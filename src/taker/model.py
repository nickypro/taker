from collections import defaultdict
from typing import List, Tuple

import einops
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch import Tensor as TT
from transformers import (AutoImageProcessor, AutoModelForCausalLM,
                          AutoModelForImageClassification, AutoTokenizer,
                          PreTrainedModel)

from .data_classes import DtypeMap
from .hooks import HookConfig, HookMap
from .model_maps import ConfigClass, ModelMap, convert_hf_model_config


class Model:
    def __init__(self,
            model_repo: str = "nickypro/tinyllama-15m", # Which huggingface model to load
            limit: int = None, # Max amount of tokens to allow to run
            model_device: str = None, # device for transformers model ["cuda", "cpu", "mps"]
            output_device: str = None, # device for saving outputs
            device_map: str = None, # transformers device map config ["auto", "cuda"]
            use_accelerator: bool = True, # Allow multigpu
            dtype: str = "bfp16", # See DtypeConfig. ["bfp16", "fp32", "fp16", "hqq8", "int8", "hqq4", "int4", "nf4"]
            compile: bool = False, # whether to use compiled backend. Prevents "training/backprop", adds speed
            torch_dtype: torch.dtype = None, # manual torch.dtype
            svd_attn: bool = False, # whether to modify attention weights with SVD (TODO: reimplement)
            tokenizer_repo: str = None, # huggingface tokenizer to load (defaults to model_repo)
            mask_fn: str = "step", # what kind of mask to apply to model hooks ("step", "sigmoid", ...)
            eval_mode: bool = True, # set model to model.eval(), not sure what this does
            hook_config: str = None, # See HookConfig. hook configuration string.
            add_hooks: bool = True, # whether to add hooks from the start or not [True, False]
            model_kwargs: dict = None, # additional kwargs for the model
        ):
        # Handle devices and multi-gpu stuff
        self.use_accelerator = False
        if (model_device is None) and (use_accelerator) and (torch.cuda.device_count() > 1):
            self.use_accelerator = True
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
        else:
            self.device = model_device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Model device mapping and output device
        self.device_map = device_map or "auto"
        if not self.use_accelerator and self.device != "cuda":
            self.device_map = self.device
        self.output_device = output_device or self.device

        # Handle dtype
        self.dtype_map = DtypeMap(dtype, torch_dtype)
        self.dtype = self.dtype_map._dtype
        self.dtype_args = self.dtype_map._dtype_args
        self.use_quantization = "quantization_config" in self.dtype_args

        # Model configuration
        self.model_repo = model_repo
        self.tokenizer_repo = tokenizer_repo or model_repo
        self.svd_attn = svd_attn
        self.mask_fn = mask_fn
        self.eval_mode = eval_mode
        self.limit = limit

        # Initialize model components
        self.cfg: ConfigClass = None
        self.tokenizer: AutoTokenizer = None
        self.processor: AutoImageProcessor = None
        self.predictor: AutoModelForCausalLM | AutoModelForImageClassification = None
        self.orig_predictor: AutoModelForCausalLM | AutoModelForImageClassification = None
        self.peft_predictor = None
        self.map: ModelMap = None
        self.model: PreTrainedModel = None
        self.layers: list = None

        # Hooking into the model
        self.hook_config: HookConfig = hook_config or self.default_config()
        self.hooks: HookMap = HookMap(self.hook_config)
        self.hook_config = self.hooks.hook_config

        # Initialize the model
        self.compile = compile
        self.model_kwargs = model_kwargs or {}
        self.init_model(add_hooks=add_hooks)
        self.run_example_input()

    @staticmethod
    def default_config():
        config_string = """
        pre_decoder: collect
        post_decoder: collect
        pre_attn: collect
        attn_pre_out: offset, mask, replace, collect, unoffset
        post_attn: collect
        pre_mlp: collect
        mlp_pre_out: offset, mask, replace, collect, unoffset
        post_mlp: collect
        """
        return HookConfig().from_string(config_string)

    def show_details( self, verbose=True ):
        if verbose:
            print(" - n_layers :", self.cfg.n_layers)
            print(" - d_model  :", self.cfg.d_model)
            print(" - n_heads  :", self.cfg.n_heads)
            print(" - d_head   :", self.cfg.d_head)
            print(" - d_mlp    :", self.cfg.d_mlp)
        else:
            print( f" - n_layers, d_model = {self.cfg.n_layers}, {self.cfg.d_model}" )

    def import_models(self,
            tokenizer=None,
            predictor=None,
            processor=None
        ):
        # Import model components (Default: Causal Language Models)
        device_map = self.device_map
        model_args = {**self.dtype_args, **self.model_kwargs}

        if self.cfg.model_modality == "vision":
            self.tokenizer = None \
                if tokenizer is None else tokenizer
            self.processor = self.init_image_processor(device_map) \
                if processor is None else self.processor
            self.predictor = AutoModelForImageClassification.from_pretrained(
                self.model_repo, device_map=device_map, **model_args) \
                if predictor is None else predictor
        elif self.cfg.model_modality == "language":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_repo, legacy=False, padding_side='left') \
                if tokenizer is None else tokenizer
            self.processor = None \
                if processor is None else processor
            self.predictor = AutoModelForCausalLM.from_pretrained(
                self.model_repo, device_map=device_map, **model_args) \
                if predictor is None else predictor

        else:
            raise NotImplementedError(f"Model modality {self.cfg.model_modality} not implemented.")
        self.orig_predictor = self.predictor
        print(f"Loaded model '{self.model_repo}' with {self.dtype_map.str_dtype}:")

    def init_model(self, do_model_import=True, add_hooks=True):
        self.cfg = convert_hf_model_config(self.model_repo)
        if do_model_import:
            self.import_models()
        self.map = ModelMap(self.predictor, self.cfg)
        self.model = self.map["model"] #.to(self.device)
        self.layers = self.map.layers
        if add_hooks:
            self.init_hooks()
        if self.compile:
            self.init_compile()

    def init_peft(self, peft_config_or_path, reinit_hooks=False):
        """
        Initialize PEFT for the model.

        Args:
            peft_config_or_path (PeftConfig or str):
                Either a PeftConfig object or a string path to a PEFT model repository.
        """
        try:
            from peft import PeftConfig, PeftModel, get_peft_model
        except ImportError:
            raise ImportError("PEFT is not installed. Please install it with 'pip install peft'.")

        if reinit_hooks:
            self.remove_hooks()
        if isinstance(peft_config_or_path, str): # load from repo
            self.peft_predictor = PeftModel.from_pretrained(self.orig_predictor, peft_config_or_path)
        else: # create from config
            self.peft_predictor = get_peft_model(self.orig_predictor, peft_config_or_path)
        self.predictor = self.peft_predictor.base_model.model
        # Reinitialize the model
        self.init_model(do_model_import=False, add_hooks=False)
        print(f"Initialized PEFT model")
        self.peft_predictor.print_trainable_parameters()
        if reinit_hooks:
            self.init_hooks()

    def init_compile(self):
        from torch import _dynamo
        torch._dynamo.config.suppress_errors = True
        self.predictor = self.dtype_map.compile(self.predictor)

    def __getitem__(self, key):
        return self.map[key]

    def __setitem__(self, key, value):
        self.map[key] = value

    def run_example_input(self):
        with torch.no_grad():
            if self.cfg.model_modality == "language":
                self.get_outputs_embeds("a b")
            if self.cfg.model_modality == "vision":
                self.get_outputs_embeds(
                    pixel_values=torch.randn([1,3,self.cfg.image_size,self.cfg.image_size], dtype=self.dtype, device=self.device)
                )

    def init_image_processor(self, device_map):
        """ Initialize processor from raw pixel values to normalised tensors"""
        try:
            from transformers import AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_repo, device_map=device_map, **self.dtype_args)
        except:
            from .vit_processor import SsdVitProcessor
            self.processor = SsdVitProcessor()
        return self.processor

    # Functions for hooks
    def set_hook_config(self, config: HookConfig):
        self.remove_hooks()
        if isinstance(config, str):
            config = HookConfig().from_string(config)
        self.hook_config = config
        self.hooks = HookMap(config)
        self.init_hooks()
        self.run_example_input()

    def init_hooks(self):
        # (x) [attn_out] (y) -> (x) [layer] (y)
        num_hooks: int = 0
        for layer_idx, layer in enumerate(self.layers):
            for point in self.hook_config.hook_points.keys():
                hooks = self.hook_config.get_hooks(point, layer_idx)
                if hooks:
                    io_type, module = self.get_module_for_hook_point(layer, point)
                    self.register_hooks(f"layer_{layer_idx}_{point}", module, hooks, io_type)
                    num_hooks += len(hooks)
        self.hook_config.n_layers = self.cfg.n_layers
        print(f"- Added {num_hooks} hooks across {layer_idx+1} layers")

    def register_hooks(self, name, module, hooks, io_type):
        hook_fn = self.get_hook_fn(name, hooks, io_type)
        if io_type == "in":
            handle = module.register_forward_pre_hook(hook_fn)
        elif io_type == "out":
            handle = module.register_forward_hook(hook_fn)
        self.hooks.handles.append(handle)

    def get_hook_fn(self, name, hooks, io_type="in"):
        def hook_fn(module, __input, __output=None):
            # Is this input or output?
            __act = __input if io_type=="in" else __output
            activation = __act[0] if isinstance(__act, tuple) else __act

            # Does the module not use batch index? if so, we fix this
            has_batch = f"has_batch_index_{io_type}"
            if not hasattr(module, has_batch): # init run has 1 batch only
                setattr(module, has_batch, activation.shape[0] == 1)
            if not getattr(module, has_batch):
                activation = activation.unsqueeze(dim=0)

            if "attn_pre_out" in name:
                activation = self.split_attn_head_dims(activation)

            for hook in hooks:
                # split attention to [n_heads, d_head]
                curr_hook  = self.hooks.get_hook_fn(hook, name, activation, self.device, self.dtype)
                activation = curr_hook(activation)

            if "attn_pre_out" in name:
                activation = self.join_attn_head_dims(activation)
            return (activation,) + __act[1:] if isinstance(__act, tuple) else activation

        return hook_fn

    # Hook-specific methods
    def get_module_for_hook_point(self, layer, point):
        layer["attn"].is_attention = True
        # Attention Points
        if point == "pre_attn":
            return "in", layer["attn.ln_in"] if self.cfg.pre_layernorm else layer["attn"]
        elif point == "attn_pre_out":
            return "in", layer["attn.out_proj"]
        elif point == "post_attn":
            return "out", layer["attn.ln_out"] if self.cfg.post_layernorm else layer["attn"]
        # MLP Points
        elif point == "pre_mlp":
            return "in", layer["mlp.ln_in"] if self.cfg.pre_layernorm else layer["mlp.in_proj"]
        elif point == "mlp_pre_out":
            return "in", layer["mlp.out_proj"]
        elif point == "post_mlp":
            return "out", layer["mlp.ln_out"] if self.cfg.post_layernorm else layer["mlp.out_proj"]
        # decoder block points:
        elif point == "pre_decoder":
            return "in", layer.layer
        elif point == "post_decoder":
            return "out", layer.layer
        else:
            raise ValueError(f"Unknown hook point: {point}")

    def remove_hooks(self):
        for handle in self.hooks.handles:
            handle.remove()
        self.hooks.handles.clear()


    # Helper functions for reshaping activations
    def split_attn_head_dims(self, activation):
        if activation.shape[-1] == (self.cfg.n_heads * self.cfg.d_head):
            activation = einops.rearrange(
                activation, "... (n_heads d_head) -> ... n_heads d_head",
                n_heads=self.cfg.n_heads, d_head=self.cfg.d_head)
        return activation

    def join_attn_head_dims(self, activation):
        if activation.shape[-1] == self.cfg.d_head:
            activation = einops.rearrange(
                activation, "... n_heads d_head -> ... (n_heads d_head)",
                n_heads=self.cfg.n_heads, d_head=self.cfg.d_head)
        return activation

    # Functions for symmetrically altering model.
    def center_unembed(self):
        with torch.no_grad():
            if "unembed" in self.map:
                lm_head = self.map["unembed"]
            else:
                lm_head = self.model.get_output_embeddings()
            embed = self.map["embed"]
            assert embed is not lm_head, "centering unembedding not yet supported for tied weights"
            lm_head.weight = torch.nn.Parameter(lm_head.weight - lm_head.weight.mean(dim=-1, keepdim=True))

    # Get "normal" Model Activations
    # text           --[tokenizer]-----> input_ids
    def get_ids(self, text:str):
        input_ids = self.tokenizer( text, return_tensors='pt').input_ids
        if self.limit is not None:
            input_ids = torch.stack([ input_ids[0][:self.limit] ])
        return input_ids.to( self.device )


    # raw_img      --[processor]----> pixel_values
    def get_pixel_values(self, raw_img):
        img = self.processor(raw_img, return_tensors="pt")
        pixel_values = img["pixel_values"].to(self.device)
        return pixel_values

    # input_ids    --[model.embed]--> inputs_embeds
    # pixel_values --[model.embed]--> inputs_embeds
    def get_inputs_embeds(self, text:str = None, input_ids:TT = None, raw_img=None, pixel_values=None):
        if text is not None:
            input_ids = self.get_ids(text)
        if input_ids is not None:
            return self.map["embed"](input_ids)
        if raw_img is not None:
            pixel_values = self.get_pixel_values(raw_img)
        if pixel_values is not None:
            return self.map["embed"](pixel_values)
        return None

    # inputs_embeds  --[model.model]---> outputs_embeds
    def get_outputs_embeds(self, text:str=None, input_ids:TT=None, raw_img=None, pixel_values=None, inputs_embeds:TT=None, attention_mask=None, **kwargs):
        """Get output logits from input text/image"""
        if attention_mask is not None: # in order for attention to work right need to do this ???
            kwargs["output_attentions"] = True
        if self.cfg.model_modality == "vision":
            pixel_values = self.get_pixel_values(raw_img) if pixel_values is None else pixel_values
            return self.model(pixel_values, output_hidden_states=False, **kwargs ).last_hidden_state

        inputs_embeds = self.get_inputs_embeds(text, input_ids) \
            if inputs_embeds is None else inputs_embeds
        return self.model( inputs_embeds=inputs_embeds, output_hidden_states=False, attention_mask=attention_mask, **kwargs ).last_hidden_state

    def get_attn_weights(self, text=None, input_ids=None, inputs_embeds=None):
        inputs_embeds = inputs_embeds if inputs_embeds is not None \
            else self.get_inputs_embeds(text=text, input_ids=input_ids)
        outputs = self.model(inputs_embeds=inputs_embeds, output_attentions=True)
        return einops.rearrange(torch.stack(outputs.attentions),
            "layer batch head tok_i tok_j -> batch layer head tok_i tok_j",
            layer=self.cfg.n_layers, head=self.cfg.n_heads,
        )

    # outputs_embeds --[model.lm_head]-> logits
    def unembed(self, embedded_outputs: TT):
        """ Converts outputs_embeds -> token logits. Is also basically LogitLens."""
        if "unembed" in self.map.key_map: #aka "lm_head"
            lm_head = self.map["unembed"]
        else:
            lm_head = self.predictor.get_output_embeddings()
        return lm_head( embedded_outputs.to(self.device) )

    def get_logits(self, text:str=None, input_ids:TT=None, raw_img=None, pixel_values=None, inputs_embeds:TT=None, outputs_embeds:TT=None, attention_mask=None, **kwargs):
        outputs_embeds = self.get_outputs_embeds(text, input_ids, raw_img, pixel_values, inputs_embeds, attention_mask, **kwargs) \
            if outputs_embeds is None else outputs_embeds
        logits = self.unembed(outputs_embeds)
        return logits


    def generate(self, text:str=None, num=10, max_length=None,
            input_ids: TT = None,
            inputs_embeds: TT = None,
            do_sample: bool = True,
            temperature: float = 0.7,
            **kwargs,
        ):
        """ Predict the next {num} tokens from an input {text}."""

        if text is not None:
            input_ids = self.get_ids(text).to(self.device)

        if input_ids is not None:
            inputs_embeds = self.get_inputs_embeds(input_ids=input_ids).to(self.device)

        if input_ids is None:
            text_before = ""
            input_ids = torch.zeros(inputs_embeds.shape[:-1], dtype=torch.long).to(self.device)
        else:
            text_before = self.tokenizer.batch_decode( input_ids,
                skip_special_tokens=True, clean_up_tokenization_spaces=False )[0]

        assert inputs_embeds is not None, "must pass text, input_ids, or inputs_embeds"

        # Write an attention mask so that the tokenizer doesn't complain
        attn_mask = None
        if hasattr(self.tokenizer, "pad_token_id"):
            attn_mask = torch.ones_like(input_ids).bool()
            for index, _id in enumerate(attn_mask[0]):
                if _id == self.tokenizer.pad_token_id:
                    attn_mask[index] = 0

        # Hard code GPT2 Tokeniser pad_token_id to avoid warnings
        if self.cfg.architecture == "GPT2LMHeadModel":
            if "pad_token_id" not in kwargs:
                kwargs["pad_token_id"] = 50256

        new_length = len(input_ids[0])+num if max_length is None else max_length

        # Generate from input_ids if available, seems to work more reliably (eg gemma2)
        if input_ids is not None:
            generate_ids = self.predictor.generate(input_ids=input_ids, max_length=new_length,
                do_sample=do_sample, temperature=temperature, pad_token_id=self.tokenizer.pad_token_ids,
                attention_mask=attn_mask, **kwargs)
            text_after = self.tokenizer.batch_decode( generate_ids[:, len(input_ids[0]):],
                skip_special_tokens=True, clean_up_tokenization_spaces=False )[0]
            return text_before, text_after

        # Otherwise, generate from inputs_embeds
        generate_ids = self.predictor.generate(inputs_embeds=inputs_embeds, max_length=new_length,
            do_sample=do_sample, temperature=temperature,
            attention_mask=attn_mask, **kwargs)
        text_after  = self.tokenizer.batch_decode( generate_ids,
            skip_special_tokens=True, clean_up_tokenization_spaces=False )[0]
        return text_before, text_after

    def generate_batch(self,
            batch_prompts: List[str],
            num: int = 10,
            max_length: int = None,
            temperature: float = 0.7,
            do_sample: bool = True,
            **kwargs
        ) -> Tuple[List[str], List[str]]:
        """
        Generate the next {num} tokens for a batch of input prompts.

        Args:
            batch_prompts (List[str]): A list of input prompts.
            num (int, optional): Number of new tokens to generate per prompt. Defaults to 10.
            max_length (int, optional): Maximum length of the generated sequences. If None, it will be set to
                                         the length of the input plus `num`. Defaults to None.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            do_sample (bool, optional): Whether to use sampling; use greedy decoding otherwise. Defaults to True.
            **kwargs: Additional generation parameters.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing the original prompts and their generated continuations.
        """
        # Ensure the tokenizer has a pad_token
        if self.tokenizer.pad_token is not None:
            pass
        elif self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise ValueError("Tokenizer has neither pad_token nor eos_token defined.")

        # Tokenize all prompts in the batch
        batch_encodings = self.tokenizer(
            batch_prompts,
            padding=True,
            truncation=False,
            max_length=1000,
            return_tensors="pt"
        )
        orig_len = batch_encodings.input_ids.shape[1]

        # Determine the new maximum length
        new_length = orig_len + num if max_length is None else max_length

        # Generate outputs
        generate_ids = self.predictor.generate(
            input_ids=batch_encodings.input_ids.to(self.device),
            attention_mask=batch_encodings.attention_mask.to(self.device),
            max_length=new_length,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )

        # Decode all generated sequences at once
        batch_text_after = self.tokenizer.batch_decode(
            [ids[orig_len:] for ids in generate_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return batch_prompts, batch_text_after

    # Get intermediate activaitons (using HOOKS!!!)
    def get_midlayer_activations(self, text=None, input_ids=None, raw_img=None, pixel_values=None):
        # Set up the activations to be collected minimally
        self.hooks.disable_all_collect_hooks()
        self.hooks.enable_collect_hooks(["mlp_pre_out", "attn_pre_out"])

        # Run model
        _outputs_embeds = self.get_outputs_embeds(text, input_ids, raw_img, pixel_values)

        # Collect and return activaitons
        return {
            "attn": self.collect_recent_attn_pre_out(),
            "mlp":  self.collect_recent_mlp_pre_out(),
        }

    def collect_recent_mlp_pre_out(self):
        mlp_activations = torch.stack([x.to(self.device) for x in self.hooks["mlp_pre_out"]["collect"]])
        mlp_activations = einops.rearrange(mlp_activations,
            "layer batch token dim -> batch layer token dim")
        return mlp_activations

    def collect_recent_attn_pre_out(self):
        attn_activations = torch.stack([x.to(self.device) for x in self.hooks["attn_pre_out"]["collect"]])
        attn_activations = einops.rearrange(attn_activations,
            "layer batch token n_heads d_head -> batch layer token n_heads d_head",
            n_heads=self.cfg.n_heads, d_head=self.cfg.d_head
        )
        return attn_activations

    def get_residual_stream_decoder(self, text):
        self.hooks.collect_hooks()
        self.hooks.enable_collect_hooks(["post_decoder"])
        # Forward pass
        inputs_embeds = self.get_inputs_embeds(text)
        outputs_embeds = self.get_outputs_embeds(inputs_embeds=inputs_embeds)
        # Collect residual stream
        post_decoder_activations = self.hooks["post_decoder"]["collect"]

        return einops.rearrange(
            torch.stack([inputs_embeds, *post_decoder_activations]),
            "layer batch token dim -> batch layer token dim"
        )


    def get_residual_stream(self, text, split=False, decoder_blocks=False):
        if decoder_blocks:
            return self.get_residual_stream_decoder(text)
        # Set up the activations to be collected minimally
        self.hooks.disable_all_collect_hooks()
        self.hooks.enable_collect_hooks(["pre_mlp", "pre_attn"])

        # Forward pass
        outputs_embeds = self.get_outputs_embeds(text)

        # Collect residual stream
        pre_attn_activations = self.hooks["pre_attn"]["collect"]
        pre_mlp_activations  = self.hooks["pre_mlp"]["collect"]

        residual_stream = []
        for pre_attn, pre_mlp in zip(pre_attn_activations, pre_mlp_activations):
            layer_residuals = []
            if pre_attn is not None:
                layer_residuals.append(pre_attn)
            if pre_mlp is not None:
                layer_residuals.append(pre_mlp)
            if layer_residuals:
                if not split:
                    residual_stream += layer_residuals
                else:
                    residual_stream.append(torch.stack(layer_residuals))

        if residual_stream is None:
            print("WARNING: Could not get residual stream.")
            return None

        if split:
            return einops.rearrange(torch.stack(residual_stream),
                "layer component batch token dim -> component batch layer token dim"
            )

        residual_stream.append(outputs_embeds)
        return einops.rearrange(torch.stack(residual_stream),
            "layer batch token dim -> batch layer token dim"
        )

    def get_residual_diffs(self, text=None, input_ids=None, split=True):
        self.hooks.disable_all_collect_hooks()
        self.hooks.enable_collect_hooks(["post_attn", "post_mlp"])

        # Forward pass
        inputs_embeds  = self.get_inputs_embeds(text, input_ids)
        outputs_embeds = self.get_outputs_embeds(inputs_embeds=inputs_embeds)

        # Collect residual stream
        post_attn_activations = self.hooks["post_attn"]["collect"]
        post_mlp_activations  = self.hooks["post_mlp"]["collect"]
        post_attn_activations = einops.rearrange(
            torch.stack(post_attn_activations),
            "layer batch token dim -> batch layer token dim")
        post_mlp_activations  = einops.rearrange(
            torch.stack(post_mlp_activations),
            "layer batch token dim -> batch layer token dim")

        return inputs_embeds, post_attn_activations, post_mlp_activations, outputs_embeds

    def get_text_activations(self, text=None, input_ids=None):
        return self.get_residual_diffs(text=text, input_ids=input_ids)

    def components_loop(self):
        yield self.predictor
        yield self.model
        for hook in self.hooks.all_hooks:
            yield hook

    def to( self, device ):
        if self.use_accelerator: # If using accelerator, init handles multi-device
            return
        if self.dtype_map.is_low_precision: # 8bit & 4bit mode handled by accelerator
            return
        if self.use_quantization:
            return
        orig_device = self.device
        self.device = device
        if self.output_device == orig_device:
            self.output_device = self.device
        for component in self.components_loop():
            component.to(self.device)
        return self

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        return self.model(input_ids)
