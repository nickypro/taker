from collections import defaultdict
from typing import List
import torch
import torch.nn as nn
from torch import Tensor as TT
import einops
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
try:
    from .model_maps import convert_hf_model_config, ModelMap, ConfigClass
    from .hooks import ActiveHooks, NeuronMask, NeuronActAdd, NeuronPostBias, \
        NeuronSave, NeuronOffset, NeuronReplace, NeuronFunctionList
    from .data_classes import DtypeMap
except:
    from model_maps import convert_hf_model_config, ModelMap, ConfigClass
    from hooks import ActiveHooks, NeuronMask, NeuronActAdd, NeuronPostBias, \
        NeuronSave, NeuronOffset, NeuronReplace, NeuronFunctionList
    from data_classes import DtypeMap

class HookConfig:
    def __init__(self):
        self.hook_points = {
            "pre_attn": {},
            "attn_pre_out": {},
            "post_attn": {},
            "pre_mlp": {},
            "mlp_pre_out": {},
            "post_mlp": {},
        }

    def add_hook(self, point, hook_type, layer=None):
        if point not in self.hook_points:
            raise ValueError(f"Invalid hook point: {point}")
        if layer is None:
            if 'all' not in self.hook_points[point]:
                self.hook_points[point]['all'] = []
            self.hook_points[point]['all'].append(hook_type)
        else:
            if layer not in self.hook_points[point]:
                self.hook_points[point][layer] = []
            self.hook_points[point][layer].append(hook_type)

    def from_string(self, config_string):
        for line in config_string.strip().split('\n'):
            parts = line.split(':')
            if len(parts) == 2:
                point, hooks = parts
                layer = None
            elif len(parts) == 3:
                point, layer, hooks = parts
                layer = int(layer.strip())
            else:
                raise ValueError(f"Invalid config line: {line}")

            point = point.strip()
            hooks = hooks.strip()
            if hooks:  # Only add hooks if the string is not empty
                for hook in hooks.split(','):
                    self.add_hook(point, hook.strip(), layer)
        return self

    def __str__(self):
        result = []
        for point, layers in self.hook_points.items():
            for layer, hooks in layers.items():
                if hooks:
                    if layer == 'all':
                        result.append(f"{point}: {', '.join(hooks)}")
                    else:
                        result.append(f"{point}: {layer}: {', '.join(hooks)}")
        return '\n'.join(result)

    def get_hooks(self, point, layer):
        hooks = self.hook_points[point].get('all', [])
        hooks += self.hook_points[point].get(layer, [])
        return hooks

class HookMap:
    def __init__(self, model):
        self.model = model

    def __getitem__(self, component):
        return HookMapComponent(self.model, component)

    def __str__(self):
        return "HookConfig:\n" + str(self.model.hook_config)

    def delete_mlp_neurons(self, remove_indices, layer: int = None):
        return self["mlp_pre_out"].delete_neurons(remove_indices, layer)

    def delete_attn_neurons(self, remove_indices, layer: int = None):
        return self["attn_pre_out"].delete_neurons(remove_indices, layer)

class HookMapComponent:
    def __init__(self, model, component):
        self.model = model
        self.component = component

    def __getitem__(self, data_type):
        if data_type == "collect":
            return self.model.get_all_layer_activations(self.component)
        elif data_type in ["mask", "actadd", "postbias"]:
            return self.model.get_all_layer_data(self.component, data_type)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def __setitem__(self, data_type, value):
        if data_type in ["mask", "actadd", "postbias"]:
            self.model.set_all_layer_parameters(self.component, data_type, value)
        else:
            raise ValueError(f"Cannot set data type: {data_type}")

    def delete_neurons(self, remove_indices, layer: int = None):
        if layer is not None:
            mask = self.model.get_data(f"layer_{layer}_{self.component}", "mask")
            if mask is not None:
                remove_indices = self._prepare_remove_indices(remove_indices, mask.param.shape)
                keep_indices = torch.logical_not(remove_indices).flatten()
                mask.delete_neurons(keep_indices)
        else:
            masks = self["mask"]
            for layer, mask in enumerate(masks):
                if mask is not None:
                    layer_remove_indices = self._prepare_remove_indices(remove_indices[layer], mask.param.shape)
                    keep_indices = torch.logical_not(layer_remove_indices).flatten()
                    mask.delete_neurons(keep_indices)
        return self.model

    def _prepare_remove_indices(self, remove_indices: torch.Tensor, mask_shape: torch.Size) -> torch.Tensor:
        remove_indices = torch.tensor(remove_indices, dtype=torch.bool, device=self.model.device)
        if remove_indices.shape != mask_shape:
            remove_indices = remove_indices.reshape(mask_shape)
        return remove_indices

class Model:
    def __init__(self,
            model_repo: str = "nickypro/tinyllama-15m", # Which huggingface model to load
            limit: int = None, # Max amount of tokens to allow to run
            model_device: str = None, # device for transformers model ["cuda", "cpu", "mps"]
            output_device: str = None, # device for saving outputs
            device_map: str = None, # transformers device map config ["auto", "cuda"]
            use_accelerator: bool = True, # Allow multigpu
            dtype: str = "bfp16", # See DtypeConfig. ["bfp16", "fp32", "fp16", "hqq8", "int8", "hqq4", "int4", "nf4"]
            torch_dtype: torch.dtype = None, # manual torch.dtype
            svd_attn: bool = False, # whether to modify attention weights with SVD (TODO: reimplement)
            tokenizer_repo: str = None, # huggingface tokenizer to load (defaults to model_repo)
            mask_fn: str = "step", # what kind of mask to apply to model hooks ("step", "sigmoid", ...)
            eval_mode: bool = True, # set model to model.eval(), not sure what this does
            hook_config: str = None, # See HookConfig. hook configuration string.
            add_hooks: bool = True, # whether to add hooks from the start or not [True, False]
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
        self.map: ModelMap = None
        self.model: PreTrainedModel = None
        self.layers: list = None

        # Hooking into the model
        self.hook_config: HookConfig = hook_config or self.default_config()
        self.hook_handles: list = []
        self.hooks_raw: ActiveHooks = ActiveHooks()
        self.hooks: HookMap = HookMap(self)

        # Initialize the model
        self.init_model(add_hooks=add_hooks)
        with torch.no_grad():
            self.get_outputs_embeds(".")

    @staticmethod
    def default_config():
        config_string = """
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
        model_args = {}

        if self.cfg.model_modality == "vision":
            self.tokenizer = None \
                if tokenizer is None else tokenizer
            self.processor = self.init_image_processor(device_map) \
                if processor is None else self.processor
            self.predictor = AutoModelForImageClassification.from_pretrained(
                self.model_repo, device_map=device_map, **model_args) \
                if predictor is None else predictor
        elif self.cfg.model_modality == "language":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_repo, legacy=False) \
                if tokenizer is None else tokenizer
            self.processor = None \
                if processor is None else processor
            self.predictor = AutoModelForCausalLM.from_pretrained(
                self.model_repo, device_map=device_map, **model_args) \
                if predictor is None else predictor

        else:
            raise NotImplementedError(f"Model modality {self.cfg.model_modality} not implemented.")

        print(f"Loaded model '{self.model_repo}':")

    def init_model(self, do_model_import=True, add_hooks=True):
        self.cfg = convert_hf_model_config(self.model_repo)
        if do_model_import:
            self.import_models()
        self.map = ModelMap(self.predictor, self.cfg)
        self.model = self.map["model"].to(self.device)
        self.layers = self.map.layers
        if add_hooks:
            self.init_hooks()

    def init_hooks(self):
        for layer_idx, layer in enumerate(self.layers):
            for point in self.hook_config.hook_points.keys():
                hooks = self.hook_config.get_hooks(point, layer_idx)
                if hooks:
                    io_type, module = self.get_module_for_hook_point(layer, point)
                    self.register_hooks(f"layer_{layer_idx}_{point}", module, hooks, io_type)
        print(f"- Added hooks for {layer_idx+1} layers")

    def set_hook_config(self, config):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.hooks_raw = ActiveHooks()
        self.hook_config = config
        self.init_hooks()

    # Hook-specific methods
    def get_module_for_hook_point(self, layer, point):
        layer["attn"].is_attention = True
        # Attention Points
        if point == "pre_attn":
            return "in", layer["ln1"] if not self.cfg.post_layernorm else layer["attn"]
        elif point == "attn_pre_out":
            return "in", layer["attn.out_proj"]
        elif point == "post_attn":
            return "out", layer["attn"] if not self.cfg.post_layernorm else layer["ln1"]
        # MLP Points
        elif point == "pre_mlp":
            return "in", layer["ln2"] if not self.cfg.post_layernorm else layer["mlp.in_proj"]
        elif point == "mlp_pre_out":
            return "in", layer["mlp.out_proj"]
        elif point == "post_mlp":
            return "out", layer["mlp.out_proj"] if not self.cfg.post_layernorm else layer["ln2"]
        else:
            raise ValueError(f"Unknown hook point: {point}")

    def get_hook_fn(self, name, hooks, io_type="in"):
        def hook_fn(module, __input, __output=None):
            __act = __input if io_type=="in" else __output
            activation = __act[0] if isinstance(__act, tuple) else __act

            for hook in hooks:
                if hook == "collect":
                    if name not in self.hooks_raw.collects:
                        self.hooks_raw.collects[name] = NeuronSave()
                    curr_hook = self.hooks_raw.collects[name]
                elif hook == "mask":
                    if name not in self.hooks_raw.neuron_masks:
                         self.hooks_raw.neuron_masks[name] = NeuronMask(activation.shape[2:]).to(self.device)
                    curr_hook = self.hooks_raw.neuron_masks[name]
                elif hook == "actadd":
                    if name not in self.hooks_raw.neuron_actadds:
                        self.hooks_raw.neuron_actadds[name] = NeuronActAdd(self.device, self.dtype)
                    curr_hook = self.hooks_raw.neuron_actadds[name]
                elif hook == "postbias":
                    if name not in self.hooks_raw.neuron_postbiases:
                        self.hooks_raw.neuron_postbiases[name] = NeuronPostBias(activation.shape[2:]).to(self.device)
                    curr_hook = self.hooks_raw.neuron_postbiases[name]
                elif hook == "offset":
                    if name not in self.hooks_raw.neuron_offsets:
                        self.hooks_raw.neuron_offsets[name] = NeuronOffset(activation.shape[2:]).to(self.device)
                    curr_hook = self.hooks_raw.neuron_offsets[name]
                elif hook == "unoffset":
                    if name not in self.hooks_raw.neuron_unoffsets:
                        assert name in self.hooks_raw.neuron_offsets
                        self.hooks_raw.neuron_unoffsets[name] = self.hooks_raw.neuron_offsets[name]
                    curr_hook = lambda __x : self.hooks_raw.neuron_unoffsets[name].undo(__x)
                elif hook == "replace":
                    if name not in self.hooks_raw.neuron_replace:
                        self.hooks_raw.neuron_replace[name] = NeuronReplace(self.device, self.dtype)
                    curr_hook = self.hooks_raw.neuron_replace[name]
                else:
                    print(f"Warning: '{hook}' not found")
                    continue
                activation = curr_hook(activation)

            return (activation,) + __act[1:] if isinstance(__act, tuple) else activation

        return hook_fn

    def register_hooks(self, name, module, hooks, io_type):
        hook_fn = self.get_hook_fn(name, hooks, io_type)
        if io_type == "in":
            handle = module.register_forward_pre_hook(hook_fn)
        elif io_type == "out":
            handle = module.register_forward_hook(hook_fn)
        self.hook_handles.append(handle)

    def get_data(self, name=None, data_type=None):
        if data_type == "collect":
            # collect is desctructively gotten (to save memory)
            if name in self.hooks_raw.collects:
                data = self.hooks_raw.collects[name].activation
                self.hooks_raw.collects[name].activation = None
                return data
            return None
        elif data_type == "mask":
            return self.hooks_raw.neuron_masks.get(name)
        elif data_type == "actadd":
            return self.hooks_raw.neuron_actadds.get(name)
        elif data_type == "postbias":
            return self.hooks_raw.neuron_postbiases.get(name)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def set_hook_parameter(self, name, param_type, value):
        if param_type == "mask":
            self.hooks_raw.neuron_masks[name].set_mask(value)
        elif param_type == "actadd":
            self.hooks_raw.neuron_actadds[name].set_actadd(value)
        elif param_type == "postbias":
            self.hooks_raw.neuron_postbiases[name].param.data = value
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def get_layer_names(self, component):
        return [f"layer_{i}_{component}" for i in range(len(self.layers))]

    def get_all_layer_activations(self, component):
        layer_names = self.get_layer_names(component)
        return [self.get_data(name, "collect") for name in layer_names]

    def get_all_layer_data(self, component, data_type):
        layer_names = self.get_layer_names(component)
        return [self.get_data(name, data_type) for name in layer_names]

    def set_all_layer_parameters(self, component, param_type, values):
        layer_names = self.get_layer_names(component)
        for name, value in zip(layer_names, values):
            self.set_hook_parameter(name, param_type, value)

    def disable_all_collect_hooks(self):
        for name, hook in self.hooks_raw.collects.items():
            hook.enabled = False

    def enable_collect_hooks(self, components=None, layers=None):
        if components is None:
            components = self.hook_config.hook_points.keys()
        if layers is None:
            layers = range(len(self.layers))

        for component in components:
            for layer in layers:
                hook_name = f"layer_{layer}_{component}"
                if hook_name in self.hooks_raw.collects:
                    self.hooks_raw.collects[hook_name].enabled = True

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
    def get_outputs_embeds(self, text:str=None, input_ids:TT=None, raw_img=None, pixel_values=None, inputs_embeds:TT=None):
        """Get output logits from input token ids"""
        inputs_embeds = self.get_inputs_embeds(text, input_ids, raw_img, pixel_values) \
            if inputs_embeds is None else inputs_embeds
        outputs = self.model( inputs_embeds=inputs_embeds, output_hidden_states=False ).last_hidden_state
        return outputs

    # outputs_embeds --[model.lm_head]-> logits
    def unembed(self, embedded_outputs: TT):
        """ Converts outputs_embeds -> token logits. Is also basically LogitLens."""
        if "lm_head" in self.map.key_map:
            lm_head = self.map["lm_head"]
        else:
            lm_head = self.predictor.get_output_embeddings()
        return lm_head( embedded_outputs.to(self.device) )

    def get_logits(self, text:str=None, input_ids:TT=None, raw_img=None, pixel_values=None, inputs_embeds:TT=None, outputs_embeds:TT=None):
        outputs_embeds = self.get_outputs_embeds(text, input_ids, raw_img, pixel_values, inputs_embeds) \
            if outputs_embeds is None else outputs_embeds
        logits = self.unembed(outputs_embeds)
        return logits

    # Get intermediate activaitons (using HOOKS!!!)
    def get_midlayer_activations(self, text=None, input_ids=None, raw_img=None, pixel_values=None):
        # Set up the activations to be collected minimally
        self.disable_all_collect_hooks()
        self.enable_collect_hooks(["mlp_pre_out", "attn_pre_out"])

        # Run model
        _outputs_embeds = self.get_outputs_embeds(text, input_ids, raw_img, pixel_values)

        # Collect and return activaitons
        return {
            "attn": self.collect_recent_attn_pre_out(),
            "mlp":  self.collect_recent_mlp_pre_out(),
        }

    def collect_recent_mlp_pre_out(self):
        mlp_activations = self.hooks["mlp_pre_out"]["collect"]
        mlp_activations = einops.rearrange(torch.stack(mlp_activations),
            "layer batch token dim -> batch layer token dim")
        return mlp_activations

    def collect_recent_attn_pre_out(self):
        attn_activations = self.hooks["attn_pre_out"]["collect"]
        attn_activations = einops.rearrange(torch.stack(attn_activations),
            "layer batch token (n_heads d_head) -> batch layer token n_heads d_head",
            n_heads=self.cfg.n_heads, d_head=self.cfg.d_head
        )
        return attn_activations

    def get_residual_stream(self, text, split=False):
        # Set up the activations to be collected minimally
        self.disable_all_collect_hooks()
        self.enable_collect_hooks(["pre_mlp", "pre_attn"])

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
        self.disable_all_collect_hooks()
        self.enable_collect_hooks(["post_attn", "post_mlp"])

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

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        return self.model(input_ids)

if __name__ == "__main__":
    model = Model("gpt2")  # This will use the default config


    print(model.hooks)
    # Get midlayer activations
    txt = "Hello, world!"
    midlayer_activations = model.get_midlayer_activations(txt)
    print("MLP activations shape:", midlayer_activations["mlp"].shape)
    print("Attention activations shape:", midlayer_activations["attn"].shape)

    # Get residual stream
    res = model.get_residual_stream(txt, split=True)
    print("Residual stream shape:", res.shape)
    res = model.get_residual_stream(txt)
    print("Residual stream shape:", res.shape)

    mask0 = model.get_data("layer_0_attn_pre_out", "mask")
    new_mask = torch.ones_like(mask0.param)  # Assuming d_model is 768
    new_mask[:384] = 0  # Set first half to 0
    model.set_hook_parameter("layer_0_attn_pre_out", "mask", new_mask)

    print(model.hooks)
    print(model.hook_config.hook_points)

    # Run with new mask
    midlayer_activations = model.get_midlayer_activations(txt)
    print("New MLP activations shape:", midlayer_activations["mlp"].shape)
