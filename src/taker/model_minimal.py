from collections import defaultdict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
try:
    from .model_maps import convert_hf_model_config, ModelMap, ConfigClass
    from .nn import ActiveHooks, NeuronMask, NeuronActAdd, NeuronPostBias, NeuronSave, NeuronFunctionList
    from .data_classes import DtypeMap
except:
    from model_maps import convert_hf_model_config, ModelMap, ConfigClass
    from nn import ActiveHooks, NeuronMask, NeuronActAdd, NeuronPostBias, NeuronSave, NeuronFunctionList
    from data_classes import DtypeMap

class ActivationsConfig:
    def __init__(self):
        self.hook_points = {
            "pre_attn": {},
            "post_attn": {},
            "pre_mlp": {},
            "post_mlp": {},
            "attn_pre_out": {},
            "mlp_pre_out": {}
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

class Model:
    def __init__(self,
            model_repo: str = "nickypro/tinyllama-15m",
            limit: int = None,
            model_device: str = None,
            output_device: str = None,
            device_map: str = None,
            use_accelerator: bool = True,
            dtype: str = "bfp16",
            torch_dtype: torch.dtype = None,
            svd_attn: bool = False,
            tokenizer_repo: str = None,
            mask_fn: str = "step",
            use_inverse_out: bool = False,
            eval_mode: bool = True,
            activations_config: str = None,
            add_hooks: bool = True,
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
        #self.activations_config = None
        self.activations_config = activations_config or self.default_config()
        self.hook_handles = []
        self.hooks = ActiveHooks()

        # Initialize the model
        self.init_model(add_hooks=add_hooks)

    @staticmethod
    def default_config():
        config_string = """
        pre_attn: collect
        attn_pre_out: collect, mask
        post_attn: collect
        pre_mlp: collect
        mlp_pre_out: collect, mask
        post_mlp: collect
        """
        return ActivationsConfig().from_string(config_string)

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
            for point in self.activations_config.hook_points.keys():
                hooks = self.activations_config.get_hooks(point, layer_idx)
                if hooks:
                    io_type, module = self.get_module_for_hook_point(layer, point)
                    self.register_hooks(f"layer_{layer_idx}_{point}", module, hooks, io_type)

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
        elif point == "post_mlp":
            return "in", layer["mlp.out_proj"] if not self.cfg.post_layernorm else layer["ln2"]
        elif point == "mlp_pre_out":
            return "out", layer["mlp.out_proj"]
        else:
            raise ValueError(f"Unknown hook point: {point}")

    def get_hook_fn(self, name, hooks, io_type="in"):
        def hook_fn(module, __input, __output=None):
            __act = __input if io_type=="in" else __output
            activation = __act[0] if isinstance(__act, tuple) else __act

            for hook in hooks:
                if hook == "collect":
                    if name not in self.hooks.collects:
                        self.hooks.collects[name] = NeuronSave()
                    curr_hook = self.hooks.collects[name]
                elif hook == "mask":
                    if name not in self.hooks.neuron_masks:
                         self.hooks.neuron_masks[name] = NeuronMask(activation.shape[1:]).to(self.device)
                    curr_hook = self.hooks.neuron_masks[name]
                elif hook == "actadd":
                    if name not in self.hooks.neuron_actadds:
                        self.hooks.neuron_actadds[name] = NeuronActAdd(self.device, self.dtype)
                    curr_hook = self.hooks.neuron_actadds[name]
                elif hook == "postbias":
                    if name not in self.hooks.neuron_postbiases:
                        self.hooks.neuron_postbiases[name] = NeuronPostBias(activation.shape[1:])
                    curr_hook = self.hooks.neuron_postbiases[name]
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
            return self.hooks.collects.get(name).activation if name in self.hooks.collects else None
            #return self.hooks.collects.get(name).activation if name in self.hooks.collects else None
        elif data_type == "mask":
            return self.hooks.neuron_masks.get(name)
        elif data_type == "actadd":
            return self.hooks.neuron_actadds.get(name)
        elif data_type == "postbias":
            return self.hooks.neuron_postbiases.get(name)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def set_hook_parameter(self, name, param_type, value):
        if param_type == "mask":
            self.hooks.neuron_masks[name].set_mask(value)
        elif param_type == "actadd":
            self.hooks.neuron_actadds[name].set_actadd(value)
        elif param_type == "postbias":
            self.hooks.neuron_postbiases[name].param.data = value
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def get_config(self, point, layer=None):
        if layer is None:
            return self.activations_config.hook_points.get(point, {})
        return self.activations_config.get_hooks(point, layer)

    def set_activations_config(self, config):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.hooks = ActiveHooks()
        self.activations_config = config
        self.init_hooks()

    def get_midlayer_activations(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        self.forward(input_ids)

        mlp_activations = []
        attn_activations = []

        for layer in range(len(self.layers)):
            mlp_act = self.get_data(f"layer_{layer}_mlp_pre_out", "collect")
            attn_act = self.get_data(f"layer_{layer}_attn_pre_out", "collect")

            if mlp_act is not None:
                mlp_activations.append(mlp_act)
            if attn_act is not None:
                attn_activations.append(attn_act)

        return {
            "mlp": torch.stack(mlp_activations) if mlp_activations else None,
            "attn": torch.stack(attn_activations) if attn_activations else None
        }

    def get_residual_stream(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        # Forward pass
        output = self.forward(input_ids)

        # Collect residual stream
        residual_stream = []
        for layer in range(len(self.layers)):
            layer_residuals = []
            for point, data_type in [("pre_attn", "collect"), ("pre_mlp", "collect")]:
                act = self.get_data(f"layer_{layer}_{point}", data_type)
                if act is not None:
                    layer_residuals.append(act)
            if layer_residuals:
                residual_stream.append(torch.stack(layer_residuals))

        # Restore original configuration
        return torch.stack(residual_stream) if residual_stream else None


    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        return self.model(input_ids)

if __name__ == "__main__":
    model = Model("gpt2")  # This will use the default config


    print(model.hooks)
    # Get midlayer activations
    text = "Hello, world!"
    midlayer_activations = model.get_midlayer_activations(text)
    print("MLP activations shape:", midlayer_activations["mlp"].shape)
    print("Attention activations shape:", midlayer_activations["attn"].shape)

    # Get residual stream
    residual_stream = model.get_residual_stream(text)
    print("Residual stream shape:", residual_stream.shape)

    mask0 = model.get_data("layer_0_attn_pre_out", "mask")
    new_mask = torch.ones_like(mask0.param)  # Assuming d_model is 768
    new_mask[:384] = 0  # Set first half to 0
    model.set_hook_parameter("layer_0_attn_pre_out", "mask", new_mask)

    print(model.hooks)

    # Run with new mask
    midlayer_activations = model.get_midlayer_activations(text)
    print("New MLP activations shape:", midlayer_activations["mlp"].shape)
