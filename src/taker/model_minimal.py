import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from .model_maps import convert_hf_model_config, ModelMap, ConfigClass
    from .nn import ActiveHooks, NeuronMask, NeuronActAdd, NeuronPostBias, NeuronInputSave, NeuronFunctionList
except:
    from model_maps import convert_hf_model_config, ModelMap, ConfigClass
    from nn import ActiveHooks, NeuronMask, NeuronActAdd, NeuronPostBias, NeuronInputSave, NeuronFunctionList

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
    def __init__(self, model_repo, activations_config=None, device_map="auto"):
        self.model_repo = model_repo
        self.device_map = device_map
        self.cfg = None
        self.tokenizer = None
        self.predictor = None
        self.map = None
        self.model = None
        self.layers = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activations_config = activations_config or self.default_config()
        self.hook_handles = []
        self.hooks = ActiveHooks()
        self.dtype = torch.bfloat16
        self.init_model()

    @staticmethod
    def default_config():
        config_string = """
        pre_attn: save_input, mask
        post_attn: save_output, actadd
        pre_mlp: save_input, mask
        post_mlp: save_output, actadd
        attn_pre_out: save_input, mask
        mlp_pre_out: save_input, mask
        """
        return ActivationsConfig().from_string(config_string)

    def import_models(self, tokenizer=None, predictor=None):
        device_map = self.device_map
        model_args = {}

        if self.cfg.model_modality == "language":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo, legacy=False) \
                if tokenizer is None else tokenizer
            self.predictor = AutoModelForCausalLM.from_pretrained(
                self.model_repo, device_map=device_map, **model_args) \
                if predictor is None else predictor
        else:
            raise NotImplementedError(f"Model modality {self.cfg.model_modality} not implemented.")

    def init_model(self, do_model_import=True):
        self.cfg = convert_hf_model_config(self.model_repo)
        if do_model_import:
            self.import_models()
        self.map = ModelMap(self.predictor, self.cfg)
        self.model = self.map["model"].to(self.device)
        self.layers = self.map.layers
        self.init_hooks()

    def init_hooks(self):
        for layer_idx, layer in enumerate(self.layers):
            for point in self.activations_config.hook_points.keys():
                hooks = self.activations_config.get_hooks(point, layer_idx)
                if hooks:
                    module = self.get_module_for_hook_point(layer, point)
                    self.register_hooks(f"layer_{layer_idx}_{point}", module, hooks)

    def get_module_for_hook_point(self, layer, point):
        if point == "pre_attn":
            return layer["ln1"]
        elif point == "post_attn":
            return layer["attn"]
        elif point == "pre_mlp":
            return layer["ln2"]
        elif point == "post_mlp":
            return layer["mlp"]
        elif point == "attn_pre_out":
            return layer["attn.out_proj"]
        elif point == "mlp_pre_out":
            return layer["mlp.out_proj"]
        else:
            raise ValueError(f"Unknown hook point: {point}")

    def get_hook_fn(self, name, hooks, is_attention):
        def hook_fn(module, input, output):
            activation = output[0] if is_attention and isinstance(output, tuple) else output

            for hook in hooks:
                if hook == "save_input":
                    if name not in self.hooks.input_saves:
                        self.hooks.input_saves[name] = NeuronInputSave()
                    self.hooks.input_saves[name](module, input[0])
                elif hook == "save_output":
                    if name not in self.hooks.output_saves:
                        self.hooks.output_saves[name] = NeuronInputSave()
                    self.hooks.output_saves[name](module, input[0], activation)
                elif hook == "mask":
                    if name not in self.hooks.neuron_masks:
                        self.hooks.neuron_masks[name] = NeuronMask(activation.shape[1:]).to(self.device)
                    activation = self.hooks.neuron_masks[name](activation)
                elif hook == "actadd":
                    if name not in self.hooks.neuron_actadds:
                        self.hooks.neuron_actadds[name] = NeuronActAdd(self.device, activation.dtype)
                    activation = self.hooks.neuron_actadds[name](activation)
                elif hook == "postbias":
                    if name not in self.hooks.neuron_postbiases:
                        self.hooks.neuron_postbiases[name] = NeuronPostBias(activation.shape[1:])
                    activation = self.hooks.neuron_postbiases[name](activation)

            return (activation,) + output[1:] if is_attention and isinstance(output, tuple) else activation

        return hook_fn

    def register_hooks(self, name, module, hooks):
        is_attention = "attn" in name
        hook_fn = self.get_hook_fn(name, hooks, is_attention)
        handle = module.register_forward_hook(hook_fn)
        self.hook_handles.append(handle)

    def get_data(self, name=None, data_type=None):
        if data_type == "mask":
            return self.hooks.neuron_masks.get(name)
        elif data_type == "actadd":
            return self.hooks.neuron_actadds.get(name)
        elif data_type == "postbias":
            return self.hooks.neuron_postbiases.get(name)
        elif data_type == "input":
            return self.hooks.input_saves.get(name).activation if name in self.hooks.input_saves else None
        elif data_type == "output":
            return self.hooks.output_saves.get(name).activation if name in self.hooks.output_saves else None
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
            mlp_act = self.get_data(f"layer_{layer}_mlp_pre_out", "input")
            attn_act = self.get_data(f"layer_{layer}_attn_pre_out", "input")

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
            for point, data_type in [("pre_attn", "input"), ("pre_mlp", "input")]:
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

    mask0 = model.get_data("layer_0_pre_attn", "mask")
    new_mask = torch.ones_like(mask0.param)  # Assuming d_model is 768
    new_mask[:384] = 0  # Set first half to 0
    model.set_hook_parameter("layer_0_pre_attn", "mask", new_mask)


    # Run with new mask
    midlayer_activations = model.get_midlayer_activations(text)
    print("New MLP activations shape:", midlayer_activations["mlp"].shape)
