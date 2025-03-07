""" This file cointains hook modules which are attached to the model.
"""

from typing import Dict, List
import torch
from torch import Tensor
import warnings

######################################################################################
# Define Hooks on Neuron Activations we can use. EG: Neuron Mask Class
######################################################################################

# Model way of storing hooks that are currently in use
class HookConfig:
    def __init__(self, n_layers=None):
        self.hook_points = {
            "pre_attn": {},
            "attn_pre_out": {},
            "post_attn": {},
            "pre_mlp": {},
            "mlp_pre_out": {},
            "post_mlp": {},
            "pre_decoder": {},
            "post_decoder": {},
        }
        self.n_layers = n_layers

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
    """Class that holds all hooks."""
    def __init__(self, hook_config):
        self.hook_config: HookConfig = hook_config
        if isinstance(hook_config, str):
            self.hook_config = HookConfig().from_string(hook_config)
        self.collects = {}
        self.neuron_masks = {}
        self.neuron_actadds = {}
        self.neuron_postbiases = {}
        self.neuron_offsets = {}
        self.neuron_unoffsets = {}
        self.neuron_replace = {}
        self.neuron_whiten = {}
        self.neuron_unwhiten = {}
        self.neuron_sae_encode = {}
        self.neuron_sae_decode = {}
        self.handles: list = []

    def __str__(self):
        attributes = []
        for attr, value in self.__dict__.items():
            attributes.append(f"{attr}: {value}")

        return f"ActiveHooks({', '.join(attributes)})"

    def __getitem__(self, component: str):
        return HookMapComponent(self, component)

    @property
    def hooks_raw(self):
        return {
            "collect":  self.collects,
            "mask":     self.neuron_masks,
            "actadd":   self.neuron_actadds,
            "postbias": self.neuron_postbiases,
            "offset":   self.neuron_offsets,
            "unoffset": self.neuron_offsets,
            "replace":  self.neuron_replace,
            "whiten":   self.neuron_whiten,
            "unwhiten": self.neuron_unwhiten,
            "sae_encode": self.neuron_sae_encode,
            "sae_decode": self.neuron_sae_decode,
        }

    @property
    def all_hooks(self):
        hook_list = []
        for _hook_type_dict in self.hooks_raw.values():
            for _location, hook in _hook_type_dict.items():
                hook_list.append(hook)
        return hook_list

    def get_hook_fn(self, hook_type, name, activation, device, dtype):
        _hooks = self.hooks_raw[hook_type]
        if name not in _hooks:
            if hook_type == "collect":
                _hooks[name] = NeuronSave()
            elif hook_type == "mask":
                _hooks[name] = NeuronMask(activation.shape[2:]).to(device, dtype)
            elif hook_type == "actadd":
                _hooks[name] = NeuronActAdd(device, dtype)
            elif hook_type == "postbias":
                _hooks[name] = NeuronPostBias(activation.shape[2:]).to(device, dtype)
            elif hook_type == "offset":
                _hooks[name] = NeuronOffset(activation.shape[2:]).to(device, dtype)
            elif hook_type == "unoffset":
                assert name in self.neuron_offsets
                _hooks[name] = self.neuron_offsets[name]
            elif hook_type == "replace":
                _hooks[name] = NeuronReplace(device, dtype)
            elif hook_type == "whiten":
                _hooks[name] = NeuronWhiten(activation.shape[2:]).to(device, dtype)
            elif hook_type == "unwhiten":
                assert name in self.neuron_whiten
                _hooks[name] = self.neuron_whiten[name]
            elif hook_type == "sae_encode":
                _hooks[name] = NeuronSAE(device, dtype)
            elif hook_type == "sae_decode":
                assert name in self.neuron_sae_encode
                _hooks[name] = self.neuron_sae_encode[name]

        curr_hook = _hooks[name]
        if hook_type == "unoffset":
            return curr_hook.undo
        if hook_type == "unwhiten":
            return curr_hook.undo
        if hook_type == "sae_encode":
            return curr_hook.encode
        if hook_type == "sae_decode":
            return curr_hook.decode
        return curr_hook

    # Generic Helper functions for get/set hook data
    def get_data(self, name=None, data_type=None):
        if data_type == "collect":
            # collect is desctructively gotten (to save memory)
            if name in self.collects:
                data = self.collects[name].activation
                self.collects[name].activation = None
                return data
            return None
        elif data_type in self.hooks_raw:
            return self.hooks_raw[data_type][name]
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def set_hook_parameter(self, name, param_type, value):
        if param_type == "mask":
            self.neuron_masks[name].set_mask(value)
        elif param_type == "actadd":
            self.neuron_actadds[name].set_actadd(value)
        elif param_type == "postbias":
            self.neuron_postbiases[name].param.data = value
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def get_layer_names(self, component, layers=None):
        layers = range(self.hook_config.n_layers) if layers is None else layers
        return [f"layer_{i}_{component}" for i in layers]

    def get_all_layer_data(self, component, data_type, layers=None):
        layer_names = self.get_layer_names(component, layers)
        return [self.get_data(name, data_type) for name in layer_names]

    def set_all_layer_parameters(self, component, param_type, values, layers=None):
        layer_names = self.get_layer_names(component, layers)
        for name, value in zip(layer_names, values):
            self.set_hook_parameter(name, param_type, value)

    # Methods for specific hook types
    def get_all_layer_activations(self, component: str, layers: List[int] | None =None):
        layer_names = self.get_layer_names(component, layers)
        return [self.get_data(name, "collect") for name in layer_names]

    def disable_all_collect_hooks(self):
        for name, hook in self.collects.items():
            hook.enabled = False

    def enable_collect_hooks(self, components=None, layers=None, run_assert=False):
        if components is None:
            components = self.hook_config.hook_points.keys()
        if isinstance(components, str):
            components = [components]
        if layers is None:
            layers = range(self.hook_config.n_layers)
        if isinstance(layers, int):
            layers = [layers]

        for component in components:
            for layer in layers:
                hook_name = f"layer_{layer}_{component}"
                if run_assert:
                    assert hook_name in self.collects
                if hook_name in self.collects:
                    self.collects[hook_name].enabled = True

    def delete_mlp_neurons(self, remove_indices, layer: int = None):
        return self["mlp_pre_out"].delete_neurons(remove_indices, layer)

    def delete_attn_neurons(self, remove_indices, layer: int = None):
        return self["attn_pre_out"].delete_neurons(remove_indices, layer)

    def reset_neuron_replace(self):
        [h.reset() for h in self.neuron_replace.values()]

    def reset(self):
        [h.reset() for h in self.all_hooks]

class HookMapComponent:
    def __init__(self, hooks: HookMap, component: str):
        self.hooks: HookMap = hooks
        self.component: str = component

    def __getitem__(self, data_type):
        layers = None
        if isinstance(data_type, tuple):
            data_type, layer = data_type
            layers = [layer]
        if data_type == "collect":
            data = self.hooks.get_all_layer_activations(self.component, layers)
        elif data_type in ["mask", "actadd", "postbias", "offset"]:
            data = self.hooks.get_all_layer_data(self.component, data_type, layers)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        if len(data) == 1:
            return data[0]
        return data

    def __setitem__(self, data_type, value):
        if data_type in ["mask", "actadd", "postbias", "offset"]:
            self.hooks.set_all_layer_parameters(self.component, data_type, value)
        else:
            raise ValueError(f"Cannot set data type: {data_type}")

    # Hook-specific functions
    def delete_neurons(self, remove_indices, layer: int = None):
        def delete_layer_neurons(nn_mask: NeuronMask, rm_idx):
            device, dtype = nn_mask.param.device, bool
            rm_idx = rm_idx.to(device, dtype).reshape(nn_mask.param.shape)
            keep_indices = torch.logical_not(rm_idx)
            nn_mask.delete_neurons(keep_indices)
        nn_masks = self["mask"]
        if layer is not None:
            return delete_layer_neurons(nn_masks[layer], remove_indices)
        for layer, nn_mask in enumerate(nn_masks):
            delete_layer_neurons(nn_mask, remove_indices[layer])

    def set_offsets(self, offset_val, layer:int = None):
        def set_layer_offset(nn_offset, val):
            device, dtype = nn_offset.param.device, nn_offset.param.dtype
            val = val.to(device, dtype).reshape(nn_offset.shape)
            nn_offset.set_offset(val)
        neuron_offsets = self["offset"]
        if layer is not None:
            return set_layer_offset(neuron_offsets[layer], offset_val)
        for layer, neuron_offset in enumerate(neuron_offsets):
            set_layer_offset(neuron_offset, offset_val[layer])

#####################################################################################
# Define Hook Classes
#####################################################################################

class NeuronFunctionList(torch.nn.Module):
    """ Class for storing all the Neuron Masks"""

    def __init__(self, neuron_function_list):
        super(NeuronFunctionList, self).__init__()
        # list all the Neuron Masks as a torch accessible list of parameters
        self.layers = torch.nn.ModuleList(neuron_function_list)

    def forward(self, x):
        "Given [layer, activation], returns all the activations masked for each layer."
        y = []
        for act in x:
            y.append(act)
        return torch.stack(y)

    def __getitem__(self, index: int):
        return self.layers[index]

class NeuronSave(torch.nn.Module):
    """Class that saves activations to self"""

    def __init__(self):
        super().__init__()
        self.activation = None
        self.enabled = False
        self.concat_mode = False

    def forward(self, x: Tensor):
        if self.enabled and self.concat_mode and self.activation is not None:
            self.activation = torch.concat([self.activation, x], dim=1) # batch token *dims
        elif self.enabled:
            self.activation = x
        return x

    def reset(self):
        self.activation = None

# Neuron Mask. EG: [a, b, c] -> [a, 0, c]
class NeuronMask(torch.nn.Module):
    """Class for creating a mask for a single layer of a neural network."""

    def __init__(self, shape, act_fn: str = "step"):
        super(NeuronMask, self).__init__()
        self.act_fn = act_fn
        self.shape: torch.Size = None
        self.param: torch.nn.Parameter = None
        self.offset: torch.nn.Parameter = None
        self.reinit_hook(shape=shape)

    def check_shapes_match(self, x):
        curr_shape  = torch.Size(self.shape)
        input_shape = torch.Size(x.shape[-len(curr_shape):])
        return curr_shape == input_shape, f"{curr_shape} vs {input_shape} (from {x.shape})"

    def reinit_hook(self, x=None, shape=None):
        # batch, token, (d_model or otherwise)
        if x is not None:
            new_shape, new_dtype = x.shape[2:], x.dtype
        elif shape is not None:
            new_shape, new_dtype = shape, torch.float32
        else:
            raise ValueError("Either x or shape must be provided to init NeuronMask")

        self.shape = new_shape
        vec = torch.ones(new_shape, dtype=new_dtype)
        # initialize mask as nn.Parameter of ones
        if self.act_fn == "sigmoid":
            vec[...] = torch.inf
        self.param = torch.nn.Parameter(vec)
        self.offset = torch.nn.Parameter(torch.zeros_like(vec))

    def get_mask(self):
        # if step, we want heaviside step function. ie: mask = mask > 0
        if self.act_fn == "step":
            return (self.param > 0)
        if self.act_fn == "sigmoid":
            return torch.sigmoid(self.param)
        if self.act_fn == "tanh":
            return torch.tanh(self.param)
        if self.act_fn == "relu":
            return torch.relu(self.param)
        if callable(self.act_fn):
            return self.act_fn(self.param)
        raise ValueError(f"Unknown activation function: {self.act_fn}")

    def get_offset(self, x):
        mask = self.get_mask().to(x.dtype)
        inv_mask = 1 - mask
        offset = self.offset * inv_mask
        return offset

    def set_mask(self, new_mask: Tensor):
        params: Dict[str, Tensor] = self.state_dict()
        params["param"] = new_mask.view(self.shape)
        self.load_state_dict(params)

    def set_offset(self, offset: Tensor):
        params: Dict[str, Tensor] = self.state_dict()
        params["offset"] = offset.view(self.shape)
        self.load_state_dict(params)

    def delete_neurons(self, keep_indices: Tensor):
        params: Dict[str, Tensor] = self.state_dict()
        params["param"] = params["param"] * keep_indices.to(params["param"].device)
        self.load_state_dict(params)

    def get_inverse_mask(self, x):
        mask = self.get_mask().to(x.dtype)
        return 1 - mask

    def inverse_mask(self, x, offset=False):
        inv_mask = self.get_inverse_mask(x)
        # TODO: allow inverse mask to work with offset. Not sure when needed though.
        assert offset == False
        return x * inv_mask

    def forward(self, x):
        is_match, msg = self.check_shapes_match(x)
        if not is_match:
            print(f"Shape mismatch: {msg}, reinitialising mask hook")
            self.reinit_hook(x)
        self.to(x.device)
        mask = self.get_mask()
        offset = self.get_offset(x)
        return x * mask + offset

    def __str__(self):
        return f"""NeuronMask(
        act_fn: {self.act_fn}
        param: {self.param}
        offset: {self.offset}
        )"""

    def reset(self):
        self.param.data = torch.ones_like(self.param)
        self.offset.data = torch.zeros_like(self.offset)

# Positional Neuron Activation Addition.
class NeuronActAdd(torch.nn.Module):
    """ MVP for Position Dependant Neuron Activation Offseting.
    Inspired by ActAdd paper.
    # EG: [[a], [b], [c], [d], ...] -> [[a+x], [b'+y], [c'+z], [d'], ...]
    # where a,b,c,d are the "real activations", x,y,z are the added vectors.

    the "autorestart" works by assuming the first input is of size > 1 and is cached.
    """
    def __init__(self, device, dtype, autorestart: bool=True):
        super(NeuronActAdd, self).__init__()
        self.device = device
        self.dtype = dtype

        # leave as uninitialised initially
        self.shape = [0]
        self.param = torch.nn.Parameter(torch.zeros(self.shape))
        self.max_tokens = 0

        self.tokens_seen = 0
        self.autorestart = autorestart

    def restart(self):
        self.tokens_seen = 0

    def reset(self):
        self.param = torch.nn.Parameter(torch.zeros(self.shape))
        self.max_tokens = 0
        self.restart()

    def set_actadd(self, offset: Tensor):
        self.shape = offset.shape
        self.max_tokens = self.shape[0]
        self.param = torch.nn.Parameter(
            offset.to(device=self.device, dtype=self.dtype)
        )
        self.restart()

    def to(self, device=None, dtype=None, *args, **kwargs):
        super(NeuronActAdd, self).to(device, dtype, *args, **kwargs)
        if not device is None:
            self.device = device
        if not dtype is None:
            self.dtype = dtype

    def forward(self, x):
        # load input vector (do not modify in place)
        x = x.clone()
        n_new_tokens = x.shape[1]
        # autoreset assuming cache goes like [0,1,2,3], [4], [5], ...
        if self.autorestart and n_new_tokens > 1:
            self.restart()
        # if we used up all the add act vectors we can skip
        if self.tokens_seen >= self.max_tokens:
            return x
        # othewise, do the act add stuff with remaining vectors left
        tokens_left = self.max_tokens - self.tokens_seen
        n_tokens    = min([tokens_left, n_new_tokens])

        x[:, :n_tokens]     += self.param[ self.tokens_seen:self.tokens_seen+n_tokens]
        self.tokens_seen += n_tokens
        return x

class NeuronReplace(torch.nn.Module):
    """ Replace neuron activations at specific token indices
    # EG: [[a], [b], [c], [d], ...] -> [[a], [b], [X], [d], ...]

    the "autorestart" works by assuming the first input is of size > 1 and is cached.
    """
    def __init__(self, device, dtype, autorestart: bool=True):
        super(NeuronReplace, self).__init__()
        self.device = device
        self.dtype = dtype

        # Use a ParameterDict to store parameters
        self.param = torch.nn.ParameterDict()
        self.max_tokens = 0

        self.tokens_seen = 0
        self.autorestart = autorestart

    def restart(self):
        self.tokens_seen = 0

    def reset(self):
        self.max_tokens = 0
        self.param.clear()
        self.restart()

    def add_token(self, token_index, value):
        # Convert the value to a Parameter before adding
        self.param[str(token_index)] = torch.nn.Parameter(value.to(self.device, self.dtype))
        self.max_tokens = max([self.max_tokens, token_index+1])

    def to(self, device=None, dtype=None, *args, **kwargs):
        super(NeuronReplace, self).to(device, dtype, *args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype

    def forward(self, x):
        # load input vector (do not modify in place)
        x = x.clone()
        n_new_tokens = x.shape[1]
        # autorestart assuming cache goes like [0,1,2,3], [4], [5], ...
        if self.autorestart and n_new_tokens > 1:
            self.restart()
        # if we used up all the add act vectors we can skip
        if self.tokens_seen >= self.max_tokens:
            return x

        # othewise, do the neuron replacement stuff with remaining vectors left
        tokens_left = self.max_tokens - self.tokens_seen
        n_tokens    = min([tokens_left, n_new_tokens])
        for local_index in range(n_tokens):
            token_index = local_index + self.tokens_seen
            if str(token_index) in self.param:
                x[:, local_index] = self.param[str(token_index)].reshape(x.shape[2:])

        self.tokens_seen += n_tokens
        return x

class NeuronOffset(torch.nn.Module):
    def __init__(self, shape):
        super(NeuronOffset, self).__init__()
        self.shape = shape
        _vec = torch.zeros(shape, dtype=torch.float32)
        self.param = torch.nn.Parameter(_vec)

    def set_offset(self, offset: Tensor):
        params: Dict[str, Tensor] = self.state_dict()
        params["param"] = offset.view(self.shape)
        self.load_state_dict(params)

    def forward(self, x):
        self.to(x.device)
        return x + self.param

    def undo(self, x):
        return x - self.param

    def reset(self):
        self.param.data = torch.zeros_like(self.param)

# Neuron Post Bias (EG: For SVD and stuff) out -> out + bias
class NeuronPostBias(torch.nn.Module):
    """Container for holding after-the-fact biases in the model."""

    def __init__(self, shape):
        super(NeuronPostBias, self).__init__()
        self.shape = shape
        _vec = torch.zeros(shape, dtype=torch.float32)
        self.param = torch.nn.Parameter(_vec)

    def get_bias(self, x):
        shape = x.shape
        bias  = self.param
        if self.shape == shape:
            return bias
        try:
            bias = bias.view(shape[-1]) # normal shape
        except:
            bias = bias.view(-1, shape[-1]) # multi head attention shape
        return bias

    def forward(self, x):
        return x + self.get_bias(x)

    def reset(self):
        self.param.data = torch.zeros_like(self.param)

class NeuronWhiten(torch.nn.Module):
    def __init__(self, shape):
        super(NeuronWhiten, self).__init__()
        self.shape = shape
        self.d_model = self.shape.numel()
        self.offset   = torch.nn.Parameter(torch.zeros(self.shape))
        self.rotate   = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.scale    = torch.nn.Parameter(torch.ones(self.d_model))
        self.rotate_inv = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.reset()

    def reset(self):
        self.offset.data = torch.zeros(self.shape)
        self.rotate.weight.data = torch.diag(torch.ones(self.d_model))
        self.scale.data = torch.ones(self.d_model)
        self.rotate_inv.weight.data = torch.diag(torch.ones(self.d_model))

    @property
    def unscale(self):
        return torch.clamp( self.scale ** -1, max=1e6 )

    def forward(self, x):
        x = x + self.offset
        x = x.reshape([*x.shape[:2], -1])
        x = self.rotate(x)
        x = x * self.scale
        return x

    def undo(self, x):
        x = x * self.unscale
        x = self.rotate_inv(x)
        x = x.reshape(x.shape[:2] + self.shape)
        x = x - self.offset
        return x

class NeuronSAE(torch.nn.Module):
    def __init__(self, device, dtype):
        super(NeuronSAE, self).__init__()
        self.device = device
        self.dtype = dtype
        self.sae = None
        self.sae_config = None

    def load_sae(self, sae, sae_config=None):
        self.sae = sae.to(self.device, self.dtype)
        self.sae_config = sae_config

    def load_sae_from_pretrained(self, release, sae_id):
        try:
            from sae_lens import SAE
        except ImportError:
            raise ImportError("sae_lens not installed. Please install it with `pip install sae-lens`.")

        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = release,
            sae_id =  sae_id,
            device = self.device,
        )
        self.load_sae(sae, cfg_dict)

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        if self.sae is None:
            warnings.warn("SAE not loaded. Call load_sae() first.")
            return x
        return self.sae.encode(x)

    def decode(self, x):
        if self.sae is None:
            return x
        return self.sae.decode(x)

    def reset(self):
        self.sae = None
        self.sae_config = None
