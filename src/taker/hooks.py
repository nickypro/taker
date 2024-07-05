""" This file cointains hook modules which are attached to the model.
"""

from typing import Dict
import torch
from torch import Tensor

######################################################################################
# Define Hooks on Neuron Activations we can use. EG: Neuron Mask Class
######################################################################################

# Model way of storing hooks that are currently in use
class ActiveHooks:
    def __init__(self):
        self.collects = {}
        self.neuron_masks = {}
        self.neuron_actadds = {}
        self.neuron_postbiases = {}
        self.neuron_offsets = {}
        self.neuron_unoffsets = {}
        self.neuron_replace = {}

    def __str__(self):
        attributes = []
        for attr, value in self.__dict__.items():
            attributes.append(f"{attr}: {value}")

        return f"ActiveHooks({', '.join(attributes)})"

# Class for holding the hook classes
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

    def forward(self, x: Tensor):
        if self.enabled:
            self.activation = x.detach()
        return x

# Neuron Mask. EG: [a, b, c] -> [a, 0, c]
class NeuronMask(torch.nn.Module):
    """Class for creating a mask for a single layer of a neural network."""

    def __init__(self, shape, act_fn: str = "step"):
        super(NeuronMask, self).__init__()
        self.shape = shape
        self.act_fn = act_fn
        # initialize mask as nn.Parameter of ones
        _vec = torch.ones(shape, dtype=torch.float32)
        if self.act_fn == "sigmoid":
            _vec[...] = torch.inf
        self.param = torch.nn.Parameter(_vec)
        self.offset = torch.nn.Parameter(torch.zeros_like(_vec))

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
        mask = self.get_mask()
        offset = self.get_offset(x)
        return x * mask + offset

    def __str__(self):
        return f"""NeuronMask(
        act_fn: {self.act_fn}
        param: {self.param}
        offset: {self.offset}
        )"""

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

        # leave as uninitialised initially
        self.param = {}
        self.max_tokens = 0

        self.tokens_seen = 0
        self.autorestart = autorestart

    def restart(self):
        self.tokens_seen = 0

    def reset(self):
        self.max_tokens = 0
        self.param = {}
        self.restart()

    def add_token(self, token_index, value):
        self.param[token_index] = value.to(self.device, self.dtype)
        self.max_tokens = max([self.max_tokens, token_index+1])

    def to(self, device=None, dtype=None, *args, **kwargs):
        super(NeuronReplace, self).to(device, dtype, *args, **kwargs)
        if not device is None:
            self.device = device
        if not dtype is None:
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
        print(f"Got {n_new_tokens}, already saw {self.tokens_seen}, checking {n_tokens} out of max {self.max_tokens}")
        for local_index in range(n_tokens):
            token_index = local_index + self.tokens_seen
            if token_index not in self.param:
                continue
            x[:, local_index] = self.param[token_index].reshape(x.shape[2:])

        self.tokens_seen += n_tokens
        return x

class NeuronOffset(torch.nn.Module):
    def __init__(self, shape):
        super(NeuronOffset, self).__init__()
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
        return x + self.param

    def undo(self, x):
        return x - self.param

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
