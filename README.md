![Tools for Lanugage Model Activations](https://github.com/nickypro/taker)

# taker

My basic library for studying LLMs, with support for Multi-GPU inference and
editing, as well as quantilised inference (not editing yet).
This includes functions for analysing the activations of the models for
different inputs, and for pruning different parts of the model based on those
activations.

The currently tested list of models is:
- EleutherAI's Pythia
- EleutherAI's GPT-NeoX
- Meta Opt
- Meta Galactica
- Meta Llama / Llama 2
- MistralAI Mistral 7B
- GPT-2
- RoBERTa
- Google's Vision Transformer (ViT)

For check out the [examples folder](https://github.com/nickypro/separability/blob/main/examples) to see in more detail how the library can be used.

## Using the Model

To load up the model, simply do:
```
from taker import Model

# choose any model from huggingface, though note that not all will be supported
# dtype can be anything from int4, int8, fp16, fp32, fp64
m = Model("nickypro/tinyllama-15m", dtype="fp16")

# You can access different attributes of the model
print(m.cfg.n_layer) # model has 6 layers
print(m.cfg.d_model) # 288 model width
print(m.cfg.d_mlp)   # 768 mlp width
print(m.cfg.n_heads) # 6 heads
print(m.cfg.d_head)  # 48 head size
print(m.cfg.d_vocab) # 32000 token options
```

Once the model is loaded, you can try using it for some generative tasks:
```
output = m.generate("I am hungry and want to", num=10)
print(output)
# ('I am hungry and want to', ' eat something." I said, "I am sorry')
```

If you want to inspect the activations, you can try to look at it's residual stream activations:
```
m.get_residual_stream("I am hungry")
print(m.shape)
# torch.Size([13, 5, 288])
# formatted as [n_layer, n_tokens, d_model]
# where n_layers is [input], [attn_0_output], [mlp_0_output], [attn_1_input], ...
```

If you want to manipulate the model, you can use one of the implemented hooks:
- `NeuronMask`, used to set neuron activations to zero (or some other constant).
- `NeuronActAdd`, used to do "activation addition" to different tokens.
- `NeuronPostBias`, used to add a bias to the ouputs when the huggingface model does not otherwise support it.

For example, you can set the masks of some mlp neurons in layer 2 to zero:
```
import torch
neurons_to_keep = torch.ones(m.cfg.d_mlp)
neurons_to_keep[:10] = 0

m.masks["mlp_pre_out"][2].delete_neurons(keep_indices=neurons_to_keep)
```

You can also make it so that the neurons are not set to zero, but rather some other value
```
neuron_cenetering_vector = torch.randn(m.cfg.d_mlp)


m.masks["mlp_pre_out"][2].set_offset(neuron_centering_vector)
```


## Model Map

Taker works by using a map, `ModelMap` which converts a standardised query like `mlp.W_out` to
the rekevant saved component. For example:
```
mlp_weights = m.layers[2]["mlp.W_out"]

mlp_weights[..., 0] = 3.14159

m.layers[2]["mlp.W_out"] = mlp_weights
```


## Pruning based on Capabilities

For a full example, see `src/examples/prune_30.py`.

The simple example is:
```
from taker.data_classes import PruningConfig
from taker.parser import cli_parser
from taker.prune import run_pruning

# Configure initial model and tests
c = PruningConfig(
    wandb_project = "testing",
    model_repo   = "facebook/opt-125m",
    token_limit  = 1000,
    run_pre_test = True,

    # Removals parameters
    ff_scoring = "abs"
    ff_frac   = 0.02,
    ff_eps    = 0.001,
    attn_scoring = "abs",
    attn_frac = 0.00,
    attn_eps  = 1e-4,

    # Eval
    focus     = "pile_codeless",
    cripple   = "code",
    additional_datasets=tuple(),
)

# optionally, use parser to get CLI arguments.
# c, args = cli_parser(c)

# Run the iterated pruning
model, history = run_pruning(c)

```

## model.py
This defines a wrapper function that encapsulates the HuggingFace implementation of Meta OPT.
To get the model, simply run:

```
from taker import Model

m = Model("facebook/opt-125m", limit=1000)
```

Where you can provide any of the model sizes that are pre-trained for OPT, and the token limit must be smaller than the max token length that the model is able to handle.

Next, you can run the model to do 2 tokens of predictions, by, for example, running:
```
text = 'Hello, my name is'
inpt, output = opt.predict( text, num=2 )
```

We can look at the residual stream of how the output changes over time.
```
residual_stream = opt.get_residual_stream( text )
```
This will return a tensor of size `2 + 2*n_layers`.
i.e:
- the input (w/ positional encoding)
- n attention layer outputs
- n feed forward layer outputs
- the final output

If we want just the output of the attention / feed forward layers, we can instead look at the activations:
```
inpt, attn_out, ff_out, output = opt.get_text_activations( text )
```
or alternatively:
```
inpt, attn_out, ff_out, output = opt.get_text_activations( residual_stream=residual_stream )
```

To get the activations for the input text at all of the MLP mid layers, we can look at:
`opt.get_ff_key_activations( text )` or `opt.get_ff_key_activations( residual_stream=residual_stream )`.

## texts.py
Has some basic tools for loading the text datasets I am using:
- 'pile', ( EleutherAI's 'The Pile' dataset)
- 'pile-codeless' (Pile without GitHub)
- 'code' (CodeParrot's 'github-code' dataset)
- 'python', (Subset of only python)
- 'wiki', (WikiText)
- 'civil', (Civil comments with toxicity < 0.2)
- 'toxic', (Civil comments with toxicity > 0.8)
- ...

## activations.py
Has code specific to the datasets I am using to analyze and attempt to remove capabilities from the models.

