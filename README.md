# taker

`taker` is a Python library for studying and manipulating Language Models (LLMs), with support for multi-GPU inference, editing, and quantized inference. It provides tools for analyzing model activations and pruning different parts of the model based on those activations.

## Features

- Support for various popular LLM architectures
- Multi-GPU inference and editing
- Quantized inference
- Activation analysis and model pruning
- Hooks for manipulating model behavior

## Installation

```bash
pip install taker
```

## Supported Models

- EleutherAI's Pythia and GPT-NeoX
- Meta's OPT, Galactica, Llama, Llama 2, and Llama 3 models
- MistralAI's Mistral 7B
- OpenAI's GPT-2
- RoBERTa
- Google's Vision Transformer (ViT)
- Google's Gemma and Gemma 2 models

## Quick Start

### Loading a Model

```python
from taker import Model

# Load a model from Hugging Face
# You can choose your level of quantization
# Note that some do not support multi-GPU, so use device_map='cuda:0'
dtype = "fp16" # "fp32", "bf16", "hqq8", "int8", "hqq4", "nf4", "int4"
model = Model("facebook/opt-125m", dtype=dtype)

# Access model attributes
print(f"Number of layers: {model.cfg.n_layer}")
print(f"Model width: {model.cfg.d_model}")
print(f"MLP width: {model.cfg.d_mlp}")
print(f"Number of heads: {model.cfg.n_heads}")
print(f"Head size: {model.cfg.d_head}")
print(f"Vocabulary size: {model.cfg.d_vocab}")
```

### Text Generation

```python
prompt = "I am hungry and want to"
output = model.generate(prompt, num=10)
print(output)
```

### Analyzing Residual Stream Activations

```python
residual_stream = model.get_residual_stream("I am hungry")
print(residual_stream.shape)
# Output: torch.Size([13, 5, 288])
# Format: [n_layer, n_tokens, d_model]
# Layers: [input], [attn_0_output], [mlp_0_output], [attn_1_input], ...

residual_stream_decoder = model.get_residual_stream_decoder("I am hungry")
print(residual_stream_decoder.shape)
# Output: torch.Size([7, 5, 288])
# Format: [n_layer, n_tokens, d_model]
# Layers: [decoder_0_input], [decoder_1_input], ..., [decoder_n_output]
```

## Model Maps
## Model Maps

Model maps in `taker` provide a unified interface for interacting with different model architectures, abstracting away implementation details and allowing consistent access across various models.

### Example Usage

Here's an expanded look at how you can use model maps in `taker`:

```python
from taker import Model

# Initialize a model
model = Model("facebook/opt-125m")

# Accessing high-level model components
embedding = model["embed"]
unembed = model["unembed"]
ln_final = model["ln_final"]

# Working with layers
layer_0 = model.layers[0]
layer_1 = model.layers[1]

# Accessing attention components
attn_weights_q = layer_0["attn.W_Q"]
attn_weights_k = layer_0["attn.W_K"]
attn_weights_v = layer_0["attn.W_V"]
attn_weights_o = layer_0["attn.W_O"]

# Accessing MLP components
mlp_in_weights = layer_0["mlp.W_in"]
mlp_out_weights = layer_0["mlp.W_out"]

# Accessing normalization layers
attn_ln = layer_0["attn.ln_in"]
mlp_ln = layer_0["mlp.ln_in"]

# Modifying model components
import torch

# Change the embedding for a specific token
model["embed.W_E"][1000] = torch.randn_like(model["embed.W_E"][1000])

# Zero out some attention weights
layer_0["attn.W_Q"][:, :10] = 0

# Print model configuration
print(f"Model has {len(model.layers)} layers")
print(f"Hidden size: {model.cfg.d_model}")
print(f"Number of attention heads: {model.cfg.n_heads}")

# Accessing and modifying biases (if present in the model)
if "attn.b_Q" in layer_0:
    attn_bias_q = layer_0["attn.b_Q"]
    layer_0["attn.b_Q"] += 0.1  # Add a small bias

# Iterating over all layers
for i, layer in enumerate(model.layers):
    print(f"Layer {i} attention output shape: {layer['attn.W_O'].shape}")
```

This example demonstrates how to access and modify various components of the model using the model map interface. It shows operations on embeddings, attention weights, MLP weights, and layer normalization parameters.

### Types of Model Maps

`taker` uses two types of model maps:

1. **Model-level maps**: For high-level model components.
2. **Layer-level maps**: For the internal structure of individual layers.

### Customization

Advanced users can extend model maps for new architectures or modify existing
ones. For this, see `src/taker/model_maps.py`.

## Advanced Hook Operations

`taker` provides powerful hooks for manipulating model behavior. Here's an in-depth look at Activation Addition, Neuron Replacement, and related operations.

### Activation Addition

Activation Addition allows you to modify neuron activations at specific positions in the input sequence.

```python
from taker import Model
import torch

model = Model("facebook/opt-125m")

# Add a custom activation to a specific token position
custom_activation = torch.randn(model.cfg.d_mlp)
model.hooks["mlp_pre_out"][2].add_token(token_index=0, value=custom_activation)

# Add activations for multiple tokens
token_activations = torch.randn(5, model.cfg.d_mlp)  # 5 tokens
for i in range(5):
    model.hooks["mlp_pre_out"][2].add_token(token_index=i, value=token_activations[i])

# Reset the activation additions
model.hooks["mlp_pre_out"][2].reset()

# Manually set all activation additions
all_token_activations = torch.randn(10, model.cfg.d_mlp)  # 10 tokens
model.hooks["mlp_pre_out"][2].set_actadd(all_token_activations)

# Restart the token counter (useful for processing multiple sequences)
model.hooks["mlp_pre_out"][2].restart()
```

### Neuron Replacement

Neuron Replacement allows you to completely replace neuron activations at specific positions.

```python
# Replace neuron activations at a specific token index
replacement_activation = torch.randn(model.cfg.d_mlp)
model.hooks.neuron_replace["layer_2_mlp_pre_out"].add_token(token_index=1, value=replacement_activation)

# Replace activations for multiple tokens
for i in range(3):
    replacement = torch.randn(model.cfg.d_mlp)
    model.hooks.neuron_replace["layer_2_mlp_pre_out"].add_token(token_index=i, value=replacement)

# Reset all neuron replacements
model.hooks.neuron_replace["layer_2_mlp_pre_out"].reset()

# Restart the token counter
model.hooks.neuron_replace["layer_2_mlp_pre_out"].restart()

# Reset all neuron replace hooks across the model
model.hooks.reset_neuron_replace()
```

### Neuron Masking

Neuron Masking allows you to selectively zero out certain neurons.

```python
# Create a mask for MLP neurons in layer 2
neurons_to_keep = torch.ones(model.cfg.d_mlp)
neurons_to_keep[:10] = 0  # Mask the first 10 neurons

model.hooks["mlp_pre_out"][2].delete_neurons(keep_indices=neurons_to_keep)

# Manually set the mask
new_mask = torch.randint(0, 2, (model.cfg.d_mlp,)).float()
model.hooks["mlp_pre_out"][2]["mask"].set_mask(new_mask)

# Set an offset for masked neurons
offset = torch.randn(model.cfg.d_mlp)
model.hooks["mlp_pre_out"][2]["mask"].set_offset(offset)
```

### Neuron Offsetting

Neuron Offsetting allows you to add a constant offset to neuron activations.

```python
# Set an offset for all neurons in a layer
offset = torch.randn(model.cfg.d_mlp)
model.hooks["mlp_pre_out"][2]["offset"].set_offset(offset)

# Set offsets for all layers at once
all_layer_offsets = torch.randn(model.cfg.n_layers, model.cfg.d_mlp)
model.hooks["mlp_pre_out"].set_offsets(all_layer_offsets)
```

### Collecting Activations

You can use hooks to collect activations from specific parts of the model.

```python
# Enable collection for specific components and layers
model.hooks.enable_collect_hooks(components=["mlp_pre_out", "attn_pre_out"], layers=[0, 1, 2])

# Run a forward pass
_ = model.get_logits("Hello, world!")

# Retrieve collected activations
mlp_activations = model.hooks["mlp_pre_out"]["collect"]
attn_activations = model.hooks["attn_pre_out"]["collect"]

# Disable all collect hooks
model.hooks.disable_all_collect_hooks()
```

### Global Hook Operations

You can perform operations on all hooks of a certain type across the model.

```python
# Delete neurons across all MLP layers
remove_indices = torch.randint(0, 2, (model.cfg.n_layers, model.cfg.d_mlp)).bool()
model.hooks.delete_mlp_neurons(remove_indices)

# Delete neurons in attention layers
attn_remove_indices = torch.randint(0, 2, (model.cfg.n_layers, model.cfg.n_heads * model.cfg.d_head)).bool()
model.hooks.delete_attn_neurons(attn_remove_indices)
```

These advanced hook operations provide fine-grained control over the model's
behavior, allowing for detailed analysis and manipulation of the model's
internal representations.

## Example: Pruning Based on Capabilities

```python
from taker.data_classes import PruningConfig
from taker.prune import run_pruning

config = PruningConfig(
    wandb_project="my_project",
    model_repo="facebook/opt-125m",
    token_limit=1000,
    run_pre_test=True,
    ff_scoring="abs",
    ff_frac=0.02,
    ff_eps=0.001,
    attn_scoring="abs",
    attn_frac=0.00,
    attn_eps=1e-4,
    focus="pile_codeless",
    cripple="code",
)

pruned_model, history = run_pruning(config)
```

## Contributing

Contributions to `taker` are welcome! Please check out our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.