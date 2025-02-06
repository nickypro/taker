
# MLP hooks gemma-2-2b (non-it)
# note: 26x layers with 16k width, d_in=2306, d_sae=16384
# total parameters: 2 * 26 * 2306 * 16384 = 1_964_638_208

import torch
from tqdm import tqdm
from taker import Model
torch.set_grad_enabled(False)

hook_config = """
post_mlp: sae_encode, mask, collect, sae_decode
post_attn: sae_encode, mask, collect, sae_decode
pre_decoder: sae_encode, mask, collect, sae_decode
"""
m = Model("google/gemma-2-2b", limit=1000, hook_config=hook_config)

# evaluate the model
from taker.eval import evaluate_all
evaluate_all(m, 1e4, ["pile"])

# show that the collection hooks are working, and that the SAE is not loaded
m.hooks.enable_collect_hooks(["post_mlp"])
m.get_outputs_embeds("Hello, world!")
for layer in range(m.cfg.n_layers):
    act = m.hooks.collects[f"layer_{layer}_post_mlp"].activation
    print(act.shape)
m.hooks.disable_all_collect_hooks()

# load the SAE
for layer in tqdm(range(m.cfg.n_layers), desc="Loading SAE"):
    sae_hook = m.hooks.neuron_sae_encode[f"layer_{layer}_post_mlp"]
    sae_hook.load_sae_from_pretrained("gemma-scope-2b-pt-mlp-canonical", f"layer_{layer}/width_16k/canonical")

# test that the SAE collection hooks are good
m.hooks.enable_collect_hooks(["post_mlp"])
m.get_outputs_embeds("Hello, world!")
for layer in range(m.cfg.n_layers):
    act = m.hooks.collects[f"layer_{layer}_post_mlp"].activation
    print(act.shape)
m.hooks.disable_all_collect_hooks()

# evaluate the model again
evaluate_all(m, 1e4, ["pile"])
