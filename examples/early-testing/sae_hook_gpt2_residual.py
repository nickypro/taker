
# Pre-residual hooks
from taker import Model

hook_config = """
pre_decoder: sae_encode, collect, sae_decode
"""
m = Model("gpt2", hook_config=hook_config)

for layer in range(m.cfg.n_layers):
    sae_hook = m.hooks.neuron_sae_encode[f"layer_{layer}_pre_decoder"]
    sae_hook.load_sae_from_pretrained("gpt2-small-res-jb", f"blocks.{layer}.hook_resid_pre")

m.hooks.enable_collect_hooks(["pre_decoder"])

m.get_outputs_embeds("Hello, world!")

for layer in range(m.cfg.n_layers):
    act = m.hooks.collects[f"layer_{layer}_pre_decoder"].activation
    print(act.shape)
