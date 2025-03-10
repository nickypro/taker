
from taker import Model
import torch
from taker.activations import get_midlayer_data
from taker.prune import prune_and_evaluate
from taker.data_classes import PruningConfig

#TODO: 
# 1. copy pruning code in here
# 2. add hooks and other things needed for all the sae stuff
# 3. grab the sae activations

hook_config = """
pre_decoder: sae_encode, mask, collect, sae_decode
mlp_pre_out: collect
"""
c = PruningConfig("nickypro/tinyllama-15m",
    attn_mode="pre-out", do_attn_mean_offset=False, use_accelerator=False,
    ff_frac=0.0, attn_frac=0.0, sae_frac=0.1,
    token_limit=100, focus="civil", cripple="toxic", wandb_entity="seperability", recalculate_activations=False,
    wandb_project="bens-tests", wandb_run_name="test notebook2", n_steps=10, scoring_normalization="peak_centered")
m = Model("gpt2", hook_config=hook_config)

for layer in range(m.cfg.n_layers):
    sae_hook = m.hooks.neuron_sae_encode[f"layer_{layer}_pre_decoder"]
    sae_hook.load_sae_from_pretrained("gpt2-small-res-jb", f"blocks.{layer}.hook_resid_pre")

m.hooks.enable_collect_hooks(["mlp_pre_out"], run_assert=True)
m.hooks.enable_collect_hooks(["pre_decoder"], run_assert=True)

focus_data = get_midlayer_data( m, "civil", 10, collect_sae=True, calculate_sae=True, collect_attn=False, collect_ff=True, calculate_attn=False, calculate_ff=False)
cripple_data = get_midlayer_data( m, "toxic", 10, collect_sae=True, calculate_sae=True, collect_attn=False, collect_ff=True, calculate_attn=False, calculate_ff=False)

with torch.no_grad():
    for i in range(c.n_steps):
        print(f"Step {i}")
        data = prune_and_evaluate(m, c, focus_data, cripple_data, i)