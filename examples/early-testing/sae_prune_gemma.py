import wandb
import torch
import sys
sys.path.append('/root/taker/src')

from taker import Model
from taker.activations import get_midlayer_data
from taker.prune import prune_and_evaluate, evaluate_all
from taker.data_classes import PruningConfig, RunDataHistory, RunDataItem

hook_config = """
post_mlp: sae_encode, mask, collect, sae_decode
mlp_pre_out: collect
"""
c = PruningConfig("doesnt matter",
    attn_mode="pre-out", do_attn_mean_offset=False, use_accelerator=False,
    ff_frac=0.0, attn_frac=0.0, sae_frac=0.2,
    token_limit=100, focus="civil", cripple="toxic", wandb_entity="seperability", recalculate_activations=False,
    wandb_project="bens-tests", wandb_run_name="gemma-2b mlp sae prune test default scoring", n_steps=10)
m = Model("google/gemma-2-2b", hook_config=hook_config)

for layer in range(m.cfg.n_layers):
    sae_hook = m.hooks.neuron_sae_encode[f"layer_{layer}_post_mlp"]
    sae_hook.load_sae_from_pretrained("gemma-scope-2b-pt-mlp-canonical", f"layer_{layer}/width_16k/canonical")

#grabbing mlp activations seems needed so things dont break (to create raw activations dict?)
m.hooks.enable_collect_hooks(["mlp_pre_out"], run_assert=True)
m.hooks.enable_collect_hooks(["post_mlp"], run_assert=True)

#TODO: seems we have to have collect_ff=True to get the activations for sae
focus_data = get_midlayer_data( m, "civil", 10, collect_sae=True, calculate_sae=True, collect_attn=False, collect_ff=True, calculate_attn=False, calculate_ff=False)
cripple_data = get_midlayer_data( m, "toxic", 10, collect_sae=True, calculate_sae=True, collect_attn=False, collect_ff=True, calculate_attn=False, calculate_ff=False)

history = RunDataHistory(c.datasets)
wandb.init(
    project=c.wandb_project,
    entity=c.wandb_entity,
    name=c.wandb_run_name,
    )
wandb.config.update(c.to_dict(), allow_val_change=True)

with torch.no_grad():
    #evaluate without pruning first
    data = RunDataItem()
    eval_out = evaluate_all(m, c.eval_sample_size, c.datasets,
                            dataset_tokens_to_skip=c.collection_sample_size)
    data.update(eval_out)
    history.add(data)

    for i in range(c.n_steps):
        print(f"Step {i}")
        data = prune_and_evaluate(m, c, focus_data, cripple_data, i)
        history.add(data)