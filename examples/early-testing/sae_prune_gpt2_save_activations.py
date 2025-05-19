import wandb
import torch
import sys
import os
from pathlib import Path
sys.path.append('/root/taker/src')

from taker import Model
from taker.activations import get_midlayer_data
from taker.prune import prune_and_evaluate, evaluate_all
from taker.data_classes import PruningConfig, RunDataHistory, RunDataItem

# Create directory for saving activations
SAVE_DIR = Path("examples/early-testing/sae_activations")
SAVE_DIR.mkdir(exist_ok=True)

hook_config = """
pre_decoder: sae_encode, mask, collect, sae_decode
mlp_pre_out: collect
attn_pre_out: collect
""" #last line is needed to recalc activations
c = PruningConfig("doesnt matter",
    attn_mode="pre-out", do_attn_mean_offset=False, use_accelerator=False,
    ff_frac=0.0, attn_frac=0.0, sae_frac=0.2,
    token_limit=100, focus="civil", cripple="toxic", wandb_entity="seperability", recalculate_activations=False,
    wandb_project="bens-tests", wandb_run_name="delete me", n_steps=10)
m = Model("gpt2", hook_config=hook_config)

for layer in range(m.cfg.n_layers):
    sae_hook = m.hooks.neuron_sae_encode[f"layer_{layer}_pre_decoder"]
    sae_hook.load_sae_from_pretrained("gpt2-small-res-jb", f"blocks.{layer}.hook_resid_pre")

#grabbing mlp activations seems needed so things dont break (to create raw activations dict?)
m.hooks.enable_collect_hooks(["mlp_pre_out"], run_assert=True)
m.hooks.enable_collect_hooks(["pre_decoder"], run_assert=True)
m.hooks.enable_collect_hooks(["attn_pre_out"], run_assert=True) #Needed to recalc activations

# Save activations for each layer
def save_activations(data, dataset_type, step):
    for layer in range(m.cfg.n_layers):
        layer_dir = SAVE_DIR / f"layer_{layer}"
        layer_dir.mkdir(exist_ok=True)
        
        # Get activations for this layer from the raw data
        activations = data.raw["sae"]["pre_decoder"][:, layer, :]  # Shape: [batch, d_sae]
        
        # Save as torch tensor with step number in filename
        save_path = layer_dir / f"step_{step:03d}_{dataset_type}_activations.pt"
        torch.save(activations.detach().cpu(), save_path)
        
        # Save metadata
        metadata = {
            "shape": activations.shape,
            "mean": float(activations.mean().item()),
            "std": float(activations.std().item()),
            "min": float(activations.min().item()),
            "max": float(activations.max().item()),
            "step": step,
            "layer": layer,
            "dataset_type": dataset_type
        }
        torch.save(metadata, layer_dir / f"step_{step:03d}_{dataset_type}_metadata.pt")

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

    # Save initial activations (step -1)
    focus_data = get_midlayer_data(m, "civil", 10, collect_sae=True, calculate_sae=True, collect_attn=False, collect_ff=True, calculate_attn=False, calculate_ff=False)
    cripple_data = get_midlayer_data(m, "toxic", 10, collect_sae=True, calculate_sae=True, collect_attn=False, collect_ff=True, calculate_attn=False, calculate_ff=False)
    save_activations(focus_data, "focus", -1)
    save_activations(cripple_data, "cripple", -1)

    for i in range(c.n_steps):
        print(f"Step {i}")
        data = prune_and_evaluate(m, c, focus_data, cripple_data, i)
        history.add(data)
        
        # Get and save activations after each pruning step
        focus_data = get_midlayer_data(m, "civil", 10, collect_sae=True, calculate_sae=True, collect_attn=False, collect_ff=True, calculate_attn=False, calculate_ff=False)
        cripple_data = get_midlayer_data(m, "toxic", 10, collect_sae=True, calculate_sae=True, collect_attn=False, collect_ff=True, calculate_attn=False, calculate_ff=False)
        save_activations(focus_data, "focus", i)
        save_activations(cripple_data, "cripple", i) 