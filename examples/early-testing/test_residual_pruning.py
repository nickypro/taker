#!/usr/bin/env python3
"""Simple test script for residual stream pruning functionality."""

from taker import Model
from taker.data_classes import PruningConfig, RunDataHistory, RunDataItem
from taker.prune import prune_and_evaluate
from taker.activations import get_midlayer_data
from taker.eval import evaluate_all
import torch
import wandb

hook_config = """
post_decoder: mask, collect
"""

c = PruningConfig("gpt2",
    dtype="fp32",  # MPS compatibility
    wandb_entity = "seperability",
    wandb_project = "bens-tests", 
    wandb_run_name = "gpt2 residual stream prune test",
    token_limit  = 100,
    # Residual stream pruning only
    ff_frac   = 0.0,
    attn_frac = 0.0,
    residual_frac = 0.1,     # Prune 10% of residual dimensions
    residual_scoring = "abs",
    focus     = "civil",
    cripple   = "toxic",
    recalculate_activations = False,
    collection_sample_size = 100,
    eval_sample_size = 100,
    n_steps = 3,
)

m = Model("gpt2", hook_config=hook_config)
m.hooks.enable_collect_hooks(["post_decoder"], run_assert=True)

# Get initial activations
focus_data = get_midlayer_data(m, "civil", 100, collect_residual=True, calculate_residual=True, 
                              collect_ff=False, calculate_ff=False, collect_attn=False, calculate_attn=False)
cripple_data = get_midlayer_data(m, "toxic", 100, collect_residual=True, calculate_residual=True,
                                collect_ff=False, calculate_ff=False, collect_attn=False, calculate_attn=False)

history = RunDataHistory(list(c.datasets))
wandb.init(
    project=c.wandb_project,
    entity=c.wandb_entity,
    name=c.wandb_run_name,
)
wandb.config.update(c.to_dict(), allow_val_change=True)

torch.set_grad_enabled(False)

# Run the pruning
with torch.no_grad():
    # Evaluate without pruning first
    data = RunDataItem()
    eval_out = evaluate_all(m, c.eval_sample_size, c.datasets,
                            dataset_tokens_to_skip=c.collection_sample_size)
    data.update(eval_out)
    history.add(data)

    for i in range(c.n_steps):
        print(f"Step {i}")
        data = prune_and_evaluate(m, c, focus_data, cripple_data, i)
        history.add(data)
        print(f"Residual dimensions pruned: {data.deletions['residual_del']}")
        print(f"Residual threshold: {data.deletions['residual_threshold']}")
        
        # Get new activations after pruning for next iteration
        focus_data = get_midlayer_data(m, "civil", 100, collect_residual=True, calculate_residual=True,
                                      collect_ff=False, calculate_ff=False, collect_attn=False, calculate_attn=False)
        cripple_data = get_midlayer_data(m, "toxic", 100, collect_residual=True, calculate_residual=True,
                                        collect_ff=False, calculate_ff=False, collect_attn=False, calculate_attn=False)