from taker.data_classes import PruningConfig
from taker.parser import cli_parser
from taker.prune import run_pruning
import torch

# Configure initial model and tests
c = PruningConfig(
    wandb_entity = "seperability",
    wandb_project = "testing",
    model_repo   = "facebook/opt-1.3b",
    token_limit  = 1000,
    run_pre_test = True,
    # Removals parameters
    ff_frac   = 0.02,
    ff_eps    = 0.001,
    attn_frac = 0.00,
    attn_eps  = 1e-4,
    focus     = "pile_codeless",
    cripple   = "code",
    additional_datasets=tuple(),
    recalculate_activations = True, # iterative vs non-iterative pruning
)

# Parse CLI for arguments
c, args = cli_parser(c)

import torch
torch.set_grad_enabled(False)

# Run the iterated pruning
with torch.no_grad():
    model, history = run_pruning(c)
