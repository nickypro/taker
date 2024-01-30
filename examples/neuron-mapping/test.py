
from taker.data_classes import PruningConfig
from taker.parser import cli_parser
from taker.prune import run_pruning
import torch

# Configure initial model and tests
c = PruningConfig(
    wandb_project = "testing", # repo to push results to
    model_repo   = "nickypro/tinyllama-15M",
    token_limit  = 1000,  # trim the input to this max length
    run_pre_test = True,  # evaluate the unpruned model 
    eval_sample_size = 1e4,
    collection_sample_size = 1e4,
    # Removals parameters
    ff_frac   = 0.2,     # % of feed forward neurons to prune
    attn_frac = 0.00,     # % of attention neurons to prune
    focus     = "pile_codeless", # the “reference” dataset
    cripple   = "code",          # the “unlearned” dataset
    additional_datasets=tuple(), # any extra datasets to evaluate on
    recalculate_activations = False, # iterative vs non-iterative
)

# Parse CLI for arguments
c, args = cli_parser(c)

# Run the iterated pruning
with torch.no_grad():
    model, history = run_pruning(c)
