
from taker.data_classes import PruningConfig
from taker.parser import cli_parser
from taker.prune import run_pruning
import torch

# Configure initial model and tests
c = PruningConfig(
    wandb_project = "testing", # repo to push results to
    model_repo   = "nickypro/tinyllama-15M",
    # "metallama/llama-2-7b"
    token_limit  = 1000,  # trim the input to this max length
    run_pre_test = True,  # evaluate the unpruned model
    eval_sample_size = 1e3,
    collection_sample_size = 1e3,
    # Removals parameters
    ff_frac   = 0.2,     # % of feed forward neurons to prune
    attn_frac = 0.00,     # % of attention neurons to prune
    focus     = "pile", # the “reference” dataset
    cripple   = "physics",          # the “unlearned” dataset
    additional_datasets=tuple(), # any extra datasets to evaluate on
    recalculate_activations = False, # iterative vs non-iterative
    n_steps = 1,
)

# Parse CLI for arguments
# c, args = cli_parser(c)

#list of repos to cripple
cripple_repos = ["physics", "bio", "code"]

# Run the iterated pruning for each cripple repo
for repo in cripple_repos:
    c.cripple = repo
    print("running iteration for ", c.cripple, " vs ", c.focus)
    with torch.no_grad():
        model, history = run_pruning(c)

for repo1 in cripple_repos:
    for repo2 in cripple_repos:
        if repo1 == repo2:
            continue
        #compare ff_criteria from files from each
        repo1_tensors = torch.load("/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/15M/"+repo1+"-"+c.focus+"-15M-recent.pt")
        repo2_tensors = torch.load("/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/15M/"+repo2+"-"+c.focus+"-15M-recent.pt")
        
        print("repo1: ", repo1, " tensors: ", repo1_tensors)
        print("repo2: ", repo2, " tensors: ", repo2_tensors)