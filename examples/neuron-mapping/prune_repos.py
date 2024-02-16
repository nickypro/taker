
from taker.data_classes import PruningConfig
from taker.parser import cli_parser
from taker.prune import run_pruning
import torch

def compare_pruned_ff_criteria(cripple_repos: list[str], model_size: str):
    directory = "/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/"+model_size+"/"
    focus_repo = "pile"
    suffix = "-"+model_size+"-recent.pt"
    ratios = {}
    ratios["model_size"] = model_size
    
    for repo1 in cripple_repos:
        #load ff_criteria from repo1
        repo1_tensors = torch.load(directory+repo1+"-"+focus_repo+suffix)
        repo1_ff_criteria = repo1_tensors["ff_criteria"]
        ratios[repo1] = {}
        for repo2 in cripple_repos:
            if repo1 == repo2:
                continue
            #load ff_criteria from repo2
            repo2_tensors = torch.load(directory+repo2+"-"+focus_repo+suffix)
            repo2_ff_criteria = repo2_tensors["ff_criteria"]

            matches = torch.logical_and(repo1_ff_criteria, repo2_ff_criteria)
            ratio = torch.sum(matches)/torch.sum(repo1_ff_criteria)
            ratios[repo1][repo2] = ratio
            
    return ratios
    

# Configure initial model and tests
c = PruningConfig(
    wandb_project = "testing", # repo to push results to
    # model_repo   = "nickypro/tinyllama-15M",
    # model_repo   = "facebook/opt-1.3b",
    model_repo   = "NousResearch/Llama-2-7b-hf",
    token_limit  = 1000,  # trim the input to this max length
    run_pre_test = False,  # evaluate the unpruned model
    eval_sample_size = 1e5,
    collection_sample_size = 1e5,
    # Removals parameters
    ff_frac   = 0.01,     # % of feed forward neurons to prune
    attn_frac = 0.00,     # % of attention neurons to prune
    focus     = "pile", # the “reference” dataset
    cripple   = "physics",          # the “unlearned” dataset
    additional_datasets=tuple(), # any extra datasets to evaluate on
    recalculate_activations = False, # iterative vs non-iterative
    dtype = "int8",
    n_steps = 1,
)

#list of repos to cripple
cripple_repos = ["emotion", "pile_FreeLaw", "pile_PubMed_Abstracts", "pile_PubMed_Central", "pile_NIH_ExPorter", "pile_Enron_Emails", "pile_Github", "pile_StackExchange", "pile_HackerNews", "pile_ArXiv", "pile_Wikipedia", "pile_Ubuntu_IRC", "pile_USPTO_Backgrounds", "pile_PhilPapers", "pile_EuroParl", "pile_Gutenberg", "pile_PhilPapers", "pile_EuroParl", "pile_Gutenberg"]

#prune each repo and save tensors, doing some extra computation but only really need ff_scores for each repo, will do actual pruning for different values of ff_frac in compare.py
for repo in cripple_repos:
    c.cripple = repo
    print("running iteration for ", c.cripple, " vs ", c.focus, "with ff_frac: ", c.ff_frac)
    with torch.no_grad():
        model, history = run_pruning(c)

