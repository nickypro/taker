
from taker.data_classes import PruningConfig
from taker.parser import cli_parser
from taker.prune import run_pruning
import os
import datetime
import torch

def save_pruning_data_dict( model_size: str,
        data: any,
        name: str ):
    now = datetime.datetime.now().strftime( "%Y-%m-%d_%H:%M:%S" )
    os.makedirs( f'saved_tensors/{model_size}', exist_ok=True )
    filename = f'saved_tensors/{model_size}/{name}-{model_size}-recent.pt'
    torch.save( data, filename )
    print( f'Saved {filename} to {model_size}' )
    filename = f'saved_tensors/{model_size}/{name}-{model_size}-{now}.pt'
    torch.save( data, filename )
    print( f'Saved {filename} to {model_size}' )
    return filename

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

# Parse CLI for arguments
# c, args = cli_parser(c)

#list of repos to cripple
#done
#"emotion", "pile_FreeLaw", "pile_PubMed_Abstracts", "pile_PubMed_Central", "pile_NIH_ExPorter", "pile_Enron_Emails", "pile_Github", "pile_StackExchange", "pile_HackerNews", "pile_ArXiv", "pile_Wikipedia"
#not done, to do next
#, "pile_EuroParl", "pile_Gutenberg"
cripple_repos = ["pile_Ubuntu_IRC", "pile_USPTO_Backgrounds", "pile_PhilPapers"]
ff_frac_to_prune = [0.01]
# model_size = c.model_repo.split('-')[-1]

# Run the iterated pruning for each cripple repo, for a range of ff_frac pruned
# shared_pruning_data = {}
for ff_frac in ff_frac_to_prune:
    c.ff_frac = ff_frac
    #only want pretest for each repo once, so running it on first value of ff_frac.
    for repo in cripple_repos:
        c.cripple = repo
        print("running iteration for ", c.cripple, " vs ", c.focus, "with ff_frac: ", ff_frac)
        with torch.no_grad():
            model, history = run_pruning(c)
    # ratios = compare_pruned_ff_criteria(cripple_repos, model_size)
    # shared_pruning_data[ff_frac] = ratios

# shared_pruning_data["config"] = {
#     "model_repo": c.model_repo, 
#     "token_limit": c.token_limit, 
#     "eval_sample_size": c.eval_sample_size, 
#     "collection_sample_size": c.collection_sample_size, 
#     "n_steps": c.n_steps, 
#     "dtype": c.dtype,
#     }

# pruning_data_filename = save_pruning_data_dict(model_size, shared_pruning_data, "shared_pruning_data")
# print("data saved to: ", pruning_data_filename, "data: ", shared_pruning_data)
