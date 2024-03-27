import torch
import os
import numpy as np
import random
from datetime import datetime
from taker.eval import run_evaluation
from taker.data_classes import PruningConfig, EvalConfig, RunDataHistory
from taker.activations import get_top_frac, get_midlayer_activations
from taker.model import Model
from taker.texts import infer_dataset_config
from taker.eval import evaluate_all

c = PruningConfig(
    wandb_project = "testing", # repo to push results to
    # model_repo   = "nickypro/tinyllama-15M",
    # model_repo   = "facebook/opt-1.3b",
    model_repo   = "nickypro/llama-7b-hf-rand",
    token_limit  = 1000,  # trim the input to this max length
    run_pre_test = False,  # evaluate the unpruned model
    eval_sample_size = 1e3,
    collection_sample_size = 1e3,
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

def set_seed(seed):
    random.seed(seed)               # Python random module.
    np.random.seed(seed)            # Numpy module.
    torch.manual_seed(seed)         # PyTorch random number generator for CPU.
    torch.cuda.manual_seed(seed)    # PyTorch random number generator for CUDA.
    torch.backends.cudnn.deterministic = True  # To ensure that CUDA selects deterministic algorithms.
    torch.backends.cudnn.benchmark = False

def save_data_dict( model_size: str,
        data: any,
        name: str ):
    now = datetime.now().strftime( "%Y-%m-%d_%H:%M:%S" )
    os.makedirs( f'saved_tensors/{model_size}', exist_ok=True )
    filepath = f'saved_tensors/{model_size}/{name}-{model_size}-recent.pt'
    torch.save( data, filepath )
    print( f'Saved {filepath} to {model_size}' )
    filepath = f'saved_tensors/{model_size}/{name}-{model_size}-{now}.pt'
    torch.save( data, filepath )
    print( f'Saved {filepath} to {model_size}' )
    return filepath

def load_pt_file(filepath: str):
    data = torch.load(filepath)
    for key in data.keys():
        print(key)
    return data


def get_activations(c: PruningConfig, datasets: list[str]):
    # Initilaise Model and show details about model
    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
        )

    results = {}
    for dataset in datasets:
        midlayer_activations = get_midlayer_activations(
            opt,
            dataset,
            c.collection_sample_size,
            c.attn_mode,
            collect_attn=True
            )

        results[dataset] = midlayer_activations.attn.orig

    return results

all_datasets = ["biology",
            "chemistry", 
            "civil", 
            "code", 
            "emotion", 
            "math", 
            "physics", 
            "pile_ArXiv", 
            "pile_Enron_Emails", 
            "pile_EuroParl", 
            "pile_FreeLaw",
            "pile_Github",
            "pile_Gutenberg",
            "pile_HackerNews",
            "pile_NIH_ExPorter",
            "pile_PhilPapers",
            "pile_PubMed_Abstracts",
            "pile_PubMed_Central",
            "pile_StackExchange",
            "pile_Ubuntu_IRC",
            "pile_USPTO_Backgrounds",
            "pile_Wikipedia",
            "poems"]

test_datasets = ["physics"]

data = get_activations(c, test_datasets)

filepath = save_data_dict("llama-7b", data, "test_activations")
print("file saved to: ", filepath)
loaded_data = load_pt_file(filepath)["physics"]

print(loaded_data)