from taker.eval import run_evaluation
from taker.data_classes import PruningConfig, EvalConfig
from taker.activations import get_top_frac
from taker.model import Model
from taker.texts import infer_dataset_config
import torch
import numpy as np

#filepath hardcoded, can't get relative path to work. may need to be edited based on where stuff is cloned
def load_tensors_for_repo(repo, model_size="hf", timestamp="recent"):
    directory = "/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/"+model_size+"/"
    filename = repo+"-pile-"+model_size+"-"+timestamp+".pt"
    data = torch.load(directory+filename)
    return data["ff_scores"], data["ff_criteria"]


def get_ff_criteria_for_ff_frac(repo, ff_frac):
    ff_scores, _ = load_tensors_for_repo(repo)
    criteria, _ = get_top_frac(ff_scores, ff_frac)
    return criteria

c = PruningConfig(
    wandb_project = "testing", # repo to push results to
    # model_repo   = "nickypro/tinyllama-15M",
    # model_repo   = "facebook/opt-1.3b",
    model_repo   = "NousResearch/Llama-2-7b-hf",
    token_limit  = 1000,  # trim the input to this max length
    run_pre_test = False,  # evaluate the unpruned model
    eval_sample_size = 1e4,
    collection_sample_size = 1e4,
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

datasets = ["biology", 
            "chemistry", 
            "civil", 
            "code", 
            "emotion", 
            "math", 
            "physics", 
            "pile_ArXiv", 
            "pile_Enron", 
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
            "pile_Ubuntu",
            "pile_USPTO_Backgrounds",
            "pile_Wikipedia",
            "poems"]

test_datasets = ["biology", "chemistry", "physics"]

def find_accuracy(dataset, ff_criteria):
    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
        )
    
    opt.delete_ff_keys(ff_criteria)

    eval_config: EvalConfig = infer_dataset_config(dataset)
    eval_config.num_tokens_to_skip = c.collection_sample_size
    eval_config.sample_size = c.eval_sample_size
   
    eval_data = run_evaluation(opt, eval_config)
    return eval_data["percent"]


def find_correct_ff_frac(dataset: str, target_accuracy: float, ff_frac_step_size: float):
    for ff_frac in np.arange(1-ff_frac_step_size, ff_frac_step_size):
        ff_criteria = get_ff_criteria_for_ff_frac(dataset, ff_frac)
        accuracy = find_accuracy(dataset, ff_criteria)
        if accuracy < target_accuracy:
            return ff_frac
    return 1

def compareEvaluations(datasets):
    final_data = {}
    for dataset1 in datasets:
        ff_frac = find_correct_ff_frac(dataset1, 0.5, 0.02)
        ff_criteria = get_ff_criteria_for_ff_frac(dataset1, ff_frac)
        final_data[dataset1] = {}

        for dataset2 in datasets:
            final_data[dataset1][dataset2] = find_accuracy(dataset2, ff_criteria)
    
    return final_data


answer = compareEvaluations(test_datasets)
print(answer)