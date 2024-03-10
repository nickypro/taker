import torch
import numpy as np
from taker.eval import run_evaluation
from taker.data_classes import PruningConfig, EvalConfig
from taker.activations import get_top_frac
from taker.model import Model
from taker.texts import infer_dataset_config


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

all_datasets = ["biology",
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

test_datasets = ["biology", "chemistry"]

def find_accuracy(dataset, ff_frac):
    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
        )
    
    eval_config: EvalConfig = infer_dataset_config(dataset)
    eval_config.num_tokens_to_skip = c.collection_sample_size
    eval_config.sample_size = c.eval_sample_size
    print(f"ff_frac is: {ff_frac}")
    if ff_frac == 0:
        base_accuracy = run_evaluation(opt, eval_config)
        print(f"base accuracy is: {base_accuracy}")
        return base_accuracy.percent["base"]
    ff_criteria = get_ff_criteria_for_ff_frac(dataset, ff_frac)
    opt.delete_ff_keys(ff_criteria)
    eval_data = run_evaluation(opt, eval_config)
    print(f"accuracy for ff_frac {ff_frac} is {eval_data.percent}")
    return eval_data.percent["base"]

#binary search to find ff_frac for a given target accuracy, upto given precision while being below target
#also check if should compare base accuracy or topk accuracy.
def find_correct_ff_frac(dataset: str, target_accuracy: float, accuracy_precision: float, lower=0, upper=1):
    print(f"trying to find correct ff_frac for {dataset} with target accuracy {target_accuracy}")
    

    while upper >= lower:
        print(f"binary search with lower {lower} and upper {upper}")
        acc_mid = find_accuracy(dataset, (lower + upper)/2)
        if acc_mid >= target_accuracy-accuracy_precision and acc_mid <= target_accuracy:
            return (lower + upper)/2
        elif acc_mid < target_accuracy:
            upper = (lower + upper)/2
        else:
            lower = (lower + upper)/2
    
    return lower

def compareEvaluations(datasets):
    final_data = {}
    for dataset1 in datasets:
        print("for neurons pruned based on: ", dataset1)
        unpruned_accuracy = find_accuracy(dataset1, 0)
        ff_frac = find_correct_ff_frac(dataset1, target_accuracy=(0.8*unpruned_accuracy), accuracy_precision=5, upper=0.2)
        final_data[dataset1] = {}
        final_data[dataset1]['unpruned_accuracy'] = unpruned_accuracy
        for dataset2 in datasets:
            print("finding accuracy for: ", dataset2)
            final_data[dataset1][dataset2] = find_accuracy(dataset2, ff_frac)
    return final_data

answer = compareEvaluations(test_datasets)
print(answer)
