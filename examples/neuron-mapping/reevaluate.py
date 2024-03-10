import torch
import numpy as np
from datetime import datetime
from taker.eval import run_evaluation
from taker.data_classes import PruningConfig, EvalConfig
from taker.activations import get_top_frac
from taker.model import Model
from taker.texts import infer_dataset_config


#most of pruningconfig is not used, but some eval functions use this as copied from elsewhere.
c = PruningConfig(
    wandb_project = "testing", # repo to push results to
    # model_repo   = "nickypro/tinyllama-15M",
    # model_repo   = "facebook/opt-1.3b",
    model_repo   = "NousResearch/Llama-2-7b-hf",
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

test_datasets = ["biology", "chemistry", "physics", "code", "pile_Github"]

file_load_counter = 0

#filepath hardcoded, can't get relative path to work. may need to be edited based on where stuff is cloned
def load_tensors_for_repo(repo, model_size="hf", timestamp="recent"):
    directory = "/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/"+model_size+"/"
    filename = repo+"-pile-"+model_size+"-"+timestamp+".pt"
    data = torch.load(directory+filename)
    return data["ff_scores"], data["ff_criteria"]


def get_ff_criteria_for_ff_frac(repo, ff_frac):
    startTime = datetime.now()
    ff_scores, _ = load_tensors_for_repo(repo)
    midTime = datetime.now()
    criteria, _ = get_top_frac(ff_scores, ff_frac)
    endTime = datetime.now()
    global file_load_counter
    file_load_counter += 1
    print("Time taken to load tensors for ff_scores: ", midTime - startTime, "total loads so far: ", file_load_counter)
    print("Time taken to get ff_criteria from ff_scores: ", endTime - midTime)
    print("Total Time taken to get ff_criteria from ff_frac: ", endTime - startTime)
    return criteria


#pruning dataset is the dataset to use to determine which neurons to prune, target dataset is the dataset to find the accuracy of
def find_accuracy(pruning_dataset, target_dataset, ff_frac):
    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
        )
    
    eval_config: EvalConfig = infer_dataset_config(target_dataset)
    eval_config.num_tokens_to_skip = c.collection_sample_size
    eval_config.sample_size = c.eval_sample_size
    print(f"checking accuracy for ff_frac: {ff_frac}")
    if ff_frac == 0:
        unpruned_accuracy = run_evaluation(opt, eval_config)
        print(f"unpruned accuracy is: {unpruned_accuracy.percent}")
        return unpruned_accuracy.percent["base"]
    ff_criteria = get_ff_criteria_for_ff_frac(pruning_dataset, ff_frac)
    opt.delete_ff_keys(ff_criteria)
    eval_data = run_evaluation(opt, eval_config)
    print(f"accuracy for ff_frac {ff_frac} is {eval_data.percent}")
    return eval_data.percent["base"]


#binary search to find ff_frac for a given target accuracy,
#upto given precision while being below target
#also check if should compare base accuracy or topk accuracy.
def find_correct_ff_frac(dataset: str, target_accuracy: float, accuracy_precision: float, ff_frac_precision=1e-5, lower=0, upper=1):
    print(f"trying to find correct ff_frac for {dataset} with target accuracy {target_accuracy}")
    #check if upper and lower are reasonable, if not default to 1 and 0.
    if upper < 1:
        acc_upper = find_accuracy(dataset, dataset, upper)
        if acc_upper > target_accuracy:
            upper = 1
    if lower > 0:
        acc_lower = find_accuracy(dataset, dataset, lower)
        if acc_lower < target_accuracy:
            lower = 0
    #binary search for ff_frac that reaches accuracy below target accuracy
    while upper >= lower + ff_frac_precision:
        acc_mid = find_accuracy(dataset, dataset, (lower + upper)/2)
        if acc_mid >= target_accuracy-accuracy_precision and acc_mid <= target_accuracy:
            return (lower + upper)/2
        elif acc_mid < target_accuracy:
            upper = (lower + upper)/2
        else:
            lower = (lower + upper)/2
    print("reached limit of ff_frac_precision without reaching target accuracy, likely due to evaluation not being deterministic")
    return lower

def compareEvaluations(datasets):
    final_data = {}
    for dataset1 in datasets:
        final_data[dataset1] = {}
        unpruned_accuracy = find_accuracy(dataset1, dataset1, 0)
        target_accuracy=0.8*unpruned_accuracy
        final_data[dataset1]["unpruned_accuracy"] = unpruned_accuracy
        final_data[dataset1]["target_accuracy"] = target_accuracy

        ff_frac = find_correct_ff_frac(dataset1, target_accuracy=target_accuracy, accuracy_precision=5, upper=0.128)
        final_data[dataset1]["ff_frac"] = ff_frac

        for dataset2 in datasets:
            print("finding accuracy for: ", dataset2, "with neurons pruned based on: ", dataset1)
            unpruned_accuracy = find_accuracy(dataset1, dataset2, 0)
            pruned_accuracy = find_accuracy(dataset1, dataset2, ff_frac)
            final_data[dataset1][dataset2]["unpruned_accuracy"] = unpruned_accuracy
            final_data[dataset1][dataset2]["pruned_accuracy"] = pruned_accuracy
            final_data[dataset1][dataset2]["accuracy_difference"] = unpruned_accuracy - pruned_accuracy
            final_data[dataset1][dataset2]["accuracy_ratio"] = pruned_accuracy/unpruned_accuracy

    return final_data

startTime = datetime.now()
print("run started at: ", startTime)
answer = compareEvaluations(test_datasets)
print(answer)
endTime = datetime.now()
print("run ended at: ", endTime, "time elapsed: ", endTime - startTime)