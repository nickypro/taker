import torch
import os
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
    model_repo   = "mistralai/Mistral-7b-instruct-v0.2",
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
    dtype = "int4",
    n_steps = 1,
)

opt = Model(
    c.model_size,
    limit=c.token_limit,
    dtype=c.dtype,
    svd_attn=c.svd_attn,
    use_accelerator=c.use_accelerator,
    model_device=c.model_device,
    mask_fn=c.mask_fn,
)

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

test_datasets = ["biology", "chemistry", "physics", "code", "pile_Github"]

pile_datasets = ["pile_ArXiv", 
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
                "pile_Wikipedia"]

mmlu_subsets = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

mmlu_datasets = ["mmlu:"+x for x in mmlu_subsets]

#filepath hardcoded, can't get relative path to work. may need to be edited based on where stuff is cloned
def load_tensors_for_repo(repo, model_size="v0.2", timestamp="recent"):
    directory = "/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/"+model_size+"/"
    filename = repo+"-pile-"+model_size+"-"+timestamp+".pt"
    data = torch.load(directory+filename)
    return data["ff_scores"], data["ff_criteria"]

def save_data_dict( model_size: str,
        data: any,
        name: str ):
    now = datetime.now().strftime( "%Y-%m-%d_%H:%M:%S" )
    os.makedirs( f'saved_tensors/{model_size}', exist_ok=True )
    filename = f'saved_tensors/{model_size}/{name}-{model_size}-recent.pt'
    torch.save( data, filename )
    print( f'Saved {filename} to {model_size}' )
    filename = f'saved_tensors/{model_size}/{name}-{model_size}-{now}.pt'
    torch.save( data, filename )
    print( f'Saved {filename} to {model_size}' )
    return filename


def get_ff_criteria_for_ff_frac(repo, ff_frac):
    # ff_start_time = datetime.now()
    ff_scores, _ = load_tensors_for_repo(repo)
    # ff_mid_time = datetime.now()
    criteria, _ = get_top_frac(ff_scores, ff_frac)
    # ff_end_time = datetime.now()

    # print("Time taken to load tensors for ff_scores: ", ff_mid_time - ff_start_time)
    # print("Time taken to get ff_criteria from ff_scores: ", ff_end_time - ff_mid_time)
    # print("Time taken to get ff_criteria from ff_frac: ", ff_end_time - ff_start_time)
    return criteria


#pruning dataset is the dataset to use to determine which neurons to prune, target dataset is the dataset to find the accuracy of
def find_accuracy(pruning_dataset, target_dataset, ff_frac):
    for i in range(3):
        try:
            find_accuracy_fn(pruning_dataset, target_dataset, ff_frac)
        except:
            pass
    return -1

def find_accuracy_fn(pruning_dataset, target_dataset, ff_frac):
    # find_acc_start = datetime.now()
    opt.init_model(do_model_import=False)
    
    if target_dataset[:4] == "mmlu":
        eval_config: EvalConfig = infer_dataset_config("mmlu")
        eval_config.dataset_subset = target_dataset[5:]
    else:
        eval_config: EvalConfig = infer_dataset_config(target_dataset)
    eval_config.num_tokens_to_skip = c.collection_sample_size
    eval_config.sample_size = c.eval_sample_size
    # print(f"checking accuracy for ff_frac: {ff_frac} for pruning dataset: {pruning_dataset} and target_dataset: {target_dataset}")
    if ff_frac == 0:
        unpruned_accuracy = run_evaluation(opt, eval_config)
        # find_acc_mid = datetime.now()
        # print(f"unpruned accuracy is: {unpruned_accuracy.percent}")
        # print(f"time to find unpruned accuracy for dataset: {target_dataset} is: {find_acc_mid - find_acc_start}")
        return unpruned_accuracy.percent["base"]
    ff_criteria = get_ff_criteria_for_ff_frac(pruning_dataset, ff_frac)
    opt.delete_ff_keys(ff_criteria)
    eval_data = run_evaluation(opt, eval_config)
    # find_acc_end = datetime.now()
    # print(f"time to find accuracy for pruning dataset: {pruning_dataset} and target_dataset: {target_dataset} with ff_frac: {ff_frac} is: {find_acc_end - find_acc_start} and has accuracy: {eval_data.percent}")
    return eval_data.percent["base"]


#binary search to find ff_frac for a given target accuracy,
#upto given precision while being below target
#also check if should compare base accuracy or topk accuracy.
def find_correct_ff_frac(dataset: str, target_accuracy: float, accuracy_precision: float, ff_frac_precision=1e-6, lower=0, upper=1):
    print(f"trying to find correct ff_frac for {dataset} with target accuracy {target_accuracy}")
    #check if upper and lower are reasonable, if not default to 1 and 0.
    # ff_start = datetime.now()
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
            # ff_end = datetime.now()
            # print(f"time to find correct ff_frac for target accuracy is: {ff_end - ff_start}")
            return (lower + upper)/2
        elif acc_mid < target_accuracy:
            upper = (lower + upper)/2
        else:
            lower = (lower + upper)/2
    print("reached limit of ff_frac_precision without reaching target accuracy, likely due to evaluation not being deterministic")
    return lower

def compareEvaluations(datasets):
    final_data = {}
    final_data["sample_size"] = c.eval_sample_size
    for dataset1 in datasets:
        dataset_start = datetime.now()
        final_data[dataset1] = {}
        unpruned_accuracy = find_accuracy(dataset1, dataset1, 0)
        target_accuracy=0.8*unpruned_accuracy
        final_data[dataset1]["unpruned_accuracy"] = unpruned_accuracy
        final_data[dataset1]["target_accuracy"] = target_accuracy

        ff_frac = find_correct_ff_frac(dataset1, target_accuracy=target_accuracy, accuracy_precision=2, upper=0.128)
        final_data[dataset1]["ff_frac"] = ff_frac

        for dataset2 in [*datasets, *mmlu_datasets]:
            final_data[dataset1][dataset2] = {}
            # print("finding accuracy for: ", dataset2, "with neurons pruned based on: ", dataset1)
            unpruned_accuracy = find_accuracy(dataset1, dataset2, 0)
            pruned_accuracy = find_accuracy(dataset1, dataset2, ff_frac)
            final_data[dataset1][dataset2]["unpruned_accuracy"] = unpruned_accuracy
            final_data[dataset1][dataset2]["pruned_accuracy"] = pruned_accuracy
            final_data[dataset1][dataset2]["accuracy_difference"] = unpruned_accuracy - pruned_accuracy
            final_data[dataset1][dataset2]["accuracy_ratio"] = pruned_accuracy/unpruned_accuracy
        dataset_end = datetime.now()
        print("time to get data for dataset: ", dataset1, "is ", dataset_end - dataset_start, "results: ", final_data[dataset1])

    return final_data

startTime = datetime.now()
print("run started at: ", startTime)
answer = compareEvaluations([ *all_datasets[1:], all_datasets[0]])
saved_file_name = save_data_dict("v0.2", answer, "cross_pruning_accuracy")
print("saved to: ", saved_file_name, "data: ", answer)
endTime = datetime.now()
print("run ended at: ", endTime, "time elapsed: ", endTime - startTime)
