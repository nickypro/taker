import torch

def compare_pruned_ff_criteria(cripple_repos: list[str], model_size: str):
    # cripple_repos = ["physics", "bio", "code"]
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
    

def load_pt_file(directory: str, filename: str):
    data = torch.load(directory+filename)
    for key in data.keys():
        print(key)
    return data

# /home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/1.3b/shared_pruning_data-1.3b-recent.pt
directory = "/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/1.3b/"
filename = "shared_pruning_data-1.3b-recent.pt"
data = load_pt_file(directory,filename)
print(data)
