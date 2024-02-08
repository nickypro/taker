import torch

def compare_pruned_ff_criteria(cripple_repos: list[str], model_size: str, path: str="/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/", focus_repo: str = "pile",):
    # cripple_repos = ["physics", "bio", "code"]
    directory = f"{path}{model_size}/"
    suffix = "-"+model_size+"-0.01-recent.pt"
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
    
print(compare_pruned_ff_criteria(["cifar20-trees", "cifar20-veh1", "cifar20-veh2"], "Cifar100", path="/home/ubuntu/tetra/taker/examples/neuron-mapping/saved_tensors/", focus_repo="cifar20-split"))