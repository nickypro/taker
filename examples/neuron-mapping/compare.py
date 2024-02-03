import torch

cripple_repos = ["physics", "bio", "code"]

for repo1 in cripple_repos:
    for repo2 in cripple_repos:
        if repo1 == repo2:
            continue
        #compare ff_criteria from files from each
        repo1_tensors = torch.load("/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/15M/"+repo1+"-pile-15M-recent.pt")
        repo2_tensors = torch.load("/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/15M/"+repo2+"-pile-15M-recent.pt")
        print("repo1: ", repo1, " tensors: ", repo1_tensors)
        print("repo2: ", repo2, " tensors: ", repo2_tensors)
