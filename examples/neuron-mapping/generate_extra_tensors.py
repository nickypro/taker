from taker.activations import get_top_frac
import torch

cripple_repos = ["biology", "chemistry", "physics", "math", "code", "poems", "civil", "emotion", "pile_FreeLaw", "pile_PubMed_Abstracts", "pile_PubMed_Central", "pile_NIH_ExPorter", "pile_Enron_Emails", "pile_Github", "pile_StackExchange", "pile_HackerNews", "pile_ArXiv", "pile_Wikipedia", "pile_Ubuntu_IRC", "pile_USPTO_Backgrounds", "pile_PhilPapers", "pile_EuroParl", "pile_Gutenberg", "pile_PhilPapers", "pile_EuroParl", "pile_Gutenberg"]
ff_fracs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

#takes repo name, model size and timestamp as strings and returns ff_scores tensor by loading relevant file
def load_tensors_for_repo(repo, model_size="hf", timestamp="recent"):
    directory = "/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/"+model_size+"/"
    filename = repo+"-pile-"+model_size+"-"+timestamp+".pt"
    data = torch.load(directory+filename)
    return data["ff_scores"], data["ff_criteria"]


def get_ff_criteria_for_ff_frac(repo, ff_frac):
    ff_scores, _ = load_tensors_for_repo(repo)
    criteria, _ = get_top_frac(ff_scores, ff_frac)
    return criteria

def compare_pruned_ff_criteria(repos, ff_fracs, model_size="hf"):
    ratios = {}
    for ff_frac in ff_fracs:
        ratios[ff_frac] = {}
        for repo1 in repos:
            ratios[ff_frac][repo1] = {}
            ff_criteria_repo1 = get_ff_criteria_for_ff_frac(repo1, ff_frac)
            for repo2 in repos:
                if repo1 == repo2:
                    continue
                ff_criteria_repo2 = get_ff_criteria_for_ff_frac(repo2, ff_frac)
                matches = torch.logical_and(ff_criteria_repo1, ff_criteria_repo2)
                ratio = torch.sum(matches)/torch.sum(ff_criteria_repo1)
                ratios[ff_frac][repo1][repo2] = ratio
                
    return ratios

ratios = compare_pruned_ff_criteria(cripple_repos, ff_fracs)
for ff_frac in ratios:
    print(ff_frac, "\n")
    for repo in ratios[ff_frac]:
        print(repo, "\n", ratios[ff_frac][repo])

