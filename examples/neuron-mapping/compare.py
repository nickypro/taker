import torch
from taker.activations import get_top_frac
import matplotlib.pyplot as plt
import numpy as np

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

def load_pt_file(directory: str, filename: str):
    data = torch.load(directory+filename)
    for key in data.keys():
        print(key)
    return data

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

def plot_ratios(ratios):
    datasets = cripple_repos
    grid = [
        [
            ratios[0.01][dataset_a][dataset_b].item() if dataset_a != dataset_b else np.nan
            for dataset_b in datasets
        ]
        for dataset_a in datasets
    ]
    grid = np.ma.masked_where(np.isnan(grid), grid)

    average = np.mean(grid)

    plt.imshow(grid)
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            plt.text(
                j,
                i,
                f"{grid[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if grid[i, j] > average else "white",
            )
    plt.xticks(
        range(len(datasets)), [dataset for dataset in datasets], rotation=90
    )
    plt.yticks(range(len(datasets)), [dataset for dataset in datasets])

    plt.subplots_adjust(bottom=0.3)

    plt.title(f"FF Criteria Overlap for Prune Ratio 0.01")

    plt.show()

cripple_repos = ["biology", "chemistry", "physics", "math", "code", "poems", "civil", "emotion", "pile_FreeLaw", "pile_PubMed_Abstracts", "pile_PubMed_Central", "pile_NIH_ExPorter", "pile_Enron_Emails", "pile_Github", "pile_StackExchange", "pile_HackerNews", "pile_ArXiv", "pile_Wikipedia", "pile_Ubuntu_IRC", "pile_USPTO_Backgrounds", "pile_PhilPapers", "pile_EuroParl", "pile_Gutenberg", "pile_PhilPapers", "pile_EuroParl", "pile_Gutenberg"]
ff_fracs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

ratios = compare_pruned_ff_criteria(cripple_repos, ff_fracs)
filename = save_pruning_data_dict("hf", ratios, "pruning_ratios")
print("saved pruning ratios to: ", filename)