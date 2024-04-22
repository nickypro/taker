import torch
import numpy as np
import matplotlib.pyplot as plt

def load_pt_file(directory: str, filename: str):
    data = torch.load(directory+filename)
    for key in data.keys():
        print(key)
    return data

cripple_repos = ["pile_FreeLaw", "biology", "chemistry", "pile_PubMed_Abstracts", "pile_PubMed_Central", "pile_NIH_ExPorter", "pile_Enron_Emails", "code", "pile_Github",  "pile_StackExchange", "pile_Ubuntu_IRC", "pile_HackerNews", "poems", "civil", "emotion", "physics", "math", "pile_ArXiv", "pile_Wikipedia", "pile_USPTO_Backgrounds", "pile_PhilPapers", "pile_EuroParl", "pile_Gutenberg"]

def plot_ratios(ratios, ff_frac):
    datasets = cripple_repos
    grid = [
        [
            ratios[ff_frac][dataset_a][dataset_b].item() if dataset_a != dataset_b else np.nan
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

    plt.subplots_adjust(bottom=0.19, top=0.97)

    plt.title(f"FF Criteria Overlap for Prune Ratio {ff_frac}")

    plt.show()


ratios = load_pt_file("/home/rashid/ml/trajectories/rash92_fork/taker/examples/neuron-mapping/saved_tensors/hf/","pruning_ratios-hf-recent.pt")
ff_fracs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
for ff_frac in ff_fracs:
    plot_ratios(ratios, ff_frac)
