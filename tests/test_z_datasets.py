import torch
from taker import Model
from taker.eval import evaluate_all

def test_text_datasets():
    dataset_list = [
        "pile",
        "biology",
        "chemistry",
        "civil",
        "emotion",
        "poems",
        "math",
        "physics",
        "code",
        'pile_ArXiv',
        'pile_BookCorpus2',
        'pile_Books3',
        'pile_DM_Mathematics',
        'pile_Enron_Emails',
        'pile_EuroParl',
        'pile_FreeLaw',
        'pile_Github',
        'pile_Gutenberg',
        'pile_HackerNews',
        'pile_NIH_ExPorter',
        'pile_OpenSubtitles',
        'pile_OpenWebText2',
        'pile_PhilPapers',
        'pile_Pile-CC',
        'pile_PubMed_Abstracts',
        'pile_PubMed_Central',
        'pile_StackExchange',
        'pile_USPTO_Backgrounds',
        'pile_Ubuntu_IRC',
        'pile_Wikipedia',
        'pile_YoutubeSubtitles',
    ]
    m = Model("nickypro/tinyllama-15m", limit=1000)
    with torch.no_grad():
        for dataset_name in dataset_list:
            evaluate_all(m, 1e2, [dataset_name])

def test_img_datasets():
    dataset_list = [
        "cifar100",
        "cifar20-aquatic_mammals",
        "cifar20-fish",
        "cifar20-flowers",
        "cifar20-food_containers",
        "cifar20-fruit_and_vegetables",
        "cifar20-household_electrical_devices",
        "cifar20-household_furniture",
        "cifar20-insects",
        "cifar20-large_carnivores",
        "cifar20-large_manmade",
        "cifar20-large_outdoor",
        "cifar20-large_omnivores_and_herbivores",
        "cifar20-medium_mammals",
        "cifar20-non_insect_invertebrates",
        "cifar20-people",
        "cifar20-reptiles",
        "cifar20-small_mammals",
        "cifar20-trees",
        "cifar20-veh1",
        "cifar20-veh2"
    ]
    m = Model("nickypro/vit-cifar100", limit=1000)
    with torch.no_grad():
        for dataset_name in dataset_list:
            evaluate_all(m, 2, [dataset_name])

if __name__ == "__main__":
    test_text_datasets()
    test_img_datasets()
