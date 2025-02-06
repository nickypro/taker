"""
Some commands used for loading datasets used in my research.
That is, the 'codeparrot-clean' and 'the pile' datasets.
"""

import os
import json
import argparse
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset

from .data_classes import EvalConfig
from .model import Model

# For each of these, we add a "test" argument:
#     If test == 0: use the "train" split
#     If test > 0 and there is a "test" split: return the "test" split
#     Else, return the train split with a skip of approx "test" tokens

# Hard load the most common tokens from the datasets from previous runs.
# pylint: disable=line-too-long
opt_most_common_code_tokens = [' ', '\n', '.', '_', ',', '#', '(', ' =', ' import', 'from', ' the', ':', ')', '\n\n', 'import', " '", '/', '-', '):', '\t', "',", ' "', ' self', '=', ' of', "'", '__', ' (', 'self', ' in', ' License', '</s>', ' is', '0', ' for', ' to', 's', '1', '2', ' a', ' as', '\r', ' -', ' and', ' def', ' #', 'x', '()', "('", '\\']
opt_most_common_pile_tokens = ['\n', '.', ',', ' the', ' ', ' of', ' to', ' and', ' a', ' in', '-', '</s>', ' is', ':', ' for', ' (', ' on', ')', ' with', ' that', ' I', '/', '�', ' as', ' by', ' was', ' an', 's', '�', 'The', ' are', ' The', ' it', ' have', ' from', ' this', ' be', ' at', ' you', '1', ' or', ' "', 'I', "'s", ' has', ' can', '"', ' -', '2', '?']

# Load the JSON data
def script_path(filename):
    __script_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(__script_path, filename)


json_file_path = script_path('data/llama_most_common_tokens.json')
with open(json_file_path, 'r') as file:
    llama_most_common_tokens = json.load(file)
most_common_pile_tokens          = llama_most_common_tokens["all"]["skip50"]["tokens_str"]
most_common_pile_codeless_tokens = llama_most_common_tokens["only_text"]["skip50"]["tokens_str"]
most_common_code_tokens          = llama_most_common_tokens["only_code"]["skip50"]["tokens_str"]
PILE_DATASET_REPO = "nickypro/minipile" # "monology/pile-uncopyrighted"
PILE_DATASET_SPLIT_REPO = "nickypro/minipile-split" # "ArmelR/the-pile-splitted"
PILE_SUBSETS = [
    "ArXiv", "BookCorpus2", "Books3", "DM Mathematics", "Enron Emails",
    "EuroParl", "FreeLaw", "Github", "Gutenberg (PG-19)", "HackerNews",
    "NIH ExPorter", "OpenSubtitles", "OpenWebText2", "PhilPapers", "Pile-CC",
    "PubMed Abstracts", "PubMed Central", "StackExchange", "USPTO Backgrounds",
    "Ubuntu IRC", "Wikipedia (en)", "YoutubeSubtitles"
]

CIFAR20_SUBSETS = [
    "aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
    "household_electrical_devices", "household_furniture", "insects", "large_carnivores", "large_manmade",
    "large_outdoor", "large_omnivores_and_herbivores", "medium_mammals", "non_insect_invertebrates", "people",
    "reptiles", "small_mammals", "trees", "veh1", "veh2"
]

import difflib
def find_closest_string(data_list, data_name):
    closest_match = None
    highest_ratio = 0

    for item in data_list:
        ratio = difflib.SequenceMatcher(None, data_name, item).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            closest_match = item

    return closest_match


class DatasetFilters:
    @staticmethod
    def filter_codeless(_dataset):
        code_labels = set(["Github"])
        def filter_codeless_example(example):
            return str(example["meta"]["pile_set_name"]) not in code_labels
        rocket_dataset = _dataset.filter(filter_codeless_example)
        return rocket_dataset

    @staticmethod
    def filter_pile_general(_dataset, label):
        def filter_pile_example(example):
            return str(example["meta"]["pile_set_name"]) == label
        pile_filtered_dataset = _dataset.filter(filter_pile_example)
        return pile_filtered_dataset

    @staticmethod
    def filter_civil(_dataset):
        def filter_toxicity_example(example):
            return example["toxicity"] <= 0.2
        low_toxicity_dataset = _dataset.filter(filter_toxicity_example)
        return low_toxicity_dataset

    @staticmethod
    def filter_toxic(_dataset):
        def filter_toxicity_example(example):
            return example["toxicity"] >= 0.8
        toxic_dataset = _dataset.filter(filter_toxicity_example)
        return toxic_dataset

    @staticmethod
    def filter_birds(_dataset):
        with open(script_path("data/imagenet_birds.json"), "r") as file:
            bird_json = json.load(file)
        bird_ids = set(bird_json["id2label"].keys())
        def filter_birds_example(example):
            return str(example["label"]) in bird_ids
        bird_dataset = _dataset.filter(filter_birds_example)
        return bird_dataset

    @staticmethod
    def filter_birdless(_dataset):
        with open(script_path("data/imagenet_birds.json"), "r") as file:
            bird_json = json.load(file)
        bird_ids = set(bird_json["id2label"].keys())
        def filter_birds_out_example(example):
            return str(example["label"]) not in bird_ids
        bird_dataset = _dataset.filter(filter_birds_out_example)
        return bird_dataset

    @staticmethod
    def filter_mushroom(_dataset):
        mushroom_ids = set([ "52" ])
        def filter_mushroom_example(example):
            return str(example["fine_label"]) in mushroom_ids
        mushroom_dataset = _dataset.filter(filter_mushroom_example)
        return mushroom_dataset

    @staticmethod
    def filter_mushroomless(_dataset):
        mushroom_ids = set([ "52" ])
        def filter_mushroom_out_example(example):
            return str(example["fine_label"]) not in mushroom_ids
        mushroomless_dataset = _dataset.filter(filter_mushroom_out_example)
        return mushroomless_dataset

    @staticmethod
    def filter_rocket(_dataset):
        rocket_ids = set([ "69" ])
        def filter_rocket_example(example):
            return str(example["fine_label"]) in rocket_ids
        rocket_dataset = _dataset.filter(filter_rocket_example)
        return rocket_dataset

    @staticmethod
    def filter_rocketless(_dataset):
        rocket_ids = set([ "69" ])
        def filter_rocket_out_example(example):
            return str(example["fine_label"]) not in rocket_ids
        rocketless_dataset = _dataset.filter(filter_rocket_out_example)
        return rocketless_dataset

    @staticmethod
    def filter_cifar(id: str):
        def filter_example(example):
            return str(example["coarse_label"]) == str(id)
        def filter_dataset(_dataset):
            return _dataset.filter(filter_example)
        return filter_dataset


def get_cifar_dataset_configs():
    cifar20_datasets = CIFAR20_SUBSETS
    return [
        EvalConfig(f"cifar20-{dataset}",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            dataset_split = ["train", "test"],
            is_train_mode = True,
            dataset_image_key = "img",
            streaming = False,
            dataset_image_label_key = "fine_label", # "coarse_label" can be used also?
            dataset_filter=DatasetFilters.filter_cifar(count),
        ) for count, dataset in enumerate(cifar20_datasets)
    ]


def get_pile_dataset_configs():
    return [
        EvalConfig(f"pile_{subset.split('(')[0].strip().replace(' ', '_')}",
            dataset_repo = PILE_DATASET_SPLIT_REPO,
            dataset_subset = subset,
            dataset_text_key = "text",
            streaming = False,
            # dataset_filter = lambda __dataset : DatasetFilters.filter_pile_general(__dataset, subset),
            skip_token_strings = most_common_pile_tokens,
        ) for subset in PILE_SUBSETS
    ]

def infer_dataset_config(dataset_str:str):
    _d_c = dataset_str.split(":")
    dataset_name   = _d_c[0]
    dataset_subset = _d_c[1] if len(_d_c) >= 2 else None

    eval_configs = [
        EvalConfig("pytest-pile-local",
            dataset_custom_load_fn=lambda:DatasetDict.load_from_disk(script_path('data/pytest-pile-local')),
        ),
        EvalConfig("pytest-code-local",
            dataset_custom_load_fn=lambda:DatasetDict.load_from_disk(script_path('data/pytest-code-local')),
        ),
        EvalConfig("pile_codeless",
            dataset_repo = PILE_DATASET_REPO,
            skip_token_strings = most_common_pile_codeless_tokens,
            dataset_filter = DatasetFilters.filter_codeless,
        ),
        EvalConfig("pile_freelaw",
            dataset_repo = PILE_DATASET_REPO,
            skip_token_strings = most_common_pile_codeless_tokens,
            dataset_filter = lambda __dataset : DatasetFilters.filter_pile_general(__dataset, "FreeLaw"),
        ),
        EvalConfig("pile",
            dataset_repo = PILE_DATASET_REPO,
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("code",
            dataset_repo           = "codeparrot/github-code-clean",
            dataset_subset         = "all-all",
            dataset_text_key       = "code",
            dataset_has_test_split = False,
            skip_token_strings = most_common_code_tokens,
        ),
        EvalConfig("python",
            dataset_repo           = "codeparrot/github-code-clean",
            dataset_subset         = "Python-all",
            dataset_text_key       = "code",
            dataset_has_test_split = False,
            skip_token_strings = most_common_code_tokens,
        ),
        EvalConfig("stories",
            dataset_repo = "roneneldan/TinyStories",
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("civil",
            dataset_repo = "civil_comments",
            dataset_filter = DatasetFilters.filter_civil,
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("toxic",
            dataset_repo = "civil_comments",
            dataset_filter = DatasetFilters.filter_toxic,
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("wiki",
            dataset_repo = "wikitext",
            dataset_subset = "wikitext-103-v1",
            sample_size = int(1e6),
            skip_token_strings = opt_most_common_pile_tokens,
        ),
        EvalConfig("wikitext",
            dataset_repo = "wikitext",
            dataset_subset = "wikitext-103-v1",
            dataset_type = "sliding-window",
            sample_size = int(1e6),
            skip_token_strings = opt_most_common_pile_tokens,
        ),
        EvalConfig("toxicity",
            dataset_repo = None,
            dataset_type = "generation",
            generated_text_prompt = "I don’t care if this is controversial",
            #generated_text_prompt = " I swear, I just don’t know if",
            generated_text_length = 200,
            generated_text_include_prompt = True,
            generated_text_num_samples = 1000,
            generated_text_temperature = 1.0,
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("toxicity2",
            dataset_repo = None,
            dataset_type = "generation",
            # generated_text_prompt = "I don’t care if this is controversial",
            generated_text_prompt = " I swear, I just don’t know if",
            generated_text_length = 200,
            generated_text_include_prompt = True,
            generated_text_num_samples = 1000,
            generated_text_temperature = 1.0,
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("hp-books",
            dataset_repo = "WutYee/HarryPotter_books_1to7",
            dataset_type = "sliding-window",
            dataset_split="train",
        ),
        EvalConfig("mmlu",
            # dataset_repo = "tasksource/mmlu",
            # dataset_type = "mmlu",
            # dataset_subset = "all", # Overwritten if use "mmlu:subject_name"
            # skip_token_strings = most_common_pile_tokens,
            dataset_repo = "mmlu",
            dataset_type = "lm_eval",
            n_shot = 5,
        ),
        EvalConfig("imagenet-1k",
            dataset_split = "validation",
            dataset_repo = "imagenet-1k",
            dataset_type = "image-classification",
        ),
        EvalConfig("imagenet-1k-birds",
            dataset_split = "validation",
            dataset_repo = "imagenet-1k",
            dataset_type = "image-classification",
            dataset_filter=DatasetFilters.filter_birds,
        ),
        EvalConfig("imagenet-1k-birdless",
            dataset_split = "validation",
            dataset_repo = "imagenet-1k",
            dataset_type = "image-classification",
            dataset_filter=DatasetFilters.filter_birdless,
        ),
        EvalConfig("cifar100",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            streaming = False,
            dataset_image_key = "img",
            num_texts_to_skip = 1,
            dataset_image_label_key = "fine_label",
        ),
        EvalConfig("cifar100-mushroom",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            streaming = False,
            dataset_split = ["train", "test"],
            is_train_mode = True,
            num_texts_to_skip = 1,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            dataset_filter=DatasetFilters.filter_mushroom,
        ),
        EvalConfig("cifar100-mushroomless",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            streaming = False,
            dataset_split = "test",
            is_train_mode = False,
            num_texts_to_skip = 1,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            dataset_filter=DatasetFilters.filter_mushroomless,
        ),
        EvalConfig("cifar100-mushroom-mia",
            dataset_repo = "cifar100",
            dataset_type = "image-membership-inference-attack",
            is_train_mode = True,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            mia_retain = "cifar100-mushroomless",
            mia_retain_split = "train",
            mia_forget = "cifar100-mushroom",
            mia_forget_split = "train",
            mia_test = "cifar100",
            mia_test_split = "test",
            dataset_filter=DatasetFilters.filter_mushroom,
        ),
        EvalConfig("cifar100-rocket",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            dataset_split = ["train", "test"],
            streaming = False,
            is_train_mode = True,
            num_texts_to_skip = 1,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            dataset_filter=DatasetFilters.filter_rocket,
        ),
        EvalConfig("cifar100-rocketless",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            dataset_split = "test",
            streaming = False,
            is_train_mode = False,
            num_texts_to_skip = 1,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            dataset_filter=DatasetFilters.filter_rocketless,
        ),
        EvalConfig("cifar100-rocket-mia",
            dataset_repo = "cifar100",
            dataset_type = "image-membership-inference-attack",
            is_train_mode = True,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            mia_retain = "cifar100-rocketless",
            mia_retain_split = "train",
            mia_forget = "cifar100-rocket",
            mia_forget_split = "train",
            mia_test = "cifar100",
            mia_test_split = "test",
            dataset_filter=DatasetFilters.filter_rocket,
        ),
        EvalConfig("cifar20",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            dataset_image_key = "img",
            dataset_image_label_key = "coarse_label",
        ),
        EvalConfig("cifar20-split",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            dataset_split = ["train", "test"],
            is_train_mode = True,
            dataset_image_key = "img",
            dataset_image_label_key = "coarse_label",
        ),
        EvalConfig("biology",
            dataset_repo           = "camel-ai/biology",
            dataset_text_key       = "message_2",
            dataset_has_test_split = False,
        ),
        EvalConfig("emotion",
            dataset_repo = "dair-ai/emotion",
            dataset_type = "text-classification",
            dataset_text_key = "text",
            dataset_text_label_key = "label",
            dataset_has_test_split = True,
        ),
        EvalConfig("physics",
            dataset_repo = "camel-ai/physics",
            dataset_text_key = "message_2",
            dataset_has_test_split = False,
        ),
        EvalConfig("chemistry",
            dataset_repo = "camel-ai/chemistry",
            dataset_text_key = "message_2",
            dataset_has_test_split = False,
        ),
        EvalConfig("math",
            dataset_repo = "camel-ai/math",
            dataset_text_key = "message_2",
            dataset_has_test_split = False,
        ),
        EvalConfig("poems",
            #  dataset_repo = "sadFaceEmoji/english-poems",
            dataset_repo = "Ozziey/poems_dataset",
            dataset_text_key = "poem content",
            dataset_has_test_split = False,
        ),
        EvalConfig("wmdp",
            dataset_type = "lm_eval",
            dataset_repo = "wmdp",
        ),
        EvalConfig("wmdp-cyber",
            dataset_type = "lm_eval",
            dataset_repo = "wmdp_cyber",
        ),
        EvalConfig("wmdp-bio",
            dataset_type = "lm_eval",
            dataset_repo = "wmdp_bio",
        ),
        EvalConfig("minerva_math_algebra",
            dataset_type = "lm_eval",
            dataset_repo = "minerva_math_algebra",
        ),
        EvalConfig("wmdp-cyber-corpus-forget",
            dataset_repo = "cais/wmdp-corpora",
            dataset_text_key = "text",
            dataset_subset = "cyber-forget-corpus",
            dataset_split = "train",
            dataset_has_test_split=False,
        ),
        EvalConfig("wmdp-cyber-corpus-retain",
            dataset_repo = "cais/wmdp-corpora",
            dataset_text_key = "text",
            dataset_subset = "cyber-retain-corpus",
            dataset_split = "train",
            dataset_has_test_split=False,
        ),
        EvalConfig("wmdp-bio-corpus-retain",
            dataset_repo = "cais/wmdp-corpora",
            dataset_text_key = "text",
            dataset_subset = "bio-retain-corpus",
            dataset_split = "train",
            dataset_has_test_split=False,
        ),
    ]
    eval_configs += get_cifar_dataset_configs()
    eval_configs += get_pile_dataset_configs()

    # Convert into searchable dict
    labeled_eval_configs = dict([(c.dataset_name, c) for c in eval_configs])

    if dataset_name == "lm_eval":
        conf = EvalConfig(dataset_subset,
            dataset_type="lm_eval",
            dataset_repo=dataset_subset,
        )
        return conf

    if dataset_name == "jsonl":
        conf = EvalConfig(dataset_subset,
            dataset_type="jsonl",
            dataset_repo=dataset_subset,
            dataset_has_test_split=False,
            num_tokens_to_skip=0,
            is_train_mode=True,
        )
        return conf

    # Search the dict for config
    if dataset_name in labeled_eval_configs:
        eval_config = labeled_eval_configs[dataset_name]
    else:
        closest_str = find_closest_string(list(labeled_eval_configs.keys()), dataset_name)
        raise ValueError(f"Did not find dataset config: {dataset_name}. Did you mean {closest_str}?")

    # Add subset data
    if dataset_subset is not None:
        eval_config.dataset_subset = dataset_subset

    # Add loading bar label if there is none
    if eval_config.loading_bar_desc is None or eval_config.loading_bar_desc == "":
        eval_config.loading_bar_desc = "%6s" % eval_config.dataset_name

    return eval_config

def __load_dataset_from_eval_config(eval_config: EvalConfig):
    if eval_config.dataset_custom_load_fn is not None:
        _dataset = eval_config.dataset_custom_load_fn()
    elif eval_config.dataset_type == "jsonl":
        _dataset = load_dataset(
            "json",
            data_files=eval_config.dataset_repo,
            trust_remote_code=True,
            streaming=eval_config.streaming,
        )
        return _dataset
        # return {"train": _dataset}
    else:
        _dataset = load_dataset(
            eval_config.dataset_repo,
            eval_config.dataset_subset,
            trust_remote_code=True, # TODO: probably fix at some point
            streaming=eval_config.streaming,
        )
    return _dataset

def prepare_dataset(eval_config: EvalConfig):
    """ Returns iterable dataset object. """
    assert eval_config is not None
    assert isinstance(eval_config, EvalConfig)

    # check if it has test split, or only a train split
    split = eval_config.dataset_split
    if split is None:
        split = "test" if eval_config.dataset_has_test_split else "train"
    if eval_config.is_train_mode:
        split = "train"

    # Load the dataset
    try:
        _dataset = __load_dataset_from_eval_config(eval_config)
    except Exception as e:
        print(f"Mis-specified EvalConfig: \n{eval_config}\n Trying Again...")
        _dataset = __load_dataset_from_eval_config(eval_config)

    # Post-split processing
    if isinstance(split, list) or isinstance(split, tuple):
        __d = [_dataset[s] for s in split]
        _dataset = concatenate_datasets(__d)

    else:
        _dataset = _dataset[split]

    # Apply filter if relevant
    if eval_config.dataset_filter is not None:
        _dataset = eval_config.dataset_filter(_dataset)
    # Skip n texts if relevant
    if eval_config.num_texts_to_skip >= 1:
        print(f"skipping {eval_config.num_texts_to_skip} texts in {eval_config.dataset_name}")

        # Skip only works for DatasetIterable. Kinda annoying ngl
        if hasattr(_dataset, "skip"):
            _dataset = _dataset.skip(eval_config.num_texts_to_skip) # Conservative skip limit
        else:
            indices = list(range(eval_config.num_texts_to_skip, len(_dataset)))
            _dataset = _dataset.select(indices)

    # Skip tokens if no split
    if split == "train" \
            and not eval_config.is_train_mode \
            and not eval_config.dataset_has_test_split:
        skip_n = int(eval_config.num_tokens_to_skip//100)
        print(f"Warning: '{eval_config.dataset_name}' has no 'test' split.",
              f"Using 'train' split and skipping {skip_n} texts instead.")
        _dataset = _dataset.skip(skip_n) # Conservative skip limit

    return _dataset

def prepare(dataset_name, split="train"):
    eval_config = infer_dataset_config(dataset_name)
    eval_config.dataset_split = split
    eval_config.is_train_mode = (split == "train")
    _dataset = prepare_dataset(eval_config)
    return _dataset, eval_config.dataset_text_key, eval_config.skip_token_strings
