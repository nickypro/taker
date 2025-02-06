from datasets import load_dataset
from huggingface_hub import HfApi
import random
from itertools import islice

def sample_10k(dataset):
    sampled = []
    for item in dataset:
        if random.random() < 0.01:  # Approximate sampling, adjust as needed
            sampled.append(item)
            if len(sampled) >= 10000:
                break
    return sampled

# Download and sample 10k examples from the train split
train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
sampled_train = sample_10k(train_dataset)

# Download and sample 10k examples from the test split
test_dataset = load_dataset("ILSVRC/imagenet-1k", split="test", streaming=True)
sampled_test = sample_10k(test_dataset)

# Create a new dataset dictionary
new_dataset = {
    "train": sampled_train,
    "test": sampled_test
}

# Initialize the Hugging Face API
api = HfApi()

# Upload the dataset to Hugging Face Hub
# Note: You need to be authenticated for this to work
api.create_repo(repo_id="nickypro/mini-imagenet", repo_type="dataset", exist_ok=True)

for split, data in new_dataset.items():
    # Convert list to Dataset object
    dataset = Dataset.from_list(data)
    dataset.push_to_hub("nickypro/mini-imagenet", split=split)

print("Dataset uploaded successfully!")
