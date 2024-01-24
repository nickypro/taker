from torchvision import transforms
from typing import Any

# Improves model performance (https://github.com/weiaicunzai/pytorch-cifar100)
CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

class SsdTensorDict:
    def __init__(self, tensor_dict):
        self.tensor_dict = tensor_dict

    def to(self, device):
        """Move all tensors in the dictionary to the specified device."""
        return {key: value.to(device) for key, value in self.tensor_dict.items()}

    def __getitem__(self, key):
        return self.tensor_dict[key]

    def __setitem__(self, key, value):
        self.tensor_dict[key] = value

class SsdVitProcessor:
    """ Processor used in Selective Synaptic Dampening paper"""
    def __init__(self, train=False):
        self.transform = []

        # If training from scratch, add random crop and flip for robustness
        if train:
            self.transform = [*self.transform,
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ]

        # Convert to ViT format
        self.transform = [*self.transform,
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            transforms.Resize(224),
        ]

        self.transform = transforms.Compose(self.transform)

    def __call__(self, img, return_tensors="pt") -> Any:
        if return_tensors != "pt":
            raise ValueError(f"return_tensors must be 'pt', got {return_tensors}")

        # Transform image based on the defined transform
        processed_img =  self.transform(img)
        if len(processed_img.shape) == 3:
            processed_img = processed_img.unsqueeze(0)

        # return in correct format
        return SsdTensorDict({
            "pixel_values": processed_img,
        })


