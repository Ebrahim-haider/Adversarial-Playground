"""
Model loading and ImageNet utilities for Adversarial Playground.
Loads lightweight, pre-trained PyTorch models (ResNet18, MobileNetV3).
"""

import json
import urllib.request
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NormalizedModel(torch.nn.Module):
    """
    Wraps a base model with an input normalization layer so that the raw input
    tensor stays in [0, 1] space. This makes gradient clipping and epsilon calculations
    strictly intuitive and visually correct.
    """
    def __init__(self, base_model, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.base_model = base_model
        # Register mean and std as buffers
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        normalized_x = (x - self.mean) / self.std
        return self.base_model(normalized_x)


def get_model(model_name: str = "ResNet18", device: str = "cpu"):
    """
    Loads pretrained vision model wrapped with normalization.
    """
    if model_name == "ResNet18":
        weights = models.ResNet18_Weights.DEFAULT
        base_model = models.resnet18(weights=weights)
    elif model_name == "MobileNetV3":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        base_model = models.mobilenet_v3_small(weights=weights)
    elif model_name == "ResNet50":
        weights = models.ResNet50_Weights.DEFAULT
        base_model = models.resnet50(weights=weights)
    else:
        weights = models.ResNet18_Weights.DEFAULT
        base_model = models.resnet18(weights=weights)

    base_model.eval()
    wrapped_model = NormalizedModel(base_model).to(device)
    categories = weights.meta["categories"]
    return wrapped_model, categories


def preprocess_image(pil_image: Image.Image, target_size: int = 224) -> torch.Tensor:
    """
    Prepares a PIL Image into a [0, 1] range float tensor (1, 3, target_size, target_size).
    """
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()  # Output values in [0.0, 1.0]
    ])
    tensor = transform(pil_image).unsqueeze(0)
    return tensor
