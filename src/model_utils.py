"""Model factory for the fruit/vegetable quality classifier.

Supports three backbones for the model comparison required by the AAI
brief: EfficientNet-B0 (strong efficient baseline), ResNet18 (classic,
well-documented baseline), and MobileNetV2 (edge/low-power deployment
comparison point). All three use ImageNet-pretrained weights with the
final classification head replaced for this task.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

SUPPORTED_ARCHITECTURES = ("efficientnet_b0", "resnet18", "mobilenet_v2")


def build_model(architecture: str, num_classes: int) -> nn.Module:
    """Builds a pretrained model with its classification head replaced
    for the given number of classes.

    Args:
        architecture: one of SUPPORTED_ARCHITECTURES.
        num_classes: number of output classes for this dataset.

    Raises:
        ValueError: if architecture is not supported.
    """
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Unknown architecture '{architecture}'. "
        f"Supported: {SUPPORTED_ARCHITECTURES}"
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Returns total and trainable parameter counts.

    Useful for the Computational Efficiency discussion in the report -
    a smaller parameter count is one proxy (alongside measured inference
    time) for suitability on low-cost or edge hardware.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def get_device() -> torch.device:
    """Returns the CUDA device if available, otherwise CPU.

    Training scripts should log the device used, since it materially
    affects timing comparisons between architectures.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")