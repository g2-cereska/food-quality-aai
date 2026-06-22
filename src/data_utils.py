"""Data loading and preprocessing for the fruit/vegetable quality dataset.

Expects the Kaggle dataset extracted under data/ with one subfolder per
class, e.g. data/CaseStudyDataset/Apple_Fresh, data/CaseStudyDataset/Apple_Rotten.
See docs/dataset.md for download and setup instructions.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import UnidentifiedImageError
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

IMAGE_SIZE = 224
SEED = 42

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SafeImageFolder(datasets.ImageFolder):
    """An ImageFolder that skips corrupted or unreadable images instead of
    crashing the whole training run.

    Kaggle image datasets occasionally contain a handful of truncated or
    non-image files. Rather than letting one bad file kill an hours-long
    training job, this returns the next valid sample instead.
    """

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (OSError, UnidentifiedImageError):
            # Fall back to the next index (wrapping around) rather than
            # raising mid-epoch.
            return self.__getitem__((index + 1) % len(self.samples))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class TransformSubset(Dataset):
    """Wraps a Subset so train/val/test splits can each apply their own
    transform, while still being drawn from a single underlying
    ImageFolder and a single reproducible random split.

    This avoids the common data-leakage mistake of fitting the dataset
    twice with different transforms and splitting each independently.
    """

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _load_raw_dataset(data_dir: str | Path) -> SafeImageFolder:
    """Loads the dataset without any transform applied yet (transform=None),
    so the same underlying samples can be reused across train/val/test
    splits with different transforms applied per-split.
    """
    return SafeImageFolder(root=str(data_dir), transform=None)


def build_splits(
    data_dir: str | Path,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = SEED,
) -> tuple[TransformSubset, TransformSubset, TransformSubset, list[str]]:
    """Builds reproducible train/val/test splits from a single ImageFolder.

    Returns:
        (train_dataset, val_dataset, test_dataset, class_names)
    """
    raw = _load_raw_dataset(data_dir)
    class_names = raw.classes

    total = len(raw)
    val_size = int(total * val_fraction)
    test_size = int(total * test_fraction)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        raw, [train_size, val_size, test_size], generator=generator
    )

    train_dataset = TransformSubset(train_subset, TRAIN_TRANSFORMS)
    val_dataset = TransformSubset(val_subset, EVAL_TRANSFORMS)
    test_dataset = TransformSubset(test_subset, EVAL_TRANSFORMS)

    return train_dataset, val_dataset, test_dataset, class_names


def class_counts(raw_dataset: SafeImageFolder) -> dict[str, int]:
    """Returns a {class_name: image_count} mapping, useful for spotting
    class imbalance before training (see docs/technical_report.md,
    Class Imbalance section).
    """
    counts: dict[str, int] = {name: 0 for name in raw_dataset.classes}
    for _, label_index in raw_dataset.samples:
        counts[raw_dataset.classes[label_index]] += 1
    return counts
