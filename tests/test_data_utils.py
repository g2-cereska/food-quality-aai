"""Tests for src/data_utils.py.

Builds a small synthetic dataset on disk (tiny coloured PNGs across a
few fake classes) rather than depending on the real 8GB Kaggle dataset,
so these tests run anywhere without a download. This mirrors the real
folder layout (one subfolder per class) that SafeImageFolder expects.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.data_utils import (
    IMAGE_SIZE,
    SafeImageFolder,
    _load_raw_dataset,
    build_splits,
    class_counts,
)

CLASS_COUNTS = {
    "Apple__Healthy": 20,
    "Apple__Rotten": 20,
    "Guava__Healthy": 6,
}


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Creates tmp_path/<class_name>/img_*.png for each class, plus one
    corrupted file in Apple__Healthy to exercise SafeImageFolder's
    skip-and-continue behaviour.
    """
    for class_name, n in CLASS_COUNTS.items():
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(n):
            Image.new("RGB", (32, 32), color=(i % 256, 0, 0)).save(
                class_dir / f"img_{i}.png"
            )

    # A corrupted "image" — valid extension, invalid content.
    (tmp_path / "Apple__Healthy" / "corrupted.png").write_bytes(b"not a real png")

    return tmp_path


def test_class_counts_matches_files_on_disk(synthetic_dataset):
    raw = _load_raw_dataset(synthetic_dataset)
    counts = class_counts(raw)

    # +1 on Apple__Healthy because the corrupted file still counts as a
    # sample on disk (ImageFolder indexes by file presence, not
    # validity) — SafeImageFolder only skips it at __getitem__ time.
    assert counts["Apple__Healthy"] == CLASS_COUNTS["Apple__Healthy"] + 1
    assert counts["Apple__Rotten"] == CLASS_COUNTS["Apple__Rotten"]
    assert counts["Guava__Healthy"] == CLASS_COUNTS["Guava__Healthy"]


def test_safe_image_folder_skips_corrupted_file(synthetic_dataset):
    """Indexing every item in the dataset should never raise, even
    though one file is corrupted — this is the entire point of
    SafeImageFolder over a plain ImageFolder."""
    raw = SafeImageFolder(root=str(synthetic_dataset), transform=None)
    for i in range(len(raw)):
        image, label = raw[i]
        assert image is not None
        assert isinstance(label, int)


def test_build_splits_sizes_sum_to_total(synthetic_dataset):
    train, val, test, class_names = build_splits(
        synthetic_dataset, val_fraction=0.2, test_fraction=0.2
    )
    total = sum(CLASS_COUNTS.values()) + 1  # +1 for the corrupted file
    assert len(train) + len(val) + len(test) == total
    assert set(class_names) == set(CLASS_COUNTS.keys())


def test_build_splits_are_reproducible_with_same_seed(synthetic_dataset):
    train_a, _, _, _ = build_splits(synthetic_dataset, seed=42)
    train_b, _, _, _ = build_splits(synthetic_dataset, seed=42)

    # Same seed -> same underlying indices -> same first sample label.
    _, label_a = train_a[0]
    _, label_b = train_b[0]
    assert label_a == label_b


def test_train_transform_produces_expected_tensor_shape(synthetic_dataset):
    train, _, _, _ = build_splits(synthetic_dataset)
    image, _ = train[0]
    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
