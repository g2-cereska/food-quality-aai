"""Quick sanity check: confirms the dataset loads, prints class counts,
and confirms split sizes - run this before a full training job.

Usage:
    python -m src.check_dataset
"""

from __future__ import annotations

from pathlib import Path

from src.data_utils import _load_raw_dataset, build_splits, class_counts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Fruit And Vegetable Diseases Dataset"


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found at {DATA_DIR}. "
            f"See docs/dataset.md for download instructions."
        )

    raw = _load_raw_dataset(DATA_DIR)
    print(f"Total images: {len(raw)}")
    print(f"Classes found: {len(raw.classes)}")

    counts = class_counts(raw)
    print("\nPer-class image counts:")
    for class_name, count in sorted(counts.items()):
        print(f"  {class_name}: {count}")

    train_dataset, val_dataset, test_dataset, class_names = build_splits(DATA_DIR)
    print(f"\nSplit sizes - train: {len(train_dataset)}, "
          f"val: {len(val_dataset)}, test: {len(test_dataset)}")

    sample_image, sample_label = train_dataset[0]
    print(f"\nSample tensor shape: {sample_image.shape}")
    print(f"Sample label index: {sample_label} ({class_names[sample_label]})")
    print("\nDataset check passed.")


if __name__ == "__main__":
    main()