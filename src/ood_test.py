"""Out-of-distribution generalisation test.

Evaluates a trained model against a sample of the Freshness44 dataset
(https://www.kaggle.com/datasets/siavash93/freshness44), which is built
from five different photo sources than the training dataset. This tests
whether the high in-distribution test accuracy holds up on images from
different cameras, lighting, and backgrounds - see docs/dataset.md and
the Generalisation section of the technical report.

26 of the 28 trained classes have a direct equivalent in Freshness44
(Grape__Healthy/Rotten is excluded - Freshness44's "Grape" folder is
grapefruit, not the same produce type, so there is no valid match).

Usage:
    python -m src.ood_test --architecture efficientnet_b0 --samples-per-class 100

Expects the trimmed Freshness44 folders (only the matched classes) under
ood_test_data/Freshness44/, e.g. ood_test_data/Freshness44/Apple_Fresh.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix

from src.data_utils import EVAL_TRANSFORMS
from src.model_utils import build_model, get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OOD_DIR = PROJECT_ROOT / "ood_test_data" / "Freshness44"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ood_test"

CLASS_NAME_MAP = {
    "Apple__Healthy": "Apple_Fresh",
    "Apple__Rotten": "Apple_Rotten",
    "Banana__Healthy": "Banana_Fresh",
    "Banana__Rotten": "Banana_Rotten",
    "Bellpepper__Healthy": "Bellpepper_Fresh",
    "Bellpepper__Rotten": "Bellpepper_Rotten",
    "Carrot__Healthy": "Carrot_Fresh",
    "Carrot__Rotten": "Carrot_Rotten",
    "Cucumber__Healthy": "Cucumber_Fresh",
    "Cucumber__Rotten": "Cucumber_Rotten",
    "Guava__Healthy": "Guava_Fresh",
    "Guava__Rotten": "Guava_Rotten",
    "Jujube__Healthy": "Jujube_Fresh",
    "Jujube__Rotten": "Jujube_Rotten",
    "Mango__Healthy": "Mango_Fresh",
    "Mango__Rotten": "Mango_Rotten",
    "Orange__Healthy": "Orange_Fresh",
    "Orange__Rotten": "Orange_Rotten",
    "Pomegranate__Healthy": "Pomegranate_Fresh",
    "Pomegranate__Rotten": "Pomegranate_Rotten",
    "Potato__Healthy": "Potato_Fresh",
    "Potato__Rotten": "Potato_Rotten",
    "Strawberry__Healthy": "Strawberry_Fresh",
    "Strawberry__Rotten": "Strawberry_Rotten",
    "Tomato__Healthy": "Tomato_Fresh",
    "Tomato__Rotten": "Tomato_Rotten",
}

EXCLUDED_CLASSES = ["Grape__Healthy", "Grape__Rotten"]

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        required=True,
        choices=("efficientnet_b0", "resnet18", "mobilenet_v2"),
    )
    parser.add_argument("--ood-dir", default=str(DEFAULT_OOD_DIR))
    parser.add_argument("--samples-per-class", type=int, default=100)
    return parser.parse_args()


def load_class_names(architecture: str) -> list[str]:
    metrics_path = PROJECT_ROOT / "outputs" / architecture / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Train this architecture first with "
            f"python -m src.train --architecture {architecture}"
        )
    with open(metrics_path) as f:
        metrics = json.load(f)
    return metrics["class_names"]


def sample_image_paths(
    ood_dir: Path, folder_name: str, samples_per_class: int, seed: int
) -> list[Path]:
    class_dir = ood_dir / folder_name
    if not class_dir.exists():
        raise FileNotFoundError(
            f"Expected folder not found: {class_dir}. "
            f"Check that the Freshness44 folders were copied into {ood_dir}."
        )
    all_images = sorted(
        p for p in class_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    rng = random.Random(seed)
    rng.shuffle(all_images)
    return all_images[:samples_per_class]


def load_image_tensor(path: Path):
    try:
        image = Image.open(path).convert("RGB")
    except (OSError, UnidentifiedImageError):
        return None
    return EVAL_TRANSFORMS(image)


def main() -> None:
    args = parse_args()
    device = get_device()
    ood_dir = Path(args.ood_dir)

    class_names = load_class_names(args.architecture)
    print(f"Architecture: {args.architecture}")
    print(f"Device: {device}")
    print(f"Trained classes: {len(class_names)}")
    print(f"Excluded from OOD test (no Freshness44 equivalent): {EXCLUDED_CLASSES}")

    model = build_model(args.architecture, num_classes=len(class_names)).to(device)
    weights_path = PROJECT_ROOT / "outputs" / args.architecture / "best_model.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    true_labels = []
    predicted_labels = []
    skipped_unreadable = 0

    for class_name in class_names:
        if class_name in EXCLUDED_CLASSES:
            continue
        folder_name = CLASS_NAME_MAP[class_name]
        true_label_index = class_names.index(class_name)

        image_paths = sample_image_paths(
            ood_dir, folder_name, args.samples_per_class, SEED
        )
        print(f"{class_name} -> {folder_name}: sampled {len(image_paths)} images")

        for path in image_paths:
            tensor = load_image_tensor(path)
            if tensor is None:
                skipped_unreadable += 1
                continue
            tensor = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
                predicted_index = torch.argmax(output, dim=1).item()
            true_labels.append(true_label_index)
            predicted_labels.append(predicted_index)

    present_label_indices = sorted(set(true_labels) | set(predicted_labels))
    present_class_names = [class_names[i] for i in present_label_indices]

    report = classification_report(
        true_labels, predicted_labels,
        labels=present_label_indices, target_names=present_class_names,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=present_label_indices)

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(matrix)
    ax.set_xticks(range(len(present_class_names)))
    ax.set_yticks(range(len(present_class_names)))
    ax.set_xticklabels(present_class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(present_class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"OOD confusion matrix ({args.architecture} on Freshness44 sample)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    summary = {
        "architecture": args.architecture,
        "source_dataset": "Freshness44 (Kaggle: siavash93/freshness44)",
        "samples_per_class_requested": args.samples_per_class,
        "total_images_evaluated": len(true_labels),
        "skipped_unreadable_images": skipped_unreadable,
        "excluded_classes": EXCLUDED_CLASSES,
        "ood_accuracy": round(report["accuracy"], 4),
        "ood_macro_f1": round(report["macro avg"]["f1-score"], 4),
    }
    (output_dir / "ood_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nOOD accuracy: {summary['ood_accuracy']:.4f}")
    print(f"OOD macro F1: {summary['ood_macro_f1']:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()