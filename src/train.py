"""Train a single model architecture on the fruit/vegetable dataset.

Usage:
    python -m src.train --architecture efficientnet_b0 --epochs 10
    python -m src.train --architecture resnet18 --epochs 10
    python -m src.train --architecture mobilenet_v2 --epochs 10

Writes outputs/<architecture>/metrics.json, confusion_matrix.png,
training_curves.png, classification_report.csv, and best_model.pth.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data_utils import build_splits, class_counts, _load_raw_dataset
from src.model_utils import build_model, count_parameters, get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "Fruit And Vegetable Diseases Dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        required=True,
        choices=("efficientnet_b0", "resnet18", "mobilenet_v2"),
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--weighted-sampling", action="store_true",
                    help="Use WeightedRandomSampler to compensate for class imbalance.")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Runs one epoch. If optimizer is None, runs in evaluation mode
    (no gradient updates).
    """
    is_training = optimizer is not None
    model.train(mode=is_training)

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            if is_training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


def evaluate_test_set(
    model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str]
) -> dict:
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels.tolist())

    report = classification_report(
        all_labels, all_predictions, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(all_labels, all_predictions)
    return {"report": report, "confusion_matrix": matrix, "labels": all_labels,
            "predictions": all_predictions}


def save_confusion_matrix(matrix, class_names: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_training_curves(
    train_losses: list[float], val_losses: list[float],
    train_accs: list[float], val_accs: list[float],
    output_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label="train")
    ax1.plot(val_losses, label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(train_accs, label="train")
    ax2.plot(val_accs, label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Architecture: {args.architecture}")
    print(f"Device: {device}")

    suffix = "_weighted" if args.weighted_sampling else ""
    output_dir = PROJECT_ROOT / "outputs" / f"{args.architecture}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)


    train_dataset, val_dataset, test_dataset, class_names = build_splits(args.data_dir)
    print(f"Classes: {len(class_names)}")
    print(f"Train/val/test sizes: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    if args.weighted_sampling:
        # Each training sample is weighted by the inverse of its class
        # frequency, so smaller classes are sampled more often per epoch.
        # This addresses the 14x imbalance between the largest classes
        # (Apple, Banana ~2000-3000 images) and smallest (Grape, Guava,
        # Jujube, Pomegranate - exactly 200 images each).
        raw = _load_raw_dataset(args.data_dir)
        counts = class_counts(raw)
        total = sum(counts.values())
        class_weights = {name: total / count for name, count in counts.items()}
        sample_weights = [
            class_weights[class_names[label]]
            for _, label in raw.samples
        ]
        train_indices = train_dataset.subset.indices
        train_weights = [sample_weights[i] for i in train_indices]
        sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=sampler, num_workers=args.num_workers,
        )
        print("Using WeightedRandomSampler to compensate for class imbalance.")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    model = build_model(args.architecture, num_classes=len(class_names)).to(device)
    param_counts = count_parameters(model)
    print(f"Parameters: {param_counts}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    best_model_path = output_dir / "best_model.pth"

    training_start = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} - "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    training_time_seconds = time.time() - training_start

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    inference_start = time.time()
    test_results = evaluate_test_set(model, test_loader, device, class_names)
    inference_time_seconds = time.time() - inference_start
    avg_inference_ms_per_image = (inference_time_seconds / len(test_dataset)) * 1000

    save_confusion_matrix(test_results["confusion_matrix"], class_names,
                           output_dir / "confusion_matrix.png")
    save_training_curves(train_losses, val_losses, train_accs, val_accs,
                          output_dir / "training_curves.png")

    report_df = pd.DataFrame(test_results["report"]).transpose()
    report_df.to_csv(output_dir / "classification_report.csv")

    metrics = {
        "architecture": args.architecture,
        "device": str(device),
        "num_classes": len(class_names),
        "class_names": class_names,
        "dataset_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weighted_sampling": args.weighted_sampling,
        },
        "parameters": param_counts,
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_results["report"]["accuracy"], 4),
        "test_macro_f1": round(test_results["report"]["macro avg"]["f1-score"], 4),
        "training_time_seconds": round(training_time_seconds, 1),
        "avg_inference_ms_per_image": round(avg_inference_ms_per_image, 2),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nDone. Test accuracy: {metrics['test_accuracy']:.4f}, "
          f"macro F1: {metrics['test_macro_f1']:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()