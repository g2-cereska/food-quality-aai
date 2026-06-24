"""Grad-CAM explainability for the trained classifier.

Generates heatmap overlays showing which regions of an image most
strongly influenced the model's prediction, using Grad-CAM (Selvaraju
et al. 2017). This addresses the AAI brief's requirement to "demonstrate
which image regions contribute most strongly to the model's decisions"
and to "discuss situations where explanations may reveal weaknesses or
biases in the model."

Runs on BOTH in-distribution images (the original training dataset) and
out-of-distribution images (the Freshness44 sample, see ood_test.py) for
the same set of classes, so heatmaps can be compared side by side to see
whether the model's attention pattern changes on unfamiliar images - this
is the most direct way to investigate *why* some classes (Pomegranate,
Guava) collapsed under distribution shift while others held up.

Target classes contrast known weak/OOD-fragile classes against known
strong ones (see outputs/<architecture>/classification_report.csv and
outputs/ood_test/classification_report.csv):
  Weak / OOD-fragile: Pomegranate__Healthy, Guava__Rotten, Bellpepper__Rotten
  Strong:             Apple__Healthy, Banana__Rotten

Usage:
    python -m src.xai --architecture efficientnet_b0 --images-per-class 4

Requires: pip install grad-cam
Writes overlay images to outputs/<architecture>/gradcam/<source>/<class_name>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data_utils import EVAL_TRANSFORMS, _load_raw_dataset
from src.model_utils import build_model, get_device
from src.ood_test import CLASS_NAME_MAP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "Fruit And Vegetable Diseases Dataset"
DEFAULT_OOD_DIR = PROJECT_ROOT / "ood_test_data" / "Freshness44"

DEFAULT_TARGET_CLASSES = [
    "Pomegranate__Healthy",
    "Guava__Rotten",
    "Bellpepper__Rotten",
    "Apple__Healthy",
    "Banana__Rotten",
]

# Maps architecture name to the layer Grad-CAM should hook into. This is
# architecture-specific: each backbone exposes its final convolutional
# block under a different attribute path. Confirmed against the actual
# torchvision model structure (model.features is a Sequential for
# efficientnet_b0 and mobilenet_v2; model.layer4 for resnet18).
TARGET_LAYER_MAP = {
    "efficientnet_b0": lambda model: [model.features[-1]],
    "mobilenet_v2": lambda model: [model.features[-1]],
    "resnet18": lambda model: [model.layer4[-1]],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        required=True,
        choices=("efficientnet_b0", "resnet18", "mobilenet_v2"),
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--ood-dir", default=str(DEFAULT_OOD_DIR))
    parser.add_argument("--images-per-class", type=int, default=4)
    parser.add_argument(
        "--target-classes",
        nargs="+",
        default=DEFAULT_TARGET_CLASSES,
        help="Class names to generate Grad-CAM heatmaps for.",
    )
    return parser.parse_args()


def load_class_names(architecture: str) -> list[str]:
    metrics_path = PROJECT_ROOT / "outputs" / architecture / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Train this architecture first."
        )
    with open(metrics_path) as f:
        return json.load(f)["class_names"]


def find_in_distribution_paths(data_dir: Path, class_name: str, n: int) -> list[Path]:
    """Finds up to n image paths belonging to the given class folder in
    the original training dataset.
    """
    raw = _load_raw_dataset(data_dir)
    class_index = raw.class_to_idx[class_name]
    matching_paths = [
        Path(path) for path, label in raw.samples if label == class_index
    ]
    return matching_paths[:n]


def find_ood_paths(ood_dir: Path, class_name: str, n: int) -> list[Path]:
    """Finds up to n image paths for the equivalent Freshness44 folder,
    using the same class name mapping as ood_test.py. Returns an empty
    list if this class has no OOD equivalent (e.g. Grape__*).
    """
    folder_name = CLASS_NAME_MAP.get(class_name)
    if folder_name is None:
        return []
    class_dir = ood_dir / folder_name
    if not class_dir.exists():
        return []
    images = sorted(
        p for p in class_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    return images[:n]


def overlay_for_image(
    image_path: Path, model, cam: GradCAM, device: torch.device,
    target_class_index: int,
):
    """Returns (overlay_rgb_uint8, predicted_class_index, confidence)."""
    pil_image = Image.open(image_path).convert("RGB").resize((224, 224))
    rgb_float = np.array(pil_image).astype(np.float32) / 255.0

    input_tensor = EVAL_TRANSFORMS(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = int(torch.argmax(logits, dim=1).item())
        confidence = float(probabilities[0, predicted_index].item())

    targets = [ClassifierOutputTarget(target_class_index)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return overlay, predicted_index, confidence


def process_source(
    source_name: str,
    image_paths: list[Path],
    class_name: str,
    class_names: list[str],
    target_class_index: int,
    model,
    cam: GradCAM,
    device: torch.device,
    output_root: Path,
    log_rows: list[dict],
) -> None:
    if not image_paths:
        print(f"  [{source_name}] no images found for {class_name} - skipping")
        return

    class_output_dir = output_root / source_name / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(image_paths):
        overlay, predicted_index, confidence = overlay_for_image(
            image_path, model, cam, device, target_class_index
        )
        predicted_name = class_names[predicted_index]
        correct = predicted_name == class_name

        output_path = class_output_dir / f"{i}_{'correct' if correct else 'WRONG'}.png"
        Image.fromarray(overlay).save(output_path)

        log_rows.append({
            "source": source_name,
            "true_class": class_name,
            "predicted_class": predicted_name,
            "correct": correct,
            "confidence": round(confidence, 4),
            "source_image": str(image_path),
            "output_image": str(output_path),
        })
        print(f"  [{source_name}] {class_name} [{i}]: predicted={predicted_name} "
              f"({'correct' if correct else 'WRONG'}, conf={confidence:.3f})")


def main() -> None:
    args = parse_args()
    device = get_device()
    data_dir = Path(args.data_dir)
    ood_dir = Path(args.ood_dir)

    class_names = load_class_names(args.architecture)
    model = build_model(args.architecture, num_classes=len(class_names)).to(device)
    weights_path = PROJECT_ROOT / "outputs" / args.architecture / "best_model.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    target_layers = TARGET_LAYER_MAP[args.architecture](model)
    output_root = PROJECT_ROOT / "outputs" / args.architecture / "gradcam"
    output_root.mkdir(parents=True, exist_ok=True)

    log_rows: list[dict] = []

    with GradCAM(model=model, target_layers=target_layers) as cam:
        for class_name in args.target_classes:
            if class_name not in class_names:
                print(f"Skipping unknown class: {class_name}")
                continue
            target_class_index = class_names.index(class_name)
            print(f"\n{class_name}:")

            in_dist_paths = find_in_distribution_paths(
                data_dir, class_name, args.images_per_class
            )
            process_source(
                "in_distribution", in_dist_paths, class_name, class_names,
                target_class_index, model, cam, device, output_root, log_rows,
            )

            ood_paths = find_ood_paths(ood_dir, class_name, args.images_per_class)
            process_source(
                "ood", ood_paths, class_name, class_names,
                target_class_index, model, cam, device, output_root, log_rows,
            )

    (output_root / "gradcam_log.json").write_text(json.dumps(log_rows, indent=2))

    total = len(log_rows)
    correct = sum(1 for r in log_rows if r["correct"])
    print(f"\n{correct}/{total} correctly classified across sampled images.")
    print(f"Saved {total} Grad-CAM overlays to {output_root}")


if __name__ == "__main__":
    main()