# Fruit & Vegetable Quality Classification — Advanced AI (UFCFUR-15-3)

AI-based computer vision solution for the Bristol Regional Food Network case
study: classifying fresh vs rotten/stale fruit and vegetables from images,
with an emphasis on explainability and fairness as required by the resit
brief.

## Status

This repository is being built incrementally. See commit history for the
build sequence: data pipeline → baseline model → model comparison → class
imbalance handling → explainability (Grad-CAM) → fairness analysis →
reporting.

## Dataset

[Fruit and Vegetable Disease (Healthy vs Rotten) — Kaggle](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten)

Not committed to this repository (see `.gitignore`). Download instructions
are in `docs/dataset.md`.

## Project structure

```
src/            Core pipeline: data loading, model definitions, training,
                inference, explainability (Grad-CAM), fairness evaluation
app/            Flask web app for live demo
outputs/        Generated metrics, plots, model comparison results
docs/           Dataset notes, GenAI usage log, technical report,
                executive summary
tests/          pytest test suite
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Models compared

| Model | Purpose |
|---|---|
| EfficientNet-B0 | Strong baseline, efficient architecture |
| ResNet18 | Well-understood classic baseline |
| MobileNetV2 | Edge/low-power deployment comparison point |

## Module context

This project is submitted for **Advanced Artificial Intelligence
(UFCFUR-15-3)**. It is intentionally kept separate from a related DESD
(UFCFTR-30-3) marketplace project — the two assess different learning
outcomes and are not meant to share a codebase or submission.