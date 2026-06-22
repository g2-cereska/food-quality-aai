# Dataset setup

This project uses the **Fruit and Vegetable Disease (Healthy vs Rotten)**
dataset from Kaggle:

https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten

## Download

1. Download the dataset zip from the Kaggle link above (requires a free
   Kaggle account), or use the Kaggle CLI:

   ```bash
   pip install kaggle
   kaggle datasets download -d muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten
   ```

2. Extract it into `data/` at the repo root, so the structure looks like:

   ```
   data/
     CaseStudyDataset/
       Apple_Fresh/
       Apple_Rotten/
       Banana_Fresh/
       Banana_Rotten/
       ...
   ```

3. The `data/` folder is gitignored — the raw images are never committed to
   this repository. Anyone cloning the repo needs to download the dataset
   separately following the steps above.

## Splits

`src/data_utils.py` builds train/validation/test splits at runtime
(default 70/15/15) using a fixed random seed (42), so results are
reproducible without needing to commit a pre-split copy of the dataset.

## Known characteristics (to revisit once trained)

- Class counts are imbalanced — some categories (e.g. apple, banana,
  tomato) have several thousand images, others (e.g. grape, guava) have a
  few hundred. See `class_counts()` in `data_utils.py` and the Class
  Imbalance section of the technical report once populated.
- Image quality, lighting, and background vary considerably across
  classes, since the images are sourced from different photographers and
  conditions rather than a single controlled capture setup.