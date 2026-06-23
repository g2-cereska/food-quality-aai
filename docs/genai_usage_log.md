# Generative AI usage log

This log documents every use of generative AI tools throughout this
project, per the AAI resit brief's requirement to cite, evaluate, and
reflect on GenAI use (10% of the module mark).

Entries are added chronologically as the project progresses.

---

## Entry 1

- **Tool**: Claude (Anthropic), Sonnet 4.6, claude.ai web interface
- **Date**: 22 June 2026
- **Purpose**: Project planning and repo structure design
- **Prompt (summarised)**: Asked for a review of a previous attempt's
  codebase against the AAI marking criteria, then for help planning a new
  repository structure with an incremental, commit-by-commit build
  sequence reflecting the AAI brief's requirements (model comparison,
  XAI, fairness analysis, GenAI documentation).
- **What was used**: The proposed repository structure (`src/`, `app/`,
  `docs/`, `tests/`), the 15-commit build roadmap, and the initial
  scaffolding files (`.gitignore`, `requirements.txt`, `README.md`,
  `docs/dataset.md`).
- **What was changed/rejected**: N/A at this stage — structure was
  accepted as proposed and will be evaluated as the project develops.
- **Reflection**: Useful for translating the marking rubric into a
  concrete, demonstrable build order rather than a vague task list. The
  main risk of over-relying on this kind of planning is that it can
  front-load structure before the actual technical decisions (e.g. model
  choice, hyperparameters) have been tested — those decisions still need
  to be made and justified independently as training results come in.

## Entry 2

- **Tool**: Claude (Anthropic), Sonnet 4.6
- **Date**: 22 June 2026
- **Purpose**: Writing `src/data_utils.py` (data loading, splitting,
  transforms)
- **Prompt (summarised)**: Asked to adapt a data loading pipeline from a
  previous project (which had a `SafeImageFolder` and `TransformSubset`
  pattern) into a clean, documented module for this new repository.
- **What was used**: The overall structure (skip-corrupted-image
  handling, single-source-of-truth split with per-split transforms to
  avoid data leakage), reproducible seeded splitting, and a
  `class_counts()` helper added in anticipation of the class imbalance
  analysis later in the project.
- **What was changed/rejected**: N/A yet — to be tested once the real
  dataset is downloaded locally. Will note here if directory structure
  assumptions need correcting once run against the actual Kaggle folder
  layout.
- **Reflection**: The data leakage avoidance pattern (single raw dataset
  + split indices + transform-per-subset) is a genuine technical decision
  worth understanding rather than copying blindly — confirmed
  understanding of why fitting transforms separately per split would be
  incorrect (train-time augmentation must not be applied to val/test
  data, but all three splits must come from the same underlying sample
  pool and same random seed for a fair comparison).

<!-- Add new entries below this line as the project progresses -->

## Entry 3

- **Tool**: Claude (Anthropic), Sonnet 4.6
- **Date**: 23 June 2026
- **Purpose**: Sourcing and integrating a second dataset (Freshness44)
  for an out-of-distribution generalisation test
- **Prompt (summarised)**: I raised a concern about whether high
  in-distribution test accuracy would hold up on real-world images from
  a different source. I mentioned a dataset I had used in a previous
  attempt at this assignment (Freshness44). Asked Claude to look up what
  the dataset actually contained, help map its class folders onto my
  trained model's 28 classes, and write a script to run my trained model
  against a sample of it.
- **What was used**: The web search confirming Freshness44's structure
  and origin (a merge of 5 datasets, built for generalisation testing).
  The class mapping dictionary between my dataset's naming convention
  (`Apple__Healthy`) and Freshness44's (`Apple_Fresh`). The OOD test
  script structure (sampling, inference loop, metrics output matching my
  existing outputs/ format).
- **What was changed/rejected**: I personally found, during manual
  inspection of the downloaded folders, that Freshness44's "Grape"
  folder actually contains grapefruit images, not grapes - this was not
  something Claude or the search results caught, and it meant excluding
  Grape__Healthy/Grape__Rotten from the test rather than using a
  mismatched class as Claude's first mapping attempt assumed.
- **Reflection**: This is a good example of why I shouldn't take a
  generated class-mapping at face value - the folder *names* looked
  like a correct match, but only opening the actual images caught the
  error. Automated/LLM-assisted mapping is useful for the bulk of
  correct matches but still needs a human sanity check on ambiguous
  ones, especially when working across datasets that weren't built by
  the same team with the same naming conventions.