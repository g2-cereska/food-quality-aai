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
- **Reflection**: The roadmap felt right as soon as I saw it, so I went
  with it straight away rather than questioning the structure. In
  hindsight I think that was a reasonable call given it directly mapped
  onto the marking criteria, but it also means the early structure
  decisions weren't independently stress-tested by me at the time - any
  problems with it would only have shown up once I started actually
  building against it.

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
- **Reflection**: I had a rough sense that training and test data should
  be handled differently, but I hadn't actually thought through why
  augmentation specifically can't be applied to validation/test data, or
  why all three splits need to come from the same underlying random
  split rather than being loaded separately. Seeing it laid out made the
  reasoning click properly rather than just being a rule I was following
  without understanding.

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
- **What was changed/rejected**: Something about the "Grape" folder felt
  off to me when I was going through the folders before deleting the
  unmatched ones, so I opened a few images to check rather than trusting
  the name. That's when I realised it was actually grapefruit, not
  grapes - not something Claude or the earlier web search had caught,
  since the folder name itself looked like a clean match. This meant
  excluding Grape__Healthy/Grape__Rotten from the OOD test rather than
  using a mismatched class as the first mapping attempt assumed.
- **Reflection**: I'm glad I checked rather than just trusting the
  folder name, since the mapping would otherwise have silently tested
  the model against the wrong fruit and the result would have looked
  valid without actually being valid. It's made me more cautious about
  taking a generated class-mapping at face value generally - the names
  matching is not the same as the contents matching, especially across
  datasets that weren't built by the same team with the same naming
  conventions.