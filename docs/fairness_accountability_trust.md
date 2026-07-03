# Fairness, Accountability and Trust analysis

This document analyses the trained EfficientNet-B0 classifier against
the AAI brief's Fairness, Accountability and Trust (FAT) considerations,
using the evidence gathered from the model comparison, the
out-of-distribution (OOD) generalisation test, and the Grad-CAM
explainability investigation (see outputs/, docs/dataset.md, and
src/ood_test.py / src/xai.py).

This is a working document for assembling the technical report - it is
intentionally more detailed than the final report's word count will
allow, so material can be selected and trimmed deliberately rather than
cut under pressure.

## 1. Is model performance consistent across classes?

No. In-distribution test performance (97.47% accuracy, 0.9595 macro F1
across 28 classes) looks strong as a single headline number, but per-class
performance is uneven:

- Strongest classes (F1 > 0.98): Apple__Healthy, Banana__Healthy,
  Banana__Rotten, Orange__Healthy, Guava__Healthy, Strawberry__Rotten
- Weakest classes (F1 < 0.90): Bellpepper__Rotten (0.830),
  Carrot__Rotten (0.848), Potato__Rotten (0.875), Carrot__Healthy (0.919)

Critically, the dataset's smallest classes by image count
(Grape, Guava, Jujube, Pomegranate - 200 images each in the training
data) did not correspond to the weakest in-distribution performance.
The weakest classes (Bellpepper, Carrot, Potato Rotten) are mid-sized
classes. This suggests the in-distribution weaknesses are driven more by
visual similarity between classes (e.g. rotten bellpepper and rotten
tomato share blistered, darkened texture - see section 3) than by simple
data scarcity.

## 2. Does performance hold up on images from a different source?

No - and the gap is substantial. Testing the same model against a sample
of Freshness44 (a dataset built from five different photo sources,
covering 26 of the 28 trained classes) found:

| Metric | In-distribution | Out-of-distribution | Drop |
|---|---|---|---|
| Accuracy | 97.47% | 76.77% | -20.7 points |
| Macro F1 | 0.9595 | 0.6858 | -0.274 |

This is a materially large generalisation gap. A 97% accuracy figure
quoted in isolation substantially overstates how the model would likely
perform on images sourced differently to the training set (different
cameras, lighting, backgrounds, photographers).

Critically, the classes that struggled in-distribution did not reliably
predict which classes would fail under distribution shift:

- Pomegranate_Healthy and Guava_Rotten were not in-distribution weak
  classes (F1 ~0.97-0.99) but collapsed hardest under distribution shift
  (Pomegranate_Healthy OOD recall: 0.06; Guava_Rotten OOD recall: 0.15).
- Apple_Healthy, a near-perfect in-distribution class, also failed
  completely on OOD images (4/4 sampled images misclassified).
- Conversely, Bellpepper_Rotten (already weak in-distribution) degraded
  further but less catastrophically, and Banana_Rotten held up
  perfectly on OOD samples (4/4 correct, 100% confidence).

Implication: in-distribution test accuracy is not a reliable proxy for
real-world deployment performance for this model. A class performing
well in testing provides no guarantee it will perform well on images
captured under different conditions than the training data.

## 3. What is causing the generalisation failures? (Grad-CAM evidence)

Grad-CAM heatmaps were generated for five classes (three OOD-fragile:
Pomegranate_Healthy, Guava_Rotten, Bellpepper_Rotten; two controls:
Apple_Healthy, Banana_Rotten), each on four in-distribution and four OOD
images. This reveals at least three distinct failure modes, not one
undifferentiated "the model is brittle" phenomenon:

Mode A - confident shortcut (Pomegranate_Healthy). All 4 in-distribution
images correctly classified at 99-100% confidence, with attention
concentrated on dark surface markings near the stem/calyx region. All 4
OOD images confidently misclassified as Tomato__Healthy at 97-99%
confidence - not uncertain, confidently wrong. The model appears to have
learned "dark blotch on round fruit surface" as a distinguishing feature
for pomegranate, rather than its actual shape, crown, or skin texture.
This shortcut happens to also resemble tomato under different
lighting/photography conditions.

Mode B - genuine uncertainty (Guava_Rotten). OOD predictions scattered
across three different wrong classes (Mango_Rotten, Cucumber_Rotten,
Bellpepper_Rotten) at volatile, often low confidence (0.25-0.88). This
looks like the model genuinely does not know what it is looking at,
rather than confidently misreading it - a different and arguably less
concerning failure than Mode A, since a downstream system could
plausibly use low confidence as a signal to flag for human review.

Mode C - real visual similarity (Bellpepper_Rotten). All 4 OOD images
misclassified as Tomato_Rotten at consistent but modest confidence
(0.55-0.57). Rotten bellpepper and rotten tomato share blistered,
darkened, irregular surface texture. This looks like a genuine,
dataset-independent visual similarity problem rather than an artifact of
either dataset specifically, consistent with this class already being
the weakest in-distribution performer.

Control comparison. Banana_Rotten generalised perfectly (4/4 OOD correct
at 100% confidence), demonstrating the OOD drop is not uniform across
all classes - some features genuinely transfer. Apple_Healthy, included
as a second control on the assumption that a near-perfect in-distribution
class would be a "safe" comparison point, unexpectedly also failed
completely on OOD images, with no consistent wrong answer (predictions
scattered, confidence 0.48-0.78). This is itself a finding: even classes
assumed safe cannot be assumed to generalise.

## 4. Bias and dataset composition

The training dataset (Fruit and Vegetable Diseases Dataset, Kaggle) has
substantial class imbalance: Apple, Banana, Orange, Mango each have
1,800-3,000+ images per class, while Grape, Guava, Jujube, and
Pomegranate each have exactly 200 images per Healthy/Rotten split. This
is roughly a 14x imbalance between the largest and smallest classes.

To investigate whether this imbalance was a root cause of the weaker
performance on small classes, EfficientNet-B0 was retrained using
WeightedRandomSampler (each training sample weighted by the inverse of
its class frequency, so smaller classes are sampled more often per
epoch). Results (outputs/efficientnet_b0_weighted/):

| Metric | Baseline | Weighted | Change |
|---|---|---|---|
| Overall accuracy | 97.47% | 97.56% | +0.09pp |
| Macro F1 | 0.9595 | 0.9647 | +0.005 |
| Training time | 976s | 1570s | +61% |

Per-class, weighting improved the four originally-smallest classes
(Jujube_Healthy +0.048, Pomegranate_Healthy +0.029, Guava_Rotten +0.033,
Bellpepper_Rotten +0.027), but came with a trade-off: some
previously-strong classes slipped slightly (Grape, Guava_Healthy,
Pomegranate_Rotten all down 0.012-0.017), and Potato_Rotten - one of
the original weak classes - actually declined marginally (-0.020).
Training took 61% longer for a negligible gain in overall accuracy.

Interpretation: the class imbalance was a partial contributor to the
weaknesses in small classes, but not the sole cause. Bellpepper_Rotten
improved with weighting but remained the weakest class overall (F1
0.857), consistent with Mode C (real visual similarity to Tomato_Rotten)
being the primary driver of its difficulty rather than data scarcity.
For Pomegranate_Healthy, the in-distribution F1 improved to 1.000 with
weighting, but the Grad-CAM evidence from Mode A suggests the model
still relies on a dataset-specific shortcut that weighting alone would
not eliminate - the OOD generalisation problem for this class would need
targeted data collection under varied conditions to address properly.

Whether the improved small-class in-distribution F1 from weighting also
improves OOD generalisation for those classes is an open question not
tested here (running a full OOD re-evaluation after weighted retraining
would be the natural next step in a longer project).

Image quality, lighting, and background also vary considerably within
the training set, since images are sourced from different photographers
and capture conditions rather than a single controlled setup - this is a
plausible contributor to the OOD generalisation gap, since the model may
have learned source-specific visual cues (background colour, typical
lighting setup) rather than produce-specific features for some classes.

## 5. Monitoring and drift detection strategy

Given the demonstrated generalisation gap, a production deployment of
this model would need active monitoring rather than a one-off validation
at training time. A reasonable strategy:

- Confidence-based flagging: route low-confidence predictions (e.g.
  below 0.7, informed by the Mode B uncertainty pattern observed in
  Guava_Rotten) to human review rather than auto-accepting them.
- Confident-wrong detection is harder: Mode A (Pomegranate) shows the
  model can be both wrong and highly confident, which a simple confidence
  threshold would not catch. A more robust approach would periodically
  sample live predictions and have a human spot-check a subset,
  regardless of reported confidence, particularly for classes
  historically prone to confident shortcuts.
- Drift indicators: track the distribution of predicted classes over
  time; a sudden shift (e.g. a spike in Tomato predictions) could
  indicate the input distribution has changed (new camera, new lighting
  setup, new supplier) rather than that produce quality has genuinely
  changed.
- Periodic re-validation against a held-out external sample: the
  Freshness44 OOD test used here is a one-off snapshot. A deployed
  system should periodically re-run an equivalent check against newly
  collected real-world images to catch degradation early.

## 6. Legal, ethical and professional considerations

- Transparency of decision-making: the Grad-CAM evidence shows the
  model's stated confidence is not always a trustworthy signal of
  correctness (Mode A). Any deployment that surfaces a confidence score
  to end users (e.g. warehouse staff) risks creating false trust in
  wrong predictions for specific produce types.
- Consequences of errors: misclassifying rotten produce as healthy (a
  false negative on spoilage) has a direct food-safety and waste-cost
  implication for the Bristol Regional Food Network; misclassifying
  healthy produce as rotten (a false positive) causes unnecessary waste
  and economic loss. These two error types are not symmetric in
  consequence, but the model's standard accuracy/F1 metrics treat them
  as equally weighted - this is worth flagging as a limitation rather
  than treating overall accuracy as sufficient.
- Fitness for purpose given current evidence: the 20.7 percentage point
  OOD accuracy drop means the model should not be considered
  deployment-ready based on the in-distribution numbers alone, despite
  those numbers looking strong in isolation.

## 7. Deployment recommendation

Recommendation: adopt after further work, not now.

Justification:
- The model demonstrates strong in-distribution performance and a sound
  experimental comparison across three architectures, but the OOD test
  provides direct evidence that this performance does not reliably
  transfer to images from different sources.
- The Grad-CAM analysis identifies specific, named failure modes
  (confident shortcuts, genuine uncertainty, real visual similarity)
  rather than a vague "needs more data" recommendation - this gives a
  concrete basis for targeted further work rather than blanket retraining.
- Suggested next steps before production deployment: (1) targeted data
  collection for classes showing Mode A confident-shortcut behaviour
  (Pomegranate), since weighting alone did not fix the shortcut - new
  data captured under deliberately varied conditions would be needed,
  (2) a confidence-monitoring pipeline as described in section 5,
  deployed initially in a human-in-the-loop advisory capacity rather
  than fully automated decision-making, (3) re-running the OOD test
  after the weighted retrain to confirm whether improved small-class
  in-distribution performance also transfers to new image sources.