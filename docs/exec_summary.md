# Executive Summary
## AI-Powered Produce Quality Assessment
### Bristol Regional Food Network — Prepared for management review

---

## What this project set out to do

The Bristol Regional Food Network currently relies on manual inspection
to determine whether fruit and vegetables are fresh or spoiled before
they reach customers. This process is time-consuming, inconsistent
between inspectors, and difficult to scale as the network grows.

This project investigated whether an AI system trained on images of
produce could automate that quality assessment — classifying items as
healthy or rotten across 14 types of fruit and vegetables, quickly
enough to be useful in a real logistics setting.

---

## What was built

A computer vision system that accepts a photograph of produce and
returns a quality classification (healthy or rotten) with a confidence
score, typically within a fraction of a second per image. The system
also generates a visual explanation showing which parts of the image
drove the decision — so a warehouse operative or quality manager can
see at a glance whether the AI is looking at the right thing.

A working web interface was built for demonstration: upload a photo,
receive a classification and explanation immediately.

---

## How well does it work?

Tested against nearly 4,400 images it had not seen during training, the
system correctly classified 97.5% of produce — a strong result for a
14-category, 28-class problem.

However, a more challenging test was also run: the same system was
tested against produce images from five completely different photographic
sources — different cameras, different lighting, different backgrounds
— to simulate what would happen if the network's suppliers photographed
produce differently to the training images. Under those conditions,
accuracy dropped to around 77%.

**What this means in practice:** the system performs very well on images
that resemble its training data. If deployed in a setting where produce
photographs are taken in a consistent, controlled way — same camera,
same lighting setup, same distance — performance would likely stay close
to the 97.5% figure. If photographs vary significantly between suppliers
or locations, performance could be considerably lower.

---

## Challenges the problem presented

**Consistency of produce photography.** The training images came from a
single photographic source. Real-world deployment would involve images
from different phones, cameras, and lighting conditions — and the system
has not yet been trained to handle that variation reliably.

**Some produce types are harder than others.** Rotten bellpepper and
rotten tomato look visually very similar, and the system struggles to
reliably distinguish them. This is a genuine perceptual difficulty, not
just a data problem — even with more training data, some confusion
between these two classes would likely remain.

**Confident but wrong predictions.** For certain produce types
(pomegranate in particular), the system was found to make incorrect
classifications at very high confidence — meaning it appeared certain
when it was wrong. A human operator seeing a "99% confident" result
might not think to double-check it, which is a risk in a safety-relevant
application.

---

## Ethical and legal considerations

**Food safety asymmetry.** Calling rotten produce healthy (a missed
spoilage) and calling healthy produce rotten (unnecessary waste) are not
equally harmful. The system's current performance metrics treat both
errors as equally bad, which does not reflect the real-world cost
difference. Any deployment should decide which error is more serious and
configure the system accordingly.

**Transparency of decisions.** The visual explanation feature (showing
where the AI is looking) was specifically built to address a requirement
that automated food quality decisions be interpretable rather than
opaque. A decision to reject produce that cannot be explained to a
producer is a legal and reputational risk. The system provides that
explanation, but — as noted above — the explanation is not always
reliable for all produce types.

**Data.** All training images used are publicly available research
datasets. No personal data, supplier-specific data, or proprietary
imagery was used. There are no data protection concerns with the current
system.

---

## Recommendation

**Adopt after further development, not immediately.**

The system demonstrates genuine potential for automating produce quality
assessment at speed and scale. The 97.5% in-distribution accuracy, the
sub-second response time, and the explainability feature all represent
a solid foundation.

However, two issues need to be resolved before production deployment
would be responsible:

1. **Photography consistency.** The system needs to be either tested
   against — and if necessary retrained on — images from the network's
   actual suppliers and camera setups, rather than assumed to generalise
   from the existing training data. The 20-point accuracy gap between
   controlled and varied images is too large to ignore for a food safety
   application.

2. **Human oversight for high-stakes decisions.** Given the confident-
   but-wrong behaviour identified for certain produce types, the system
   should not make fully automated accept/reject decisions without human
   review in its first deployment phase. A recommended starting point
   would be using the AI as an advisory tool — flagging borderline cases
   for human inspection rather than replacing the inspector entirely.

With targeted data collection from real supplier photographs and a
human-in-the-loop deployment model, the system could realistically
support production use. The technical foundations are sound; the
remaining work is validation against the network's specific operating
conditions, not fundamental redesign.

---

*Technical details, full evaluation results, and methodology are
documented in the accompanying technical report and GitHub repository.*