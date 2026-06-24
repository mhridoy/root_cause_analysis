"""
Clinically-grounded root-cause modelling.

THE PROBLEM THIS FIXES
----------------------
The old root-cause label was `argmax(keyword_counts)`, and the model was trained
on TF-IDF of the same words -> it just relearned keyword counting. That is why
the output "felt like text extraction".

THE FIX
-------
Ground the root-cause target in two independent, clinically meaningful signals
and learn from a richer representation:

  1. Diagnosis-informed PRIOR. The primary diagnosis (reliably labelled from the
     report category) implies a clinically expected distribution over
     contributing-factor groups. ASD/ADHD -> Neurodevelopmental; Depression ->
     Mood; GAD/OCD -> Anxiety/Stress; Dyslexia -> Learning/Cognitive, etc.
     This knowledge does NOT come from surface keywords.

  2. Evidence DISTRIBUTION (not argmax). How strongly the narrative discusses
     each factor, with diminishing returns (sqrt) so one repeated word can't
     dominate, and normalised to a distribution.

The training label is the argmax of (prior x evidence) -- a posterior that
blends clinical knowledge with what the report actually describes. Because the
prior is independent of the bag-of-words, a model predicting this label must
capture diagnosis-level meaning, not just count words.
"""
import math
import re
from .lexicons import ROOT_CAUSE_GROUPS

ROOT_CAUSE_LIST = list(ROOT_CAUSE_GROUPS.keys())
_IDX = {g: i for i, g in enumerate(ROOT_CAUSE_LIST)}

# Diagnosis -> clinically expected contributing-factor weights (unnormalised).
# Curated from standard developmental/clinical associations. These are PRIORS:
# starting beliefs about likely contributing factors given the diagnosis, to be
# updated by the report's actual evidence.
DIAGNOSIS_PRIOR = {
    "ASD": {
        "Neurodevelopmental Differences": 6, "Sensory / Motor": 3,
        "Social / Environmental Pressure": 1, "Anxiety / Stress Reactivity": 1,
    },
    "ADHD": {
        "Neurodevelopmental Differences": 6, "Learning / Cognitive Processing": 2,
        "Sleep / Physiological Regulation": 1, "Social / Environmental Pressure": 1,
    },
    "Dyslexia": {
        "Learning / Cognitive Processing": 6, "Neurodevelopmental Differences": 2,
        "Social / Environmental Pressure": 1,
    },
    "Depression": {
        "Mood / Emotional Vulnerability": 6, "Social / Environmental Pressure": 2,
        "Sleep / Physiological Regulation": 2, "Anxiety / Stress Reactivity": 1,
    },
    "GAD": {
        "Anxiety / Stress Reactivity": 6, "Mood / Emotional Vulnerability": 2,
        "Social / Environmental Pressure": 2, "Sleep / Physiological Regulation": 1,
    },
    "OCD": {
        "Anxiety / Stress Reactivity": 6, "Neurodevelopmental Differences": 1,
        "Mood / Emotional Vulnerability": 1,
    },
    "Other / Complex": {  # flat-ish prior; rely on evidence
        g: 1 for g in ROOT_CAUSE_LIST
    },
}


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())


def evidence_distribution(text):
    """Return (probs list aligned to ROOT_CAUSE_LIST, raw_hits dict).
    sqrt-dampened, normalised distribution of factor evidence in the narrative."""
    hay = _norm(text)
    raw = {}
    damped = []
    for g in ROOT_CAUSE_LIST:
        hits = sum(hay.count(kw) for kw in ROOT_CAUSE_GROUPS[g])
        raw[g] = hits
        damped.append(math.sqrt(hits))
    total = sum(damped)
    if total == 0:
        probs = [1.0 / len(ROOT_CAUSE_LIST)] * len(ROOT_CAUSE_LIST)
    else:
        probs = [d / total for d in damped]
    return probs, raw


def prior_distribution(diagnosis):
    pri = DIAGNOSIS_PRIOR.get(diagnosis, DIAGNOSIS_PRIOR["Other / Complex"])
    vec = [pri.get(g, 0.3) for g in ROOT_CAUSE_LIST]  # small floor for unlisted
    s = sum(vec)
    return [v / s for v in vec]


def posterior(diagnosis, text, prior_weight=0.55):
    """Blend clinical prior with report evidence -> normalised posterior over
    contributing-factor groups. Returns list aligned to ROOT_CAUSE_LIST."""
    pri = prior_distribution(diagnosis)
    ev, _ = evidence_distribution(text)
    blended = [prior_weight * p + (1 - prior_weight) * e
               for p, e in zip(pri, ev)]
    s = sum(blended)
    return [b / s for b in blended]


def grounded_label(diagnosis, text):
    """The clinically-grounded ground-truth root-cause group (argmax posterior)."""
    post = posterior(diagnosis, text)
    return ROOT_CAUSE_LIST[max(range(len(post)), key=lambda i: post[i])]
