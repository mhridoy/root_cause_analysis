"""
Symptom-signature clinical prior for the PRIMARY DIAGNOSIS.

Why this exists
---------------
The learned TF-IDF model was trained on a templated corpus where reports usually
NAME the disorder ("Primary Diagnosis: OCD ..."). On a realistic, paraphrased,
symptom-only report that never says the word, the model flounders (a textbook
OCD case scored OCD last). Clinicians don't rely on the label being written —
they recognise the *symptom pattern*. This module encodes characteristic
symptom clusters (DSM-5-style) per disorder and scores them, NEGATION-AWARE, so
ruled-out symptoms ("no history of restricted interests") do not count.

It is blended with the learned model in analyze.py. It is decision-support
vocabulary, NOT a diagnostic instrument.
"""
import numpy as np
from .textproc import marked_tokens, split_negation, phrase_present, term_present

# Trauma / PTSD recognition from a NARRATIVE (no explicit diagnosis line).
# A trauma-driven presentation often mimics ADHD (restlessness, distractibility),
# so the learned model picks ADHD. Requiring an explicit trauma EXPOSURE *event*
# PLUS several trauma-SPECIFIC symptoms separates it from ADHD — and from OCD,
# whose "intrusive thoughts" overlap (so that term is deliberately NOT counted).
TRAUMA_EXPOSURE = [
    "witnessed", "witnessing", "domestic violence", "physical abuse",
    "sexual abuse", "emotional abuse", "assault", "car accident",
    "road traffic accident", "natural disaster", "traumatic event",
    "traumatic incident", "following the event", "following the incident",
    "after the event", "after the incident", "since the event",
    "since the incident", "exposure to violence", "exposure to trauma",
    "house fire", "bereavement", "sudden death",
]
# Trauma-SPECIFIC symptoms (excludes "intrusive thoughts", shared with OCD).
TRAUMA_SYMPTOMS = [
    "nightmare", "flashback", "hypervigilan", "exaggerated startle",
    "easily startled", "startle response", "avoid reminders", "avoidance of",
    "re-experienc", "reexperienc", "dissociat", "on guard", "threat scanning",
    "scans the room", "scans for threat", "emotional numbing", "hyperarousal",
    "trauma reminders",
]


def trauma_signal(text):
    """Return (has_exposure_event, n_trauma_specific_symptoms) from the ASSERTED
    (non-negated) text, so a denied/absent trauma history does not count.
    Word-boundary matching avoids 'accidental' -> 'accident'."""
    asserted, _ = split_negation(text)
    exposure = any(term_present(k, asserted) for k in TRAUMA_EXPOSURE)
    n = sum(1 for k in TRAUMA_SYMPTOMS if term_present(k, asserted))
    return exposure, n

# Characteristic, fairly SPECIFIC signature terms per disorder. Deliberately
# avoids generic words ("attention", "anxiety", "social communication") that
# appear incidentally; uses discriminating phrases instead. Substring-matched
# against asserted (non-negated) text, so morphology is tolerated
# (e.g. "ritual" matches "ritualized").
DISEASE_SIGNATURES = {
    "OCD": [
        "obsession", "obsessive", "compulsion", "compulsive", "intrusive thought",
        "ritual", "checking", "washing", "contamination", "germ", "reassurance",
        "perfectionism", "unwanted thought", "repeatedly check", "symmetry",
        "ordering", "hoarding", "unable to stop", "must be perfect",
    ],
    "ADHD": [
        "inattention", "inattentive", "hyperactivity", "hyperactive",
        "impulsivity", "impulsive", "distractib", "fidget", "forgetful",
        "disorganiz", "careless mistake", "difficulty sustaining attention",
        "interrupts", "blurts", "difficulty waiting", "squirm", "off task",
        "restlessness motor",
    ],
    "ASD": [
        "restricted interest", "repetitive motor", "stereotyp", "echolalia",
        "insistence on sameness", "poor eye contact", "limited eye contact",
        "social reciprocity", "social communication deficit", "sensory sensitivit",
        "rigid routine", "literal interpretation", "difficulty with transitions",
        "lack of social",
    ],
    "GAD": [
        "excessive worry", "generalized anxiety", "generalised anxiety",
        "worry about", "uncontrollable worry", "apprehension", "free-floating",
        "worries about", "constant worry", "chronic worry", "muscle tension",
    ],
    "Depression": [
        "depressed mood", "anhedonia", "loss of interest", "hopeless",
        "worthless", "persistent sadness", "low mood", "tearful", "suicidal",
        "guilt", "psychomotor retardation", "fatigue and", "crying spells",
    ],
    "Dyslexia": [
        "reading difficult", "decoding", "phonolog", "spelling difficult",
        "reading fluency", "letter reversal", "word recognition",
        "reading comprehension", "sounding out", "slow reading", "misreads",
    ],
}

DISEASES = list(DISEASE_SIGNATURES.keys())


def asserted_text(text):
    """Negation-cleaned, space-joined token string (ruled-out tokens removed)."""
    toks = marked_tokens(text, use_negation=True)
    return " " + " ".join(t for t in toks if not t.startswith("neg_")) + " "


def symptom_hits(text):
    """Return {disease: [matched signature terms]} on asserted text."""
    at = asserted_text(text)
    hits = {}
    for dis, terms in DISEASE_SIGNATURES.items():
        hits[dis] = [t for t in terms if t in at]
    return hits


def symptom_scores(text):
    """Return {disease: count_of_distinct_signature_terms_present}."""
    return {d: len(v) for d, v in symptom_hits(text).items()}


def symptom_distribution(text):
    """Normalised distribution over the 6 named diseases (None if no signal)."""
    scores = symptom_scores(text)
    total = sum(scores.values())
    if total == 0:
        return None, scores
    return {d: scores[d] / total for d in DISEASES}, scores


def blend_weight(scores):
    """Adaptive weight for the symptom prior: stronger when the signal is strong
    AND concentrated (clear leader), near-zero when weak/ambiguous. Capped so the
    learned model is never fully overridden."""
    vals = sorted(scores.values(), reverse=True)
    if not vals or vals[0] == 0:
        return 0.0
    top = vals[0]
    second = vals[1] if len(vals) > 1 else 0
    margin = top - second
    # need at least a few hits and a clear lead to intervene strongly
    w = 0.14 * top + 0.10 * margin
    return float(max(0.0, min(0.85, w)))


# Confidence at/above which the learned model is trusted outright (no symptom
# blending). Tuned so held-out test accuracy is unchanged while paraphrased,
# symptom-only reports the model is unsure about get corrected. Raise it for
# stronger symptom influence (slightly lower accuracy); lower it for less.
DEFAULT_GATE = 0.45


def blend_with_symptoms(ml_probs, classes, text, gate=DEFAULT_GATE):
    """Blend learned-model probabilities with the symptom-signature prior, but
    only when the model is UNSURE (top prob < gate). Returns (blended_probs,
    info) where info documents whether/why the prior was applied (explainable).
    """
    ml = np.asarray(ml_probs, dtype=float).copy()
    info = {"applied": False, "weight": 0.0, "matched": {},
            "top_symptom_disease": None}
    dist, scores = symptom_distribution(text)
    if dist is None:
        return ml, info
    ml_top = float(ml.max())
    if ml_top >= gate:
        return ml, info
    uncertainty = min(1.0, (gate - ml_top) / (gate * 0.6))
    w = blend_weight(scores) * uncertainty
    if w <= 0:
        return ml, info
    ci = {c: i for i, c in enumerate(classes)}
    sym = np.zeros(len(classes))
    for d in DISEASES:
        if d in ci:
            sym[ci[d]] = dist[d]
    out = (1 - w) * ml + w * sym
    out = out / out.sum()
    hits = symptom_hits(text)
    info = {"applied": True, "weight": round(float(w), 3),
            "matched": {d: h for d, h in hits.items() if h},
            "top_symptom_disease": max(DISEASES, key=lambda d: scores[d])}
    return out, info
