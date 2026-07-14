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


# Anxiety subtype recognition from a NARRATIVE (no scores, no conclusion). Social
# Anxiety has no model class and merges into GAD at low probability, so a
# narrative description must be recognised explicitly.
SOCIAL_ANX_SIGNALS = [
    "social anxiety", "performance anxiety", "fear of being evaluated",
    "fear of evaluation", "fear of negative evaluation", "embarrass", "humiliat",
    "scrutiny", "self-conscious", "fear of speaking", "afraid to speak",
    "avoids speaking", "oral presentation", "presentation", "fear of public",
    "public speaking", "social evaluative", "fear of judgement",
    "fear of judgment", "blush", "social situation", "evaluative anxiety",
    "speaking in front", "in front of", "others will laugh", "peers laughed",
    "raising a hand", "raise a hand", "avoid raising", "speak publicly",
]
GAD_SIGNALS = [
    "excessive worry", "generali", "uncontrollable worry", "constant worry",
    "worry about many", "worries about everything", "chronic worry",
    "worry for days", "worries for days", "free-floating", "worry about everything",
]
GENERIC_ANX = [
    "anxiety", "anxious", "worry", "worried", "nervous", "fearful",
    "school avoidance", "reluctance to attend", "avoids", "reassurance", "panic",
    "dread", "tense", "stomachache", "scared", "freeze", "freezes",
]


def anxiety_narrative_signal(text):
    """Return (label, strength) for a clear anxiety narrative, else (None, 0).
    Distinguishes Social Anxiety from generalized worry."""
    asserted, _ = split_negation(text)
    social = sum(1 for k in SOCIAL_ANX_SIGNALS if phrase_present(k, asserted))
    gad = sum(1 for k in GAD_SIGNALS if phrase_present(k, asserted))
    generic = sum(1 for k in GENERIC_ANX if phrase_present(k, asserted))
    if social >= 2:
        return "Social Anxiety", social + 1
    if gad >= 2:
        return "GAD", gad + 1
    if social >= 1 and generic >= 3:
        return "Social Anxiety", social + 1
    if gad >= 1 and generic >= 4:
        return "GAD", gad + 1
    return None, 0


# Depression and ADHD share concentration, motivation, restlessness, and school
# performance problems.  These signatures deliberately separate a CURRENT mood
# episode from a DEVELOPMENTAL attention pattern so a depressive episode is not
# demoted merely because the report also describes longstanding ADHD.
DEPRESSION_CORE = [
    "depressed mood", "feeling depressed", "feeling down", "persistent sadness",
    "low mood", "anhedonia", "loss of interest", "little interest", "hopeless",
    "worthless", "low self worth", "feeling empty", "no longer enjoys",
]
DEPRESSION_ASSOCIATED = [
    "low energy", "reduced energy", "fatigue", "sleep disturbance", "insomnia",
    "hypersomnia", "early waking", "appetite", "tearful", "crying", "withdrawn",
    "social withdrawal", "social isolation", "irritability", "guilt",
    "psychomotor", "suicidal", "self harm", "poor motivation",
    "reduced motivation", "flat affect",
]
DEPRESSION_IMPAIRMENT = [
    "academic decline", "declining grades", "declined significantly",
    "impaired functioning", "functional impairment", "clinically significant",
    "reduced school participation", "school refusal", "difficulty functioning",
    "stopped participating", "marked impairment", "marked withdrawal",
]
DEPRESSION_CURRENT = [
    "past two weeks", "two week period", "for weeks", "for months",
    "several months", "nearly every day", "most days", "current episode",
    "current symptoms", "current presentation", "presenting concern",
    "change from previous functioning", "recent decline", "over the past year",
]

ADHD_DEVELOPMENTAL_CORE = [
    "inattention", "inattentive", "distractib", "disorganiz", "forgetful",
    "careless mistake", "difficulty sustaining attention", "hyperactivity",
    "hyperactive", "impulsivity", "impulsive", "fidget", "interrupts",
    "difficulty waiting", "off task", "incomplete assignment",
]
ADHD_CHRONICITY = [
    "longstanding", "long standing", "since early childhood", "since childhood",
    "since elementary school", "before age 12", "before the age of 12",
    "onset before", "from an early age", "over several years", "for years",
    "persistent pattern", "predated the mood", "prior to the mood",
    "before the depressive", "before depression",
]
ADHD_CROSS_SETTING = [
    "across home and school", "both home and school", "home and at school",
    "school and home", "multiple settings", "more than one setting",
    "two settings", "across settings", "parent and teacher", "parents and teachers",
    "teacher and parent", "teachers and parents",
]


def _matched_terms(terms, asserted):
    return [term for term in terms if term_present(term, asserted)]


def depression_narrative_signal(text):
    """Describe support for a current depressive episode in asserted text.

    A strong signal requires multiple core mood symptoms plus associated symptoms
    and either current-duration or functional-impact evidence. Concentration
    problems alone are intentionally excluded because they overlap heavily with
    ADHD, anxiety, sleep problems, and learning difficulties.
    """
    asserted, _ = split_negation(text)
    core = _matched_terms(DEPRESSION_CORE, asserted)
    associated = _matched_terms(DEPRESSION_ASSOCIATED, asserted)
    impairment = _matched_terms(DEPRESSION_IMPAIRMENT, asserted)
    current = _matched_terms(DEPRESSION_CURRENT, asserted)
    strength = (2 * len(core) + min(len(associated), 6)
                + 2 * min(len(impairment), 3) + min(len(current), 2))
    strong = (len(core) >= 2 and len(associated) >= 2
              and bool(impairment or current))
    return {
        "strong": strong,
        "strength": strength,
        "core": core,
        "associated": associated,
        "impairment": impairment,
        "current": current,
    }


def adhd_developmental_signal(text):
    """Describe whether attention symptoms form a developmental ADHD pattern.

    Mood-related concentration difficulty is not enough. The signal requires a
    cluster of characteristic symptoms plus chronicity or cross-setting evidence.
    Standardized attention scores can add support in the caller, but cannot turn
    an isolated concentration complaint into ADHD by themselves.
    """
    asserted, _ = split_negation(text)
    core = _matched_terms(ADHD_DEVELOPMENTAL_CORE, asserted)
    chronicity = _matched_terms(ADHD_CHRONICITY, asserted)
    cross_setting = _matched_terms(ADHD_CROSS_SETTING, asserted)
    developmental = len(core) >= 3 and bool(chronicity or cross_setting)
    return {
        "supported": developmental,
        "strength": len(core) + 2 * bool(chronicity) + 2 * bool(cross_setting),
        "core": core,
        "chronicity": chronicity,
        "cross_setting": cross_setting,
    }

# Characteristic, fairly SPECIFIC signature terms per disorder. Deliberately
# avoids generic words ("attention", "anxiety", "social communication") that
# appear incidentally; uses discriminating phrases instead. Substring-matched
# against asserted (non-negated) text, so morphology is tolerated
# (e.g. "ritual" matches "ritualized").
DISEASE_SIGNATURES = {
    "OCD": [
        "obsession", "obsessive", "compulsion", "compulsive", "intrusive thought",
        "ritual", "checking", "washing", "contamination", "germ",
        "unwanted thought", "repeatedly check", "symmetry",
        "ordering", "hoarding", "unable to stop", "must be perfect",
        # NB: 'reassurance' and 'perfectionism' removed — both are common in
        # general/social anxiety and were firing OCD on anxiety reports.
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
