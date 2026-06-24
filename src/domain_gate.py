"""
Domain gate: decide whether an uploaded document is actually a clinical /
developmental assessment report BEFORE the classifier is allowed to label it.

Why this exists: a TF-IDF + softmax classifier is mathematically forced to
output one of its trained classes for ANY text, including an invoice or a
recipe. Without a gate the app would present a confident-looking diagnosis for
non-clinical input. This module rejects out-of-domain documents up front and
explains why.

The decision combines independent signals so it is hard to fool:
  * anchor_hits      -- distinct clinical-assessment vocabulary present
  * symptom_hits     -- clinical symptom phrases present (from lexicons)
  * model_confidence -- top softmax probability
  * vocab_coverage   -- fraction of document terms the model has ever seen
"""
import re
from .lexicons import SYMPTOM_PHRASES

# Vocabulary that characterises a psychological / developmental assessment
# report (not any single diagnosis). Invoices, recipes, resumes etc. contain
# almost none of these.
CLINICAL_ANCHORS = [
    "assessment", "evaluation", "diagnosis", "diagnostic", "referral",
    "referred", "clinical", "clinician", "psychologist", "psychological",
    "developmental", "milestone", "behaviou", "behavior", "cognitive",
    "symptom", "presenting", "presentation", "impairment", "functioning",
    "history", "observation", "intervention", "recommendation", "therapy",
    "therapist", "treatment", "patient", "child", "adolescent", "parent",
    "school", "teacher", "social", "emotional", "attention", "anxiety",
    "mood", "speech", "language", "sensory", "motor", "dsm", "icd",
    "screening", "scale", "subtest", "standardized", "standardised",
    "comprehension", "regulation", "neurodevelopmental", "examiner",
]


def _distinct_hits(text_low, terms):
    return sum(1 for t in terms if t in text_low)


def assess_domain(text, model_confidence=None, symptom_count=None,
                  vocab_coverage=None):
    """Return a dict describing whether `text` is an in-domain clinical report.

    Thresholds are calibrated so that essentially all genuine assessment reports
    pass while non-clinical documents (invoices, recipes, letters, resumes) are
    rejected. See calibrate_gate.py for the empirical basis.
    """
    low = " " + re.sub(r"\s+", " ", text.lower()) + " "
    anchor_hits = _distinct_hits(low, CLINICAL_ANCHORS)
    if symptom_count is None:
        symptom_count = _distinct_hits(low, [p.lower() for p in SYMPTOM_PHRASES])

    reasons = []
    # Hard rejects
    if len(text.split()) < 40:
        reasons.append("document too short to be an assessment report")
    if anchor_hits < 4:
        reasons.append(f"very few clinical terms present ({anchor_hits})")

    # Graded decision
    strong = anchor_hits >= 9
    moderate = anchor_hits >= 6 and symptom_count >= 1
    weak_ok = anchor_hits >= 4 and symptom_count >= 2

    is_clinical = (strong or moderate or weak_ok) and len(text.split()) >= 40

    # Confidence/coverage can only DOWNGRADE a borderline pass, never rescue a
    # hard fail. If the model is also unsure AND the doc is borderline, reject.
    borderline = is_clinical and not strong and anchor_hits < 7
    if borderline and model_confidence is not None and model_confidence < 0.30:
        is_clinical = False
        reasons.append("borderline clinical content with very low model confidence")
    if vocab_coverage is not None and vocab_coverage < 0.05:
        is_clinical = False
        reasons.append(f"almost none of the text matches the trained medical "
                       f"vocabulary (coverage {vocab_coverage:.0%})")

    # A relevance score in [0,1] for display.
    score = min(1.0, anchor_hits / 12.0) * 0.6 + min(1.0, symptom_count / 4.0) * 0.4
    return {
        "is_clinical": bool(is_clinical),
        "relevance_score": round(score, 3),
        "anchor_hits": anchor_hits,
        "symptom_hits": symptom_count,
        "reasons": reasons,
    }
