"""
Inference engine: turn one report's text into the structured, explainable result.

Output fields (exactly the requested format):
  Child Condition / Diagnosis
  Co-occurring Diseases or Disorders
  Key Symptoms Found
  Probable Root Cause / Contributing Factors
  Evidence from Report
  Confidence Score
  Risk Level
  Recommendation
  Needs Doctor Review (Yes/No)

Explainability: per-class linear weights x present TF-IDF features identify the
tokens that pushed the prediction, mapped back to sentences in the report.
"""
import os
import re
import json
import numpy as np
from .textproc import (TfidfVectorizer, tokenize, ngrams,
                       split_negation, phrase_present)
from .model import SoftmaxClassifier, OneVsRestClassifier
from .lexicons import (SYMPTOM_PHRASES, COOCCURRING_KEYWORDS, ROOT_CAUSE_GROUPS,
                       RISK_HIGH_SIGNALS, RISK_MODERATE_SIGNALS)
from .domain_gate import assess_domain
from .diagnosis_priors import blend_with_symptoms

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(HERE, "..", "models", "bundle.json")

DISCLAIMER = (
    "This is an AI-assisted analysis of the report text, NOT a confirmed medical "
    "diagnosis. It can make mistakes and does not replace evaluation by a qualified "
    "clinician. Always seek professional medical review, especially when confidence "
    "is low or the case is complex.")

# Confidence policy
HIGH_CONF = 0.70
LOW_CONF = 0.50
# If the winner beats the runner-up by less than this margin, the case is
# ambiguous between two conditions -> always flag for expert review.
AMBIGUITY_MARGIN = 0.10


class Analyzer:
    def __init__(self, bundle_path=BUNDLE_PATH):
        with open(bundle_path) as f:
            b = json.load(f)
        self.dis_vec = TfidfVectorizer.from_dict(b["disease"]["vec"])
        self.dis_clf = SoftmaxClassifier.from_dict(b["disease"]["clf"])
        self.dis_labels = b["disease"]["labels"]
        self.co_vec = TfidfVectorizer.from_dict(b["cooccurring"]["vec"])
        self.co_clf = OneVsRestClassifier.from_dict(b["cooccurring"]["clf"])
        try:
            from .rootcause import get_rootcause_engine
            self.rc_engine = get_rootcause_engine()
        except Exception:
            self.rc_engine = None
            self.rc_vec = TfidfVectorizer.from_dict(b["rootcause"]["vec"])
            self.rc_clf = SoftmaxClassifier.from_dict(b["rootcause"]["clf"])

    def _vocab_coverage(self, text):
        """Fraction of document tokens represented in the trained vocabulary."""
        toks = tokenize(text)
        if not toks:
            return 0.0
        known = sum(1 for token in toks if token in self.dis_vec.vocab_)
        return known / len(toks)

    # ---- explainability -------------------------------------------------
    def _top_evidence_terms(self, text, pred_label, k=8):
        """Tokens present in the doc with the largest positive contribution to
        the predicted class (TF-IDF value x class weight)."""
        vec = self.dis_vec
        X = vec.transform([text])[0]
        cls_i = self.dis_clf.classes_.index(pred_label)
        w = self.dis_clf.W[:-1, cls_i]  # drop bias
        contrib = X * w
        names = vec.feature_names_
        order = np.argsort(contrib)[::-1]
        terms = []
        for j in order:
            if contrib[j] <= 0:
                break
            terms.append(names[j].replace("_", " "))
            if len(terms) >= k:
                break
        return terms

    def _evidence_sentences(self, text, terms, max_sent=4):
        sents = re.split(r"(?<=[.!?])\s+|\n+", text)
        scored = []
        low_terms = [t.lower() for t in terms]
        for s in sents:
            sl = s.lower()
            hits = [t for t in low_terms if t in sl]
            if hits and 15 < len(s.strip()) < 320:
                scored.append((len(hits), s.strip()))
        scored.sort(key=lambda x: -x[0])
        seen, out = set(), []
        for _, s in scored:
            if s not in seen:
                out.append(s); seen.add(s)
            if len(out) >= max_sent:
                break
        return out

    # Map the primary diagnosis to its own co-occurring label so we don't list
    # the primary condition as its own "co-occurring" disorder.
    _PRIMARY_COOC = {"ADHD": "ADHD", "ASD": "ASD", "Depression": "Depression",
                     "GAD": "Anxiety", "OCD": "OCD", "Dyslexia": "Learning Disorder"}

    def _symptoms(self, text):
        """Only symptoms that are ASSERTED (not negated/ruled-out) in the text."""
        asserted, _ = split_negation(text)
        return [p for p in SYMPTOM_PHRASES if phrase_present(p, asserted)][:14]

    def _cooccurring(self, text, primary):
        """Negation-aware, TIERED co-occurring detection. Conditions that are
        only ruled out/denied are surfaced separately, never as positives. The
        noisy weak-label OvR model is no longer used."""
        asserted, negated = split_negation(text)
        skip = self._PRIMARY_COOC.get(primary)
        likely, possible, ruled_out = [], [], []
        for cond, kws in COOCCURRING_KEYWORDS.items():
            if cond == skip:
                continue
            a = sum(1 for kw in kws if phrase_present(kw, asserted))
            n = sum(1 for kw in kws if phrase_present(kw, negated))
            if a >= 2:
                likely.append(cond)
            elif a == 1:
                possible.append(cond)
            elif n > 0:
                ruled_out.append(cond)
        return {"likely": likely, "possible": possible, "ruled_out": ruled_out}

    def _root_cause(self, text, diagnosis):
        """Blend learned text evidence with a diagnosis-informed posterior."""
        if self.rc_engine is not None:
            detail = self.rc_engine.predict(text, diagnosis)
            top = [(group, probability)
                   for group, probability in detail["ranked"][:3]]
            return top, [group for group, _ in top], detail

        # Backward-compatible fallback for deployments without the new artifact.
        P = self.rc_vec.transform([text])
        probs = self.rc_clf.predict_proba(P)[0]
        order = np.argsort(probs)[::-1]
        top = [(self.rc_clf.classes_[i], float(probs[i])) for i in order[:3]]
        return top, [group for group, _ in top], None

    # Impairment language -> at least Moderate clinical concern (not acute risk).
    _IMPAIRMENT = ["interfere", "impair", "distress", "disrupt", "delayed",
                   "difficulty", "withdraw", "decline", "avoidance"]

    def _risk(self, text, primary):
        """Negation-aware risk. HIGH only when acute-danger signals (self-harm,
        suicide, abuse, neglect, severe) are ASSERTED — never when they are
        explicitly denied ('no suicidal thoughts...'). Returns
        (level, asserted_signals, denied_signals)."""
        asserted, negated = split_negation(text)
        high = [s for s in RISK_HIGH_SIGNALS if phrase_present(s, asserted)]
        denied = [s for s in RISK_HIGH_SIGNALS if phrase_present(s, negated)]
        mod = [s for s in RISK_MODERATE_SIGNALS if phrase_present(s, asserted)]
        impaired = any(phrase_present(w, asserted) for w in self._IMPAIRMENT)
        if high:
            return "High", high[:5], denied
        if mod or impaired or primary == "Other / Complex":
            return "Moderate", mod[:5], denied
        return "Low", [], denied

    def _not_a_report(self, gate, coverage):
        """Return a structured refusal instead of producing a diagnosis."""
        why = gate["reasons"][0] if gate["reasons"] else (
            "the document does not contain enough clinical-assessment content")
        return {
            "out_of_domain": True,
            "diagnosis": "Not a recognized clinical assessment report",
            "diagnosis_ranked": [],
            "diagnosis_probs": [],
            "decision_margin": 0.0,
            "cooccurring": [],
            "symptoms": [],
            "root_cause_top": [],
            "root_cause_factors": [],
            "root_cause_detail": None,
            "evidence": [],
            "evidence_terms": [],
            "confidence": 0.0,
            "confidence_band": "N/A",
            "risk_level": "N/A",
            "risk_signals": [],
            "relevance_score": gate["relevance_score"],
            "vocab_coverage": round(coverage, 3),
            "recommendation": (
                "This file does not look like a psychological or developmental "
                f"assessment report ({why}). No diagnosis was produced. "
                "Please upload an assessment/evaluation report. If this is a "
                "clinical report, verify that its text was extracted correctly."),
            "needs_doctor_review": False,
            "disclaimer": DISCLAIMER,
        }

    # ---- main -----------------------------------------------------------
    def analyze(self, text):
        if not text or len(text.strip()) < 80:
            return {"error": "Not enough readable text to analyze."}

        P = self.dis_vec.transform([text])
        ml_probs = self.dis_clf.predict_proba(P)[0]

        # Blend with the negation-aware symptom-signature prior, but ONLY when the
        # learned model is unsure (paraphrased / symptom-only reports). On the
        # held-out test set this leaves accuracy unchanged; it rescues realistic
        # reports that never name the disorder. Done BEFORE the domain gate so a
        # genuine clinical report the model is unsure about is judged on its
        # rescued confidence, not the raw (low) model confidence.
        probs, prior_info = blend_with_symptoms(ml_probs, self.dis_clf.classes_, text)
        blended_conf = float(probs.max())

        symptoms = self._symptoms(text)
        coverage = self._vocab_coverage(text)
        gate = assess_domain(text, model_confidence=blended_conf,
                             symptom_count=len(symptoms),
                             vocab_coverage=coverage)
        if not gate["is_clinical"]:
            return self._not_a_report(gate, coverage)

        order = np.argsort(probs)[::-1]
        primary = self.dis_clf.classes_[order[0]]
        confidence = float(probs[order[0]])
        runner_up = float(probs[order[1]]) if len(order) > 1 else 0.0
        margin = confidence - runner_up
        ranked = [(self.dis_clf.classes_[i], float(probs[i])) for i in order[:3]]
        all_probs = [(self.dis_clf.classes_[i], round(float(probs[i]), 4))
                     for i in order]

        terms = self._top_evidence_terms(text, primary)
        evidence = self._evidence_sentences(text, terms + symptoms)
        cooc = self._cooccurring(text, primary)
        rc_top, rc_factors, rc_detail = self._root_cause(text, primary)
        risk, risk_signals, risk_denied = self._risk(text, primary)

        # confidence -> review policy
        if confidence < LOW_CONF:
            needs_review = True
            conf_band = "Low"
        elif confidence < HIGH_CONF:
            needs_review = True
            conf_band = "Moderate"
        else:
            conf_band = "High"
            needs_review = risk == "High" or primary == "Other / Complex"
        # ambiguous between two conditions -> always escalate
        if margin < AMBIGUITY_MARGIN:
            needs_review = True
        if rc_detail and rc_detail.get("abstain"):
            needs_review = True

        recommendation = self._recommendation(primary, conf_band, risk, needs_review)
        if rc_detail and rc_detail.get("abstain"):
            recommendation += (
                " Contributing-factor analysis was inconclusive because its "
                "learned and clinical views disagreed or had a narrow margin.")
        if prior_info.get("applied"):
            recommendation += (
                " The learned model was uncertain, so this leans on the "
                "clinical symptom pattern in the report (the disorder may not be "
                "named explicitly); confirm against full diagnostic criteria.")

        return {
            "diagnosis": primary,
            "diagnosis_ranked": ranked,
            "diagnosis_probs": all_probs,
            "decision_margin": round(margin, 4),
            "cooccurring": cooc["likely"] + cooc["possible"],
            "cooccurring_detail": cooc,
            "symptoms": symptoms,
            "root_cause_top": rc_top,
            "root_cause_factors": rc_factors,
            "root_cause_detail": rc_detail,
            "evidence": evidence,
            "evidence_terms": terms,
            "confidence": round(confidence, 4),
            "confidence_band": conf_band,
            "risk_level": risk,
            "risk_signals": risk_signals,
            "risk_denied": risk_denied,
            "recommendation": recommendation,
            "needs_doctor_review": needs_review,
            "symptom_prior": prior_info,
            "disclaimer": DISCLAIMER,
        }

    def _recommendation(self, primary, conf_band, risk, needs_review):
        parts = []
        if risk == "High":
            parts.append("Report contains language suggesting elevated risk "
                         "(e.g. safety, self-harm, severe impairment). Prioritise "
                         "urgent review by a qualified clinician.")
        if conf_band == "Low":
            parts.append("Model confidence is low; the report may be atypical, "
                         "incomplete, or describe a condition outside the trained "
                         "set. Treat the prediction as a weak hint only.")
        elif conf_band == "Moderate":
            parts.append("Model confidence is moderate; corroborate with the full "
                         "clinical record before drawing conclusions.")
        else:
            parts.append(f"The text is consistent with patterns the model "
                         f"associates with {primary}. Confirm against standardized "
                         f"diagnostic instruments and clinical judgement.")
        if needs_review:
            parts.append("Flagged: NEEDS EXPERT REVIEW.")
        return " ".join(parts)


_singleton = None


def get_analyzer():
    global _singleton
    if _singleton is None:
        _singleton = Analyzer()
    return _singleton


if __name__ == "__main__":
    import sys
    from .extract import extract_text
    text, meta = extract_text(sys.argv[1])
    res = get_analyzer().analyze(text)
    print(json.dumps(res, indent=2))
