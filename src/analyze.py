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
from .textproc import TfidfVectorizer, tokenize, ngrams
from .model import SoftmaxClassifier, OneVsRestClassifier
from .lexicons import (SYMPTOM_PHRASES, COOCCURRING_KEYWORDS, ROOT_CAUSE_GROUPS,
                       RISK_HIGH_SIGNALS, RISK_MODERATE_SIGNALS)

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


class Analyzer:
    def __init__(self, bundle_path=BUNDLE_PATH):
        with open(bundle_path) as f:
            b = json.load(f)
        self.dis_vec = TfidfVectorizer.from_dict(b["disease"]["vec"])
        self.dis_clf = SoftmaxClassifier.from_dict(b["disease"]["clf"])
        self.dis_labels = b["disease"]["labels"]
        self.rc_vec = TfidfVectorizer.from_dict(b["rootcause"]["vec"])
        self.rc_clf = SoftmaxClassifier.from_dict(b["rootcause"]["clf"])
        self.co_vec = TfidfVectorizer.from_dict(b["cooccurring"]["vec"])
        self.co_clf = OneVsRestClassifier.from_dict(b["cooccurring"]["clf"])

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

    def _symptoms(self, text):
        low = text.lower()
        found = [p for p in SYMPTOM_PHRASES if p in low]
        # dedupe near-duplicates, cap
        return found[:14]

    def _cooccurring(self, text):
        """Combine lexicon hits (high precision) with the model (recall)."""
        low = text.lower()
        lex = set()
        for cond, kws in COOCCURRING_KEYWORDS.items():
            if any(kw in low for kw in kws):
                lex.add(cond)
        # model
        P = self.co_vec.transform([text])
        probs = self.co_clf.predict_proba(P)[0]
        model_hits = {self.co_clf.labels_[i] for i, p in enumerate(probs)
                      if p >= self.co_clf.threshold}
        return sorted(lex | model_hits)

    def _root_cause(self, text):
        P = self.rc_vec.transform([text])
        probs = self.rc_clf.predict_proba(P)[0]
        order = np.argsort(probs)[::-1]
        top = [(self.rc_clf.classes_[i], float(probs[i])) for i in order[:3]]
        # lexicon-based contributing factors for transparency
        low = text.lower()
        factors = []
        for grp, kws in ROOT_CAUSE_GROUPS.items():
            c = sum(low.count(kw) for kw in kws)
            if c:
                factors.append((grp, c))
        factors.sort(key=lambda x: -x[1])
        return top, [f[0] for f in factors[:3]]

    def _risk(self, text, confidence, primary):
        low = text.lower()
        high = [s for s in RISK_HIGH_SIGNALS if s in low]
        mod = [s for s in RISK_MODERATE_SIGNALS if s in low]
        if high:
            return "High", high[:5]
        if mod or primary == "Other / Complex":
            return "Moderate", mod[:5]
        return "Low", []

    # ---- main -----------------------------------------------------------
    def analyze(self, text):
        if not text or len(text.strip()) < 80:
            return {"error": "Not enough readable text to analyze."}

        P = self.dis_vec.transform([text])
        probs = self.dis_clf.predict_proba(P)[0]
        order = np.argsort(probs)[::-1]
        primary = self.dis_clf.classes_[order[0]]
        confidence = float(probs[order[0]])
        ranked = [(self.dis_clf.classes_[i], float(probs[i])) for i in order[:3]]

        terms = self._top_evidence_terms(text, primary)
        evidence = self._evidence_sentences(text, terms + self._symptoms(text))
        symptoms = self._symptoms(text)
        cooc = self._cooccurring(text)
        rc_top, rc_factors = self._root_cause(text)
        risk, risk_signals = self._risk(text, confidence, primary)

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

        recommendation = self._recommendation(primary, conf_band, risk, needs_review)

        return {
            "diagnosis": primary,
            "diagnosis_ranked": ranked,
            "cooccurring": cooc,
            "symptoms": symptoms,
            "root_cause_top": rc_top,
            "root_cause_factors": rc_factors,
            "evidence": evidence,
            "evidence_terms": terms,
            "confidence": round(confidence, 4),
            "confidence_band": conf_band,
            "risk_level": risk,
            "risk_signals": risk_signals,
            "recommendation": recommendation,
            "needs_doctor_review": needs_review,
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
