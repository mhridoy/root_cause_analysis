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
                       split_negation, phrase_present, term_present)
from .model import SoftmaxClassifier, OneVsRestClassifier
from .lexicons import (SYMPTOM_PHRASES, COOCCURRING_KEYWORDS, ROOT_CAUSE_GROUPS,
                       RISK_HIGH_SIGNALS, RISK_MODERATE_SIGNALS)
from .domain_gate import assess_domain
from .diagnosis_priors import blend_with_symptoms, trauma_signal
from .diagnosis_extract import extract as extract_diagnosis
from .scores import (parse as parse_scores, domain as score_domain,
                     indicated_conditions)

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

    # Map the primary diagnosis (modelled OR extracted) to its own co-occurring
    # label so we don't list the primary condition as its own "co-occurring".
    _PRIMARY_COOC = {"ADHD": "ADHD", "ASD": "ASD", "Depression": "Depression",
                     "GAD": "Anxiety", "OCD": "OCD", "Dyslexia": "Learning Disorder",
                     "PTSD / Trauma": "PTSD / Trauma",
                     "Intellectual Disability": "Intellectual Disability",
                     "Speech / Language Disorder": "Speech / Language",
                     "DCD / Dyspraxia": "Motor / Coordination",
                     "Tourette / Tics": "Tics / Tourette",
                     "Panic Disorder": "Anxiety", "Separation Anxiety": "Anxiety",
                     "Social Anxiety": "Anxiety",
                     "Social Communication Disorder": "Speech / Language",
                     "Selective Mutism": "Anxiety", "Dyscalculia": "Learning Disorder",
                     "Borderline Intellectual Functioning": "Intellectual Disability",
                     "SLD - Written Expression": "Learning Disorder",
                     "Reactive Attachment Disorder": "PTSD / Trauma"}

    def _symptoms(self, text):
        """Only symptoms that are ASSERTED (not negated/ruled-out) in the text."""
        asserted, _ = split_negation(text)
        return [p for p in SYMPTOM_PHRASES if phrase_present(p, asserted)][:14]

    def _cooccurring(self, text, primary):
        """Negation- AND rule-out-aware, TIERED co-occurring detection. A
        condition mentioned only in a rule-out / differential sentence ("X was
        considered but not met", "no evidence of X") or in a negated span is
        surfaced as ruled-out, never as a positive co-occurring condition. The
        noisy weak-label OvR model is no longer used."""
        from .diagnosis_extract import RULEOUT, _sentences
        # asserted text built ONLY from sentences that are not rule-outs
        kept = " ".join(s for s in _sentences(text) if not RULEOUT.search(s.lower()))
        asserted, _ = split_negation(kept)
        # everything ruled out: negated spans (full text) + rule-out sentences
        _, neg_full = split_negation(text)
        ruleout_txt = neg_full + " " + " ".join(
            re.sub(r"[-_]", " ", s.lower()) for s in _sentences(text)
            if RULEOUT.search(s.lower()))
        skip = self._PRIMARY_COOC.get(primary)
        likely, possible, ruled_out = [], [], []
        for cond, kws in COOCCURRING_KEYWORDS.items():
            if cond == skip:
                continue
            # word-start matching so "tics" doesn't match "mathematics", etc.
            a = sum(1 for kw in kws if term_present(kw, asserted))
            ro = sum(1 for kw in kws if term_present(kw, ruleout_txt))
            if a >= 2:
                likely.append(cond)
            elif a == 1:
                possible.append(cond)
            elif ro > 0:
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
            "primary_label": "Not a recognized clinical assessment report",
            "diagnosis_source": "out_of_domain",
            "diagnosis_statement": "",
            "explanation": "Input did not look like a clinical assessment report.",
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

    def _not_in_scope(self, reason):
        """Refusal for a clinical report that is the WRONG population/domain
        (e.g. an adult TBI evaluation)."""
        return {
            "out_of_domain": True,
            "diagnosis": "Out of scope (not a paediatric neurodevelopmental report)",
            "primary_label": "Out of scope (not a paediatric neurodevelopmental report)",
            "diagnosis_source": "out_of_scope",
            "diagnosis_statement": "",
            "explanation": reason,
            "diagnosis_ranked": [], "diagnosis_probs": [], "decision_margin": 0.0,
            "cooccurring": [], "cooccurring_detail": {"likely": [], "possible": [],
                                                      "ruled_out": []},
            "symptoms": [], "root_cause_top": [], "root_cause_factors": [],
            "root_cause_detail": None, "evidence": [], "evidence_terms": [],
            "confidence": 0.0, "confidence_band": "N/A", "risk_level": "N/A",
            "risk_signals": [], "risk_denied": [],
            "recommendation": reason + " Please use an assessment tool appropriate "
                              "to the patient's age and presentation.",
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

        # Read any explicit stated diagnosis up front. A report that formally
        # states a diagnosis (or explicitly states "no diagnosis") is, by
        # definition, a clinical report and must bypass the out-of-domain gate.
        ext = extract_diagnosis(text)
        trauma_exposure, trauma_n = trauma_signal(text)
        strong_trauma = trauma_exposure and trauma_n >= 2
        sc = parse_scores(text)
        dom, dom_reason = score_domain(text)

        # Wrong-POPULATION rejection: an adult / acquired-injury (TBI, dementia)
        # neuropsych evaluation is out of scope for a paediatric neurodevelopmental
        # classifier — refuse it explicitly rather than emit a child diagnosis.
        if dom == "adult":
            return self._not_in_scope(
                "This looks like an ADULT / acquired-injury neuropsychological "
                "evaluation (" + dom_reason + "). This tool only analyzes "
                "paediatric neurodevelopmental assessments, so no diagnosis was "
                "produced.")

        # A paediatric report carrying standardized test scores is, by definition,
        # a clinical assessment -> bypass the format-only out-of-domain gate.
        strong_clinical = (bool(ext["stated"]) or ext["no_diagnosis"]
                           or strong_trauma or (sc["has_tests"] and dom != "adult"))

        symptoms = self._symptoms(text)
        coverage = self._vocab_coverage(text)
        gate = assess_domain(text, model_confidence=blended_conf,
                             symptom_count=len(symptoms),
                             vocab_coverage=coverage)
        if not gate["is_clinical"] and not strong_clinical:
            return self._not_a_report(gate, coverage)

        order = np.argsort(probs)[::-1]
        all_probs = [(self.dis_clf.classes_[i], round(float(probs[i]), 4))
                     for i in order]

        # ---- DECIDE THE PRIMARY -------------------------------------------
        # Priority: (1) an explicitly STATED diagnosis in the report (most
        # reliable, and covers conditions outside the 6 trained classes);
        # (2) an explicit "no diagnosis"; otherwise (3) the learned model +
        # symptom-signature prior.  (ext computed before the gate, above.)
        dx_source = "model"
        dx_evidence = ""
        conflict_note = ""
        # ML's own top pick among real conditions (ignore the Other/Complex bucket)
        ml_real = [(self.dis_clf.classes_[i], float(probs[i])) for i in order
                   if self.dis_clf.classes_[i] != "Other / Complex"]
        ml_top_label = ml_real[0][0] if ml_real else self.dis_clf.classes_[order[0]]
        ml_top_prob = ml_real[0][1] if ml_real else float(probs[order[0]])
        if ext["stated"] and ext["confidence"] >= 0.70:
            primary = ext["stated"]
            confidence = ext["confidence"]
            if not ext["modelled"]:
                confidence = min(confidence, 0.75)
            dx_source = "stated"
            dx_evidence = ext["evidence"]
            # CONFLICT GUARD (two independent checks): do not silently amplify a
            # possibly-wrong stated conclusion.
            #  (a) the learned MODEL confidently indicates a different condition;
            #  (b) the standardized SCORES indicate a different condition than the
            #      stated label and do not support the stated one (catches the case
            #      where the model ALSO agrees with a wrong label but the anxiety/
            #      depression scores contradict both).
            score_conds = indicated_conditions(sc)
            conflict_with = None
            if (ext["modelled"] and ml_top_label != primary
                    and ml_top_label not in (ext.get("stated_secondary"),)
                    and ml_top_prob >= 0.40):
                conflict_with = f"{ml_top_label} ({int(round(ml_top_prob*100))}%)"
            elif (score_conds and primary not in score_conds
                  and ext.get("stated_secondary") not in score_conds):
                conflict_with = " / ".join(sorted(score_conds)) + " (standardized scores)"
            if conflict_with:
                confidence = min(confidence, 0.55)
                conflict_note = (
                    f" ⚠ The report states {primary}, but the report's own "
                    f"measures more strongly indicate {conflict_with}. This "
                    f"discrepancy must be resolved by a clinician before relying "
                    f"on either label.")
            others = [(self.dis_clf.classes_[i], float(probs[i])) for i in order
                      if self.dis_clf.classes_[i] != primary]
            ranked = [(primary, confidence)] + others[:2]
            margin = confidence - (others[0][1] if others else 0.0)
        elif ext.get("uncertain"):
            # the report itself says no definitive diagnosis can be made
            # ("neither X nor Y can be diagnosed", "further assessment required")
            primary = "Inconclusive — further assessment needed"
            confidence = 0.40
            dx_source = "uncertain"
            ranked = [(primary, confidence)]
            margin = confidence
        elif (ext["no_diagnosis"] and ml_top_prob < 0.45
              and not (sc["iq"] is not None and sc["iq"] < 80)
              and not sc["reading_low"]):
            # genuine "no diagnosis" ONLY when (a) the learned model is not itself
            # strongly indicating a condition (the missing-conclusion trap: a full
            # ADHD report with ML 58% must not be cleared just because a comorbidity
            # was ruled out), and (b) the scores don't show impairment.
            primary = "No diagnosis indicated"
            confidence = 0.80
            dx_source = "no_diagnosis"
            ranked = [(primary, confidence)]
            margin = confidence
        elif sc["iq"] is not None and sc["iq"] < 70:
            primary = "Intellectual Disability"
            confidence = 0.70
            dx_source = "score_pattern"
            dx_evidence = f"Full-Scale IQ {sc['iq']} (below the diagnostic cutoff)"
            ranked = [(primary, confidence)]
            margin = confidence
        elif sc["iq"] is not None and 70 <= sc["iq"] < 80:
            primary = "Borderline Intellectual Functioning"
            confidence = 0.68
            dx_source = "score_pattern"
            dx_evidence = f"Full-Scale IQ {sc['iq']} (borderline range, 70-79)"
            ranked = [(primary, confidence)]
            margin = confidence
        elif strong_trauma:
            # trauma narrative (event + >=2 trauma symptoms) that the learned
            # model would otherwise mistake for ADHD/anxiety.
            primary = "PTSD / Trauma"
            confidence = 0.62
            dx_source = "symptom_pattern"
            others = [(self.dis_clf.classes_[i], float(probs[i])) for i in order
                      if self.dis_clf.classes_[i] != "ADHD"]
            ranked = [(primary, confidence)] + others[:2]
            margin = confidence - (others[0][1] if others else 0.0)
        else:
            primary = self.dis_clf.classes_[order[0]]
            confidence = float(probs[order[0]])
            runner_up = float(probs[order[1]]) if len(order) > 1 else 0.0
            margin = confidence - runner_up
            ranked = [(self.dis_clf.classes_[i], float(probs[i])) for i in order[:3]]

        # ---- CO-PRIMARY: honest about mixed presentations -----------------
        # A near-tie between two real conditions, or score evidence for two
        # (mixed ADHD/ASD; twice-exceptional ADHD + Dyslexia), should be shown as
        # a dual primary instead of an arbitrary coin-flip single pick.
        co_primary = None
        if dx_source == "stated" and ext.get("stated_secondary"):
            # an explicitly stated DUAL diagnosis ("ADHD AND Dyslexia")
            co_primary = ext["stated_secondary"]
        if dx_source == "model":
            l0 = self.dis_clf.classes_[order[0]]
            l1 = self.dis_clf.classes_[order[1]] if len(order) > 1 else ""
            if (margin < 0.12 and float(probs[order[1]]) >= 0.18
                    and l0 != "Other / Complex" and l1 != "Other / Complex"):
                co_primary = l1
            if sc["srs_high"] and sc["attention_high"]:        # ADHD + ASD
                co_primary = "ASD" if primary == "ADHD" else (
                    "ADHD" if primary == "ASD" else co_primary)
                if primary not in ("ADHD", "ASD"):
                    primary, co_primary = "ADHD", "ASD"
            if sc["reading_low"] and sc["attention_high"]:     # 2e: ADHD + Dyslexia
                if primary == "ADHD":
                    co_primary = "Dyslexia"
                elif primary == "Dyslexia":
                    co_primary = "ADHD"
            if co_primary == primary:
                co_primary = None

        terms = self._top_evidence_terms(text, primary if dx_source == "model" else
                                         self.dis_clf.classes_[order[0]])
        evidence = self._evidence_sentences(text, terms + symptoms)
        cooc = self._cooccurring(text, primary)
        rc_top, rc_factors, rc_detail = self._root_cause(text, primary)
        risk, risk_signals, risk_denied = self._risk(text, primary)

        # confidence band
        if confidence < LOW_CONF:
            conf_band = "Low"
        elif confidence < HIGH_CONF:
            conf_band = "Moderate"
        else:
            conf_band = "High"

        # review policy depends on how the diagnosis was reached
        if dx_source == "no_diagnosis":
            needs_review = risk == "High"
        elif dx_source == "stated":
            # explicit statement: confident, but flag if outside trained scope,
            # if risk is high, if it is a non-specific bucket, or if the learned
            # model disagrees with the stated conclusion
            needs_review = (risk == "High" or not ext["modelled"]
                            or primary == "Other / Complex" or bool(conflict_note))
        elif dx_source in ("symptom_pattern", "score_pattern"):
            needs_review = True   # an inference outside the trained classes
        else:
            needs_review = (conf_band != "High" or risk == "High"
                            or primary == "Other / Complex"
                            or margin < AMBIGUITY_MARGIN)
        if dx_source == "model" and rc_detail and rc_detail.get("abstain"):
            needs_review = True
        if co_primary:           # mixed / 2e presentation -> always review
            needs_review = True

        # explanation (fixes "no explanation generated")
        if dx_source == "stated":
            explanation = (f'Read from an explicit diagnosis statement in the '
                           f'report: "{dx_evidence}".'
                           + ("" if ext["modelled"] else
                              " This condition is outside the model's six trained "
                              "classes, so it is surfaced from the report text and "
                              "should be confirmed clinically.")
                           + conflict_note)
        elif dx_source == "no_diagnosis":
            explanation = ("The report explicitly indicates no diagnosis / "
                           "age-appropriate development; no condition was predicted.")
        elif dx_source == "symptom_pattern":
            explanation = ("Recognised from a trauma-exposure narrative (a "
                           "precipitating event plus trauma symptoms such as "
                           "nightmares, hypervigilance, exaggerated startle, or "
                           "avoidance). Trauma-related attention problems can mimic "
                           "ADHD; this is outside the trained model and must be "
                           "confirmed clinically.")
        elif dx_source == "score_pattern":
            explanation = (f"Inferred from standardized scores ({dx_evidence}). "
                           "This is outside the trained model and must be "
                           "confirmed against full diagnostic criteria.")
        elif dx_source == "uncertain":
            explanation = ("The report explicitly states a definitive diagnosis "
                           "cannot be made yet (e.g. conflicting informants, "
                           "further assessment required). No diagnosis is asserted.")
        else:
            explanation = ("Predicted by the learned model from the report's "
                           "language and symptom pattern.")
            if prior_info.get("applied"):
                explanation += (" The model was uncertain, so this leans on the "
                                "clinical symptom pattern (the disorder may not be "
                                "named explicitly).")
        if co_primary:
            explanation += (f" Mixed presentation: evidence also supports "
                            f"{co_primary}, so both are shown as co-primary — the "
                            "single best label is unreliable here; clinician "
                            "differentiation is needed.")

        recommendation = self._recommendation(primary, conf_band, risk, needs_review)
        if dx_source == "stated":
            recommendation = ("Diagnosis read from the report's explicit statement. "
                              + recommendation)
        if rc_detail and rc_detail.get("abstain"):
            recommendation += (
                " Contributing-factor analysis was inconclusive because its "
                "learned and clinical views disagreed or had a narrow margin.")

        display_primary = f"{primary} + {co_primary}" if co_primary else primary
        return {
            "diagnosis": display_primary,
            "primary_label": display_primary,  # explicit alias for API clients
            "co_primary": co_primary,
            "diagnosis_source": dx_source,     # stated|no_diagnosis|model|score_pattern|symptom_pattern
            "diagnosis_statement": dx_evidence,
            "explanation": explanation,
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
