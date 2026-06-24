"""
Root-cause inference engine (the reliable, explainable replacement for the old
keyword-argmax logic).

At inference it combines two independent views and shows its work:
  * LEARNED view  -- the cross-validated text model (linear winner, or the deep
    MLP if it wins on a future, larger dataset) predicting the clinically-grounded
    contributing-factor label from the report's latent semantics.
  * CLINICAL view -- the diagnosis-informed posterior (prior x evidence) from
    clinical_priors, using the *predicted* diagnosis.

The final answer is a calibrated blend. When the two views disagree, or the
blended confidence/margin is low, the engine ABSTAINS and flags the case for
expert review rather than guessing. Per-factor evidence sentences are returned
so the user sees *why*, not just a label.
"""
import os
import re
import json
import numpy as np
from .textproc import TfidfVectorizer
from .model import SoftmaxClassifier
from .deep_models import TruncatedSVD, MLPClassifier
from .clinical_priors import (posterior as clinical_posterior,
                              prior_distribution, ROOT_CAUSE_LIST)
from .lexicons import ROOT_CAUSE_GROUPS

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "..", "models", "rootcause_deep.json")

BLEND_W = 0.6          # weight on the learned text model vs clinical posterior
MIN_CONFIDENCE = 0.42  # below this -> abstain / needs review
MIN_MARGIN = 0.12      # top1-top2 gap below this -> abstain / needs review


class RootCauseEngine:
    def __init__(self, path=DEFAULT_PATH):
        with open(path) as f:
            b = json.load(f)
        self.type = b.get("type", "linear")
        self.tfidf = TfidfVectorizer.from_dict(b["tfidf"])
        self.groups = b.get("rootcause_list", ROOT_CAUSE_LIST)
        self.choice = b.get("choice")
        if self.type == "linear":
            self.clf = SoftmaxClassifier.from_dict(b["linear"])
        else:
            self.svd = TruncatedSVD.from_dict(b["svd"])
            self.mu = np.array(b["scaler"]["mu"]); self.sd = np.array(b["scaler"]["sd"])
            self.mlp = MLPClassifier.from_dict(b["mlp"])
            self.use_prior = b.get("use_prior", False)
            self.diag_clf = (SoftmaxClassifier.from_dict(b["diag_clf"])
                             if b.get("diag_clf") else None)

    def _text_probs(self, text, diagnosis):
        """Return probability over ROOT_CAUSE_LIST from the learned model."""
        X = self.tfidf.transform([text])
        if self.type == "linear":
            classes = self.clf.classes_
            p = self.clf.predict_proba(X)[0]
        else:
            Z = (self.svd.transform(X) - self.mu) / self.sd
            if self.use_prior:
                pri = np.array([prior_distribution(diagnosis)])
                Z = np.hstack([Z, pri])
            classes = self.mlp.classes_
            p = self.mlp.predict_proba(Z)[0]
        # map onto canonical group order
        out = np.zeros(len(self.groups))
        idx = {g: i for i, g in enumerate(self.groups)}
        for c, pv in zip(classes, p):
            if c in idx:
                out[idx[c]] = pv
        s = out.sum()
        return out / s if s > 0 else np.ones(len(self.groups)) / len(self.groups)

    def _evidence_for(self, text, group, max_sent=2):
        kws = ROOT_CAUSE_GROUPS[group]
        sents = re.split(r"(?<=[.!?])\s+|\n+", text)
        scored = []
        for s in sents:
            sl = s.lower()
            hits = sum(sl.count(k) for k in kws)
            if hits and 15 < len(s.strip()) < 280:
                scored.append((hits, s.strip()))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:max_sent]]

    def predict(self, text, diagnosis):
        learned = self._text_probs(text, diagnosis)
        clinical = np.array(clinical_posterior(diagnosis, text))
        blended = BLEND_W * learned + (1 - BLEND_W) * clinical
        blended = blended / blended.sum()

        order = np.argsort(blended)[::-1]
        ranked = [(self.groups[i], float(blended[i])) for i in order]
        top_group = self.groups[order[0]]
        confidence = float(blended[order[0]])
        margin = float(blended[order[0]] - blended[order[1]]) if len(order) > 1 else confidence

        learned_top = self.groups[int(np.argmax(learned))]
        clinical_top = self.groups[int(np.argmax(clinical))]
        disagree = learned_top != clinical_top

        abstain = (confidence < MIN_CONFIDENCE) or (margin < MIN_MARGIN) or disagree

        evidence = {g: self._evidence_for(text, g) for g, _ in ranked[:3]}
        return {
            "top": top_group,
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "ranked": [(g, round(p, 4)) for g, p in ranked],
            "evidence": evidence,
            "learned_top": learned_top,
            "clinical_top": clinical_top,
            "agreement": not disagree,
            "abstain": bool(abstain),
            "learned_probs": {g: round(float(learned[i]), 4)
                              for i, g in enumerate(self.groups)},
            "clinical_probs": {g: round(float(clinical[i]), 4)
                               for i, g in enumerate(self.groups)},
        }


_engine = None


def get_rootcause_engine():
    global _engine
    if _engine is None:
        _engine = RootCauseEngine()
    return _engine
