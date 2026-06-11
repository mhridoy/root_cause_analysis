"""
Text preprocessing + a self-contained TF-IDF vectorizer (numpy only).

No scikit-learn dependency: the sandbox blocks PyPI, and keeping the runtime to
numpy means the trained model runs on almost any Python install.
"""
import re
import math
import json
import numpy as np
from .lexicons import LEAKAGE_PATTERNS, DISEASE_KEYWORDS

_TOKEN_RE = re.compile(r"[a-z][a-z\-']+")

STOPWORDS = set("""
a an the and or but if then else of to in on at by for with without from into
is are was were be been being it its this that these those as he she they them
his her their we you your i me my our not no yes do does did has have had will
would can could should may might must also than too very more most some any
report assessment patient client child adolescent results result page date name
""".split())


def mask_leakage(text: str) -> str:
    """Remove diagnosis-revealing statements + disease names for the
    leakage-controlled evaluation. The model must then rely on described
    symptoms and findings rather than the stated answer."""
    t = text
    for pat in LEAKAGE_PATTERNS:
        t = re.sub(pat, " [DX] ", t, flags=re.IGNORECASE)
    for kws in DISEASE_KEYWORDS.values():
        for kw in sorted(kws, key=len, reverse=True):
            t = re.sub(re.escape(kw), " [DX] ", t, flags=re.IGNORECASE)
    return t


def tokenize(text: str):
    toks = _TOKEN_RE.findall(text.lower())
    out = []
    for w in toks:
        if w in STOPWORDS or len(w) < 3:
            continue
        out.append(w)
    return out


def ngrams(tokens, n_max=2):
    grams = list(tokens)
    for n in range(2, n_max + 1):
        grams += ["_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return grams


class TfidfVectorizer:
    """Minimal TF-IDF with sublinear TF, smoothed IDF, L2 row norm, n-grams,
    min_df / max_df filtering."""

    def __init__(self, ngram_max=2, min_df=2, max_df=0.9, max_features=8000):
        self.ngram_max = ngram_max
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.vocab_ = {}
        self.idf_ = None

    def _docs_to_grams(self, docs):
        return [ngrams(tokenize(d), self.ngram_max) for d in docs]

    def fit(self, docs):
        grammed = self._docs_to_grams(docs)
        n_docs = len(grammed)
        df = {}
        for grams in grammed:
            for g in set(grams):
                df[g] = df.get(g, 0) + 1
        max_df_count = self.max_df * n_docs
        # candidate terms passing df thresholds, ranked by document frequency
        cand = [(g, c) for g, c in df.items()
                if c >= self.min_df and c <= max_df_count]
        cand.sort(key=lambda x: (-x[1], x[0]))
        cand = cand[:self.max_features]
        self.vocab_ = {g: i for i, (g, _) in enumerate(cand)}
        self.idf_ = np.zeros(len(self.vocab_), dtype=np.float64)
        for g, i in self.vocab_.items():
            self.idf_[i] = math.log((1 + n_docs) / (1 + df[g])) + 1.0
        return self

    def transform(self, docs):
        grammed = self._docs_to_grams(docs)
        X = np.zeros((len(grammed), len(self.vocab_)), dtype=np.float64)
        for r, grams in enumerate(grammed):
            counts = {}
            for g in grams:
                j = self.vocab_.get(g)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                X[r, j] = (1.0 + math.log(c)) * self.idf_[j]  # sublinear tf * idf
        # L2 normalise rows
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

    @property
    def feature_names_(self):
        inv = [None] * len(self.vocab_)
        for g, i in self.vocab_.items():
            inv[i] = g
        return inv

    def to_dict(self):
        return {"ngram_max": self.ngram_max, "min_df": self.min_df,
                "max_df": self.max_df, "max_features": self.max_features,
                "vocab": self.vocab_, "idf": self.idf_.tolist()}

    @classmethod
    def from_dict(cls, d):
        v = cls(d["ngram_max"], d["min_df"], d["max_df"], d["max_features"])
        v.vocab_ = d["vocab"]
        v.idf_ = np.array(d["idf"], dtype=np.float64)
        return v
