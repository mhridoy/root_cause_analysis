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


# Negation handling -----------------------------------------------------------
# Clinical reports constantly RULE OUT conditions ("no history of restricted
# interests", "denies hyperactivity", "teachers do not report impulsivity").
# A plain bag-of-words counts those ruled-out symptoms as PRESENT and votes for
# the wrong diagnosis. We detect negation cues and prefix the following in-scope
# tokens with "neg_" so they form distinct features and stop poisoning the
# asserted-symptom signal.
NEG_CUES = {
    "no", "not", "n't", "never", "without", "denies", "denied", "deny",
    "denying", "nor", "negative", "absence", "absent", "ruled", "none",
    "neither", "cannot", "lacks", "lacking", "unremarkable", "rule",
}
# Words/punctuation that END a negation scope (contrastive / clause break).
NEG_BREAK = {"but", "however", "although", "though", "yet", "except",
             "aside", "whereas", "still", "nevertheless"}
# Connectors that should NOT consume the negation window, so list negations like
# "no history of major family conflict, abuse, or neglect" stay fully negated.
NEG_CONNECTORS = {"of", "or", "and", "the", "a", "an", "to", "with", "in", "on",
                  "for", "history", "reported", "any", "were", "was", "is",
                  "are", "showed", "show", "signs", "evidence", "presence"}
# Post-position (backward) negation: "Suicidal ideation was explicitly DENIED",
# "Tics were NOT OBSERVED", "ASD was RULED OUT" -> the clinical terms BEFORE the
# trigger are negated. Forward cues ("denied X") are handled separately, so these
# only fire in passive / clause-final position to avoid over-negating "...and
# denied suicidal thoughts" (where the object follows).
NEG_BACKWARD = {"denied", "denies", "absent", "unremarkable", "negative"}
NEG_BACKWARD_AFTER_NOT = {"reported", "observed", "present", "noted", "endorsed",
                          "elicited", "identified", "evident", "indicated",
                          "met", "seen", "found", "detected"}
_BE = {"was", "were", "is", "are", "been", "be"}


def _negation_mask(toks, window=8):
    """Boolean mask of tokens inside a negation scope (forward + backward)."""
    n = len(toks)
    mask = [False] * n
    # forward: cue negates following tokens
    for i, w in enumerate(toks):
        if w in NEG_CUES:
            span, j = 0, i + 1
            while j < n and span <= window:
                if toks[j] in NEG_BREAK:
                    break
                mask[j] = True
                if toks[j] not in NEG_CONNECTORS:
                    span += 1
                j += 1
    # backward: passive / clause-final trigger negates preceding tokens
    for k, t in enumerate(toks):
        backward = False
        if t in NEG_BACKWARD:
            if k == n - 1 or (k + 1 < n and toks[k + 1] in NEG_BREAK):
                backward = True
            elif any(toks[m] in _BE for m in range(max(0, k - 3), k)):
                backward = True
        elif t == "out" and k > 0 and toks[k - 1] == "ruled":
            backward = True
        elif t in NEG_BACKWARD_AFTER_NOT and k > 0 and toks[k - 1] in ("not", "never"):
            backward = True
        if backward:
            span, j = 0, k - 1
            while j >= 0 and span <= window:
                if toks[j] in NEG_BREAK:
                    break
                mask[j] = True
                if toks[j] not in NEG_CONNECTORS:
                    span += 1
                j -= 1
    return mask


def marked_tokens(text, use_negation=True, window=8):
    """Tokenize, dropping stopwords/short tokens; if use_negation, prefix tokens
    that fall inside a negation scope with 'neg_'. Scope starts at a negation
    cue and ends at sentence punctuation, a contrastive conjunction, or after
    `window` tokens (commas/'or'/'and' do NOT end it, so negated lists stay
    negated)."""
    if not use_negation:
        return tokenize(text)
    out = []
    for clause in re.split(r"[.;:!?]", text.lower()):
        toks = _TOKEN_RE.findall(clause)
        mask = _negation_mask(toks, window)
        for w, neg in zip(toks, mask):
            if w in STOPWORDS or len(w) < 3:
                continue
            out.append(("neg_" + w) if neg else w)
    return out


def split_negation(text, window=8):
    """Split text into (asserted_text, negated_text) at the PHRASE level.

    Unlike marked_tokens (which drops stopwords for the classifier), this keeps
    every word so multi-word clinical phrases survive ("loss of interest",
    "no suicidal thoughts"). Words inside a negation scope go to negated_text;
    everything else to asserted_text. Used by the risk / symptom / co-occurring
    detectors so ruled-out findings are never counted as present.
    """
    asserted, negated = [], []
    for clause in re.split(r"[.;:!?]", text.lower()):
        words = re.findall(r"[a-z][a-z\-']+", clause)
        mask = _negation_mask(words, window)
        for w, neg in zip(words, mask):
            (negated if neg else asserted).append(w)
    norm = lambda lst: " " + re.sub(r"[-_]", " ", " ".join(lst)) + " "
    return norm(asserted), norm(negated)


def phrase_present(phrase, hay_normalized):
    """True if phrase (hyphens/underscores normalised to spaces) occurs in the
    already-normalised haystack produced by split_negation()."""
    return re.sub(r"[-_]", " ", phrase.lower()) in hay_normalized


def term_present(term, hay_normalized):
    """Like phrase_present but anchored at a WORD START, so short keywords don't
    match inside longer words (e.g. 'tics' must not match 'mathematics',
    'odd' must not match 'toddler'). The end is left open so morphological
    variants still match ('depress' -> 'depression')."""
    t = re.sub(r"[-_]", " ", term.lower())
    return re.search(r"\b" + re.escape(t), hay_normalized) is not None


def char_ngrams_tokens(tokens, n_min=3, n_max=5):
    """Character n-grams from a token list (skips negated tokens so ruled-out
    words contribute no character evidence either)."""
    grams = []
    for word in tokens:
        if word.startswith("neg_") or word in STOPWORDS:
            continue
        padded = f" {word} "
        L = len(padded)
        for n in range(n_min, n_max + 1):
            if L < n:
                continue
            for i in range(L - n + 1):
                grams.append("c#" + padded[i:i + n])
    return grams


def ngrams(tokens, n_max=2):
    grams = list(tokens)
    for n in range(2, n_max + 1):
        grams += ["_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return grams


_WS_RE = re.compile(r"\s+")


def char_ngrams(text, n_min=3, n_max=5):
    """Character n-grams over the normalised lowercase text ('char_wb'-style:
    computed within word boundaries, padded with spaces). Robust to typos,
    OCR noise, inflections and hyphenation."""
    grams = []
    for word in _TOKEN_RE.findall(text.lower()):
        if word in STOPWORDS:
            continue
        padded = f" {word} "
        L = len(padded)
        for n in range(n_min, n_max + 1):
            if L < n:
                continue
            for i in range(L - n + 1):
                grams.append("c#" + padded[i:i + n])
    return grams


class TfidfVectorizer:
    """Minimal TF-IDF with sublinear TF, smoothed IDF, L2 row norm, n-grams,
    min_df / max_df filtering. Optionally augments word n-grams with
    character n-grams (prefixed 'c#') for typo/OCR robustness."""

    def __init__(self, ngram_max=2, min_df=2, max_df=0.9, max_features=8000,
                 use_char=False, char_min=3, char_max=5, use_negation=False):
        self.ngram_max = ngram_max
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.use_char = use_char
        self.char_min = char_min
        self.char_max = char_max
        self.use_negation = use_negation
        self.vocab_ = {}
        self.idf_ = None

    def _docs_to_grams(self, docs):
        out = []
        for d in docs:
            toks = marked_tokens(d, self.use_negation)
            grams = ngrams(toks, self.ngram_max)
            if self.use_char:
                grams += char_ngrams_tokens(toks, self.char_min, self.char_max)
            out.append(grams)
        return out

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
                "use_char": self.use_char, "char_min": self.char_min,
                "char_max": self.char_max, "use_negation": self.use_negation,
                "vocab": self.vocab_, "idf": self.idf_.tolist()}

    @classmethod
    def from_dict(cls, d):
        v = cls(d["ngram_max"], d["min_df"], d["max_df"], d["max_features"],
                d.get("use_char", False), d.get("char_min", 3),
                d.get("char_max", 5), d.get("use_negation", False))
        v.vocab_ = d["vocab"]
        v.idf_ = np.array(d["idf"], dtype=np.float64)
        return v
