"""
Multinomial (softmax) logistic-regression classifier in pure numpy.

Why this model:
  * Linear-in-TF-IDF => fast, robust on small/medium text corpora.
  * Softmax outputs => usable per-class confidence scores.
  * Weight matrix => directly explainable (per-class term importances), which
    drives the "evidence from report" highlighting.

Trained with full-batch gradient descent + L2 regularisation. Deterministic.
"""
import json
import numpy as np


def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


def balanced_sample_weight(y):
    """Inverse-frequency sample weights so minority classes (e.g.
    'Other / Complex') are not drowned out by majority classes."""
    from collections import Counter
    counts = Counter(y)
    n, k = len(y), len(counts)
    return np.array([n / (k * counts[label]) for label in y], dtype=float)


class SoftmaxClassifier:
    def __init__(self, l2=1e-3, lr=0.5, epochs=400, seed=0, temperature=1.0):
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.temperature = temperature  # >1 softens overconfident probabilities
        self.classes_ = None
        self.W = None  # (n_features+1, n_classes), last row = bias

    def fit(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.seed)
        self.classes_ = sorted(set(y))
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        n, d = X.shape
        k = len(self.classes_)
        Xb = np.hstack([X, np.ones((n, 1))])
        Y = np.zeros((n, k))
        for i, label in enumerate(y):
            Y[i, cls_idx[label]] = 1.0
        if sample_weight is None:
            sw = np.ones(n)
        else:
            sw = np.asarray(sample_weight, dtype=float)
        sw = (sw / sw.mean()).reshape(-1, 1)
        self.W = rng.normal(0, 0.01, size=(d + 1, k))
        for _ in range(self.epochs):
            P = _softmax(Xb @ self.W)
            grad = Xb.T @ ((P - Y) * sw) / n
            grad[:-1] += self.l2 * self.W[:-1]  # regularise weights, not bias
            self.W -= self.lr * grad
        return self

    def decision_function(self, X):
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        return Xb @ self.W

    def predict_proba(self, X):
        return _softmax(self.decision_function(X) / max(self.temperature, 1e-6))

    def calibrate_temperature(self, X, y_true, grid=None):
        """Fit a single temperature on held-out data by minimising NLL
        (Guo et al. 2017). Cheap, never changes the argmax prediction."""
        if grid is None:
            grid = [round(0.6 + 0.05 * i, 2) for i in range(49)]  # 0.6 .. 3.0
        Z = self.decision_function(X)
        idx = np.array([self.classes_.index(t) for t in y_true])
        best_t, best_nll = 1.0, np.inf
        for t in grid:
            P = _softmax(Z / t)
            nll = -np.log(np.clip(P[np.arange(len(idx)), idx], 1e-12, 1)).mean()
            if nll < best_nll:
                best_nll, best_t = nll, t
        self.temperature = float(best_t)
        return self.temperature

    def predict(self, X):
        P = self.predict_proba(X)
        idx = P.argmax(axis=1)
        return [self.classes_[i] for i in idx]

    def to_dict(self):
        return {"l2": self.l2, "lr": self.lr, "epochs": self.epochs,
                "seed": self.seed, "temperature": self.temperature,
                "classes": self.classes_, "W": self.W.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["l2"], d["lr"], d["epochs"], d["seed"],
                d.get("temperature", 1.0))
        m.classes_ = d["classes"]
        m.W = np.array(d["W"], dtype=np.float64)
        return m


class OneVsRestClassifier:
    """Independent binary logistic regressions per label for MULTI-label
    targets (co-occurring conditions). Each label predicted independently."""

    def __init__(self, l2=1e-3, lr=0.5, epochs=300, seed=0, threshold=0.5):
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.threshold = threshold
        self.labels_ = None
        self.W = None  # (n_features+1, n_labels)

    def fit(self, X, Y):
        # Y: binary matrix (n, n_labels); self.labels_ set by caller separately
        n, d = X.shape
        m = Y.shape[1]
        Xb = np.hstack([X, np.ones((n, 1))])
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 0.01, size=(d + 1, m))
        for _ in range(self.epochs):
            P = 1.0 / (1.0 + np.exp(-(Xb @ self.W)))
            grad = Xb.T @ (P - Y) / n
            grad[:-1] += self.l2 * self.W[:-1]
            self.W -= self.lr * grad
        return self

    def predict_proba(self, X):
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        return 1.0 / (1.0 + np.exp(-(Xb @ self.W)))

    def to_dict(self):
        return {"l2": self.l2, "lr": self.lr, "epochs": self.epochs,
                "seed": self.seed, "threshold": self.threshold,
                "labels": self.labels_, "W": self.W.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["l2"], d["lr"], d["epochs"], d["seed"], d["threshold"])
        m.labels_ = d["labels"]
        m.W = np.array(d["W"], dtype=np.float64)
        return m
