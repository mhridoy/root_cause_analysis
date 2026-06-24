"""
Deep models, implemented from scratch in numpy (PyTorch/TF can't be installed in
this environment, and on ~360 documents a from-scratch regularised MLP is the
honest sweet spot anyway).

Contents:
  * TruncatedSVD  -- latent semantic analysis: turns sparse keyword TF-IDF into
    dense semantic dimensions, so the classifier sees latent topics rather than
    literal word counts. This is the key step that makes the result "not text
    extraction".
  * MLPClassifier -- multilayer perceptron (2 hidden layers, ReLU, dropout,
    softmax) trained with the Adam optimiser, mini-batches, L2 weight decay and
    early stopping. Real backprop, written out by hand.
  * temperature_scale -- post-hoc confidence calibration so probabilities mean
    what they say.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Latent Semantic Analysis
# ---------------------------------------------------------------------------
class TruncatedSVD:
    def __init__(self, n_components=200, seed=0):
        self.n_components = n_components
        self.seed = seed
        self.components_ = None
        self.singular_values_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        k = min(self.n_components, min(X.shape) - 1)
        # full economy SVD is fine at this scale (n<=~360)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = Vt[:k]
        self.singular_values_ = s[:k]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def to_dict(self):
        return {"n_components": self.n_components,
                "components": self.components_.tolist(),
                "singular_values": self.singular_values_.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["n_components"])
        m.components_ = np.array(d["components"], dtype=np.float64)
        m.singular_values_ = np.array(d["singular_values"], dtype=np.float64)
        return m


def _relu(z):
    return np.maximum(0, z)


def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Multilayer Perceptron (deep net) — numpy + Adam
# ---------------------------------------------------------------------------
class MLPClassifier:
    def __init__(self, hidden=(256, 128), l2=1e-4, lr=2e-3, epochs=300,
                 batch_size=32, dropout=0.3, seed=0, patience=30):
        self.hidden = tuple(hidden)
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed
        self.patience = patience
        self.classes_ = None
        self.params = None
        self.T = 1.0  # calibration temperature

    # -- init / serialise --------------------------------------------------
    def _init_params(self, d_in, d_out):
        rng = np.random.default_rng(self.seed)
        sizes = [d_in] + list(self.hidden) + [d_out]
        P = {}
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            # He initialisation for ReLU layers
            P[f"W{i}"] = rng.normal(0, np.sqrt(2.0 / fan_in),
                                    size=(sizes[i], sizes[i + 1]))
            P[f"b{i}"] = np.zeros(sizes[i + 1])
        return P

    def _forward(self, X, train=False, rng=None):
        cache = {"A0": X}
        A = X
        n_layers = len(self.hidden) + 1
        for i in range(n_layers):
            Z = A @ self.params[f"W{i}"] + self.params[f"b{i}"]
            if i < n_layers - 1:
                A = _relu(Z)
                if train and self.dropout > 0:
                    mask = (rng.random(A.shape) >= self.dropout) / (1 - self.dropout)
                    A = A * mask
                    cache[f"mask{i}"] = mask
                cache[f"Z{i}"] = Z
                cache[f"A{i+1}"] = A
            else:
                logits = Z
        cache["logits"] = logits
        return logits, cache

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = sorted(set(y))
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        d_in, d_out = X.shape[1], len(self.classes_)
        Y = np.zeros((len(y), d_out));
        for i, lab in enumerate(y):
            Y[i, cls_idx[lab]] = 1.0
        self.params = self._init_params(d_in, d_out)
        rng = np.random.default_rng(self.seed)

        # Adam state
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        n = X.shape[0]
        n_layers = len(self.hidden) + 1
        best_val, best_params, wait, t = -1, None, 0, 0

        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                bidx = perm[start:start + self.batch_size]
                xb, yb = X[bidx], Y[bidx]
                logits, cache = self._forward(xb, train=True, rng=rng)
                P = _softmax(logits)
                grads = {}
                dZ = (P - yb) / xb.shape[0]
                for i in reversed(range(n_layers)):
                    A_prev = cache[f"A{i}"]
                    grads[f"W{i}"] = A_prev.T @ dZ + self.l2 * self.params[f"W{i}"]
                    grads[f"b{i}"] = dZ.sum(axis=0)
                    if i > 0:
                        dA = dZ @ self.params[f"W{i}"].T
                        if self.dropout > 0 and f"mask{i-1}" in cache:
                            dA = dA * cache[f"mask{i-1}"]
                        dZ = dA * (cache[f"Z{i-1}"] > 0)
                # Adam update
                t += 1
                for k in self.params:
                    m[k] = b1 * m[k] + (1 - b1) * grads[k]
                    v[k] = b2 * v[k] + (1 - b2) * (grads[k] ** 2)
                    mhat = m[k] / (1 - b1 ** t)
                    vhat = v[k] / (1 - b2 ** t)
                    self.params[k] -= self.lr * mhat / (np.sqrt(vhat) + eps)

            # early stopping on val accuracy
            if X_val is not None and len(X_val):
                acc = np.mean([p == t_ for p, t_ in zip(self.predict(X_val), y_val)])
                if acc > best_val:
                    best_val = acc
                    best_params = {k: vv.copy() for k, vv in self.params.items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_params is not None:
            self.params = best_params
        return self

    def decision_logits(self, X):
        X = np.asarray(X, dtype=np.float64)
        logits, _ = self._forward(X, train=False)
        return logits

    def predict_proba(self, X):
        return _softmax(self.decision_logits(X) / self.T)

    def predict(self, X):
        P = self.predict_proba(X)
        return [self.classes_[i] for i in P.argmax(axis=1)]

    def calibrate(self, X_val, y_val):
        """Temperature scaling: find scalar T>0 minimising val NLL."""
        if X_val is None or not len(X_val):
            return
        idx = {c: i for i, c in enumerate(self.classes_)}
        # keep only val items whose class the model can represent
        keep = [j for j, c in enumerate(y_val) if c in idx]
        if not keep:
            return
        Xv = np.asarray(X_val)[keep]
        logits = self.decision_logits(Xv)
        yv = np.array([idx[y_val[j]] for j in keep])
        best_T, best_nll = 1.0, 1e9
        for T in np.linspace(0.5, 4.0, 36):
            P = _softmax(logits / T)
            nll = -np.mean(np.log(P[np.arange(len(yv)), yv] + 1e-12))
            if nll < best_nll:
                best_nll, best_T = nll, T
        self.T = float(best_T)

    def to_dict(self):
        return {"hidden": list(self.hidden), "l2": self.l2, "lr": self.lr,
                "epochs": self.epochs, "batch_size": self.batch_size,
                "dropout": self.dropout, "seed": self.seed, "T": self.T,
                "classes": self.classes_,
                "params": {k: v.tolist() for k, v in self.params.items()}}

    @classmethod
    def from_dict(cls, d):
        m = cls(hidden=d["hidden"], l2=d["l2"], lr=d["lr"], epochs=d["epochs"],
                batch_size=d["batch_size"], dropout=d["dropout"], seed=d["seed"])
        m.T = d.get("T", 1.0)
        m.classes_ = d["classes"]
        m.params = {k: np.array(v, dtype=np.float64) for k, v in d["params"].items()}
        return m
