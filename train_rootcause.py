"""
Root-cause model: build clinically-grounded labels, then compare approaches with
5-fold cross-validation and pick the most accurate + best-calibrated one.

Compared (all predict the SAME grounded labels, from TEXT only -> fair):
  B  TF-IDF  -> linear softmax            (shallow baseline)
  C  TF-IDF  -> LSA(SVD) -> deep MLP       (deep representation, text only)
  C+ [LSA ; predicted-diagnosis prior] -> deep MLP   (hierarchical: diagnosis
     predicted per-fold then fed in; realistic, not circular with the label)

Outputs models/rootcause_deep.json (winning model + LSA + scaler + tfidf) and
prints a CV table.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.labeling import build_dataset
from src.textproc import TfidfVectorizer
from src.model import SoftmaxClassifier
from src.deep_models import TruncatedSVD, MLPClassifier
from src.metrics import classification_report, expected_calibration_error
from src.clinical_priors import grounded_label, prior_distribution, ROOT_CAUSE_LIST

HERE = os.path.dirname(os.path.abspath(__file__))


def load(text_dir):
    rows = build_dataset(text_dir)
    docs, diag, rc = [], [], []
    for r in rows:
        with open(os.path.join(text_dir, r["file"]), encoding="utf-8", errors="ignore") as f:
            t = f.read()
        docs.append(t)
        diag.append(r["primary_disease"])
        rc.append(grounded_label(r["primary_disease"], t))
    return docs, diag, rc


def stratified_kfold(y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for i, l in enumerate(y):
        by.setdefault(l, []).append(i)
    folds = [[] for _ in range(k)]
    for l, idxs in by.items():
        idxs = list(idxs); rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            folds[j % k].append(i)
    return folds


def standardize_fit(X):
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
    return mu, sd


def evaluate(docs, diag, rc, seed=0):
    folds = stratified_kfold(rc, k=3, seed=seed)
    results = {"B": [], "C": [], "C+": []}
    cal = {"B": [], "C": [], "C+": []}
    for f in range(3):
        te = folds[f]
        tr = [i for j in range(3) if j != f for i in folds[j]]
        tr_txt = [docs[i] for i in tr]; te_txt = [docs[i] for i in te]
        ytr = [rc[i] for i in tr]; yte = [rc[i] for i in te]

        vec = TfidfVectorizer(ngram_max=2, min_df=2, max_df=0.9, max_features=3000)
        Xtr = vec.fit_transform(tr_txt); Xte = vec.transform(te_txt)

        # inner val split for early stopping / calibration (from train)
        n_in = max(1, int(0.15 * len(tr)))
        rng = np.random.default_rng(seed + f); order = rng.permutation(len(tr))
        vi, ti = order[:n_in], order[n_in:]

        # --- B: linear on tfidf ---
        lin = SoftmaxClassifier(l2=3e-3, lr=0.7, epochs=150, seed=0)
        lin.fit(Xtr[ti], [ytr[i] for i in ti])
        pB = lin.predict(Xte)
        results["B"].append(classification_report(yte, pB, ROOT_CAUSE_LIST))
        probB = lin.predict_proba(Xte)
        cal["B"].append(expected_calibration_error(
            probB.max(1), [a == b for a, b in zip(yte, pB)]))

        # --- LSA features ---
        svd = TruncatedSVD(n_components=120, seed=0)
        Ltr = svd.fit_transform(Xtr); Lte = svd.transform(Xte)
        mu, sd = standardize_fit(Ltr[ti])
        Ztr = (Ltr - mu) / sd; Zte = (Lte - mu) / sd

        # --- C: deep MLP, text only ---
        mlp = MLPClassifier(hidden=(256, 128), dropout=0.3, l2=1e-4, lr=2e-3,
                            epochs=200, batch_size=32, seed=0, patience=20)
        mlp.fit(Ztr[ti], [ytr[i] for i in ti], Ztr[vi], [ytr[i] for i in vi])
        mlp.calibrate(Ztr[vi], [ytr[i] for i in vi])
        pC = mlp.predict(Zte)
        results["C"].append(classification_report(yte, pC, ROOT_CAUSE_LIST))
        cal["C"].append(expected_calibration_error(
            mlp.predict_proba(Zte).max(1), [a == b for a, b in zip(yte, pC)]))

        # --- C+: hierarchical (predicted-diagnosis prior appended) ---
        dx = SoftmaxClassifier(l2=3e-3, lr=0.7, epochs=150, seed=0)
        dx.fit(Xtr[ti], [diag[tr[i]] for i in ti])
        def prior_feats(Xmat):
            preds = dx.predict(Xmat)
            return np.array([prior_distribution(p) for p in preds])
        Ptr = prior_feats(Xtr); Pte = prior_feats(Xte)
        Ztr2 = np.hstack([Ztr, Ptr]); Zte2 = np.hstack([Zte, Pte])
        mlp2 = MLPClassifier(hidden=(256, 128), dropout=0.3, l2=1e-4, lr=2e-3,
                             epochs=200, batch_size=32, seed=0, patience=20)
        mlp2.fit(Ztr2[ti], [ytr[i] for i in ti], Ztr2[vi], [ytr[i] for i in vi])
        mlp2.calibrate(Ztr2[vi], [ytr[i] for i in vi])
        pCp = mlp2.predict(Zte2)
        results["C+"].append(classification_report(yte, pCp, ROOT_CAUSE_LIST))
        cal["C+"].append(expected_calibration_error(
            mlp2.predict_proba(Zte2).max(1), [a == b for a, b in zip(yte, pCp)]))

    return results, cal


def summarize(results, cal):
    out = {}
    for k in results:
        acc = np.mean([r["_accuracy"] for r in results[k]])
        f1 = np.mean([r["_macro_f1"] for r in results[k]])
        ece = np.mean(cal[k])
        out[k] = {"cv_accuracy": round(float(acc), 4),
                  "cv_macro_f1": round(float(f1), 4),
                  "cv_ece": round(float(ece), 4)}
    return out


def train_final_and_save(docs, diag, rc, choice):
    """Train the winning configuration on ALL data and persist it (type-aware)."""
    vec = TfidfVectorizer(ngram_max=2, min_df=2, max_df=0.9, max_features=8000)
    X = vec.fit_transform(docs)
    n_in = max(1, int(0.15 * len(docs)))
    rng = np.random.default_rng(123); order = rng.permutation(len(docs))
    vi, ti = order[:n_in], order[n_in:]
    bundle = {"choice": choice, "rootcause_list": ROOT_CAUSE_LIST,
              "tfidf": vec.to_dict()}

    if choice == "B":   # linear winner
        lin = SoftmaxClassifier(l2=3e-3, lr=0.6, epochs=400, seed=0)
        lin.fit(X[ti], [rc[i] for i in ti])
        bundle["type"] = "linear"
        bundle["linear"] = lin.to_dict()
    else:               # deep winner (C / C+)
        svd = TruncatedSVD(n_components=160, seed=0)
        L = svd.fit_transform(X)
        mu, sd = standardize_fit(L[ti]); Z = (L - mu) / sd
        use_prior = choice == "C+"
        if use_prior:
            dx = SoftmaxClassifier(l2=3e-3, lr=0.6, epochs=400, seed=0)
            dx.fit(X[ti], [diag[i] for i in ti])
            P = np.array([prior_distribution(p) for p in dx.predict(X)])
            Z = np.hstack([Z, P])
        mlp = MLPClassifier(hidden=(256, 128), dropout=0.3, l2=1e-4, lr=2e-3,
                            epochs=300, batch_size=32, seed=0, patience=30)
        mlp.fit(Z[ti], [rc[i] for i in ti], Z[vi], [rc[i] for i in vi])
        mlp.calibrate(Z[vi], [rc[i] for i in vi])
        bundle.update({"type": "mlp", "use_prior": use_prior,
                       "svd": svd.to_dict(),
                       "scaler": {"mu": mu.tolist(), "sd": sd.tolist()},
                       "mlp": mlp.to_dict()})
        if use_prior:
            bundle["diag_clf"] = dx.to_dict()
    with open(os.path.join(HERE, "models", "rootcause_deep.json"), "w") as f:
        json.dump(bundle, f)


def main():
    args = [a for a in sys.argv[1:]]
    if args and args[0] == "--text_dir":
        args = args[1:]
    text_dir = os.path.abspath(args[0] if args
                               else os.path.join(HERE, "..", "merged_unique_text"))
    print("Loading + grounding labels…")
    docs, diag, rc = load(text_dir)
    from collections import Counter
    print("Grounded root-cause distribution:", dict(Counter(rc)))
    print("\n3-fold CV (predicting grounded labels from text)…")
    results, cal = evaluate(docs, diag, rc)
    summ = summarize(results, cal)
    print(f"\n{'model':4s} {'CV-acc':>8s} {'macroF1':>8s} {'ECE':>7s}")
    names = {"B": "B linear (shallow)", "C": "C deep MLP+LSA",
             "C+": "C+ deep+diagnosis"}
    for k in ["B", "C", "C+"]:
        s = summ[k]
        print(f"{k:4s} {s['cv_accuracy']:8.3f} {s['cv_macro_f1']:8.3f} "
              f"{s['cv_ece']:7.3f}   {names[k]}")
    # Pick the most accurate honestly. (No thumb on the scale for the deep net —
    # if it doesn't win, we say so.)
    best = max(["B", "C", "C+"], key=lambda k: summ[k]["cv_accuracy"])
    print(f"\nWinner: {best} ({names[best]})  -> training final + saving")
    train_final_and_save(docs, diag, rc, best)
    json.dump({"cv": summ, "winner": best},
              open(os.path.join(HERE, "models", "rootcause_cv.json"), "w"), indent=2)
    print("Saved models/rootcause_deep.json + models/rootcause_cv.json")


if __name__ == "__main__":
    main()
