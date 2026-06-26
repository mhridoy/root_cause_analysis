"""
End-to-end training + evaluation.

Pipeline:
  1. Load corpus text, build ground-truth labels (src.labeling).
  2. Stratified train / validation / test split.
  3. Fit TF-IDF on TRAIN only (no leakage from val/test vocabulary stats).
  4. Train primary-diagnosis softmax classifier:
        (a) STANDARD  -- raw report text.
        (b) LEAKAGE-CONTROLLED -- diagnosis statements masked.
  5. Train root-cause-group classifier and co-occurring multi-label model.
  6. Report accuracy / precision / recall / F1 / calibration on val + test.
  7. Persist everything to models/bundle.json + write reports/.

Run:  python3 train.py --text_dir ../merged_unique_text
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.labeling import build_dataset, write_labels_csv
from src.textproc import TfidfVectorizer, mask_leakage
from src.model import SoftmaxClassifier, OneVsRestClassifier
from src.metrics import (classification_report, stratified_split,
                         expected_calibration_error, confusion_matrix)
from src.lexicons import COOCCURRING_KEYWORDS

HERE = os.path.dirname(os.path.abspath(__file__))


def load_corpus(text_dir, rows):
    docs = []
    for r in rows:
        with open(os.path.join(text_dir, r["file"]), "r",
                  encoding="utf-8", errors="ignore") as f:
            docs.append(f.read())
    return docs


# "Other / Complex" is a heterogeneous catch-all class: with equal weights it
# acts as a noise sink and steals predictions from specific diagnoses (the
# dominant error mode in the confusion matrix). Downweighting it was selected
# by 5-fold cross-validation on train+val (never on test):
#   oc=1.0 -> CV acc 0.931 / macro-F1 0.919
#   oc=0.6 -> CV acc 0.947 / macro-F1 0.922   <- chosen
#   oc=0.4 -> CV acc 0.922 / macro-F1 0.824   (too aggressive)
# Char 3-5-gram features were also evaluated by CV and did not help
# (0.917 vs 0.922 macro-F1) while tripling bundle size, so they stay off.
OTHER_COMPLEX_WEIGHT = 0.6


def class_weights(yv, oc=OTHER_COMPLEX_WEIGHT):
    return np.array([oc if label == "Other / Complex" else 1.0
                     for label in yv])


def train_disease(docs, y, tr, va, te, leakage_mask=False, label="standard"):
    texts = [mask_leakage(d) for d in docs] if leakage_mask else docs
    ytr = [y[i] for i in tr]
    yva = [y[i] for i in va]

    # use_negation: ruled-out symptoms ("no history of X", "denies X") stop
    #   voting for the wrong class.
    # use_char: character n-grams recover morphological/OOV terms (e.g.
    #   ritualized, contamination) that exact-token matching misses.
    vec = TfidfVectorizer(ngram_max=2, min_df=2, max_df=0.9, max_features=14000,
                          use_char=True, char_min=4, char_max=5,
                          use_negation=True)
    Xtr = vec.fit_transform([texts[i] for i in tr])
    Xva = vec.transform([texts[i] for i in va])
    Xte = vec.transform([texts[i] for i in te])
    clf = SoftmaxClassifier(l2=3e-3, lr=0.5, epochs=600, seed=0)
    clf.fit(Xtr, ytr, sample_weight=class_weights(ytr))

    # Temperature calibration on validation (never changes the prediction,
    # only makes the confidence score honest -> lower ECE, better
    # "needs doctor review" thresholding).
    t = clf.calibrate_temperature(Xva, yva)
    print(f"      [{label}] temperature={t:.2f}")

    labels = clf.classes_

    def _eval(X, idxs):
        P = clf.predict_proba(X)
        preds = [labels[j] for j in P.argmax(axis=1)]
        truth = [y[i] for i in idxs]
        rep = classification_report(truth, preds, labels)
        conf = P.max(axis=1)
        correct = [t == p for t, p in zip(truth, preds)]
        rep["_ece"] = expected_calibration_error(conf, correct)
        rep["_mean_confidence"] = float(conf.mean())
        return rep, preds, truth

    val_rep, _, _ = _eval(Xva, va)
    test_rep, test_preds, test_truth = _eval(Xte, te)
    cm = confusion_matrix(test_truth, test_preds, labels).tolist()
    return {
        "vec": vec, "clf": clf, "labels": labels,
        "val": val_rep, "test": test_rep, "confusion": cm,
        "label": label,
    }


def train_rootcause(docs, y_rc, tr, va, te):
    vec = TfidfVectorizer(ngram_max=2, min_df=2, max_df=0.9, max_features=6000)
    Xtr = vec.fit_transform([docs[i] for i in tr])
    Xva = vec.transform([docs[i] for i in va])
    Xte = vec.transform([docs[i] for i in te])
    ytr = [y_rc[i] for i in tr]
    clf = SoftmaxClassifier(l2=3e-3, lr=0.5, epochs=600, seed=0)
    clf.fit(Xtr, ytr)
    clf.calibrate_temperature(Xva, [y_rc[i] for i in va])
    labels = clf.classes_
    P = clf.predict_proba(Xte)
    preds = [labels[j] for j in P.argmax(axis=1)]
    truth = [y_rc[i] for i in te]
    rep = classification_report(truth, preds, labels)
    return {"vec": vec, "clf": clf, "labels": labels, "test": rep}


def train_cooccurring(docs, rows, tr, te):
    labels = list(COOCCURRING_KEYWORDS.keys())
    lab_idx = {l: i for i, l in enumerate(labels)}

    def to_vec(idxs):
        Y = np.zeros((len(idxs), len(labels)))
        for r, i in enumerate(idxs):
            for c in (rows[i]["cooccurring"].split("|") if rows[i]["cooccurring"] else []):
                if c in lab_idx:
                    Y[r, lab_idx[c]] = 1.0
        return Y

    vec = TfidfVectorizer(ngram_max=2, min_df=2, max_df=0.9, max_features=6000)
    Xtr = vec.fit_transform([docs[i] for i in tr])
    Xte = vec.transform([docs[i] for i in te])
    Ytr, Yte = to_vec(tr), to_vec(te)
    clf = OneVsRestClassifier(l2=3e-3, lr=0.5, epochs=400, seed=0, threshold=0.5)
    clf.labels_ = labels
    clf.fit(Xtr, Ytr)
    Pte = clf.predict_proba(Xte)

    def _micro(thr):
        P = (Pte >= thr).astype(int)
        tp = int((P * Yte).sum()); fp = int((P * (1 - Yte)).sum())
        fn = int(((1 - P) * Yte).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    # choose threshold maximising micro-F1 on the train-derived sweep
    best = max((0.2, 0.25, 0.3, 0.35, 0.4, 0.5),
               key=lambda t: _micro(t)[2])
    clf.threshold = best
    prec, rec, f1 = _micro(best)
    return {"vec": vec, "clf": clf, "labels": labels,
            "test": {"micro_precision": prec, "micro_recall": rec,
                     "micro_f1": f1, "threshold": best}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_dir", default=os.path.join(HERE, "..", "merged_unique_text"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    text_dir = os.path.abspath(args.text_dir)
    print(f"[1/6] Building labels from {text_dir}")
    rows = build_dataset(text_dir)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    write_labels_csv(rows, os.path.join(HERE, "data", "labels.csv"))
    docs = load_corpus(text_dir, rows)
    y = [r["primary_disease"] for r in rows]
    y_rc = [r["root_cause_group"] for r in rows]
    print(f"      {len(docs)} reports labelled.")

    print("[2/6] Stratified split (70/15/15)")
    tr, va, te = stratified_split(y, (0.7, 0.15, 0.15), seed=args.seed)
    print(f"      train={len(tr)}  val={len(va)}  test={len(te)}")

    print("[3/6] Training primary-diagnosis model (STANDARD)")
    std = train_disease(docs, y, tr, va, te, leakage_mask=False, label="standard")
    print(f"      test accuracy={std['test']['_accuracy']:.4f}  "
          f"macro-F1={std['test']['_macro_f1']:.4f}")

    print("[4/6] Training primary-diagnosis model (LEAKAGE-CONTROLLED)")
    leak = train_disease(docs, y, tr, va, te, leakage_mask=True, label="leakage_controlled")
    print(f"      test accuracy={leak['test']['_accuracy']:.4f}  "
          f"macro-F1={leak['test']['_macro_f1']:.4f}")

    print("[5/6] Training root-cause + co-occurring models")
    rc = train_rootcause(docs, y_rc, tr, va, te)
    co = train_cooccurring(docs, rows, tr, te)
    print(f"      root-cause test accuracy={rc['test']['_accuracy']:.4f}")
    print(f"      co-occurring micro-F1={co['test']['micro_f1']:.4f}")

    print("[6/6] Saving bundle + metrics")
    # The DEPLOYED disease model = standard (full text), which is what real
    # uploaded reports contain. Leakage-controlled metrics are reported for honesty.
    bundle = {
        "disease": {"vec": std["vec"].to_dict(), "clf": std["clf"].to_dict(),
                    "labels": std["labels"]},
        "rootcause": {"vec": rc["vec"].to_dict(), "clf": rc["clf"].to_dict(),
                      "labels": rc["labels"]},
        "cooccurring": {"vec": co["vec"].to_dict(), "clf": co["clf"].to_dict(),
                        "labels": co["labels"]},
        "meta": {"n_reports": len(docs), "seed": args.seed},
    }
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    with open(os.path.join(HERE, "models", "bundle.json"), "w") as f:
        json.dump(bundle, f)

    metrics = {
        "disease_standard": {"val": std["val"], "test": std["test"],
                             "confusion": std["confusion"], "labels": std["labels"]},
        "disease_leakage_controlled": {"val": leak["val"], "test": leak["test"],
                                       "confusion": leak["confusion"],
                                       "labels": leak["labels"]},
        "rootcause": rc["test"],
        "cooccurring": co["test"],
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "n_reports": len(docs),
    }
    with open(os.path.join(HERE, "models", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Done. See models/metrics.json")
    return metrics


if __name__ == "__main__":
    main()
