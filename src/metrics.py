"""Evaluation metrics in pure numpy (no scikit-learn)."""
import numpy as np


def confusion_matrix(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M


def classification_report(y_true, y_pred, labels):
    """Return dict with per-class precision/recall/f1/support + macro & accuracy."""
    M = confusion_matrix(y_true, y_pred, labels)
    report = {}
    precisions, recalls, f1s, supports = [], [], [], []
    for i, lab in enumerate(labels):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        support = M[i, :].sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        report[lab] = {"precision": prec, "recall": rec, "f1": f1,
                       "support": int(support)}
        if support > 0:
            precisions.append(prec); recalls.append(rec); f1s.append(f1)
        supports.append(support)
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    report["_accuracy"] = correct / total if total else 0.0
    report["_macro_precision"] = float(np.mean(precisions)) if precisions else 0.0
    report["_macro_recall"] = float(np.mean(recalls)) if recalls else 0.0
    report["_macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    # weighted f1
    wsum = sum(supports)
    report["_weighted_f1"] = (
        sum(report[l]["f1"] * report[l]["support"] for l in labels) / wsum
        if wsum else 0.0)
    return report


def stratified_split(y, ratios=(0.7, 0.15, 0.15), seed=0):
    """Return (train_idx, val_idx, test_idx) preserving class proportions."""
    rng = np.random.default_rng(seed)
    by_class = {}
    for i, lab in enumerate(y):
        by_class.setdefault(lab, []).append(i)
    train, val, test = [], [], []
    for lab, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = max(1, int(round(ratios[0] * n)))
        n_va = int(round(ratios[1] * n))
        # ensure at least 1 in test when class has >=3
        if n >= 3 and n_tr + n_va >= n:
            n_va = max(1, n_va - 1)
        train += idxs[:n_tr]
        val += idxs[n_tr:n_tr + n_va]
        test += idxs[n_tr + n_va:]
    return sorted(train), sorted(val), sorted(test)


def expected_calibration_error(confidences, correct, n_bins=10):
    """ECE: |confidence - accuracy| averaged over confidence bins."""
    confidences = np.asarray(confidences)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for b in range(n_bins):
        mask = (confidences > bins[b]) & (confidences <= bins[b + 1])
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
