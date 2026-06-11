from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


MANIFEST = Path("/Users/moshiur/Downloads/authentic_reports/merged_manifest.csv")
ARTIFACT_DIR = Path("/Users/moshiur/Downloads/archives/disease_name_model")
DISPLAY = {
    "adhd": "ADHD",
    "asd": "ASD",
    "depression": "Depression",
    "dyslexia": "Dyslexia",
    "gad": "GAD",
    "ocd": "OCD",
}


def infer_disease(file_name: str) -> str | None:
    name = file_name.lower()

    if name.startswith("zip__adhd_report_"):
        return "adhd"
    if name.startswith("zip__asd_report_"):
        return "asd"
    if name.startswith("zip__depression_report_"):
        return "depression"
    if name.startswith("zip__dyslexia_report_"):
        return "dyslexia"
    if name.startswith("zip__gad_report_"):
        return "gad"
    if name.startswith("zip__ocd_report_"):
        return "ocd"

    if "major_depressive_disorder" in name or "severe_depression" in name:
        return "depression"
    if "psychological_diagnostic_report_gad" in name or "_gad_" in name:
        return "gad"
    if "dyslexia" in name:
        return "dyslexia"
    if "autism" in name or "asd" in name:
        return "asd"
    if "adhd" in name:
        return "adhd"
    return None


def load_dataset() -> tuple[list[str], list[str], list[str]]:
    rows = list(csv.DictReader(MANIFEST.open()))
    texts: list[str] = []
    labels: list[str] = []
    names: list[str] = []
    for row in rows:
        if row["ml_included"] != "true":
            continue
        text_file = row["text_file"]
        if not text_file:
            continue
        name = Path(text_file).name
        label = infer_disease(name)
        if label is None:
            continue
        texts.append(Path(text_file).read_text(errors="ignore"))
        labels.append(label)
        names.append(name)
    return texts, labels, names


def build_features(texts: list[str], fit: bool, vectorizers: dict[str, TfidfVectorizer] | None = None):
    if fit:
        word = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        x_word = word.fit_transform(texts)
        x_char = char.fit_transform(texts)
        return hstack([x_word, x_char]), {"word": word, "char": char}
    assert vectorizers is not None
    return hstack(
        [vectorizers["word"].transform(texts), vectorizers["char"].transform(texts)]
    ), vectorizers


def cross_validate(texts: list[str], labels: list[str]) -> dict[str, object]:
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = []
    micro_scores = []
    macro_scores = []
    all_true: list[int] = []
    all_pred: list[int] = []

    for train_idx, test_idx in splitter.split(texts, y):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        x_train, vecs = build_features(train_texts, fit=True)
        x_test, _ = build_features(test_texts, fit=False, vectorizers=vecs)
        clf = LogisticRegression(max_iter=4000, class_weight="balanced")
        clf.fit(x_train, y[train_idx])
        preds = clf.predict(x_test)
        acc_scores.append(accuracy_score(y[test_idx], preds))
        micro_scores.append(f1_score(y[test_idx], preds, average="micro", zero_division=0))
        macro_scores.append(f1_score(y[test_idx], preds, average="macro", zero_division=0))
        all_true.extend(y[test_idx].tolist())
        all_pred.extend(preds.tolist())

    report = classification_report(
        all_true,
        all_pred,
        target_names=encoder.classes_.tolist(),
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(np.mean(acc_scores)),
        "micro_f1": float(np.mean(micro_scores)),
        "macro_f1": float(np.mean(macro_scores)),
        "class_report": report,
    }


def train() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    texts, labels, names = load_dataset()
    metrics = cross_validate(texts, labels)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    x_all, vecs = build_features(texts, fit=True)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced")
    clf.fit(x_all, y)
    probs = clf.predict_proba(x_all)
    preds = clf.predict(x_all)

    joblib.dump(
        {"classifier": clf, "vectorizers": vecs, "encoder": encoder, "display": DISPLAY, "unknown_threshold": 0.55},
        ARTIFACT_DIR / "disease_name_classifier.joblib",
    )

    with (ARTIFACT_DIR / "training_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "true_label", "predicted_label", *[f"prob_{c}" for c in encoder.classes_]])
        for name, true, pred, row in zip(names, labels, encoder.inverse_transform(preds), probs):
            writer.writerow([name, true, pred, *row.tolist()])

    label_counts: dict[str, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    (ARTIFACT_DIR / "model_report.json").write_text(
        json.dumps(
            {
                "sample_count": len(texts),
                "labels": encoder.classes_.tolist(),
                "label_counts": label_counts,
                "metrics": metrics,
            },
            indent=2,
        )
    )


def predict(path: Path) -> dict[str, object]:
    from root_cause_model import read_report_text

    artifact = joblib.load(ARTIFACT_DIR / "disease_name_classifier.joblib")
    text = read_report_text(path)
    x, _ = build_features([text], fit=False, vectorizers=artifact["vectorizers"])
    probs = artifact["classifier"].predict_proba(x)[0]
    encoder: LabelEncoder = artifact["encoder"]
    ranked = sorted(
        [
            {
                "disease_code": code,
                "disease": artifact["display"].get(code, code.upper()),
                "probability": round(float(prob), 4),
            }
            for code, prob in zip(encoder.classes_, probs)
        ],
        key=lambda row: row["probability"],
        reverse=True,
    )
    best = ranked[0]
    threshold = artifact["unknown_threshold"]
    if best["probability"] < threshold:
        primary = {"disease_code": "unknown", "disease": "Unknown", "probability": best["probability"]}
    else:
        primary = best
    return {"file": str(path), "primary_disease": primary, "ranked_diseases": ranked}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train")
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        train()
    else:
        print(json.dumps(predict(args.path), indent=2))


if __name__ == "__main__":
    main()
