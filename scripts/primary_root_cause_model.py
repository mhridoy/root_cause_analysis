from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder


MANIFEST = Path("/Users/moshiur/Downloads/authentic_reports/merged_manifest.csv")
TEXT_ROOT = Path("/Users/moshiur/Downloads/authentic_reports/merged_unique_text")
ARTIFACT_DIR = Path("/Users/moshiur/Downloads/archives/primary_root_cause_model")

GROUP_DISPLAY = {
    "neurodevelopmental": "Neurodevelopmental Differences",
    "anxiety_stress": "Anxiety / Stress Reactivity",
    "mood_emotional": "Mood / Emotional Vulnerability",
    "learning_cognitive": "Learning / Cognitive Processing",
}


def normalize_name(name: str) -> str:
    return name.lower().replace("__", "_")


def infer_primary_group(file_name: str) -> str | None:
    name = normalize_name(file_name)

    if name.startswith("zip_adhd_report_"):
        return "neurodevelopmental"
    if name.startswith("zip_asd_report_"):
        return "neurodevelopmental"
    if name.startswith("zip_dyslexia_report_"):
        return "learning_cognitive"
    if name.startswith("zip_depression_report_"):
        return "mood_emotional"
    if name.startswith("zip_gad_report_"):
        return "anxiety_stress"
    if name.startswith("zip_ocd_report_"):
        return "anxiety_stress"

    neuro_patterns = [
        "adhd",
        "autism",
        "asd",
        "gdd",
        "spcd",
        "neurodevelopmental",
        "executive_function",
        "executive function",
    ]
    learning_patterns = [
        "dyslexia",
        "learning_disorder",
        "learning disorder",
        "nvld",
        "borderline_intellectual",
        "borderline intellectual",
        "language_disorder",
        "language disorder",
        "cognitive_processing",
        "cognitive processing",
    ]
    mood_patterns = [
        "bipolar",
        "major_depressive",
        "major depressive",
        "severe_depression",
        "severe depression",
        "dmdd",
        "emotional_regulation_disorder",
        "emotional regulation disorder",
        "adjustment_disorder",
        "adjustment disorder",
    ]
    anxiety_patterns = [
        "gad",
        "anxiety",
        "panic",
        "school_avoidance",
        "school_refusal",
        "social_anxiety",
        "social anxiety",
        "separation_anxiety",
        "separation anxiety",
        "selective_mutism",
        "selective mutism",
        "ptsd",
        "post-traumatic",
        "post_traumatic",
        "ocd",
        "behavioral_inhibition",
        "behavioral inhibition",
    ]
    motor_patterns = [
        "speech_sound",
        "speech & language",
        "speech_language",
        "fine_motor",
        "fine motor",
        "gross_motor",
        "gross motor",
        "sensory_processing",
        "sensory processing",
        "dcd",
        "dyspraxia",
        "feeding_disorder",
        "feeding disorder",
    ]

    if any(p in name for p in learning_patterns):
        return "learning_cognitive"
    if any(p in name for p in mood_patterns):
        return "mood_emotional"
    if any(p in name for p in anxiety_patterns):
        return "anxiety_stress"
    if any(p in name for p in motor_patterns):
        return "neurodevelopmental"
    if any(p in name for p in neuro_patterns):
        return "neurodevelopmental"
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
        label = infer_primary_group(Path(text_file).name)
        if label is None:
            continue
        text = Path(text_file).read_text(errors="ignore")
        texts.append(text)
        labels.append(label)
        names.append(Path(text_file).name)
    return texts, labels, names


def build_features(texts: list[str], fit: bool, vectorizers: dict[str, TfidfVectorizer] | None = None):
    if fit:
        word = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        x_word = word.fit_transform(texts)
        x_char = char.fit_transform(texts)
        return hstack([x_word, x_char]), {"word": word, "char": char}
    assert vectorizers is not None
    x_word = vectorizers["word"].transform(texts)
    x_char = vectorizers["char"].transform(texts)
    return hstack([x_word, x_char]), vectorizers


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

    artifact = {
        "classifier": clf,
        "vectorizers": vecs,
        "encoder": encoder,
        "group_display": GROUP_DISPLAY,
    }
    joblib.dump(artifact, ARTIFACT_DIR / "primary_root_cause_classifier.joblib")

    with (ARTIFACT_DIR / "training_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["file_name", "true_label", "predicted_label"] + [f"prob_{cls}" for cls in encoder.classes_]
        writer.writerow(header)
        for name, true, pred, row in zip(names, labels, encoder.inverse_transform(preds), probs):
            writer.writerow([name, true, pred, *row.tolist()])

    label_counts: dict[str, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    report = {
        "sample_count": len(texts),
        "labels": encoder.classes_.tolist(),
        "label_counts": label_counts,
        "metrics": metrics,
    }
    (ARTIFACT_DIR / "model_report.json").write_text(json.dumps(report, indent=2))


def predict(path: Path) -> dict[str, object]:
    artifact = joblib.load(ARTIFACT_DIR / "primary_root_cause_classifier.joblib")
    from root_cause_model import read_report_text  # reuse file readers

    text = read_report_text(path)
    x, _ = build_features([text], fit=False, vectorizers=artifact["vectorizers"])
    probs = artifact["classifier"].predict_proba(x)[0]
    encoder: LabelEncoder = artifact["encoder"]
    best_idx = int(np.argmax(probs))
    best_code = encoder.classes_[best_idx]
    ranked = sorted(
        [
            {
                "group_code": code,
                "group": artifact["group_display"].get(code, code),
                "probability": round(float(prob), 4),
            }
            for code, prob in zip(encoder.classes_, probs)
        ],
        key=lambda row: row["probability"],
        reverse=True,
    )
    return {
        "file": str(path),
        "primary_root_cause": {
            "group_code": best_code,
            "group": artifact["group_display"].get(best_code, best_code),
            "probability": round(float(probs[best_idx]), 4),
        },
        "ranked_groups": ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train")
    p = sub.add_parser("predict")
    p.add_argument("path", type=Path)
    args = parser.parse_args()

    if args.command == "train":
        train()
    else:
        print(json.dumps(predict(args.path), indent=2))


if __name__ == "__main__":
    main()
