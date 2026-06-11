from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


DEFAULT_ROOT = Path("/Users/moshiur/Downloads/archives/50 Reports_txt")
DEFAULT_ARTIFACT_DIR = Path("/Users/moshiur/Downloads/archives/root_cause_model")

SECTION_PATTERNS = {
    "reason": r"(reason for referral|presenting concerns?)",
    "developmental": r"(developmental history|background information|developmental profile)",
    "medical": r"(medical history|health history)",
    "family": r"(family (and psychosocial )?history|family background|psychosocial history)",
    "educational": r"(educational history|academic history|school history)",
    "social": r"(social and emotional functioning|social functioning|behavioral observations|emotional and behavioral profile)",
    "diagnosis": r"(dsm-5(-tr)? diagnosis|diagnostic impression|diagnosis:|primary diagnos(es|is))",
    "formulation": r"(diagnostic formulation|clinical formulation|case conceptualization|integrated formulation)",
}

GROUP_KEYWORDS = {
    "Anxiety_Stress": [
        "anxiety", "panic", "worry", "rumination", "fear", "school avoidance", "school refusal",
        "social anxiety", "separation anxiety", "stress", "avoidance",
    ],
    "Neurodevelopmental": [
        "developmental delay", "neurodevelopment", "adhd", "autism", "asd", "gdd",
        "executive functioning", "inattention", "hyperactivity", "impulsivity", "social communication",
    ],
    "Sensory_Motor": [
        "sensory", "dyspraxia", "dcd", "motor delay", "fine motor", "gross motor",
        "feeding", "coordination", "speech sound", "language disorder",
    ],
    "Learning_Cognitive": [
        "learning disorder", "dyslexia", "dysgraphia", "working memory", "processing speed",
        "cognitive", "academic struggles", "intellectual", "borderline intellectual", "nonverbal learning",
    ],
    "Mood_Emotional": [
        "emotional dysregulation", "mood instability", "depression", "depressive", "bipolar",
        "dmdd", "irritability", "hopeless", "anhedonia", "low mood",
    ],
    "Social_Environmental": [
        "peer", "bullying", "social isolation", "family conflict", "communication patterns",
        "parental expectations", "academic pressure", "social comparison", "psychosocial stressors",
    ],
    "Trauma_Adversity": [
        "trauma", "ptsd", "abuse", "neglect", "adversity", "post-traumatic", "traumatic",
    ],
    "Sleep_Regulation": [
        "sleep", "insomnia", "night awakenings", "fatigue", "appetite", "restless",
    ],
}

SECTION_WEIGHTS = {
    "reason": 1.0,
    "developmental": 1.6,
    "medical": 1.3,
    "family": 1.8,
    "educational": 1.2,
    "social": 1.2,
    "diagnosis": 0.7,
    "formulation": 2.0,
}

GROUP_DISPLAY = {
    "Anxiety_Stress": "Anxiety / Stress Reactivity",
    "Neurodevelopmental": "Neurodevelopmental Differences",
    "Sensory_Motor": "Sensory / Motor / Communication",
    "Learning_Cognitive": "Learning / Cognitive Processing",
    "Mood_Emotional": "Mood / Emotional Vulnerability",
    "Social_Environmental": "Social / Environmental Pressure",
    "Trauma_Adversity": "Trauma / Adversity",
    "Sleep_Regulation": "Sleep / Physiological Regulation",
}


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", " ").replace("\u2028", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def heading_to_section(line: str) -> str | None:
    s = " ".join(line.lower().split())
    for section, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, s):
            return section
    return None


def extract_sections(text: str) -> dict[str, str]:
    current = "full_text"
    sections: dict[str, list[str]] = defaultdict(list)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        section = heading_to_section(line)
        if section:
            current = section
            continue
        sections[current].append(line)
    return {key: "\n".join(value) for key, value in sections.items()}


def score_groups(sections: dict[str, str]) -> dict[str, float]:
    scores = {group: 0.0 for group in GROUP_KEYWORDS}
    for section, weight in SECTION_WEIGHTS.items():
        text = sections.get(section, "").lower()
        if not text:
            continue
        for group, keywords in GROUP_KEYWORDS.items():
            hits = sum(text.count(keyword.lower()) for keyword in keywords)
            if hits:
                scores[group] += weight * hits
    full_text = sections.get("full_text", "").lower()
    for group, keywords in GROUP_KEYWORDS.items():
        scores[group] += 0.15 * sum(full_text.count(keyword.lower()) for keyword in keywords)
    return scores


def choose_labels(scores: dict[str, float]) -> list[str]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ranked = [(group, score) for group, score in ranked if score > 0]
    if not ranked:
        return ["Social_Environmental"]

    labels = [ranked[0][0]]
    if len(ranked) > 1 and ranked[1][1] >= 0.6 * ranked[0][1]:
        labels.append(ranked[1][0])
    if len(ranked) > 2 and ranked[2][1] >= 0.8 * ranked[1][1]:
        labels.append(ranked[2][0])
    return labels


def build_training_frame(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*.txt")):
        text = normalize_text(path.read_text(errors="ignore"))
        sections = extract_sections(text)
        scores = score_groups(sections)
        analysis_text = " ".join(
            sections.get(key, "")
            for key in ["reason", "developmental", "medical", "family", "educational", "social", "diagnosis", "formulation"]
        ).strip() or text
        labels = choose_labels(scores)
        row = {
            "file_name": path.name,
            "text": analysis_text,
            "labels": labels,
        }
        for group, score in scores.items():
            row[f"score_{group}"] = round(score, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def build_features(texts: list[str], fit: bool, vectorizers: dict[str, TfidfVectorizer] | None = None):
    if fit:
        word = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
        x_word = word.fit_transform(texts)
        x_char = char.fit_transform(texts)
        return hstack([x_word, x_char]), {"word": word, "char": char}

    assert vectorizers is not None
    x_word = vectorizers["word"].transform(texts)
    x_char = vectorizers["char"].transform(texts)
    return hstack([x_word, x_char]), vectorizers


def cross_validate(texts: list[str], y: np.ndarray) -> dict[str, float]:
    if len(texts) < 10:
        return {"micro_f1": float("nan"), "macro_f1": float("nan")}

    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    micro_scores = []
    macro_scores = []
    for train_idx, test_idx in splitter.split(texts):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        x_train, vecs = build_features(train_texts, fit=True)
        x_test, _ = build_features(test_texts, fit=False, vectorizers=vecs)
        clf = OneVsRestClassifier(LogisticRegression(max_iter=4000, class_weight="balanced"))
        clf.fit(x_train, y[train_idx])
        probs = clf.predict_proba(x_test)
        preds = ensure_one_label((probs >= 0.45).astype(int), probs)
        micro_scores.append(f1_score(y[test_idx], preds, average="micro", zero_division=0))
        macro_scores.append(f1_score(y[test_idx], preds, average="macro", zero_division=0))
    return {"micro_f1": float(np.mean(micro_scores)), "macro_f1": float(np.mean(macro_scores))}


def ensure_one_label(preds: np.ndarray, probs: np.ndarray) -> np.ndarray:
    fixed = preds.copy()
    for i in range(fixed.shape[0]):
        if not fixed[i].any():
            fixed[i, int(np.argmax(probs[i]))] = 1
    return fixed


def train_model(root: Path, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    df = build_training_frame(root)

    label_counts = defaultdict(int)
    for labels in df["labels"]:
        for label in labels:
            label_counts[label] += 1
    trainable_labels = sorted([label for label, count in label_counts.items() if count >= 4])
    df["train_labels"] = df["labels"].apply(lambda labels: [label for label in labels if label in trainable_labels] or [max(labels, key=lambda label: label_counts[label])])

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["train_labels"])
    texts = df["text"].tolist()
    metrics = cross_validate(texts, y)

    x_all, vectorizers = build_features(texts, fit=True)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=4000, class_weight="balanced"))
    clf.fit(x_all, y)
    probs = clf.predict_proba(x_all)
    preds = ensure_one_label((probs >= 0.45).astype(int), probs)

    artifact = {
        "classifier": clf,
        "vectorizers": vectorizers,
        "mlb": mlb,
        "group_display": GROUP_DISPLAY,
        "threshold": 0.45,
        "trainable_labels": trainable_labels,
        "label_counts": dict(label_counts),
    }
    joblib.dump(artifact, artifact_dir / "root_cause_classifier.joblib")

    out = df[["file_name"]].copy()
    out["weak_labels"] = ["; ".join(labels) for labels in df["labels"]]
    out["train_labels"] = ["; ".join(labels) for labels in df["train_labels"]]
    out["predicted_labels"] = ["; ".join(mlb.classes_[row.astype(bool)]) for row in preds]
    for idx, cls in enumerate(mlb.classes_):
        out[f"prob_{cls}"] = probs[:, idx]
    out.to_csv(artifact_dir / "training_predictions.csv", index=False)

    report = {
        "sample_count": int(len(df)),
        "label_count": int(len(mlb.classes_)),
        "labels": list(mlb.classes_),
        "raw_label_counts": dict(label_counts),
        "metrics": metrics,
    }
    (artifact_dir / "model_report.json").write_text(json.dumps(report, indent=2))


def read_report_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return normalize_text(path.read_text(errors="ignore"))
    if suffix == ".docx":
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            subprocess.run(
                ["/usr/bin/textutil", "-convert", "txt", str(path), "-output", tmp.name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return normalize_text(Path(tmp.name).read_text(errors="ignore"))
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return normalize_text("\n".join((page.extract_text() or "") for page in reader.pages))
    raise ValueError(f"Unsupported file type: {path.suffix}")


def predict(path: Path, artifact_dir: Path) -> dict[str, object]:
    artifact = joblib.load(artifact_dir / "root_cause_classifier.joblib")
    text = read_report_text(path)
    sections = extract_sections(text)
    analysis_text = " ".join(
        sections.get(key, "")
        for key in ["reason", "developmental", "medical", "family", "educational", "social", "diagnosis", "formulation"]
    ).strip() or text

    x, _ = build_features([analysis_text], fit=False, vectorizers=artifact["vectorizers"])
    probs = artifact["classifier"].predict_proba(x)[0]
    threshold = artifact["threshold"]
    classes = artifact["mlb"].classes_
    keyword_scores = score_groups(extract_sections(text))
    max_keyword = max(keyword_scores.values()) if keyword_scores else 0.0

    blended = []
    for i, group_code in enumerate(classes):
        heuristic = keyword_scores.get(group_code, 0.0)
        heuristic_norm = (heuristic / max_keyword) if max_keyword > 0 else 0.0
        combined = 0.65 * float(probs[i]) + 0.35 * heuristic_norm
        blended.append((group_code, float(probs[i]), heuristic, combined))

    blended.sort(key=lambda item: item[3], reverse=True)
    selected = [row for row in blended if row[3] >= 0.45]
    if not selected:
        selected = [blended[0]]

    groups = [
        {
            "group_code": group_code,
            "group": artifact["group_display"].get(group_code, group_code),
            "model_probability": round(prob, 4),
            "heuristic_score": round(float(heuristic), 3),
            "combined_score": round(combined, 4),
        }
        for group_code, prob, heuristic, combined in selected
    ]

    ranked_keywords = sorted(keyword_scores.items(), key=lambda item: item[1], reverse=True)
    return {
        "file": str(path),
        "predicted_groups": groups,
        "keyword_support": [
            {"group": GROUP_DISPLAY[group], "score": round(score, 3)}
            for group, score in ranked_keywords[:5]
            if score > 0
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--corpus", type=Path, default=DEFAULT_ROOT)
    train_parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("path", type=Path)
    predict_parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)

    args = parser.parse_args()
    if args.command == "train":
        train_model(args.corpus, args.artifact_dir)
    else:
        result = predict(args.path, args.artifact_dir)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
