from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path("/Users/moshiur/Downloads/archives/50 Reports_txt")
OUTPUT = Path("/Users/moshiur/Downloads/archives/50 Reports_analysis")


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


FACTOR_KEYWORDS = {
    "Family History / Genetic Loading": [
        "family history",
        "first-degree relative",
        "genetic",
        "mother has",
        "father has",
        "runs in the family",
        "heredit",
        "mood disorders",
        "anxiety disorders",
        "adhd in the family",
    ],
    "Neurodevelopmental Differences": [
        "developmental delay",
        "neurodevelopment",
        "adhd",
        "autism",
        "asd",
        "gdd",
        "executive functioning",
        "inattention",
        "hyperactivity",
        "impulsivity",
        "social communication",
    ],
    "Learning / Cognitive Processing": [
        "learning disorder",
        "dyslexia",
        "dysgraphia",
        "working memory",
        "processing speed",
        "cognitive",
        "academic struggles",
        "intellectual",
        "borderline intellectual",
        "nonverbal learning",
    ],
    "Emotional Dysregulation / Mood Vulnerability": [
        "emotional dysregulation",
        "mood instability",
        "depression",
        "depressive",
        "bipolar",
        "dmdd",
        "irritability",
        "hopeless",
        "anhedonia",
        "low mood",
    ],
    "Anxiety / Stress Reactivity": [
        "anxiety",
        "panic",
        "worry",
        "rumination",
        "fear",
        "school avoidance",
        "school refusal",
        "social anxiety",
        "separation anxiety",
        "stress",
        "avoidance",
    ],
    "Sleep / Physiological Regulation": [
        "sleep",
        "insomnia",
        "night awakenings",
        "fatigue",
        "appetite",
        "physiological",
        "restless",
    ],
    "Social / Environmental Pressure": [
        "peer",
        "bullying",
        "social isolation",
        "family conflict",
        "communication patterns",
        "parental expectations",
        "academic pressure",
        "social comparison",
        "psychosocial stressors",
    ],
    "Trauma / Adversity": [
        "trauma",
        "ptsd",
        "abuse",
        "neglect",
        "adversity",
        "post-traumatic",
        "traumatic",
    ],
    "Sensory / Motor / Feeding": [
        "sensory",
        "dyspraxia",
        "dcd",
        "motor delay",
        "fine motor",
        "gross motor",
        "feeding",
        "coordination",
        "speech sound",
        "language disorder",
    ],
    "Behavioral Reinforcement / Addiction": [
        "behavioral addiction",
        "gaming",
        "oppositional",
        "odd",
        "conduct",
        "aggression",
        "defiance",
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


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", " ").replace("\u2028", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def clean_filename_label(name: str) -> str:
    base = name.replace(".txt", "")
    base = re.sub(r"^\d+\s*", "", base)
    base = base.replace("_", " ")
    base = re.sub(r"\b(report|assessment|psychological|diagnostic|evaluation|fnl|autorecovered|docx)\b", "", base, flags=re.I)
    base = re.sub(r"\s+", " ", base).strip(" -_.")
    return base or name


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
        candidate = heading_to_section(line)
        if candidate:
            current = candidate
            continue
        sections[current].append(line)
    return {key: "\n".join(value) for key, value in sections.items()}


def extract_diagnoses(text: str, fallback: str) -> list[str]:
    diagnoses: list[str] = []
    lines = [line.strip() for line in text.splitlines()]
    heading_markers = (
        "primary diagnosis",
        "secondary diagnosis",
        "associated diagnoses",
        "diagnosis",
        "diagnostic impression",
    )

    for i, line in enumerate(lines):
        lower = line.lower()
        if any(marker in lower for marker in heading_markers):
            inline = re.sub(r"^.*?:", "", line).strip(" .:-")
            if inline and inline.lower() not in {"primary diagnosis", "secondary diagnosis", "associated diagnoses", "diagnosis"}:
                diagnoses.append(inline)

            for follow in lines[i + 1 : i + 4]:
                candidate = follow.strip(" .:-")
                if not candidate:
                    continue
                if heading_to_section(candidate):
                    break
                if len(candidate) < 3:
                    continue
                if candidate.lower().startswith(("primary diagnosis", "secondary diagnosis", "diagnosis", "condition", "rule-outs")):
                    continue
                diagnoses.append(candidate)
                break

    if not diagnoses:
        for line in lines:
            s = line.strip()
            if re.match(r"^(adhd|attention-deficit|autism|major depressive|generalized anxiety|panic disorder|bipolar|dyslexia|ptsd|post-traumatic|selective mutism|language disorder|adjustment disorder|social \(pragmatic\)|developmental coordination disorder|speech sound disorder|separation anxiety)", s, flags=re.I):
                diagnoses.append(s.strip(" ."))

    if not diagnoses:
        diagnoses = [fallback]

    cleaned = []
    seen = set()
    for diagnosis in diagnoses:
        diagnosis = re.sub(r"\s+", " ", diagnosis).strip()
        if diagnosis.lower() in {
            "primary diagnosis",
            "secondary diagnosis",
            "associated diagnoses",
            "condition",
            "(rule-outs)",
            "(dsm-5-tr)",
            "(dsm-5-tr aligned)",
            "(dsm-5)",
            "considerations",
        }:
            continue
        if diagnosis.endswith(":"):
            continue
        if len(diagnosis.split()) > 20:
            continue
        if diagnosis.lower() not in seen:
            seen.add(diagnosis.lower())
            cleaned.append(diagnosis)
    if not cleaned:
        cleaned = [fallback]
    return cleaned


def score_factors(sections: dict[str, str]) -> dict[str, float]:
    scores = {factor: 0.0 for factor in FACTOR_KEYWORDS}
    for section, weight in SECTION_WEIGHTS.items():
        text = sections.get(section, "").lower()
        if not text:
            continue
        for factor, keywords in FACTOR_KEYWORDS.items():
            hits = sum(text.count(keyword.lower()) for keyword in keywords)
            if hits:
                scores[factor] += weight * hits

    full_text = sections.get("full_text", "").lower()
    for factor, keywords in FACTOR_KEYWORDS.items():
        scores[factor] += 0.15 * sum(full_text.count(keyword.lower()) for keyword in keywords)
    return scores


def top_factors(scores: dict[str, float], limit: int = 3) -> list[str]:
    ranked = [(factor, score) for factor, score in scores.items() if score > 0]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [factor for factor, _ in ranked[:limit]]


def build_corpus_record(path: Path) -> dict[str, object]:
    raw = normalize_text(path.read_text(errors="ignore"))
    sections = extract_sections(raw)
    fallback = clean_filename_label(path.name)
    diagnoses = extract_diagnoses(raw, fallback)
    scores = score_factors(sections)

    analysis_text = " ".join(
        sections.get(key, "")
        for key in ["reason", "developmental", "medical", "family", "educational", "social", "diagnosis", "formulation"]
    ).strip()
    if not analysis_text:
        analysis_text = raw

    record: dict[str, object] = {
        "file_name": path.name,
        "diagnoses": "; ".join(diagnoses),
        "primary_label": diagnoses[0],
        "top_factors": "; ".join(top_factors(scores)),
        "analysis_text": analysis_text,
    }
    for factor, score in scores.items():
        record[factor] = round(score, 2)
    return record


def cluster_labels(texts: list[str], k: int) -> tuple[np.ndarray, list[list[str]]]:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.85)
    matrix = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(matrix)

    terms = np.array(vectorizer.get_feature_names_out())
    top_terms: list[list[str]] = []
    for center in model.cluster_centers_:
        idx = center.argsort()[::-1][:10]
        top_terms.append(terms[idx].tolist())
    return labels, top_terms


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(ROOT.glob("*.txt"))
    rows = [build_corpus_record(path) for path in paths]
    df = pd.DataFrame(rows)

    k = min(6, max(3, math.ceil(len(df) / 10)))
    labels, cluster_terms = cluster_labels(df["analysis_text"].tolist(), k=k)
    df["cluster_id"] = labels
    df["cluster_keywords"] = df["cluster_id"].map(lambda cid: ", ".join(cluster_terms[cid][:6]))

    factor_columns = list(FACTOR_KEYWORDS.keys())
    global_factor_scores = df[factor_columns].sum().sort_values(ascending=False)
    diagnosis_counts = Counter()
    for value in df["diagnoses"]:
        for diagnosis in [v.strip() for v in value.split(";") if v.strip()]:
            diagnosis_counts[diagnosis] += 1

    df = df.drop(columns=["analysis_text"])
    df.to_csv(OUTPUT / "report_factor_analysis.csv", index=False)

    lines = []
    lines.append("# ML-Assisted Analysis of 50 Psychological Reports")
    lines.append("")
    lines.append("## Method")
    lines.append("- Source: 50 extracted `.docx` psychological assessment reports.")
    lines.append("- Pipeline: text extraction, section-aware keyword scoring, TF-IDF vectorization, KMeans clustering.")
    lines.append("- Important limitation: this estimates likely contributing drivers from narrative reports; it does not establish medical causality.")
    lines.append("")
    lines.append("## Global Factor Ranking")
    for factor, score in global_factor_scores.items():
        lines.append(f"- {factor}: {score:.2f}")
    lines.append("")
    lines.append("## Most Frequent Diagnoses / Labels")
    for diagnosis, count in diagnosis_counts.most_common(15):
        lines.append(f"- {diagnosis}: {count}")
    lines.append("")
    lines.append("## Cluster Summary")
    for cid in sorted(df['cluster_id'].unique()):
        subset = df[df["cluster_id"] == cid]
        labels_in_cluster = subset["primary_label"].value_counts().head(5)
        lines.append(f"### Cluster {cid}")
        lines.append(f"- Reports: {len(subset)}")
        lines.append(f"- Top keywords: {', '.join(cluster_terms[cid][:8])}")
        lines.append("- Common labels: " + ", ".join(f"{label} ({count})" for label, count in labels_in_cluster.items()))
        lines.append("")
    lines.append("## Per-Report Likely Root Drivers")
    for _, row in df.sort_values("file_name").iterrows():
        lines.append(f"### {row['file_name']}")
        lines.append(f"- Label: {row['diagnoses']}")
        lines.append(f"- Likely drivers: {row['top_factors']}")
        lines.append(f"- Cluster: {row['cluster_id']} ({row['cluster_keywords']})")
        lines.append("")

    (OUTPUT / "analysis_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
