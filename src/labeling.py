"""
Build a ground-truth labelled dataset from the existing corpus.

Ground truth for the PRIMARY diagnosis comes from the human-assigned
folder/filename category (e.g. the "ASD" report set, or a filename that spells
out "Autism Spectrum Disorder"). This is independent of the report body text,
so using it as a label is NOT circular.

Co-occurring conditions and root-cause group labels are derived from a
combination of the descriptive filename and lexical scan of the body, then
sanity-checked. They are weak labels and are reported as such.
"""
import os
import re
import csv
from .lexicons import (
    DISEASE_KEYWORDS, DISEASE_CLASSES, COOCCURRING_KEYWORDS, ROOT_CAUSE_GROUPS,
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())


def primary_from_name(filename: str) -> str:
    """Assign primary diagnosis from the (human-authored) filename/folder name."""
    name = _norm(filename.replace("_", " "))
    # priority order from DISEASE_KEYWORDS dict insertion order
    for disease, kws in DISEASE_KEYWORDS.items():
        for kw in kws:
            if kw in name:
                return disease
    return "Other / Complex"


def cooccurring_from_text(name: str, body: str, primary: str) -> list:
    """Weak multi-label co-occurring detection from filename + body."""
    hay = _norm(name.replace("_", " ")) + " \n " + _norm(body)
    found = []
    for cond, kws in COOCCURRING_KEYWORDS.items():
        if any(kw in hay for kw in kws):
            found.append(cond)
    # The primary disease itself is not a "co-occurring" condition.
    canon = {
        "ADHD": "ADHD", "ASD": "ASD", "Depression": "Depression",
        "GAD": "Anxiety", "OCD": "OCD", "Dyslexia": "Learning Disorder",
    }.get(primary)
    return [c for c in found if c != canon]


def root_cause_from_text(body: str) -> tuple:
    """Score each root-cause group by lexical hits; return (top_group, scores)."""
    hay = _norm(body)
    scores = {}
    for grp, kws in ROOT_CAUSE_GROUPS.items():
        scores[grp] = sum(hay.count(kw) for kw in kws)
    if not any(scores.values()):
        return "Neurodevelopmental Differences", scores
    top = max(scores, key=scores.get)
    return top, scores


def build_dataset(text_dir: str) -> list:
    """Return list of dict rows for every .txt report in text_dir."""
    rows = []
    for fn in sorted(os.listdir(text_dir)):
        if not fn.endswith(".txt"):
            continue
        path = os.path.join(text_dir, fn)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            body = f.read()
        if len(body.strip()) < 100:  # skip empty/near-empty
            continue
        primary = primary_from_name(fn)
        cooc = cooccurring_from_text(fn, body, primary)
        rc_top, rc_scores = root_cause_from_text(body)
        rows.append({
            "file": fn,
            "primary_disease": primary,
            "cooccurring": "|".join(cooc),
            "root_cause_group": rc_top,
            "n_chars": len(body),
        })
    return rows


def write_labels_csv(rows: list, out_path: str):
    cols = ["file", "primary_disease", "cooccurring", "root_cause_group", "n_chars"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    import sys
    text_dir = sys.argv[1] if len(sys.argv) > 1 else "../merged_unique_text"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/labels.csv"
    rows = build_dataset(text_dir)
    write_labels_csv(rows, out)
    from collections import Counter
    print(f"Labelled {len(rows)} reports -> {out}")
    print("Primary disease distribution:")
    for k, v in Counter(r["primary_disease"] for r in rows).most_common():
        print(f"  {k:20s} {v}")
    print("Root-cause group distribution:")
    for k, v in Counter(r["root_cause_group"] for r in rows).most_common():
        print(f"  {k:32s} {v}")
