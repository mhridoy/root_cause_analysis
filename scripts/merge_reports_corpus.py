from __future__ import annotations

import csv
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PyPDF2 import PdfReader


ROOT = Path("/Users/moshiur/Downloads/authentic_reports")
FIFTY_ROOT = ROOT / "50 Reports"
ZIP_ROOT = ROOT / "ADHD Disorder Assessment Reports" / "Disorder Assesment Reports"
MERGED_FILES = ROOT / "merged_unique_reports"
MERGED_TEXT = ROOT / "merged_unique_text"
MANIFEST = ROOT / "merged_manifest.csv"
SUMMARY = ROOT / "merged_summary.txt"


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", " ").replace("\u2028", " ").replace("\r", "\n")
    text = text.lower()
    text = re.sub(r"patient_\d+", "patient", text)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", "date", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def source_label(path: Path) -> str:
    if FIFTY_ROOT in path.parents:
        return "fifty"
    if ZIP_ROOT in path.parents:
        return "zip"
    return "public"


def include_in_ml(path: Path) -> bool:
    label = source_label(path)
    if label in {"fifty", "zip"}:
        return True
    return False


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            subprocess.run(
                ["/usr/bin/textutil", "-convert", "txt", str(path), "-output", tmp.name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return Path(tmp.name).read_text(errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def safe_name(label: str, path: Path) -> str:
    stem = re.sub(r"\s+", "_", path.stem.strip())
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    suffix = path.suffix.lower()
    return f"{label}__{stem}{suffix}"


def main() -> None:
    MERGED_FILES.mkdir(parents=True, exist_ok=True)
    MERGED_TEXT.mkdir(parents=True, exist_ok=True)

    for old in MERGED_FILES.iterdir():
        if old.is_file():
            old.unlink()
    for old in MERGED_TEXT.iterdir():
        if old.is_file():
            old.unlink()

    files = sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".docx", ".pdf"}
        and MERGED_FILES not in p.parents
        and MERGED_TEXT not in p.parents
    )

    rows: list[dict[str, str]] = []
    seen_hashes: dict[str, str] = {}
    counts = {
        "input_files": 0,
        "valid_files": 0,
        "ml_included": 0,
        "duplicates_removed": 0,
        "invalid_files": 0,
    }

    for path in files:
        counts["input_files"] += 1
        row = {
            "source_path": str(path),
            "source_label": source_label(path),
            "file_type": path.suffix.lower(),
            "ml_included": "false",
            "status": "skipped",
            "duplicate_of": "",
            "merged_file": "",
            "text_file": "",
            "notes": "",
        }

        try:
            raw_text = extract_text(path)
        except Exception as exc:
            counts["invalid_files"] += 1
            row["status"] = "invalid"
            row["notes"] = str(exc)
            rows.append(row)
            continue

        normalized = normalize_text(raw_text)
        if not normalized:
            counts["invalid_files"] += 1
            row["status"] = "invalid"
            row["notes"] = "empty extracted text"
            rows.append(row)
            continue

        text_hash = hashlib.sha256(normalized.encode()).hexdigest()
        if text_hash in seen_hashes:
            counts["duplicates_removed"] += 1
            row["status"] = "duplicate"
            row["duplicate_of"] = seen_hashes[text_hash]
            rows.append(row)
            continue

        counts["valid_files"] += 1
        merged_name = safe_name(row["source_label"], path)
        merged_path = MERGED_FILES / merged_name
        shutil.copy2(path, merged_path)
        seen_hashes[text_hash] = merged_name
        row["merged_file"] = str(merged_path)

        if include_in_ml(path):
            counts["ml_included"] += 1
            row["ml_included"] = "true"
            text_path = MERGED_TEXT / f"{merged_path.stem}.txt"
            text_path.write_text(raw_text)
            row["text_file"] = str(text_path)
            row["status"] = "kept_for_ml"
        else:
            row["status"] = "kept_reference_only"
            row["notes"] = "public aggregate/non-patient report excluded from ML corpus"

        rows.append(row)

    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_path",
                "source_label",
                "file_type",
                "ml_included",
                "status",
                "duplicate_of",
                "merged_file",
                "text_file",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = [
        f"Input files: {counts['input_files']}",
        f"Valid files kept: {counts['valid_files']}",
        f"ML corpus files: {counts['ml_included']}",
        f"Duplicates removed: {counts['duplicates_removed']}",
        f"Invalid files skipped: {counts['invalid_files']}",
        f"Merged files dir: {MERGED_FILES}",
        f"Merged text dir: {MERGED_TEXT}",
        f"Manifest: {MANIFEST}",
    ]
    SUMMARY.write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
