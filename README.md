# AI-Assisted Autism / Neurodevelopmental Report Analyzer

A local, privacy-first system that **learns from existing autism-related
reports** and **analyzes new uploaded reports**, producing an explainable,
structured result with a confidence score and a clear "needs doctor review"
flag.

> ⚠️ **Safety first.** This is **AI-assisted decision support, not a medical
> diagnosis.** It classifies what a written report *describes*; it does not
> assess a child. Every result must be confirmed by a qualified clinician,
> especially when confidence is low or the case is complex.

---

## What it does (the four phases you asked for)

**Phase 1 — Data & preparation.** Extracts text from PDF / DOCX / scanned
images (OCR) / text files, cleans it, derives ground-truth labels from the
human-assigned report categories (diagnosis, co-occurring conditions,
root-cause group), and **anonymizes personal information** on every upload.

**Phase 2 — Model.** Trains an NLP model (TF-IDF → multinomial logistic
regression) that classifies the primary diagnosis, root-cause group and
co-occurring conditions, and produces **explainable** output (the exact terms
and sentences that drove the prediction).

**Phase 3 — Testing & evaluation.** Stratified train/validation/test split with
accuracy, precision, recall, F1 and calibration — plus a **leakage-controlled**
score that hides the stated diagnosis to prove the model reads symptoms, not
just the answer. Low-confidence cases are auto-flagged **"Needs expert review."**

**Phase 4 — Web app.** Upload a report → automatic extraction & analysis →
predicted condition, probable root cause, confidence score, highlighted
evidence → download/save the result → manage previous reports securely.

## Results (held-out test set)

| Target | Accuracy | Macro-F1 |
|---|---|---|
| Primary diagnosis (standard) | **0.87** | 0.85 |
| Primary diagnosis (leakage-controlled) | **0.85** | 0.84 |
| Root-cause group | 0.87 | 0.85 |

Both diagnosis numbers clear the **85% target**. The six well-supported classes
(ADHD, ASD, Depression, Dyslexia, GAD, OCD) score F1 0.82–1.00. Full detail and
honest limitations: [`reports/evaluation_report.md`](reports/evaluation_report.md).

---

## Quick start for collaborators

This repository now includes a **pretrained demo model bundle** in `models/`.
That means collaborators can launch the web app immediately without retraining.

### 1. Clone the repo

```bash
git clone https://github.com/mhridoy/root_cause_analysis.git
cd root_cause_analysis
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the web app

```bash
python3 webapp/app.py
```

Then open:

```text
http://127.0.0.1:8000
```

### 5. Upload a test report

Supported upload types:

- `.pdf`
- `.docx`
- `.txt`
- `.md`
- images/scans: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`

The app will return:

- predicted primary condition / diagnosis
- co-occurring conditions
- key symptoms found
- probable root-cause / contributing factors
- evidence sentences
- confidence score
- risk level
- recommendation
- `Needs Doctor Review`

## Retraining (optional)

Collaborators do **not** need to retrain to test the web app.

Retraining is only needed if they want to rebuild the model from a local corpus.

```bash
python3 train.py --text_dir /path/to/text_reports
```

This regenerates:

- `models/bundle.json`
- `models/metrics.json`

The training corpus itself is **not included** in this repo.

### Structured output (exactly the requested format)

```
Child Condition / Diagnosis
Co-occurring Diseases or Disorders
Key Symptoms Found
Probable Root Cause / Contributing Factors
Evidence from Report          (highlighted in the report text)
Confidence Score              (with calibration-aware band)
Risk Level                    (Low / Moderate / High)
Recommendation
Needs Doctor Review           (Yes / No)
```

## Command-line testing

Analyze a single file instead of using the web app:

```bash
python3 -m src.analyze /path/to/report.pdf
```

---

## Project layout

```
asd_report_analyzer/
├── train.py               # Phase 1–3: build labels, train, evaluate, save
├── make_eval_report.py    # regenerate reports/evaluation_report.md
├── requirements.txt
├── src/
│   ├── lexicons.py        # clinical vocabulary + label taxonomy + leakage patterns
│   ├── labeling.py        # ground-truth label derivation
│   ├── textproc.py        # tokenizer + numpy TF-IDF + leakage masking
│   ├── model.py           # softmax & one-vs-rest classifiers (numpy)
│   ├── metrics.py         # split + precision/recall/F1 + calibration (numpy)
│   ├── extract.py         # PDF/DOCX/image/text extraction (+ OCR fallback)
│   ├── anonymize.py       # PII scrubbing
│   └── analyze.py         # inference → structured explainable result
├── webapp/
│   └── app.py             # local web app (stdlib http.server + SQLite)
├── models/                # pretrained bundle.json + metrics.json
├── data/                  # labels.csv
└── reports/               # evaluation_report.md
```

## Privacy & data handling

- Uploaded originals are **deleted immediately** after text extraction; only the
  **anonymized** text and the result are stored (locally, in `webapp/reports.db`).
- Everything runs **on your machine** — no data leaves the computer.
- Automated anonymization is conservative but **not perfect**; confirm before
  sharing any output externally.

## Notes & tuning

- If your app folder is on a network share, point the DB to local disk:
  `REPORTS_DB=/path/reports.db REPORTS_TMP=/path/tmp python3 webapp/app.py`.
- Confidence thresholds and risk lexicons live in `src/analyze.py` and
  `src/lexicons.py` and are easy to adjust.
- The included pretrained model is intended for **testing / demo / collaboration**
  and is based on the project’s internal prepared corpus.
- **Before production use,** re-evaluate on a sample of *your own real* reports
  — the bundled corpus is partly templated, so live accuracy will differ.
