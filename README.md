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

## Quick start

```bash
cd asd_report_analyzer

# 1) (optional) install extras for PDF/DOCX/OCR — plain text needs nothing but numpy
pip install -r requirements.txt

# 2) (re)train from the corpus  — produces models/bundle.json + metrics
python3 train.py --text_dir ../merged_unique_text

# 3) launch the web app  (no Flask needed — uses Python's built-in server)
python3 webapp/app.py
#    then open http://127.0.0.1:8000
```

Upload a child's report on the page, and you'll get the structured result below.

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

Analyze a single file from the command line instead of the web app:

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
├── models/                # bundle.json (trained), metrics.json
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
- **Before production use,** re-evaluate on a sample of *your own real* reports
  — the bundled corpus is partly templated, so live accuracy will differ.
```
