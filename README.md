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

Root-cause analysis uses two independent views: a cross-validated text model
and a diagnosis-informed clinical posterior. Their probabilities are blended,
and disagreement or a narrow margin causes the system to abstain and require
expert review instead of presenting a confident guess.

**Phase 3 — Testing & evaluation.** Stratified train/validation/test split with
accuracy, precision, recall, F1 and calibration — plus a **leakage-controlled**
score that hides the stated diagnosis to prove the model reads symptoms, not
just the answer. Low-confidence cases are auto-flagged **"Needs expert review."**

**Phase 4 — Web app.** Upload a report → automatic extraction & analysis →
predicted condition, probable root cause, confidence score, highlighted
evidence → download/save the result → manage previous reports securely.

## Results (held-out test set)

| Target | Accuracy | Macro-F1 | Eval |
|---|---|---|---|
| Primary diagnosis (standard) | **0.93** | 0.87 | held-out test |
| Primary diagnosis (leakage-controlled) | **0.93** | 0.87 | held-out test |
| Root-cause group (clinically grounded) | **0.92** | 0.82 | cross-validation |

Both diagnosis numbers clear the **85% target** with margin. The six
well-supported classes (ADHD, ASD, Depression, Dyslexia, GAD, OCD) score
F1 0.90–1.00. Confidence scores are **temperature-calibrated** on the
validation split, so the "Needs doctor review" thresholds act on honest
probabilities. Model selection was done by cross-validation on train+val only —
the test set was touched once, for the numbers above.

**Reads symptoms, not just labels.** Two robustness fixes mean a realistic,
paraphrased report that *never names the disorder* is still recognized:

- **Negation-aware throughout** — ruled-out findings ("**no** history of restricted
  interests", "**no** suicidal thoughts, self-harm, abuse, or neglect", "did **not**
  show hyperactivity") no longer count as present. This governs the diagnosis
  features **and** the risk, symptom, and co-occurring detectors, so the system no
  longer raises a false High-risk flag from a denied "no suicidal thoughts…" line,
  no longer lists denied symptoms, and no longer over-lists co-occurring
  conditions. Co-occurring is now tiered (Likely / Possible / explicitly ruled
  out), and risk is High only when danger signals are actually asserted.
- **Symptom-signature clinical prior** — when the learned model is unsure, the
  prediction is blended with negation-aware DSM-style symptom clusters. On a
  textbook OCD case written without the word "OCD", this moved OCD from **5.7%
  (last) to ~75% (first)** — with **no change to held-out test accuracy**, because
  the prior only engages when the model is uncertain (it stays off for the
  confident templated cases). Character n-grams additionally recover
  morphological/out-of-vocabulary terms (e.g. *ritualized, contamination*).
- **Reads the stated diagnosis (incl. ICD-10) and conditions beyond the six
  classes** — most reports record their conclusion ("Diagnosis: … (F70)"), so a
  high-precision reader (`src/diagnosis_extract.py`) surfaces it as the primary
  label with an explanation. It names **Speech/Language Disorder, Intellectual
  Disability, PTSD, Panic, Bipolar**, etc. (flagged for review as outside the
  trained model), keeps a co-occurring condition from becoming the primary
  ("ASD with co-occurring ADHD" → ASD primary), and returns **"No diagnosis"**
  for a typical-development report instead of forcing a class. **Backward
  negation** ("suicidal ideation *was explicitly denied*", "tics *were not
  observed*") prevents false risk flags and false co-occurring entries.

**Root cause was rebuilt** so it is no longer keyword-counting: labels are now
clinically grounded (a diagnosis-informed prior blended with text evidence) and
the engine shows a learned view *and* a clinical view, abstaining when they
disagree. A from-scratch **deep MLP was cross-validated head-to-head against the
linear model and lost** (0.85 vs **0.92**) — deep nets need far more data, so the
linear model ships and the deep net is kept for larger future datasets. Full
detail and honest limitations: [`reports/evaluation_report.md`](reports/evaluation_report.md)
(see §7 for the deep-vs-linear comparison).

### Live web app features

The Vercel app (`app.py`) now includes: drag-and-drop upload with **batch
analysis** (up to 5 files), a paste-text mode with a built-in synthetic sample,
a **probability chart** across all trained conditions, the **anonymized report
with evidence sentences and model terms highlighted inline**, a decision-margin
indicator, per-report **JSON download** and **print / save-as-PDF**, and a JSON
API (`POST /api/analyze`, `GET /api/health`, `GET /api/model-info`).

Non-clinical uploads are rejected before diagnosis using clinical vocabulary,
symptom anchors, model confidence, and trained-vocabulary coverage.

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

For scanned/image reports (OCR, local use only):

```bash
pip install -r requirements-ocr.txt   # + install the tesseract OS package
```

### 4. Launch the web app

```bash
python3 webapp/app.py
```

Then open:

```text
http://127.0.0.1:8000
```

For the stateless Vercel-compatible version instead, run:

```bash
python3 app.py
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

Dataset metadata used for training/evaluation labels is included at
`data/labels.csv` so collaborators can inspect the prepared dataset table.

## Vercel deployment

This repo now includes a Vercel entrypoint:

- `app.py` -> stateless Flask app for Vercel
- `vercel.json` -> routes all traffic to the Python app

Deploy steps:

```bash
npm i -g vercel
vercel login
vercel
vercel --prod
```

Important Vercel constraints:

- the live deployment is **stateless**; it does not keep the SQLite report history
- OCR/image uploads are **disabled** in Vercel; use `.txt`, `.md`, `.docx`, or native-text `.pdf`
- Vercel function request payloads are limited, so keep uploads small
- the richer local app remains `webapp/app.py`

## Retraining (optional)

Collaborators do **not** need to retrain to test the web app.

Retraining is only needed if they want to rebuild the model from a local corpus.

```bash
python3 train.py --text_dir /path/to/text_reports
```

This regenerates `models/bundle.json` and `models/metrics.json` (diagnosis,
co-occurring, legacy root-cause).

To rebuild the **clinically-grounded root-cause model** and re-run the
deep-vs-linear cross-validation:

```bash
python3 train_rootcause.py --text_dir /path/to/text_reports
```

This regenerates `models/rootcause_deep.json` (the deployed winner) and
`models/rootcause_cv.json` (the CV comparison table). The training corpus itself
is **not included** in this repo.

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

Verify the system is genuinely analyzing (not returning a canned answer):

```bash
python3 test_system.py      # 45 checks, expect "45 passed, 0 failed"
```

This confirms non-clinical files (invoice, recipe, resume, news) are **refused**,
the six diagnosis families get **correct, distinct** labels, out-of-scope reports
(Bipolar/PTSD/Panic) route to *Other / Complex* and are flagged, and the
root-cause engine is grounded + abstains on uncertain cases.

---

## Project layout

```
asd_report_analyzer/
├── app.py                 # Vercel/Flask stateless web app (+ JSON API)
├── vercel.json            # Vercel routing
├── webapp/app.py          # richer LOCAL web app (stdlib http.server + SQLite)
├── train.py               # Phase 1–3: build labels, train, evaluate, save
├── train_rootcause.py     # grounded root-cause: deep-vs-linear CV + train winner
├── make_eval_report.py    # regenerate reports/evaluation_report.md
├── test_system.py         # self-test: refuses junk, verifies real analysis (45 checks)
├── requirements.txt       # lean core (numpy, Flask, pdfplumber, python-docx)
├── requirements-ocr.txt   # optional OCR extras (PyMuPDF, pytesseract, Pillow)
├── src/
│   ├── lexicons.py        # clinical vocabulary + label taxonomy + leakage patterns
│   ├── labeling.py        # ground-truth label derivation
│   ├── textproc.py        # tokenizer + numpy TF-IDF (word + char n-grams)
│   ├── model.py           # softmax / one-vs-rest classifiers + temperature calibration
│   ├── metrics.py         # split + precision/recall/F1 + calibration (numpy)
│   ├── extract.py         # PDF/DOCX/image/text extraction (page-bounded; OCR optional)
│   ├── anonymize.py       # PII scrubbing
│   ├── domain_gate.py     # rejects non-clinical uploads
│   ├── diagnosis_priors.py # negation-aware symptom-signature prior for diagnosis
│   ├── clinical_priors.py # diagnosis-informed root-cause grounding (fixes circular labels)
│   ├── deep_models.py     # from-scratch LSA + deep MLP (numpy) + calibration
│   ├── rootcause.py       # calibrated blended root-cause engine + abstention
│   ├── embeddings_optional.py # optional pretrained-transformer embeddings (transfer learning)
│   └── analyze.py         # inference → structured explainable result
├── scripts/               # standalone corpus-prep / earlier model scripts
├── models/                # bundle.json, metrics.json, rootcause_deep.json, rootcause_cv.json
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
