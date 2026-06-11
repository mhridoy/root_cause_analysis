# Final ML Model Summary

## Active Outputs

- Strongest current model artifacts: `/Users/moshiur/Downloads/archives/primary_root_cause_model`
- Secondary hybrid model artifacts: `/Users/moshiur/Downloads/archives/root_cause_model_merged`
- Active merged file corpus: `/Users/moshiur/Downloads/authentic_reports/merged_unique_reports`
- Active merged text corpus for training: `/Users/moshiur/Downloads/authentic_reports/merged_unique_text`
- Corpus manifest: `/Users/moshiur/Downloads/authentic_reports/merged_manifest.csv`
- Training script: `/Users/moshiur/Downloads/archives/root_cause_model.py`
- Stronger primary model script: `/Users/moshiur/Downloads/archives/primary_root_cause_model.py`
- Merge / cleanup script: `/Users/moshiur/Downloads/archives/merge_reports_corpus.py`

## Cleanup Performed

Removed:

- old 50-report-only analysis output
- old 50-report-only extracted text folder
- old 50-report-only model artifacts
- invalid CDC HTML files saved as `.pdf`
- `.DS_Store` junk files

Kept:

- original 50 DOCX reports
- original ZIP archive
- merged unique corpus
- current merged model artifacts
- public UPenn reference PDFs

## Final Corpus Used For ML

- Total scanned documents: `373`
- Valid documents kept: `371`
- ML training documents: `361`
- Public reference documents excluded from ML: `10`
- Invalid documents skipped: `2`
- Exact text duplicates removed: `0`

### ML Corpus Composition

- From ZIP reports: `311`
- From original 50 reports: `50`
- File types in ML corpus:
  - `.pdf`: `302`
  - `.docx`: `59`

## Model Accuracy

### Best Current Model: Primary Root Cause Classifier

This is the strongest current model for your use case because it uses cleaner supervision from explicit disease-family labels in filenames, instead of relying only on weak text-derived labels.

Metrics from `5-fold cross-validation`:

- Accuracy: `0.9556`
- Micro F1: `0.9556`
- Macro F1: `0.9535`

Training sample count:

- `360` labeled patient-level reports

Predicted primary groups:

- Neurodevelopmental Differences
- Anxiety / Stress Reactivity
- Mood / Emotional Vulnerability
- Learning / Cognitive Processing

Only one ML-eligible file remained excluded from this stricter model:

- `zip__Diagnosis_Free_Test_Report.txt`

because it does not describe a clear diagnostic family.

### Earlier Hybrid Weak-Label Model

This model is still useful as a secondary evidence layer, but it is not the strongest final answer.

Metrics from `5-fold cross-validation` on the merged corpus:

- Micro F1: `0.8705`
- Macro F1: `0.8312`

### What This Means

- The model is strong at predicting the **group labels it was trained against**.
- Those labels are **weak labels**, derived automatically from report text and section-aware keyword logic.
- This is **not the same as true clinical accuracy**.

The practical interpretation is:

- **Good** for coarse root-cause group suggestion
- **Useful** for ranking likely contributing factors
- **Not sufficient** to claim proven medical causation

## Groups The Model Predicts

- Anxiety / Stress Reactivity
- Learning / Cognitive Processing
- Mood / Emotional Vulnerability
- Neurodevelopmental Differences
- Sensory / Motor
- Sleep / Physiological Regulation
- Social / Environmental Pressure

`Trauma / Adversity` was too rare to train as a stable supervised label.

## How The Model Works

### 1. Text extraction

- `.docx` files are converted to text with `textutil`
- `.pdf` files are read with `PyPDF2`

### 2. Section-aware parsing

The script looks for report sections such as:

- reason for referral
- developmental history
- medical history
- family history
- educational history
- social / behavioral observations
- diagnosis
- formulation

### 3. Weak-label generation

Before training, each report is scored against root-cause groups using domain keywords and section weights.

Example:

- developmental and formulation sections are weighted more heavily than generic text
- repeated evidence for ADHD / ASD / executive dysfunction raises `Neurodevelopmental`
- repeated evidence for worry / panic / avoidance raises `Anxiety_Stress`

These scores produce the training labels.

### 4. Supervised classifier

The trained classifier uses:

- word-level TF-IDF features
- character n-gram TF-IDF features
- One-vs-Rest Logistic Regression

This is a better fit than deep learning here because:

- the corpus is still relatively small
- many reports are template-heavy
- classical linear text models are more stable and easier to interpret on this scale

### 5. Prediction output

For a new report, the model returns:

- predicted root-cause groups
- model probabilities
- heuristic keyword support
- a combined score that blends the classifier with section-aware heuristic evidence

## Why Public Reports Were Not Added To Training

The public UPenn PDFs were kept in the workspace as reference documents, but not used as training samples.

Reason:

- they are aggregate public research/needs-assessment reports
- they are not patient-level diagnostic reports
- adding them to training would reduce validity rather than improve it

They remain useful as:

- background domain reference
- topic vocabulary support
- provenance that some external public material is available in the workspace

## Current Strengths

- Much stronger than the original 50-report-only version
- Handles `.txt`, `.docx`, and `.pdf`
- Works across mixed report styles
- Produces ranked group outputs rather than a single blind label

## Current Limitations

- Labels are auto-derived, not clinician-validated
- Many ZIP reports appear highly templated
- High cross-validation scores may partly reflect template regularity
- The model predicts **group-level drivers**, not a medically proven root cause
- Public research PDFs were excluded because they are not patient-level assessments

## Best Accuracy Statement

The most accurate statement you can make today is:

> The strongest current model has strong internal performance for predicting a **primary root-cause group** from the merged patient-report corpus, with cross-validated Accuracy `0.9556` and Macro F1 `0.9535`, but those scores are still against programmatically inferred labels rather than clinician-confirmed ground truth.

## How To Use

Train the stronger primary model:

```bash
python3 '/Users/moshiur/Downloads/archives/primary_root_cause_model.py' train
```

Predict with the stronger primary model:

```bash
python3 '/Users/moshiur/Downloads/archives/primary_root_cause_model.py' predict \
  '/path/to/report.pdf'
```

Train the secondary hybrid model:

```bash
python3 '/Users/moshiur/Downloads/archives/root_cause_model.py' train \
  --corpus '/Users/moshiur/Downloads/authentic_reports/merged_unique_text' \
  --artifact-dir '/Users/moshiur/Downloads/archives/root_cause_model_merged'
```

Predict with the secondary hybrid model:

```bash
python3 '/Users/moshiur/Downloads/archives/root_cause_model.py' predict \
  --artifact-dir '/Users/moshiur/Downloads/archives/root_cause_model_merged' \
  '/path/to/report.pdf'
```

## Best Next Step

If you want a genuinely more accurate model, the next improvement should be:

1. remove or downweight highly templated reports
2. add clinician-reviewed `primary root cause` labels
3. separate `primary cause` from `secondary contributors`
4. retrain on reviewed labels

That will improve real-world accuracy more than switching to a bigger model.
