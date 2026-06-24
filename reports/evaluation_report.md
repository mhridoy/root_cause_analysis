# Evaluation Report — AI-Assisted Report Analyzer

**Corpus:** 361 de-duplicated patient-level reports.
**Split:** train 252 / validation 55 / test 54 (stratified by diagnosis).
**Model:** TF-IDF (uni+bigram) → multinomial softmax logistic regression (numpy).
All metrics below are on the **held-out test set** the model never saw during training.

---

## 1. Primary Diagnosis — STANDARD (deployed model, full report text)

Test accuracy **0.870**, macro-F1 **0.850**,
mean confidence 0.78, calibration error (ECE) 0.125.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ADHD | 1.00 | 0.80 | 0.89 | 10 |
| ASD | 0.88 | 0.78 | 0.82 | 9 |
| Depression | 1.00 | 0.89 | 0.94 | 9 |
| Dyslexia | 1.00 | 1.00 | 1.00 | 8 |
| GAD | 1.00 | 0.88 | 0.93 | 8 |
| OCD | 1.00 | 1.00 | 1.00 | 7 |
| Other / Complex | 0.25 | 0.67 | 0.36 | 3 |
| **Accuracy** | | | **0.870** | |
| **Macro F1** | | | **0.850** | |
| **Weighted F1** | | | **0.895** | |

## 2. Primary Diagnosis — LEAKAGE-CONTROLLED (diagnosis statements masked)

To test whether the model relies on described symptoms rather than just reading
the stated answer, every diagnosis-revealing phrase ("Primary Diagnosis: …",
DSM/ICD codes, the disease names themselves) was masked, then the model retrained
and re-evaluated from scratch.

Test accuracy **0.852**, macro-F1 **0.835**,
ECE 0.105.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ADHD | 1.00 | 0.80 | 0.89 | 10 |
| ASD | 0.86 | 0.67 | 0.75 | 9 |
| Depression | 1.00 | 0.89 | 0.94 | 9 |
| Dyslexia | 1.00 | 1.00 | 1.00 | 8 |
| GAD | 1.00 | 0.88 | 0.93 | 8 |
| OCD | 1.00 | 1.00 | 1.00 | 7 |
| Other / Complex | 0.22 | 0.67 | 0.33 | 3 |
| **Accuracy** | | | **0.852** | |
| **Macro F1** | | | **0.835** | |
| **Weighted F1** | | | **0.881** | |

**Interpretation:** performance drops only slightly when the answer is hidden
(0.870 → 0.852), which means the model
is genuinely learning from symptom/finding language, not only copying the stated
diagnosis. Both numbers clear the 85% target. The six well-supported classes
(ADHD, ASD, Depression, Dyslexia, GAD, OCD) score F1 0.82–1.00; the small,
heterogeneous *Other / Complex* bucket is the weak spot and is always flagged for
expert review.

## 3. Root-Cause / Contributing-Factor Group

Test accuracy **0.870**, macro-F1 **0.851**
across groups: Neurodevelopmental Differences, Anxiety / Stress Reactivity,
Mood / Emotional Vulnerability, Learning / Cognitive Processing.

## 4. Co-occurring Conditions (multi-label, supplementary)

Micro-precision 0.65,
micro-recall 0.70,
micro-F1 **0.68** (threshold 0.2).
This target uses *weak labels* (derived from filenames + lexical scan), so it is
treated as a supplementary hint and combined with a high-precision keyword
detector at inference time — not a primary output.

## 5. Confidence & Expert-Review policy

Per-prediction softmax confidence drives routing:

| Confidence | Band | Needs doctor review? |
|---|---|---|
| ≥ 0.70 | High | Only if High risk or *Other/Complex* |
| 0.50–0.70 | Moderate | **Yes** |
| < 0.50 | Low | **Yes** |

Calibration error (ECE ≈ 0.13) is reported so confidence numbers
can be interpreted honestly rather than taken as exact probabilities.

## 6. Honest limitations

- **Report-level text classification, not clinical diagnosis.** The system
  classifies what a written report describes. It does not assess a child.
- **Synthetic / curated corpus.** Many training reports are structured templates;
  real-world reports are messier, so live accuracy will be lower than the test
  numbers here. Re-evaluate on a sample of your own real reports before relying on it.
- **Weak co-occurring & root-cause labels.** These were derived heuristically and
  should be read as hints.
- **Six supported diagnosis families.** Anything else lands in *Other / Complex*
  and is flagged for review.
- **Not a medical device.** Output is AI-assisted decision support and must be
  confirmed by a qualified clinician.


---

## 7. Root-cause: clinical grounding + deep-learning comparison

**Problem fixed.** The previous root-cause label was literally
`argmax(keyword_counts)`, and the model was trained on TF-IDF of the same words
— so it just relearned keyword counting. That is why it "felt like text
extraction."

**New approach.** Labels are now *clinically grounded*: a diagnosis-informed
prior (independent of surface words) blended with sqrt-dampened evidence. At
inference the learned model is combined with the diagnosis-informed clinical
posterior, the two views are shown side by side, and the engine **abstains and
flags for review when they disagree or confidence is low**.

**Did "deeper" help? Measured, 3-fold CV (predicting grounded labels from text):**

| Model | CV accuracy | Macro-F1 | Calibration (ECE) |
|---|---|---|---|
| Linear softmax (shallow) | 0.925 | 0.823 | 0.125 |
| Deep MLP + LSA | 0.848 | 0.758 | 0.187 |
| Deep MLP + LSA + diagnosis prior | 0.858 | 0.764 | 0.156 |

**Honest conclusion:** on ~360 reports the **Linear softmax (shallow)** was the most
accurate and best-calibrated; the from-scratch deep MLP did **not** beat it
(deep nets need far more data). The deep network is included and inspectable
(`src/deep_models.py`, `train_rootcause.py`) and will be selected automatically
if it wins on a larger dataset. For a genuine deep-learning gain now, plug in
pretrained clinical-language-model embeddings (see README, optional) — that is
where deep learning pays off at this data scale, not a bigger net trained from
scratch.
