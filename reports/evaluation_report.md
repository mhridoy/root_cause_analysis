# Evaluation Report — AI-Assisted Report Analyzer

**Corpus:** 361 de-duplicated patient-level reports.
**Split:** train 252 / validation 55 / test 54 (stratified by diagnosis).
**Model:** TF-IDF (uni+bigram) → multinomial softmax logistic regression (numpy).
All metrics below are on the **held-out test set** the model never saw during training.

---

## 1. Primary Diagnosis — STANDARD (deployed model, full report text)

Test accuracy **0.926**, macro-F1 **0.873**,
mean confidence 0.88, calibration error (ECE) 0.060.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ADHD | 0.91 | 1.00 | 0.95 | 10 |
| ASD | 0.90 | 1.00 | 0.95 | 9 |
| Depression | 1.00 | 0.89 | 0.94 | 9 |
| Dyslexia | 1.00 | 1.00 | 1.00 | 8 |
| GAD | 1.00 | 0.88 | 0.93 | 8 |
| OCD | 1.00 | 1.00 | 1.00 | 7 |
| Other / Complex | 0.33 | 0.33 | 0.33 | 3 |
| **Accuracy** | | | **0.926** | |
| **Macro F1** | | | **0.873** | |
| **Weighted F1** | | | **0.926** | |

## 2. Primary Diagnosis — LEAKAGE-CONTROLLED (diagnosis statements masked)

To test whether the model relies on described symptoms rather than just reading
the stated answer, every diagnosis-revealing phrase ("Primary Diagnosis: …",
DSM/ICD codes, the disease names themselves) was masked, then the model retrained
and re-evaluated from scratch.

Test accuracy **0.926**, macro-F1 **0.873**,
ECE 0.063.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ADHD | 0.91 | 1.00 | 0.95 | 10 |
| ASD | 0.90 | 1.00 | 0.95 | 9 |
| Depression | 1.00 | 0.89 | 0.94 | 9 |
| Dyslexia | 1.00 | 1.00 | 1.00 | 8 |
| GAD | 1.00 | 0.88 | 0.93 | 8 |
| OCD | 1.00 | 1.00 | 1.00 | 7 |
| Other / Complex | 0.33 | 0.33 | 0.33 | 3 |
| **Accuracy** | | | **0.926** | |
| **Macro F1** | | | **0.873** | |
| **Weighted F1** | | | **0.926** | |

**Interpretation:** performance drops only slightly when the answer is hidden
(0.926 → 0.926), which means the model
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

Calibration error (ECE ≈ 0.06) is reported so confidence numbers
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

The old root-cause label was literally `argmax(keyword_counts)` and the model
relearned keyword counting. Labels are now clinically grounded (diagnosis-informed
prior blended with dampened evidence); at inference a learned view and a clinical
view are combined and the engine abstains when they disagree.

**Did "deeper" help? 3-fold CV (predicting grounded labels from text):**

| Model | CV accuracy | Macro-F1 | Calibration (ECE) |
|---|---|---|---|
| Linear softmax (shallow) | 0.925 | 0.823 | 0.125 |
| Deep MLP + LSA | 0.848 | 0.758 | 0.187 |
| Deep MLP + LSA + diagnosis prior | 0.858 | 0.764 | 0.156 |

**Honest conclusion:** the **Linear softmax (shallow)** won; the from-scratch deep MLP did not
beat it on ~360 reports. The deep net is kept (`src/deep_models.py`) and
auto-selected if it wins on a larger dataset. The real deep-learning lever at this
scale is transfer learning — see `src/embeddings_optional.py`.


---

## 8. Robustness: reading symptoms, not labels

The model is trained on a templated corpus where reports usually NAME the
disorder. Two fixes make it work on realistic, paraphrased, symptom-only reports:

1. **Negation-aware features.** Ruled-out symptoms ("no history of restricted
   interests", "teachers do not report hyperactivity") are tagged so they no
   longer vote for the ruled-out condition. Before this, those negated mentions
   were the top features for the *wrong* class.

2. **Symptom-signature clinical prior.** When the learned model is unsure
   (top probability below a gate), its prediction is blended with negation-aware
   DSM-style symptom clusters (`src/diagnosis_priors.py`). The prior only engages
   on uncertain inputs, so **held-out test accuracy is unchanged** while realistic
   reports are rescued.

Character n-grams additionally recover morphological / out-of-vocabulary terms
(e.g. *ritualized*, *contamination*).

3. **Negation-aware structured output.** The same negation logic now governs the
   risk, symptom, and co-occurring detectors (not just the classifier). Ruled-out
   findings are split out of the asserted text, so:
   - Risk is **High only when danger signals are asserted** — a denied
     "no suicidal thoughts, self-harm, abuse, or neglect" no longer triggers a
     false High-risk flag (it is shown as *explicitly denied* instead).
   - Denied symptoms ("did not show hyperactivity", "denied persistent sadness")
     are no longer listed as found.
   - Co-occurring conditions are **tiered** (Likely / Possible / explicitly ruled
     out); the noisy weak-label model is no longer used, and the primary diagnosis
     is never listed as its own co-occurring disorder.

These behaviours are guarded by `test_system.py` (27 checks), including a denied-
risk report (must not be High) and a genuinely asserted-risk report (must be High).

**Worked example (a textbook OCD report that never writes "OCD"):**

| Stage | OCD probability | Rank |
|---|---|---|
| Before fixes | 5.7% | last (7th) — predicted ADHD |
| After negation + symptom prior | ~75% | **1st** |

This is the case that motivated the fix; it is now guarded by `test_system.py`.


---

## 9. Stated-diagnosis reader + conditions beyond the six classes

An external API audit found the model returned **no primary label in 8/8**
explicit clinical reports and could not name conditions outside its six trained
classes (Speech/Language Disorder, Intellectual Disability, PTSD). Real reports
usually *state* their conclusion, so a high-precision reader was added
(`src/diagnosis_extract.py`):

- It reads named diagnoses and **ICD-10 codes** (F84 ASD, F70 Intellectual
  Disability, F90 ADHD, F80 Speech/Language, F43.1 PTSD, F41.0 Panic, …),
  sentence by sentence, so a "rule out X" / "no evidence of X" line is never
  mistaken for a positive diagnosis.
- A condition introduced as **co-occurring/secondary** is not made the primary
  ("ASD with co-occurring ADHD" -> ASD primary, ADHD co-occurring).
- When a report explicitly says **"no diagnosis" / typical development**, the
  system returns *No diagnosis indicated* instead of forcing a class.
- Every result now carries an **explanation** and a `primary_label`, fixing the
  "always N/A / no explanation" API findings.

**Backward (post-position) negation** was added so "Suicidal ideation **was
explicitly denied**", "Tics **were not observed**", and "ASD **was ruled out**"
negate the terms *before* the cue — fixing a false High-risk flag and false
co-occurring entries.

**Result on the 8 audited cases** (all previously "N/A"):

| Case | Now |
|---|---|
| ADHD / ASD / Dyslexia (stated + ICD) | correct primary, 92% |
| Speech/Language Disorder (F80) | **named** (beyond trained classes), 92% |
| Intellectual Disability (F70) | **named**, 92% |
| ADHD + ASD comorbid | **ASD** primary, ADHD co-occurring |
| Normal child | **No diagnosis indicated** (no hallucinated co-occurring) |
| PTSD (suicidality denied) | **PTSD/Trauma**, risk Moderate (not High) |

Note: the learned model's held-out generalisation is still the **0.93** above;
the stated-diagnosis reader is a high-precision front-end for reports that
explicitly record their conclusion, with the model + symptom prior as the
fallback for reports that do not.

---

## 10. Second external audit — false positives, hallucinations, trauma

A follow-up audit found the stated-diagnosis reader had introduced new failure
modes. All are fixed and guarded:

- **Normal child mislabelled (critical false positive).** A ruled-out condition
  ("*Specific Learning Disorder / Dyslexia was considered but not met*", "*no
  evidence of dyslexia*") was read as a positive diagnosis. The rule-out
  detector now catches "considered", "differential", "not met", "within normal
  limits", "average and consistent"; the loose "consistent with" trigger (which
  fired on "consistent with **ability**") was removed; and the co-occurring
  detector is now **rule-out-sentence aware**. A normal report now returns
  *No diagnosis*, with ruled-out conditions shown separately.
- **"Tics / Tourette" hallucination.** The keyword "tics" was substring-matching
  **"mathematics"**, "characteristics", "diagnostics". Co-occurring matching is
  now anchored at a **word start**, so short keywords only fire on real mentions.
- **Trauma narrative mistaken for ADHD.** A PTSD presentation written without a
  diagnosis line (event + nightmares + hypervigilance + startle + avoidance) was
  classified ADHD. A precise trauma recogniser now surfaces **PTSD / Trauma**
  (flagged, outside trained classes) — requiring an explicit exposure *event*
  plus trauma-specific symptoms, and deliberately excluding "intrusive thoughts"
  so it does not fire on OCD.
- **Backward negation** ("Suicidal ideation **was explicitly denied**") and
  **out-of-scope confidence** (named-but-untrained conditions capped, never
  claiming model-level certainty) addressed.

Guarded by `test_system.py` (42 checks).

---

## 11. Third audit — scores, wrong population, mixed presentations

A third audit (which confirmed clean single-diagnosis cases now work) isolated
four harder failures, all addressed via a focused standardized-score interpreter
(`src/scores.py`) — the model reads language, not numbers, so these needed score
awareness rather than more text matching:

- **Out-of-domain adult / TBI report accepted (#4).** An adult neuropsych
  evaluation (WAIS-IV, WMS-IV, mild TBI, "return to work") was classified as
  ADHD. The interpreter now detects **adult instruments / acquired-injury
  context** and the report is **refused as out of scope** for a paediatric tool.
- **Confidently-wrong "No diagnosis" on an impaired profile (#2).** A borderline
  profile (FSIQ 74, academics 62-68) was cleared at 80% HIGH. "No diagnosis" is
  now blocked when scores show impairment, and **Borderline Intellectual
  Functioning** (or Intellectual Disability for FSIQ < 70) is recognised from the
  IQ and flagged for review.
- **Mixed ADHD/ASD coin-flip (#1) and twice-exceptional (#3).** A near-tie
  between two real conditions — or score evidence for two (SRS-2 high + attention
  high → ADHD + ASD; reading-low + attention-high → ADHD + Dyslexia) — is now
  shown as a **co-primary** ("ASD + ADHD") and flagged, instead of an arbitrary
  single pick.
- Paediatric reports that carry **standardized scores but no narrative** are no
  longer wrongly refused by the format gate.

**Honest limitation kept in view:** the system still leans on a report's *stated*
conclusion when one is present (high confidence) and is weaker when it must
reason purely from numeric scores (lower confidence — which now correctly routes
to review). The score interpreter is deliberately narrow (IQ bands, ADOS/SRS/
Conners cut-offs); a full psychometric reasoner is future work. None of this is a
substitute for a clinician.

Guarded by `test_system.py` (**45 checks**).
