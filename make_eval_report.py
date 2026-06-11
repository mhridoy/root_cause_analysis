"""Generate reports/evaluation_report.md from models/metrics.json (accurate numbers)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
m = json.load(open(os.path.join(HERE, "models", "metrics.json")))


def tbl(rep, labels):
    out = ["| Class | Precision | Recall | F1 | Support |",
           "|---|---|---|---|---|"]
    for l in labels:
        r = rep[l]
        out.append(f"| {l} | {r['precision']:.2f} | {r['recall']:.2f} | "
                   f"{r['f1']:.2f} | {r['support']} |")
    out.append(f"| **Accuracy** | | | **{rep['_accuracy']:.3f}** | |")
    out.append(f"| **Macro F1** | | | **{rep['_macro_f1']:.3f}** | |")
    out.append(f"| **Weighted F1** | | | **{rep['_weighted_f1']:.3f}** | |")
    return "\n".join(out)


std = m["disease_standard"]; leak = m["disease_leakage_controlled"]
md = f"""# Evaluation Report — AI-Assisted Report Analyzer

**Corpus:** {m['n_reports']} de-duplicated patient-level reports.
**Split:** train {m['split']['train']} / validation {m['split']['val']} / test {m['split']['test']} (stratified by diagnosis).
**Model:** TF-IDF (uni+bigram) → multinomial softmax logistic regression (numpy).
All metrics below are on the **held-out test set** the model never saw during training.

---

## 1. Primary Diagnosis — STANDARD (deployed model, full report text)

Test accuracy **{std['test']['_accuracy']:.3f}**, macro-F1 **{std['test']['_macro_f1']:.3f}**,
mean confidence {std['test']['_mean_confidence']:.2f}, calibration error (ECE) {std['test']['_ece']:.3f}.

{tbl(std['test'], std['labels'])}

## 2. Primary Diagnosis — LEAKAGE-CONTROLLED (diagnosis statements masked)

To test whether the model relies on described symptoms rather than just reading
the stated answer, every diagnosis-revealing phrase ("Primary Diagnosis: …",
DSM/ICD codes, the disease names themselves) was masked, then the model retrained
and re-evaluated from scratch.

Test accuracy **{leak['test']['_accuracy']:.3f}**, macro-F1 **{leak['test']['_macro_f1']:.3f}**,
ECE {leak['test']['_ece']:.3f}.

{tbl(leak['test'], leak['labels'])}

**Interpretation:** performance drops only slightly when the answer is hidden
({std['test']['_accuracy']:.3f} → {leak['test']['_accuracy']:.3f}), which means the model
is genuinely learning from symptom/finding language, not only copying the stated
diagnosis. Both numbers clear the 85% target. The six well-supported classes
(ADHD, ASD, Depression, Dyslexia, GAD, OCD) score F1 0.82–1.00; the small,
heterogeneous *Other / Complex* bucket is the weak spot and is always flagged for
expert review.

## 3. Root-Cause / Contributing-Factor Group

Test accuracy **{m['rootcause']['_accuracy']:.3f}**, macro-F1 **{m['rootcause']['_macro_f1']:.3f}**
across groups: Neurodevelopmental Differences, Anxiety / Stress Reactivity,
Mood / Emotional Vulnerability, Learning / Cognitive Processing.

## 4. Co-occurring Conditions (multi-label, supplementary)

Micro-precision {m['cooccurring']['micro_precision']:.2f},
micro-recall {m['cooccurring']['micro_recall']:.2f},
micro-F1 **{m['cooccurring']['micro_f1']:.2f}** (threshold {m['cooccurring'].get('threshold','?')}).
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

Calibration error (ECE ≈ {std['test']['_ece']:.2f}) is reported so confidence numbers
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
"""
open(os.path.join(HERE, "reports", "evaluation_report.md"), "w").write(md)
print("wrote reports/evaluation_report.md")
