# Report Authenticity / Similarity Audit

## Executive Summary

- The 50 reports are **not** mostly near-duplicate copies.
- They **do** show strong evidence of **template reuse** and repeated narrative scaffolding.
- Content-level duplication is low.
- Structure-level duplication is high.

## Quantitative Summary

- Total reports reviewed: `50`
- High-similarity pairs at cosine similarity `>= 0.60`: `0`
- High-similarity pairs at cosine similarity `>= 0.55`: `1`
- Reused structural template groups: `10`
- Reports participating in reused structural templates: `39 / 50`

## Interpretation

- If the question is whether many reports are almost exact copies: **No**
- If the question is whether many reports use the same writing skeleton with different diagnoses/content swapped in: **Yes**

This pattern is more consistent with:

- automated drafting support
- heavy report templating
- or repeated reuse of a small number of report formats

It is less consistent with:

- fully independent, case-specific report writing for every file

## Strongest Content-Similarity Pairs

Only one pair crossed `0.55` cosine similarity:

- `41 Autism Spectrum Disorder (ASD) + Emotional Regulation Disorder.txt`
- `42 Borderline Intellectual Functioning.txt`
- Similarity: `0.556`

Next highest pairs were substantially lower:

- `40 ASD +ADHD Dual Diagnosis.txt` vs `42 Borderline Intellectual Functioning.txt` -> `0.375`
- `35 Autism Spectrum Disorder (ASD) + Intellectual Disability (Mild–Moderate).txt` vs `42 Borderline Intellectual Functioning.txt` -> `0.325`
- `21 ADHD + Specific Learning Disorder.txt` vs `24 Fine Motor Delay + Learning Difficulties.txt` -> `0.313`
- `40 ASD +ADHD Dual Diagnosis.txt` vs `35 Autism Spectrum Disorder (ASD) + Intellectual Disability (Mild–Moderate).txt` -> `0.311`

## Structural Template Groups

These groups reuse the same broad section-order / narrative skeleton.

### Group A

Count: `11`

- `11 PANIC DISORDE Assessment Report.txt`
- `20 Speech Sound Disorder + Social Difficulties.txt`
- `21 ADHD + Specific Learning Disorder.txt`
- `24 Fine Motor Delay + Learning Difficulties.txt`
- `25 Gross Motor Delay + Sensory Processing Difficulties.txt`
- `27 Post-Traumatic Stress Disorder (PTSD)  Report.txt`
- `28 Sensory Processing Difficulties + Anxiety + School Avoidance.txt`
- `5 Autism Spectrum Disorder (ASD) + Tourette Traits + Anxiety.txt`
- `7 Disruptive Mood Dysregulation Disorder (DMDD)  Assessment Report (1).txt`
- `8 Dyspraxia  DCD and ADHD Traits Report.txt`
- `9 Global Developmental Delay (GDD) Assessment Report.txt`

### Group B

Count: `6`

- `44 Nonverbal Learning Disorder (NVLD) .txt`
- `46 ADHD + Behavioral Addiction (Gaming).txt`
- `47 ASD + Cognitive Processing Deficits.txt`
- `48 ASD + Feeding Disorder.txt`
- `49 Childhood Behavioral Inhibition Disorder.txt`
- `50 Bipolar1 +ADHD.txt`

### Group C

Count: `5`

- `10 Major Depressive Disorder + Generalized Anxiety DisorderReport.txt`
- `2 ADHD COMBINED PRESENTATION Report (AutoRecovered).txt`
- `26 Language Disorder + Academic Struggles Report.txt`
- `3 Anxiety + School Avoidance fnl Report.txt`
- `6 Depression + School Refusal + Anxiety Report.txt`

### Group D

Count: `3`

- `23 Developmental Coordination Disorder (DCD) + Dysgraphia + ADHD Traits.txt`
- `32 Selective Mutism.txt`
- `39 Social Anxiety Disorder.txt`

### Group E

Count: `3`

- `33 ADHD + DMDD AND Anxiety.txt`
- `37 Autism Spectrum Disorder (ASD) + Executive Function Disorder.txt`
- `38 Autism Spectrum Disorder (ASD) + Social Anxiety Disorder + Communication Disorder.txt`

### Group F

Count: `3`

- `35 Autism Spectrum Disorder (ASD) + Intellectual Disability (Mild–Moderate).txt`
- `41 Autism Spectrum Disorder (ASD) + Emotional Regulation Disorder.txt`
- `42 Borderline Intellectual Functioning.txt`

### Group G

Count: `2`

- `22 ADHD + Speech & Language Delay Report.txt`
- `29 Social (Pragmatic) Communication Disorder (SPCD).txt`

### Group H

Count: `2`

- `30 Neurodevelopmental Disorder NOS.txt`
- `31 Emotional Regulation Disorder.txt`

### Group I

Count: `2`

- `34 Autism Spectrum Disorder (ASD) + Sensory Avoidance + Anxiety.txt`
- `36 Adjustment Disorder with Mixed Disturbance of Emotions and Conduct.txt`

### Group J

Count: `2`

- `43 Multiple Comorbidity Profile (ASD + ADHD + Learning Disorder + Anxiety).txt`
- `45 Separation Anxiety Disorder (SAD).txt`

## Bottom-Line Judgment

These reports are best described as:

- **low exact duplication**
- **high template reuse**
- **likely automated or template-heavy drafting in many files**

That does not automatically mean the reports are clinically wrong. It does mean:

- the writing style itself is not reliable evidence of authenticity
- an ML model trained directly on this prose may learn template patterns instead of true clinical reasoning
- a structured-label approach is safer than trusting the raw narratives
