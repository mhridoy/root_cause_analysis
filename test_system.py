"""
Self-test / trust check. Run this to confirm the system actually analyzes input
rather than returning a canned answer:

    python3 test_system.py

It checks three things:
  1. Non-clinical documents (invoice, recipe, resume, news) are REFUSED.
  2. Real reports from each diagnosis family get the CORRECT, DIFFERENT label.
  3. Genuine reports outside the 6 trained classes route to 'Other / Complex'
     and are flagged for expert review (not forced into a wrong label).
"""
import os
import sys
import glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.analyze import get_analyzer

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "merged_unique_text")
A = get_analyzer()

NON_CLINICAL = {
    "invoice": "INVOICE #INV-2026-0481\nBill To: Acme Retail Corp\nDescription Qty Unit Price Amount\n"
               "Web hosting 1 1200.00 1200.00\nConsulting hours 10 150.00 1500.00\n"
               "Subtotal 2880.00 Tax 230.40 Total Due 3110.40\nPayment terms Net 30. Thank you for your business.",
    "recipe":  "Banana Bread Recipe. Ingredients: 3 ripe bananas, butter, sugar, egg, vanilla, baking soda, flour. "
               "Preheat oven to 350F. Mash bananas, mix butter, add sugar and egg, stir in flour, bake 50 minutes, cool and serve.",
    "resume":  "Jane Doe Senior Software Engineer. Led backend teams building cloud services in Python and Go. "
               "Designed REST APIs and CI/CD pipelines on AWS. Skills Python JavaScript SQL Docker. B.Sc Computer Science.",
    "news":    "City Council Approves New Transit Budget. The council voted to approve a record transit budget to expand "
               "bus routes and repair rail lines. Critics argued the plan raises taxes. The mayor praised the decision.",
}

passed = failed = 0


def check(name, cond):
    global passed, failed
    print(("  PASS " if cond else "  FAIL ") + name)
    if cond: passed += 1
    else: failed += 1


print("1) Non-clinical documents must be REFUSED")
for name, text in NON_CLINICAL.items():
    r = A.analyze(text)
    check(f"{name} refused", r.get("out_of_domain") is True)

print("\n2) Real reports must get the CORRECT, distinct label")
expect = {"ASD": "ASD", "ADHD": "ADHD", "Depression": "Depression",
          "Dyslexia": "Dyslexia", "GAD": "GAD", "OCD": "OCD"}
seen = {}
for fam, want in expect.items():
    files = sorted(glob.glob(os.path.join(CORPUS, f"zip__{fam}_Report_*.txt")))
    if not files:
        continue
    r = A.analyze(open(files[0], encoding="utf-8", errors="ignore").read())
    seen[fam] = r["diagnosis"]
    check(f"{os.path.basename(files[0])} -> {r['diagnosis']} (want {want}, conf {r['confidence']:.2f})",
          r["diagnosis"] == want)
check("the six predictions are not all identical",
      len(set(seen.values())) >= 5)

print("\n3) Genuine but out-of-scope reports -> 'Other / Complex' + review flag")
for pat in ["*Bipolar*", "*PTSD*", "*PANIC*"]:
    files = glob.glob(os.path.join(CORPUS, "fifty__*" + pat.strip("*") + "*.txt"))
    if not files:
        continue
    r = A.analyze(open(files[0], encoding="utf-8", errors="ignore").read())
    check(f"{os.path.basename(files[0])[:46]} -> {r['diagnosis']} review={r['needs_doctor_review']}",
          r["needs_doctor_review"] is True)

print("\n4) Root cause is grounded + explainable (not keyword argmax)")
expect_rc = {"ASD": "Sensory / Motor", "Depression": "Mood / Emotional Vulnerability",
             "GAD": "Anxiety / Stress Reactivity",
             "Dyslexia": "Learning / Cognitive Processing"}
for fam, want_rc in expect_rc.items():
    files = sorted(glob.glob(os.path.join(CORPUS, f"zip__{fam}_Report_*.txt")))
    if not files:
        continue
    r = A.analyze(open(files[0], encoding="utf-8", errors="ignore").read())
    rc = r.get("root_cause_detail") or {}
    ok = (rc.get("top") == want_rc and not rc.get("abstain"))
    check(f"{fam}: root cause {rc.get('top')} (want {want_rc}), "
          f"learned+clinical exposed={('learned_top' in rc)}", ok)
# uncertain case must abstain
files = glob.glob(os.path.join(CORPUS, "fifty__27_Post*PTSD*.txt"))
if files:
    rc = A.analyze(open(files[0], encoding="utf-8", errors="ignore").read()).get("root_cause_detail") or {}
    check("out-of-scope PTSD root cause abstains (flagged uncertain)",
          rc.get("abstain") is True)

print("\n5) Paraphrased, symptom-only report (disorder never named) is recognized")
# A textbook OCD presentation that never writes 'OCD' or 'obsessive-compulsive'.
OCD_PARAPHRASE = (
    "The student spends large portions of the day washing hands and cleaning "
    "belongings, and repeatedly checks that doors, windows, and appliances are "
    "secure. There are persistent intrusive thoughts about germs, illness, and "
    "accidental harm, and fears that something terrible will happen if certain "
    "routines are not completed correctly. The student recognizes the thoughts "
    "are excessive but cannot stop them; resisting the urges causes marked "
    "distress. Ritualized checking, perfectionism, reassurance-seeking, and "
    "compulsive cleaning are reported across home and school. The student spends "
    "more than two hours each day on these compulsive behaviors. There is no "
    "history of restricted interests, repetitive motor behaviors, or "
    "developmental delays, and teachers do not report hyperactivity, "
    "impulsivity, or distractibility. Attention during instruction is good and "
    "academic skills are average to above average."
)
r = A.analyze(OCD_PARAPHRASE)
sp = r.get("symptom_prior", {})
check(f"paraphrased OCD -> {r['diagnosis']} at {int(round(r['confidence']*100))}% "
      f"(symptom prior applied={sp.get('applied')})",
      r["diagnosis"] == "OCD" and r["confidence"] >= 0.5)
# negated rule-outs must NOT win
check(f"negated rule-outs (ADHD/ASD) did not win: got {r['diagnosis']}",
      r["diagnosis"] not in ("ADHD", "ASD"))

print("\n6) Negation-aware risk / symptoms / co-occurring (no false positives)")
NEG_REPORT = (
    "Psychological assessment. A 12-year-old presents with intrusive contamination "
    "fears, repetitive checking, handwashing, bedtime rituals, and reassurance "
    "seeking that interfere with schoolwork and sleep. He did not show signs of "
    "hyperactivity. He denied persistent sadness and loss of interest. No suicidal "
    "thoughts, self-harm behaviors, or aggression were reported. No history of "
    "major family conflict, abuse, or neglect. Academic skills are age-appropriate. "
    "Provisional Diagnostic Impression: Obsessive-Compulsive Disorder."
)
r = A.analyze(NEG_REPORT)
check(f"denied risk NOT flagged High (got {r['risk_level']})", r["risk_level"] != "High")
check("denied risk terms recognized as denied (suicidal/abuse/neglect)",
      any(s in r.get("risk_denied", []) for s in ("suicidal", "abuse", "neglect")))
check(f"negated symptoms excluded (no hyperactivity/persistent sadness): {r['symptoms']}",
      "hyperactivity" not in r["symptoms"] and "persistent sadness" not in r["symptoms"])
check(f"primary not listed as its own co-occurring: {r['cooccurring']}",
      "OCD" not in r["cooccurring"])
check(f"unsupported co-occurring not over-listed (<=2 items): {r['cooccurring']}",
      len(r["cooccurring"]) <= 2)

ASSERTED_RISK = (
    "Psychological assessment of a 15-year-old with persistent depressed mood, "
    "anhedonia, hopelessness, and fatigue. He disclosed recurrent suicidal thoughts "
    "and a recent episode of self-harm. A history of abuse is documented. Marked "
    "functional impairment across school and home. Diagnosis: Major Depressive Disorder."
)
r2 = A.analyze(ASSERTED_RISK)
check(f"genuine asserted risk STILL flagged High (got {r2['risk_level']})",
      r2["risk_level"] == "High")

print("\n7) Reads explicitly stated diagnoses (incl. beyond the 6 trained classes)")
STATED = {
    "ADHD": "Neurodevelopmental assessment. Conners-3 T-scores 78-82. Meets all "
            "DSM-5 criteria for ADHD, combined presentation. "
            "Diagnosis: Attention-Deficit/Hyperactivity Disorder (ADHD) (F90.2).",
    "ASD": "Autism assessment. ADOS-2 comparison score 21 (cutoff 11). ADI-R "
           "above cutoffs. Diagnosis: Autism Spectrum Disorder (F84.0).",
    "Dyslexia": "Psychoeducational assessment. CTOPP-2 phonological deficits; WIAT-III "
                "reading below average. Diagnosis: Specific Learning Difficulty - "
                "Dyslexia (F81.0).",
    "Speech / Language Disorder": "Speech and language assessment. CELF-P2 core "
            "language severely delayed; GFTA-3 multiple errors. Nonverbal cognition "
            "within normal limits. Diagnosis: Developmental Language Disorder (F80.2).",
    "Intellectual Disability": "Assessment. WISC-V Full Scale IQ 52; Vineland-3 "
            "composite 51 with developmental onset. Diagnosis: Mild Intellectual "
            "Disability (F70).",
}
for want, rep in STATED.items():
    r = A.analyze(rep)
    check(f"stated {want} -> {r.get('primary_label')} ({int(round(r['confidence']*100))}%, "
          f"src={r.get('diagnosis_source')})",
          r.get("primary_label") == want and r["diagnosis_source"] == "stated"
          and r["confidence"] >= 0.7)

# comorbid: "ASD with co-occurring ADHD" -> ASD primary, ADHD co-occurring
COMORBID = ("Comprehensive assessment. ADOS-2 total 13 (cutoff 11), meeting autism "
            "classification; continues to meet ADHD criteria. Diagnostic Impression: "
            "Autism Spectrum Disorder (F84.0) with co-occurring ADHD (F90.2).")
rc2 = A.analyze(COMORBID)
check(f"comorbid -> primary {rc2['primary_label']} (want ASD), ADHD co-occurring",
      rc2["primary_label"] == "ASD" and "ADHD" in rc2["cooccurring"])

# normal child -> no diagnosis, no hallucinated co-occurring
NORMAL = ("Routine assessment. WISC-V IQ 104; WIAT-III all average. No concerns "
          "regarding attention, mood, social communication, or development. "
          "Behavioral observations unremarkable. No evidence of any disorder. "
          "Impression: No diagnosis; development is age-appropriate.")
rn = A.analyze(NORMAL)
check(f"normal child -> {rn['primary_label']} (no diagnosis), cooc={rn['cooccurring']}",
      "No diagnosis" in rn["primary_label"] and len(rn["cooccurring"]) == 0
      and "Tics / Tourette" not in rn["cooccurring"])

# PTSD: stated, with "Suicidal ideation was explicitly denied" -> not High risk
PTSD = ("Assessment following exposure to domestic violence. Nightmares, intrusive "
        "thoughts, exaggerated startle, avoidance, hypervigilance. Suicidal ideation "
        "and self-harm were explicitly denied. No compulsive rituals. Provisional "
        "Diagnostic Impression: Post-Traumatic Stress Disorder (PTSD).")
rp = A.analyze(PTSD)
check(f"PTSD stated -> {rp['primary_label']} (want PTSD / Trauma)",
      rp["primary_label"] == "PTSD / Trauma")
check(f"PTSD 'suicidal ideation was explicitly denied' NOT High (got {rp['risk_level']})",
      rp["risk_level"] != "High")

print("\n8) v2 audit regressions (false positives, hallucinations, trauma)")
# Normal child where Dyslexia is RULED OUT must not be diagnosed as Dyslexia.
NORMAL_RO = ("Psychoeducational assessment. WISC-V IQ 103 (average). WIAT-III "
             "reading, spelling and mathematics all within the average range and "
             "consistent with ability. Differential considerations: Specific "
             "Learning Disorder / Dyslexia was considered but not met — reading and "
             "spelling are average. No evidence of dyslexia, ADHD, or autism. "
             "Impression: No diagnosis; development is age-appropriate.")
rno = A.analyze(NORMAL_RO)
check(f"ruled-out Dyslexia NOT diagnosed (got {rno['primary_label']})",
      "Dyslexia" not in rno["primary_label"])
check(f"normal child -> no positive co-occurring (got {rno['cooccurring']})",
      len(rno["cooccurring"]) == 0)
check("no Tics/Tourette hallucination from 'mathematics' etc.",
      "Tics / Tourette" not in rno["cooccurring"])

# Trauma narrative (no stated dx) -> PTSD/Trauma, not ADHD; risk not High.
TRAUMA = ("Psychological assessment. New-onset inattentiveness and irritability "
          "that began after the child witnessed domestic violence four months ago. "
          "He has frequent nightmares, is easily startled, scans the room for "
          "threats, and avoids reminders of the event; he is restless and "
          "distractible in class. Before the event he had no attention problems. "
          "Suicidal ideation, self-harm, and abuse were explicitly denied.")
rt = A.analyze(TRAUMA)
check(f"trauma narrative -> {rt['primary_label']} (want PTSD / Trauma, not ADHD)",
      rt["primary_label"] == "PTSD / Trauma")
check(f"trauma 'self-harm/abuse explicitly denied' NOT High risk (got {rt['risk_level']})",
      rt["risk_level"] != "High")

# Out-of-scope stated dx should not claim model-level confidence.
SPL = ("Assessment. CELF-P2 core language severely delayed. Diagnosis: "
       "Developmental Language Disorder (F80.2).")
rs = A.analyze(SPL)
check(f"out-of-scope stated confidence capped <= 0.75 (got {rs['confidence']:.2f})",
      rs["confidence"] <= 0.75 and rs["needs_doctor_review"] is True)

print("\n9) v3 audit: adult/TBI rejection, borderline ID, co-primary, scores")
ADULT_TBI = ("Adult Neuropsychological Evaluation. A 47-year-old man referred "
             "following a mild traumatic brain injury after a motor vehicle "
             "accident. WAIS-IV Full Scale IQ 98. WMS-IV memory indices reduced. "
             "Trail Making B slowed. Recommend cognitive rehabilitation and "
             "graded return to work.")
ra = A.analyze(ADULT_TBI)
check(f"adult TBI rejected as out-of-scope (got {ra['primary_label'][:30]})",
      ra.get("out_of_domain") is True)

BID = ("Psychological assessment of a 10-year-old. WISC-V Full Scale IQ 74 "
       "(Borderline range); indices 70-78. WIAT-III reading 65, spelling 62, "
       "mathematics 68 (well below average). Vineland-3 composite 77. Pervasive "
       "since early development. He does not meet the strict IQ cutoff for "
       "Intellectual Disability. Impression: borderline intellectual functioning.")
rb = A.analyze(BID)
check(f"borderline-ID NOT confidently 'No diagnosis' (got {rb['primary_label']})",
      "No diagnosis" not in rb["primary_label"] and rb["needs_doctor_review"] is True)

OVERLAP = ("Neurodevelopmental assessment of an 8-year-old. Conners-3 Inattention "
           "T=75 and BASC-3 Attention Problems T=74 (elevated). SRS-2 Total T=74 "
           "(moderate-to-severe range) with limited eye contact, rigid thinking, "
           "insistence on routines, and sensory sensitivities.")
ro = A.analyze(OVERLAP)
check(f"mixed ADHD/ASD shown as co-primary (got {ro['primary_label']})",
      ro.get("co_primary") is not None and "+" in ro["primary_label"])

print("\n10) round-3 audit: missing conclusion, label-space gaps, dual dx")
# Complete ADHD report with NO conclusion sentence (and a ruled-out comorbidity)
# must NOT be cleared as 'No diagnosis'.
NOCONCL = ("Neurodevelopmental assessment of an 8-year-old. Conners-3 Inattention "
           "T 74-80 and Hyperactivity T 76-80 (very elevated). DSM-5: 8 of 9 "
           "inattentive and 7 of 9 hyperactive symptoms met, onset before age 6, "
           "two settings. There is no evidence of autism spectrum disorder and no "
           "evidence of a specific learning disorder. Recommendations: classroom "
           "accommodations and a medication review.")
rnc = A.analyze(NOCONCL)
check(f"no-conclusion ADHD report NOT cleared as no-diagnosis (got {rnc['primary_label']})",
      "No diagnosis" not in rnc["primary_label"])

# Social Communication Disorder must NOT collapse into ASD.
SCD = ("Diagnostic assessment of a 7-year-old. ADOS-2 Social Affect 6 (below the "
       "autism cutoff of 8); zero restricted or repetitive behaviours. The profile "
       "is NOT consistent with Autism Spectrum Disorder. The profile is more "
       "consistent with Social (Pragmatic) Communication Disorder (DSM-5 315.39).")
rscd = A.analyze(SCD)
check(f"SCD stays SCD, not ASD (got {rscd['primary_label']})",
      rscd["primary_label"] == "Social Communication Disorder")

# Selective Mutism must be named, not mapped to Speech/Language.
SM = ("Psychological assessment. A 6-year-old who speaks fluently at home but is "
      "silent at school for 14 months, with marked social anxiety. Diagnosis: "
      "Selective Mutism (F94.0).")
rsm = A.analyze(SM)
check(f"Selective Mutism named, not Speech/Language (got {rsm['primary_label']})",
      rsm["primary_label"] == "Selective Mutism")

# Giftedness with no concerns -> No diagnosis (not ADHD).
GIFT = ("Psychoeducational assessment. WISC-V Full Scale IQ 142. All academic "
        "scores above the 99th percentile. No attentional, behavioural, social, or "
        "emotional concerns. Impression: intellectually gifted; no diagnosis is warranted.")
rg = A.analyze(GIFT)
check(f"gifted -> No diagnosis, not ADHD (got {rg['primary_label']})",
      "No diagnosis" in rg["primary_label"])

# Dual STATED diagnosis -> co-primary with Dyslexia named.
DUAL = ("Psychoeducational assessment. Meets DSM-5 criteria for ADHD; CTOPP-2 "
        "composites at the 1st-2nd percentile. Diagnosis: Attention-Deficit/"
        "Hyperactivity Disorder (F90.2) AND Dyslexia / Specific Learning Disorder "
        "in reading (F81.0).")
rd = A.analyze(DUAL)
check(f"dual stated ADHD+Dyslexia -> co-primary (got {rd['primary_label']})",
      rd.get("co_primary") is not None and "Dyslexia" in rd["primary_label"]
      and "ADHD" in rd["primary_label"])

print("\n11) round-4 audit: stated-vs-ML conflict, conclusion negation, SLD subtypes")
# Wrong stated conclusion that contradicts strong ML evidence -> flagged, not amplified.
CONFLICT = ("Psychoeducational Assessment. An 8-year-old assessed for literacy "
            "difficulties. CTOPP-2 phonological awareness and rapid naming severely "
            "impaired at the 1st percentile. WIAT-III word reading and spelling at the "
            "2nd percentile; reading fluency markedly impaired; decoding poor. "
            "Cognitive ability average. There is a classic double deficit in "
            "phonological processing and reading. Diagnosis: Autism Spectrum "
            "Disorder (F84.0).")
rcf = A.analyze(CONFLICT)
check(f"wrong stated conclusion flagged + de-confidenced (conf {rcf['confidence']:.2f}, "
      f"review {rcf['needs_doctor_review']})",
      rcf["confidence"] <= 0.60 and rcf["needs_doctor_review"] is True
      and "discrepancy" in rcf["explanation"].lower())

# "NOT ADHD, NOT ASD" with RAD stated -> RAD, not the excluded conditions.
RAD = ("Psychological assessment. History of early institutional care and disrupted "
       "attachment; emotionally withdrawn toward caregivers. Diagnosis: Reactive "
       "Attachment Disorder. This is NOT ADHD and NOT Autism Spectrum Disorder.")
rr = A.analyze(RAD)
check(f"RAD stated, ADHD/ASD excluded (got {rr['primary_label']})",
      rr["primary_label"] == "Reactive Attachment Disorder")

# "Neither ASD nor ADHD can be diagnosed" -> not ASD+ADHD.
NEITHER = ("Neurodevelopmental assessment. Informants diverged. Neither Autism "
           "Spectrum Disorder nor ADHD can be definitively diagnosed at this time; "
           "further assessment is required before a diagnosis can be established.")
rne = A.analyze(NEITHER)
check(f"'neither ASD nor ADHD' NOT output as ASD/ADHD (got {rne['primary_label']})",
      "ASD" not in rne["primary_label"] and rne["primary_label"] != "ADHD")

# Written-expression SLD must not collapse to Dyslexia.
SLDW = ("Psychoeducational assessment. WIAT-III Reading Comprehension at the 96th "
        "percentile; Written Expression at the 3rd percentile with severe spelling and "
        "handwriting difficulty. Diagnosis: Specific Learning Disability in Written "
        "Expression (dysgraphia).")
rw = A.analyze(SLDW)
check(f"writing-only SLD not labelled Dyslexia (got {rw['primary_label']})",
      rw["primary_label"] != "Dyslexia")

# Family-history mention must not become the child's diagnosis source.
FAM = ("Neurodevelopmental assessment. Conners-3 Inattention T 75-79; meets 7 of 9 "
       "inattentive symptoms. Mother has adult ADHD (diagnosed age 38).")
rf = A.analyze(FAM)
check(f"family-history ADHD not read as a stated dx (src={rf['diagnosis_source']})",
      rf["diagnosis_source"] != "stated")

print("\n12) round-5 audit: subtest-note grab, score conflict, historical dx")
# Depressed child, no conclusion, with a subtest score note -> must NOT grab
# 'Written Expression 86' as a diagnosis.
DEPNC = ("Psychological assessment of an 8-year-old. CDI-2 Self-Report T=76. BASC-3 "
         "parent Depression T=80; teacher Depression T=76. RCADS Major Depressive "
         "Disorder T=78. Persistent sadness, anhedonia, low energy, sleep "
         "disturbance, tearfulness. Academic testing: Written Expression 86 (below "
         "expectation given verbal ability, a 32-point gap, consistent with "
         "motivational impact of low mood). Recommend mood-focused intervention.")
rdn = A.analyze(DEPNC)
check(f"subtest-score note NOT read as SLD-Written (got {rdn['primary_label']})",
      "Written" not in rdn["primary_label"])

# Anxiety scores but ADHD conclusion (and ML also leans ADHD) -> flag + review.
GADADHD = ("Psychological assessment of a 9-year-old. MASC-2 Total T=78; RCADS "
           "Generalised Anxiety T=78; BASC-3 Anxiety T=80 (clinically significant). "
           "Conners-3 Inattention T=58 and Hyperactivity T=46 (average). Excessive "
           "worry and physical tension. Diagnosis: Attention-Deficit/Hyperactivity Disorder.")
rga = A.analyze(GADADHD)
check(f"anxiety-scores vs ADHD label: flagged + review (conf {rga['confidence']:.2f}, "
      f"review {rga['needs_doctor_review']})",
      rga["confidence"] <= 0.60 and rga["needs_doctor_review"] is True)

# Prior/background diagnosis must not override the current conclusion.
PRIOR = ("Neurodevelopmental assessment of a 12-year-old. Background: a prior "
         "diagnosis of PTSD at age 7, now resolved. Current presentation: persistent "
         "social-communication difficulties, restricted interests, sensory "
         "sensitivities since early childhood; ADOS-2 above cutoff. Primary "
         "Diagnosis: Autism Spectrum Disorder, Level 1.")
rpr = A.analyze(PRIOR)
check(f"current ASD wins over prior PTSD (got {rpr['primary_label']})",
      rpr["primary_label"] == "ASD")

print(f"\n==== {passed} passed, {failed} failed ====")
sys.exit(0 if failed == 0 else 1)
