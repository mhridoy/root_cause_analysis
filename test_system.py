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

print(f"\n==== {passed} passed, {failed} failed ====")
sys.exit(0 if failed == 0 else 1)
