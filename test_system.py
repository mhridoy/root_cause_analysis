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

print(f"\n==== {passed} passed, {failed} failed ====")
sys.exit(0 if failed == 0 else 1)
