"""
Explicit diagnosis-statement extractor.

Real assessment reports usually STATE their conclusion ("Diagnosis: Specific
Learning Difficulty - Dyslexia (F81.0)", "Mild Intellectual Disability (F70)",
"Provisional Diagnostic Impression: PTSD"). The trained classifier only knows
six families and was getting these wrong (or refusing). A clinician reads the
stated diagnosis; so should the tool.

This module finds the stated diagnosis with high precision:
  * sentence-level, so a "rule out X" / "no evidence of X" / denied sentence is
    NOT mistaken for a positive diagnosis;
  * recognises conditions BEYOND the six modelled classes (PTSD, Intellectual
    Disability, Speech/Language Disorder, DCD, Tourette, Bipolar, etc.);
  * raises confidence when the statement is a formal diagnosis line or carries an
    ICD-10 code.

It is used as a high-precision front-end; when no explicit statement is found the
system falls back to the learned model + symptom-signature prior.
"""
import re
from .textproc import split_negation, term_present

# Canonical label -> list of regex patterns (searched case-insensitively).
CONDITION_PATTERNS = [
    ("ADHD", [r"\badhd\b", r"attention[-\s]?deficit", r"\bf90\b", r"hyperkinetic"]),
    # Social (Pragmatic) Communication Disorder BEFORE ASD — it is the "looks like
    # autism but is not" diagnosis and must not collapse into ASD.
    ("Social Communication Disorder",
     [r"social (\(?pragmatic\)? )?communication disorder", r"\bscpd\b",
      r"pragmatic language (disorder|impairment)", r"\bf80\.89\b", r"315\.39"]),
    ("ASD", [r"\basd\b", r"autism", r"autistic", r"asperger", r"\bf84",
             r"pervasive developmental"]),
    ("Selective Mutism", [r"selective mutism", r"\bf94\.0", r"313\.23"]),
    ("Reactive Attachment Disorder",
     [r"reactive attachment disorder", r"\brad\b", r"\bf94\.1", r"313\.89"]),
    # SLD subtypes BEFORE generic Dyslexia, and each kept distinct so a
    # writing-only or maths-only disability is not mislabelled as dyslexia.
    # NB: requires a DISORDER term — bare "Written Expression" is a subtest name
    # (a score), not a diagnosis, and must not be grabbed as one.
    ("SLD - Written Expression",
     [r"dysgraphi", r"disorder of written", r"written expression disorder",
      r"impair\w*\s+(in\s+)?written expression",
      r"specific learning.{0,25}(in\s+)?writ", r"\bf81\.81\b"]),
    ("Dyscalculia", [r"dyscalculia", r"mathematics disorder",
                     r"specific learning.{0,20}math", r"\bf81\.2"]),
    ("Dyslexia", [r"dyslexi", r"specific learning.{0,20}read", r"\bf81\.0",
                  r"\bspld\b", r"reading disorder"]),
    ("Speech / Language Disorder",
     [r"developmental language disorder", r"\bdld\b", r"language disorder",
      r"speech[-\s]?(and|&)?[-\s]?language (disorder|delay)",
      r"speech sound disorder", r"\bf80"]),
    ("Intellectual Disability",
     [r"intellectual disabilit", r"intellectual developmental disorder",
      r"\bf7[0-3]\b", r"global developmental delay", r"\bgdd\b",
      r"mental retardation"]),
    ("Borderline Intellectual Functioning",
     [r"borderline intellectual", r"\bbif\b", r"\br41\.83\b", r"slow learner"]),
    ("PTSD / Trauma",
     [r"\bptsd\b", r"post[-\s]?traumatic stress", r"\bf43\.?1",
      r"acute stress disorder", r"trauma[-\s]?(and|&|related|focused)"]),
    ("OCD", [r"\bocd\b", r"obsessive[-\s]?compulsive", r"\bf42\b"]),
    # Specific anxiety subtypes BEFORE generic GAD so they aren't absorbed by the
    # broad "anxiety disorder" pattern.
    ("Panic Disorder", [r"panic disorder", r"\bf41\.?0"]),
    ("Separation Anxiety", [r"separation anxiety", r"\bf93\.?0"]),
    ("Social Anxiety", [r"social anxiety", r"social phobia", r"\bf40\.?1"]),
    ("GAD", [r"generali[sz]ed anxiety", r"\bgad\b", r"\bf41\.?1",
             r"anxiety disorder"]),
    ("Depression", [r"major depress", r"depressive disorder", r"\bmdd\b",
                    r"\bf3[23]", r"dysthymi", r"persistent depressive"]),
    ("DCD / Dyspraxia", [r"developmental coordination disorder", r"\bdcd\b",
                         r"dyspraxi", r"\bf82"]),
    ("Tourette / Tics", [r"tourette", r"\btic disorder\b", r"\bf95"]),
    ("Bipolar", [r"bipolar", r"\bf31"]),
    ("ODD / Conduct", [r"oppositional defiant", r"\bodd\b", r"conduct disorder",
                       r"\bf91"]),
    ("DMDD", [r"disruptive mood dysregulation", r"\bdmdd\b"]),
]

# A sentence that announces a diagnosis. "(more )?consistent with" is included
# again — benign uses like "consistent with ability/age" are now caught by
# RULEOUT (so they are excluded), so this no longer over-fires.
DX_STATEMENT = re.compile(
    r"(diagnos|diagnostic impression|impression\s*[:\-]|meets? (the )?(dsm[-\s]?5 )?"
    r"criteria for|met (the )?criteria|criteria (for [\w /]+ )?(are|were|is|was) met|"
    r"(more |most )?consistent with|clinically consistent with|classification of|"
    r"confirms?\b|provisional)", re.I)
# ICD-10 (F-codes) and DSM-5 numeric codes both count as strong diagnostic anchors.
ICD = re.compile(r"\bf\d{2}(\.\d{1,2})?\b|\b3\d{2}\.\d{1,2}\b", re.I)

# Markers that a condition is SECONDARY / co-occurring (so not the primary).
SECONDARY = re.compile(
    r"(co-?occurring|comorbid|secondary|alongside|in addition|as well as|"
    r"\bplus\b|together with|along with)", re.I)

# Rule-out / negative / differential context -> a positive diagnosis must NOT be
# read here. Broadened to catch "considered but not met", "differential",
# "within normal limits", "average and consistent", etc. (the normal-child bug).
RULEOUT = re.compile(
    r"(rule[ds]?\s+out|ruled out|to exclude|no evidence|does not meet|did not meet|"
    r"not (be |been )?met|not meet|criteria (were |are |was )?not met|differential|"
    r"consider(ed|ing|ation)|negative for|\bdenied\b|\bdenies\b|was absent|were absent|"
    r"unremarkable|\bquery\b|\bqueried\b|\bversus\b|\bvs\.?\b|\br/o\b|"
    r"within normal limits|average and consistent|no concerns|sub[-\s]?clinical|"
    r"below (the )?(clinical )?threshold|does not have|not present|not indicated|"
    r"not warranted|not suggestive|no diagnosis|"
    # explicit "not X" / uncertainty phrasing
    r"not consistent with|not (suggestive|indicative) of|not (autistic|autism)|"
    r"inconsistent with|cannot be made|cannot be confirmed)", re.I)

# Diagnostic uncertainty -> do NOT emit a confident positive label.
UNCERTAIN = re.compile(
    r"(definitive diagnosis (cannot|could not) be (made|reached|confirmed)|"
    r"(diagnosis|diagnose[ds]?) (cannot|could not|can not) be (made|confirmed|"
    r"established|determined|reached)|cannot be (definitively |formally )?"
    r"(diagnosed|established|determined)|neither[\w\s,]{0,40}\bnor\b[\w\s]{0,40}"
    r"(diagnos|can be)|inconclusive|further (assessment|evaluation|testing) "
    r"(is )?(required|needed|recommended) (before|to|prior))", re.I)

# Family-history / informant context — a condition named here is NOT the child's
# diagnosis ("Mother has adult ADHD", "family history of autism").
FAMILY = re.compile(
    r"(family history|mother|father|maternal|paternal|sibling|brother|sister|"
    r"parent|grandparent|aunt|uncle|cousin|relative|hereditary|runs in the family)",
    re.I)

# Historical / prior diagnosis — a PAST diagnosis is not the CURRENT one.
HISTORICAL = re.compile(
    r"(prior diagnos|previous diagnos|previously diagnosed|past diagnos|"
    r"earlier diagnos|formerly diagnosed|history of|historically|in the past|"
    r"at age \d|background[:\s]|since resolved|now resolved|no longer (meets|has))",
    re.I)

# Explicit "no diagnosis / typical development".
NO_DX = re.compile(
    r"(no (formal |clinical )?diagnos(is|es)( is)?( indicated| warranted| made| required)?|"
    r"does not meet (the )?criteria for any|no evidence of (any )?(neurodevelopmental|"
    r"psychiatric|psychological|mental health)?\s*(disorder|condition|diagnos)|"
    r"typically developing|development is age[-\s]appropriate|"
    r"no neurodevelopmental (diagnos|disorder|condition)|within normal limits)", re.I)

MODELLED = {"ADHD", "ASD", "Depression", "Dyslexia", "GAD", "OCD"}


def _sentences(text):
    # NB: do not split on ':' — it separates "Diagnosis:" / "Impression:" from
    # the condition that follows.
    return re.split(r"(?<=[.;!?])\s+|\n+", text)


def extract(text):
    """Return a dict describing the explicitly stated diagnosis (if any):
      stated: canonical label or None
      confidence: 0..1 strength of the statement
      modelled: whether `stated` is one of the 6 trained classes
      evidence: the sentence the diagnosis was read from
      no_diagnosis: True if the report explicitly indicates no disorder
      ruled_out: list of conditions explicitly ruled out / denied
      mentioned: list of conditions positively mentioned (for co-occurring)
    """
    low = text.lower()
    scores = {}            # condition -> [score, best_evidence_sentence]
    ruled_out = set()
    mentioned = set()

    uncertain = bool(UNCERTAIN.search(text))
    for sent in _sentences(text):
        sl = sent.lower()
        if len(sl.strip()) < 4:
            continue
        # a rule-out OR an uncertainty sentence ("diagnosis cannot be made") must
        # not yield a positive diagnosis
        is_ruleout = bool(RULEOUT.search(sl)) or bool(UNCERTAIN.search(sl))
        is_stmt = bool(DX_STATEMENT.search(sl))
        has_icd = bool(ICD.search(sl))
        # word-level negation within the sentence ("NOT ADHD", "neither ASD nor
        # ADHD") so an excluded condition is not read as positive
        sent_neg = split_negation(sent)[1]
        for cond, pats in CONDITION_PATTERNS:
            m = next((mm for p in pats for mm in [re.search(p, sl)] if mm), None)
            if m:
                # the matched surface form falls inside the negated span?
                matched = re.sub(r"[-_]", " ", m.group(0))
                cond_negated = matched in sent_neg
                if is_ruleout or cond_negated:
                    ruled_out.add(cond)
                    continue
                # named in a family-history / informant or HISTORICAL (prior
                # diagnosis) clause -> not the child's CURRENT diagnosis
                ctx = sl[max(0, m.start() - 60):m.start()]
                if FAMILY.search(ctx) or HISTORICAL.search(ctx):
                    continue
                mentioned.add(cond)
                # a condition introduced as co-occurring/secondary is NOT the
                # primary, even in a diagnosis statement ("ASD with co-occurring
                # ADHD" -> ASD primary, ADHD secondary).
                before = sl[max(0, m.start() - 45):m.start()]
                is_secondary = bool(SECONDARY.search(before))
                w = 1.0
                if is_stmt:
                    w += 2.5
                if has_icd:
                    w += 2.0
                if is_secondary:
                    w *= 0.3
                # keep the single strongest statement per condition (no
                # accumulation — incidental repeated mentions must not add up to
                # a "stated" diagnosis).
                if cond not in scores or w > scores[cond][0]:
                    scores[cond] = [w, sent.strip()]

    stated, confidence, evidence = None, 0.0, ""
    if scores:
        stated = max(scores, key=lambda c: scores[c][0])
        w = scores[stated][0]
        evidence = scores[stated][1]
        # map statement strength -> confidence. Require a real diagnosis
        # statement or ICD code (w >= 3) — a bare incidental mention (w == 1)
        # is NOT a stated diagnosis and is left to the learned model.
        if w >= 5:        # diagnosis line + ICD code
            confidence = 0.92
        elif w >= 3.5:    # formal diagnosis statement
            confidence = 0.85
        elif w >= 3.0:    # statement or ICD present
            confidence = 0.78
        else:             # only an incidental mention -> weak, let ML decide
            stated, confidence = None, 0.0

    # A genuine DUAL stated diagnosis ("ADHD AND Dyslexia") -> capture the second
    # strongly-stated condition so the caller can present a co-primary.
    stated_secondary = None
    if stated:
        ranked = sorted(scores.items(), key=lambda kv: -kv[1][0])
        for cond, (w, _) in ranked:
            if cond != stated and w >= 3.5:   # also a formal statement, not incidental
                stated_secondary = cond
                break

    no_dx = bool(NO_DX.search(text)) and not stated
    return {
        "stated": stated,
        "stated_secondary": stated_secondary,
        "confidence": confidence,
        "modelled": stated in MODELLED if stated else False,
        "evidence": evidence,
        "no_diagnosis": no_dx,
        "uncertain": uncertain,
        "ruled_out": sorted(ruled_out),
        "mentioned": sorted(mentioned),
    }
