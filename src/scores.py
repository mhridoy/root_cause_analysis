"""
Standardized-score interpreter.

The learned model reads language, not numbers, so it mishandles reports whose
diagnosis lives in the SCORES (FSIQ 74 -> borderline; SRS-2 T=74 -> autism range;
phonological 25th %ile -> dyslexia). It also can't tell a paediatric
neurodevelopmental report from an ADULT neuropsych / TBI evaluation. This module
extracts the few high-value signals needed to fix those cases — deliberately
narrow and high-precision, not a full psychometric engine.
"""
import re

# --- instrument vocabularies -------------------------------------------------
PEDIATRIC_TESTS = [
    "wisc", "wppsi", "wiat", "ados", "adi-r", "adir", "conners", "ctopp",
    "basc", "vineland", "srs-2", "srs2", "social responsiveness", "basciii",
    "celf", "gfta", "movement abc", "das-ii", "leiter", " kabc", "wj-iv",
    "brief", "sensory profile", "ucla ptsd",
]
ADULT_TESTS = ["wais", "wms", "trail making", "boston naming", "rbans",
               "d-kefs", "wcst", "cvlt", "mmse", "moca", "wtar"]
ADULT_CONTEXT = [
    "traumatic brain injury", "\\btbi\\b", "post-concussion", "concussion",
    "stroke", "dementia", "alzheimer", "parkinson", "return to work",
    "cognitive rehabilitation", "geriatric", "retirement", "occupational injury",
    "workers compensation", "neurodegenerative",
]
PEDIATRIC_CONTEXT = [
    "school", "teacher", "classroom", "parent", "developmental", "milestones",
    "iep", "year-old (boy|girl|child|student)", "grade", "nursery", "preschool",
]
_ADULT_AGE = re.compile(r"\b(1[89]|[2-9]\d)[\s-]*year[\s-]*old\b", re.I)
_CHILD_AGE = re.compile(r"\b([1-9]|1[0-7])[\s-]*year[\s-]*old\b", re.I)


def _count(text, terms):
    return sum(1 for t in terms if re.search(t, text, re.I))


def _fsiq(text):
    """Lowest Full-Scale IQ value mentioned (the diagnostically relevant one)."""
    vals = []
    for m in re.finditer(
            r"(full[\s-]?scale iq|fsiq|full scale|general (ability|intellectual)|"
            r"\biq\b)[^\d]{0,15}(\d{2,3})", text, re.I):
        v = int(m.group(m.lastindex))
        if 40 <= v <= 160:
            vals.append(v)
    return min(vals) if vals else None


def domain(text):
    """Return ('paediatric'|'adult'|'unknown', reason)."""
    adult = _count(text, ADULT_TESTS) + _count(text, ADULT_CONTEXT)
    paed = _count(text, PEDIATRIC_TESTS) + _count(text, PEDIATRIC_CONTEXT)
    adult_age = bool(_ADULT_AGE.search(text)) and not _CHILD_AGE.search(text)
    if (adult >= 2 or (adult >= 1 and adult_age)) and adult > paed:
        return "adult", "adult instruments / acquired-injury context"
    if paed >= 1:
        return "paediatric", "paediatric instruments / context"
    return "unknown", ""


def parse(text):
    """High-value score signals (all best-effort, high precision)."""
    iq = _fsiq(text)
    has_tests = (_count(text, PEDIATRIC_TESTS) + _count(text, ADULT_TESTS)) >= 1 \
        or re.search(r"\bt[\s-]?score|percentile|standard score|scaled score", text, re.I)
    ados_pos = bool(re.search(
        r"ados[-\s]?2?[^\.]{0,40}(above (the )?cutoff|exceed|autism (range|classification)|"
        r"comparison score (of )?(1[1-9]|[2-9]\d))", text, re.I))
    srs_high = bool(re.search(
        r"srs[-\s]?2?[^\.]{0,30}(t[\s=:-]*?(6[6-9]|[7-9]\d)|above cutoff|"
        r"(moderate|severe|elevated))", text, re.I))
    # must be an ATTENTION score (not just any Conners/BASC subscale — a BASC
    # Anxiety T-score must not count as an attention signal).
    attn_high = bool(re.search(
        r"((conners|vanderbilt|snap[-\s]?iv|adhd rating scale)[^\.]{0,35}"
        r"(inattention|hyperactivit|attention)|"
        r"(inattention|hyperactivit|attention problems)[^\.]{0,15})"
        r"[^\.]{0,15}(t[\s=:-]*?(6[5-9]|[7-9]\d)|very elevated|clinically (significant|elevated))",
        text, re.I))
    reading_low = bool(re.search(
        r"(phonolog|reading|decoding|word reading|spelling)[^\.]{0,40}"
        r"(below average|well below|impaired|2[0-9](th)? percentile|"
        r"1\d(th)? percentile|deficit)", text, re.I))
    anxiety_high = bool(re.search(
        r"(masc|scared|spence|rcads[^.]{0,25}(anxiet|gad|generali[sz]ed)|"
        r"basc[^.]{0,20}anxiet)[^.]{0,30}"
        r"(t[\s=:-]*?(6[5-9]|[7-9]\d)|elevated|clinically (significant|elevated))",
        text, re.I))
    depression_high = bool(re.search(
        r"(cdi|mfq|rcads[^.]{0,25}(depress|mdd|major)|basc[^.]{0,20}depress|"
        r"children'?s depression)[^.]{0,30}"
        r"(t[\s=:-]*?(6[5-9]|[7-9]\d)|elevated|clinically (significant|elevated))",
        text, re.I)) or bool(re.search(
        r"phq[-\s]?(9|a)[^.]{0,45}"
        r"((total\s+)?score\s*[:=]?\s*(1[5-9]|2[0-7])\b|"
        r"moderately severe|severe depression)", text, re.I))
    return {
        "iq": iq,
        "has_tests": bool(has_tests),
        "ados_positive": ados_pos,
        "srs_high": srs_high,
        "attention_high": attn_high,
        "reading_low": reading_low,
        "anxiety_high": anxiety_high,
        "depression_high": depression_high,
    }


def indicated_conditions(s):
    """Map parsed score signals -> set of strongly-score-indicated conditions
    (in the model's label space). Used to flag stated labels that the scores
    do not support."""
    conds = set()
    if s.get("attention_high"):
        conds.add("ADHD")
    if s.get("srs_high") or s.get("ados_positive"):
        conds.add("ASD")
    if s.get("reading_low"):
        conds.add("Dyslexia")
    if s.get("anxiety_high"):
        conds.add("GAD")
    if s.get("depression_high"):
        conds.add("Depression")
    if s.get("iq") is not None and s["iq"] < 70:
        conds.add("Intellectual Disability")
    return conds
