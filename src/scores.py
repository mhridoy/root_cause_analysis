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
    attn_high = bool(re.search(
        r"(conners|basc)[^\.]{0,40}(t[\s=:-]*?(6[5-9]|[7-9]\d)|elevated|"
        r"very elevated|clinically (significant|elevated))", text, re.I))
    reading_low = bool(re.search(
        r"(phonolog|reading|decoding|word reading|spelling)[^\.]{0,40}"
        r"(below average|well below|impaired|2[0-9](th)? percentile|"
        r"1\d(th)? percentile|deficit)", text, re.I))
    return {
        "iq": iq,
        "has_tests": bool(has_tests),
        "ados_positive": ados_pos,
        "srs_high": srs_high,
        "attention_high": attn_high,
        "reading_low": reading_low,
    }
