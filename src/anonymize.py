"""
PII anonymisation for patient-protection. Runs on every uploaded report before
storage/analysis. Conservative regex + label-driven scrubbing; redactions are
counted so the UI can show what was removed.

NOTE: No automated scrubber is perfect. This reduces obvious identifiers
(names after labels, emails, phones, IDs, dates, addresses) but a human should
confirm before sharing externally.
"""
import re

_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("MRN", re.compile(r"\b(?:MRN|mrn|record\s*(?:no|number|#))\s*[:#]?\s*\w+", re.I)),
    ("DATE", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
    ("DOB", re.compile(r"\b(?:d\.?o\.?b\.?|date of birth)\s*[:\-]?\s*[^\n,;]{1,30}", re.I)),
    ("ADDRESS", re.compile(r"\b\d{1,5}\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Blvd)\b")),
]

# Labelled name fields: "Name: John Smith", "Patient: ...", "Child's Name: ..."
_NAME_FIELD = re.compile(
    r"((?:patient|client|child|student|name|guardian|parent|mother|father|examiner|evaluator|clinician|therapist)"
    r"(?:'s)?\s*name)\s*[:\-]\s*([^\n,;]{1,40})", re.I)
_NAME_FIELD2 = re.compile(
    r"\b(name|patient|client)\s*[:\-]\s*([A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){0,3})")


def anonymize(text: str):
    """Return (clean_text, counts dict)."""
    counts = {}
    out = text

    def _bump(tag, n=1):
        counts[tag] = counts.get(tag, 0) + n

    # Name fields first (keep the label, redact the value)
    def _name_sub(m):
        _bump("NAME")
        return f"{m.group(1)}: [REDACTED_NAME]"
    out = _NAME_FIELD.sub(_name_sub, out)

    def _name_sub2(m):
        _bump("NAME")
        return f"{m.group(1)}: [REDACTED_NAME]"
    out = _NAME_FIELD2.sub(_name_sub2, out)

    for tag, pat in _PATTERNS:
        def _sub(m, tag=tag):
            _bump(tag)
            return f"[REDACTED_{tag}]"
        out = pat.sub(_sub, out)

    return out, counts


def anonymization_summary(counts: dict) -> str:
    if not counts:
        return "No obvious personal identifiers detected."
    parts = [f"{v} {k.lower().replace('_', ' ')}" for k, v in sorted(counts.items())]
    return "Redacted: " + ", ".join(parts) + "."
