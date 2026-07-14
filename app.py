"""
Vercel-compatible stateless web app for the Clinical Report Analyzer.

Features
  * Professional clinical UI (light theme, print-friendly)
  * Drag & drop upload, multi-file batch analysis, paste-text mode
  * Inline evidence highlighting inside the anonymized report text
  * Probability breakdown chart for every trained condition
  * Per-report JSON download + printable report (browser "Save as PDF")
  * JSON API: POST /api/analyze, GET /api/health, GET /api/model-info

Stateless by design: nothing is persisted server-side. The richer local app
with SQLite history remains webapp/app.py.
"""
import html as html_lib
import json
import os
import re
import tempfile

from flask import Flask, jsonify, render_template_string, request

from src.analyze import DISCLAIMER, get_analyzer
from src.anonymize import anonymization_summary, anonymize
from src.extract import extract_text

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

MAX_BATCH_FILES = 5
MAX_HIGHLIGHT_CHARS = 20000

ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md"}
OCR_ONLY_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_model_info():
    """Best-effort summary of the deployed model's held-out performance."""
    try:
        with open(os.path.join(HERE, "models", "metrics.json")) as f:
            m = json.load(f)
        std = m.get("disease_standard", {}).get("test", {})
        leak = m.get("disease_leakage_controlled", {}).get("test", {})
        return {
            "n_reports": m.get("n_reports"),
            "test_accuracy": round(std.get("_accuracy", 0), 4),
            "test_macro_f1": round(std.get("_macro_f1", 0), 4),
            "leakage_controlled_accuracy": round(leak.get("_accuracy", 0), 4),
            "ece": round(std.get("_ece", 0), 4),
        }
    except Exception:
        return {}


MODEL_INFO = _load_model_info()

SAMPLE_TEXT = (
    "Psychoeducational Assessment Summary (synthetic sample for demo purposes). "
    "The student is a 9-year-old referred by the classroom teacher for persistent "
    "difficulty sustaining attention during seatwork, frequent careless mistakes, "
    "and incomplete assignments. Parent and teacher rating scales describe "
    "fidgeting, leaving the seat during lessons, interrupting peers, and "
    "difficulty waiting for a turn. Symptoms have been present since early "
    "childhood, occur both at home and at school, and are not better explained "
    "by a recent stressor. Reading fluency and decoding are age-appropriate. "
    "Sleep routine is consistent. There is no reported history of developmental "
    "regression. Working memory and processing speed scores were relatively "
    "weaker than verbal comprehension. The family reports significant homework "
    "battles and growing frustration. Hearing and vision screening were normal."
)


# --------------------------------------------------------------------------
# Highlighting
# --------------------------------------------------------------------------
def build_highlight_html(anon_text, evidence_sentences, terms):
    """Escape the anonymized report and wrap evidence sentences / model terms
    in highlight markup. Uses control-char placeholders so the replacement
    passes can never corrupt each other's HTML."""
    truncated = len(anon_text) > MAX_HIGHLIGHT_CHARS
    esc = html_lib.escape(anon_text[:MAX_HIGHLIGHT_CHARS])
    SO, SC, TO, TC = "\x02", "\x03", "\x04", "\x05"
    for s in evidence_sentences or []:
        es = html_lib.escape(s)
        if es and es in esc:
            esc = esc.replace(es, SO + es + SC, 1)
    for t in sorted({t for t in (terms or []) if len(t) >= 3},
                    key=len, reverse=True):
        et = re.escape(html_lib.escape(t))
        esc = re.sub(et, lambda m: TO + m.group(0) + TC, esc,
                     count=30, flags=re.IGNORECASE)
    out = (esc.replace(SO, '<span class="hl-sent">').replace(SC, "</span>")
              .replace(TO, '<mark class="hl-term">').replace(TC, "</mark>"))
    out = out.replace("\r\n", "\n").replace("\n", "<br>")
    if truncated:
        out += '<br><em class="sub">… text truncated for display …</em>'
    return out


# --------------------------------------------------------------------------
# Analysis plumbing
# --------------------------------------------------------------------------
def _analyze_text(text, source_name):
    anon_text, counts = anonymize(text)
    result = get_analyzer().analyze(anon_text)
    if result.get("error"):
        raise ValueError(result["error"])
    if result.get("out_of_domain"):
        raise ValueError(result["recommendation"])
    highlight = build_highlight_html(
        anon_text, result.get("evidence"), result.get("evidence_terms"))
    return {
        "name": source_name,
        "result": result,
        "result_json": json.dumps(result, indent=2),
        "anon_summary": anonymization_summary(counts),
        "highlight_html": highlight,
        "confidence_pct": int(round(result.get("confidence", 0) * 100)),
    }


def _extract_uploaded_file(upload):
    filename = upload.filename or "report"
    ext = os.path.splitext(filename)[1].lower()
    if ext in OCR_ONLY_EXT:
        raise ValueError(
            "Image/OCR uploads are disabled in this live deployment. "
            "Use text, DOCX, or native-text PDF.")
    if ext not in ALLOWED_EXT:
        raise ValueError(f"Unsupported file type: {ext or 'unknown'}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir="/tmp") as tmp:
        upload.save(tmp)
        tmp_path = tmp.name
    try:
        text, meta = extract_text(tmp_path)
    except Exception as exc:
        # extract_text can raise RuntimeError (e.g. legacy .doc) or any
        # library-level error -- surface it as a user-visible message
        # instead of a blank 500.
        raise ValueError(f"Could not read {filename}: {exc}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    if not meta["ok"]:
        raise ValueError(meta["warning"])
    if len(text.strip()) < 80:
        raise ValueError(
            f"Could not extract enough readable text from {filename}. "
            "If this is a scanned/image PDF, OCR is disabled on this "
            "deployment — paste the text instead.")
    return text, meta


def _run_request(files, report_text):
    """Returns (views, errors). Each view = one analyzed report."""
    views, errors = [], []
    real_files = [f for f in files if f and f.filename]
    if real_files:
        if len(real_files) > MAX_BATCH_FILES:
            raise ValueError(
                f"Please upload at most {MAX_BATCH_FILES} files per batch.")
        for f in real_files:
            try:
                text, _meta = _extract_uploaded_file(f)
                views.append(_analyze_text(text, f.filename))
            except ValueError as exc:
                errors.append(f"{f.filename}: {exc}")
            except Exception as exc:  # never let one bad file blank the page
                errors.append(f"{f.filename}: unexpected error ({exc})")
        if not views and errors:
            # every file failed -> make sure the user sees why
            raise ValueError(" | ".join(errors))
    elif report_text:
        if len(report_text) < 80:
            raise ValueError("Pasted text is too short to analyze reliably.")
        views.append(_analyze_text(report_text, "Pasted text"))
    else:
        raise ValueError("Upload a report or paste report text.")
    return views, errors


# --------------------------------------------------------------------------
# Template
# --------------------------------------------------------------------------
PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroReport Insight — Clinical Report Analyzer</title>
<style>
:root{
  --bg:#f4f6fa; --card:#ffffff; --line:#e3e8f1; --fg:#17213a; --muted:#5e6b85;
  --accent:#2456d6; --accent-d:#1c44ab; --accent-soft:#e9effc;
  --ok:#0e8a4c; --ok-soft:#e2f5ea; --warn:#b96e00; --warn-soft:#fdf3de;
  --danger:#c93434; --danger-soft:#fdecec; --shadow:0 1px 2px rgba(16,24,40,.05),0 1px 3px rgba(16,24,40,.06);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
a{color:var(--accent);text-decoration:none}
.wrap{max-width:1040px;margin:0 auto;padding:0 20px 56px}
/* header */
.topbar{background:var(--card);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:9}
.topbar-in{max-width:1040px;margin:0 auto;padding:14px 20px;display:flex;align-items:center;gap:12px}
.logo{width:38px;height:38px;border-radius:10px;flex:0 0 auto;
  background:linear-gradient(135deg,#2456d6,#7c3aed);display:flex;align-items:center;justify-content:center}
.logo svg{width:22px;height:22px;stroke:#fff;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.brand b{font-size:16.5px;letter-spacing:-.01em}
.brand .sub{display:block}
.sub{color:var(--muted);font-size:12.5px}
.topbar .stats{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
.stat{background:var(--accent-soft);color:var(--accent-d);border-radius:8px;
  padding:5px 10px;font-size:12px;font-weight:600;white-space:nowrap}
/* cards & layout */
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:22px;margin:18px 0;box-shadow:var(--shadow)}
h1{font-size:22px;margin:26px 0 4px;letter-spacing:-.02em}
h2{font-size:16px;margin:0 0 12px}
h3{font-size:13.5px;margin:18px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.disclaimer{background:var(--warn-soft);border:1px solid #ecd9ae;color:#7a5200;
  padding:12px 16px;border-radius:12px;margin:14px 0;font-size:13px}
/* tabs */
.tabs{display:flex;gap:4px;background:#eef1f7;border-radius:10px;padding:4px;width:max-content;margin-bottom:18px}
.tab{border:0;background:transparent;color:var(--muted);font-size:13.5px;font-weight:600;
  padding:8px 18px;border-radius:8px;cursor:pointer}
.tab.active{background:var(--card);color:var(--fg);box-shadow:var(--shadow)}
.pane{display:none}.pane.active{display:block}
/* dropzone */
.drop{border:2px dashed #c6d0e2;border-radius:12px;padding:34px 20px;text-align:center;
  color:var(--muted);cursor:pointer;transition:.15s}
.drop.on,.drop:hover{border-color:var(--accent);background:var(--accent-soft);color:var(--accent-d)}
.drop b{color:var(--fg)}
.filelist{margin:10px 0 0;padding:0;list-style:none;font-size:13.5px}
.filelist li{display:flex;align-items:center;gap:8px;padding:7px 12px;margin-top:6px;
  background:#f7f9fc;border:1px solid var(--line);border-radius:8px}
.filelist .x{margin-left:auto;cursor:pointer;color:var(--danger);font-weight:700;border:0;background:none}
.vhide{position:absolute;width:1px;height:1px;opacity:0;overflow:hidden;clip:rect(0 0 0 0)}
.vhide.shown{position:static;width:auto;height:auto;opacity:1;clip:auto;margin-top:10px;display:block}
textarea{width:100%;min-height:230px;background:#fbfcfe;color:var(--fg);border:1px solid var(--line);
  border-radius:10px;padding:14px;font:inherit;resize:vertical}
textarea:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
.row{display:flex;gap:10px;align-items:center;margin-top:14px;flex-wrap:wrap}
.btn{background:var(--accent);color:#fff;border:0;padding:11px 22px;border-radius:9px;
  cursor:pointer;font-size:14px;font-weight:600}
.btn:hover{background:var(--accent-d)}
.btn.ghost{background:var(--card);color:var(--accent);border:1px solid var(--line)}
.btn.ghost:hover{background:var(--accent-soft)}
.btn.small{padding:7px 14px;font-size:12.5px}
.err{background:var(--danger-soft);border:1px solid #efc1c1;color:#8f2020;
  padding:12px 16px;border-radius:12px;margin:14px 0;font-size:13.5px}
/* result */
.flag{display:flex;align-items:center;gap:10px;background:var(--danger-soft);
  border:1px solid #efc1c1;color:#8f2020;padding:11px 16px;border-radius:12px;font-weight:700;margin:14px 0 0}
.okflag{display:flex;align-items:center;gap:10px;background:var(--ok-soft);
  border:1px solid #bfe5cd;color:#0c6b3c;padding:11px 16px;border-radius:12px;font-weight:700;margin:14px 0 0}
.res-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.res-head .fname{font-size:13px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:14px}
.cell{background:#f8fafd;border:1px solid var(--line);border-radius:12px;padding:14px}
.cell .k{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.cell .v{font-size:18px;font-weight:700;margin-top:4px}
.badge{display:inline-block;padding:3px 12px;border-radius:999px;font-size:12.5px;font-weight:700}
.badge.low{background:var(--ok-soft);color:var(--ok)}
.badge.moderate{background:var(--warn-soft);color:var(--warn)}
.badge.high{background:var(--danger-soft);color:var(--danger)}
.meter{height:8px;background:#e8edf5;border-radius:6px;overflow:hidden;margin-top:8px}
.meter>i{display:block;height:100%;border-radius:6px}
.chip{display:inline-block;background:var(--accent-soft);color:var(--accent-d);
  padding:4px 12px;border-radius:999px;margin:3px 5px 0 0;font-size:12.5px;font-weight:600}
.chip.gray{background:#eef1f6;color:var(--muted)}
/* probability bars */
.bars{margin-top:6px}
.bar-row{display:grid;grid-template-columns:170px 1fr 52px;gap:10px;align-items:center;margin:7px 0;font-size:13px}
.bar-track{height:10px;background:#e8edf5;border-radius:6px;overflow:hidden;display:block}
.bar-fill{height:100%;border-radius:6px;background:#b9c8e8;display:block}
.bar-row.win .bar-fill{background:var(--accent)}
.bar-row.win{font-weight:700}
.bar-pct{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
.ev{padding:9px 14px;margin:8px 0;font-size:13.5px;background:#f4f8ff;color:#2a3a5e;
  border-left:3px solid var(--accent);border-radius:0 10px 10px 0}
.hl-sent{background:#fff3c4;border-radius:3px;padding:1px 2px}
.hl-term{background:#cfe0ff;color:#16306e;border-radius:3px;padding:0 2px;font-weight:600}
details.report{margin-top:10px}
details.report summary{cursor:pointer;font-weight:600;color:var(--accent);font-size:13.5px}
.report-text{margin-top:10px;padding:16px;background:#fbfcfe;border:1px solid var(--line);
  border-radius:10px;font-size:13px;line-height:1.75;max-height:420px;overflow:auto}
.legend{font-size:12px;color:var(--muted);margin-top:8px}
.legend .hl-sent,.legend .hl-term{margin-right:4px}
/* batch summary table */
table.summary{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:6px}
table.summary th{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;
  text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
table.summary td{padding:9px 10px;border-bottom:1px solid var(--line)}
table.summary tr:last-child td{border-bottom:0}
.footer{margin-top:28px;color:var(--muted);font-size:12.5px;text-align:center}
@media print{
  .topbar,.input-card,.disclaimer,.no-print,.footer{display:none!important}
  body{background:#fff}
  .card{box-shadow:none;border-color:#ccc;break-inside:avoid}
  .report-text{max-height:none}
}
@media (max-width:640px){
  .bar-row{grid-template-columns:110px 1fr 46px}
  .topbar .stats{display:none}
}
</style>
</head>
<body>
<div class="topbar">
  <div class="topbar-in">
    <div class="logo">
      <svg viewBox="0 0 24 24"><path d="M3 12h4l2.5-7 4 14 2.5-7h5"/></svg>
    </div>
    <div class="brand"><b>NeuroReport Insight</b>
      <span class="sub">Clinical decision support for neurodevelopmental &amp; psychological reports</span>
    </div>
    {% if model_info %}
    <div class="stats">
      <span class="stat">Test accuracy {{ (model_info.test_accuracy * 100)|round(1) }}%</span>
      <span class="stat">Macro-F1 {{ (model_info.test_macro_f1 * 100)|round(1) }}%</span>
      <span class="stat">{{ model_info.n_reports }} training reports</span>
    </div>
    {% endif %}
  </div>
</div>

<div class="wrap">
  <h1>Analyze a clinical / school report</h1>
  <div class="sub">Upload up to {{ max_batch }} documents or paste text. Files are processed in memory,
  anonymized automatically, and never stored on the server.</div>

  <div class="disclaimer"><b>Decision support only.</b> {{ disclaimer }}</div>

  <div class="card input-card">
    <div class="tabs no-print">
      <button class="tab active" data-pane="upload" type="button">Upload files</button>
      <button class="tab" data-pane="paste" type="button">Paste text</button>
    </div>
    <form method="post" action="/analyze" enctype="multipart/form-data" id="aform">
      <div class="pane active" id="pane-upload">
        <div class="drop" id="drop">
          <b>Drop report files here</b> or click to browse<br>
          <span class="sub">.pdf · .docx · .txt · .md — up to {{ max_batch }} files, 4&nbsp;MB total. OCR/image scans are disabled on this live deployment.</span>
        </div>
        <input type="file" name="reports" id="finput" accept=".pdf,.docx,.txt,.md" multiple class="vhide">
        <ul class="filelist" id="flist"></ul>
        <div class="err" id="upload-err" style="display:none"></div>
      </div>
      <div class="pane" id="pane-paste">
        <textarea name="report_text" id="ptext" placeholder="Paste the report text here (at least 80 characters)…">{{ pasted or "" }}</textarea>
        <div class="row no-print">
          <button class="btn ghost small" type="button" id="sample-btn">Load synthetic sample</button>
          <span class="sub">A made-up example so you can try the analyzer instantly.</span>
        </div>
      </div>
      <div class="row">
        <button class="btn" type="submit" id="go">Analyze report</button>
        <span class="sub" id="busy" style="display:none">Analyzing…</span>
      </div>
    </form>
  </div>

  {% for e in errors %}<div class="err">{{ e }}</div>{% endfor %}
  {% if error %}<div class="err">{{ error }}</div>{% endif %}

  {% if views and views|length > 1 %}
  <div class="card">
    <h2>Batch summary — {{ views|length }} reports</h2>
    <table class="summary">
      <tr><th>Report</th><th>Diagnosis</th><th>Confidence</th><th>Risk</th><th>Review</th></tr>
      {% for v in views %}
      <tr>
        <td><a href="#r{{ loop.index }}">{{ v.name }}</a></td>
        <td><b>{{ v.result.diagnosis }}</b></td>
        <td>{{ v.confidence_pct }}% ({{ v.result.confidence_band }})</td>
        <td><span class="badge {{ v.result.risk_level|lower }}">{{ v.result.risk_level }}</span></td>
        <td>{{ "Needs review" if v.result.needs_doctor_review else "—" }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}

  {% for v in views %}
  {% set r = v.result %}
  <div class="card" id="r{{ loop.index }}">
    <div class="res-head">
      <h2 style="margin:0">Analysis result</h2>
      <span class="fname">{{ v.name }}</span>
      <span style="margin-left:auto" class="no-print">
        <button class="btn ghost small" type="button" onclick="dlJson({{ loop.index }})">Download JSON</button>
        <button class="btn ghost small" type="button" onclick="window.print()">Print / PDF</button>
      </span>
    </div>

    {% if r.needs_doctor_review %}
    <div class="flag">&#9888;&#65039; Needs doctor review — do not act on this result without a qualified clinician.</div>
    {% else %}
    <div class="okflag">&#10003; No automatic review trigger — still confirm with a clinician before any decision.</div>
    {% endif %}

    <div class="grid">
      <div class="cell">
        <div class="k">Primary condition / diagnosis</div>
        <div class="v">{{ r.diagnosis }}</div>
        {% if r.secondary_diagnoses %}
        <div class="sub" style="margin-top:6px">Secondary: <b>{{ r.secondary_diagnoses|join(", ") }}</b></div>
        {% endif %}
      </div>
      <div class="cell">
        <div class="k">Confidence ({{ r.confidence_band }})</div>
        <div class="v">{{ v.confidence_pct }}%</div>
        <div class="meter"><i style="width:{{ v.confidence_pct }}%;background:{{ v.conf_color }}"></i></div>
      </div>
      <div class="cell">
        <div class="k">Risk level</div>
        <div class="v"><span class="badge {{ r.risk_level|lower }}">{{ r.risk_level }}</span></div>
        {% if r.risk_signals %}<div class="sub" style="margin-top:6px">signals: {{ r.risk_signals|join(", ") }}</div>{% endif %}
      </div>
      <div class="cell">
        <div class="k">Decision margin</div>
        <div class="v">{{ (r.decision_margin * 100)|round|int if r.decision_margin is defined else "–" }}%</div>
        <div class="sub" style="margin-top:6px">gap between the top two candidate conditions</div>
      </div>
    </div>

    <h3>Condition probabilities</h3>
    <div class="bars">
      {% for name, p in r.diagnosis_probs or r.diagnosis_ranked %}
      <div class="bar-row {{ 'win' if loop.first }}">
        <span>{{ name }}</span>
        <span class="bar-track"><span class="bar-fill" style="width:{{ (p * 100)|round(1) }}%"></span></span>
        <span class="bar-pct">{{ (p * 100)|round(1) }}%</span>
      </div>
      {% endfor %}
    </div>

    <h3>Co-occurring conditions</h3>
    <div>{% for item in r.cooccurring %}<span class="chip">{{ item }}</span>{% else %}<span class="chip gray">None detected</span>{% endfor %}</div>

    <h3>Key symptoms found</h3>
    <div>{% for item in r.symptoms %}<span class="chip gray">{{ item }}</span>{% else %}<span class="chip gray">None matched</span>{% endfor %}</div>

    <h3>Probable root cause / contributing factors</h3>
    <div class="bars">
      {% for name, p in r.root_cause_top %}
      <div class="bar-row {{ 'win' if loop.first }}">
        <span>{{ name }}</span>
        <span class="bar-track"><span class="bar-fill" style="width:{{ (p * 100)|round(1) }}%"></span></span>
        <span class="bar-pct">{{ (p * 100)|round(1) }}%</span>
      </div>
      {% endfor %}
    </div>
    {% if r.root_cause_detail %}
      {% set rc = r.root_cause_detail %}
      <div class="sub" style="margin-top:8px">
        {{ "Learned and clinical views agree" if rc.agreement else
           "Learned and clinical views disagree" }}
        · confidence {{ (rc.confidence * 100)|round|int }}%
        · {{ "uncertain — expert review required" if rc.abstain else "consistent" }}
      </div>
      {% for group, sentences in rc.evidence.items() %}
        {% if sentences %}
          <div class="k" style="margin-top:12px">{{ group }} evidence</div>
          {% for sentence in sentences %}<div class="ev">{{ sentence }}</div>{% endfor %}
        {% endif %}
      {% endfor %}
    {% elif r.root_cause_factors %}
      <div class="sub" style="margin-top:6px">Factors detected: {{ r.root_cause_factors|join(", ") }}</div>
    {% endif %}

    <h3>Evidence from report</h3>
    {% for item in r.evidence %}<div class="ev">{{ item }}</div>{% else %}<div class="sub">No evidence sentences extracted.</div>{% endfor %}

    <details class="report">
      <summary>View anonymized report with highlighted evidence</summary>
      <div class="legend">
        <mark class="hl-term">model evidence term</mark>
        <span class="hl-sent">evidence sentence</span>
      </div>
      <div class="report-text">{{ v.highlight_html|safe }}</div>
    </details>

    <h3>Recommendation</h3>
    <div>{{ r.recommendation }}</div>

    <h3>Anonymization</h3>
    <div class="sub">{{ v.anon_summary }}</div>
  </div>
  <script type="application/json" id="json-{{ loop.index }}">{{ v.result_json }}</script>
  {% endfor %}

  <div class="footer">
    Stateless demo deployment — uploads are analyzed in memory and discarded.
    {% if model_info %}Held-out test: accuracy {{ (model_info.test_accuracy * 100)|round(1) }}%,
    leakage-controlled {{ (model_info.leakage_controlled_accuracy * 100)|round(1) }}%.{% endif %}
    API: <code>POST /api/analyze</code> · <code>GET /api/model-info</code>
  </div>
</div>

<script>
// tabs
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.pane').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('pane-' + t.dataset.pane).classList.add('active');
}));
// dropzone -- with graceful degradation: if anything in the fancy path
// fails, the native file input is revealed and keeps working.
const drop = document.getElementById('drop'), finput = document.getElementById('finput'),
      flist = document.getElementById('flist'), uerr = document.getElementById('upload-err');
let picked = [];
let managed = true;            // can we rebuild input.files ourselves?
try { new DataTransfer(); } catch (e) { managed = false; }

function showUploadError(msg) { uerr.textContent = msg; uerr.style.display = 'block'; }
function fallbackToNative(msg) {
  managed = false;
  finput.classList.add('shown');
  if (msg) showUploadError(msg);
}
drop.addEventListener('click', () => {
  try { finput.click(); } catch (e) { fallbackToNative('Use the file picker below.'); }
});
['dragover','dragenter'].forEach(ev => drop.addEventListener(ev, e => {e.preventDefault(); drop.classList.add('on');}));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => {e.preventDefault(); drop.classList.remove('on');}));
drop.addEventListener('drop', e => {
  if (!managed) { fallbackToNative('Drag & drop is not supported by this browser — use the file picker below.'); return; }
  addFiles(e.dataTransfer.files);
});
finput.addEventListener('change', () => {
  uerr.style.display = 'none';
  if (managed) { addFiles(finput.files); }
  else { renderList(Array.from(finput.files), false); }
});
function addFiles(fl) {
  try {
    for (const f of Array.from(fl)) {
      if (picked.length >= {{ max_batch }}) { showUploadError('Maximum {{ max_batch }} files per batch.'); break; }
      if (!picked.some(p => p.name === f.name && p.size === f.size)) picked.push(f);
    }
    sync();
  } catch (e) { fallbackToNative('Use the file picker below.'); }
}
function rmFile(i) { picked.splice(i, 1); sync(); }
function sync() {
  const dt = new DataTransfer();
  picked.forEach(f => dt.items.add(f));
  finput.files = dt.files;
  renderList(picked, true);
}
function renderList(files, removable) {
  flist.innerHTML = files.map((f, i) =>
    `<li>&#128196; ${f.name} <span class="sub">(${(f.size/1024).toFixed(0)} KB)</span>` +
    (removable ? `<button type="button" class="x" onclick="rmFile(${i})">&times;</button>` : '') +
    `</li>`).join('');
}
// sample text
document.getElementById('sample-btn').addEventListener('click', () => {
  document.getElementById('ptext').value = {{ sample_text|tojson }};
});
// submit: validate before sending so an empty submit can't look like a failure
document.getElementById('aform').addEventListener('submit', e => {
  const onUpload = document.getElementById('pane-upload').classList.contains('active');
  const nFiles = finput.files ? finput.files.length : 0;
  const txt = document.getElementById('ptext').value.trim();
  if (onUpload && nFiles === 0 && !txt) {
    e.preventDefault();
    showUploadError('Choose at least one file first (click the box above or drag files into it).');
    return;
  }
  if (!onUpload && txt.length < 80 && nFiles === 0) {
    e.preventDefault();
    alert('Pasted text is too short to analyze reliably (need at least 80 characters).');
    return;
  }
  let totalKB = 0;
  for (const f of Array.from(finput.files || [])) totalKB += f.size / 1024;
  if (totalKB > 4096) {
    e.preventDefault();
    showUploadError('Files exceed the 4 MB upload limit — remove some files or use smaller ones.');
    return;
  }
  document.getElementById('go').disabled = true;
  document.getElementById('busy').style.display = 'inline';
});
// json download
function dlJson(i) {
  const data = document.getElementById('json-' + i).textContent;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([data], {type: 'application/json'}));
  a.download = 'analysis-' + i + '.json';
  a.click();
  URL.revokeObjectURL(a.href);
}
// jump to first result after analysis
{% if views %}window.addEventListener('load', () => {
  const el = document.getElementById('r1');
  if (el) el.scrollIntoView({behavior: 'smooth'});
});{% endif %}
</script>
</body>
</html>
"""


def _conf_color(band):
    return {"High": "#0e8a4c", "Moderate": "#b96e00", "Low": "#c93434"}.get(
        band, "#5e6b85")


def _render(views=None, errors=None, error=None, pasted=None):
    views = views or []
    for v in views:
        v["conf_color"] = _conf_color(v["result"].get("confidence_band"))
    return render_template_string(
        PAGE, views=views, errors=errors or [], error=error,
        disclaimer=DISCLAIMER, model_info=MODEL_INFO,
        max_batch=MAX_BATCH_FILES, sample_text=SAMPLE_TEXT, pasted=pasted)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return _render()


@app.post("/analyze")
def analyze_form():
    report_text = (request.form.get("report_text") or "").strip()
    files = request.files.getlist("reports") or []
    # backward compatibility with the old single-file field name
    legacy = request.files.get("report")
    if legacy and legacy.filename:
        files.append(legacy)
    try:
        views, errors = _run_request(files, report_text)
        return _render(views=views, errors=errors, pasted=report_text)
    except ValueError as exc:
        return _render(error=str(exc), pasted=report_text)


@app.post("/api/analyze")
def analyze_api():
    report_text = (request.form.get("report_text") or "").strip()
    if not report_text and request.is_json:
        report_text = ((request.get_json(silent=True) or {})
                       .get("report_text") or "").strip()
    files = request.files.getlist("reports") or []
    legacy = request.files.get("report")
    if legacy and legacy.filename:
        files.append(legacy)
    try:
        views, errors = _run_request(files, report_text)
        payload = [{
            "name": v["name"],
            "anonymization_summary": v["anon_summary"],
            "result": v["result"],
        } for v in views]
        body = {"ok": True, "count": len(payload), "analyses": payload,
                "errors": errors}
        # keep the old single-result shape for existing clients
        if len(payload) == 1:
            body["result"] = payload[0]["result"]
            body["anonymization_summary"] = payload[0]["anonymization_summary"]
        return jsonify(body)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.errorhandler(413)
def too_large(_e):
    msg = ("Upload too large — the combined size limit is 4 MB. "
           "Remove some files or upload a smaller/native-text version.")
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": msg}), 413
    return _render(error=msg), 413


@app.errorhandler(500)
def server_error(_e):
    msg = ("Something went wrong while analyzing the report. If this was a "
           "PDF, it may be a scanned/image file (OCR is disabled on this "
           "deployment) — try pasting the text instead.")
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": msg}), 500
    return _render(error=msg), 500


@app.get("/api/health")
def health():
    try:
        get_analyzer()
        return jsonify({"ok": True, "status": "ready"})
    except Exception as exc:  # pragma: no cover
        return jsonify({"ok": False, "status": str(exc)}), 500


@app.get("/api/model-info")
def model_info():
    return jsonify({"ok": True, "model_info": MODEL_INFO,
                    "disclaimer": DISCLAIMER})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
