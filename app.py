import json
import os
import tempfile

from flask import Flask, jsonify, render_template_string, request

from src.analyze import DISCLAIMER, get_analyzer
from src.anonymize import anonymization_summary, anonymize
from src.extract import extract_text

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

ALLOWED_EXT = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}
OCR_ONLY_EXT = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}

STYLE = """
<style>
:root{--bg:#0f1320;--card:#181d2e;--muted:#8a93a8;--fg:#eef1f7;--accent:#5b8cff;
--ok:#34c77b;--warn:#f5a623;--danger:#ef5b6b;--line:#27304a}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
background:var(--bg);color:var(--fg)}
a{color:var(--accent);text-decoration:none}
.wrap{max-width:1080px;margin:0 auto;padding:24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0}
.sub{color:var(--muted);font-size:13px}
.disclaimer{background:#3a2a14;border:1px solid var(--warn);color:#ffd591;
padding:12px 16px;border-radius:10px;margin:16px 0;font-size:13px}
.btn{background:var(--accent);color:#fff;border:0;padding:10px 16px;border-radius:8px;cursor:pointer;font-size:14px}
.chip{display:inline-block;background:#222a42;border:1px solid var(--line);padding:4px 10px;
border-radius:16px;margin:3px 4px 0 0;font-size:13px}
.kv{margin:10px 0}.kv .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.kv .v{font-size:15px;margin-top:2px}.meter{height:10px;background:#222a42;border-radius:6px;overflow:hidden;margin-top:6px}
.meter>i{display:block;height:100%}.ev{padding:8px 12px;margin:8px 0;font-size:14px;color:#cdd6ec;background:#141a2c;border-left:3px solid var(--accent);border-radius:0 8px 8px 0}
.flag{background:rgba(239,91,107,.14);border:1px solid var(--danger);color:#ffb3bc;padding:10px 14px;border-radius:8px;font-weight:600;margin-bottom:12px}
.err{background:rgba(239,91,107,.14);border:1px solid var(--danger);color:#ffb3bc;padding:12px 16px;border-radius:10px;margin:16px 0}
textarea,input[type=file]{width:100%}textarea{min-height:220px;background:#111624;color:var(--fg);border:1px solid var(--line);border-radius:10px;padding:12px}
input[type=file]{color:var(--muted)}pre{white-space:pre-wrap;word-break:break-word;background:#111624;color:#dbe5ff;border:1px solid var(--line);padding:14px;border-radius:10px;overflow:auto}
</style>
"""

PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AI Report Analyzer</title>
  {{ style|safe }}
</head>
<body>
  <div class="wrap">
    <h1>AI-Assisted Report Analyzer</h1>
    <div class="sub">Vercel demo deployment: stateless analysis of report text.</div>
    <div class="disclaimer">{{ disclaimer }}</div>
    <div class="card">
      <h2>Upload a report or paste text</h2>
      <div class="sub">Supported in this live deployment: .txt, .md, .docx, native-text .pdf. OCR image/scanned uploads are disabled on Vercel.</div>
      <form method="post" action="/analyze" enctype="multipart/form-data">
        <p><input type="file" name="report" accept=".pdf,.docx,.txt,.md"></p>
        <p class="sub">or paste report text below</p>
        <p><textarea name="report_text" placeholder="Paste report text here"></textarea></p>
        <p><button class="btn" type="submit">Analyze</button></p>
      </form>
    </div>
    {% if error %}
    <div class="err">{{ error }}</div>
    {% endif %}
    {% if result %}
    {% if result.needs_doctor_review %}
    <div class="flag">Needs Doctor Review</div>
    {% endif %}
    <div class="card">
      <div class="kv"><div class="k">Child Condition / Diagnosis</div><div class="v">{{ result.diagnosis }}</div></div>
      <div class="kv"><div class="k">Confidence</div><div class="v">{{ confidence_pct }}% ({{ result.confidence_band }})</div><div class="meter"><i style="width:{{ confidence_pct }}%;background:{{ confidence_color }}"></i></div></div>
      <div class="kv"><div class="k">Risk Level</div><div class="v">{{ result.risk_level }}</div></div>
      <div class="kv"><div class="k">Co-occurring Conditions</div><div class="v">{% for item in result.cooccurring %}<span class="chip">{{ item }}</span>{% else %}None detected{% endfor %}</div></div>
      <div class="kv"><div class="k">Symptoms Found</div><div class="v">{% for item in result.symptoms %}<span class="chip">{{ item }}</span>{% else %}None matched{% endfor %}</div></div>
      <div class="kv"><div class="k">Probable Root Cause / Contributing Factors</div><div class="v">{% for item in result.root_cause_top %}<div>{{ item[0] }} - {{ (item[1] * 100)|round|int }}%</div>{% endfor %}</div></div>
      <div class="kv"><div class="k">Evidence</div><div class="v">{% for item in result.evidence %}<div class="ev">{{ item }}</div>{% else %}No evidence sentences extracted{% endfor %}</div></div>
      <div class="kv"><div class="k">Recommendation</div><div class="v">{{ result.recommendation }}</div></div>
      <div class="kv"><div class="k">Anonymization Summary</div><div class="v">{{ anon_summary }}</div></div>
    </div>
    <div class="card">
      <h2>JSON Output</h2>
      <pre>{{ result_json }}</pre>
    </div>
    {% endif %}
  </div>
</body>
</html>
"""


def _confidence_color(band):
    return {
        "High": "#34c77b",
        "Moderate": "#f5a623",
        "Low": "#ef5b6b",
    }.get(band, "#8a93a8")


def _analyze_text(text):
    anon_text, counts = anonymize(text)
    result = get_analyzer().analyze(anon_text)
    return result, anonymization_summary(counts)


def _extract_uploaded_file(upload):
    filename = upload.filename or "report"
    ext = os.path.splitext(filename)[1].lower()
    if ext in OCR_ONLY_EXT:
        raise ValueError("Image/OCR uploads are disabled in the Vercel deployment. Use text, DOCX, or native-text PDF.")
    if ext not in ALLOWED_EXT:
        raise ValueError(f"Unsupported file type: {ext or 'unknown'}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir="/tmp") as tmp:
        upload.save(tmp)
        tmp_path = tmp.name
    try:
        text, meta = extract_text(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if not meta["ok"]:
        raise ValueError(meta["warning"])
    if len(text.strip()) < 80:
        raise ValueError("Could not extract enough readable text from the uploaded file.")
    return text, meta


@app.get("/")
def index():
    return render_template_string(PAGE, style=STYLE, disclaimer=DISCLAIMER, result=None, error=None)


@app.post("/analyze")
def analyze_form():
    report_text = (request.form.get("report_text") or "").strip()
    upload = request.files.get("report")
    error = None
    result = None
    anon_summary = ""

    try:
        if upload and upload.filename:
            text, _meta = _extract_uploaded_file(upload)
            result, anon_summary = _analyze_text(text)
        elif report_text:
            if len(report_text) < 80:
                raise ValueError("Pasted text is too short to analyze reliably.")
            result, anon_summary = _analyze_text(report_text)
        else:
            raise ValueError("Upload a report or paste report text.")
        if result.get("error"):
            raise ValueError(result["error"])
    except ValueError as exc:
        error = str(exc)

    return render_template_string(
        PAGE,
        style=STYLE,
        disclaimer=DISCLAIMER,
        result=result,
        error=error,
        anon_summary=anon_summary,
        result_json=json.dumps(result, indent=2) if result else "",
        confidence_pct=int(round((result or {}).get("confidence", 0) * 100)),
        confidence_color=_confidence_color((result or {}).get("confidence_band")),
    )


@app.post("/api/analyze")
def analyze_api():
    report_text = (request.form.get("report_text") or "").strip()
    upload = request.files.get("report")

    try:
        if upload and upload.filename:
            text, meta = _extract_uploaded_file(upload)
        elif report_text:
            if len(report_text) < 80:
                raise ValueError("Pasted text is too short to analyze reliably.")
            text = report_text
            meta = {"method": "pasted-text"}
        else:
            raise ValueError("Upload a report or include report_text.")

        result, anon_summary = _analyze_text(text)
        if result.get("error"):
            raise ValueError(result["error"])
        return jsonify({
            "ok": True,
            "extract_method": meta["method"],
            "anonymization_summary": anon_summary,
            "result": result,
        })
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
