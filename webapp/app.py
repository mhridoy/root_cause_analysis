"""
Local web app for the ASD / neurodevelopmental report analyzer.

Built on Python's standard-library http.server so it runs with NO third-party
web framework (no Flask install needed). Document extraction libraries
(pdfplumber/python-docx/pytesseract) are optional and only needed for those
formats; plain-text and the ML model work with stdlib + numpy alone.

Run:   python3 webapp/app.py
Then:  open http://127.0.0.1:8000

Features: upload report -> anonymize -> analyze -> structured explainable
result with highlighted evidence + confidence + risk; download result;
securely manage previous uploads (stored locally in SQLite, originals kept
out of the DB -- only anonymized text + results are retained).
"""
import os
import re
import io
import sys
import json
import html
import sqlite3
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, quote

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.extract import extract_text
from src.anonymize import anonymize, anonymization_summary
from src.analyze import get_analyzer, DISCLAIMER

# Vercel Functions have an ephemeral filesystem. In that environment results are
# rendered immediately and never written to SQLite. Local execution retains the
# existing report-history behavior.
SERVERLESS = bool(os.environ.get("VERCEL") or
                  os.environ.get("REPORTS_STATELESS"))
DEFAULT_DB = "/tmp/asd-report-analyzer.db" if SERVERLESS else os.path.join(HERE, "reports.db")
DEFAULT_TMP = "/tmp/asd-report-uploads" if SERVERLESS else os.path.join(HERE, "uploads")
DB_PATH = os.environ.get("REPORTS_DB", DEFAULT_DB)
UPLOAD_TMP = os.environ.get("REPORTS_TMP", DEFAULT_TMP)
os.makedirs(UPLOAD_TMP, exist_ok=True)
ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg",
               ".tif", ".tiff", ".bmp"}


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT, created TEXT,
            diagnosis TEXT, confidence REAL, confidence_band TEXT,
            risk_level TEXT, needs_review INTEGER,
            result_json TEXT, anon_text TEXT, anon_summary TEXT,
            extract_method TEXT)""")


def save_report(filename, result, anon_text, anon_summary, method):
    with db() as c:
        cur = c.execute(
            """INSERT INTO reports (filename, created, diagnosis, confidence,
               confidence_band, risk_level, needs_review, result_json,
               anon_text, anon_summary, extract_method)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (filename, datetime.datetime.now().isoformat(timespec="seconds"),
             result.get("diagnosis"), result.get("confidence"),
             result.get("confidence_band"), result.get("risk_level"),
             1 if result.get("needs_doctor_review") else 0,
             json.dumps(result), anon_text, anon_summary, method))
        return cur.lastrowid


def list_reports():
    with db() as c:
        return c.execute("SELECT * FROM reports ORDER BY id DESC").fetchall()


def get_report(rid):
    with db() as c:
        return c.execute("SELECT * FROM reports WHERE id=?", (rid,)).fetchone()


def delete_report(rid):
    with db() as c:
        c.execute("DELETE FROM reports WHERE id=?", (rid,))


# --------------------------------------------------------------------------
# Minimal multipart/form-data parser (cgi module removed in py3.13)
# --------------------------------------------------------------------------
def parse_multipart(body, boundary):
    files, fields = {}, {}
    delim = b"--" + boundary
    parts = body.split(delim)
    for part in parts:
        if not part or part in (b"--\r\n", b"--", b"\r\n"):
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        head_end = part.find(b"\r\n\r\n")
        if head_end == -1:
            continue
        raw_head = part[:head_end].decode("utf-8", "ignore")
        data = part[head_end + 4:]
        disp = ""
        for line in raw_head.split("\r\n"):
            if line.lower().startswith("content-disposition"):
                disp = line
        name_m = re.search(r'name="([^"]*)"', disp)
        file_m = re.search(r'filename="([^"]*)"', disp)
        if not name_m:
            continue
        name = name_m.group(1)
        if file_m:
            files[name] = (file_m.group(1), data)
        else:
            fields[name] = data.decode("utf-8", "ignore")
    return fields, files


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------
def esc(s):
    return html.escape(str(s if s is not None else ""))


STYLE = """
:root{--bg:#0f1320;--card:#181d2e;--muted:#8a93a8;--fg:#eef1f7;--accent:#5b8cff;
--ok:#34c77b;--warn:#f5a623;--danger:#ef5b6b;--line:#27304a}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
background:var(--bg);color:var(--fg)}
a{color:var(--accent);text-decoration:none}
.wrap{max-width:1080px;margin:0 auto;padding:24px}
header h1{font-size:20px;margin:0}
.sub{color:var(--muted);font-size:13px}
.disclaimer{background:#3a2a14;border:1px solid var(--warn);color:#ffd591;
padding:12px 16px;border-radius:10px;margin:16px 0;font-size:13px}
.grid{display:grid;grid-template-columns:300px 1fr;gap:20px;margin-top:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}
.card h2{font-size:14px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 10px}
.btn{background:var(--accent);color:#fff;border:0;padding:10px 16px;border-radius:8px;
cursor:pointer;font-size:14px}.btn.sec{background:#2a3350}.btn.danger{background:var(--danger)}
.drop{border:2px dashed var(--line);border-radius:10px;padding:22px;text-align:center;color:var(--muted)}
.list a{display:block;padding:8px 10px;border-radius:8px;color:var(--fg);font-size:13px}
.list a:hover{background:#222a42}.list .meta{color:var(--muted);font-size:11px}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600}
.b-high{background:rgba(239,91,107,.18);color:#ff96a2;border:1px solid var(--danger)}
.b-mod{background:rgba(245,166,35,.16);color:#ffce80;border:1px solid var(--warn)}
.b-low{background:rgba(52,199,123,.16);color:#7fe3ab;border:1px solid var(--ok)}
.kv{margin:10px 0}.kv .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.kv .v{font-size:15px;margin-top:2px}
.chip{display:inline-block;background:#222a42;border:1px solid var(--line);padding:4px 10px;
border-radius:16px;margin:3px 4px 0 0;font-size:13px}
.meter{height:10px;background:#222a42;border-radius:6px;overflow:hidden;margin-top:6px}
.meter > i{display:block;height:100%}
mark{background:#3a4d8a;color:#dce6ff;padding:0 2px;border-radius:3px}
.ev{background:#11162400;border-left:3px solid var(--accent);padding:8px 12px;margin:8px 0;
font-size:14px;color:#cdd6ec;background:#141a2c;border-radius:0 8px 8px 0}
.rank{font-size:13px;color:var(--muted)}
table{width:100%;border-collapse:collapse}td{padding:4px 0;font-size:13px}
.flag{background:rgba(239,91,107,.14);border:1px solid var(--danger);color:#ffb3bc;
padding:10px 14px;border-radius:8px;font-weight:600;margin-bottom:12px}
footer{color:var(--muted);font-size:12px;margin-top:30px;text-align:center}
"""


def page(body, title="Report Analyzer"):
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{STYLE}</style></head><body><div class=wrap>
<header><h1>🧩 Clinical Report Analyzer</h1>
<div class=sub>Autism &amp; neurodevelopmental report decision-support — local &amp; private</div></header>
<div class=disclaimer>⚠️ {esc(DISCLAIMER)}</div>
{body}
<footer>Runs locally on your machine. Reports are stored only on this computer.
Anonymized before storage. Not a medical device.</footer>
</div></body></html>"""


def confidence_color(band):
    return {"High": "var(--ok)", "Moderate": "var(--warn)", "Low": "var(--danger)"}.get(band, "var(--muted)")


def risk_badge(level):
    cls = {"High": "b-high", "Moderate": "b-mod", "Low": "b-low"}.get(level, "b-low")
    return f'<span class="badge {cls}">{esc(level)} risk</span>'


def index_page(msg=""):
    rows = [] if SERVERLESS else list_reports()
    items = ""
    for r in rows:
        rev = " ⚑" if r["needs_review"] else ""
        items += (f'<a href="/report/{r["id"]}"><b>{esc(r["diagnosis"])}</b>{rev}'
                  f'<div class=meta>{esc(r["filename"])[:38]} · '
                  f'{esc(r["confidence_band"])} conf · {esc(r["created"][:16])}</div></a>')
    if not items:
        items = '<div class=sub>No reports analyzed yet.</div>'
    msg_html = f'<div class=flag>{esc(msg)}</div>' if msg else ""
    upload = f"""
    <div class=card>
      <h2>Upload a child's report</h2>
      {msg_html}
      <form method=post action=/upload enctype=multipart/form-data>
        <div class=drop>
          <p>Choose a PDF, DOCX, image/scan, or text report.</p>
          <input type=file name=report accept=".pdf,.docx,.txt,.md,.png,.jpg,.jpeg,.tif,.tiff,.bmp" required>
        </div>
        <p style="margin-top:14px"><button class=btn type=submit>Analyze report</button></p>
        <div class=sub>Files are anonymized on upload. Supported model classes:
        ADHD, ASD, Depression, Dyslexia, GAD, OCD (others flagged for review).</div>
      </form>
    </div>"""
    if SERVERLESS:
        sidebar = ('<div class=card><h2>Privacy mode</h2>'
                   '<div class=sub>Hosted uploads are analyzed in memory and are '
                   'not saved after the response. Download the result before '
                   'leaving the page.</div></div>')
    else:
        sidebar = f'<div class=card><h2>Previous reports ({len(rows)})</h2><div class=list>{items}</div></div>'
    return page(f'<div class=grid><div>{sidebar}</div><div>{upload}</div></div>')


def render_cooccurring(res):
    """Tiered co-occurring rendering: Likely / Possible / Ruled-out (denied)."""
    d = res.get("cooccurring_detail")
    if not d:  # legacy flat list
        items = res.get("cooccurring", [])
        return ("".join(f'<span class=chip>{esc(c)}</span>' for c in items)
                or "<span class=sub>None detected</span>")
    out = ""
    if d.get("likely"):
        out += ('<div class=k>Likely</div>' + "".join(
            f'<span class=chip>{esc(c)}</span>' for c in d["likely"]))
    if d.get("possible"):
        out += ('<div class=k style="margin-top:6px">Possible (weak)</div>' +
                "".join(f'<span class=chip style="opacity:.75">{esc(c)}</span>'
                        for c in d["possible"]))
    if d.get("ruled_out"):
        out += ('<div class=k style="margin-top:6px">Explicitly ruled out / denied</div>' +
                "".join(f'<span class=chip style="opacity:.5;text-decoration:line-through">'
                        f'{esc(c)}</span>' for c in d["ruled_out"]))
    if not out:
        out = "<span class=sub>None clearly supported</span>"
    return out


def render_rootcause(res):
    """Rich, explainable root-cause block: ranked bars, learned-vs-clinical
    agreement, abstain badge, and per-factor evidence."""
    rc = res.get("root_cause_detail")
    if not rc:  # legacy / fallback
        rows = "".join(f'<div class=rank>{esc(g)} — {int(round(p*100))}%</div>'
                       for g, p in res.get("root_cause_top", []))
        return rows or "<span class=sub>n/a</span>"
    bars = ""
    for g, p in rc["ranked"][:5]:
        pct = int(round(p * 100))
        bars += (f'<div style="margin:6px 0"><div style="display:flex;'
                 f'justify-content:space-between;font-size:13px">'
                 f'<span>{esc(g)}</span><span class=sub>{pct}%</span></div>'
                 f'<div class=meter><i style="width:{pct}%;background:var(--accent)"></i></div></div>')
    if rc["abstain"]:
        badge = ('<span class="badge b-mod">⚠ uncertain — flagged for review</span>')
    else:
        badge = ('<span class="badge b-low">consistent</span>')
    agree = ("learned model and clinical expectation <b>agree</b>"
             if rc["agreement"] else
             f"learned model says <b>{esc(rc['learned_top'])}</b> but clinical "
             f"expectation says <b>{esc(rc['clinical_top'])}</b> — <b>disagreement</b>")
    # per-factor evidence
    ev_html = ""
    for g, sents in rc.get("evidence", {}).items():
        if sents:
            inner = "".join(f'<div class=ev>{esc(s)}</div>' for s in sents)
            ev_html += f'<div style="margin-top:8px"><div class=k>{esc(g)}</div>{inner}</div>'
    top_line = (f'<div class=v style="font-size:18px;font-weight:700">{esc(rc["top"])}</div>'
                f'<div class=sub style="margin:4px 0">{badge} · confidence '
                f'{int(round(rc["confidence"]*100))}% · {agree}</div>')
    return top_line + bars + ev_html


def highlight(text, terms):
    out = esc(text)
    for t in sorted(set(terms), key=len, reverse=True):
        if len(t) < 3:
            continue
        out = re.sub(f"({re.escape(esc(t))})", r"<mark>\1</mark>", out, flags=re.IGNORECASE)
    return out


def report_page(r, ephemeral=False):
    res = json.loads(r["result_json"])
    band = res["confidence_band"]
    pct = int(round(res["confidence"] * 100))
    ranked = "".join(
        f'<div class=rank>{esc(d)} — {int(round(p*100))}%</div>'
        for d, p in res["diagnosis_ranked"])
    cooc = render_cooccurring(res)
    symp = "".join(f'<span class=chip>{esc(s)}</span>' for s in res["symptoms"]) or "<span class=sub>None matched</span>"
    rootcause_html = render_rootcause(res)
    ev_terms = res.get("evidence_terms", []) + res.get("symptoms", [])
    evidence = "".join(f'<div class=ev>{highlight(s, ev_terms)}</div>'
                       for s in res["evidence"]) or "<span class=sub>No clear evidence sentences extracted.</span>"
    flag = '<div class=flag>⚑ NEEDS DOCTOR REVIEW</div>' if res["needs_doctor_review"] else ""
    risk_extra = ""
    if res.get("risk_signals"):
        risk_extra = " · signals: " + ", ".join(esc(x) for x in res["risk_signals"])
    if res.get("risk_denied"):
        risk_extra += ('<div class=sub style="margin-top:4px">Explicitly denied in '
                       'report (not counted): ' +
                       ", ".join(esc(x) for x in dict.fromkeys(res["risk_denied"])) +
                       "</div>")

    if ephemeral:
        json_url = ("data:application/json;charset=utf-8," +
                    quote(r["result_json"]))
        text_url = ("data:text/plain;charset=utf-8," +
                    quote(result_to_text(r)))
        actions = f"""
          <a class=btn download="analysis.json" href="{json_url}">⬇ Download result (JSON)</a>
          <a class=btn sec download="analysis.txt" href="{text_url}" style="margin-left:8px">⬇ Download (text)</a>
        """
    else:
        actions = f"""
          <a class=btn href="/download/{r['id']}">⬇ Download result (JSON)</a>
          <a class=btn sec href="/download/{r['id']}?fmt=txt" style="margin-left:8px">⬇ Download (text)</a>
          <form method=post action="/delete/{r['id']}" style="display:inline">
            <button class="btn danger" style="margin-left:8px"
              onclick="return confirm('Delete this report permanently?')">Delete</button></form>
        """

    body = f"""
    <p><a href="/">← Back</a></p>
    {flag}
    <div class=card>
      <h2>Structured Result — {esc(r['filename'])}</h2>
      <div class=kv><div class=k>Primary Condition / Diagnosis</div>
        <div class=v style="font-size:20px;font-weight:700">{esc(res['diagnosis'])}</div>
        {('<div class=sub style="margin-top:6px">Secondary: <b>'
          + esc(', '.join(res.get('secondary_diagnoses', []))) + '</b></div>')
         if res.get('secondary_diagnoses') else ''}
        {ranked}
        <div class=sub style="margin-top:6px">{esc(res.get('explanation',''))}</div></div>
      <div class=kv><div class=k>Confidence Score</div>
        <div class=v style="color:{confidence_color(band)};font-weight:700">{pct}% ({esc(band)})</div>
        <div class=meter><i style="width:{pct}%;background:{confidence_color(band)}"></i></div></div>
      <div class=kv><div class=k>Risk Level</div><div class=v>{risk_badge(res['risk_level'])}{risk_extra}</div></div>
      <div class=kv><div class=k>Needs Doctor Review</div>
        <div class=v>{'<b style=color:#ff96a2>Yes</b>' if res['needs_doctor_review'] else '<b style=color:#7fe3ab>No</b>'}</div></div>
    </div>

    <div class=card><h2>Co-occurring Diseases or Disorders</h2>{cooc}</div>
    <div class=card><h2>Key Symptoms Found</h2>{symp}</div>
    <div class=card><h2>Probable Root Cause / Contributing Factors</h2>
      {rootcause_html}</div>
    <div class=card><h2>Evidence from Report</h2>{evidence}</div>
    <div class=card><h2>Recommendation</h2><div class=v>{esc(res['recommendation'])}</div></div>
    <div class=card><h2>Privacy</h2><div class=sub>{esc(r['anon_summary'])} · Extraction: {esc(r['extract_method'])}</div></div>

    <p>
      {actions}
    </p>"""
    return page(body, f"Result — {r['filename']}")


def result_to_text(r):
    res = json.loads(r["result_json"])
    L = []
    L.append("CLINICAL REPORT ANALYSIS (NOT A MEDICAL DIAGNOSIS)")
    L.append("=" * 60)
    L.append(f"File: {r['filename']}")
    L.append(f"Generated: {r['created']}")
    L.append("")
    L.append(f"Primary Condition / Diagnosis: {res['diagnosis']}")
    if res.get("secondary_diagnoses"):
        L.append("Secondary Diagnosis: " + ", ".join(res["secondary_diagnoses"]))
    L.append("  Ranked: " + ", ".join(f"{d} ({int(round(p*100))}%)" for d, p in res['diagnosis_ranked']))
    if res.get("explanation"):
        L.append(f"  Basis: {res['explanation']}")
    _cd = res.get("cooccurring_detail")
    if _cd:
        L.append("Co-occurring Diseases or Disorders:")
        L.append(f"  Likely: {', '.join(_cd['likely']) or 'none'}")
        L.append(f"  Possible (weak): {', '.join(_cd['possible']) or 'none'}")
        if _cd.get("ruled_out"):
            L.append(f"  Explicitly ruled out / denied: {', '.join(_cd['ruled_out'])}")
    else:
        L.append(f"Co-occurring Diseases or Disorders: {', '.join(res['cooccurring']) or 'None detected'}")
    L.append(f"Key Symptoms Found: {', '.join(res['symptoms']) or 'None matched'}")
    L.append("Probable Root Cause / Contributing Factors: " +
             ", ".join(f"{g} ({int(round(p*100))}%)" for g, p in res['root_cause_top']))
    _rc = res.get("root_cause_detail")
    if _rc:
        L.append(f"  Root-cause method: learned model blended with diagnosis-informed "
                 f"clinical posterior (learned_top={_rc['learned_top']}, "
                 f"clinical_top={_rc['clinical_top']}, "
                 f"{'UNCERTAIN/flagged' if _rc['abstain'] else 'consistent'})")
    L.append("Evidence from Report:")
    for e in res['evidence']:
        L.append(f"  - {e}")
    L.append(f"Confidence Score: {int(round(res['confidence']*100))}% ({res['confidence_band']})")
    L.append(f"Risk Level: {res['risk_level']}"
             + (f" (signals: {', '.join(res['risk_signals'])})" if res.get('risk_signals') else ""))
    if res.get("risk_denied"):
        L.append("  Explicitly denied in report (not counted as risk): "
                 + ", ".join(dict.fromkeys(res["risk_denied"])))
    L.append(f"Recommendation: {res['recommendation']}")
    L.append(f"Needs Doctor Review: {'Yes' if res['needs_doctor_review'] else 'No'}")
    L.append("")
    L.append("DISCLAIMER: " + res['disclaimer'])
    return "\n".join(L)


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8", headers=None):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass  # quiet

    def _route(self):
        """Return the public route, including paths forwarded by Vercel."""
        u = urlparse(self.path)
        q = parse_qs(u.query, keep_blank_values=True)
        forwarded = q.pop("_path", [None])[0]
        if forwarded is not None:
            path = "/" + forwarded.lstrip("/")
        elif SERVERLESS and u.path == "/api/index":
            path = "/"
        else:
            path = u.path
        return path, q

    def do_GET(self):
        path, q = self._route()
        if path == "/" or path == "":
            self._send(200, index_page(q.get("msg", [""])[0]))
        elif path.startswith("/report/"):
            rid = path.split("/")[-1]
            r = get_report(rid)
            self._send(200, report_page(r) if r else page("<p>Not found.</p>"))
        elif path.startswith("/download/"):
            rid = path.split("/")[-1]
            r = get_report(rid)
            if not r:
                self._send(404, page("<p>Not found.</p>")); return
            if q.get("fmt", [""])[0] == "txt":
                self._send(200, result_to_text(r), "text/plain; charset=utf-8",
                           {"Content-Disposition": f'attachment; filename="analysis_{rid}.txt"'})
            else:
                self._send(200, r["result_json"], "application/json",
                           {"Content-Disposition": f'attachment; filename="analysis_{rid}.json"'})
        else:
            self._send(404, page("<p>Not found.</p>"))

    def do_POST(self):
        path, _ = self._route()
        if path == "/upload":
            self._handle_upload()
        elif path.startswith("/delete/"):
            rid = path.split("/")[-1]
            delete_report(rid)
            self._send(303, "", headers={"Location": "/"})
        else:
            self._send(404, page("<p>Not found.</p>"))

    def _handle_upload(self):
        ctype = self.headers.get("Content-Type", "")
        m = re.search(r"boundary=(.+)", ctype)
        if not m:
            self._redirect_msg("Upload failed: bad form encoding."); return
        boundary = m.group(1).strip('"').encode()
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _, files = parse_multipart(body, boundary)
        if "report" not in files or not files["report"][1]:
            self._redirect_msg("No file received."); return
        filename, data = files["report"]
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXT:
            self._redirect_msg(f"Unsupported file type: {ext}"); return

        tmp = os.path.join(UPLOAD_TMP, "incoming" + ext)
        with open(tmp, "wb") as f:
            f.write(data)
        try:
            text, meta = extract_text(tmp)
        finally:
            try:
                os.remove(tmp)  # do not retain original (privacy)
            except OSError:
                pass
        if not meta["ok"]:
            self._redirect_msg(meta["warning"]); return
        if len(text.strip()) < 80:
            self._redirect_msg("Could not read enough text from the file "
                               "(possibly a low-quality scan)."); return

        anon_text, counts = anonymize(text)
        summary = anonymization_summary(counts)
        result = get_analyzer().analyze(anon_text)
        if result.get("error"):
            self._redirect_msg(result["error"]); return
        # Out-of-domain files are refused, not stored as "reports".
        if result.get("out_of_domain"):
            self._redirect_msg("❌ '" + filename + "' was not recognized as a "
                               "clinical assessment report, so it was not "
                               "analyzed. " + result["recommendation"])
            return
        if SERVERLESS:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds")
            row = {
                "id": "current",
                "filename": filename,
                "created": now,
                "result_json": json.dumps(result),
                "anon_summary": summary,
                "extract_method": meta["method"],
            }
            self._send(200, report_page(row, ephemeral=True))
        else:
            rid = save_report(filename, result, anon_text, summary, meta["method"])
            self._send(303, "", headers={"Location": f"/report/{rid}"})

    def _redirect_msg(self, msg):
        from urllib.parse import quote
        self._send(303, "", headers={"Location": f"/?msg={quote(msg)}"})


def main(port=8000):
    init_db()
    get_analyzer()  # warm-load model
    print(f"Report Analyzer running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    main(p)
