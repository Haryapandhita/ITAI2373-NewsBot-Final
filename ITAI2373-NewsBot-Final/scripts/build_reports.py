"""
Render the Markdown documents to professional PDFs (headless Chrome).

    python scripts/build_reports.py

Inputs : docs/technical_documentation.md, reports/src/executive_summary.md,
         reports/src/reflective_journal.md
Outputs: reports/*.pdf (+ Canvas-named copies in ../Canvas_Submission/, outside the repository)
"""

import base64
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SUBMITTER, GROUP = "MourinhoHernanto", "Solo"

CSS = """
@page { size: Letter; margin: 0.8in 0.8in 0.85in 0.8in;
        @bottom-center { content: counter(page); } }
body { font-family: 'Inter', 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1e293b; }
h1 { font-size: 22pt; color: #1d4ed8; margin: 0 0 4pt 0; }
h2 { font-size: 14.5pt; color: #0f172a; border-bottom: 2px solid #dbeafe; padding-bottom: 3pt; margin-top: 20pt; page-break-after: avoid; }
h3 { font-size: 12pt; color: #1e3a8a; margin-top: 14pt; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin-top: 10pt; }
p, li { text-align: left; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt 0; font-size: 9.2pt; page-break-inside: avoid; }
th { background: #eff6ff; color: #1e3a8a; text-align: left; }
th, td { border: 1px solid #cbd5e1; padding: 4pt 6pt; vertical-align: top; }
tr:nth-child(even) td { background: #f8fafc; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.8pt; background: #f1f5f9; padding: 0 2pt; border-radius: 2pt; }
pre { background: #0f172a; color: #e2e8f0; padding: 8pt 10pt; border-radius: 4pt; font-size: 7.6pt; line-height: 1.35;
      white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; color: inherit; padding: 0; }
blockquote { border-left: 4px solid #93c5fd; margin: 8pt 0; padding: 4pt 10pt; background: #f8fafc; color: #334155; }
img { max-width: 100%; display: block; margin: 8pt auto; page-break-inside: avoid; }
.figure { text-align: center; font-size: 9pt; color: #475569; margin-bottom: 10pt; page-break-inside: avoid; }
.cover { text-align: left; border-left: 6px solid #2563eb; padding: 6pt 0 6pt 14pt; margin-bottom: 18pt; }
.cover .sub { font-size: 12pt; color: #475569; }
.kpi { box-sizing: border-box; display: inline-block; width: 23.5%; margin: 0 1.3% 8pt 0; padding: 7pt; border: 1px solid #bfdbfe; border-radius: 6pt;
       background: #eff6ff; vertical-align: top; }
.kpi b { display: block; font-size: 15pt; color: #1d4ed8; }
.kpi span { font-size: 8.5pt; color: #334155; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 14pt 0; }
.pb { page-break-before: always; }
"""


def embed_images(html: str, base: Path) -> str:
    """Inline local images as base64 so Chrome renders them without file access issues."""
    def repl(m):
        src = m.group(1)
        path = (base / src).resolve()
        if not path.exists():
            return m.group(0)
        data = base64.b64encode(path.read_bytes()).decode()
        return m.group(0).replace(src, f"data:image/png;base64,{data}")
    return re.sub(r'<img[^>]+src="([^"]+)"', repl, html)


def render(md_path: Path, pdf_path: Path, title: str) -> Path:
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "attr_list", "md_in_html"])
    body = embed_images(body, md_path.parent)
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>"
    tmp = pdf_path.with_suffix(".html")
    tmp.write_text(html, encoding="utf-8")
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        sys.exit("Chrome/Chromium is required to render PDFs")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", tmp.as_uri()], check=True, capture_output=True, timeout=180)
    tmp.unlink()
    print("wrote", pdf_path.relative_to(ROOT))
    return pdf_path


def main():
    submission = ROOT.parent / "Canvas_Submission"
    submission.mkdir(parents=True, exist_ok=True)
    jobs = [
        (ROOT / "docs" / "technical_documentation.md", REPORTS / "technical_report.pdf",
         f"FP_TechnicalDoc_{SUBMITTER}_{GROUP}_ITAI2373.pdf", "NewsBot 2.0 – Technical Documentation"),
        (REPORTS / "src" / "executive_summary.md", REPORTS / "executive_summary.pdf",
         f"FP_ExecutiveSummary_{SUBMITTER}_{GROUP}_ITAI2373.pdf", "NewsBot 2.0 – Executive Summary"),
        (REPORTS / "src" / "reflective_journal.md", REPORTS / "reflective_journal.pdf",
         f"FP_ReflectiveJournal_{GROUP}_ITAI2373.pdf", "NewsBot 2.0 – Reflective Journal"),
    ]
    for src, out, canvas_name, title in jobs:
        if src.exists():
            render(src, out, title)
            shutil.copy(out, submission / canvas_name)


if __name__ == "__main__":
    main()
