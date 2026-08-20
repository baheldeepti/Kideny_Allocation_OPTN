#!/usr/bin/env python3
"""Wrap the styled paper.html into a print-optimized standalone document and
render it to PDF with headless Chrome (figures are embedded as data URIs)."""
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
CONTENT = (BASE / "paper.html").read_text()
PRINT_HTML = BASE / "paper_print.html"
PDF = BASE / "PAPER.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PRINT_CSS = """
<style>
  @page { size: A4; margin: 16mm 15mm; }
  html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .wrap { padding: 0 !important; min-height: 0 !important; }
  .hero { padding-top: 8px !important; }
  /* keep figures / tiles whole, but let long tables flow across pages */
  figure, .kpi, h1.title { break-inside: avoid; }
  .body h2, .body h3 { break-after: avoid; }
  .kpis { break-inside: avoid; }
  .tablewrap, .body table { break-inside: auto; overflow-x: visible; }
  .body tr { break-inside: avoid; }        /* never split a row */
  .body thead { display: table-header-group; }  /* repeat header on each page */
  .fig-grid { grid-template-columns: 1fr 1fr; }
  a { color: inherit; }               /* links readable on paper */
</style>
"""

# force light theme for print, wrap as a full HTML document
doc = f"""<!doctype html>
<html lang="en" data-theme="light">
<head><meta charset="utf-8">
<title>Kidney Waitlist Risk & Time-to-Transplant</title>
{PRINT_CSS}
</head>
<body>
{CONTENT}
</body></html>"""

PRINT_HTML.write_text(doc)
print(f"wrote {PRINT_HTML.name}")

subprocess.run([
    CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
    f"--print-to-pdf={PDF}", PRINT_HTML.as_uri(),
], check=True, capture_output=True)
print(f"wrote {PDF.name}  ({PDF.stat().st_size/1024:.0f} KB)")
