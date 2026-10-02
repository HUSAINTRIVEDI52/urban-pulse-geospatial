"""
UrbanPulse - PDF Report Exporter
Converts docs/report/index.html to docs/report/UrbanPulse_Report.pdf using Playwright Chromium.
"""

import argparse
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def export_pdf(
    html_path: Path | str = PROJECT_ROOT / "docs" / "report" / "index.html",
    pdf_path: Path | str = PROJECT_ROOT / "docs" / "report" / "UrbanPulse_Report.pdf",
) -> Path:
    html_file = Path(html_path).resolve()
    pdf_file = Path(pdf_path).resolve()

    if not html_file.exists():
        raise FileNotFoundError(f"HTML report not found at {html_file}")

    pdf_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Launching Playwright Chromium to export PDF...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        # Open local HTML via file:// protocol
        file_url = html_file.as_uri()
        page.goto(file_url, wait_until="networkidle")

        page.pdf(
            path=str(pdf_file),
            format="A4",
            print_background=True,
            margin={
                "top": "12mm",
                "bottom": "14mm",
                "left": "12mm",
                "right": "12mm",
            },
            display_header_footer=True,
            header_template='<div style="font-size:8px; color:#94a3b8; width:100%; text-align:right; padding-right:12mm;">UrbanPulse Technical Report</div>',
            footer_template='<div style="font-size:8px; color:#94a3b8; width:100%; text-align:center;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>',
        )
        browser.close()

    print(f"[+] Successfully exported PDF report: {pdf_file}")
    return pdf_file


def main():
    parser = argparse.ArgumentParser(description="Export UrbanPulse HTML report to PDF.")
    parser.add_argument(
        "--html",
        type=Path,
        default=PROJECT_ROOT / "docs" / "report" / "index.html",
        help="Input HTML path (default: docs/report/index.html)",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=PROJECT_ROOT / "docs" / "report" / "UrbanPulse_Report.pdf",
        help="Output PDF path (default: docs/report/UrbanPulse_Report.pdf)",
    )
    args = parser.parse_args()

    export_pdf(args.html, args.pdf)


if __name__ == "__main__":
    main()
