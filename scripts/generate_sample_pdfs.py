"""Generate sample earnings-call PDF documents for demo purposes."""

from pathlib import Path

from fpdf import FPDF

from config import COMPANIES, DOCUMENTS_DIR, DOCUMENT_COMPANY_MAP


EARNINGS_CONTENT = {
    "company_a": {
        "title": "Alpha Corp Q4 2025 Earnings Call Transcript",
        "sections": [
            (
                "Financial Highlights",
                "Alpha Corp reported Q4 revenue of $4.2 billion, up 12% year-over-year. "
                "Net income reached $680 million with EPS of $1.45. Full-year revenue "
                "totaled $15.8 billion. Operating margin improved to 18.3%.",
            ),
            (
                "Segment Performance",
                "Cloud services grew 28% to $1.1 billion. Enterprise software revenue "
                "was flat at $2.4 billion due to extended sales cycles. International "
                "markets contributed 35% of total revenue.",
            ),
            (
                "Outlook",
                "Management expects Q1 2026 revenue between $4.0 and $4.3 billion. "
                "Capital expenditure is projected at $450 million for data center expansion.",
            ),
        ],
    },
    "company_b": {
        "title": "Beta Industries Q4 2025 Earnings Call Transcript",
        "sections": [
            (
                "Financial Highlights",
                "Beta Industries posted Q4 revenue of $2.9 billion, a 7% increase. "
                "Net income was $410 million with EPS of $0.92. Gross margin expanded "
                "to 42.1% driven by supply chain optimization.",
            ),
            (
                "Product Updates",
                "The new BetaOne platform launched in November with 2 million active users. "
                "Hardware division revenue declined 3% due to component shortages.",
            ),
            (
                "Guidance",
                "Beta expects full-year 2026 revenue growth of 8-10%. R&D investment "
                "will increase by $120 million focused on AI-powered analytics.",
            ),
        ],
    },
    "company_c": {
        "title": "Gamma Holdings Q4 2025 Earnings Call Transcript",
        "sections": [
            (
                "Financial Highlights",
                "Gamma Holdings reported Q4 revenue of $6.1 billion, down 2% year-over-year. "
                "Net income was $890 million with EPS of $2.10. The decline was attributed "
                "to currency headwinds in European markets.",
            ),
            (
                "Strategic Initiatives",
                "Gamma completed the acquisition of NovaTech for $1.5 billion. "
                "Retail division same-store sales grew 5%. Digital channels now represent "
                "40% of total sales.",
            ),
            (
                "Outlook",
                "Management forecasts Q1 2026 revenue of $5.8 to $6.2 billion. "
                "Cost reduction program targets $200 million in annual savings.",
            ),
        ],
    },
    "company_d": {
        "title": "Delta Systems Q4 2025 Earnings Call Transcript",
        "sections": [
            (
                "Financial Highlights",
                "Delta Systems achieved Q4 revenue of $3.5 billion, up 15% year-over-year. "
                "Net income reached $520 million with EPS of $1.18. Backlog increased to "
                "$12 billion, the highest in company history.",
            ),
            (
                "Operations",
                "Defense contracts contributed $1.8 billion in Q4 revenue. Commercial "
                "aviation segment recovered with 22% growth. Manufacturing efficiency "
                "improved with 95% on-time delivery.",
            ),
            (
                "Guidance",
                "Delta expects 2026 revenue between $14 and $15 billion. "
                "Free cash flow guidance is $2.1 billion for the full year.",
            ),
        ],
    },
    "company_e": {
        "title": "Epsilon Energy Q4 2025 Earnings Call Transcript",
        "sections": [
            (
                "Financial Highlights",
                "Epsilon Energy reported Q4 revenue of $5.4 billion, up 9% year-over-year. "
                "Net income was $760 million with EPS of $1.67. Renewable energy segment "
                "grew 35% and now represents 28% of total revenue.",
            ),
            (
                "Sustainability",
                "Epsilon committed to net-zero emissions by 2040. Solar and wind capacity "
                "increased to 8.2 GW. Carbon capture pilot projects launched in Texas and Norway.",
            ),
            (
                "Outlook",
                "Q1 2026 revenue guidance is $5.2 to $5.6 billion. Capital expenditure "
                "of $1.8 billion is planned for renewable infrastructure.",
            ),
        ],
    },
}


def _write_pdf(path: Path, company_key: str) -> None:
    content = EARNINGS_CONTENT[company_key]
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, content["title"])
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, f"Company: {COMPANIES[company_key]}")
    pdf.ln(8)

    for heading, body in content["sections"]:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, heading)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body)
        pdf.ln(6)

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))


def generate_sample_pdfs() -> None:
    for filename, company_key in DOCUMENT_COMPANY_MAP.items():
        _write_pdf(DOCUMENTS_DIR / filename, company_key)
        print(f"Created {filename}")


if __name__ == "__main__":
    generate_sample_pdfs()
