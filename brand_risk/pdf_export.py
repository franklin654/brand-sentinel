"""Board-report PDF generator using reportlab.

Renders dossier cards into a clean A4 PDF suitable for board minutes and CRO briefings.

Public API:
    generate_pdf(dossiers: list) -> bytes   # call with st.download_button
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

_RISK_COLOUR = {
    "low":      colors.HexColor("#27ae60"),
    "medium":   colors.HexColor("#f39c12"),
    "high":     colors.HexColor("#e67e22"),
    "critical": colors.HexColor("#c0392b"),
}


def generate_pdf(dossiers: list) -> bytes:
    """Render a list of ReputationDossier objects into a board-ready PDF.

    Args:
        dossiers: List of ReputationDossier instances from the pipeline run.

    Returns:
        PDF content as bytes, ready for st.download_button.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=45, leftMargin=45, topMargin=55, bottomMargin=45,
    )
    styles = getSampleStyleSheet()
    story = _build_story(dossiers, styles)
    doc.build(story)
    return buffer.getvalue()


def _build_story(dossiers: list, styles) -> list:
    story: list = []
    story.append(Paragraph("Brand &amp; Reputational Risk Report", styles["Title"]))
    story.append(Paragraph(
        f"Generated {datetime.utcnow().isoformat()[:19]}Z · TCS-AMD Hackathon",
        styles["Normal"],
    ))
    story.append(Spacer(1, 18))
    for d in dossiers:
        _add_dossier(story, d, styles)
    return story


def _add_dossier(story: list, d, styles) -> None:
    colour = _RISK_COLOUR.get(d.overall_risk, colors.grey)
    story.append(Paragraph(d.entity_name, styles["Heading1"]))
    story.append(Paragraph(
        f"Risk score: <font color='#{colour.hexval()[2:]}' size='12'>"
        f"<b>{d.adverse.risk_score}/100</b></font>"
        f" &nbsp;·&nbsp; {d.overall_risk.upper()}",
        styles["Heading2"],
    ))
    story.append(Paragraph(d.headline, styles["Normal"]))
    story.append(Spacer(1, 6))

    attr = d.risk_attribution
    if attr:
        story.append(Paragraph(
            f"Attribution — Social: {attr.get('social_pct', 0):.0f}%  "
            f"&nbsp;·&nbsp; Media: {attr.get('media_pct', 0):.0f}%  "
            f"&nbsp;·&nbsp; Vendor: {attr.get('vendor_pct', 0):.0f}%",
            styles["Normal"],
        ))
        story.append(Spacer(1, 6))

    if d.vendor_impacts:
        data = [["Vendor", "Exposure", "Action"]] + [
            [v.vendor_name, v.exposure, v.recommended_action]
            for v in d.vendor_impacts
        ]
        table = Table(data, colWidths=[180, 100, 100])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))

    if d.suggested_response:
        story.append(Paragraph("Suggested response", styles["Heading3"]))
        story.append(Paragraph(d.suggested_response[:600], styles["Normal"]))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 18))
