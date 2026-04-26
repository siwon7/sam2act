#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer


def ensure_cjk_fonts():
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
        pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
        return "HYGothic-Medium", "HYSMyeongJo-Medium"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def build_styles():
    styles = getSampleStyleSheet()
    base_font, bold_font = ensure_cjk_fonts()
    base = ParagraphStyle(
        "Base",
        parent=styles["BodyText"],
        fontName=base_font,
        fontSize=9.5,
        leading=13,
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=base,
        fontName=bold_font,
        fontSize=18,
        leading=22,
        spaceBefore=8,
        spaceAfter=10,
        textColor=colors.black,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=base,
        fontName=bold_font,
        fontSize=13,
        leading=17,
        spaceBefore=8,
        spaceAfter=6,
    )
    h3 = ParagraphStyle(
        "H3",
        parent=base,
        fontName=bold_font,
        fontSize=11,
        leading=14,
        spaceBefore=6,
        spaceAfter=4,
    )
    code = ParagraphStyle(
        "Code",
        parent=base,
        fontName="Courier",
        fontSize=8,
        leading=10,
        backColor=colors.whitesmoke,
        borderPadding=4,
        leftIndent=4,
        rightIndent=4,
        spaceBefore=4,
        spaceAfter=6,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=base,
        leftIndent=14,
        firstLineIndent=-8,
    )
    return {"base": base, "h1": h1, "h2": h2, "h3": h3, "code": code, "bullet": bullet}


def esc(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    return text


def parse_markdown(text: str, styles):
    lines = text.splitlines()
    story = []
    in_code = False
    code_lines = []

    for raw in lines:
        line = raw.rstrip()

        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                story.append(Preformatted("\n".join(code_lines), styles["code"]))
                story.append(Spacer(1, 2))
                in_code = False
                code_lines = []
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            story.append(Spacer(1, 4))
            continue

        if line.startswith("# "):
            story.append(Paragraph(esc(line[2:].strip()), styles["h1"]))
            continue
        if line.startswith("## "):
            story.append(Paragraph(esc(line[3:].strip()), styles["h2"]))
            continue
        if line.startswith("### "):
            story.append(Paragraph(esc(line[4:].strip()), styles["h3"]))
            continue
        if line.startswith("- "):
            story.append(Paragraph("&bull; " + esc(line[2:].strip()), styles["bullet"]))
            continue
        if re.match(r"^\d+\.\s", line):
            story.append(Paragraph(esc(line), styles["bullet"]))
            continue

        story.append(Paragraph(esc(line), styles["base"]))

    if code_lines:
        story.append(Preformatted("\n".join(code_lines), styles["code"]))

    return story


def main():
    if len(sys.argv) != 3:
        print("usage: md_to_pdf_simple.py <input.md> <output.pdf>")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    text = src.read_text(encoding="utf-8")
    styles = build_styles()
    story = parse_markdown(text, styles)

    doc = SimpleDocTemplate(
        str(dst),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=src.stem,
    )
    doc.build(story)


if __name__ == "__main__":
    main()
