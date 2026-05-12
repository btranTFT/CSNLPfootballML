from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def wrap_line(text: str, max_chars: int = 95) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if len(trial) <= max_chars:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def main() -> None:
    input_md = Path("docs/05_emnlp_paper_draft.md")
    output_pdf = Path("paper/emnlp_paper_draft.pdf")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    lines = input_md.read_text(encoding="utf-8").splitlines()
    c = canvas.Canvas(str(output_pdf), pagesize=LETTER)
    width, height = LETTER
    y = height - inch
    c.setFont("Times-Roman", 11)

    for raw in lines:
        text = raw.strip()
        if text.startswith("## "):
            y -= 8
            c.setFont("Times-Bold", 13)
            text = text[3:]
            wrapped = wrap_line(text, max_chars=80)
        elif text.startswith("### "):
            y -= 4
            c.setFont("Times-Bold", 11)
            text = text[4:]
            wrapped = wrap_line(text, max_chars=86)
        else:
            c.setFont("Times-Roman", 11)
            wrapped = wrap_line(text, max_chars=95)

        for row in wrapped:
            if y < inch:
                c.showPage()
                c.setFont("Times-Roman", 11)
                y = height - inch
            c.drawString(inch, y, row)
            y -= 14
        y -= 2

    c.save()
    print(f"Exported PDF: {output_pdf.resolve()}")


if __name__ == "__main__":
    main()
