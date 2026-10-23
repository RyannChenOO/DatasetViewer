"""Create the two-page HW3 experiment summary PDF."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "output" / "pdf" / "HW3_Experiment_Summary.pdf"
INK = colors.HexColor("#17382d")
GREEN = colors.HexColor("#4e8067")
LIME = colors.HexColor("#d8f5b6")
ORANGE = colors.HexColor("#df7b49")
MUTED = colors.HexColor("#66756d")
PAPER = colors.HexColor("#f5f7f3")
LINE = colors.HexColor("#d7dfd8")


class ArchitectureDiagram(Flowable):
    def __init__(self, width: float, height: float = 92) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def draw(self) -> None:
        canvas = self.canv
        boxes = [
            (0, 104, "RGB", "192 → 64"),
            (112, 104, "Goal", "2 → 16"),
            (230, 172, "Gated residual", "v + sigmoid(Wg) x tanh(Wc)"),
            (410, 100, "Classifier", "64 → 32 → 4"),
        ]
        y = 17
        for x, width, title, detail in boxes:
            canvas.setFillColor(colors.white if title != "Gated residual" else colors.HexColor("#fff2ea"))
            canvas.setStrokeColor(LINE if title != "Gated residual" else ORANGE)
            canvas.roundRect(x, y, width, 55, 6, fill=1, stroke=1)
            canvas.setFillColor(MUTED)
            canvas.setFont("Helvetica-Bold", 7)
            canvas.drawString(x + 10, y + 37, title.upper())
            canvas.setFillColor(INK)
            canvas.setFont("Helvetica-Bold", 8 if title != "Gated residual" else 7.4)
            canvas.drawString(x + 10, y + 20, detail)
        canvas.setStrokeColor(GREEN)
        canvas.setLineWidth(1.6)
        for start, end in ((104, 230), (216, 230), (402, 410)):
            canvas.line(start + 2, y + 27, end - 4, y + 27)
            canvas.line(end - 8, y + 31, end - 4, y + 27)
            canvas.line(end - 8, y + 23, end - 4, y + 27)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(6, 3, "Separate modality encoders")
        canvas.drawCentredString(316, 3, "Goal controls a bounded correction to the visual path")


def page_decor(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(INK)
    canvas.rect(0, 0, letter[0], 18, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawString(0.48 * inch, 6, "DS7400 · HW3 · Goal-Conditioned Gated Residual Fusion")
    canvas.drawRightString(letter[0] - 0.48 * inch, 6, f"{doc.page}")
    canvas.restoreState()


def pct(metric: dict[str, float]) -> str:
    return f"{metric['mean'] * 100:.2f} ± {metric['std'] * 100:.2f}%"


def main() -> None:
    results = json.loads((ROOT / "model" / "hw3_results.json").read_text(encoding="utf-8"))["summary"]
    output_dir = OUTPUT.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleHW", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=23, textColor=INK, alignment=TA_CENTER, spaceAfter=5))
    styles.add(ParagraphStyle(name="SubtitleHW", parent=styles["Normal"], fontSize=8.3, leading=11, textColor=MUTED, alignment=TA_CENTER, spaceAfter=9))
    styles.add(ParagraphStyle(name="SectionHW", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.2, leading=12, textColor=INK, spaceBefore=6, spaceAfter=3))
    styles.add(ParagraphStyle(name="BodyHW", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10.7, textColor=colors.HexColor("#33463e"), spaceAfter=4))
    styles.add(ParagraphStyle(name="CalloutHW", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.6, leading=10.4, textColor=INK, leftIndent=8, rightIndent=8, spaceBefore=3, spaceAfter=5))
    styles.add(ParagraphStyle(name="SmallHW", parent=styles["BodyText"], fontName="Helvetica", fontSize=6.8, leading=8.7, textColor=MUTED))

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, rightMargin=0.48 * inch, leftMargin=0.48 * inch,
        topMargin=0.38 * inch, bottomMargin=0.36 * inch, title="HW3 Experiment Summary",
        author="Ryann Chen",
    )
    width = letter[0] - doc.leftMargin - doc.rightMargin
    story = [
        Paragraph("HW3 Experiment Summary", styles["TitleHW"]),
        Paragraph("Goal-Conditioned Gated Residual Fusion for Habitat PointNav Action Prediction", styles["SubtitleHW"]),
    ]

    chips = Table([
        ["TASK", "RGB + goal → next action", "DATA", "2,880 emulated samples", "REPEATS", "5 seeds"],
    ], colWidths=[0.45*inch, 1.45*inch, 0.42*inch, 1.38*inch, 0.55*inch, 0.7*inch])
    chips.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PAPER), ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE), ("TEXTCOLOR", (0, 0), (-1, -1), MUTED),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 6.7),
        ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"), ("FONTNAME", (2, 0), (2, 0), "Helvetica-Bold"),
        ("FONTNAME", (4, 0), (4, 0), "Helvetica-Bold"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [chips, Spacer(1, 5)]

    story += [
        Paragraph("1. Limitation and original hypothesis", styles["SectionHW"]),
        Paragraph("The HW2 baseline concatenates 192 RGB values with only two goal values at the first layer (194 → 64 → 32 → 4). This early fusion treats heterogeneous modalities identically: the high-dimensional image may dominate compact but decision-critical goal information, and no direct path preserves a visual representation while goal context modifies it.", styles["BodyHW"]),
        Table([[Paragraph("<b>Hypothesis.</b> Separate encoders will make goal information harder to ignore. A learned goal-conditioned gate on a residual correction will preserve visual context while suppressing unhelpful updates. <b>Failure mode.</b> Sigmoid gates may saturate and become unresponsive.", styles["CalloutHW"])]], colWidths=[width], style=TableStyle([("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#fff8e8")), ("BOX",(0,0),(-1,-1),0.6,colors.HexColor("#dfc777")), ("LEFTPADDING",(0,0),(-1,-1),5), ("RIGHTPADDING",(0,0),(-1,-1),5)])),
        Paragraph("2. Proposed design", styles["SectionHW"]),
        ArchitectureDiagram(width),
        Paragraph("The model computes v = ReLU(Wv·RGB), g = ReLU(Wg·goal), c = tanh(Wc[v,g]), and z = v + sigmoid(Wgate·g) x c. A 64 → 32 → 4 classifier predicts the next action.", styles["BodyHW"]),
        Paragraph("3. Controlled experiment", styles["SectionHW"]),
        Paragraph("Baseline, full design, and a no-gate ablation (z = v + c) used the identical episode split, preprocessing, 90 epochs, batch size 128, Adam (lr 0.0025), and cross-entropy. Each model was trained with seeds 7400–7404.", styles["BodyHW"]),
    ]

    header = ["Model", "Params", "Test accuracy", "Macro F1", "Test loss"]
    rows = [header]
    for key in ("baseline", "gated_residual", "ungated_residual"):
        model = results["models"][key]
        rows.append([model["label"], f"{model['parameter_count']:,}", pct(model["test_accuracy"]), pct(model["test_macro_f1"]), f"{model['test_loss']['mean']:.4f} ± {model['test_loss']['std']:.4f}"])
    table = Table(rows, colWidths=[1.45*inch, 0.62*inch, 1.12*inch, 1.06*inch, 1.04*inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), INK), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 6.8), ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("GRID", (0,0), (-1,-1), 0.4, LINE), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, PAPER]),
        ("BACKGROUND", (0,2), (-1,2), colors.HexColor("#fff1e9")),
        ("FONTNAME", (0,2), (-1,2), "Helvetica-Bold"), ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [table, PageBreak()]

    story += [
        Paragraph("Observed results and failure analysis", styles["TitleHW"]),
        Paragraph("The full design improved mean test accuracy by 3.10 percentage points over HW2 and reduced run-to-run variation. Its clearest class-level improvement was MOVE_FORWARD: 76.3% (baseline) → 93.0% (gated). This is consistent with the separate goal branch preserving the narrow goal-angle information used for forward decisions.", styles["BodyHW"]),
    ]

    accuracy_image = Image(str(ROOT / "public" / "model" / "hw3_accuracy_comparison.png"), width=4.85*inch, height=2.36*inch)
    accuracy_image.hAlign = "CENTER"
    story += [accuracy_image, Paragraph("Figure 1. Test accuracy, mean ± sample standard deviation across five seeds.", styles["SmallHW"])]

    finding_table = Table([
        [Paragraph("<b>Key ablation result</b><br/>The no-gate model reached 98.38%, only 0.32 points below the full model. Therefore, separate encoders and residual fusion captured most of the gain; adaptive gating alone was not the main explanation.", styles["CalloutHW"]),
         Paragraph("<b>Gate behavior</b><br/>Mean activation = 0.853. Near-saturation fraction = 52.1%. The gate often behaved as an almost-open switch, matching the predicted failure mode and limiting fine-grained control.", styles["CalloutHW"])],
    ], colWidths=[width/2-4, width/2-4])
    finding_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), colors.HexColor("#edf6e9")), ("BACKGROUND", (1,0), (1,0), colors.HexColor("#fff1e9")),
        ("BOX", (0,0), (0,0), 0.6, GREEN), ("BOX", (1,0), (1,0), 0.6, ORANGE),
        ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story += [Spacer(1, 4), finding_table]

    story += [Paragraph("Evidence by class", styles["SectionHW"])]
    confusion = Image(str(ROOT / "public" / "model" / "hw3_confusion_matrices.png"), width=6.75*inch, height=2.07*inch)
    confusion.hAlign = "CENTER"
    story += [confusion, Paragraph("Figure 2. Row-normalized mean test confusion matrices. Most remaining errors involve MOVE_FORWARD being predicted as a turn.", styles["SmallHW"])]

    story += [
        Paragraph("Conclusion, trade-offs, and next experiment", styles["SectionHW"]),
        Paragraph("The system-level hypothesis was supported: explicit modality handling plus residual fusion improved accuracy and stability. The mechanism-level gating hypothesis was only partly supported because the ablation retained most of the gain and the gate frequently saturated. The full model also has 42% more parameters and about 2.8× the measured single-sample CPU latency, so capacity is a confound. Next, I would add a parameter-matched two-branch control and use gate-entropy regularization or a temperature schedule. I would then validate on true Habitat rollouts with scene-disjoint splits rather than repeated public visualization frames.", styles["BodyHW"]),
        Paragraph("Reproduction: python training/run_hw3_experiments.py · Full per-seed results: model/hw3_results.json", styles["SmallHW"]),
    ]

    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    print(OUTPUT)


if __name__ == "__main__":
    main()
