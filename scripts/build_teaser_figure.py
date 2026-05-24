from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_cikm2026" / "figures" / "1.pdf"

W, H = 628.321, 455.844


def hex_color(value: str) -> colors.Color:
    value = value.lstrip("#")
    r = int(value[0:2], 16) / 255
    g = int(value[2:4], 16) / 255
    b = int(value[4:6], 16) / 255
    return colors.Color(r, g, b)


def rounded_box(c: canvas.Canvas, x: float, y: float, w: float, h: float, fill: str, stroke: str) -> None:
    c.setFillColor(hex_color(fill))
    c.setStrokeColor(hex_color(stroke))
    c.setLineWidth(1.15)
    c.roundRect(x, y, w, h, 8, stroke=1, fill=1)


def text(c: canvas.Canvas, x: float, y: float, value: str, size: float = 8.5, color: str = "#222222", font: str = "Helvetica") -> None:
    c.setFillColor(hex_color(color))
    c.setFont(font, size)
    c.drawString(x, y, value)


def centered(c: canvas.Canvas, x: float, y: float, value: str, size: float = 9.0, color: str = "#222222", font: str = "Helvetica") -> None:
    c.setFillColor(hex_color(color))
    c.setFont(font, size)
    c.drawCentredString(x, y, value)


def bullet(c: canvas.Canvas, x: float, y: float, value: str, color: str = "#444444") -> None:
    c.setFillColor(hex_color("#555555"))
    c.circle(x, y + 2.2, 1.5, stroke=0, fill=1)
    text(c, x + 7, y, value, size=7.3, color=color)


def arrow(c: canvas.Canvas, x0: float, y0: float, x1: float, y1: float, color: str = "#667085") -> None:
    c.setStrokeColor(hex_color(color))
    c.setFillColor(hex_color(color))
    c.setLineWidth(1.1)
    c.line(x0, y0, x1, y1)
    c.line(x1, y1, x1 - 5, y1 + 3)
    c.line(x1, y1, x1 - 5, y1 - 3)


def node(c: canvas.Canvas, x: float, y: float, label: str, fill: str = "#FFFFFF", stroke: str = "#667085") -> None:
    c.setFillColor(hex_color(fill))
    c.setStrokeColor(hex_color(stroke))
    c.setLineWidth(0.9)
    c.circle(x, y, 10, stroke=1, fill=1)
    centered(c, x, y - 3, label, size=6.8, color="#1F2937")


def mini_mlp(c: canvas.Canvas, x: float, y: float) -> None:
    xs = [x, x + 36, x + 72]
    ys = [[y + 30, y, y - 30], [y + 22, y - 22], [y]]
    for i in range(2):
        for y0 in ys[i]:
            for y1 in ys[i + 1]:
                c.setStrokeColor(hex_color("#C5CDD8"))
                c.setLineWidth(0.55)
                c.line(xs[i], y0, xs[i + 1], y1)
    for col, values in enumerate(ys):
        for yy in values:
            node(c, xs[col], yy, "+", fill="#FFFFFF", stroke="#8EA4C8")


def mini_basis(c: canvas.Canvas, x: float, y: float) -> None:
    labels = ["rbf1", "rbf2", "rbf3", "..."]
    for idx, label in enumerate(labels):
        rounded_box(c, x + idx * 30, y, 25, 20, "#FFFFFF", "#4D908E")
        centered(c, x + idx * 30 + 12.5, y + 7, label, size=5.7, color="#1F2937")
    arrow(c, x + 124, y + 10, x + 152, y + 10, "#4D908E")
    node(c, x + 167, y + 10, "sum", fill="#FFFFFF", stroke="#4D908E")


def mini_eml(c: canvas.Canvas, x: float, y: float) -> None:
    rounded_box(c, x, y + 24, 72, 22, "#FFFFFF", "#C2410C")
    centered(c, x + 36, y + 32, "bounded exp", size=6.5, color="#1F2937")
    rounded_box(c, x, y - 8, 72, 22, "#FFFFFF", "#0F766E")
    centered(c, x + 36, y, "positive log", size=6.5, color="#1F2937")
    node(c, x + 100, y + 18, "+", fill="#FFFFFF", stroke="#6B7280")
    arrow(c, x + 74, y + 35, x + 90, y + 23, "#C2410C")
    arrow(c, x + 74, y + 3, x + 90, y + 13, "#0F766E")
    rounded_box(c, x + 122, y + 6, 38, 22, "#FFF7ED", "#C2410C")
    centered(c, x + 141, y + 14, "gate", size=6.4, color="#1F2937")


def draw_column(c: canvas.Canvas, x: float, title: str, subtitle: str, fill: str, stroke: str, bullets: list[str], kind: str) -> None:
    rounded_box(c, x, 118, 180, 220, fill, stroke)
    centered(c, x + 90, 318, title, size=10.6, color="#111827", font="Helvetica-Bold")
    centered(c, x + 90, 301, subtitle, size=7.6, color="#344054")
    if kind == "mlp":
        mini_mlp(c, x + 54, 252)
    elif kind == "basis":
        mini_basis(c, x + 18, 236)
    else:
        mini_eml(c, x + 20, 231)
    y = 185
    for item in bullets:
        bullet(c, x + 18, y, item)
        y -= 17


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    c.setTitle("EML-KAN checkpoint export motivation")

    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    centered(c, W / 2, 420, "Checkpoint-to-Equation-DAG Audit", size=18, color="#111827", font="Helvetica-Bold")
    centered(
        c,
        W / 2,
        398,
        "The audit object is a trained neural checkpoint, not an independent symbolic-search formula.",
        size=9.0,
        color="#475467",
    )

    rounded_box(c, 222, 354, 184, 28, "#F8FAFC", "#98A2B3")
    centered(c, 314, 363, "trained scientific predictor", size=8.4, color="#1F2937", font="Helvetica-Bold")
    arrow(c, 272, 350, 178, 338)
    arrow(c, 314, 350, 314, 338)
    arrow(c, 356, 350, 450, 338)

    draw_column(
        c,
        28,
        "Dense MLP export",
        "faithful, but long",
        "#F4F7FF",
        "#3B5BA9",
        ["many affine/activation assignments", "little operator-level structure", "larger graph to diff and inspect"],
        "mlp",
    )
    draw_column(
        c,
        224,
        "KAN / RBF-KAN export",
        "basis-parametrized edges",
        "#F2FBFA",
        "#247C78",
        ["basis-expanded serialization", "basis terms dominate audit length", "useful predictor, heavier DAG"],
        "basis",
    )
    draw_column(
        c,
        420,
        "EML-KAN export",
        "bounded exp/log edges",
        "#FFF7F1",
        "#C75A12",
        ["explicit scientific operator motifs", "no RBF basis expansion", "shorter exact checkpoint DAG"],
        "eml",
    )

    rounded_box(c, 64, 54, 500, 38, "#FFFFFF", "#CBD5E1")
    centered(
        c,
        314,
        76,
        "Audit reports separate export fidelity, DAG tokens, canonical tokens, basis terms, and reduction diagnostics.",
        size=7.6,
        color="#334155",
    )
    centered(c, 314, 62, "PySR remains a compact symbolic-search reference, not a checkpoint export.", size=7.2, color="#475467")

    c.showPage()
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
