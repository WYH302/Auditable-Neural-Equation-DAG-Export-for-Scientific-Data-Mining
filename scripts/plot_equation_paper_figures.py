from __future__ import annotations

import shutil
from io import BytesIO
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Keep the audit-reduction figure reproducible in the bundled runtime.
    matplotlib = None
    plt = None
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2" / "equation_export"
FIG_DIR = RESULTS / "figures"
PAPER_FIG_DIRS = [
    ROOT.parent / "paper" / "paper_cikm2026" / "figures",
    ROOT.parent / "paper" / "paper_cikm2026" / "build" / "figures",
    ROOT / "paper_cikm2026" / "figures",
    ROOT / "paper_cikm2026" / "build" / "figures",
    ROOT / "paper_elscas" / "figures_updated",
]
WORKFLOW = ROOT / "els-cas-templates" / "figs" / "workflow.png"
METHOD_ORDER = ["MLP", "RBF-KAN", "StableEML", "EML-KAN", "PySR"]
COLORS = {
    "MLP": "#396AB1",
    "RBF-KAN": "#11A579",
    "StableEML": "#F2B701",
    "EML-KAN": "#E73F74",
    "PySR": "#7F3C8D",
}
MARKER = "o"


def fmt_sci(value: float) -> str:
    if value == 0:
        return "0"
    if not np.isfinite(value):
        return "--"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.1e}"
    return f"{value:.3g}"


def annotate_y_extrema(ax, x: np.ndarray, y: np.ndarray, color: str, prefix: str = "") -> None:
    finite = np.isfinite(y)
    if not finite.any():
        return
    xf = x[finite]
    yf = y[finite]
    for idx, label, offset in [
        (int(np.argmin(yf)), "min", (10, -4)),
        (int(np.argmax(yf)), "max", (10, 4)),
    ]:
        ax.scatter([xf[idx]], [yf[idx]], s=24, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(
            f"{prefix}{label}: {yf[idx]:.3g}",
            (xf[idx], yf[idx]),
            textcoords="offset points",
            xytext=offset,
            fontsize=7.2,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.75),
        )


def crop_workflow() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 2800, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_root = Path("C:/Windows/Fonts")

    def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        candidates = ["arialbd.ttf" if bold else "arial.ttf", "timesbd.ttf" if bold else "times.ttf"]
        for name in candidates:
            path = font_root / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
        return ImageFont.load_default()

    title_font = font(46, True)
    body_font = font(35)
    small_font = font(29)
    tiny_font = font(25)

    def box(x0: int, y0: int, x1: int, y1: int, text: str, fill: str, outline: str, fnt: ImageFont.ImageFont = body_font) -> None:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=22, fill=fill, outline=outline, width=4)
        lines = text.split("\n")
        heights = [draw.textbbox((0, 0), line, font=fnt)[3] - draw.textbbox((0, 0), line, font=fnt)[1] for line in lines]
        total_h = sum(heights) + 12 * (len(lines) - 1)
        y = y0 + (y1 - y0 - total_h) // 2
        for line, line_h in zip(lines, heights):
            bbox = draw.textbbox((0, 0), line, font=fnt)
            x = x0 + (x1 - x0 - (bbox[2] - bbox[0])) // 2
            draw.text((x, y), line, font=fnt, fill="#111111")
            y += line_h + 12

    def arrow(x0: int, y0: int, x1: int, y1: int, color: str = "#314F8F") -> None:
        draw.line((x0, y0, x1, y1), fill=color, width=8)
        head = 34
        draw.polygon([(x1, y1), (x1 - head, y1 - head // 2), (x1 - head, y1 + head // 2)], fill=color)

    def chip(x: int, y: int, text: str, color: str) -> None:
        bbox = draw.textbbox((0, 0), text, font=small_font)
        w = bbox[2] - bbox[0] + 44
        h = bbox[3] - bbox[1] + 24
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill="#FFFFFF", outline=color, width=3)
        draw.text((x + 22, y + 12), text, font=small_font, fill="#222222")

    draw.text((90, 55), "Checkpoint-to-DAG audit asks a different question from symbolic search", font=title_font, fill="#111111")
    draw.text(
        (92, 112),
        "Given a trained predictor, can we export its computation faithfully, shorten the neural DAG, and expose audit motifs?",
        font=body_font,
        fill="#333333",
    )

    box(120, 230, 690, 430, "trained neural\ncheckpoint", "#F4F8FF", "#396AB1", title_font)
    box(930, 230, 1490, 430, "assignment-preserving\nequation DAG", "#F8FCFA", "#0B7A75", title_font)
    box(1730, 230, 2600, 430, "audit report\nnot a final physical law", "#FFF9ED", "#B57418", title_font)
    arrow(705, 330, 910, 330)
    arrow(1505, 330, 1710, 330)

    draw.text((145, 500), "Neural checkpoint exporters", font=small_font, fill="#333333")
    chip(140, 550, "MLP: dense graph", "#396AB1")
    chip(140, 625, "RBF-KAN: basis expansion", "#0B7A75")
    chip(140, 700, "EML-KAN: bounded exp-log edges", "#E73F74")
    chip(140, 775, "bounded variant avoided NaN/Inf in our audits", "#E73F74")

    draw.text((965, 500), "Audit metrics", font=small_font, fill="#333333")
    for idx, (text, color) in enumerate(
        [
            ("export fidelity", "#396AB1"),
            ("DAG tokens", "#396AB1"),
            ("exp/log operators", "#0B7A75"),
            ("basis terms", "#0B7A75"),
            ("pruning curve", "#B57418"),
            ("distillation candidate", "#B57418"),
        ]
    ):
        chip(960 + (idx % 2) * 340, 550 + (idx // 2) * 92, text, color)

    draw.text((1780, 500), "Reference boundary", font=small_font, fill="#333333")
    box(1780, 555, 2580, 685, "PySR / gplearn\ncompact symbolic-search references", "#FBF7FF", "#7F3C8D", body_font)
    box(1780, 735, 2580, 865, "not trained neural checkpoints\nso no neural export-fidelity score", "#FBF7FF", "#7F3C8D", body_font)

    image.save(FIG_DIR / "workflow_overview.png", quality=95)


def iqr(values: pd.Series) -> tuple[float, float, float]:
    array = np.asarray(values.dropna(), dtype=float)
    return float(np.percentile(array, 25)), float(np.median(array)), float(np.percentile(array, 75))


def figure_to_image(fig: plt.Figure, dpi: int = 600) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.load()
    plt.close(fig)
    return image


def plot_accuracy_complexity() -> None:
    frame = pd.read_csv(RESULTS / "equation_metrics_all.csv")
    frame["method_label"] = frame["method_label"].replace({"KAN": "RBF-KAN"})
    frame = frame[frame["status"].isin(["ok", "approx"])].copy()
    frame["test_mse"] = pd.to_numeric(frame["test_mse"], errors="coerce")
    frame["tokens"] = pd.to_numeric(frame["tokens"], errors="coerce")

    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "font.size": 6.6,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "mathtext.fontset": "dejavuserif",
        }
    ):
        fig = plt.figure(figsize=(3.35, 2.28), dpi=600)
        ax = fig.add_axes([0.13, 0.18, 0.82, 0.75])
        rows = []
        for method in METHOD_ORDER:
            group = frame[frame["method_label"] == method]
            tq1, tmed, tq3 = iqr(group["tokens"])
            yq1, ymed, yq3 = iqr(group["test_mse"])
            rows.append((method, tq1, tmed, tq3, yq1, ymed, yq3))

        x_low = min(row[1] for row in rows)
        x_high = max(row[3] for row in rows)
        y_low = min(row[4] for row in rows)
        y_high = max(row[6] for row in rows)
        ax.axvspan(x_low / 1.55, 120, color="#F4EEF8", alpha=0.72, zorder=0)
        ax.text(
            0.07,
            0.07,
            "symbolic-search\nreference",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.6,
            color="#6A2E7F",
            bbox=dict(fc="white", ec="#eadcf0", lw=0.35, alpha=0.9, pad=1.2),
            zorder=1,
        )

        label_offsets = {
            "MLP": (5, 4),
            "RBF-KAN": (-42, -10),
            "StableEML": (-44, 6),
            "EML-KAN": (5, 5),
            "PySR": (5, 7),
        }

        for method, tq1, tmed, tq3, yq1, ymed, yq3 in rows:
            xerr = np.array([[max(tmed - tq1, 1e-9)], [max(tq3 - tmed, 1e-9)]])
            yerr = np.array([[max(ymed - yq1, 1e-12)], [max(yq3 - ymed, 1e-12)]])
            is_symbolic = method == "PySR"
            is_eml = method == "EML-KAN"
            ax.errorbar(
                [tmed],
                [ymed],
                xerr=xerr,
                yerr=yerr,
                fmt=MARKER,
                ms=4.8,
                mfc=COLORS[method],
                mec=COLORS[method],
                mew=0.95,
                ecolor=COLORS[method],
                elinewidth=0.8,
                capsize=2.1,
                capthick=0.8,
                alpha=0.96,
                zorder=4 if is_eml else 3,
            )
            ax.annotate(
                method,
                (tmed, ymed),
                xytext=label_offsets[method],
                textcoords="offset points",
                color=COLORS[method],
                fontsize=6.1,
                fontweight="bold" if method in {"EML-KAN", "PySR"} else "normal",
                va="center",
                ha="left",
                bbox=dict(fc="white", ec="none", alpha=0.80, pad=0.35),
                zorder=5,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_low / 1.55, x_high * 1.85)
        ax.set_ylim(y_low / 1.50, y_high * 1.45)
        ax.set_xlabel("Exported equation-DAG tokens")
        ax.set_ylabel("Test MSE")
        ax.grid(True, which="major", color="#d0d0d0", alpha=0.45, linewidth=0.45)
        ax.grid(True, which="minor", color="#e6e6e6", alpha=0.34, linewidth=0.28)
        ax.text(
            0.98,
            0.98,
            "lower-left is better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.8,
            color="#555555",
            bbox=dict(fc="white", ec="#dddddd", lw=0.35, alpha=0.9, pad=1.4),
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(which="both", direction="out", length=2.4, pad=1.5)
        ax.tick_params(which="minor", length=1.5)
        fig.savefig(FIG_DIR / "accuracy_complexity_tradeoff.png", facecolor="white")
        plt.close(fig)
        return


def plot_case_study() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    scale = 6
    width, height = 2010, 2520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_root = Path("C:/Windows/Fonts")

    def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        candidates = ["timesbd.ttf" if bold else "times.ttf", "arialbd.ttf" if bold else "arial.ttf"]
        for name in candidates:
            path = font_root / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
        return ImageFont.load_default()

    title_font = font(56, True)
    body_font = font(44)
    body_bold = font(44, True)
    small_font = font(39)
    tiny_font = font(32)

    def xy(x: float, y: float) -> tuple[int, int]:
        return int(x * width), int(y * height)

    def rect(x0: float, y0: float, x1: float, y1: float, fill: str, outline: str = "#D9D9D9", line: int = 3) -> None:
        draw.rounded_rectangle((*xy(x0, y0), *xy(x1, y1)), radius=9 * scale, fill=fill, outline=outline, width=line)

    def text_center(x0: float, y0: float, x1: float, y1: float, text: str, fnt: ImageFont.ImageFont, fill: str = "#111111") -> None:
        lines = text.split("\n")
        line_heights = [draw.textbbox((0, 0), line, font=fnt)[3] - draw.textbbox((0, 0), line, font=fnt)[1] for line in lines]
        total_h = sum(line_heights) + (len(lines) - 1) * int(0.008 * height)
        y = int((y0 + y1) * height / 2 - total_h / 2)
        for line, line_h in zip(lines, line_heights):
            bbox = draw.textbbox((0, 0), line, font=fnt)
            x = int((x0 + x1) * width / 2 - (bbox[2] - bbox[0]) / 2)
            draw.text((x, y), line, font=fnt, fill=fill)
            y += line_h + int(0.008 * height)

    def label(x: float, y: float, text: str, fnt: ImageFont.ImageFont = body_font, fill: str = "#111111") -> None:
        draw.text(xy(x, y), text, font=fnt, fill=fill)

    def arrow(x0: float, y0: float, x1: float, y1: float, fill: str = "#224C9A") -> None:
        start, end = xy(x0, y0), xy(x1, y1)
        draw.line((start, end), fill=fill, width=5)
        head = 18 * scale
        draw.polygon([(end[0], end[1]), (end[0] - head, end[1] - head // 2), (end[0] - head, end[1] + head // 2)], fill=fill)

    def box(x0: float, y0: float, x1: float, y1: float, text: str, fill: str, outline: str, fnt: ImageFont.ImageFont = body_font) -> None:
        rect(x0, y0, x1, y1, fill, outline, line=4)
        text_center(x0, y0, x1, y1, text, fnt)

    # Panel backgrounds.
    rect(0.035, 0.035, 0.965, 0.270, "#F5FAFF")
    rect(0.035, 0.305, 0.965, 0.535, "#F8FCFA")
    rect(0.035, 0.575, 0.965, 0.955, "#FFFDF7")

    label(0.060, 0.060, "(a) Correctness layer", title_font)
    box(0.080, 0.125, 0.335, 0.190, "trained EML-KAN\ncheckpoint", "#FFFFFF", "#396AB1", body_font)
    box(0.405, 0.125, 0.610, 0.190, "exact\nserializer", "#EEF5FF", "#396AB1", body_font)
    box(0.690, 0.125, 0.910, 0.190, "faithful\nequation-DAG", "#FFFFFF", "#396AB1", body_font)
    arrow(0.340, 0.157, 0.402, 0.157)
    arrow(0.615, 0.157, 0.687, 0.157)
    box(0.155, 0.212, 0.425, 0.250, "export fidelity = 0", "#FFFFFF", "#11A579", small_font)
    box(0.545, 0.212, 0.850, 0.250, "raw DAG: thousands of tokens", "#FFFFFF", "#11A579", tiny_font)

    label(0.060, 0.330, "(b) Reduction layer", title_font)
    step_boxes = [
        (0.070, 0.395, 0.225, 0.462, "raw\nDAG"),
        (0.302, 0.395, 0.465, 0.462, "canonical\nDAG"),
        (0.540, 0.395, 0.695, 0.462, "pruned\nDAG"),
        (0.765, 0.395, 0.925, 0.462, "candidate\nformula"),
    ]
    for x0, y0, x1, y1, text in step_boxes:
        box(x0, y0, x1, y1, text, "#FFFFFF", "#0B7A75", body_font)
    for sx, ex, caption in [(0.228, 0.300, "canonicalize"), (0.468, 0.538, "gate/prune"), (0.697, 0.763, "distill")]:
        arrow(sx, 0.428, ex, 0.428, "#0B7A75")
        bbox = draw.textbbox((0, 0), caption, font=tiny_font)
        draw.text((int((sx + ex) * width / 2 - (bbox[2] - bbox[0]) / 2), int(0.480 * height)), caption, font=tiny_font, fill="#333333")
    label(
        0.070,
        0.505,
        "The reduced formula is a diagnostic hypothesis, not a replacement for the faithful checkpoint DAG.",
        tiny_font,
        "#333333",
    )

    label(0.060, 0.600, "(c) Diagnostic examples", title_font)
    table_left, table_top, table_right, table_bottom = 0.060, 0.665, 0.940, 0.875
    rect(table_left, table_top, table_right, table_bottom, "#FFFFFF", "#777777", line=3)
    header_bottom = table_top + (table_bottom - table_top) / 4
    draw.rectangle((*xy(table_left, table_top), *xy(table_right, header_bottom)), fill="#F4F4F4")
    c1, c2 = 0.310, 0.785
    for xpos in [c1, c2]:
        draw.line((*xy(xpos, table_top), *xy(xpos, table_bottom)), fill="#CFCFCF", width=2)
    row_h = (table_bottom - table_top) / 4
    for idx in range(1, 4):
        y = table_top + idx * row_h
        draw.line((*xy(table_left, y), *xy(table_right, y)), fill="#CFCFCF", width=2)
    label(0.075, 0.685, "Case", body_bold)
    label(0.330, 0.685, "Distilled candidate", body_bold)
    label(0.805, 0.685, "Sym. MSE", body_bold)
    rows = [
        ("log-response", "log(x + 2.1)", "3.31e-8"),
        ("exp-minus-log", "0.909 exp(x) -\n0.585 log(x + 2.1)", "6.83e-7"),
        ("inverse-quad.", "1/(x^2 + 1)", "3.94e-11"),
    ]
    for idx, (case, candidate, mse) in enumerate(rows):
        y0 = table_top + (idx + 1) * row_h
        label(0.075, y0 + 0.020, case, small_font)
        text_center(0.330, y0 + 0.006, 0.765, y0 + row_h - 0.006, candidate, small_font)
        label(0.810, y0 + 0.020, mse, small_font)
    label(
        0.060,
        0.905,
        "Note: examples are audit reductions after exact export; only the log-response raw graph is reported as 2609 tokens.",
        tiny_font,
        "#444444",
    )

    image.save(FIG_DIR / "equation_export_case_study.png")


def plot_pruning_appendix() -> None:
    frame = pd.read_csv(RESULTS / "pruning_curves.csv")
    frame = frame[frame["status"] == "ok"].copy()
    grouped = frame.groupby("threshold", as_index=False).agg(
        tokens=("tokens", "mean"),
        active_terms=("active_terms", "mean"),
        export_fidelity_mse=("export_fidelity_mse", "mean"),
        true_test_mse=("true_test_mse", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.3), dpi=220, sharex=True)
    label_thresholds = {0.0, 0.1, 0.2, 0.3}
    for ax, key, title, color in [
        (axes[0], "export_fidelity_mse", "Export fidelity vs pruning", "#396AB1"),
        (axes[1], "true_test_mse", "True test MSE vs pruning", "#E73F74"),
    ]:
        y = grouped[key].clip(lower=1e-8)
        ax.plot(grouped["threshold"], y, marker="o", lw=2, color=color)
        ax.set_yscale("log")
        ax.set_xlabel("gate pruning threshold")
        ax.set_title(title, fontsize=10, weight="bold")
        ax.grid(True, which="both", alpha=0.24)
        for _, row in grouped.iterrows():
            if round(float(row["threshold"]), 2) not in label_thresholds:
                continue
            ax.annotate(
                f"{row['tokens']:.0f} tok",
                (row["threshold"], max(row[key], 1e-8)),
                textcoords="offset points",
                xytext=(4, 5),
                fontsize=7,
            )
        if key == "export_fidelity_mse":
            ax.set_ylim(5e-9, max(y) * 2.5)
            ax.text(
                0.02,
                0.05,
                "0 export gap plotted at floor",
                transform=ax.transAxes,
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", alpha=0.92),
            )
        finite = np.asarray(y, dtype=float)
        thresholds = np.asarray(grouped["threshold"], dtype=float)
        min_idx = int(np.argmin(finite))
        max_idx = int(np.argmax(finite))
        for idx, label, offset in [(min_idx, "min", (7, 8)), (max_idx, "max", (7, -16))]:
            ax.scatter([thresholds[idx]], [finite[idx]], s=30, color=color, edgecolor="white", linewidth=0.8, zorder=4)
            if key == "export_fidelity_mse" and label == "min":
                continue
            ax.annotate(
                f"{label} {fmt_sci(finite[idx])}",
                (thresholds[idx], finite[idx]),
                textcoords="offset points",
                xytext=(8, 20) if label == "min" else offset,
                fontsize=7.2,
                color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.75),
            )
    axes[0].set_ylabel("MSE")
    fig.suptitle("Gate pruning as an equation-complexity audit", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pruning_to_equation_appendix.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_gate_distribution_appendix() -> None:
    source = ROOT / "results_v2" / "synthetic_surgery" / "ablation_runs_20260509.csv"
    if not source.exists():
        return
    frame = pd.read_csv(source)
    frame = frame[np.isfinite(pd.to_numeric(frame["gate_mean"], errors="coerce"))].copy()
    frame["gate_mean"] = pd.to_numeric(frame["gate_mean"], errors="coerce")
    order = [
        "bounded-tanh EML",
        "bounded-tanh, depth 3, warmup",
        "bounded-tanh, no warmup",
        "raw exp",
        "bounded-tanh tau=4",
        "clip M=8",
        "softclip M=5",
        "softclip M=8 + identity",
    ]
    short = {
        "bounded-tanh EML": "bounded T=2",
        "bounded-tanh, depth 3, warmup": "T=3 + warmup",
        "bounded-tanh, no warmup": "T=3 no warmup",
        "raw exp": "raw exp",
        "bounded-tanh tau=4": "tau=4",
        "clip M=8": "clip M=8",
        "softclip M=5": "softclip M=5",
        "softclip M=8 + identity": "softclip + id",
    }
    rows = []
    for label in order:
        values = frame.loc[frame["label"] == label, "gate_mean"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        rows.append(
            {
                "label": label,
                "short": short[label],
                "q1": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "q3": float(np.percentile(values, 75)),
            }
        )
    fig, ax = plt.subplots(figsize=(6.2, 3.9), dpi=220)
    y_pos = np.arange(len(rows))
    med = np.array([r["median"] for r in rows])
    q1 = np.array([r["q1"] for r in rows])
    q3 = np.array([r["q3"] for r in rows])
    ax.errorbar(
        med,
        y_pos,
        xerr=np.vstack([med - q1, q3 - med]),
        fmt="o",
        color="#396AB1",
        ecolor="#8AAED6",
        elinewidth=2,
        capsize=4,
    )
    ax.axvline(0.05, color="#B00020", ls="--", lw=1.2, alpha=0.75)
    ax.text(
        0.052,
        0.03,
        "threshold 0.05",
        transform=ax.get_xaxis_transform(),
        color="#B00020",
        fontsize=7.5,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#eeeeee", alpha=0.92),
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["short"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Gate value, median with IQR")
    ax.set_title("Gate values do not collapse to zero", fontsize=10.5, weight="bold", pad=8)
    ax.grid(True, axis="x", alpha=0.22)
    ax.set_xlim(0, max(0.24, float(q3.max()) * 1.15))
    for y, value in zip(y_pos, med):
        ax.text(value + 0.006, y, f"{value:.3f}", va="center", fontsize=7.2)
    fig.tight_layout()
    out = ROOT / "results_v2" / "synthetic_surgery" / "figures" / "gate_distribution.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    crop_workflow()
    plot_case_study()
    if plt is not None:
        plot_accuracy_complexity()
        plot_pruning_appendix()
        plot_gate_distribution_appendix()
    else:
        print("matplotlib is unavailable; regenerated PIL-only paper figures and reused existing matplotlib figures.")
    for name in [
        "workflow_overview.png",
        "accuracy_complexity_tradeoff.png",
        "equation_export_case_study.png",
        "pruning_to_equation_appendix.png",
    ]:
        for paper_fig_dir in PAPER_FIG_DIRS:
            if paper_fig_dir.parent.exists():
                paper_fig_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(FIG_DIR / name, paper_fig_dir / name)
    gate = ROOT / "results_v2" / "synthetic_surgery" / "figures" / "gate_distribution.png"
    if gate.exists():
        for paper_fig_dir in PAPER_FIG_DIRS:
            if paper_fig_dir.parent.exists():
                paper_fig_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(gate, paper_fig_dir / "gate_distribution.png")


if __name__ == "__main__":
    main()
