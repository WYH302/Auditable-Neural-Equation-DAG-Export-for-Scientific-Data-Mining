from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_v2" / "feynman_family_panel"
RUN_DIR = RESULT_ROOT / "runs"
PAPER = ROOT / "paper_elscas"

sys.path.append(str(ROOT / "experiments"))
sys.path.append(str(ROOT / "src"))
from symbolic_export.export_models import load_model_from_json  # noqa: E402
from symbolic_export.graph_export import export_graph  # noqa: E402


LABELS = {
    "mlp": "MLP",
    "kan": "RBF-KAN",
    "stable_eml": "StableEML",
    "eml_kan": "EML-KAN",
}
METHOD_ORDER = ["MLP", "RBF-KAN", "StableEML", "EML-KAN"]
FAMILY_ORDER = ["exp/log", "rational/power/root", "polynomial/product", "trigonometric"]
TRIG_RE = re.compile(r"\b(sin|cos|tan|asin|acos|atan|arcsin|arccos|arctan)\b", re.I)
EXPLOG_RE = re.compile(r"\b(exp|log)\b", re.I)
RATIONAL_RE = re.compile(r"(/|\*\*|sqrt|\^)", re.I)


def classify_formula(formula: str) -> str:
    if TRIG_RE.search(formula):
        return "trigonometric"
    if EXPLOG_RE.search(formula):
        return "exp/log"
    if RATIONAL_RE.search(formula):
        return "rational/power/root"
    if "*" in formula:
        return "polynomial/product"
    return "other"


def metadata() -> dict[str, dict[str, str]]:
    path = ROOT / "data" / "feynman" / "official" / "FeynmanEquations.csv"
    frame = pd.read_csv(path)
    rows = {}
    for _, row in frame.iterrows():
        formula = str(row["Formula"])
        rows[str(row["Filename"])] = {"formula": formula, "family": classify_formula(formula)}
    return rows


def fmt_value(value: float, integer: bool = False) -> str:
    if not math.isfinite(value):
        return "--"
    if integer:
        return f"{value:.0f}"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return f"{value:.4f}"


def load_rows() -> list[dict[str, object]]:
    meta = metadata()
    rows: list[dict[str, object]] = []
    for path in sorted(RUN_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        method = payload.get("model", "")
        label = LABELS.get(method, method)
        dataset = payload.get("dataset", "")
        metric = payload.get("metrics", {})
        row = {
            "dataset": dataset,
            "family": meta.get(dataset, {}).get("family", "other"),
            "formula": meta.get(dataset, {}).get("formula", payload.get("data_meta", {}).get("formula", "")),
            "seed": payload.get("seed", ""),
            "method": method,
            "method_label": label,
            "status": "ok",
            "test_mse": float(metric.get("test", {}).get("mse", math.nan)),
            "tokens": math.nan,
            "basis_terms": math.nan,
            "checkpoint_exists": path.with_suffix(".pt").exists(),
            "json_file": str(path.relative_to(ROOT)),
        }
        try:
            model, loaded_payload = load_model_from_json(path)
            graph = export_graph(
                model,
                method,
                [f"x{i + 1}" for i in range(int(loaded_payload["data_meta"]["input_dim"]))],
            )
            row["tokens"] = float(graph.tokens)
            row["basis_terms"] = float(graph.basis_terms)
        except Exception as exc:  # keep the panel auditable even if one export fails
            row["status"] = f"export_error: {exc}"
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["family"]), str(row["method_label"]))].append(row)

    summary = []
    for family in FAMILY_ORDER:
        for method in METHOD_ORDER:
            group = groups.get((family, method), [])
            if not group:
                continue
            ok = [r for r in group if str(r["status"]) == "ok"]
            equations = len({r["dataset"] for r in group})
            summary.append(
                {
                    "family": family,
                    "method": method,
                    "equations": equations,
                    "runs": len(group),
                    "failures": len(group) - len(ok),
                    "test_mse_mean": mean(float(r["test_mse"]) for r in ok) if ok else math.nan,
                    "tokens_median": median(float(r["tokens"]) for r in ok) if ok else math.nan,
                    "basis_terms_median": median(float(r["basis_terms"]) for r in ok) if ok else math.nan,
                }
            )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(summary: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\scriptsize",
        r"\caption{Low-budget family-balanced Feynman mini-panel. The panel uses one seed, the same neural export serializer as the main audit, and a deliberately small training budget; it is a coverage sanity check rather than a new SOTA benchmark. The local low-dimensional Feynman pool contains no usable ``other'' family and only two polynomial/product files, so the panel is family-stratified where possible rather than perfectly balanced.}",
        r"\label{tab:feynman-family-panel}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Family & Method & Equations & Runs & Failures & Test MSE $\downarrow$ & Median tokens $\downarrow$ & Median basis terms $\downarrow$ \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{row['family']} & {row['method']} & {row['equations']} & {row['runs']} & {row['failures']} "
            f"& ${fmt_value(float(row['test_mse_mean']))}$ & {fmt_value(float(row['tokens_median']), integer=True)} "
            f"& {fmt_value(float(row['basis_terms_median']), integer=True)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ]
    (PAPER / "table_feynman_family_panel.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit(f"No runs found in {RUN_DIR}")
    summary = summarize(rows)
    write_csv(RESULT_ROOT / "family_panel_runs.csv", rows)
    write_csv(RESULT_ROOT / "family_panel_summary.csv", summary)
    write_latex(summary)
    print(f"rows={len(rows)} summary={len(summary)}")


if __name__ == "__main__":
    main()
