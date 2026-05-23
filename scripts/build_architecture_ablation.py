from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2"
PAPER = ROOT / "paper_elscas"
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from src.symbolic_export.export_models import load_model_from_json  # noqa: E402
from src.symbolic_export.graph_export import export_graph  # noqa: E402
from symbolic_recovery import recover_one  # noqa: E402

DATASETS = {
    "exp_1d",
    "log_1d_x_plus_2",
    "exp_log_1d",
    "poly_1d_x2_x_1",
    "inverse_quadratic",
    "sin_weak_case",
    "exp_log_2d",
    "kinetic_energy",
}

CONFIGS = [
    {
        "config": "t1_weakgate",
        "label": "T=1 full",
        "path": RESULTS / "architecture_ablation" / "t1_weakgate" / "synthetic",
        "comment": "Shallow recursion baseline.",
    },
    {
        "config": "t2_current",
        "label": "T=2 full",
        "path": RESULTS / "synthetic_final" / "synthetic",
        "comment": "Original matched synthetic setting.",
    },
    {
        "config": "t3_weakgate",
        "label": "T=3 full",
        "path": RESULTS / "synthetic_surgery" / "tanh_d3_weakgate" / "synthetic",
        "comment": "Full exp-log edge family.",
    },
    {
        "config": "no_residual",
        "label": "no residual",
        "path": RESULTS / "architecture_ablation" / "no_residual_d3_weakgate" / "synthetic",
        "comment": "Removes the linear path.",
    },
    {
        "config": "no_log",
        "label": "exp branch only",
        "path": RESULTS / "architecture_ablation" / "no_log_d3_weakgate" / "synthetic",
        "comment": "Removes the logarithmic branch.",
    },
    {
        "config": "only_log",
        "label": "log branch only",
        "path": RESULTS / "architecture_ablation" / "only_log_d3_weakgate" / "synthetic",
        "comment": "Removes the exponential branch.",
    },
    {
        "config": "no_gate",
        "label": "no gates",
        "path": RESULTS / "architecture_ablation" / "no_gate_d3_weakgate" / "synthetic",
        "comment": "Keeps all recursive terms active.",
    },
    {
        "config": "no_gate_penalty",
        "label": "no gate penalty",
        "path": RESULTS / "architecture_ablation" / "no_gate_penalty_d3" / "synthetic",
        "comment": "Learns gates without sparsity pressure.",
    },
    {
        "config": "tau4",
        "label": "tau=4 full",
        "path": RESULTS / "synthetic_surgery" / "tanh_tau4_d3_weakgate" / "synthetic",
        "comment": "Less conservative bounded exponential.",
    },
]


def as_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def fmt_value(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        mantissa, exp = f"{value:.2e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exp)}}}"
    return f"{value:.4f}"


def fmt_plain(value: float, digits: int = 1) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def gate_stats(path: Path) -> tuple[float, float]:
    if not path.exists():
        return math.nan, math.nan
    state = torch.load(path, map_location="cpu")
    gates = []
    depths = []
    for key, value in state.get("model_state", {}).items():
        if key.endswith("gate_logits"):
            gate = torch.sigmoid(value.float())
            gates.append(float(gate.mean()))
            active_by_depth = (gate > 0.05).float().mean(dim=(1, 2))
            depths.append(float((active_by_depth > 0).sum()))
    if not gates:
        return math.nan, math.nan
    return float(mean(gates)), float(mean(depths))


def graph_stats(path: Path) -> tuple[float, float, float]:
    if not path.with_suffix(".pt").exists():
        return math.nan, math.nan, math.nan
    try:
        model, payload = load_model_from_json(path)
        in_dim = len(payload["metrics"]["x_mean"])
        graph = export_graph(model, payload["model"], [f"x{i + 1}" for i in range(in_dim)])
        return float(graph.tokens), float(graph.active_terms), float(graph.basis_terms)
    except Exception as exc:
        print(f"graph export failed for {path}: {exc}")
        return math.nan, math.nan, math.nan


def collect_rows() -> list[dict]:
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for cfg in CONFIGS:
        if not cfg["path"].exists():
            print(f"missing {cfg['path']}")
            continue
        for path in sorted(cfg["path"].glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("model") != "eml_kan" or payload.get("dataset") not in DATASETS:
                continue
            metrics = payload.get("metrics", {})
            test_mse = as_float(metrics.get("test", {}).get("mse"))
            ood_mse = as_float(metrics.get("ood", {}).get("mse"))
            gate_mean, eff_depth = gate_stats(path.with_suffix(".pt"))
            tokens, active_terms, basis_terms = graph_stats(path)
            try:
                recovery = recover_one(path, device)
                symbolic_mse = as_float(recovery.get("symbolic_test_mse"))
                exact = int(recovery.get("exact_recovery", 0))
            except Exception as exc:
                print(f"symbolic recovery failed for {path}: {exc}")
                symbolic_mse = math.nan
                exact = 0
            rows.append(
                {
                    "config": cfg["config"],
                    "label": cfg["label"],
                    "dataset": payload.get("dataset"),
                    "seed": payload.get("seed"),
                    "test_mse": test_mse,
                    "ood_mse": ood_mse,
                    "nan_steps": int(metrics.get("nan_steps", 0)),
                    "nonfinite": int(not math.isfinite(test_mse) or not math.isfinite(ood_mse)),
                    "tokens": tokens,
                    "active_terms": active_terms,
                    "basis_terms": basis_terms,
                    "gate_mean": gate_mean,
                    "effective_depth": eff_depth,
                    "symbolic_mse": symbolic_mse,
                    "exact_recovery": exact,
                    "comment": cfg["comment"],
                    "json_file": str(path.relative_to(ROOT)),
                }
            )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)
    order = {cfg["config"]: i for i, cfg in enumerate(CONFIGS)}
    summary = []
    for config, group in sorted(grouped.items(), key=lambda item: order.get(item[0], 999)):
        def vals(key):
            return finite([r[key] for r in group])
        summary.append(
            {
                "config": config,
                "label": group[0]["label"],
                "n": len(group),
                "test_mean": mean(vals("test_mse")) if vals("test_mse") else math.nan,
                "test_median": median(vals("test_mse")) if vals("test_mse") else math.nan,
                "ood_mean": mean(vals("ood_mse")) if vals("ood_mse") else math.nan,
                "ood_median": median(vals("ood_mse")) if vals("ood_mse") else math.nan,
                "tokens_mean": mean(vals("tokens")) if vals("tokens") else math.nan,
                "tokens_median": median(vals("tokens")) if vals("tokens") else math.nan,
                "active_terms_mean": mean(vals("active_terms")) if vals("active_terms") else math.nan,
                "gate_mean": mean(vals("gate_mean")) if vals("gate_mean") else math.nan,
                "effective_depth": mean(vals("effective_depth")) if vals("effective_depth") else math.nan,
                "symbolic_mse_mean": mean(vals("symbolic_mse")) if vals("symbolic_mse") else math.nan,
                "exact_count": sum(int(r["exact_recovery"]) for r in group),
                "nonfinite_count": sum(int(r["nonfinite"]) for r in group),
                "nan_steps": sum(int(r["nan_steps"]) for r in group),
                "comment": group[0]["comment"],
            }
        )
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(summary: list[dict], path: Path) -> None:
    best_test = min((r["test_mean"] for r in summary if math.isfinite(r["test_mean"])), default=math.nan)
    best_tokens = min((r["tokens_mean"] for r in summary if math.isfinite(r["tokens_mean"])), default=math.nan)
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Architecture ablation of EML-KAN components on the eight-function synthetic diagnostic suite. All variants use the same width, optimizer, data splits, and seeds unless noted; this table tests whether the residual path, exp/log branches, gates, recursion depth, and exponential temperature are arbitrary additions or useful design choices.}",
        r"\label{tab:architecture-ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrl}",
        r"\toprule",
        r"Variant & $n$ & Test MSE $\downarrow$ & OOD MSE $\downarrow$ & Tokens $\downarrow$ & Exact rec. $\uparrow$ & Eff. depth & NaN/Inf & Comment \\",
        r"\midrule",
    ]
    for row in summary:
        test = fmt_value(row["test_mean"])
        tokens = fmt_plain(row["tokens_mean"], 0)
        if math.isfinite(row["test_mean"]) and abs(row["test_mean"] - best_test) <= max(abs(best_test), 1.0) * 1e-12:
            test = rf"\mathbf{{{test}}}"
        if math.isfinite(row["tokens_mean"]) and abs(row["tokens_mean"] - best_tokens) <= max(abs(best_tokens), 1.0) * 1e-12:
            tokens = rf"\textbf{{{tokens}}}"
        lines.append(
            f"{row['label']} & {row['n']} & ${test}$ & ${fmt_value(row['ood_mean'])}$ & {tokens} & "
            f"{row['exact_count']}/{row['n']} & {fmt_plain(row['effective_depth'], 1)} & "
            f"{row['nonfinite_count']} & {row['comment']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(summary: list[dict], path: Path) -> None:
    lines = [
        "# Architecture Ablation Report",
        "",
        "| Variant | n | Test mean | OOD mean | Tokens mean | Exact | Eff depth | NaN/Inf | Comment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['label']} | {row['n']} | {row['test_mean']:.4g} | {row['ood_mean']:.4g} | "
            f"{row['tokens_mean']:.1f} | {row['exact_count']}/{row['n']} | {row['effective_depth']:.2f} | "
            f"{row['nonfinite_count']} | {row['comment']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    summary = summarize(rows)
    out = RESULTS / "architecture_ablation"
    write_csv(rows, out / "architecture_ablation_runs.csv")
    write_csv(summary, out / "architecture_ablation_summary.csv")
    write_latex(summary, PAPER / "table_architecture_ablation.tex")
    write_report(summary, out / "architecture_ablation_report.md")
    print(f"rows={len(rows)} summary={len(summary)}")


if __name__ == "__main__":
    main()
