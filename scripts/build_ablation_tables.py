from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2"
PAPER = ROOT / "paper_elscas"

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
        "config": "mlp_d2_current",
        "label": "MLP reference",
        "path": RESULTS / "synthetic_final" / "synthetic",
        "models": {"mlp"},
        "comment": "Dense neural reference.",
    },
    {
        "config": "bounded_tanh_d2",
        "label": "bounded-tanh EML",
        "path": RESULTS / "synthetic_final" / "synthetic",
        "models": {"eml_kan"},
        "comment": "Default stable version.",
    },
    {
        "config": "bounded_tanh_d3_warmup",
        "label": "bounded-tanh, depth 3, warmup",
        "path": RESULTS / "synthetic_surgery" / "tanh_d3_weakgate" / "synthetic",
        "models": {"eml_kan"},
        "comment": "Deeper weak-gate diagnostic.",
    },
    {
        "config": "bounded_tanh_d3_nowarmup",
        "label": "bounded-tanh, no warmup",
        "path": RESULTS / "synthetic_surgery" / "tanh_d3_weakgate_nowarmup" / "synthetic",
        "models": {"eml_kan"},
        "comment": "Isolates gate warmup.",
    },
    {
        "config": "raw_exp_d3",
        "label": "raw exp",
        "path": RESULTS / "synthetic_surgery" / "raw_d3_weakgate" / "synthetic",
        "models": {"eml_kan_raw"},
        "comment": "Unbounded exponential risk check.",
    },
    {
        "config": "tau4_d3",
        "label": "bounded-tanh tau=4",
        "path": RESULTS / "synthetic_surgery" / "tanh_tau4_d3_weakgate" / "synthetic",
        "models": {"eml_kan"},
        "comment": "Less conservative saturation.",
    },
    {
        "config": "clip_m8_d3",
        "label": "clip M=8",
        "path": RESULTS / "synthetic_surgery" / "clip_m8_d3_weakgate" / "synthetic",
        "models": {"eml_kan_clip"},
        "comment": "Hard clipping often unstable in OOD.",
    },
    {
        "config": "softclip_m5_d3",
        "label": "softclip M=5",
        "path": RESULTS / "synthetic_surgery" / "softclip_m5_d3_weakgate" / "synthetic",
        "models": {"eml_kan_softclip"},
        "comment": "Smooth clipping negative result.",
    },
    {
        "config": "softclip_m8_identity",
        "label": "softclip M=8 + identity",
        "path": RESULTS / "synthetic_surgery" / "softclip_m8_d3_identity" / "synthetic",
        "models": {"eml_kan_identity"},
        "comment": "Identity residual did not fix OOD.",
    },
]


def as_float(value: object) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def gate_stats(checkpoint_path: Path) -> tuple[float, float]:
    if not checkpoint_path.exists():
        return math.nan, math.nan
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return math.nan, math.nan
    gates = []
    depths = []
    for key, value in payload.get("model_state", {}).items():
        if key.endswith("gate_logits"):
            gate = torch.sigmoid(value.float())
            gates.append(float(gate.mean()))
            active_by_depth = (gate > 0.05).float().mean(dim=(1, 2))
            depths.append(float((active_by_depth > 0).sum()))
    if not gates:
        return math.nan, math.nan
    return float(mean(gates)), float(mean(depths))


def collect_runs() -> list[dict]:
    rows = []
    for cfg in CONFIGS:
        if not cfg["path"].exists():
            continue
        for path in sorted(cfg["path"].glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("dataset") not in DATASETS:
                continue
            if payload.get("model") not in cfg["models"]:
                continue
            metrics = payload.get("metrics", {})
            test_mse = metrics.get("test", {}).get("mse", math.nan)
            ood_mse = metrics.get("ood", {}).get("mse", math.nan)
            gate_mean, effective_depth = gate_stats(path.with_suffix(".pt"))
            rows.append(
                {
                    "config": cfg["config"],
                    "label": cfg["label"],
                    "dataset": payload.get("dataset"),
                    "seed": payload.get("seed"),
                    "model": payload.get("model"),
                    "test_mse": test_mse,
                    "ood_mse": ood_mse,
                    "test_r2": metrics.get("test", {}).get("r2", math.nan),
                    "ood_r2": metrics.get("ood", {}).get("r2", math.nan),
                    "best_epoch": metrics.get("best_epoch", math.nan),
                    "nan_steps": metrics.get("nan_steps", 0),
                    "nonfinite_metric": int(
                        not math.isfinite(float(test_mse)) or not math.isfinite(float(ood_mse))
                    ),
                    "runtime_sec": metrics.get("runtime_sec", math.nan),
                    "gate_mean": gate_mean,
                    "effective_depth": effective_depth,
                    "checkpoint_exists": path.with_suffix(".pt").exists(),
                    "comment": cfg["comment"],
                    "json_file": str(path.relative_to(ROOT)),
                }
            )
    return rows


def finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["config"]].append(row)
    order = {cfg["config"]: idx for idx, cfg in enumerate(CONFIGS)}
    summary = []
    for config, group in sorted(groups.items(), key=lambda item: order.get(item[0], 999)):
        test = finite([r["test_mse"] for r in group])
        ood = finite([r["ood_mse"] for r in group])
        depths = finite([r["effective_depth"] for r in group])
        gates = finite([r["gate_mean"] for r in group])
        summary.append(
            {
                "config": config,
                "label": group[0]["label"],
                "n": len(group),
                "test_mse_mean": mean(test) if test else math.nan,
                "test_mse_std": stdev(test) if len(test) > 1 else 0.0,
                "ood_mse_mean": mean(ood) if ood else math.nan,
                "ood_mse_std": stdev(ood) if len(ood) > 1 else 0.0,
                "nan_steps_sum": sum(int(r["nan_steps"]) for r in group),
                "nonfinite_metric_count": sum(int(r["nonfinite_metric"]) for r in group),
                "checkpoint_count": sum(1 for r in group if r["checkpoint_exists"]),
                "effective_depth_mean": mean(depths) if depths else math.nan,
                "gate_mean": mean(gates) if gates else math.nan,
                "comment": group[0]["comment"],
            }
        )
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_value(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        mantissa, exp = f"{value:.2e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exp)}}}"
    return f"{value:.4f}"


def fmt_plain(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.2f}"


def write_latex(summary: list[dict], path: Path) -> None:
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Synthetic stabilization ablation. Negative results are included because they diagnose whether OOD failure is caused by clipping, gate warmup, or residual initialization.}",
        r"\label{tab:ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Variant & Test MSE $\downarrow$ & OOD MSE $\downarrow$ & NaN/Inf runs & Eff. depth & Comment \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{row['label']} & ${fmt_value(row['test_mse_mean'])}$ & "
            f"${fmt_value(row['ood_mse_mean'])}$ & {row['nonfinite_metric_count']} & "
            f"{fmt_plain(row['effective_depth_mean'])} & {row['comment']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(summary: list[dict], path: Path) -> None:
    lines = [
        "# Synthetic Ablation Report",
        "",
        "| Variant | n | Test MSE mean | OOD MSE mean | NaN steps | NaN/Inf metric runs | Checkpoints | Eff. depth | Comment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['label']} | {row['n']} | {row['test_mse_mean']:.4g} | "
            f"{row['ood_mse_mean']:.4g} | {row['nan_steps_sum']} | "
            f"{row['nonfinite_metric_count']} | {row['checkpoint_count']} | "
            f"{fmt_plain(row['effective_depth_mean'])} | "
            f"{row['comment']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_runs()
    summary = summarize(rows)
    out_dir = RESULTS / "synthetic_surgery"
    write_csv(rows, out_dir / "ablation_runs_20260509.csv")
    write_csv(summary, out_dir / "ablation_table_20260509.csv")
    write_latex(summary, PAPER / "table_ablation.tex")
    write_report(summary, out_dir / "ablation_report_20260509.md")
    print(f"runs={len(rows)} summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
