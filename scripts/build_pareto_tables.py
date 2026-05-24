from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2"
PAPER = ROOT / "paper_elscas"
sys.path.append(str(ROOT / "experiments"))
sys.path.append(str(ROOT / "scripts"))
from run_tabular import build_model  # noqa: E402
try:
    from symbolic_recovery import recover_one  # noqa: E402
except ModuleNotFoundError:
    recover_one = None


METHOD_LABELS = {
    "mlp": "MLP",
    "stable_eml": "StableEML",
    "kan": "RBF-KAN",
    "eml_kan": "EML-KAN",
}

BUDGET_LABELS = {"small": "Small", "medium": "Medium", "large": "Large"}


def as_float(value: object, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    return float(value)


def as_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def model_input_dim(payload: dict) -> int:
    metrics = payload.get("metrics", {})
    return len(metrics.get("x_mean", []))


def build_from_payload(payload: dict):
    args = payload.get("args", {})
    return build_model(
        payload["model"],
        model_input_dim(payload),
        as_int(args.get("width"), 32),
        as_int(args.get("depth"), 2),
        as_float(args.get("tau"), 2.0),
        as_float(args.get("eta"), 0.35),
        args.get("exp_mode", "bounded_tanh"),
        as_float(args.get("clip_m"), 8.0),
        args.get("w0_init", "normal"),
    )


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def config_key(payload: dict) -> tuple:
    args = payload.get("args", {})
    return (
        payload["model"],
        model_input_dim(payload),
        as_int(args.get("width"), 32),
        as_int(args.get("depth"), 2),
        as_float(args.get("tau"), 2.0),
        as_float(args.get("eta"), 0.35),
        args.get("exp_mode", "bounded_tanh"),
        as_float(args.get("clip_m"), 8.0),
        args.get("w0_init", "normal"),
    )


def measure_latency(payload_by_key: dict[tuple, dict], batch_size: int = 1000) -> dict[tuple, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latencies = {}
    for key, payload in payload_by_key.items():
        model = build_from_payload(payload).to(device).eval()
        x = torch.randn(batch_size, model_input_dim(payload), device=device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            repeats = 50
            for _ in range(repeats):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies[key] = (time.perf_counter() - start) * 1000.0 / repeats
    return latencies


def collect_payloads() -> list[tuple[Path, dict]]:
    run_dir = RESULTS / "pareto_budget_v2" / "synthetic"
    payloads = []
    for path in sorted(run_dir.glob("*.json")):
        payloads.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return payloads


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_existing_summary(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            method_label = METHOD_LABELS.get(method, row.get("method_label", method))
            rows.append(
                {
                    "budget": row["budget"],
                    "budget_label": BUDGET_LABELS.get(row["budget"], row["budget_label"]),
                    "method": method,
                    "method_label": method_label,
                    "n": as_int(row["n"]),
                    "params_mean": as_float(row["params_mean"]),
                    "train_sec_mean": as_float(row["train_sec_mean"]),
                    "latency_ms_per_1000_mean": as_float(row["latency_ms_per_1000_mean"]),
                    "test_mse_mean": as_float(row["test_mse_mean"]),
                    "test_mse_std": as_float(row["test_mse_std"]),
                    "ood_mse_mean": as_float(row["ood_mse_mean"]),
                    "ood_mse_std": as_float(row["ood_mse_std"]),
                    "exact_count": as_int(row["exact_count"]),
                    "complexity_mean": as_float(row["complexity_mean"]),
                    "nan_steps_sum": as_int(row["nan_steps_sum"]),
                }
            )
    return rows


def finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def fmt_value(value: float) -> str:
    if not math.isfinite(float(value)):
        return "--"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        mantissa, exp = f"{value:.2e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exp)}}}"
    return f"{value:.4f}"


def bold_if(text: str, condition: bool) -> str:
    return rf"\best{{{text}}}" if condition else text


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["budget"], row["method"])].append(row)
    summary = []
    for (budget, method), group in sorted(groups.items(), key=lambda item: (["small", "medium", "large"].index(item[0][0]), item[0][1])):
        tests = finite([r["test_mse"] for r in group])
        oods = finite([r["ood_mse"] for r in group])
        params = finite([r["params"] for r in group])
        runtimes = finite([r["runtime_sec"] for r in group])
        latency = finite([r["latency_ms_per_1000"] for r in group])
        complexity = finite([r["complexity"] for r in group])
        summary.append(
            {
                "budget": budget,
                "budget_label": BUDGET_LABELS[budget],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n": len(group),
                "params_mean": mean(params),
                "train_sec_mean": mean(runtimes),
                "latency_ms_per_1000_mean": mean(latency),
                "test_mse_mean": mean(tests),
                "test_mse_std": stdev(tests) if len(tests) > 1 else 0.0,
                "ood_mse_mean": mean(oods),
                "ood_mse_std": stdev(oods) if len(oods) > 1 else 0.0,
                "exact_count": sum(int(r["exact_recovery"]) for r in group),
                "complexity_mean": mean(complexity),
                "nan_steps_sum": sum(as_int(r["nan_steps"]) for r in group),
            }
        )
    return summary


def write_latex(summary: list[dict], path: Path) -> None:
    order = ["small", "medium", "large"]
    method_order = ["MLP", "RBF-KAN", "StableEML", "EML-KAN"]
    summary = sorted(summary, key=lambda r: (order.index(r["budget"]), method_order.index(r["method_label"])))
    best_by_budget = {}
    for budget in order:
        rows = [r for r in summary if r["budget"] == budget]
        best_by_budget[budget] = {
            "train_sec_mean": min(r["train_sec_mean"] for r in rows),
            "latency_ms_per_1000_mean": min(r["latency_ms_per_1000_mean"] for r in rows),
            "test_mse_mean": min(r["test_mse_mean"] for r in rows),
            "ood_mse_mean": min(r["ood_mse_mean"] for r in rows),
            "exact_count": max(r["exact_count"] for r in rows),
            "complexity_mean": min(r["complexity_mean"] for r in rows),
        }
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Synthetic Small/Medium/Large budget Pareto summary. Parameter budgets are matched within each budget level; exact recovery is measured by sparse symbolic distillation of each trained neural model. Bold marks the best value within each budget for metrics with a clear direction.}",
        r"\label{tab:pareto-budget}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Budget & Method & Params & Train sec & Infer ms/1k & Test MSE $\downarrow$ & OOD MSE $\downarrow$ & Exact $\uparrow$ & Complexity $\downarrow$ \\",
        r"\midrule",
    ]
    for row in summary:
        best = best_by_budget[row["budget"]]
        train = bold_if(f"{row['train_sec_mean']:.1f}", row["train_sec_mean"] == best["train_sec_mean"])
        latency = bold_if(f"{row['latency_ms_per_1000_mean']:.3f}", row["latency_ms_per_1000_mean"] == best["latency_ms_per_1000_mean"])
        test = bold_if(rf"${fmt_value(row['test_mse_mean'])}$", row["test_mse_mean"] == best["test_mse_mean"])
        ood = bold_if(rf"${fmt_value(row['ood_mse_mean'])}$", row["ood_mse_mean"] == best["ood_mse_mean"])
        exact = bold_if(f"{row['exact_count']}/{row['n']}", row["exact_count"] == best["exact_count"])
        complexity = bold_if(f"{row['complexity_mean']:.2f}", row["complexity_mean"] == best["complexity_mean"])
        lines.append(
            f"{row['budget_label']} & {row['method_label']} & "
            f"{row['params_mean']:.0f} & {train} & {latency} & {test} & "
            f"{ood} & {exact} & {complexity} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(summary: list[dict], path: Path) -> None:
    lines = [
        "# Synthetic Pareto Budget Report",
        "",
        "| Budget | Method | n | Params | Train sec | Infer ms/1k | Test MSE | OOD MSE | Exact | Complexity | NaN steps |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['budget_label']} | {row['method_label']} | {row['n']} | "
            f"{row['params_mean']:.0f} | {row['train_sec_mean']:.2f} | "
            f"{row['latency_ms_per_1000_mean']:.4f} | {row['test_mse_mean']:.4g} | "
            f"{row['ood_mse_mean']:.4g} | {row['exact_count']}/{row['n']} | "
            f"{row['complexity_mean']:.2f} | {row['nan_steps_sum']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_pareto(summary: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"MLP": "#4c78a8", "RBF-KAN": "#f58518", "StableEML": "#54a24b", "EML-KAN": "#b279a2"}
    for x_key, x_label, filename in [
        ("params_mean", "Trainable parameters", "pareto_params_test_mse.png"),
        ("train_sec_mean", "Training time (sec)", "pareto_time_test_mse.png"),
        ("latency_ms_per_1000_mean", "Inference latency (ms / 1000 samples)", "pareto_latency_test_mse.png"),
    ]:
        plt.figure(figsize=(6.2, 4.2), dpi=180)
        for method in ["MLP", "RBF-KAN", "StableEML", "EML-KAN"]:
            rows = [r for r in summary if r["method_label"] == method]
            rows.sort(key=lambda r: r[x_key])
            xs = [r[x_key] for r in rows]
            ys = [r["test_mse_mean"] for r in rows]
            plt.plot(xs, ys, marker="o", label=method, color=colors[method])
            for x, y, r in zip(xs, ys, rows):
                plt.annotate(r["budget_label"][0], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        finite_rows = [r for r in summary if math.isfinite(r[x_key]) and math.isfinite(r["test_mse_mean"])]
        y_values = [r["test_mse_mean"] for r in finite_rows]
        plt.ylim(min(y_values) / 2.2, max(y_values) * 1.8)
        best = min(finite_rows, key=lambda r: r["test_mse_mean"])
        worst = max(finite_rows, key=lambda r: r["test_mse_mean"])
        for row, label, offset in [(best, "min", (8, 10)), (worst, "max", (8, -16))]:
            plt.scatter(
                [row[x_key]],
                [row["test_mse_mean"]],
                s=64,
                facecolors="none",
                edgecolors="#111111",
                linewidths=1.2,
                zorder=5,
            )
            plt.annotate(
                f"{label} {row['test_mse_mean']:.1e}",
                (row[x_key], row["test_mse_mean"]),
                textcoords="offset points",
                xytext=offset,
                fontsize=7.2,
                color="#111111",
                arrowprops=dict(arrowstyle="-", color="#111111", lw=0.7, alpha=0.75),
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(x_label)
        plt.ylabel("Synthetic test MSE")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / filename, bbox_inches="tight", pad_inches=0.05)
        plt.close()


def main() -> None:
    out_dir = RESULTS / "pareto_budget_v2"
    if recover_one is None:
        summary = load_existing_summary(out_dir / "pareto_summary.csv")
        write_latex(summary, PAPER / "table_pareto_budget.tex")
        write_report(summary, out_dir / "pareto_budget_report.md")
        plot_pareto(summary, out_dir / "figures")
        paper_fig_dir = PAPER / "figures_updated"
        paper_fig_dir.mkdir(parents=True, exist_ok=True)
        for name in ["pareto_params_test_mse.png", "pareto_time_test_mse.png", "pareto_latency_test_mse.png"]:
            shutil.copy2(out_dir / "figures" / name, paper_fig_dir / name)
        print(f"reused existing pareto summary rows={len(summary)}")
        return

    payloads = collect_payloads()
    payload_by_key = {}
    for _, payload in payloads:
        payload_by_key.setdefault(config_key(payload), payload)
    latencies = measure_latency(payload_by_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    recovery_rows = []
    for path, payload in payloads:
        metrics = payload.get("metrics", {})
        test = metrics.get("test", {})
        ood = metrics.get("ood", {})
        key = config_key(payload)
        rec = recover_one(path, device)
        recovery_rows.append(rec)
        rows.append(
            {
                "budget": payload.get("budget"),
                "dataset": payload.get("dataset"),
                "seed": payload.get("seed"),
                "method": payload.get("model"),
                "method_label": METHOD_LABELS.get(payload.get("model"), payload.get("model")),
                "params": count_params(build_from_payload(payload)),
                "runtime_sec": metrics.get("runtime_sec", math.nan),
                "best_epoch": metrics.get("best_epoch", math.nan),
                "latency_ms_per_1000": latencies.get(key, math.nan),
                "test_mse": test.get("mse", math.nan),
                "ood_mse": ood.get("mse", math.nan),
                "exact_recovery": rec["exact_recovery"],
                "complexity": rec["complexity"],
                "symbolic_test_mse": rec["symbolic_test_mse"],
                "nan_steps": metrics.get("nan_steps", 0),
                "json_file": str(path.relative_to(ROOT)),
            }
        )
    summary = summarize(rows)
    write_csv(rows, out_dir / "pareto_runs.csv")
    write_csv(recovery_rows, out_dir / "pareto_symbolic_recovery_runs.csv")
    write_csv(summary, out_dir / "pareto_summary.csv")
    write_latex(summary, PAPER / "table_pareto_budget.tex")
    write_report(summary, out_dir / "pareto_budget_report.md")
    plot_pareto(summary, out_dir / "figures")
    paper_fig_dir = PAPER / "figures_updated"
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    for name in ["pareto_params_test_mse.png", "pareto_time_test_mse.png", "pareto_latency_test_mse.png"]:
        shutil.copy2(out_dir / "figures" / name, paper_fig_dir / name)
    print(f"runs={len(rows)} summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
