from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2"
PAPER = ROOT / "paper_elscas"
sys.path.append(str(ROOT / "experiments"))
from run_tabular import build_model  # noqa: E402


METHOD_LABELS = {
    "mlp": "MLP",
    "stable_eml": "StableEML",
    "kan": "RBF-KAN",
    "eml_kan": "EML-KAN",
}

RUN_DIRS = [
    ("Feynman clean12", RESULTS / "feynman_lowdim_clean12" / "runs"),
    (
        "Feynman clean12",
        RESULTS / "baseline_supplement" / "feynman_lowdim_clean12" / "runs",
    ),
    ("SRSD full", RESULTS / "srsd_grouped" / "srsd"),
    ("SRSD full", RESULTS / "eml_rerun_strong_srsd_w80" / "srsd"),
    ("SRSD full", RESULTS / "baseline_supplement" / "srsd_grouped" / "srsd"),
    ("Synthetic final", RESULTS / "synthetic_final" / "synthetic"),
    ("Synthetic final", RESULTS / "baseline_supplement" / "synthetic" / "synthetic"),
    ("Feynman clean12", RESULTS / "baseline_matched" / "feynman_stable_eml_w14" / "runs"),
    ("SRSD full", RESULTS / "baseline_matched" / "srsd_stable_eml_w27" / "srsd"),
    ("Synthetic final", RESULTS / "baseline_matched" / "synthetic_stable_eml_w8" / "synthetic"),
]


def as_float(value: object, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    return float(value)


def as_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def model_input_dim(payload: dict) -> int:
    if payload.get("data_meta", {}).get("input_dim") is not None:
        return as_int(payload["data_meta"]["input_dim"])
    metrics = payload.get("metrics", {})
    if isinstance(metrics.get("x_mean"), list):
        return len(metrics["x_mean"])
    args = payload.get("args", {})
    return as_int(args.get("expected_cols"), 5) - 1


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
    latencies: dict[tuple, float] = {}
    for key, payload in payload_by_key.items():
        model = build_from_payload(payload).to(device).eval()
        x = torch.randn(batch_size, model_input_dim(payload), device=device)
        with torch.no_grad():
            for _ in range(20):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            repeats = 100
            for _ in range(repeats):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        latencies[key] = elapsed * 1000.0 / repeats
    return latencies


def collect_payloads() -> list[tuple[str, Path, dict]]:
    payloads = []
    for benchmark, run_dir in RUN_DIRS:
        if not run_dir.exists():
            continue
        for path in sorted(run_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payloads.append((benchmark, path, payload))
    matched_benchmarks = {
        benchmark
        for benchmark, path, payload in payloads
        if payload.get("model") == "stable_eml" and "baseline_matched" in str(path)
    }
    if matched_benchmarks:
        payloads = [
            (benchmark, path, payload)
            for benchmark, path, payload in payloads
            if not (
                payload.get("model") == "stable_eml"
                and benchmark in matched_benchmarks
                and "baseline_supplement" in str(path)
            )
        ]
    strong_eml_benchmarks = {
        benchmark
        for benchmark, path, payload in payloads
        if payload.get("model") == "eml_kan" and "eml_rerun_strong" in str(path)
    }
    if strong_eml_benchmarks:
        payloads = [
            (benchmark, path, payload)
            for benchmark, path, payload in payloads
            if not (
                payload.get("model") == "eml_kan"
                and benchmark in strong_eml_benchmarks
                and "srsd_grouped" in str(path)
            )
        ]
    return payloads


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "benchmark",
        "dataset",
        "seed",
        "method",
        "method_label",
        "input_dim",
        "width",
        "depth",
        "grid_size",
        "eml_edge_depth",
        "exp_mode",
        "params",
        "runtime_sec",
        "best_epoch",
        "latency_ms_per_1000",
        "early_stop_metric",
        "batch_size",
        "eval_batch_size",
        "lr",
        "weight_decay",
        "nan_steps",
        "json_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def fmt_num(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "--"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["benchmark"], row["method"])].append(row)
    summary = []
    mlp_params = {}
    for (benchmark, method), group in groups.items():
        params = [as_float(r["params"]) for r in group]
        if method == "mlp":
            mlp_params[benchmark] = mean(params)
    for (benchmark, method), group in sorted(groups.items()):
        params = finite([as_float(r["params"]) for r in group])
        runtimes = finite([as_float(r["runtime_sec"]) for r in group])
        epochs = finite([as_float(r["best_epoch"]) for r in group])
        latencies = finite([as_float(r["latency_ms_per_1000"]) for r in group])
        ratio = mean(params) / mlp_params.get(benchmark, mean(params))
        status = "matched" if 0.5 <= ratio <= 2.0 else "unmatched"
        summary.append(
            {
                "benchmark": benchmark,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n": len(group),
                "params_mean": mean(params),
                "params_std": stdev(params) if len(params) > 1 else 0.0,
                "params_min": min(params),
                "params_max": max(params),
                "params_vs_mlp": ratio,
                "runtime_sec_mean": mean(runtimes) if runtimes else math.nan,
                "runtime_sec_std": stdev(runtimes) if len(runtimes) > 1 else 0.0,
                "best_epoch_mean": mean(epochs) if epochs else math.nan,
                "latency_ms_per_1000_mean": mean(latencies) if latencies else math.nan,
                "nan_steps_sum": sum(as_int(r["nan_steps"]) for r in group),
                "status": status,
            }
        )
    return summary


def write_summary_csv(summary: list[dict], path: Path) -> None:
    fieldnames = list(summary[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def write_latex(summary: list[dict], path: Path) -> None:
    order = ["MLP", "StableEML", "RBF-KAN", "EML-KAN"]
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Capacity and runtime audit for the completed comparison. Parameter ratios are relative to MLP within each benchmark; values outside $[0.5,2]$ indicate that the row is not parameter matched. Bold marks the fastest training or inference entry within each benchmark.}",
        r"\label{tab:fairness-audit}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Benchmark & Method & $n$ & Params & Params/MLP & Train sec & Infer ms/1k \\",
        r"\midrule",
    ]
    for benchmark in ["Feynman clean12", "SRSD full", "Synthetic final"]:
        items = [r for r in summary if r["benchmark"] == benchmark]
        items.sort(key=lambda r: order.index(r["method_label"]))
        finite_train = [r["runtime_sec_mean"] for r in items if math.isfinite(r["runtime_sec_mean"])]
        finite_latency = [r["latency_ms_per_1000_mean"] for r in items if math.isfinite(r["latency_ms_per_1000_mean"])]
        best_train = min(finite_train) if finite_train else math.nan
        best_latency = min(finite_latency) if finite_latency else math.nan
        for row in items:
            train = fmt_num(row["runtime_sec_mean"], 1)
            latency = fmt_num(row["latency_ms_per_1000_mean"], 3)
            if math.isfinite(best_train) and row["runtime_sec_mean"] == best_train:
                train = rf"\best{{{train}}}"
            if math.isfinite(best_latency) and row["latency_ms_per_1000_mean"] == best_latency:
                latency = rf"\best{{{latency}}}"
            lines.append(
                f"{benchmark} & {row['method_label']} & {row['n']} & "
                f"{fmt_num(row['params_mean'], 0)} & {fmt_num(row['params_vs_mlp'], 2)} & "
                f"{train} & {latency} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(summary: list[dict], path: Path) -> None:
    lines = [
        "# Fairness Audit",
        "",
        "This audit uses the completed main-table runs. StableEML is taken from the matched-budget rerun when available, replacing the earlier larger-capacity StableEML pilot.",
        "",
        "| Benchmark | Method | n | Params mean | Params/MLP | Train sec | Inference ms/1k | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['benchmark']} | {row['method_label']} | {row['n']} | "
            f"{fmt_num(row['params_mean'], 0)} | {fmt_num(row['params_vs_mlp'], 2)} | "
            f"{fmt_num(row['runtime_sec_mean'], 1)} | "
            f"{fmt_num(row['latency_ms_per_1000_mean'], 3)} | {row['status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payloads = collect_payloads()
    unique_payloads = {}
    for _, _, payload in payloads:
        unique_payloads.setdefault(config_key(payload), payload)
    latencies = measure_latency(unique_payloads)
    rows = []
    for benchmark, path, payload in payloads:
        args = payload.get("args", {})
        metrics = payload.get("metrics", {})
        model = build_from_payload(payload)
        key = config_key(payload)
        rows.append(
            {
                "benchmark": benchmark,
                "dataset": payload.get("dataset", ""),
                "seed": payload.get("seed", ""),
                "method": payload.get("model", ""),
                "method_label": METHOD_LABELS.get(payload.get("model", ""), payload.get("model", "")),
                "input_dim": model_input_dim(payload),
                "width": as_int(args.get("width"), 32),
                "depth": as_int(args.get("depth"), 2),
                "grid_size": 16 if payload.get("model") == "kan" else "",
                "eml_edge_depth": as_int(args.get("depth"), 2) if payload.get("model") == "eml_kan" else "",
                "exp_mode": args.get("exp_mode", "bounded_tanh"),
                "params": count_params(model),
                "runtime_sec": metrics.get("runtime_sec", ""),
                "best_epoch": metrics.get("best_epoch", ""),
                "latency_ms_per_1000": latencies.get(key, math.nan),
                "early_stop_metric": metrics.get("early_stop_metric", args.get("early_stop_metric", "")),
                "batch_size": args.get("batch_size", ""),
                "eval_batch_size": args.get("eval_batch_size", ""),
                "lr": args.get("lr", ""),
                "weight_decay": args.get("weight_decay", ""),
                "nan_steps": metrics.get("nan_steps", 0),
                "json_file": str(path.relative_to(ROOT)),
            }
        )
    summary = summarize(rows)
    write_csv(rows, RESULTS / "fairness_audit_runs_20260509.csv")
    write_summary_csv(summary, RESULTS / "fairness_audit_summary_20260509.csv")
    write_latex(summary, PAPER / "table_fairness_audit.tex")
    write_report(summary, RESULTS / "fairness_audit_report_20260509.md")
    print(f"runs={len(rows)} configs={len(unique_payloads)} summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
