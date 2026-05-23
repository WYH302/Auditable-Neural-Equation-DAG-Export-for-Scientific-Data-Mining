from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RESULTS = ROOT / "results_v2"
SWEEP = RESULTS / "review_tau_sweep"
SUPPLEMENT = RESULTS / "review_supplement"
PAPER = PROJECT_ROOT / "paper" / "paper_cikm2026"


def tau_from_path(path: Path) -> float:
    for part in path.parts:
        match = re.fullmatch(r"tau([0-9]+(?:\.[0-9]+)?)", part)
        if match:
            return float(match.group(1))
    raise ValueError(f"Could not infer tau from {path}")


def gate_mean(path: Path) -> float:
    checkpoint = path.with_suffix(".pt")
    if not checkpoint.exists():
        return math.nan
    payload = torch.load(checkpoint, map_location="cpu")
    gates = []
    for key, value in payload.get("model_state", {}).items():
        if key.endswith("gate_logits"):
            gates.append(float(torch.sigmoid(value.float()).mean()))
    return mean(gates) if gates else math.nan


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(SWEEP.glob("tau*/synthetic/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "tau": tau_from_path(path),
                "dataset": payload.get("dataset", ""),
                "seed": payload.get("seed", ""),
                "test_mse": float(metrics.get("test", {}).get("mse", math.nan)),
                "ood_mse": float(metrics.get("ood", {}).get("mse", math.nan)),
                "nan_steps": int(metrics.get("nan_steps", 0)),
                "best_epoch": int(metrics.get("best_epoch", 0)),
                "runtime_sec": float(metrics.get("runtime_sec", math.nan)),
                "gate_mean": gate_mean(path),
                "json_file": str(path.relative_to(ROOT)),
            }
        )
    return rows


def finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[float(row["tau"])].append(row)
    out = []
    for tau in sorted(grouped):
        group = grouped[tau]
        tests = finite([row["test_mse"] for row in group])
        oods = finite([row["ood_mse"] for row in group])
        gates = finite([row["gate_mean"] for row in group])
        runtimes = finite([row.get("runtime_sec", math.nan) for row in group])
        out.append(
            {
                "tau": tau,
                "n": len(group),
                "test_mse_mean": mean(tests) if tests else math.nan,
                "ood_mse_mean": mean(oods) if oods else math.nan,
                "nan_steps_sum": sum(int(row["nan_steps"]) for row in group),
                "gate_mean": mean(gates) if gates else math.nan,
                "runtime_sec_mean": mean(runtimes) if runtimes else math.nan,
            }
        )
    return out


def fmt(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "--"
    if abs(value) >= 1 or abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def render_latex(summary: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Small tau sanity sweep on three one-dimensional synthetic OOD failure cases. This is a targeted seed-0 check with 300 training epochs, not an exhaustive hyperparameter search.}",
        r"\label{tab:small-tau-sweep}",
        r"\resulttablesetup",
        r"\begin{tabularx}{\columnwidth}{@{}lrrrrr@{}}",
        r"\toprule",
        r"$\tau$ & $n$ & Test & OOD & Gate & NaN \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{row['tau']:.0f} & {row['n']} & ${fmt(row['test_mse_mean'])}$ & ${fmt(row['ood_mse_mean'])}$ & "
            f"{fmt(row['gate_mean'])} & {row['nan_steps_sum']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = collect_rows()
    summary = summarize(rows)
    write_csv(SUPPLEMENT / "review_small_tau_sweep_runs.csv", rows)
    write_csv(SUPPLEMENT / "review_small_tau_sweep_summary.csv", summary)
    (PAPER / "table_small_tau_sweep.tex").write_text(render_latex(summary), encoding="utf-8")
    print(f"small tau sweep runs={len(rows)} summary={len(summary)}")


if __name__ == "__main__":
    main()
