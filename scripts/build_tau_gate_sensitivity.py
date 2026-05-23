from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RESULTS = ROOT / "results_v2"
PAPER = PROJECT_ROOT / "paper" / "paper_cikm2026"
SUPPLEMENT = RESULTS / "review_supplement"

ARCH_SUMMARY = RESULTS / "architecture_ablation" / "architecture_ablation_summary.csv"
SURGERY_SUMMARY = RESULTS / "synthetic_surgery" / "ablation_table_20260509.csv"

CONFIG_ORDER = [
    "t2_current",
    "t3_weakgate",
    "bounded_tanh_d3_nowarmup",
    "no_gate",
    "no_gate_penalty",
    "tau4",
    "raw_exp_d3",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def as_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def variant_family(config: str) -> str:
    if config.startswith("t") and ("current" in config or "weakgate" in config):
        return "depth"
    if "tau" in config:
        return "temperature"
    if "gate" in config or "warmup" in config:
        return "gate"
    if "raw" in config or "clip" in config:
        return "stability"
    return "depth"


def normalize_arch_row(row: dict[str, str]) -> dict:
    return {
        "source": "architecture_ablation_summary.csv",
        "config": row["config"],
        "label": row["label"],
        "n": as_int(row["n"]),
        "test_mse": as_float(row["test_mean"]),
        "ood_mse": as_float(row["ood_mean"]),
        "tokens": as_float(row["tokens_mean"]),
        "gate_mean": as_float(row["gate_mean"]),
        "effective_depth": as_float(row["effective_depth"]),
        "nonfinite": as_int(row["nonfinite_count"]),
        "nan_steps": as_int(row["nan_steps"]),
        "family": variant_family(row["config"]),
    }


def normalize_surgery_row(row: dict[str, str]) -> dict:
    return {
        "source": "ablation_table_20260509.csv",
        "config": row["config"],
        "label": row["label"],
        "n": as_int(row["n"]),
        "test_mse": as_float(row["test_mse_mean"]),
        "ood_mse": as_float(row["ood_mse_mean"]),
        "tokens": math.nan,
        "gate_mean": as_float(row.get("gate_mean")),
        "effective_depth": as_float(row.get("effective_depth_mean")),
        "nonfinite": as_int(row.get("nonfinite_metric_count")),
        "nan_steps": as_int(row.get("nan_steps_sum")),
        "family": variant_family(row["config"]),
    }


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    rows.extend(normalize_arch_row(row) for row in read_csv(ARCH_SUMMARY))
    surgery = {row["config"]: normalize_surgery_row(row) for row in read_csv(SURGERY_SUMMARY)}
    if "bounded_tanh_d3_nowarmup" in surgery:
        rows.append(surgery["bounded_tanh_d3_nowarmup"])
    if "raw_exp_d3" in surgery:
        rows.append(surgery["raw_exp_d3"])
    return pick_rows(rows, CONFIG_ORDER)


def pick_rows(rows: list[dict], configs: list[str]) -> list[dict]:
    lookup = {row["config"]: row for row in rows}
    return [lookup[config] for config in configs if config in lookup]


def fmt(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        mantissa, exp = f"{value:.2e}".split("e")
        return rf"{mantissa}e{int(exp)}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.3f}"


def render_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Existing ablation sensitivity for temperature and gates on the synthetic diagnostic suite. This is a summary of completed runs rather than a new exhaustive sweep; it reports the available test/OOD error, token count, gate mean, effective depth, nonfinite metric count, and NaN-step total.}",
        r"\label{tab:tau-gate-sensitivity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Factor & Variant & $n$ & Test & OOD & Tok. & Gate & Eff. depth & Fail \\",
        r"\midrule",
    ]
    for row in rows:
        fail = f"{row['nonfinite']}/{row['nan_steps']}"
        lines.append(
            f"{row['family']} & {row['label']} & {row['n']} & ${fmt(row['test_mse'])}$ & ${fmt(row['ood_mse'])}$ & "
            f"{fmt(row['tokens'])} & {fmt(row['gate_mean'])} & {fmt(row['effective_depth'])} & {fail} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["source", "config", "label", "family", "n", "test_mse", "ood_mse", "tokens", "gate_mean", "effective_depth", "nonfinite", "nan_steps"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def main() -> None:
    rows = collect_rows()
    write_csv(SUPPLEMENT / "review_tau_gate_sensitivity.csv", rows)
    (PAPER / "table_tau_gate_sensitivity.tex").write_text(render_latex(rows), encoding="utf-8")
    print(f"tau/gate sensitivity rows={len(rows)}")


if __name__ == "__main__":
    main()
