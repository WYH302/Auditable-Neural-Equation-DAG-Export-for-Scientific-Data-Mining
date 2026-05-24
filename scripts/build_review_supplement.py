from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2"
OUT_DIR = RESULTS / "review_supplement"
sys.path.append(str(ROOT / "src"))
from symbolic_export.export_models import load_model_from_json  # noqa: E402
from symbolic_export.graph_export import export_graph  # noqa: E402


def as_float(value: object) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite(values: list[float] | tuple[float, ...]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def mean_finite(values: list[float] | tuple[float, ...]) -> float:
    clean = finite(values)
    return mean(clean) if clean else math.nan


def geomean_finite(values: list[float] | tuple[float, ...]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value)) and float(value) > 0]
    if not clean:
        return math.nan
    return math.exp(mean([math.log(value) for value in clean]))


def sign_test_p_value(candidate_wins: int, baseline_wins: int) -> float:
    trials = candidate_wins + baseline_wins
    if trials == 0:
        return math.nan
    tail = min(candidate_wins, baseline_wins)
    probability = sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def paired_summary(rows: list[dict], baseline_key: str, candidate_key: str) -> dict:
    ratios: list[float] = []
    candidate_wins = 0
    baseline_wins = 0
    ties = 0
    for row in rows:
        baseline = as_float(row.get(baseline_key))
        candidate = as_float(row.get(candidate_key))
        if not math.isfinite(baseline) or not math.isfinite(candidate):
            continue
        if baseline > 0:
            ratios.append(candidate / baseline)
        if candidate < baseline:
            candidate_wins += 1
        elif baseline < candidate:
            baseline_wins += 1
        else:
            ties += 1
    return {
        "n": candidate_wins + baseline_wins + ties,
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "mean_ratio": mean_finite(ratios),
        "geomean_ratio": geomean_finite(ratios),
        "sign_p": sign_test_p_value(candidate_wins, baseline_wins),
    }


def make_pairwise_row(
    comparison: str,
    rows: list[dict],
    baseline_key: str,
    candidate_key: str,
) -> dict:
    summary = paired_summary(rows, baseline_key, candidate_key)
    return {
        "comparison": comparison,
        "n": summary["n"],
        "eml_wins": summary["candidate_wins"],
        "mlp_wins": summary["baseline_wins"],
        "ties": summary["ties"],
        "mean_eml_over_mlp": summary["mean_ratio"],
        "geomean_eml_over_mlp": summary["geomean_ratio"],
        "sign_test_p": summary["sign_p"],
    }


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])


def fmt_float(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def group_metric_summary(records: list[dict], group_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[tuple(record.get(key, "") for key in group_keys)].append(record)
    rows = []
    for key, group in sorted(grouped.items()):
        row = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        row.update(
            {
                "n": len(group),
                "test_mse_mean": mean_finite([as_float(item.get("test_mse")) for item in group]),
                "ood_mse_mean": mean_finite([as_float(item.get("ood_mse")) for item in group]),
                "runtime_sec_mean": mean_finite([as_float(item.get("runtime_sec")) for item in group]),
                "nan_steps_sum": sum(int(as_float(item.get("nan_steps")) or 0) for item in group),
            }
        )
        if any("tokens" in item for item in group):
            row["tokens_mean"] = mean_finite([as_float(item.get("tokens")) for item in group])
            row["basis_terms_mean"] = mean_finite([as_float(item.get("basis_terms")) for item in group])
        rows.append(row)
    return rows


def flatten_payload(payload: dict) -> dict:
    metrics = payload.get("metrics", {})
    return {
        "dataset": payload.get("dataset", ""),
        "seed": payload.get("seed", ""),
        "method": payload.get("model", ""),
        "variant": payload.get("variant", payload.get("model", "")),
        "noise_frac": payload.get("noise_frac", ""),
        "test_mse": metrics.get("test", {}).get("mse", math.nan),
        "ood_mse": metrics.get("ood", {}).get("mse", math.nan),
        "val_mse": metrics.get("val", {}).get("mse", math.nan),
        "runtime_sec": metrics.get("runtime_sec", math.nan),
        "nan_steps": metrics.get("nan_steps", 0),
        "best_epoch": metrics.get("best_epoch", math.nan),
    }


def add_export_stats(json_path: Path, row: dict) -> dict:
    model, payload = load_model_from_json(json_path)
    variable_names = [f"x{i + 1}" for i in range(len(payload["metrics"]["x_mean"]))]
    export = export_graph(model, payload["model"], variable_names)
    row = dict(row)
    row.update(
        {
            "tokens": export.tokens,
            "basis_terms": export.basis_terms,
            "exp_log_ops": export.exp_log_ops,
            "export_kind": export.kind,
        }
    )
    return row


def build_mlp_large_summary() -> tuple[list[dict], list[dict]]:
    root = OUT_DIR / "mlp_large" / "equation_native"
    records = []
    for payload in read_json_records(root):
        path = Path(payload["_path"])
        records.append(add_export_stats(path, flatten_payload(payload)))
    if not records:
        return [], []
    fieldnames = [
        "dataset",
        "seed",
        "method",
        "variant",
        "test_mse",
        "ood_mse",
        "runtime_sec",
        "nan_steps",
        "tokens",
        "basis_terms",
        "exp_log_ops",
        "export_kind",
    ]
    write_csv(OUT_DIR / "review_mlp_large_runs.csv", records, fieldnames)
    summary = group_metric_summary(records, ["variant"])
    write_csv(
        OUT_DIR / "review_mlp_large_summary.csv",
        summary,
        [
            "variant",
            "n",
            "test_mse_mean",
            "ood_mse_mean",
            "runtime_sec_mean",
            "nan_steps_sum",
            "tokens_mean",
            "basis_terms_mean",
        ],
    )
    return records, summary


def build_noise_summary() -> tuple[list[dict], list[dict], list[dict]]:
    root = OUT_DIR / "noise_robustness"
    records = [flatten_payload(payload) for payload in read_json_records(root)]
    if not records:
        return [], [], []
    write_csv(
        OUT_DIR / "review_noise_robustness_runs.csv",
        records,
        [
            "dataset",
            "seed",
            "method",
            "noise_frac",
            "test_mse",
            "ood_mse",
            "val_mse",
            "runtime_sec",
            "nan_steps",
            "best_epoch",
        ],
    )
    summary = group_metric_summary(records, ["noise_frac", "method"])
    write_csv(
        OUT_DIR / "review_noise_robustness_summary.csv",
        summary,
        [
            "noise_frac",
            "method",
            "n",
            "test_mse_mean",
            "ood_mse_mean",
            "runtime_sec_mean",
            "nan_steps_sum",
        ],
    )
    paired_rows = build_noise_pairwise(records)
    return records, summary, paired_rows


def build_noise_pairwise(records: list[dict]) -> list[dict]:
    by_key = {
        (row["dataset"], int(row["seed"]), float(row["noise_frac"]), row["method"]): row
        for row in records
    }
    noise_levels = sorted({float(row["noise_frac"]) for row in records})
    paired_rows = []
    for noise_frac in noise_levels:
        pair_records = []
        for dataset, seed, level, method in sorted(by_key):
            if level != noise_frac or method != "mlp":
                continue
            mlp = by_key[(dataset, seed, level, "mlp")]
            eml = by_key.get((dataset, seed, level, "eml_kan"))
            if eml is None:
                continue
            pair_records.append(
                {
                    "mlp_test_mse": mlp["test_mse"],
                    "eml_kan_test_mse": eml["test_mse"],
                    "mlp_ood_mse": mlp["ood_mse"],
                    "eml_kan_ood_mse": eml["ood_mse"],
                }
            )
        test = make_pairwise_row(f"Noise {noise_frac:g} test", pair_records, "mlp_test_mse", "eml_kan_test_mse")
        ood = make_pairwise_row(f"Noise {noise_frac:g} OOD", pair_records, "mlp_ood_mse", "eml_kan_ood_mse")
        paired_rows.extend([test, ood])
    write_csv(
        OUT_DIR / "review_noise_pairwise.csv",
        paired_rows,
        [
            "comparison",
            "n",
            "eml_wins",
            "mlp_wins",
            "ties",
            "mean_eml_over_mlp",
            "geomean_eml_over_mlp",
            "sign_test_p",
        ],
    )
    return paired_rows


def build_pairwise_significance() -> list[dict]:
    specs = [
        (
            "Feynman clean12 test",
            RESULTS / "feynman_lowdim_clean12_pairwise_by_seed.csv",
            "mlp_test_mse",
            "eml_kan_test_mse",
        ),
        (
            "SRSD grouped test",
            RESULTS / "srsd_grouped" / "srsd_full_pairwise_by_seed.csv",
            "mlp_test_mse",
            "eml_kan_test_mse",
        ),
        (
            "Synthetic test",
            RESULTS / "synthetic_final" / "synthetic_final_pairwise_by_seed.csv",
            "mlp_test_mse",
            "eml_kan_test_mse",
        ),
        (
            "Synthetic OOD",
            RESULTS / "synthetic_final" / "synthetic_final_pairwise_by_seed.csv",
            "mlp_ood_mse",
            "eml_kan_ood_mse",
        ),
    ]
    rows = []
    for comparison, path, baseline_key, candidate_key in specs:
        if not path.exists():
            continue
        rows.append(make_pairwise_row(comparison, read_csv(path), baseline_key, candidate_key))
    write_csv(
        OUT_DIR / "review_pairwise_significance.csv",
        rows,
        [
            "comparison",
            "n",
            "eml_wins",
            "mlp_wins",
            "ties",
            "mean_eml_over_mlp",
            "geomean_eml_over_mlp",
            "sign_test_p",
        ],
    )
    return rows


def build_cost_summary() -> list[dict]:
    source = RESULTS / "fairness_audit_summary_20260509.csv"
    if not source.exists():
        return []
    keep = []
    for row in read_csv(source):
        keep.append(
            {
                "benchmark": row["benchmark"],
                "method": row["method_label"],
                "n": row["n"],
                "params_mean": as_float(row["params_mean"]),
                "params_vs_mlp": as_float(row["params_vs_mlp"]),
                "train_sec": as_float(row["runtime_sec_mean"]),
                "infer_ms_per_1k": as_float(row["latency_ms_per_1000_mean"]),
                "status": row["status"],
            }
        )
    write_csv(
        OUT_DIR / "review_cost_summary.csv",
        keep,
        [
            "benchmark",
            "method",
            "n",
            "params_mean",
            "params_vs_mlp",
            "train_sec",
            "infer_ms_per_1k",
            "status",
        ],
    )
    return keep


def optional_environment_status() -> dict:
    modules = ["pysr", "gplearn", "kan", "pykan", "aifeynman"]
    return {module: importlib.util.find_spec(module) is not None for module in modules}


def read_json_records(root: Path) -> list[dict]:
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_path"] = str(path)
        rows.append(payload)
    return rows


def write_report(
    pairwise_rows: list[dict],
    cost_rows: list[dict],
    mlp_large_summary: list[dict],
    noise_summary: list[dict],
    noise_pairwise: list[dict],
) -> None:
    env = optional_environment_status()
    lines = [
        "# Review Supplement Report",
        "",
        "## Existing Pairwise Evidence",
        "",
        "| Comparison | n | EML wins | MLP wins | Ties | Geomean EML/MLP | Sign-test p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pairwise_rows:
        lines.append(
            f"| {row['comparison']} | {row['n']} | {row['eml_wins']} | {row['mlp_wins']} | "
            f"{row['ties']} | {fmt_float(row['geomean_eml_over_mlp'])} | {fmt_float(row['sign_test_p'])} |"
        )
    lines += [
        "",
        "## Capacity And Runtime Evidence",
        "",
        "| Benchmark | Method | n | Params/MLP | Train sec | Infer ms/1k |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in cost_rows:
        lines.append(
            f"| {row['benchmark']} | {row['method']} | {row['n']} | "
            f"{fmt_float(row['params_vs_mlp'])} | {fmt_float(row['train_sec'])} | "
            f"{fmt_float(row['infer_ms_per_1k'])} |"
        )
    lines += [
        "",
        "## MLP-Large Supplement",
        "",
        "| Variant | n | Test MSE | OOD MSE | Tokens | Train sec | NaN steps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mlp_large_summary:
        lines.append(
            f"| {row['variant']} | {row['n']} | {fmt_float(row['test_mse_mean'])} | "
            f"{fmt_float(row['ood_mse_mean'])} | {fmt_float(row.get('tokens_mean', math.nan))} | "
            f"{fmt_float(row['runtime_sec_mean'])} | {row['nan_steps_sum']} |"
        )
    lines += [
        "",
        "## Noise Robustness Supplement",
        "",
        "| Noise | Method | n | Test MSE | OOD MSE | Train sec | NaN steps |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in noise_summary:
        lines.append(
            f"| {fmt_float(float(row['noise_frac']))} | {row['method']} | {row['n']} | "
            f"{fmt_float(row['test_mse_mean'])} | {fmt_float(row['ood_mse_mean'])} | "
            f"{fmt_float(row['runtime_sec_mean'])} | {row['nan_steps_sum']} |"
        )
    lines += [
        "",
        "### Noise Pairwise EML-KAN vs MLP",
        "",
        "| Comparison | n | EML wins | MLP wins | Geomean EML/MLP | Sign-test p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in noise_pairwise:
        lines.append(
            f"| {row['comparison']} | {row['n']} | {row['eml_wins']} | {row['mlp_wins']} | "
            f"{fmt_float(row['geomean_eml_over_mlp'])} | {fmt_float(row['sign_test_p'])} |"
        )
    lines += [
        "",
        "## Local Optional Baseline Availability",
        "",
        "| Module | Available |",
        "|---|---:|",
    ]
    for module, available in env.items():
        lines.append(f"| {module} | {'yes' if available else 'no'} |")
    (OUT_DIR / "review_supplement_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairwise_rows = build_pairwise_significance()
    cost_rows = build_cost_summary()
    _, mlp_large_summary = build_mlp_large_summary()
    _, noise_summary, noise_pairwise = build_noise_summary()
    write_report(pairwise_rows, cost_rows, mlp_large_summary, noise_summary, noise_pairwise)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
