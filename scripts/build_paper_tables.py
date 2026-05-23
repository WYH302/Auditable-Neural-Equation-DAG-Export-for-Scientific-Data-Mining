from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2"
PAPER = ROOT / "paper_elscas"


METHOD_LABELS = {
    "mlp": "MLP",
    "stable_eml": "StableEML",
    "kan": "RBF-KAN",
    "eml_kan": "EML-KAN",
}

EXPECTED = {
    ("Feynman clean12", "test", "mlp"): 36,
    ("Feynman clean12", "test", "eml_kan"): 36,
    ("Feynman clean12", "test", "stable_eml"): 36,
    ("Feynman clean12", "test", "kan"): 36,
    ("SRSD full", "test", "mlp"): 3,
    ("SRSD full", "test", "eml_kan"): 3,
    ("SRSD full", "test", "stable_eml"): 3,
    ("SRSD full", "test", "kan"): 3,
    ("Synthetic final", "test", "mlp"): 30,
    ("Synthetic final", "test", "eml_kan"): 30,
    ("Synthetic final", "test", "stable_eml"): 30,
    ("Synthetic final", "test", "kan"): 30,
    ("Synthetic final", "ood", "mlp"): 30,
    ("Synthetic final", "ood", "eml_kan"): 30,
    ("Synthetic final", "ood", "stable_eml"): 30,
    ("Synthetic final", "ood", "kan"): 30,
}


def as_float(value: object) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def as_int(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


def read_csv_records(path: Path, benchmark: str, source: str) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            dataset = row["dataset"]
            seed = as_int(row["seed"])
            nan_steps = as_int(row.get("nan_steps", 0))
            checkpoint_exists = boolish(row.get("checkpoint_exists", False))
            for metric in ("test", "ood"):
                mse_key = f"{metric}_mse"
                r2_key = f"{metric}_r2"
                if mse_key not in row:
                    continue
                records.append(
                    {
                        "benchmark": benchmark,
                        "metric": metric,
                        "dataset": dataset,
                        "seed": seed,
                        "model": model,
                        "mse": as_float(row[mse_key]),
                        "r2": as_float(row.get(r2_key)),
                        "nan_steps": nan_steps,
                        "checkpoint_exists": checkpoint_exists,
                        "source": source,
                    }
                )
    return records


def read_json_records(glob_root: Path, benchmark: str, source: str) -> list[dict]:
    records: list[dict] = []
    if not glob_root.exists():
        return records
    for path in sorted(glob_root.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        model = payload["model"]
        dataset = payload["dataset"]
        seed = as_int(payload["seed"])
        metrics = payload.get("metrics", {})
        nan_steps = as_int(metrics.get("nan_steps", 0))
        checkpoint_exists = path.with_suffix(".pt").exists()
        for metric in ("test", "ood"):
            split = metrics.get(metric)
            if not isinstance(split, dict) or "mse" not in split:
                continue
            records.append(
                {
                    "benchmark": benchmark,
                    "metric": metric,
                    "dataset": dataset,
                    "seed": seed,
                    "model": model,
                    "mse": as_float(split.get("mse")),
                    "r2": as_float(split.get("r2")),
                    "nan_steps": nan_steps,
                    "checkpoint_exists": checkpoint_exists,
                    "source": source,
                }
            )
    return records


def finite_values(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def fmt_value(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value < 1e-3 or abs_value >= 1e4:
        mantissa, exp = f"{value:.2e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exp)}}}"
    if abs_value < 1:
        return f"{value:.5f}"
    return f"{value:.2f}"


def fmt_fixed(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.5f}"


def collect_records() -> list[dict]:
    records: list[dict] = []
    records += read_csv_records(
        RESULTS / "feynman_lowdim_clean12_summary.csv",
        "Feynman clean12",
        "formal_mlp_eml",
    )
    records += read_csv_records(
        RESULTS / "srsd_grouped" / "srsd_full_summary.csv",
        "SRSD full",
        "formal_mlp_eml",
    )
    records += read_csv_records(
        RESULTS / "synthetic_final" / "synthetic_final_summary.csv",
        "Synthetic final",
        "formal_mlp_eml",
    )
    records += read_json_records(
        RESULTS / "baseline_supplement" / "feynman_lowdim_clean12" / "runs",
        "Feynman clean12",
        "baseline_supplement",
    )
    records += read_json_records(
        RESULTS / "baseline_supplement" / "srsd_grouped" / "srsd",
        "SRSD full",
        "baseline_supplement",
    )
    records += read_json_records(
        RESULTS / "baseline_supplement" / "synthetic" / "synthetic",
        "Synthetic final",
        "baseline_supplement",
    )
    records += read_json_records(
        RESULTS / "baseline_matched" / "feynman_stable_eml_w14" / "runs",
        "Feynman clean12",
        "matched_budget",
    )
    records += read_json_records(
        RESULTS / "baseline_matched" / "srsd_stable_eml_w27" / "srsd",
        "SRSD full",
        "matched_budget",
    )
    records += read_json_records(
        RESULTS / "eml_rerun_strong_srsd_w80" / "srsd",
        "SRSD full",
        "eml_strong_rerun",
    )
    records += read_json_records(
        RESULTS / "baseline_matched" / "synthetic_stable_eml_w8" / "synthetic",
        "Synthetic final",
        "matched_budget",
    )
    matched_benchmarks = {
        r["benchmark"]
        for r in records
        if r["model"] == "stable_eml" and r["source"] == "matched_budget"
    }
    if matched_benchmarks:
        records = [
            r
            for r in records
            if not (
                r["model"] == "stable_eml"
                and r["benchmark"] in matched_benchmarks
                and r["source"] == "baseline_supplement"
            )
        ]
    strong_eml_benchmarks = {
        r["benchmark"]
        for r in records
        if r["model"] == "eml_kan" and r["source"] == "eml_strong_rerun"
    }
    if strong_eml_benchmarks:
        records = [
            r
            for r in records
            if not (
                r["model"] == "eml_kan"
                and r["benchmark"] in strong_eml_benchmarks
                and r["source"] == "formal_mlp_eml"
            )
        ]
    return records


def summarize(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["benchmark"], record["metric"], record["model"])].append(record)

    mlp_lookup = {
        (r["benchmark"], r["metric"], r["dataset"], r["seed"]): r["mse"]
        for r in records
        if r["model"] == "mlp"
    }

    rows: list[dict] = []
    for key in sorted(grouped):
        benchmark, metric, model = key
        group = grouped[key]
        mses = finite_values([r["mse"] for r in group])
        r2s = finite_values([r["r2"] for r in group])
        win_total = 0
        win_count = 0
        for r in group:
            mlp_mse = mlp_lookup.get((benchmark, metric, r["dataset"], r["seed"]))
            if mlp_mse is None or not math.isfinite(mlp_mse) or model == "mlp":
                continue
            win_total += 1
            if r["mse"] < mlp_mse:
                win_count += 1
        expected = EXPECTED.get((benchmark, metric, model), len(group))
        checkpoint_count = sum(1 for r in group if r["checkpoint_exists"])
        nan_steps_sum = sum(as_int(r["nan_steps"]) for r in group)
        rows.append(
            {
                "benchmark": benchmark,
                "metric": metric,
                "method": model,
                "method_label": METHOD_LABELS.get(model, model),
                "n": len(group),
                "expected_n": expected,
                "complete": len(group) == expected,
                "mse_mean": mean(mses) if mses else math.nan,
                "mse_std": stdev(mses) if len(mses) > 1 else 0.0,
                "r2_mean": mean(r2s) if r2s else math.nan,
                "r2_std": stdev(r2s) if len(r2s) > 1 else 0.0,
                "win_vs_mlp_count": win_count if model != "mlp" else "",
                "win_vs_mlp_total": win_total if model != "mlp" else "",
                "win_rate": (win_count / win_total) if win_total else (1.0 if model == "mlp" else math.nan),
                "nan_steps_sum": nan_steps_sum,
                "checkpoint_count": checkpoint_count,
                "source": "+".join(sorted({r["source"] for r in group})),
                "status": "complete" if len(group) == expected and checkpoint_count == len(group) and nan_steps_sum == 0 else "check",
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "metric",
        "method",
        "method_label",
        "n",
        "expected_n",
        "complete",
        "mse_mean",
        "mse_std",
        "r2_mean",
        "r2_std",
        "win_vs_mlp_count",
        "win_vs_mlp_total",
        "win_rate",
        "nan_steps_sum",
        "checkpoint_count",
        "source",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(rows: list[dict], path: Path) -> None:
    order = [
        ("Feynman clean12", "test"),
        ("SRSD full", "test"),
        ("Synthetic final", "test"),
        ("Synthetic final", "ood"),
    ]
    methods = ["mlp", "stable_eml", "kan", "eml_kan"]
    lookup = {(r["benchmark"], r["metric"], r["method"]): r for r in rows}
    ranks = {}
    for benchmark, metric in order:
        ranked = sorted(
            [
                r
                for r in rows
                if r["benchmark"] == benchmark
                and r["metric"] == metric
                and math.isfinite(r["mse_mean"])
            ],
            key=lambda r: r["mse_mean"],
        )
        for idx, row in enumerate(ranked):
            ranks[(benchmark, metric, row["method"])] = idx + 1
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Unified paper-level comparison. Means and standard deviations are computed over matched dataset--seed runs. Win counts are pairwise against MLP on the same dataset and seed. Bold marks the best mean MSE and underline marks the second-best method within each benchmark split.}",
        r"\label{tab:main-results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Benchmark & Split & Method & $n$ & MSE mean$\pm$std & $R^2$ mean & Wins vs MLP \\",
        r"\midrule",
    ]
    for benchmark, metric in order:
        for model in methods:
            row = lookup.get((benchmark, metric, model))
            if row is None:
                continue
            wins = "baseline" if model == "mlp" else f"{row['win_vs_mlp_count']}/{row['win_vs_mlp_total']}"
            n = f"{row['n']}/{row['expected_n']}"
            method_label = row["method_label"]
            rank = ranks.get((benchmark, metric, model))
            if rank == 1:
                method_label = rf"\best{{{method_label}}}"
            elif rank == 2:
                method_label = rf"\underline{{{method_label}}}"
            mse_text = f"${fmt_value(row['mse_mean'])}\\pm{fmt_value(row['mse_std'])}$"
            if rank == 1:
                mse_text = rf"\best{{{mse_text}}}"
            lines.append(
                f"{benchmark} & {metric} & {method_label} & {n} & "
                f"{mse_text} & "
                f"{fmt_fixed(row['r2_mean'])} & {wins} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(rows: list[dict], path: Path) -> None:
    incomplete = [r for r in rows if r["status"] != "complete"]
    lines = [
        "# Paper Table Build Report",
        "",
        f"- Total summary rows: {len(rows)}",
        f"- Incomplete/check rows: {len(incomplete)}",
        "",
        "## Completeness",
        "",
        "| Benchmark | Split | Method | n/expected | ckpt | nan_steps | status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['benchmark']} | {r['metric']} | {r['method_label']} | "
            f"{r['n']}/{r['expected_n']} | {r['checkpoint_count']} | "
            f"{r['nan_steps_sum']} | {r['status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pairwise_report(records: list[dict], path: Path) -> None:
    pairs = [
        ("eml_kan", "mlp"),
        ("eml_kan", "stable_eml"),
        ("eml_kan", "kan"),
        ("kan", "mlp"),
        ("stable_eml", "mlp"),
    ]
    grouped: dict[tuple[str, str, str, int], dict[str, dict]] = defaultdict(dict)
    for record in records:
        key = (
            record["benchmark"],
            record["metric"],
            record["dataset"],
            record["seed"],
        )
        grouped[key][record["model"]] = record

    pair_rows = []
    for benchmark, metric in sorted({(r["benchmark"], r["metric"]) for r in records}):
        for left, right in pairs:
            total = 0
            left_wins = 0
            ratios = []
            for key, by_model in grouped.items():
                if key[0] != benchmark or key[1] != metric:
                    continue
                if left not in by_model or right not in by_model:
                    continue
                lmse = by_model[left]["mse"]
                rmse = by_model[right]["mse"]
                if not (math.isfinite(lmse) and math.isfinite(rmse)):
                    continue
                total += 1
                left_wins += int(lmse < rmse)
                if rmse > 0:
                    ratios.append(lmse / rmse)
            if total:
                pair_rows.append(
                    {
                        "benchmark": benchmark,
                        "metric": metric,
                        "left": METHOD_LABELS.get(left, left),
                        "right": METHOD_LABELS.get(right, right),
                        "wins": left_wins,
                        "total": total,
                        "mean_mse_ratio": mean(ratios) if ratios else math.nan,
                        "median_mse_ratio": median(ratios) if ratios else math.nan,
                    }
                )

    lines = [
        "# Pairwise Baseline Report",
        "",
        "Lower MSE wins on matched dataset--seed pairs. Ratio is left MSE divided by right MSE; values below 1 favor the left method.",
        "",
        "| Benchmark | Split | Pair | Wins | Median MSE ratio | Mean MSE ratio |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in pair_rows:
        mean_ratio = f"{row['mean_mse_ratio']:.4g}" if math.isfinite(row["mean_mse_ratio"]) else "--"
        median_ratio = f"{row['median_mse_ratio']:.4g}" if math.isfinite(row["median_mse_ratio"]) else "--"
        lines.append(
            f"| {row['benchmark']} | {row['metric']} | "
            f"{row['left']} vs {row['right']} | {row['wins']}/{row['total']} | "
            f"{median_ratio} | {mean_ratio} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records = collect_records()
    rows = summarize(records)
    write_csv(rows, RESULTS / "paper_unified_table_20260509.csv")
    write_latex_table(rows, PAPER / "table_main_results.tex")
    write_report(rows, RESULTS / "baseline_supplement" / "baseline_supplement_report.md")
    write_pairwise_report(records, RESULTS / "baseline_supplement" / "baseline_pairwise_report.md")
    print(f"records={len(records)} summary_rows={len(rows)}")


if __name__ == "__main__":
    main()
