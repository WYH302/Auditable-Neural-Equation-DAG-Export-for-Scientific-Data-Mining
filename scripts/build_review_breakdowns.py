from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RESULTS = ROOT / "results_v2"
SUPPLEMENT = RESULTS / "review_supplement"
PAPER = PROJECT_ROOT / "paper" / "paper_cikm2026"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sign_test_p(eml_wins: int, mlp_wins: int) -> float:
    n = eml_wins + mlp_wins
    if n == 0:
        return math.nan
    smaller = min(eml_wins, mlp_wins)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def bootstrap_ci(values: list[float], bootstrap_reps: int = 10_000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        value = math.exp(values[0])
        return value, value
    rng = random.Random(seed)
    estimates: list[float] = []
    n = len(values)
    for _ in range(bootstrap_reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(math.exp(mean(sample)))
    estimates.sort()
    lo = estimates[int(0.025 * (len(estimates) - 1))]
    hi = estimates[int(0.975 * (len(estimates) - 1))]
    return lo, hi


def summarize_ratios(label: str, ratios: list[float], bootstrap_reps: int = 10_000) -> dict:
    clean = [ratio for ratio in ratios if math.isfinite(ratio) and ratio > 0]
    logs = [math.log(ratio) for ratio in clean]
    eml_wins = sum(1 for ratio in clean if ratio < 1.0)
    mlp_wins = sum(1 for ratio in clean if ratio > 1.0)
    ties = len(clean) - eml_wins - mlp_wins
    ci_lo, ci_hi = bootstrap_ci(logs, bootstrap_reps=bootstrap_reps, seed=17)
    return {
        "comparison": label,
        "n": len(clean),
        "eml_wins": eml_wins,
        "mlp_wins": mlp_wins,
        "ties": ties,
        "geomean_eml_over_mlp": math.exp(mean(logs)) if logs else math.nan,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "sign_test_p": sign_test_p(eml_wins, mlp_wins),
    }


def float_col(rows: list[dict[str, str]], column: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def build_pairwise_ci() -> list[dict]:
    feynman = read_csv(RESULTS / "feynman_lowdim_clean12_pairwise_by_seed.csv")
    srsd = read_csv(RESULTS / "srsd_grouped" / "srsd_full_pairwise_by_seed.csv")
    synthetic = read_csv(RESULTS / "synthetic_final" / "synthetic_final_pairwise_by_seed.csv")
    return [
        summarize_ratios("Feynman clean12 test", float_col(feynman, "eml_over_mlp_mse")),
        summarize_ratios("SRSD grouped test", float_col(srsd, "eml_over_mlp_mse")),
        summarize_ratios("Synthetic test", float_col(synthetic, "eml_over_mlp_test_mse")),
        summarize_ratios("Synthetic OOD", float_col(synthetic, "eml_over_mlp_ood_mse")),
    ]


def select_worst_cases(rows: list[dict], n: int = 4) -> list[dict]:
    return sorted(rows, key=lambda row: float(row["ratio"]), reverse=True)[:n]


def build_ood_failure_cases() -> list[dict]:
    rows = read_csv(RESULTS / "synthetic_final" / "synthetic_final_pairwise_by_dataset.csv")
    normalized: list[dict] = []
    for row in rows:
        normalized.append(
            {
                "dataset": row["dataset"],
                "test_ratio": float(row["eml_over_mlp_test_mse_mean"]),
                "ratio": float(row["eml_over_mlp_ood_mse_mean"]),
                "mlp_ood": float(row["mlp_ood_mse_mean"]),
                "eml_ood": float(row["eml_kan_ood_mse_mean"]),
                "eml_ood_wins": int(row["eml_kan_ood_wins"]),
                "mlp_ood_wins": int(row["mlp_ood_wins"]),
            }
        )
    return select_worst_cases(normalized, n=4)


def build_feynman_edge_cases() -> list[dict]:
    rows = read_csv(RESULTS / "feynman_lowdim_clean12_pairwise_by_dataset.csv")
    normalized: list[dict] = []
    for row in rows:
        ratio = float(row["eml_over_mlp_mse_mean"])
        normalized.append(
            {
                "dataset": row["dataset"],
                "input_dim": int(row["input_dim"]),
                "eml_wins": int(row["eml_kan_wins"]),
                "mlp_wins": int(row["mlp_wins"]),
                "ratio": ratio,
            }
        )
    worst = sorted(normalized, key=lambda row: row["ratio"], reverse=True)[:3]
    best = sorted(normalized, key=lambda row: row["ratio"])[:3]
    for row in worst:
        row["group"] = "High ratio"
    for row in best:
        row["group"] = "Low ratio"
    return worst + best


def fmt(value: float) -> str:
    if not math.isfinite(float(value)):
        return "--"
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    if abs(value) >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2e}"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def render_pairwise_ci(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired seed-level diagnostics for EML-KAN versus MLP. Ratios below one favor EML-KAN. Confidence intervals are descriptive bootstrap intervals for the geometric mean ratio, and sign-test p-values are uncorrected descriptive checks. SRSD is grouped aggregate only, not a per-equation claim.}",
        r"\label{tab:pairwise-ci}",
        r"\resulttablesetup",
        r"\begin{tabularx}{\columnwidth}{@{}Xrrrr@{}}",
        r"\toprule",
        r"Comparison & Wins & Geo. ratio & 95\% CI & $p$ \\",
        r"\midrule",
    ]
    for row in rows:
        wins = f"{row['eml_wins']}/{row['mlp_wins']}"
        ci = f"{fmt(row['ci_lo'])}--{fmt(row['ci_hi'])}"
        lines.append(
            f"{row['comparison']} & {wins} & {fmt(row['geomean_eml_over_mlp'])} & {ci} & {fmt(row['sign_test_p'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def render_ood_failures(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Worst synthetic OOD cases for EML-KAN versus MLP by dataset-level mean ratio. Ratios above one mean the EML-KAN OOD MSE is larger than the MLP OOD MSE.}",
        r"\label{tab:ood-failure-cases}",
        r"\resulttablesetup",
        r"\begin{tabularx}{\columnwidth}{@{}Xrrrr@{}}",
        r"\toprule",
        r"Dataset & Test ratio & OOD ratio & MLP OOD & EML OOD \\",
        r"\midrule",
    ]
    for row in rows:
        dataset = row["dataset"].replace("_", r"\_")
        lines.append(
            f"{dataset} & {fmt(row['test_ratio'])} & {fmt(row['ratio'])} & "
            f"{fmt(row['mlp_ood'])} & {fmt(row['eml_ood'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def render_feynman_edges(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Feynman clean12 per-equation edge cases. Each row aggregates three seeds for one equation ID and reports the mean EML-KAN/MLP test-MSE ratio; ratios below one favor EML-KAN.}",
        r"\label{tab:feynman-per-equation-edges}",
        r"\resulttablesetup",
        r"\begin{tabularx}{\columnwidth}{@{}Xlrrr@{}}",
        r"\toprule",
        r"Case & ID & Dim. & Wins & Ratio \\",
        r"\midrule",
    ]
    for row in rows:
        wins = f"{row['eml_wins']}/{row['mlp_wins']}"
        lines.append(
            f"{row['group']} & {row['dataset']} & {row['input_dim']} & {wins} & {fmt(row['ratio'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    pairwise = build_pairwise_ci()
    ood = build_ood_failure_cases()
    feynman_edges = build_feynman_edge_cases()
    write_csv(
        SUPPLEMENT / "review_pairwise_ci.csv",
        pairwise,
        ["comparison", "n", "eml_wins", "mlp_wins", "ties", "geomean_eml_over_mlp", "ci_lo", "ci_hi", "sign_test_p"],
    )
    write_csv(
        SUPPLEMENT / "review_ood_failure_cases.csv",
        ood,
        ["dataset", "test_ratio", "ratio", "mlp_ood", "eml_ood", "eml_ood_wins", "mlp_ood_wins"],
    )
    write_csv(
        SUPPLEMENT / "review_feynman_per_equation_edges.csv",
        feynman_edges,
        ["group", "dataset", "input_dim", "eml_wins", "mlp_wins", "ratio"],
    )
    (PAPER / "table_pairwise_ci.tex").write_text(render_pairwise_ci(pairwise), encoding="utf-8")
    (PAPER / "table_ood_failure_cases.tex").write_text(render_ood_failures(ood), encoding="utf-8")
    (PAPER / "table_feynman_per_equation_edges.tex").write_text(render_feynman_edges(feynman_edges), encoding="utf-8")
    print(f"review breakdowns pairwise={len(pairwise)} ood={len(ood)} feynman_edges={len(feynman_edges)}")


if __name__ == "__main__":
    main()
