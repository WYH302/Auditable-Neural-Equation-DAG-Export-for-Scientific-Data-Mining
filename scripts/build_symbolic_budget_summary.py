from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RESULTS = ROOT / "results_v2"
SUPPLEMENT = RESULTS / "review_supplement"
PAPER = PROJECT_ROOT / "paper" / "paper_cikm2026"


def parse_float(value) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def summarize_runs(rows: list[dict]) -> dict:
    tests = finite([parse_float(row.get("test_mse")) for row in rows])
    oods = finite([parse_float(row.get("ood_mse")) for row in rows])
    complexities = finite([parse_float(row.get("complexity")) for row in rows])
    runtimes = finite([parse_float(row.get("runtime_sec")) for row in rows])
    return {
        "n": len(rows),
        "ok": sum(1 for row in rows if str(row.get("status", "")).lower() == "ok"),
        "exact_count": sum(int(parse_float(row.get("exact_recovery")) == 1) for row in rows),
        "test_mse_mean": mean(tests) if tests else math.nan,
        "ood_mse_mean": mean(oods) if oods else math.nan,
        "complexity_mean": mean(complexities) if complexities else math.nan,
        "runtime_sec_mean": mean(runtimes) if runtimes else math.nan,
    }


def budget_label(generations: int, population_size: int) -> str:
    return f"{generations} gen, {population_size} pop"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def existing_reference_rows() -> list[dict]:
    path = RESULTS / "external_symbolic_v3" / "external_symbolic_summary.csv"
    rows = read_csv(path)
    out = []
    for row in rows:
        if row.get("benchmark") != "Synthetic symbolic":
            continue
        method = row.get("method", "")
        if method not in {"pysr", "gplearn"}:
            continue
        out.append(
            {
                "method": "PySR" if method == "pysr" else "gplearn",
                "budget": "existing reference",
                "n": int(parse_float(row.get("n"))),
                "ok": int(parse_float(row.get("ok"))),
                "exact_count": int(parse_float(row.get("exact_count"))),
                "test_mse_mean": parse_float(row.get("test_mse_mean")),
                "ood_mse_mean": parse_float(row.get("ood_mse_mean")),
                "complexity_mean": parse_float(row.get("complexity_mean")),
                "runtime_sec_mean": parse_float(row.get("runtime_sec_mean")),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return out


def longer_gplearn_row() -> list[dict]:
    path = RESULTS / "review_symbolic_budget_gplearn" / "external_symbolic_runs.csv"
    rows = read_csv(path)
    if not rows:
        return []
    summary = summarize_runs(rows)
    return [
        {
            "method": "gplearn",
            "budget": budget_label(generations=20, population_size=1000),
            **summary,
            "source": str(path.relative_to(ROOT)),
        }
    ]


def collect_rows() -> list[dict]:
    return existing_reference_rows() + longer_gplearn_row()


def fmt(value: float) -> str:
    value = parse_float(value)
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) >= 1 or abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def fmt_runtime(value: float) -> str:
    value = parse_float(value)
    if not math.isfinite(value):
        return "--"
    return f"{value:.2f}"


def render_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{External symbolic-search budget check on the ten synthetic formulas. These rows are references only: they search for formulas from data and do not export trained neural checkpoints. The longer-budget PySR attempt did not complete in this Windows run, so only the completed gplearn longer-budget row is added.}",
        r"\label{tab:symbolic-budget-check}",
        r"\resulttablesetup",
        r"\begin{tabularx}{\columnwidth}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Method & Budget & OK & Exact & Test & OOD & Runtime \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & {row['budget']} & {row['ok']}/{row['n']} & {row['exact_count']}/{row['n']} & "
            f"${fmt(row['test_mse_mean'])}$ & ${fmt(row['ood_mse_mean'])}$ & {fmt_runtime(row['runtime_sec_mean'])} \\\\"
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
    write_csv(SUPPLEMENT / "review_symbolic_budget_summary.csv", rows)
    (PAPER / "table_symbolic_budget_check.tex").write_text(render_latex(rows), encoding="utf-8")
    print(f"symbolic budget rows={len(rows)}")


if __name__ == "__main__":
    main()
