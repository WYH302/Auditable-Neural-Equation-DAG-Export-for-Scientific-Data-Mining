from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from scripts import build_token_sensitivity as sensitivity


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RESULTS = ROOT / "results_v2" / "equation_export"
PAPER = PROJECT_ROOT / "paper" / "paper_cikm2026"

CASE_DATASET = "mixed_exp_log"
CASE_METHODS = [
    ("MLP", "mlp"),
    ("RBF-KAN", "kan"),
    ("EML-KAN", "eml_kan"),
]


def parse_active_notes(notes: str) -> dict[str, int]:
    parsed = {"active_edges": 0, "active_terms": 0}
    for key in parsed:
        match = re.search(rf"{key}=([0-9]+)", notes or "")
        if match:
            parsed[key] = int(match.group(1))
    return parsed


def first_assignments(text: str, n: int = 2, include_final: bool = True) -> list[str]:
    statements = sensitivity.split_assignments(text)
    snippet = statements[:n]
    if include_final:
        finals = [statement for statement in statements if statement.strip().startswith("y =")]
        if finals and finals[-1] not in snippet:
            snippet.append(finals[-1])
    return snippet


def load_metric_rows() -> dict[tuple[str, str], dict[str, str]]:
    path = RESULTS / "equation_metrics.csv"
    out: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out[(row["dataset"], row["method"])] = row
    return out


def finite_float(value: str | float | int | None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def summarize_case(dataset: str = CASE_DATASET) -> list[dict]:
    metrics = load_metric_rows()
    rows: list[dict] = []
    for method_label, method_key in CASE_METHODS:
        core_path = RESULTS / "exported_formulas" / f"{dataset}_{method_key}_seed0.txt"
        full_path = RESULTS / "full_deployment_formulas" / f"{dataset}_{method_key}_seed0.txt"
        text = core_path.read_text(encoding="utf-8")
        full_text = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
        canonical, max_rhs_depth, assignments, ok, canonical_ms = sensitivity.canonicalize_local(text)
        cse_text, cse_ok = sensitivity.global_cse_text(text)
        metric = metrics[(dataset, method_key)]
        active = parse_active_notes(metric.get("notes", ""))
        row = {
            "dataset": dataset,
            "method": method_key,
            "method_label": method_label,
            "raw_tokens": sensitivity.token_count(text),
            "local_tokens": sensitivity.token_count(canonical) if ok else math.nan,
            "global_cse_tokens": sensitivity.token_count(cse_text) if cse_ok else math.nan,
            "full_tokens": sensitivity.token_count(full_text) if full_text else math.nan,
            "assignments": assignments,
            "max_rhs_depth": max_rhs_depth,
            "canonical_ms": canonical_ms,
            "active_edges": active["active_edges"],
            "active_terms": active["active_terms"],
            "basis_terms": int(float(metric.get("basis_terms", 0) or 0)),
            "exp_log_ops": int(float(metric.get("exp_log_ops", 0) or 0)),
            "test_mse": finite_float(metric.get("test_mse")),
            "ood_mse": finite_float(metric.get("ood_mse")),
            "target_expression": metric.get("target_expression", ""),
            "core_file": str(core_path.relative_to(ROOT)),
            "full_file": str(full_path.relative_to(ROOT)) if full_path.exists() else "",
            "snippet": "\n".join(first_assignments(text)),
        }
        rows.append(row)
    return rows


def fmt(value: float | int) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "--"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def fmt_int(value: float | int) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "--"
    return f"{value:.0f}"


def render_latex_table(dataset: str, rows: list[dict]) -> str:
    dataset_tex = rf"\texttt{{\detokenize{{{dataset}}}}}"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Direct audit case study on {dataset_tex}. The workflow starts from the raw assignment-preserving checkpoint DAG, applies the same local canonicalization pass used in Table~\ref{{tab:export}}, runs an exploratory global CSE pass for sensitivity, and records full-deployment tokens, assignment count, maximum canonical RHS depth, canonicalization time, active gates/terms, and basis expansion. Canonicalization time is one CPU Python/SymPy pass measured with \texttt{{perf\_counter}} and excludes file I/O; active terms use exporter gate threshold 0.0, i.e., retained non-pruned gate terms in the faithful export.}}",
        r"\label{tab:audit-case-study}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Method & Raw & Local & Global CSE & Full & Assign. & Max depth & Canon. ms & Active & Basis \\",
        r"\midrule",
    ]
    for row in rows:
        active = f"{row['active_edges']}/{row['active_terms']}"
        cells = [
            row["method_label"],
            fmt(row["raw_tokens"]),
            fmt(row["local_tokens"]),
            fmt(row["global_cse_tokens"]),
            fmt(row["full_tokens"]),
            fmt_int(row["assignments"]),
            fmt_int(row["max_rhs_depth"]),
            fmt(row["canonical_ms"]),
            active,
            fmt_int(row["basis_terms"]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def short_fragment(snippet: str, max_chars: int = 120) -> str:
    first = snippet.splitlines()[0] if snippet else ""
    if len(first) <= max_chars:
        return first
    return first[: max_chars - 4].rstrip() + " ..."


def render_fragment_table(dataset: str, rows: list[dict]) -> str:
    dataset_tex = rf"\texttt{{\detokenize{{{dataset}}}}}"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Representative raw DAG fragments from the {dataset_tex} audit case. Display snippets are truncated for readability; token counts are computed on the full exported checkpoint DAG files, and full snippets and file paths are included in the artifact case-study Markdown.}}",
        r"\label{tab:audit-case-fragments}",
        r"\begin{tabularx}{\textwidth}{@{}lX@{}}",
        r"\toprule",
        r"Method & Raw exported fragment \\",
        r"\midrule",
    ]
    for row in rows:
        fragment = short_fragment(row["snippet"])
        lines.append(rf"{row['method_label']} & \texttt{{\detokenize{{{fragment}}}}} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def render_markdown(dataset: str, rows: list[dict]) -> str:
    target = rows[0]["target_expression"] if rows else ""
    lines = [
        f"# Direct Audit Case Study: {dataset}",
        "",
        f"Target: `{target}`",
        "",
        "Workflow: raw assignment DAG -> local canonicalization -> exploratory global CSE -> full-deployment DAG check.",
        "",
        "| Method | Raw | Local | Global CSE | Full | Assignments | Max depth | Canon. ms | Active edges/terms | Basis |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_label']} | {fmt(row['raw_tokens'])} | {fmt(row['local_tokens'])} | "
            f"{fmt(row['global_cse_tokens'])} | {fmt(row['full_tokens'])} | {fmt_int(row['assignments'])} | "
            f"{fmt_int(row['max_rhs_depth'])} | {fmt(row['canonical_ms'])} | "
            f"{row['active_edges']}/{row['active_terms']} | {fmt_int(row['basis_terms'])} |"
        )
    lines += ["", "## Snippets", ""]
    for row in rows:
        lines += [
            f"### {row['method_label']}",
            "",
            f"Core file: `{row['core_file']}`",
            "",
            "```text",
            row["snippet"],
            "```",
            "",
        ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "dataset",
        "method",
        "method_label",
        "raw_tokens",
        "local_tokens",
        "global_cse_tokens",
        "full_tokens",
        "assignments",
        "max_rhs_depth",
        "canonical_ms",
        "active_edges",
        "active_terms",
        "basis_terms",
        "exp_log_ops",
        "test_mse",
        "ood_mse",
        "target_expression",
        "core_file",
        "full_file",
        "snippet",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def main() -> None:
    rows = summarize_case(CASE_DATASET)
    write_csv(RESULTS / "audit_case_study_summary.csv", rows)
    (RESULTS / "audit_case_study.md").write_text(render_markdown(CASE_DATASET, rows), encoding="utf-8")
    (PAPER / "table_audit_case_study.tex").write_text(render_latex_table(CASE_DATASET, rows), encoding="utf-8")
    (PAPER / "table_audit_case_fragments.tex").write_text(render_fragment_table(CASE_DATASET, rows), encoding="utf-8")
    print(f"audit case study dataset={CASE_DATASET} rows={len(rows)}")


if __name__ == "__main__":
    main()
