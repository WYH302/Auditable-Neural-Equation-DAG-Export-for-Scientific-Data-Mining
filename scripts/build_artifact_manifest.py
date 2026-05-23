from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RESULTS = ROOT / "results_v2"
PAPER = PROJECT_ROOT / "paper" / "paper_cikm2026"


def manifest_rows() -> list[dict[str, str]]:
    py = r"D:\App1\environment\envs\ai_base\python.exe"
    prefix = r"$env:PYTHONPATH='<project>\code_experiments'; "
    return [
        {
            "name": "equation_export_audit",
            "command": prefix + py + r" code_experiments\scripts\run_equation_audit.py",
            "inputs": r"code_experiments\data\equation_native\*",
            "outputs": r"code_experiments\results_v2\equation_export\equation_metrics.csv; exported_formulas\*.txt",
            "paper_items": "Table 2; Figure 3; pruning diagnostics",
        },
        {
            "name": "export_fidelity_details",
            "command": prefix + py + r" code_experiments\scripts\build_export_fidelity_details.py",
            "inputs": r"equation_export\exported_formulas\*.txt; equation_native splits; checkpoints",
            "outputs": r"equation_export\export_fidelity_details*.csv; full_deployment_formulas\*.txt; table_export_fidelity_details.tex",
            "paper_items": "export fidelity detail table; full-deployment DAG audit",
        },
        {
            "name": "token_sensitivity",
            "command": prefix + py + r" code_experiments\scripts\build_token_sensitivity.py",
            "inputs": r"equation_export\exported_formulas\*.txt; full_deployment_formulas\*.txt",
            "outputs": r"equation_export\token_sensitivity*.csv; table_token_sensitivity.tex",
            "paper_items": "token sensitivity table",
        },
        {
            "name": "direct_audit_case_study",
            "command": prefix + py + r" code_experiments\scripts\build_audit_case_study.py",
            "inputs": r"equation_export\exported_formulas\mixed_exp_log_*.txt; equation_metrics.csv",
            "outputs": r"equation_export\audit_case_study_summary.csv; audit_case_study.md; table_audit_case_*.tex",
            "paper_items": "direct audit workflow table and raw DAG fragment table",
        },
        {
            "name": "review_breakdowns",
            "command": prefix + py + r" code_experiments\scripts\build_review_breakdowns.py",
            "inputs": r"Feynman/SRSD/synthetic pairwise CSVs",
            "outputs": r"review_supplement\review_pairwise_ci.csv; review_ood_failure_cases.csv; review_feynman_per_equation_edges.csv",
            "paper_items": "paired CI table; OOD failure cases; Feynman per-equation edge cases",
        },
        {
            "name": "ood_failure_curves",
            "command": prefix + py + r" code_experiments\scripts\build_ood_failure_plots.py",
            "inputs": r"synthetic_final checkpoints; data\synthetic train/test/OOD splits",
            "outputs": r"synthetic_final\figures\ood_failure_curves.png; synthetic_final\ood_failure_curve_metrics.csv; figure_ood_failure_curves.tex",
            "paper_items": "representative OOD failure curve figure",
        },
        {
            "name": "tau_gate_sensitivity",
            "command": prefix + py + r" code_experiments\scripts\build_tau_gate_sensitivity.py",
            "inputs": r"architecture_ablation_summary.csv; synthetic_surgery\ablation_table_20260509.csv",
            "outputs": r"review_supplement\review_tau_gate_sensitivity.csv; table_tau_gate_sensitivity.tex",
            "paper_items": "existing tau/gate sensitivity table",
        },
        {
            "name": "small_tau_sweep",
            "command": prefix + py + r" code_experiments\scripts\build_review_tau_sweep.py",
            "inputs": r"review_tau_sweep\tau*\synthetic\*.json; matching checkpoints",
            "outputs": r"review_supplement\review_small_tau_sweep_*.csv; table_small_tau_sweep.tex",
            "paper_items": "targeted seed-0 tau sanity table",
        },
        {
            "name": "symbolic_budget_summary",
            "command": prefix + py + r" code_experiments\scripts\build_symbolic_budget_summary.py",
            "inputs": r"external_symbolic_v3\external_symbolic_summary.csv; review_symbolic_budget_gplearn\external_symbolic_runs.csv",
            "outputs": r"review_supplement\review_symbolic_budget_summary.csv; table_symbolic_budget_check.tex",
            "paper_items": "external symbolic-search budget sanity table",
        },
        {
            "name": "paper_build",
            "command": r"cd paper\paper_cikm2026; latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex",
            "inputs": r"paper\paper_cikm2026\main.tex; table_*.tex; figures\*; references.bib",
            "outputs": r"paper\paper_cikm2026\build\main.pdf; build\main.log",
            "paper_items": "compiled manuscript",
        },
    ]


def render_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Artifact Manifest",
        "",
        "This manifest maps the revision analyses to their inputs, commands, outputs, and paper items. Commands assume they are run from the project root with the `ai_base` Python environment and `PYTHONPATH` pointing at `code_experiments`.",
        "",
        "## Reproduction Commands",
        "",
        "| Name | Command | Inputs | Outputs | Paper items |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['command']}` | `{row['inputs']}` | `{row['outputs']}` | {row['paper_items']} |"
        )
    lines += [
        "",
        "## Scope Notes",
        "",
        "- Exact checkpoint exporters are MLP, RBF-KAN, and EML-KAN; PySR/gplearn remain symbolic-search references.",
        "- StableEML is retained as a summarized neural reference and is not assigned exact scalar-graph fidelity.",
        "- The tau/gate table summarizes completed ablation logs; it is not an exhaustive hyperparameter sweep.",
        "- The small tau sweep is a targeted seed-0 sanity check on three one-dimensional OOD failure cases and does not override the completed 24-run ablation.",
        "- The symbolic budget table treats PySR/gplearn as symbolic-search references. The longer PySR attempt did not complete in the current Windows artifact run, so it is not reported as a result.",
        "- OOD curves are representative one-dimensional failure cases selected from the worst dataset-level OOD ratios.",
        "- AI-Feynman and official pykan are not included in the completed artifact because the current Windows environment lacks the required compiled/runtime setup.",
        "",
        "## Verification",
        "",
        "- Unit tests: `python -m unittest discover -s code_experiments\\tests`.",
        "- PDF build: `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex` from `paper\\paper_cikm2026`.",
        "- Log scan: search `build\\main.log` for `Fatal`, `Undefined`, `Overfull`, `LaTeX Warning`, `Package natbib Warning`, and `Missing $`.",
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["name", "command", "inputs", "outputs", "paper_items"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = manifest_rows()
    write_csv(RESULTS / "artifact_manifest.csv", rows)
    (RESULTS / "artifact_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (RESULTS / "ARTIFACT_MANIFEST.md").write_text(render_markdown(rows), encoding="utf-8")
    print(f"artifact manifest rows={len(rows)}")


if __name__ == "__main__":
    main()
