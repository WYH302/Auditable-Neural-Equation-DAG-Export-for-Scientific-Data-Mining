from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2"
PAPER = ROOT / "paper_elscas"
sys.path.append(str(ROOT / "src"))
from symbolic_export.export_models import load_model_from_json  # noqa: E402
from symbolic_export.graph_export import export_graph  # noqa: E402

METHOD_LABELS = {
    "mlp": "MLP",
    "stable_eml": "StableEML",
    "kan": "RBF-KAN",
    "eml_kan": "EML-KAN",
}
METHOD_ORDER = ["MLP", "RBF-KAN", "StableEML", "EML-KAN"]

RUN_DIRS = [
    ("formal", RESULTS / "feynman_lowdim_clean12" / "runs"),
    ("baseline_supplement", RESULTS / "baseline_supplement" / "feynman_lowdim_clean12" / "runs"),
    ("matched_stable", RESULTS / "baseline_matched" / "feynman_stable_eml_w14" / "runs"),
]


def collect_paths() -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for source, root in RUN_DIRS:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.json")):
            paths.append((source, path))
    if any(source == "matched_stable" for source, _ in paths):
        paths = [
            (source, path)
            for source, path in paths
            if not ("stable_eml" in path.name and source == "baseline_supplement")
        ]
    return paths


def model_input_dim(payload: dict) -> int:
    meta = payload.get("data_meta", {})
    if meta.get("input_dim") is not None:
        return int(meta["input_dim"])
    return len(payload.get("metrics", {}).get("x_mean", []))


def collect_rows() -> list[dict]:
    rows = []
    for source, path in collect_paths():
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = payload["model"]
        if method not in METHOD_LABELS:
            continue
        model, payload = load_model_from_json(path)
        variable_names = [f"x{i + 1}" for i in range(model_input_dim(payload))]
        export = export_graph(model, method, variable_names)
        rows.append(
            {
                "dataset": payload["dataset"],
                "seed": payload["seed"],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "source": source,
                "test_mse": payload["metrics"]["test"]["mse"],
                "test_r2": payload["metrics"]["test"]["r2"],
                "tokens": export.tokens,
                "ast_nodes": export.ast_nodes,
                "exp_log_ops": export.exp_log_ops,
                "basis_terms": export.basis_terms,
                "active_edges": export.active_edges,
                "active_terms": export.active_terms,
                "json_file": str(path.relative_to(ROOT)),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["method_label"]].append(row)
    summary = []
    for label in METHOD_ORDER:
        group = groups.get(label, [])
        if not group:
            continue
        summary.append(
            {
                "method": label,
                "n": len(group),
                "test_mse_mean": mean(r["test_mse"] for r in group),
                "test_mse_std": stdev([r["test_mse"] for r in group]) if len(group) > 1 else 0.0,
                "tokens_mean": mean(r["tokens"] for r in group),
                "tokens_median": median(r["tokens"] for r in group),
                "exp_log_ops_mean": mean(r["exp_log_ops"] for r in group),
                "basis_terms_mean": mean(r["basis_terms"] for r in group),
                "active_terms_mean": mean(r["active_terms"] for r in group),
            }
        )
    return summary


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        mantissa, exp = f"{value:.2e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exp)}}}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.3f}"


def texnum(value: float) -> str:
    text = fmt(value)
    return f"${text}$" if r"\times" in text else text


def write_summary_csv(summary: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def write_latex(summary: list[dict], path: Path) -> None:
    best_mse = min(row["test_mse_mean"] for row in summary)
    neural_min_tokens = min(row["tokens_mean"] for row in summary)
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Feynman clean12 equation-DAG export audit from existing trained checkpoints. This is a broader real-formula export check than the controlled equation-native suite. Bold marks best test MSE; underlining marks the shortest neural exported graph.}",
        r"\label{tab:feynman-export-audit}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & $n$ & Test MSE $\downarrow$ & Tokens $\downarrow$ & Median tokens $\downarrow$ & Exp/log ops & Basis terms $\downarrow$ \\",
        r"\midrule",
    ]
    for row in summary:
        mse = f"${fmt(row['test_mse_mean'])}\\pm{fmt(row['test_mse_std'])}$"
        if row["test_mse_mean"] == best_mse:
            mse = rf"\best{{{mse}}}"
        tokens = texnum(row["tokens_mean"])
        if row["tokens_mean"] == neural_min_tokens:
            tokens = rf"\underline{{{tokens}}}"
        lines.append(
            f"{row['method']} & {row['n']} & {mse} & {tokens} & "
            f"{texnum(row['tokens_median'])} & {texnum(row['exp_log_ops_mean'])} & "
            f"{texnum(row['basis_terms_mean'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    write_csv(rows, RESULTS / "feynman_clean12_export_audit.csv")
    summary = summarize(rows)
    write_summary_csv(summary, RESULTS / "feynman_clean12_export_audit_summary.csv")
    write_latex(summary, PAPER / "table_feynman_export_audit.tex")
    print(f"feynman export rows={len(rows)} methods={len(summary)}")


if __name__ == "__main__":
    main()
