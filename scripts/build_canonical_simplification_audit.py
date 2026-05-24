from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2" / "equation_export"
PAPER = ROOT / "paper_elscas"

METHOD_ORDER = ["MLP", "RBF-KAN", "StableEML", "EML-KAN", "PySR"]
METHOD_FROM_FILE = {
    "mlp": "MLP",
    "kan": "RBF-KAN",
    "stable_eml": "StableEML",
    "eml_kan": "EML-KAN",
    "pysr": "PySR",
}


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+\.\d+|\d+|[+\-*/^(),=;]", text))


def tree_depth(expr: sp.Expr) -> int:
    if not expr.args:
        return 1
    return 1 + max(tree_depth(arg) for arg in expr.args)


def count_basis_terms(text: str) -> int:
    return len(re.findall(r"\brbf\(|\bspline\(|\bB_\d+|\bknot", text))


def count_ops(text: str) -> Counter[str]:
    return Counter(re.findall(r"\b(exp|log|tanh|softplus|silu|rbf|LayerNormAffine)\s*\(", text))


def normalize_rhs(rhs: str) -> str:
    rhs = rhs.strip()
    rhs = rhs.replace("^", "**")
    rhs = rhs.replace("+-", "-")
    rhs = re.sub(r"rbf\(([^;()]+);([^,()]+),([^)]+)\)", r"rbf(\1,\2,\3)", rhs)
    rhs = re.sub(r"LayerNormAffine\(([^;]+);([^,]+),([^)]+)\)", r"LayerNormAffine(\1,\2,\3)", rhs)
    return rhs


def canonicalize_expression(expr_text: str) -> tuple[str, int, int, bool]:
    locals_map = {
        "silu": sp.Function("silu"),
        "softplus": sp.Function("softplus"),
        "rbf": sp.Function("rbf"),
        "LayerNormAffine": sp.Function("LayerNormAffine"),
    }
    statements = [part.strip().rstrip(";") for part in re.split(r";\s*\n", expr_text) if part.strip()]
    if not statements:
        statements = [expr_text.strip()]
    simplified: list[str] = []
    depths: list[int] = []
    ast_nodes = 0
    ok = True
    for statement in statements:
        if "=" in statement:
            lhs, rhs = statement.split("=", 1)
            lhs = lhs.strip()
        else:
            lhs, rhs = "", statement
        rhs_norm = normalize_rhs(rhs)
        try:
            expr = sp.sympify(rhs_norm, locals=locals_map)
            expr = sp.factor_terms(expr)
            rhs_out = sp.sstr(expr, full_prec=False)
            depths.append(tree_depth(expr))
            ast_nodes += sum(1 for _ in sp.preorder_traversal(expr))
        except Exception:
            ok = False
            rhs_out = rhs.strip()
        simplified.append(f"{lhs} = {rhs_out}" if lhs else rhs_out)
    text = ";\n".join(simplified)
    return text, max(depths) if depths else 0, ast_nodes, ok


def method_from_name(path: Path) -> str:
    name = path.stem
    for key, label in sorted(METHOD_FROM_FILE.items(), key=lambda item: -len(item[0])):
        if f"_{key}_seed" in name:
            return label
    raise ValueError(f"Could not infer method from {path.name}")


def dataset_from_name(path: Path, method: str) -> str:
    key = {label: raw for raw, label in METHOD_FROM_FILE.items()}[method]
    return re.sub(rf"_{key}_seed\d+$", "", path.stem)


def collect_rows() -> list[dict]:
    rows = []
    for path in sorted((RESULTS / "exported_formulas").glob("*.txt")):
        method = method_from_name(path)
        dataset = dataset_from_name(path, method)
        text = path.read_text(encoding="utf-8")
        canonical, depth, ast_nodes, ok = canonicalize_expression(text)
        op_before = count_ops(text)
        op_after = count_ops(canonical)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "original_tokens": token_count(text),
                "canonical_tokens": token_count(canonical),
                "token_reduction_pct": 100.0 * (token_count(text) - token_count(canonical)) / max(token_count(text), 1),
                "canonical_ast_nodes": ast_nodes,
                "canonical_depth": depth,
                "basis_terms": count_basis_terms(text),
                "parse_ok": ok,
                "exp_log_ops_before": op_before["exp"] + op_before["log"],
                "exp_log_ops_after": op_after["exp"] + op_after["log"],
                "activation_ops_after": op_after["silu"] + op_after["softplus"] + op_after["tanh"],
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.2f}"


def write_summary(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    summary = []
    for method in METHOD_ORDER:
        group = groups.get(method, [])
        if not group:
            continue
        summary.append(
            {
                "method": method,
                "n": len(group),
                "original_tokens_mean": mean(r["original_tokens"] for r in group),
                "canonical_tokens_mean": mean(r["canonical_tokens"] for r in group),
                "canonical_tokens_median": median(r["canonical_tokens"] for r in group),
                "token_reduction_pct_mean": mean(r["token_reduction_pct"] for r in group),
                "canonical_depth_mean": mean(r["canonical_depth"] for r in group),
                "basis_terms_mean": mean(r["basis_terms"] for r in group),
                "parse_ok": sum(1 for r in group if r["parse_ok"]),
            }
        )
    return summary


def write_summary_csv(summary: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def write_latex(summary: list[dict], path: Path) -> None:
    best_all = min(row["canonical_tokens_mean"] for row in summary)
    neural = [row for row in summary if row["method"] != "PySR"]
    best_neural = min(row["canonical_tokens_mean"] for row in neural)
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Canonical simplification audit for exported equation graphs. The same local canonicalization pass is applied to all exported text: constant folding, algebraic term ordering, function-call normalization, and per-assignment symbolic simplification. Bold marks the shortest canonical export overall; underlining marks the shortest neural export.}",
        r"\label{tab:canonical-simplification}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & $n$ & Original tokens $\downarrow$ & Canonical tokens $\downarrow$ & Median canonical $\downarrow$ & Reduction \% $\uparrow$ & Depth $\downarrow$ & Basis terms $\downarrow$ \\",
        r"\midrule",
    ]
    for row in summary:
        canonical = fmt(row["canonical_tokens_mean"])
        if row["canonical_tokens_mean"] == best_all:
            canonical = rf"\best{{{canonical}}}"
        elif row["canonical_tokens_mean"] == best_neural:
            canonical = rf"\underline{{{canonical}}}"
        lines.append(
            f"{row['method']} & {row['n']} & {fmt(row['original_tokens_mean'])} & "
            f"{canonical} & {fmt(row['canonical_tokens_median'])} & "
            f"{fmt(row['token_reduction_pct_mean'])} & {fmt(row['canonical_depth_mean'])} & "
            f"{fmt(row['basis_terms_mean'])} \\\\"
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
    write_csv(rows, RESULTS / "canonical_simplification_audit.csv")
    summary = write_summary(rows)
    write_summary_csv(summary, RESULTS / "canonical_simplification_summary.csv")
    write_latex(summary, PAPER / "table_canonical_simplification.tex")
    print(f"canonical rows={len(rows)} methods={len(summary)}")


if __name__ == "__main__":
    main()
