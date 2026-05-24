from __future__ import annotations

import csv
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RESULTS = ROOT / "results_v2" / "equation_export"
PAPER = PROJECT_ROOT / "paper" / "paper_cikm2026"

METHOD_ORDER = ["MLP", "RBF-KAN", "StableEML", "EML-KAN", "PySR"]
TABLE_METHOD_ORDER = ["MLP", "RBF-KAN", "StableEML", "EML-KAN"]
EXACT_NEURAL = {"MLP", "RBF-KAN", "EML-KAN"}
METHOD_FROM_FILE = {
    "stable_eml": "StableEML",
    "eml_kan": "EML-KAN",
    "mlp": "MLP",
    "kan": "RBF-KAN",
    "pysr": "PySR",
}


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+\.\d+|\d+|[+\-*/^(),=;]", text))


def round_numeric_literals(text: str, digits: int) -> str:
    pattern = re.compile(r"(?<![A-Za-z_0-9.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:e[-+]?\d+)?", re.IGNORECASE)

    def repl(match: re.Match[str]) -> str:
        value = float(match.group(0))
        return f"{value:.{digits}g}"

    return pattern.sub(repl, text)


def split_assignments(text: str) -> list[str]:
    statements: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == ";" and depth == 0:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def normalize_rhs(rhs: str) -> str:
    rhs = rhs.strip().replace("^", "**").replace("+-", "-")
    rhs = re.sub(r"rbf\(([^;()]+);([^,()]+),([^)]+)\)", r"rbf(\1,\2,\3)", rhs)
    rhs = re.sub(r"LayerNormAffine\(([^;]+);([^,]+),([^)]+)\)", r"LayerNormAffine(\1,\2,\3)", rhs)
    return rhs


def canonicalize_local(text: str) -> tuple[str, int, int, bool, float]:
    locals_map = {
        "silu": sp.Function("silu"),
        "softplus": sp.Function("softplus"),
        "rbf": sp.Function("rbf"),
        "LayerNormAffine": sp.Function("LayerNormAffine"),
    }
    start = time.perf_counter()
    output: list[str] = []
    depths: list[int] = []
    assignment_count = 0
    ok = True
    for statement in split_assignments(text):
        if "=" not in statement:
            try:
                expr = sp.sympify(normalize_rhs(statement), locals=locals_map)
                expr = sp.factor_terms(expr)
                output.append(sp.sstr(expr, full_prec=False))
                depths.append(tree_depth(expr))
            except Exception:
                ok = False
                output.append(statement)
            continue
        lhs, rhs = statement.split("=", 1)
        lhs = lhs.strip()
        assignment_count += 1
        try:
            expr = sp.sympify(normalize_rhs(rhs), locals=locals_map)
            expr = sp.factor_terms(expr)
            output.append(f"{lhs} = {sp.sstr(expr, full_prec=False)}")
            depths.append(tree_depth(expr))
        except Exception:
            ok = False
            output.append(statement)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return ";\n".join(output), max(depths) if depths else 0, assignment_count, ok, elapsed_ms


def tree_depth(expr: sp.Expr) -> int:
    if not expr.args:
        return 1
    return 1 + max(tree_depth(arg) for arg in expr.args)


def replace_symbol_tokens(text: str, replacements: dict[str, str]) -> str:
    if not replacements:
        return text
    pattern = re.compile(r"\b(" + "|".join(re.escape(key) for key in sorted(replacements, key=len, reverse=True)) + r")\b")
    return pattern.sub(lambda match: replacements[match.group(1)], text)


def inline_final_expression(text: str, max_chars: int = 2_000_000) -> str:
    env: dict[str, str] = {}
    for statement in split_assignments(text):
        if "=" not in statement:
            continue
        lhs, rhs = statement.split("=", 1)
        lhs = lhs.strip()
        expanded = replace_symbol_tokens(rhs.strip(), {key: f"({value})" for key, value in env.items()})
        if len(expanded) > max_chars:
            raise ValueError(f"expanded expression exceeded {max_chars} characters")
        env[lhs] = expanded
    if "y" not in env:
        raise ValueError("No final y assignment found")
    return f"({env['y']})"


def global_cse_text(text: str) -> tuple[str, bool]:
    locals_map = {
        "silu": sp.Function("silu"),
        "softplus": sp.Function("softplus"),
        "rbf": sp.Function("rbf"),
        "LayerNormAffine": sp.Function("LayerNormAffine"),
    }
    lhs_names: list[str] = []
    exprs: list[sp.Expr] = []
    try:
        for statement in split_assignments(text):
            if "=" not in statement:
                exprs.append(sp.sympify(normalize_rhs(statement), locals=locals_map))
                continue
            lhs, rhs = statement.split("=", 1)
            lhs_names.append(lhs.strip())
            exprs.append(sp.sympify(normalize_rhs(rhs), locals=locals_map))
        replacements, reduced = sp.cse(exprs, symbols=sp.numbered_symbols("cse"))
    except Exception:
        return text, False
    lines = [f"{symbol} = {sp.sstr(expr, full_prec=False)}" for symbol, expr in replacements]
    if lhs_names:
        lines += [f"{lhs} = {sp.sstr(expr, full_prec=False)}" for lhs, expr in zip(lhs_names, reduced)]
    else:
        lines += [sp.sstr(expr, full_prec=False) for expr in reduced]
    return ";\n".join(lines), True


def method_from_name(path: Path) -> str:
    name = path.stem
    for key, label in sorted(METHOD_FROM_FILE.items(), key=lambda item: -len(item[0])):
        if f"_{key}_seed" in name:
            return label
    raise ValueError(f"Could not infer method from {path.name}")


def dataset_from_name(path: Path, method: str) -> str:
    raw = {label: key for key, label in METHOD_FROM_FILE.items()}[method]
    return re.sub(rf"_{raw}_seed\d+$", "", path.stem)


def collect_source_files() -> list[dict]:
    rows: list[dict] = []
    for path in sorted((RESULTS / "exported_formulas").glob("*.txt")):
        method = method_from_name(path)
        rows.append(
            {
                "dataset": dataset_from_name(path, method),
                "method": method,
                "source": "model_core",
                "path": path,
                "text": path.read_text(encoding="utf-8"),
            }
        )
    for path in sorted((RESULTS / "full_deployment_formulas").glob("*.txt")):
        method = method_from_name(path)
        rows.append(
            {
                "dataset": dataset_from_name(path, method),
                "method": method,
                "source": "full_deployment",
                "path": path,
                "text": path.read_text(encoding="utf-8"),
            }
        )
    return rows


def add_row(rows: list[dict], item: dict, setting: str, text: str, status: str = "ok", elapsed_ms: float = math.nan) -> None:
    rows.append(
        {
            "dataset": item["dataset"],
            "method": item["method"],
            "source": item["source"],
            "setting": setting,
            "tokens": token_count(text) if status == "ok" else math.nan,
            "bytes": len(text.encode("utf-8")) if status == "ok" else math.nan,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "file": str(item["path"].relative_to(ROOT)),
        }
    )


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for item in collect_source_files():
        text = item["text"]
        if item["source"] == "model_core":
            add_row(rows, item, "raw_core", text)
            add_row(rows, item, "precision4_core", round_numeric_literals(text, 4))
            add_row(rows, item, "precision8_core", round_numeric_literals(text, 8))
            canonical, _depth, _assignments, ok, elapsed_ms = canonicalize_local(text)
            add_row(rows, item, "local_canonical_core", canonical, "ok" if ok else "parse_fallback", elapsed_ms)
            cse, cse_ok = global_cse_text(text)
            add_row(rows, item, "global_cse_core", cse, "ok" if cse_ok else "parse_failed")
            try:
                start = time.perf_counter()
                inlined = inline_final_expression(text)
                add_row(rows, item, "inline_final_expr", inlined, "ok", (time.perf_counter() - start) * 1000.0)
            except Exception as exc:
                failed = dict(item)
                failed["path"] = item["path"]
                add_row(rows, failed, "inline_final_expr", "", f"failed:{type(exc).__name__}")
        elif item["source"] == "full_deployment":
            add_row(rows, item, "raw_full_deployment", text)
            add_row(rows, item, "precision4_full_deployment", round_numeric_literals(text, 4))
            canonical, _depth, _assignments, ok, elapsed_ms = canonicalize_local(text)
            add_row(rows, item, "local_canonical_full", canonical, "ok" if ok else "parse_fallback", elapsed_ms)
    return rows


def finite(values) -> list[float]:
    clean = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            clean.append(value)
    return clean


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["setting"])].append(row)
    out: list[dict] = []
    for method in METHOD_ORDER:
        for setting in sorted({key[1] for key in grouped if key[0] == method}):
            group = grouped[(method, setting)]
            token_values = finite([row["tokens"] for row in group])
            byte_values = finite([row["bytes"] for row in group])
            elapsed_values = finite([row["elapsed_ms"] for row in group])
            status_counts: dict[str, int] = defaultdict(int)
            for row in group:
                status_counts[row["status"]] += 1
            out.append(
                {
                    "method": method,
                    "setting": setting,
                    "n": len(group),
                    "ok": status_counts.get("ok", 0),
                    "tokens_mean": mean(token_values) if token_values else math.nan,
                    "tokens_median": median(token_values) if token_values else math.nan,
                    "bytes_mean": mean(byte_values) if byte_values else math.nan,
                    "elapsed_ms_mean": mean(elapsed_values) if elapsed_values else math.nan,
                    "status_counts": "; ".join(f"{key}={value}" for key, value in sorted(status_counts.items())),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])


def fmt(value: float) -> str:
    if not math.isfinite(float(value)):
        return "--"
    if abs(float(value)) >= 100:
        return f"{float(value):.0f}"
    return f"{float(value):.2f}"


def render_latex_table(summary: list[dict]) -> str:
    keep_settings = [
        "raw_core",
        "precision4_core",
        "precision8_core",
        "local_canonical_core",
        "global_cse_core",
        "raw_full_deployment",
        "local_canonical_full",
        "inline_final_expr",
    ]
    labels = {
        "raw_core": "Raw core",
        "precision4_core": "4-digit core",
        "precision8_core": "8-digit core",
        "local_canonical_core": "Local canon.",
        "global_cse_core": "Global CSE",
        "raw_full_deployment": "Full deploy",
        "local_canonical_full": "Full canon.",
        "inline_final_expr": "Inline final",
    }
    lookup = {(row["method"], row["setting"]): row for row in summary}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Token sensitivity audit for neural checkpoint-DAG exports. Values are mean lexical tokens over available equation-native exports. Numeric precision changes do not alter lexical token counts because each literal counts as one token; full-deployment rows include input standardization and target inverse-standardization. Global CSE and inline-final rows are sensitivity checks rather than the main assignment-preserving protocol; symbolic-search references are omitted because they are not checkpoint exports.}",
        r"\label{tab:token-sensitivity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & Raw core & 4-digit core & 8-digit core & Local canon. & Global CSE & Full deploy & Full canon. & Inline final \\",
        r"\midrule",
    ]
    for method in TABLE_METHOD_ORDER:
        cells = []
        for setting in keep_settings:
            row = lookup.get((method, setting))
            cells.append(fmt(row["tokens_mean"]) if row else "--")
        lines.append(f"{method} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def write_latex(summary: list[dict]) -> None:
    (PAPER / "table_token_sensitivity.tex").write_text(render_latex_table(summary), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    detail_fields = ["dataset", "method", "source", "setting", "tokens", "bytes", "status", "elapsed_ms", "file"]
    write_csv(RESULTS / "token_sensitivity.csv", rows, detail_fields)
    summary = summarize(rows)
    summary_fields = ["method", "setting", "n", "ok", "tokens_mean", "tokens_median", "bytes_mean", "elapsed_ms_mean", "status_counts"]
    write_csv(RESULTS / "token_sensitivity_summary.csv", summary, summary_fields)
    write_latex(summary)
    print(f"token sensitivity rows={len(rows)} summary={len(summary)}")


if __name__ == "__main__":
    main()
