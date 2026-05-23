from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import sympy as sp


SCIENTIFIC_OPS = {
    sp.Add,
    sp.Mul,
    sp.Pow,
    sp.exp,
    sp.log,
    sp.sqrt,
}


@dataclass
class ExpressionMetrics:
    tokens: int
    ast_nodes: int
    constants: int
    operators: int
    exp_log_ops: int
    scientific_operator_ratio: float
    tree_depth: int
    piecewise_segments: int
    basis_terms: int
    coefficient_count: int


def tree_depth(expr: sp.Expr) -> int:
    if not expr.args:
        return 1
    return 1 + max(tree_depth(arg) for arg in expr.args)


def count_piecewise(expr: sp.Expr) -> int:
    if isinstance(expr, sp.Piecewise):
        return len(expr.args)
    values = [count_piecewise(arg) for arg in expr.args]
    return max(values) if values else 0


def count_basis_terms(expr_text: str) -> int:
    return len(re.findall(r"\brbf\(|\bspline\(|\bB_\d+|\bknot", expr_text))


def token_count(expr_text: str) -> int:
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+\.\d+|\d+|[+\-*/^(),]", expr_text)
    return len(tokens)


def count_ops(expr: sp.Expr) -> tuple[int, int, float]:
    op_counter: Counter[type] = Counter()
    exp_log = 0
    sci = 0
    total = 0
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Basic) and node.args:
            total += 1
            op_counter[type(node)] += 1
            if node.func in {sp.exp, sp.log}:
                exp_log += 1
            if node.func in SCIENTIFIC_OPS or type(node) in SCIENTIFIC_OPS:
                sci += 1
    ratio = sci / total if total else 0.0
    return total, exp_log, ratio


def count_constants(expr: sp.Expr) -> int:
    return sum(1 for node in sp.preorder_traversal(expr) if node.is_number and not node.is_Integer)


def metrics_for_expr(expr: sp.Expr | str) -> ExpressionMetrics:
    if isinstance(expr, str):
        expr_obj = sp.sympify(expr)
        expr_text = expr
    else:
        expr_obj = expr
        expr_text = sp.sstr(expr)
    operators, exp_log_ops, sci_ratio = count_ops(expr_obj)
    coeff_count = count_constants(expr_obj)
    return ExpressionMetrics(
        tokens=token_count(expr_text),
        ast_nodes=sum(1 for _ in sp.preorder_traversal(expr_obj)),
        constants=coeff_count,
        operators=operators,
        exp_log_ops=exp_log_ops,
        scientific_operator_ratio=sci_ratio,
        tree_depth=tree_depth(expr_obj),
        piecewise_segments=count_piecewise(expr_obj),
        basis_terms=count_basis_terms(expr_text),
        coefficient_count=coeff_count,
    )


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def normalized_edit_distance(expr: str, target: str) -> float:
    expr = re.sub(r"\s+", "", expr)
    target = re.sub(r"\s+", "", target)
    return levenshtein(expr, target) / max(len(expr), len(target), 1)


def safe_float(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else math.nan

