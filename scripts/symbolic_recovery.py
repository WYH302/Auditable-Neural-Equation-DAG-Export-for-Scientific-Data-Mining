from __future__ import annotations

import csv
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import torch
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2"
PAPER = ROOT / "paper_elscas"
sys.path.append(str(ROOT / "experiments"))
from run_tabular import build_model, load_synthetic_dataset  # noqa: E402


RUN_DIRS = [
    RESULTS / "synthetic_final" / "synthetic",
    RESULTS / "baseline_supplement" / "synthetic" / "synthetic",
    RESULTS / "baseline_matched" / "synthetic_stable_eml_w8" / "synthetic",
]

METHOD_LABELS = {
    "mlp": "MLP",
    "stable_eml": "StableEML",
    "kan": "RBF-KAN",
    "eml_kan": "EML-KAN",
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TARGET_COEFFS = {
    "poly_1d_x2_x_1": {"1": 1.0, "x1": 1.0, "x1^2": 1.0},
    "poly_2d_x2_2y_1": {"1": 1.0, "x1^2": 1.0, "x2": 2.0},
    "exp_1d": {"exp(x1)": 1.0},
    "log_1d_x_plus_2": {"log(x1+2.1)": 1.0},
    "exp_log_1d": {"exp(x1)": 1.0, "log(x1+2.1)": -1.0},
    "exp_log_2d": {"exp(x1)": 1.0, "log(x2+2.1)": -1.0},
    "kinetic_energy": {"x2^2": 1.05, "x1*x2^2": 0.5},
    "inverse_quadratic": {"1/(x1^2+1)": 1.0},
    "sin_weak_case": {"sin(x1)": 1.0},
    "cos_plus_x_weak_case": {"cos(x1)": 1.0, "x1": 1.0},
}

TARGET_EXPR = {
    "poly_1d_x2_x_1": "x1^2 + x1 + 1",
    "poly_2d_x2_2y_1": "x1^2 + 2*x2 + 1",
    "exp_1d": "exp(x1)",
    "log_1d_x_plus_2": "log(x1+2.1)",
    "exp_log_1d": "exp(x1) - log(x1+2.1)",
    "exp_log_2d": "exp(x1) - log(x2+2.1)",
    "kinetic_energy": "0.5*x1*x2^2 + 1.05*x2^2",
    "inverse_quadratic": "1/(x1^2+1)",
    "sin_weak_case": "sin(x1)",
    "cos_plus_x_weak_case": "cos(x1) + x1",
}


def as_float(value: object, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    return float(value)


def as_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def model_input_dim(payload: dict) -> int:
    metrics = payload.get("metrics", {})
    return len(metrics.get("x_mean", []))


def build_from_payload(payload: dict):
    args = payload.get("args", {})
    return build_model(
        payload["model"],
        model_input_dim(payload),
        as_int(args.get("width"), 32),
        as_int(args.get("depth"), 2),
        as_float(args.get("tau"), 2.0),
        as_float(args.get("eta"), 0.35),
        args.get("exp_mode", "bounded_tanh"),
        as_float(args.get("clip_m"), 8.0),
        args.get("w0_init", "normal"),
    )


def predict_raw(model, payload: dict, x_raw: np.ndarray, device: torch.device) -> np.ndarray:
    metrics = payload["metrics"]
    x_mean = np.asarray(metrics["x_mean"], dtype=np.float32).reshape(1, -1)
    x_std = np.asarray(metrics["x_std"], dtype=np.float32).reshape(1, -1)
    y_mean = float(np.asarray(metrics["y_mean"], dtype=np.float32).reshape(-1)[0])
    y_std = float(np.asarray(metrics["y_std"], dtype=np.float32).reshape(-1)[0])
    x_norm = (x_raw.astype(np.float32) - x_mean) / x_std
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_norm), 4096):
            bx = torch.from_numpy(x_norm[start : start + 4096]).to(device)
            pred = model(bx).detach().cpu().numpy()
            preds.append(pred)
    pred_std = np.vstack(preds).reshape(-1)
    return pred_std * y_std + y_mean


def feature_library(dataset: str, x: np.ndarray) -> tuple[list[str], np.ndarray]:
    cols = []
    names = []
    x1 = x[:, 0].astype(np.float64)

    def add(name: str, values: np.ndarray) -> None:
        names.append(name)
        cols.append(values.astype(np.float64))

    if dataset == "poly_1d_x2_x_1":
        add("x1", x1)
        add("x1^2", x1**2)
        add("x1^3", x1**3)
    elif dataset == "poly_2d_x2_2y_1":
        x2 = x[:, 1].astype(np.float64)
        add("x1", x1)
        add("x2", x2)
        add("x1^2", x1**2)
        add("x2^2", x2**2)
        add("x1*x2", x1 * x2)
    elif dataset == "exp_1d":
        add("x1", x1)
        add("x1^2", x1**2)
        add("exp(x1)", np.exp(x1))
        add("log(x1+2.1)", np.log(x1 + 2.1))
    elif dataset == "log_1d_x_plus_2":
        add("x1", x1)
        add("x1^2", x1**2)
        add("exp(x1)", np.exp(x1))
        add("log(x1+2.1)", np.log(x1 + 2.1))
    elif dataset == "exp_log_1d":
        add("x1", x1)
        add("x1^2", x1**2)
        add("exp(x1)", np.exp(x1))
        add("log(x1+2.1)", np.log(x1 + 2.1))
    elif dataset == "exp_log_2d":
        x2 = x[:, 1].astype(np.float64)
        add("x1", x1)
        add("x2", x2)
        add("exp(x1)", np.exp(x1))
        add("exp(x2)", np.exp(x2))
        add("log(x1+2.1)", np.log(x1 + 2.1))
        add("log(x2+2.1)", np.log(x2 + 2.1))
    elif dataset == "kinetic_energy":
        x2 = x[:, 1].astype(np.float64)
        add("x1", x1)
        add("x2", x2)
        add("x2^2", x2**2)
        add("x1*x2", x1 * x2)
        add("x1*x2^2", x1 * x2**2)
    elif dataset == "inverse_quadratic":
        add("x1", x1)
        add("x1^2", x1**2)
        add("1/(x1^2+1)", 1.0 / (x1**2 + 1.0))
    elif dataset == "sin_weak_case":
        add("x1", x1)
        add("sin(x1)", np.sin(x1))
        add("cos(x1)", np.cos(x1))
    elif dataset == "cos_plus_x_weak_case":
        add("x1", x1)
        add("sin(x1)", np.sin(x1))
        add("cos(x1)", np.cos(x1))
    else:
        raise ValueError(f"Unknown synthetic dataset: {dataset}")
    return names, np.column_stack(cols)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true.reshape(-1) - y_pred.reshape(-1)) ** 2))


def fit_with_pruning(dataset, x_train, y_train, x_val, y_val_model):
    feature_names, phi_train = feature_library(dataset, x_train)
    _, phi_val = feature_library(dataset, x_val)
    candidates = []
    mu = phi_train.mean(axis=0)
    sigma = phi_train.std(axis=0)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)
    z_train = (phi_train - mu) / sigma
    z_val = (phi_val - mu) / sigma
    y_vec = y_train.reshape(-1)
    for alpha in [0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
        if alpha == 0.0:
            z_aug = np.column_stack([np.ones(len(z_train)), z_train])
            raw, *_ = np.linalg.lstsq(z_aug, y_vec, rcond=None)
            intercept_z = raw[0]
            coef_z = raw[1:]
        else:
            reg = Lasso(alpha=alpha, fit_intercept=True, max_iter=50000, tol=1e-8)
            reg.fit(z_train, y_vec)
            intercept_z = float(reg.intercept_)
            coef_z = reg.coef_.astype(np.float64)
        coef = coef_z / sigma
        intercept = intercept_z - float(np.sum(coef_z * mu / sigma))
        names = ["1"] + feature_names
        full_coef = np.concatenate([[intercept], coef])
        active = np.abs(full_coef) >= 1e-4
        if not active.any():
            active[0] = True
        active_features = active[1:]
        if active_features.any():
            refit_phi = np.column_stack([np.ones(len(phi_train)), phi_train[:, active_features]])
            refit_coef, *_ = np.linalg.lstsq(refit_phi, y_vec, rcond=None)
            final_coef = np.zeros_like(full_coef)
            final_coef[0] = refit_coef[0]
            final_coef[1:][active_features] = refit_coef[1:]
        else:
            final_coef = np.zeros_like(full_coef)
            final_coef[0] = float(np.mean(y_vec))
        val_phi = np.column_stack([np.ones(len(phi_val)), phi_val])
        pred_val = val_phi @ final_coef
        candidates.append(
            {
                "coef": final_coef,
                "active": np.abs(final_coef) >= 1e-4,
                "val_model_mse": mse(y_val_model, pred_val),
                "complexity": int((np.abs(final_coef) >= 1e-4).sum()),
            }
        )
    best = min(candidates, key=lambda c: c["val_model_mse"])
    tolerance = best["val_model_mse"] * 1.05 + 1e-10
    simple = [c for c in candidates if c["val_model_mse"] <= tolerance]
    chosen = min(simple, key=lambda c: (c["complexity"], c["val_model_mse"]))
    return names, chosen["coef"], chosen["active"], chosen["val_model_mse"]


def expression_from_coeffs(names: list[str], coeffs: np.ndarray, active: np.ndarray) -> str:
    parts = []
    for name, coef, keep in zip(names, coeffs, active):
        if not keep or abs(coef) < 1e-10:
            continue
        if name == "1":
            parts.append(f"{coef:.4g}")
        elif abs(coef - 1.0) < 5e-4:
            parts.append(name)
        elif abs(coef + 1.0) < 5e-4:
            parts.append(f"-{name}")
        else:
            parts.append(f"{coef:.4g}*{name}")
    if not parts:
        return "0"
    expr = " + ".join(parts)
    return expr.replace("+ -", "- ")


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def normalized_edit_distance(expr: str, target: str) -> float:
    expr = expr.replace(" ", "")
    target = target.replace(" ", "")
    return levenshtein(expr, target) / max(len(expr), len(target), 1)


def coeff_error(names: list[str], coeffs: np.ndarray, target: dict[str, float]) -> float:
    lookup = {name: float(coef) for name, coef in zip(names, coeffs)}
    all_names = set(target) | {name for name, coef in lookup.items() if abs(coef) > 1e-6}
    errors = [abs(lookup.get(name, 0.0) - target.get(name, 0.0)) for name in all_names]
    return float(np.mean(errors)) if errors else 0.0


def exact_recovery(names: list[str], coeffs: np.ndarray, active: np.ndarray, target: dict[str, float], symbolic_test_mse: float) -> bool:
    active_support = {name for name, keep in zip(names, active) if keep and abs(coeffs[names.index(name)]) > 1e-4}
    target_support = set(target)
    if active_support != target_support:
        return False
    max_error = max(abs(float(coeffs[names.index(name)]) - value) for name, value in target.items())
    return max_error <= 0.05 and symbolic_test_mse <= 1e-4


def collect_jsons():
    paths = []
    for run_dir in RUN_DIRS:
        paths += sorted(run_dir.glob("*.json"))
    has_matched_stable = any(
        "baseline_matched" in str(path) and "stable_eml" in path.name for path in paths
    )
    if has_matched_stable:
        paths = [
            path
            for path in paths
            if not (
                "baseline_supplement" in str(path)
                and "stable_eml" in path.name
            )
        ]
    return paths


def recover_one(path: Path, device: torch.device) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    checkpoint = path.with_suffix(".pt")
    model = build_from_payload(payload)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    dataset = payload["dataset"]
    seed = as_int(payload["seed"])
    splits = load_synthetic_dataset(dataset, seed)
    train_x, train_y = splits["train"]
    val_x, val_y = splits["val"]
    test_x, test_y = splits["test"]
    model_train = predict_raw(model, payload, train_x, device)
    model_val = predict_raw(model, payload, val_x, device)
    names, coeffs, active, val_model_mse = fit_with_pruning(dataset, train_x, model_train, val_x, model_val)
    _, phi_val_no_intercept = feature_library(dataset, val_x)
    _, phi_test_no_intercept = feature_library(dataset, test_x)
    phi_val = np.column_stack([np.ones(len(phi_val_no_intercept)), phi_val_no_intercept])
    phi_test = np.column_stack([np.ones(len(phi_test_no_intercept)), phi_test_no_intercept])
    pred_val = phi_val @ coeffs
    pred_test = phi_test @ coeffs
    expr = expression_from_coeffs(names, coeffs, active)
    target_expr = TARGET_EXPR[dataset]
    target_coeffs = TARGET_COEFFS[dataset]
    symbolic_val_mse = mse(val_y, pred_val)
    symbolic_test_mse = mse(test_y, pred_test)
    return {
        "dataset": dataset,
        "seed": seed,
        "method": payload["model"],
        "method_label": METHOD_LABELS.get(payload["model"], payload["model"]),
        "expression": expr,
        "target_expression": target_expr,
        "complexity": int(sum(1 for keep, coef in zip(active, coeffs) if keep and abs(coef) > 1e-10)),
        "target_complexity": len(target_coeffs),
        "val_model_mse": val_model_mse,
        "symbolic_val_mse": symbolic_val_mse,
        "symbolic_test_mse": symbolic_test_mse,
        "ned": normalized_edit_distance(expr, target_expr),
        "constant_error": coeff_error(names, coeffs, target_coeffs),
        "exact_recovery": int(exact_recovery(names, coeffs, active, target_coeffs, symbolic_test_mse)),
        "checkpoint_file": str(checkpoint.relative_to(ROOT)),
        "json_file": str(path.relative_to(ROOT)),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite(values):
    return [float(v) for v in values if math.isfinite(float(v))]


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    summary = []
    for method, group in sorted(groups.items()):
        symbolic = finite([r["symbolic_test_mse"] for r in group])
        complexity = finite([r["complexity"] for r in group])
        ned = finite([r["ned"] for r in group])
        const_error = finite([r["constant_error"] for r in group])
        summary.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n": len(group),
                "exact_recovery": sum(int(r["exact_recovery"]) for r in group) / len(group),
                "exact_count": sum(int(r["exact_recovery"]) for r in group),
                "symbolic_mse_mean": mean(symbolic),
                "symbolic_mse_median": median(symbolic),
                "complexity_mean": mean(complexity),
                "ned_mean": mean(ned),
                "constant_error_mean": mean(const_error),
                "symbolic_mse_std": stdev(symbolic) if len(symbolic) > 1 else 0.0,
            }
        )
    return summary


def fmt_value(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        mantissa, exp = f"{value:.2e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exp)}}}"
    return f"{value:.4f}"


def tex_text(value: str) -> str:
    return value.replace("_", r"\_")


def write_latex_summary(summary: list[dict], path: Path) -> None:
    order = ["MLP", "RBF-KAN", "StableEML", "EML-KAN"]
    summary = sorted(summary, key=lambda r: order.index(r["method_label"]))
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Synthetic symbolic distillation results. Expressions are extracted by sparse linear distillation over a fixed formula library from each trained neural predictor, then evaluated against the true symbolic target.}",
        r"\label{tab:symbolic-recovery}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Exact recovery $\uparrow$ & Symbolic MSE $\downarrow$ & Complexity $\downarrow$ & NED $\downarrow$ & Const. error $\downarrow$ \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{row['method_label']} & {row['exact_count']}/{row['n']} & "
            f"${fmt_value(row['symbolic_mse_mean'])}$ & "
            f"{row['complexity_mean']:.2f} & {row['ned_mean']:.3f} & "
            f"{row['constant_error_mean']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_examples(rows: list[dict], path: Path) -> None:
    selected = []
    for dataset in [
        "exp_1d",
        "log_1d_x_plus_2",
        "exp_log_1d",
        "poly_1d_x2_x_1",
        "inverse_quadratic",
        "kinetic_energy",
    ]:
        candidates = [
            row
            for row in rows
            if row["dataset"] == dataset and row["method"] == "eml_kan" and row["seed"] == 0
        ]
        if candidates:
            selected.append(candidates[0])
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Representative EML-KAN symbolic distillation examples on synthetic functions.}",
        r"\label{tab:symbolic-examples}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrr}",
        r"\toprule",
        r"Dataset & Target & Recovered expression & Symbolic MSE & Complexity \\",
        r"\midrule",
    ]
    for row in selected:
        expr = row["expression"].replace("*", r"\,")
        target = row["target_expression"].replace("*", r"\,")
        lines.append(
            f"{tex_text(row['dataset'])} & ${target}$ & ${expr}$ & "
            f"${fmt_value(row['symbolic_test_mse'])}$ & {row['complexity']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(summary: list[dict], rows: list[dict], path: Path) -> None:
    lines = [
        "# Symbolic Recovery Report",
        "",
        "This is a model-agnostic sparse library distillation proxy on synthetic datasets. It is not a full PySR or AI-Feynman symbolic search baseline.",
        "",
        "| Method | Exact | Symbolic MSE mean | Symbolic MSE median | Complexity | NED | Constant error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method_label']} | {row['exact_count']}/{row['n']} | "
            f"{row['symbolic_mse_mean']:.4g} | {row['symbolic_mse_median']:.4g} | "
            f"{row['complexity_mean']:.2f} | {row['ned_mean']:.3f} | "
            f"{row['constant_error_mean']:.3f} |"
        )
    lines += ["", "## EML-KAN Examples", ""]
    for row in rows:
        if row["method"] == "eml_kan" and row["seed"] == 0:
            lines.append(
                f"- {row['dataset']}: target `{row['target_expression']}`; "
                f"recovered `{row['expression']}`; symbolic_test_mse={row['symbolic_test_mse']:.4g}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for path in collect_jsons():
        rows.append(recover_one(path, device))
    summary = summarize(rows)
    out_dir = RESULTS / "symbolic_recovery"
    write_csv(rows, out_dir / "synthetic_symbolic_runs.csv")
    write_csv(summary, out_dir / "synthetic_symbolic_summary.csv")
    write_latex_summary(summary, PAPER / "table_symbolic_recovery.tex")
    write_examples(rows, PAPER / "table_symbolic_examples.tex")
    write_report(summary, rows, out_dir / "symbolic_recovery_report.md")
    print(f"runs={len(rows)} methods={len(summary)}")


if __name__ == "__main__":
    main()
