from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RESULTS = ROOT / "results_v2" / "equation_export"
PAPER = PROJECT_ROOT / "paper" / "paper_cikm2026"
EXACT_NEURAL_METHODS = {"mlp", "kan", "eml_kan"}
METHOD_LABELS = {"mlp": "MLP", "kan": "RBF-KAN", "eml_kan": "EML-KAN"}
METHOD_ORDER = ["mlp", "kan", "eml_kan"]
SPLIT_ORDER = ["test", "ood"]
TOLERANCE = 1e-6

sys.path.append(str(ROOT / "experiments"))
sys.path.append(str(ROOT / "src"))
from symbolic_export.export_models import load_model_from_json, torch_predict  # noqa: E402
from symbolic_export.graph_export import export_graph, fmt, token_count  # noqa: E402


def is_exact_neural_export(method: str) -> bool:
    return method in EXACT_NEURAL_METHODS


def summarize_errors(y_checkpoint: Iterable[float], y_export: Iterable[float]) -> dict:
    checkpoint = np.asarray(list(y_checkpoint), dtype=np.float64).reshape(-1)
    exported = np.asarray(list(y_export), dtype=np.float64).reshape(-1)
    if checkpoint.shape != exported.shape:
        raise ValueError(f"Shape mismatch: {checkpoint.shape} vs {exported.shape}")
    finite = np.isfinite(checkpoint) & np.isfinite(exported)
    if not finite.any():
        return {
            "n_total": int(checkpoint.size),
            "n_finite": 0,
            "mse": math.nan,
            "mae": math.nan,
            "max_abs": math.nan,
        }
    diff = checkpoint[finite] - exported[finite]
    abs_diff = np.abs(diff)
    return {
        "n_total": int(checkpoint.size),
        "n_finite": int(finite.sum()),
        "mse": float(np.mean(diff**2)),
        "mae": float(np.mean(abs_diff)),
        "max_abs": float(np.max(abs_diff)),
    }


def status_from_tolerance(max_abs: float, tolerance: float) -> str:
    if not math.isfinite(float(max_abs)):
        return "not_evaluated"
    return "within_tolerance" if float(max_abs) <= tolerance else "exceeds_tolerance"


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


def softplus(value):
    value = np.asarray(value, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0)


def silu(value):
    return value / (1.0 + np.exp(-value))


def rbf(value, center, gamma):
    return np.exp(-((value - center) * gamma) ** 2)


def normalize_rhs(rhs: str) -> str:
    rhs = rhs.replace("^", "**")
    return re.sub(r"rbf\(([^;()]+);([^,()]+),([^)]+)\)", r"rbf(\1,\2,\3)", rhs)


def evaluate_assignment_graph(text: str, x_raw: Iterable[Iterable[float]]) -> np.ndarray:
    x = np.asarray(list(x_raw), dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    env = {
        "exp": np.exp,
        "log": np.log,
        "tanh": np.tanh,
        "softplus": softplus,
        "silu": silu,
        "rbf": rbf,
        "np": np,
    }
    for idx in range(x.shape[1]):
        env[f"x{idx + 1}"] = x[:, idx]
    for statement in split_assignments(text):
        if "=" not in statement:
            continue
        lhs, rhs = statement.split("=", 1)
        lhs = lhs.strip()
        rhs = normalize_rhs(rhs.strip())
        env[lhs] = eval(rhs, {"__builtins__": {}}, env)
    if "y" not in env:
        raise ValueError("Serialized assignment graph did not produce `y`")
    return np.asarray(env["y"], dtype=np.float64).reshape(-1)


def model_input_dim(payload: dict) -> int:
    meta = payload.get("data_meta", {})
    if meta.get("input_dim") is not None:
        return int(meta["input_dim"])
    return len(payload.get("metrics", {}).get("x_mean", []))


def deployment_graph_text(model, method: str, payload: dict) -> tuple[str, int, int]:
    metrics = payload["metrics"]
    input_dim = model_input_dim(payload)
    raw_names = [f"x{i + 1}" for i in range(input_dim)]
    norm_names = [f"x{i + 1}_norm" for i in range(input_dim)]
    prefix = [
        f"{norm_names[i]} = ({raw_names[i]} - {fmt(metrics['x_mean'][i])})/{fmt(metrics['x_std'][i])}"
        for i in range(input_dim)
    ]
    core = export_graph(model, method, norm_names, gate_threshold=0.0)
    y_mean = float(np.asarray(metrics["y_mean"], dtype=np.float64).reshape(-1)[0])
    y_std = float(np.asarray(metrics["y_std"], dtype=np.float64).reshape(-1)[0])
    core_text = core.text.replace("y_std", fmt(y_std)).replace("y_mean", fmt(y_mean))
    text = ";\n".join(prefix + [core_text])
    return text, token_count(core.text), token_count(text)


def existing_run(dataset: str, seed: int, method: str) -> Path:
    run_dir = RESULTS / "neural" / "synthetic"
    matches = sorted(run_dir.glob(f"equation_audit_{dataset}_{method}_seed{seed}_*.json"))
    if not matches:
        raise FileNotFoundError(f"Missing equation audit run for {dataset} seed={seed} method={method}")
    return matches[-1]


def load_synthetic_split(dataset: str, seed: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "data" / "synthetic" / dataset / f"seed_{seed}" / f"{split}.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "y" not in reader.fieldnames:
            raise ValueError(f"Expected columns including `y` in {path}")
        x_columns = [column for column in reader.fieldnames if column != "y"]
        x_rows: list[list[float]] = []
        y_rows: list[list[float]] = []
        for row in reader:
            x_rows.append([float(row[column]) for column in x_columns])
            y_rows.append([float(row["y"])])
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def evaluate_run(dataset: str, seed: int, method: str, split: str, tolerance: float = TOLERANCE) -> dict:
    json_path = existing_run(dataset, seed, method)
    model, payload = load_model_from_json(json_path)
    x_raw, _ = load_synthetic_split(dataset, seed, split)
    text, core_tokens, deployment_tokens = deployment_graph_text(model, method, payload)
    formula_path = RESULTS / "full_deployment_formulas" / f"{dataset}_{method}_seed{seed}.txt"
    if split == "test":
        formula_path.parent.mkdir(parents=True, exist_ok=True)
        formula_path.write_text(text, encoding="utf-8")
    checkpoint_pred = torch_predict(model, payload, x_raw)
    export_pred = evaluate_assignment_graph(text, x_raw)
    summary = summarize_errors(checkpoint_pred, export_pred)
    return {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "split": split,
        "dtype": "float64 numpy evaluator over float32-trained checkpoint",
        "checkpoint_device": "cpu",
        "tolerance": tolerance,
        "core_tokens": core_tokens,
        "deployment_tokens": deployment_tokens,
        "deployment_token_delta": deployment_tokens - core_tokens,
        "n_total": summary["n_total"],
        "n_finite": summary["n_finite"],
        "mse": summary["mse"],
        "mae": summary["mae"],
        "max_abs": summary["max_abs"],
        "status": status_from_tolerance(summary["max_abs"], tolerance),
        "json_file": str(json_path.relative_to(ROOT)),
        "formula_file": str(formula_path.relative_to(ROOT)),
    }


def read_equation_metric_keys() -> list[tuple[str, int, str]]:
    path = RESULTS / "equation_metrics.csv"
    keys: list[tuple[str, int, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if row.get("status") == "ok" and is_exact_neural_export(method):
                keys.append((row["dataset"], int(row["seed"]), method))
    return keys


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])


def mean_finite(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return mean(clean) if clean else math.nan


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["split"])].append(row)
    summary_rows: list[dict] = []
    for method in METHOD_ORDER:
        for split in SPLIT_ORDER:
            group = grouped.get((method, split), [])
            if not group:
                continue
            statuses = defaultdict(int)
            for row in group:
                statuses[row["status"]] += 1
            summary_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "split": split,
                    "runs": len(group),
                    "eval_n_mean": mean_finite([row["n_finite"] for row in group]),
                    "mse_mean": mean_finite([row["mse"] for row in group]),
                    "mae_mean": mean_finite([row["mae"] for row in group]),
                    "max_abs_mean": mean_finite([row["max_abs"] for row in group]),
                    "max_abs_max": max(float(row["max_abs"]) for row in group),
                    "core_tokens_mean": mean_finite([row["core_tokens"] for row in group]),
                    "deployment_tokens_mean": mean_finite([row["deployment_tokens"] for row in group]),
                    "deployment_token_delta_mean": mean_finite([row["deployment_token_delta"] for row in group]),
                    "tolerance": group[0]["tolerance"],
                    "status_counts": "; ".join(f"{key}={value}" for key, value in sorted(statuses.items())),
                }
            )
    return summary_rows


def fmt_sci(value: float, digits: int = 2) -> str:
    if not math.isfinite(float(value)):
        return "--"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        mantissa, exp = f"{value:.{digits}e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exp)}}}"
    return f"{value:.{digits + 1}g}"


def write_latex(summary: list[dict]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Export-fidelity precision and deployment-token audit for exact neural exporters. Fidelity is measured by evaluating the serialized full-deployment assignment DAG, including input standardization and target inverse-standardization, against the trained checkpoint on the same split. Values are means over 10 equation-native functions unless noted; Max abs reports the worst run in the group.}",
        r"\label{tab:export-fidelity-details}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Method & Split & $n$ & MSE $\downarrow$ & MAE $\downarrow$ & Max abs $\downarrow$ & Core tok. & Full tok. \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{row['method_label']} & {row['split']} & {row['eval_n_mean']:.0f} & "
            rf"${fmt_sci(row['mse_mean'])}$ & ${fmt_sci(row['mae_mean'])}$ & "
            rf"${fmt_sci(row['max_abs_max'])}$ & {row['core_tokens_mean']:.1f} & "
            f"{row['deployment_tokens_mean']:.1f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\par\smallskip\footnotesize All rows use a CPU NumPy float64 evaluator over float32-trained checkpoints; tolerance is $10^{-6}$ max absolute error. StableEML is excluded because its LayerNorm-style block export is summarized rather than exact; PySR is excluded because it is not a neural checkpoint export.",
        r"\end{table*}",
        "",
    ]
    (PAPER / "table_export_fidelity_details.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows: list[dict] = []
    for dataset, seed, method in read_equation_metric_keys():
        for split in SPLIT_ORDER:
            rows.append(evaluate_run(dataset, seed, method, split))
    detail_fields = [
        "dataset",
        "seed",
        "method",
        "method_label",
        "split",
        "dtype",
        "checkpoint_device",
        "tolerance",
        "core_tokens",
        "deployment_tokens",
        "deployment_token_delta",
        "n_total",
        "n_finite",
        "mse",
        "mae",
        "max_abs",
        "status",
        "json_file",
        "formula_file",
    ]
    write_csv(RESULTS / "export_fidelity_details.csv", rows, detail_fields)
    summary = summarize_rows(rows)
    summary_fields = [
        "method",
        "method_label",
        "split",
        "runs",
        "eval_n_mean",
        "mse_mean",
        "mae_mean",
        "max_abs_mean",
        "max_abs_max",
        "core_tokens_mean",
        "deployment_tokens_mean",
        "deployment_token_delta_mean",
        "tolerance",
        "status_counts",
    ]
    write_csv(RESULTS / "export_fidelity_details_summary.csv", summary, summary_fields)
    write_latex(summary)
    print(f"wrote {len(rows)} fidelity detail rows and {len(summary)} summary rows")


if __name__ == "__main__":
    main()
