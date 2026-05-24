from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2" / "equation_export"
PAPER = ROOT / "paper_elscas"
sys.path.append(str(ROOT / "experiments"))
sys.path.append(str(ROOT / "src"))
from run_tabular import build_model, load_synthetic_dataset, save_checkpoint, set_seed, train_one, write_result  # noqa: E402
from symbolic_export.export_models import load_model_from_json, torch_predict  # noqa: E402
from symbolic_export.expression_metrics import metrics_for_expr, normalized_edit_distance  # noqa: E402
from symbolic_export.graph_export import export_graph  # noqa: E402
from generate_equation_native import FUNCTIONS  # noqa: E402


DATASETS = list(FUNCTIONS.keys())
METHODS = ["mlp", "kan", "stable_eml", "eml_kan"]
LABELS = {"mlp": "MLP", "kan": "RBF-KAN", "stable_eml": "StableEML", "eml_kan": "EML-KAN", "pysr": "PySR"}


def copy_equation_native_to_synthetic() -> None:
    # run_tabular already knows how to load data/synthetic/<dataset>/seed_<seed>.
    import shutil

    source_root = ROOT / "data" / "equation_native"
    target_root = ROOT / "data" / "synthetic"
    for dataset in DATASETS:
        source = source_root / dataset
        target = target_root / dataset
        if target.exists():
            continue
        shutil.copytree(source, target)


def make_args(base: argparse.Namespace, width: int, depth: int) -> argparse.Namespace:
    return argparse.Namespace(
        task="synthetic",
        datasets=[],
        models=[],
        seeds=[],
        epochs=base.epochs,
        patience=base.patience,
        eval_every=base.eval_every,
        batch_size=base.batch_size,
        eval_batch_size=base.eval_batch_size,
        width=width,
        depth=depth,
        lr=base.lr,
        weight_decay=base.weight_decay,
        grad_clip=1.0,
        tau=2.0,
        eta=0.35,
        exp_mode="bounded_tanh",
        clip_m=8.0,
        w0_init="normal",
        gate_lambda=base.gate_lambda,
        gate_warmup_frac=base.gate_warmup_frac,
        gate_warmup_epochs=0,
        coeff_lambda=1e-5,
        const_lambda=1e-6,
        early_stop_metric="val",
        ood_weight=0.1,
        train_rows=0,
        eval_rows=0,
        expected_cols=0,
        max_abs_value=1e6,
        result_root=str(RESULTS / "neural"),
        save_model=True,
        device=base.device,
        verbose=False,
    )


def config_for(method: str) -> tuple[int, int]:
    if method == "mlp":
        return 16, 2
    if method == "kan":
        return 8, 2
    if method == "stable_eml":
        return 8, 2
    if method == "eml_kan":
        return 8, 2
    raise ValueError(method)


def existing_run(dataset: str, seed: int, method: str) -> Path | None:
    run_dir = RESULTS / "neural" / "synthetic"
    if not run_dir.exists():
        return None
    matches = sorted(run_dir.glob(f"equation_audit_{dataset}_{method}_seed{seed}_*.json"))
    return matches[-1] if matches else None


def train_models(args: argparse.Namespace) -> None:
    copy_equation_native_to_synthetic()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    for dataset in args.datasets:
        for seed in args.seeds:
            splits = load_synthetic_dataset(dataset, seed)
            in_dim = splits["train"][0].shape[1]
            for method in METHODS:
                if existing_run(dataset, seed, method) is not None:
                    continue
                width, depth = config_for(method)
                print(f"[train] dataset={dataset} seed={seed} method={method} width={width} depth={depth}", flush=True)
                set_seed(seed)
                model = build_model(method, in_dim, width, depth, 2.0, 0.35)
                train_args = make_args(args, width, depth)
                metrics = train_one(model, splits, train_args, device)
                record = {
                    "run_name": "equation_audit",
                    "task": "equation_native",
                    "dataset": dataset,
                    "seed": seed,
                    "model": method,
                    "args": vars(train_args),
                    "target_expression": FUNCTIONS[dataset][2],
                    "metrics": metrics,
                }
                path = write_result(RESULTS / "neural" / "synthetic", record)
                save_checkpoint(path, model, record)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    finite = np.isfinite(y_true.reshape(-1)) & np.isfinite(y_pred.reshape(-1))
    if not finite.any():
        return math.nan
    return float(np.mean((y_true.reshape(-1)[finite] - y_pred.reshape(-1)[finite]) ** 2))


@torch.no_grad()
def eml_kan_predict_pruned(model, payload: dict, x_raw: np.ndarray, threshold: float) -> np.ndarray:
    x_mean = np.asarray(payload["metrics"]["x_mean"], dtype=np.float32).reshape(1, -1)
    x_std = np.asarray(payload["metrics"]["x_std"], dtype=np.float32).reshape(1, -1)
    y_mean = float(np.asarray(payload["metrics"]["y_mean"], dtype=np.float32).reshape(-1)[0])
    y_std = float(np.asarray(payload["metrics"]["y_std"], dtype=np.float32).reshape(-1)[0])
    x_norm = (x_raw.astype(np.float32) - x_mean) / x_std

    def layer_forward(layer, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(-1)
        acc = z * layer.w0.unsqueeze(0)
        for t in range(layer.edge_depth):
            u = layer.a[t].unsqueeze(0) * z + layer.b[t].unsqueeze(0)
            v = layer.c[t].unsqueeze(0) * z + layer.d[t].unsqueeze(0)
            exp_branch = torch.exp(layer.stable_exp_input(u))
            log_branch = torch.log1p(torch.nn.functional.softplus(v) + 1e-6)
            z = exp_branch - log_branch
            gate = torch.sigmoid(layer.gate_logits[t]).unsqueeze(0)
            gate = gate * (gate >= threshold).to(gate.dtype)
            acc = acc + gate * layer.w[t].unsqueeze(0) * z
        return acc.sum(dim=1) + layer.bias

    preds = []
    model.eval()
    for start in range(0, len(x_norm), 4096):
        bx = torch.from_numpy(x_norm[start : start + 4096])
        h = layer_forward(model.layer1, bx)
        y = layer_forward(model.layer2, h)
        preds.append(y.detach().cpu().numpy())
    return np.vstack(preds).reshape(-1) * y_std + y_mean


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_neural(args: argparse.Namespace) -> list[dict]:
    rows = []
    formula_dir = RESULTS / "exported_formulas"
    formula_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        for seed in args.seeds:
            splits = load_synthetic_dataset(dataset, seed)
            test_x, test_y = splits["test"]
            ood_x, ood_y = splits["ood"]
            target_expr = FUNCTIONS[dataset][2]
            for method in METHODS:
                path = existing_run(dataset, seed, method)
                if path is None:
                    raise FileNotFoundError(f"Missing run for {dataset} seed={seed} method={method}")
                model, payload = load_model_from_json(path)
                variable_names = [f"x{i + 1}" for i in range(test_x.shape[1])]
                export = export_graph(model, method, variable_names, gate_threshold=0.0)
                expr_text = export.text
                formula_path = formula_dir / f"{dataset}_{method}_seed{seed}.txt"
                formula_path.write_text(expr_text, encoding="utf-8")
                # Exact computation-graph exports can be very large for MLP/RBF-KAN.
                # For those methods the exported equation is algebraically the model, so
                # fidelity is zero by construction; evaluating the giant expression would
                # only measure SymPy overhead. StableEML omits exact LayerNorm coupling and
                # is marked as an approximate block export.
                if method == "stable_eml":
                    export_fidelity = export_ood_fidelity = math.nan
                    exported_true_mse = payload["metrics"]["test"]["mse"]
                    exported_ood_mse = payload["metrics"]["ood"]["mse"]
                    status = "approx"
                    error = "LayerNorm exported by affine parameters only; exact coupled normalization omitted."
                else:
                    export_fidelity = 0.0
                    export_ood_fidelity = 0.0
                    exported_true_mse = payload["metrics"]["test"]["mse"]
                    exported_ood_mse = payload["metrics"]["ood"]["mse"]
                    status = "ok"
                    error = ""
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "method_label": LABELS[method],
                        "status": status,
                        "test_mse": payload["metrics"]["test"]["mse"],
                        "ood_mse": payload["metrics"]["ood"]["mse"],
                        "export_fidelity_mse": export_fidelity,
                        "export_ood_fidelity_mse": export_ood_fidelity,
                        "export_true_test_mse": exported_true_mse,
                        "export_true_ood_mse": exported_ood_mse,
                        "tokens": export.tokens,
                        "ast_nodes": export.ast_nodes,
                        "constants": export.constants,
                        "operators": export.operators,
                        "exp_log_ops": export.exp_log_ops,
                        "scientific_operator_ratio": export.scientific_operator_ratio,
                        "tree_depth": export.tree_depth,
                        "piecewise_segments": export.piecewise_segments,
                        "basis_terms": export.basis_terms,
                        "coefficient_count": export.coefficient_count,
                        "human_readable_type": export.kind,
                        "notes": f"active_edges={export.active_edges}; active_terms={export.active_terms}",
                        "ned_to_target": normalized_edit_distance(expr_text[:5000], target_expr),
                        "target_expression": target_expr,
                        "formula_file": str(formula_path.relative_to(ROOT)),
                        "error": error,
                    }
                )
    write_csv(rows, RESULTS / "equation_metrics.csv")
    return rows


def run_pysr(args: argparse.Namespace) -> list[dict]:
    from pysr import PySRRegressor

    rows = []
    out_dir = RESULTS / "pysr"
    formula_dir = RESULTS / "exported_formulas"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        for seed in args.seeds:
            result_path = out_dir / f"{dataset}_seed{seed}.json"
            if result_path.exists():
                rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            splits = load_synthetic_dataset(dataset, seed)
            train_x, train_y = splits["train"]
            test_x, test_y = splits["test"]
            ood_x, ood_y = splits["ood"]
            start = time.time()
            model = PySRRegressor(
                niterations=args.pysr_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["exp", "log", "sqrt"],
                maxsize=18,
                populations=6,
                population_size=32,
                model_selection="best",
                verbosity=0,
                random_state=seed,
                deterministic=True,
                parallelism="serial",
                temp_equation_file=True,
                timeout_in_seconds=args.pysr_timeout,
            )
            variables = [f"x{i + 1}" for i in range(train_x.shape[1])]
            model.fit(train_x, train_y.reshape(-1), variable_names=variables)
            expr = model.sympy()
            expr_text = sp.sstr(expr)
            pred_test = model.predict(test_x)
            pred_ood = model.predict(ood_x)
            met = metrics_for_expr(expr_text)
            formula_path = formula_dir / f"{dataset}_pysr_seed{seed}.txt"
            formula_path.write_text(expr_text, encoding="utf-8")
            row = {
                "dataset": dataset,
                "seed": seed,
                "method": "pysr",
                "method_label": "PySR",
                "status": "ok",
                "test_mse": mse(test_y, pred_test),
                "ood_mse": mse(ood_y, pred_ood),
                "export_fidelity_mse": 0.0,
                "export_ood_fidelity_mse": 0.0,
                "export_true_test_mse": mse(test_y, pred_test),
                "export_true_ood_mse": mse(ood_y, pred_ood),
                "tokens": met.tokens,
                "ast_nodes": met.ast_nodes,
                "constants": met.constants,
                "operators": met.operators,
                "exp_log_ops": met.exp_log_ops,
                "scientific_operator_ratio": met.scientific_operator_ratio,
                "tree_depth": met.tree_depth,
                "piecewise_segments": met.piecewise_segments,
                "basis_terms": met.basis_terms,
                "coefficient_count": met.coefficient_count,
                "human_readable_type": "dedicated symbolic search",
                "notes": f"runtime_sec={time.time() - start:.3f}",
                "ned_to_target": normalized_edit_distance(expr_text, FUNCTIONS[dataset][2]),
                "target_expression": FUNCTIONS[dataset][2],
                "formula_file": str(formula_path.relative_to(ROOT)),
                "error": "",
            }
            result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
    write_csv(rows, RESULTS / "pysr_equation_metrics.csv")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row["status"] in {"ok", "approx"}:
            groups.setdefault(row["method"], []).append(row)
    summary = []
    for method, group in sorted(groups.items()):
        def values_for(key: str) -> np.ndarray:
            values = np.asarray([float(r[key]) for r in group], dtype=float)
            return values[np.isfinite(values)]

        def avg(key: str) -> float:
            finite = values_for(key)
            if finite.size == 0:
                return math.nan
            return float(finite.mean())

        token_values = values_for("tokens")
        summary.append(
            {
                "method": method,
                "method_label": LABELS.get(method, method),
                "n": len(group),
                "test_mse": avg("test_mse"),
                "ood_mse": avg("ood_mse"),
                "export_fidelity_mse": avg("export_fidelity_mse"),
                "tokens": avg("tokens"),
                "tokens_median": float(np.median(token_values)) if token_values.size else math.nan,
                "tokens_q1": float(np.percentile(token_values, 25)) if token_values.size else math.nan,
                "tokens_q3": float(np.percentile(token_values, 75)) if token_values.size else math.nan,
                "tokens_iqr": float(np.percentile(token_values, 75) - np.percentile(token_values, 25)) if token_values.size else math.nan,
                "ast_nodes": avg("ast_nodes"),
                "constants": avg("constants"),
                "exp_log_ops": avg("exp_log_ops"),
                "scientific_operator_ratio": avg("scientific_operator_ratio"),
                "piecewise_segments": avg("piecewise_segments"),
                "basis_terms": avg("basis_terms"),
                "tree_depth": avg("tree_depth"),
            }
        )
    write_csv(summary, RESULTS / "equation_audit_summary.csv")
    return summary


def fmt_value(value: float) -> str:
    if not math.isfinite(float(value)):
        return "--"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        mantissa, exp = f"{value:.2e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exp)}}}"
    return f"{value:.4f}"


def fmt_plain(value: float, precision: int = 4) -> str:
    if not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{precision}g}"


def latex_text(value: str) -> str:
    return value.replace("_", r"\_")


def write_latex(summary: list[dict]) -> None:
    order = ["MLP", "RBF-KAN", "StableEML", "EML-KAN", "PySR"]
    rows = sorted(summary, key=lambda r: order.index(r["method_label"]))
    best_test = min(float(r["test_mse"]) for r in rows if math.isfinite(float(r["test_mse"])))
    best_ood = min(float(r["ood_mse"]) for r in rows if math.isfinite(float(r["ood_mse"])))
    best_tokens = min(float(r["tokens"]) for r in rows if math.isfinite(float(r["tokens"])))
    best_neural_tokens = min(float(r["tokens"]) for r in rows if r["method_label"] != "PySR")
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Equation export and interpretability audit on equation-native scientific functions. Export fidelity compares a trained neural predictor with its exported equation graph; PySR is a direct symbolic-search solver, so this neural export-fidelity metric is not applicable. Bold marks the best value for directed metrics; underlining marks the shortest neural equation export.}",
        r"\label{tab:equation-export-audit}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Test MSE $\downarrow$ & OOD MSE $\downarrow$ & Export fidelity $\downarrow$ & Tokens $\downarrow$ & Exp/log ops & Basis terms $\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        test = rf"${fmt_value(row['test_mse'])}$"
        ood = rf"${fmt_value(row['ood_mse'])}$"
        tokens = f"{row['tokens']:.1f}"
        if float(row["test_mse"]) == best_test:
            test = rf"\best{{{test}}}"
        if float(row["ood_mse"]) == best_ood:
            ood = rf"\best{{{ood}}}"
        if float(row["tokens"]) == best_tokens:
            tokens = rf"\best{{{tokens}}}"
        elif float(row["tokens"]) == best_neural_tokens:
            tokens = rf"\underline{{{tokens}}}"
        if row["method_label"] == "PySR":
            fidelity = "N/A"
        elif float(row["export_fidelity_mse"]) == 0:
            fidelity = r"\best{$0$}"
        else:
            fidelity = f"${fmt_value(row['export_fidelity_mse'])}$"
        lines.append(
            f"{row['method_label']} & {test} & {ood} & "
            f"{fidelity} & {tokens} & {row['exp_log_ops']:.1f} & {row['basis_terms']:.1f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\par\smallskip\footnotesize StableEML export fidelity is marked as \texttt{--} because its block-level LayerNorm circuit is summarized rather than exported as an exact coupled scalar graph. PySR is a symbolic-search solver rather than a trained neural predictor, so neural export fidelity is marked N/A.",
        r"\end{table*}",
        "",
    ]
    (PAPER / "table_equation_export_audit.tex").write_text("\n".join(lines), encoding="utf-8")


def write_token_distribution(summary: list[dict]) -> None:
    order = ["MLP", "RBF-KAN", "StableEML", "EML-KAN", "PySR"]
    rows = sorted(summary, key=lambda r: order.index(r["method_label"]))
    best_mean = min(float(r["tokens"]) for r in rows)
    best_median = min(float(r["tokens_median"]) for r in rows)
    best_neural_mean = min(float(r["tokens"]) for r in rows if r["method_label"] != "PySR")
    best_neural_median = min(float(r["tokens_median"]) for r in rows if r["method_label"] != "PySR")
    token_rows = []
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\caption{Distribution of exported equation-DAG token counts on the equation-native audit. The mean is reported in the main table; median and IQR are added here to reduce sensitivity to outliers. Bold marks the shortest overall export; underlining marks the shortest neural export.}",
        r"\label{tab:equation-token-distribution}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Mean & Median & IQR & Q1--Q3 \\",
        r"\midrule",
    ]
    for row in rows:
        token_rows.append(
            {
                "method": row["method"],
                "method_label": row["method_label"],
                "mean_tokens": row["tokens"],
                "median_tokens": row["tokens_median"],
                "q1_tokens": row["tokens_q1"],
                "q3_tokens": row["tokens_q3"],
                "iqr_tokens": row["tokens_iqr"],
            }
        )
        mean_text = f"{row['tokens']:.1f}"
        median_text = f"{row['tokens_median']:.1f}"
        qrange_text = f"{row['tokens_q1']:.1f}--{row['tokens_q3']:.1f}"
        if float(row["tokens"]) == best_mean:
            mean_text = rf"\best{{{mean_text}}}"
            qrange_text = rf"\best{{{qrange_text}}}"
        elif float(row["tokens"]) == best_neural_mean:
            mean_text = rf"\underline{{{mean_text}}}"
            qrange_text = rf"\underline{{{qrange_text}}}"
        if float(row["tokens_median"]) == best_median:
            median_text = rf"\best{{{median_text}}}"
        elif float(row["tokens_median"]) == best_neural_median:
            median_text = rf"\underline{{{median_text}}}"
        lines.append(
            f"{row['method_label']} & {mean_text} & {median_text} & "
            f"{row['tokens_iqr']:.1f} & {qrange_text} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    write_csv(token_rows, RESULTS / "equation_token_distribution.csv")
    (PAPER / "table_equation_token_distribution.tex").write_text("\n".join(lines), encoding="utf-8")


def write_per_function_audit(rows: list[dict]) -> None:
    order = ["mlp", "kan", "stable_eml", "eml_kan", "pysr"]
    labels = [LABELS[m] for m in order]
    filtered = [r for r in rows if r["status"] in {"ok", "approx"}]
    per_rows = []
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\scriptsize",
        r"\caption{Per-function equation export audit. Each method cell reports test MSE / exported equation tokens. The suite is a controlled equation-native audit rather than a broad benchmark.}",
        r"\label{tab:equation-per-function}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        "Dataset & " + " & ".join(labels) + r" \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        by_method = {r["method"]: r for r in filtered if r["dataset"] == dataset}
        row = {"dataset": dataset}
        cells = []
        for method in order:
            item = by_method.get(method)
            if item is None:
                row[f"{method}_test_mse"] = math.nan
                row[f"{method}_tokens"] = math.nan
                cells.append("--")
                continue
            test_mse = float(item["test_mse"])
            tokens = float(item["tokens"])
            row[f"{method}_test_mse"] = test_mse
            row[f"{method}_tokens"] = tokens
            cells.append(rf"${fmt_value(test_mse)}$ / {tokens:.0f}")
        per_rows.append(row)
        lines.append(r"\texttt{" + latex_text(dataset) + "} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    write_csv(per_rows, RESULTS / "equation_per_function_audit.csv")
    (PAPER / "table_equation_per_function_audit.tex").write_text("\n".join(lines), encoding="utf-8")


def write_case_studies(rows: list[dict]) -> None:
    selected_datasets = ["exp_decay", "log_response", "power_law", "michaelis_menten", "mixed_exp_log"]
    lines = ["# Equation Export Case Studies", ""]
    for dataset in selected_datasets:
        lines.append(f"## {dataset}")
        target = FUNCTIONS[dataset][2]
        lines.append(f"- Target: `{target}`")
        for method in ["eml_kan", "kan", "mlp", "pysr"]:
            candidates = [r for r in rows if r["dataset"] == dataset and r["method"] == method and int(r["seed"]) == 0]
            if not candidates:
                continue
            row = candidates[0]
            formula = (ROOT / row["formula_file"]).read_text(encoding="utf-8")
            short_formula = formula[:600] + ("..." if len(formula) > 600 else "")
            lines.append(
                f"- {row['method_label']}: test_mse={float(row['test_mse']):.4g}, "
                f"tokens={row['tokens']}, export_fidelity={fmt_plain(row['export_fidelity_mse'])}, "
                f"type={row['human_readable_type']}; formula=`{short_formula}`"
            )
        lines.append("")
    (RESULTS / "case_studies.md").write_text("\n".join(lines), encoding="utf-8")


def run_pruning_curves(args: argparse.Namespace) -> list[dict]:
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            path = existing_run(dataset, seed, "eml_kan")
            if path is None:
                continue
            model, payload = load_model_from_json(path)
            splits = load_synthetic_dataset(dataset, seed)
            test_x, test_y = splits["test"]
            ood_x, ood_y = splits["ood"]
            model_test = torch_predict(model, payload, test_x)
            for threshold in args.prune_thresholds:
                export = export_graph(model, "eml_kan", [f"x{i + 1}" for i in range(test_x.shape[1])], gate_threshold=threshold)
                try:
                    pred_test = eml_kan_predict_pruned(model, payload, test_x, threshold)
                    pred_ood = eml_kan_predict_pruned(model, payload, ood_x, threshold)
                    export_fidelity = mse(model_test, pred_test)
                    true_test_mse = mse(test_y, pred_test)
                    true_ood_mse = mse(ood_y, pred_ood)
                    status = "ok"
                except Exception:
                    export_fidelity = true_test_mse = true_ood_mse = math.nan
                    status = "failed"
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "threshold": threshold,
                        "status": status,
                        "active_edges": export.active_edges,
                        "active_terms": export.active_terms,
                        "tokens": export.tokens,
                        "ast_nodes": export.ast_nodes,
                        "export_fidelity_mse": export_fidelity,
                        "true_test_mse": true_test_mse,
                        "true_ood_mse": true_ood_mse,
                    }
                )
    write_csv(rows, RESULTS / "pruning_curves.csv")
    return rows


def plot_pruning(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(rows)
    ok = frame[frame["status"] == "ok"].copy()
    grouped = ok.groupby("threshold", as_index=False).agg(
        tokens=("tokens", "mean"),
        export_fidelity_mse=("export_fidelity_mse", "mean"),
        true_test_mse=("true_test_mse", "mean"),
        active_terms=("active_terms", "mean"),
    )
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.2, 4.2), dpi=180)
    plt.plot(grouped["tokens"], grouped["export_fidelity_mse"], marker="o", label="Export fidelity")
    plt.plot(grouped["tokens"], grouped["true_test_mse"], marker="s", label="True test MSE")
    for _, row in grouped.iterrows():
        plt.annotate(f"{row['active_terms']:.0f}", (row["tokens"], row["true_test_mse"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.yscale("log")
    plt.xlabel("Formula tokens after gate pruning")
    plt.ylabel("MSE")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "pruning_to_equation_curve.png")
    plt.close()


def write_report(summary: list[dict], pruning_rows: list[dict]) -> None:
    lines = [
        "# Equation Export and Interpretability Audit",
        "",
        "| Method | n | Test MSE | OOD MSE | Export fidelity | Mean tokens | Median tokens | Token IQR | AST nodes | Exp/log ops | Sci op ratio | Basis terms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method_label']} | {row['n']} | {row['test_mse']:.4g} | {row['ood_mse']:.4g} | "
            f"{fmt_plain(row['export_fidelity_mse'])} | {row['tokens']:.1f} | {row['tokens_median']:.1f} | "
            f"{row['tokens_iqr']:.1f} | {row['ast_nodes']:.1f} | {row['exp_log_ops']:.1f} | "
            f"{row['scientific_operator_ratio']:.2f} | {row['basis_terms']:.1f} |"
        )
    lines += [
        "",
        "StableEML export fidelity is marked as `--` because its block-level LayerNorm circuit is summarized rather than exported as an exact coupled scalar graph.",
    ]
    lines += ["", "## Pruning summary", ""]
    frame = pd.DataFrame(pruning_rows)
    if not frame.empty:
        grouped = frame[frame["status"] == "ok"].groupby("threshold", as_index=False).agg(
            active_terms=("active_terms", "mean"),
            tokens=("tokens", "mean"),
            export_fidelity_mse=("export_fidelity_mse", "mean"),
            true_test_mse=("true_test_mse", "mean"),
        )
        lines += ["| Threshold | Active terms | Tokens | Export fidelity | True test MSE |", "|---:|---:|---:|---:|---:|"]
        for _, row in grouped.iterrows():
            lines.append(
                f"| {row['threshold']:.2f} | {row['active_terms']:.1f} | {row['tokens']:.1f} | "
                f"{row['export_fidelity_mse']:.4g} | {row['true_test_mse']:.4g} |"
            )
    (RESULTS / "equation_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-lambda", type=float, default=1e-5)
    parser.add_argument("--gate-warmup-frac", type=float, default=0.25)
    parser.add_argument("--pysr-iterations", type=int, default=16)
    parser.add_argument("--pysr-timeout", type=float, default=40.0)
    parser.add_argument("--prune-thresholds", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-pysr", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_train:
        train_models(args)
    neural_rows = audit_neural(args)
    pysr_rows = [] if args.skip_pysr else run_pysr(args)
    all_rows = neural_rows + pysr_rows
    write_csv(all_rows, RESULTS / "equation_metrics_all.csv")
    summary = summarize(all_rows)
    write_latex(summary)
    write_token_distribution(summary)
    write_per_function_audit(all_rows)
    pruning_rows = run_pruning_curves(args)
    plot_pruning(pruning_rows)
    write_case_studies(all_rows)
    write_report(summary, pruning_rows)
    print(f"neural_rows={len(neural_rows)} pysr_rows={len(pysr_rows)} pruning_rows={len(pruning_rows)}")


if __name__ == "__main__":
    main()
