from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_X_y, check_array


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2"
PAPER = ROOT / "paper_elscas"

SYNTHETIC_DATASETS = [
    "cos_plus_x_weak_case",
    "exp_1d",
    "exp_log_1d",
    "exp_log_2d",
    "inverse_quadratic",
    "kinetic_energy",
    "log_1d_x_plus_2",
    "poly_1d_x2_x_1",
    "poly_2d_x2_2y_1",
    "sin_weak_case",
]

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

TARGET_COMPLEXITY = {
    "poly_1d_x2_x_1": 3,
    "poly_2d_x2_2y_1": 3,
    "exp_1d": 1,
    "log_1d_x_plus_2": 1,
    "exp_log_1d": 2,
    "exp_log_2d": 2,
    "kinetic_energy": 3,
    "inverse_quadratic": 1,
    "sin_weak_case": 1,
    "cos_plus_x_weak_case": 2,
}

TRIG_RE = re.compile(r"\b(sin|cos|tan|asin|acos|atan|arcsin|arccos|arctan)\b", re.I)
INTEGRAL_RE = re.compile(r"\bInt_", re.I)


@dataclass
class DatasetBundle:
    benchmark: str
    dataset: str
    seed: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_ood: np.ndarray | None
    y_ood: np.ndarray | None
    variable_names: list[str]
    target_expression: str
    target_complexity: int | None


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite.any():
        return math.nan
    return float(np.mean((y_true[finite] - y_pred[finite]) ** 2))


def rel_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = mse(y_true, y_pred)
    var = float(np.var(y_true.reshape(-1)))
    return value / max(var, 1e-12) if math.isfinite(value) else math.nan


def load_synthetic(dataset: str, seed: int, train_samples: int) -> DatasetBundle:
    base = ROOT / "data" / "synthetic" / dataset / f"seed_{seed}"
    train = pd.read_csv(base / "train.csv")
    if train_samples and len(train) > train_samples:
        train = train.sample(n=train_samples, random_state=seed)
    test = pd.read_csv(base / "test.csv")
    ood = pd.read_csv(base / "ood.csv")

    def xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame.drop(columns=["y"]).to_numpy(dtype=np.float64)
        y = frame["y"].to_numpy(dtype=np.float64)
        return x, y

    x_train, y_train = xy(train)
    x_test, y_test = xy(test)
    x_ood, y_ood = xy(ood)
    return DatasetBundle(
        benchmark="Synthetic symbolic",
        dataset=dataset,
        seed=seed,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_ood=x_ood,
        y_ood=y_ood,
        variable_names=[f"x{i + 1}" for i in range(x_train.shape[1])],
        target_expression=TARGET_EXPR[dataset],
        target_complexity=TARGET_COMPLEXITY[dataset],
    )


def load_feynman_metadata() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "feynman" / "official" / "FeynmanEquations.csv")


def detect_ncols(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return len(stripped.split())
    return 0


def select_feynman_files(limit: int, include_trig: bool) -> list[dict]:
    df = load_feynman_metadata()
    data_dir = ROOT / "data" / "feynman" / "official" / "Feynman_without_units"
    rows = []
    for _, row in df.iterrows():
        filename = str(row["Filename"])
        path = data_dir / filename
        if not path.exists():
            continue
        ncols = detect_ncols(path)
        if ncols < 2:
            continue
        n_vars = ncols - 1
        formula = str(row["Formula"])
        if n_vars > 4:
            continue
        if INTEGRAL_RE.search(formula):
            continue
        if not include_trig and TRIG_RE.search(formula):
            continue
        var_names = []
        for idx in range(1, n_vars + 1):
            value = row.get(f"v{idx}_name")
            var_names.append(f"x{idx}" if pd.isna(value) else str(value))
        rows.append(
            {
                "filename": filename,
                "path": path,
                "formula": formula,
                "n_vars": n_vars,
                "variable_names": var_names,
            }
        )
    rows.sort(key=lambda r: (r["n_vars"], r["filename"]))
    return rows[:limit]


def split_feynman_table(path: Path, seed: int, train_samples: int, test_samples: int, max_abs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.loadtxt(path, dtype=np.float64, max_rows=max(train_samples + test_samples + 2000, 6000))
    if values.ndim == 1:
        values = values.reshape(1, -1)
    finite = np.isfinite(values).all(axis=1)
    bounded = np.max(np.abs(values), axis=1) <= max_abs
    values = values[finite & bounded]
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    train = values[: min(train_samples, len(values) // 2)]
    test_start = len(train)
    test = values[test_start : test_start + min(test_samples, max(1, len(values) - test_start))]
    return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]


def load_feynman(row: dict, seed: int, train_samples: int, test_samples: int, max_abs: float) -> DatasetBundle:
    x_train, y_train, x_test, y_test = split_feynman_table(
        row["path"], seed, train_samples, test_samples, max_abs
    )
    return DatasetBundle(
        benchmark="Feynman lowdim20",
        dataset=row["filename"],
        seed=seed,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_ood=None,
        y_ood=None,
        variable_names=[f"x{i + 1}" for i in range(x_train.shape[1])],
        target_expression=row["formula"],
        target_complexity=None,
    )


def patch_gplearn_validation() -> None:
    from gplearn.genetic import SymbolicRegressor

    def _validate_data(self, X, y=None, y_numeric=False):
        if y is None:
            x_checked = check_array(X)
            self.n_features_in_ = x_checked.shape[1]
            return x_checked
        x_checked, y_checked = check_X_y(X, y, y_numeric=y_numeric)
        self.n_features_in_ = x_checked.shape[1]
        return x_checked, y_checked

    SymbolicRegressor._validate_data = _validate_data


def fit_pysr(bundle: DatasetBundle, args) -> dict:
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=args.pysr_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "sqrt", "exp", "log", "sin", "cos"],
        maxsize=args.maxsize,
        populations=args.pysr_populations,
        population_size=args.pysr_population_size,
        model_selection="best",
        verbosity=0,
        random_state=bundle.seed,
        deterministic=True,
        parallelism="serial",
        temp_equation_file=True,
        timeout_in_seconds=args.pysr_timeout,
    )
    model.fit(bundle.x_train, bundle.y_train, variable_names=bundle.variable_names)
    expression = str(model.sympy())
    complexity = math.nan
    try:
        best = model.get_best()
        complexity = int(best.get("complexity", math.nan))
    except Exception:
        if getattr(model, "equations_", None) is not None and len(model.equations_) > 0:
            complexity = int(model.equations_.iloc[-1].get("complexity", math.nan))
    return {"model": model, "expression": expression, "complexity": complexity}


def protected_exp(x):
    return np.exp(np.clip(x, -10.0, 10.0))


def fit_gplearn(bundle: DatasetBundle, args) -> dict:
    patch_gplearn_validation()
    from gplearn.functions import make_function
    from gplearn.genetic import SymbolicRegressor

    exp_func = make_function(function=protected_exp, name="exp", arity=1)
    model = SymbolicRegressor(
        population_size=args.gp_population_size,
        generations=args.gp_generations,
        tournament_size=20,
        stopping_criteria=1e-12,
        const_range=(-5.0, 5.0),
        init_depth=(2, 6),
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "sin", "cos", exp_func),
        metric="mse",
        parsimony_coefficient=0.003,
        random_state=bundle.seed,
        verbose=0,
        n_jobs=1,
    )
    model.fit(bundle.x_train, bundle.y_train)
    return {
        "model": model,
        "expression": str(model._program),
        "complexity": int(getattr(model._program, "length_", math.nan)),
    }


def run_one(method: str, bundle: DatasetBundle, args) -> dict:
    start = time.time()
    status = "ok"
    error = ""
    fitted = None
    try:
        fitted = fit_pysr(bundle, args) if method == "pysr" else fit_gplearn(bundle, args)
        pred_test = fitted["model"].predict(bundle.x_test)
        pred_ood = fitted["model"].predict(bundle.x_ood) if bundle.x_ood is not None else None
        test_mse = mse(bundle.y_test, pred_test)
        test_rel_mse = rel_mse(bundle.y_test, pred_test)
        ood_mse = mse(bundle.y_ood, pred_ood) if pred_ood is not None else math.nan
        ood_rel_mse = rel_mse(bundle.y_ood, pred_ood) if pred_ood is not None else math.nan
        expression = fitted["expression"]
        complexity = fitted["complexity"]
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        test_mse = test_rel_mse = ood_mse = ood_rel_mse = math.nan
        expression = ""
        complexity = math.nan
    runtime_sec = time.time() - start
    exact_threshold = max(1e-10, 1e-8 * float(np.var(bundle.y_test.reshape(-1))))
    exact = int(math.isfinite(test_mse) and test_mse <= exact_threshold)
    return {
        "benchmark": bundle.benchmark,
        "dataset": bundle.dataset,
        "seed": bundle.seed,
        "method": method,
        "status": status,
        "test_mse": test_mse,
        "test_rel_mse": test_rel_mse,
        "ood_mse": ood_mse,
        "ood_rel_mse": ood_rel_mse,
        "exact_recovery": exact,
        "complexity": complexity,
        "runtime_sec": runtime_sec,
        "expression": expression,
        "target_expression": bundle.target_expression,
        "target_complexity": bundle.target_complexity,
        "error": error,
        "n_train": len(bundle.x_train),
        "n_test": len(bundle.x_test),
        "n_ood": 0 if bundle.x_ood is None else len(bundle.x_ood),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_value(value: float) -> str:
    if value in (None, ""):
        return "--"
    value = float(value)
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        mantissa, exp = f"{value:.2e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exp)}}}"
    return f"{value:.4f}"


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["benchmark"], row["method"]), []).append(row)
    summary = []
    for (benchmark, method), group in sorted(groups.items()):
        ok = [r for r in group if r["status"] == "ok"]
        values = lambda key: [float(r[key]) for r in ok if math.isfinite(float(r[key]))]
        test = values("test_mse")
        ood = values("ood_mse")
        comp = values("complexity")
        runtime = values("runtime_sec")
        summary.append(
            {
                "benchmark": benchmark,
                "method": method,
                "n": len(group),
                "ok": len(ok),
                "exact_count": sum(int(r["exact_recovery"]) for r in ok),
                "test_mse_mean": float(np.mean(test)) if test else math.nan,
                "ood_mse_mean": float(np.mean(ood)) if ood else math.nan,
                "complexity_mean": float(np.mean(comp)) if comp else math.nan,
                "runtime_sec_mean": float(np.mean(runtime)) if runtime else math.nan,
                "failed": len(group) - len(ok),
            }
        )
    return summary


def write_latex(summary: list[dict], path: Path) -> None:
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{External symbolic-regression baselines. PySR is the primary solver baseline; GP denotes a gplearn genetic-programming baseline. Feynman has no separate OOD split in this protocol.}",
        r"\label{tab:external-sr}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Benchmark & Method & Runs & Exact $\uparrow$ & Test MSE $\downarrow$ & OOD MSE $\downarrow$ & Complexity $\downarrow$ & Runtime sec $\downarrow$ \\",
        r"\midrule",
    ]
    labels = {"pysr": "PySR", "gplearn": "GP"}
    for row in summary:
        lines.append(
            f"{row['benchmark']} & {labels.get(row['method'], row['method'])} & "
            f"{row['ok']}/{row['n']} & {row['exact_count']}/{row['ok']} & "
            f"${fmt_value(row['test_mse_mean'])}$ & ${fmt_value(row['ood_mse_mean'])}$ & "
            f"{fmt_value(row['complexity_mean'])} & {fmt_value(row['runtime_sec_mean'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(summary: list[dict], rows: list[dict], path: Path) -> None:
    lines = [
        "# External Symbolic Regression Baselines",
        "",
        "PySR is run through its Julia backend installed by juliacall. gplearn is included as a classical GP baseline.",
        "",
        "| Benchmark | Method | OK/Total | Exact | Test MSE | OOD MSE | Complexity | Runtime sec | Failed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['benchmark']} | {row['method']} | {row['ok']}/{row['n']} | "
            f"{row['exact_count']}/{row['ok']} | {row['test_mse_mean']:.4g} | "
            f"{row['ood_mse_mean']:.4g} | {row['complexity_mean']:.2f} | "
            f"{row['runtime_sec_mean']:.2f} | {row['failed']} |"
        )
    lines += ["", "## Selected expressions", ""]
    for row in rows[:80]:
        expr = row["expression"][:220].replace("\n", " ")
        lines.append(
            f"- {row['benchmark']} / {row['dataset']} / {row['method']}: "
            f"test_mse={row['test_mse']:.4g}; expr=`{expr}`; target=`{row['target_expression']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aifeynman_status(path: Path) -> None:
    gfortran_found = False
    try:
        import shutil

        gfortran_found = shutil.which("gfortran") is not None
    except Exception:
        pass
    status = [
        "# AI-Feynman Environment Status",
        "",
        "- Local source is present at `data/feynman/AI-Feynman`.",
        "- The upstream README states that the package is supported only on Linux and Mac environments.",
        f"- `gfortran` available on PATH: {gfortran_found}.",
        "- Because this workspace is Windows and no Fortran compiler is available, AI-Feynman was not executed here.",
        "- PySR and gplearn are therefore used as executable external symbolic-regression baselines in the current draft.",
    ]
    path.write_text("\n".join(status) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["pysr", "gplearn"], choices=["pysr", "gplearn"])
    parser.add_argument("--benchmarks", nargs="+", default=["synthetic", "feynman"], choices=["synthetic", "feynman"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--max-feynman-files", type=int, default=20)
    parser.add_argument("--include-trig-feynman", action="store_true")
    parser.add_argument("--max-abs-value", type=float, default=1e6)
    parser.add_argument("--pysr-iterations", type=int, default=12)
    parser.add_argument("--pysr-timeout", type=float, default=45.0)
    parser.add_argument("--pysr-populations", type=int, default=6)
    parser.add_argument("--pysr-population-size", type=int, default=32)
    parser.add_argument("--maxsize", type=int, default=18)
    parser.add_argument("--gp-population-size", type=int, default=600)
    parser.add_argument("--gp-generations", type=int, default=12)
    parser.add_argument("--out-dir", default="results_v2/external_symbolic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    rows: list[dict] = []
    for seed in args.seeds:
        bundles: list[DatasetBundle] = []
        if "synthetic" in args.benchmarks:
            bundles.extend(load_synthetic(dataset, seed, args.train_samples) for dataset in SYNTHETIC_DATASETS)
        if "feynman" in args.benchmarks:
            for row in select_feynman_files(args.max_feynman_files, args.include_trig_feynman):
                bundles.append(load_feynman(row, seed, args.train_samples, args.test_samples, args.max_abs_value))
        for bundle in bundles:
            for method in args.methods:
                print(
                    f"[external-sr] benchmark={bundle.benchmark} dataset={bundle.dataset} "
                    f"seed={seed} method={method}",
                    flush=True,
                )
                row = run_one(method, bundle, args)
                rows.append(row)
                write_csv(rows, out_dir / "external_symbolic_runs.csv")
    summary = summarize(rows)
    write_csv(summary, out_dir / "external_symbolic_summary.csv")
    write_latex(summary, PAPER / "table_external_symbolic.tex")
    write_report(summary, rows, out_dir / "external_symbolic_report.md")
    write_aifeynman_status(out_dir / "aifeynman_status_20260509.md")
    print(f"runs={len(rows)} summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
