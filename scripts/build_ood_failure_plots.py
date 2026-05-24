from __future__ import annotations

import csv
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.symbolic_export.export_models import load_model_from_json, torch_predict


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DATA = ROOT / "data" / "synthetic"
RESULTS = ROOT / "results_v2" / "synthetic_final"
RUNS = RESULTS / "synthetic"
FIGURES = RESULTS / "figures"
PAPER_FIGURES = PROJECT_ROOT / "paper" / "paper_cikm2026" / "figures"
PAPER = PROJECT_ROOT / "paper" / "paper_cikm2026"

CASES = ["inverse_quadratic", "cos_plus_x_weak_case", "poly_1d_x2_x_1"]
METHODS = {"mlp": "MLP", "eml_kan": "EML-KAN"}


def true_function(dataset: str, x: np.ndarray) -> np.ndarray:
    x1 = x.reshape(-1)
    if dataset == "inverse_quadratic":
        return 1.0 / (x1**2 + 1.0)
    if dataset == "cos_plus_x_weak_case":
        return np.cos(x1) + x1
    if dataset == "poly_1d_x2_x_1":
        return x1**2 + x1 + 1.0
    raise KeyError(f"No plotting true function registered for {dataset}")


def latest_result(files) -> Path:
    paths = [Path(file) for file in files]
    if not paths:
        raise FileNotFoundError("No matching result files")

    def key(path: Path) -> tuple[str, str]:
        match = re.search(r"_(\d{8}_\d{6})\.json$", path.name)
        return (match.group(1) if match else "", path.name)

    return max(paths, key=key)


def input_columns(headers: list[str]) -> list[str]:
    return [header for header in headers if header != "y"]


def split_path(dataset: str, seed: int, split: str) -> Path:
    return DATA / dataset / f"seed_{seed}" / f"{split}.csv"


def is_one_dimensional_case(dataset: str, seed: int = 0) -> bool:
    with split_path(dataset, seed, "train").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    return len(input_columns(header)) == 1


def load_split(dataset: str, seed: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(split_path(dataset, seed, split))
    x_cols = input_columns(list(frame.columns))
    x = frame[x_cols].to_numpy(dtype=np.float32)
    y = frame["y"].to_numpy(dtype=np.float32)
    return x, y


def result_json(dataset: str, method: str, seed: int) -> Path:
    pattern = f"synthetic_{dataset}_{method}_seed{seed}_*.json"
    return latest_result(RUNS.glob(pattern))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite.any():
        return math.nan
    return float(np.mean((y_true[finite] - y_pred[finite]) ** 2))


def case_payload(dataset: str, seed: int = 0) -> dict:
    if not is_one_dimensional_case(dataset, seed):
        raise ValueError(f"{dataset} is not one-dimensional")
    splits = {name: load_split(dataset, seed, name) for name in ["train", "test", "ood"]}
    x_all = np.concatenate([splits[name][0][:, 0] for name in ["train", "test", "ood"]])
    pad = 0.05 * max(1e-6, float(x_all.max() - x_all.min()))
    x_grid = np.linspace(float(x_all.min() - pad), float(x_all.max() + pad), 800, dtype=np.float32).reshape(-1, 1)
    payload = {
        "dataset": dataset,
        "seed": seed,
        "splits": splits,
        "grid_x": x_grid,
        "true_grid": true_function(dataset, x_grid[:, 0]),
        "predictions": {},
        "metrics": {},
    }
    for method in METHODS:
        model, model_payload = load_model_from_json(result_json(dataset, method, seed))
        payload["predictions"][method] = torch_predict(model, model_payload, x_grid)
        payload["metrics"][method] = {}
        for split, (x, y) in splits.items():
            pred = torch_predict(model, model_payload, x)
            payload["metrics"][method][split] = mse(y, pred)
    return payload


def format_title(dataset: str) -> str:
    return dataset.replace("_", " ")


def plot_cases(cases: list[str] = CASES, seed: int = 0) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict] = []
    fig, axes = plt.subplots(1, len(cases), figsize=(12.0, 3.1), dpi=220, sharey=False)
    if len(cases) == 1:
        axes = [axes]
    colors = {"mlp": "#396AB1", "eml_kan": "#E68310"}
    for ax, dataset in zip(axes, cases):
        payload = case_payload(dataset, seed)
        train_x, train_y = payload["splits"]["train"]
        test_x, test_y = payload["splits"]["test"]
        ood_x, ood_y = payload["splits"]["ood"]
        grid_x = payload["grid_x"][:, 0]
        train_order = np.argsort(train_x[:, 0])
        test_order = np.argsort(test_x[:, 0])
        ood_order = np.argsort(ood_x[:, 0])
        ax.scatter(train_x[train_order[:400], 0], train_y[train_order[:400]], s=6, color="#B0B0B0", alpha=0.25, label="train")
        ax.scatter(test_x[test_order[:: max(1, len(test_order) // 160)], 0], test_y[test_order[:: max(1, len(test_order) // 160)]], s=7, color="#333333", alpha=0.35, label="test")
        ax.scatter(ood_x[ood_order[:: max(1, len(ood_order) // 160)], 0], ood_y[ood_order[:: max(1, len(ood_order) // 160)]], s=7, color="#7F3C8D", alpha=0.45, label="OOD")
        ax.plot(grid_x, payload["true_grid"], color="black", linewidth=1.8, label="target")
        for method, label in METHODS.items():
            ax.plot(grid_x, payload["predictions"][method], color=colors[method], linewidth=1.5, linestyle="--" if method == "mlp" else "-", label=label)
        mlp_ood = payload["metrics"]["mlp"]["ood"]
        eml_ood = payload["metrics"]["eml_kan"]["ood"]
        ratio = eml_ood / mlp_ood if mlp_ood > 0 else math.nan
        ax.set_title(format_title(dataset), fontsize=9)
        ax.set_xlabel("$x_1$")
        ax.text(0.02, 0.96, f"OOD ratio {ratio:.1f}x", transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.grid(alpha=0.18, linewidth=0.5)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "mlp_test_mse": payload["metrics"]["mlp"]["test"],
                "eml_kan_test_mse": payload["metrics"]["eml_kan"]["test"],
                "mlp_ood_mse": mlp_ood,
                "eml_kan_ood_mse": eml_ood,
                "eml_over_mlp_ood": ratio,
            }
        )
    axes[0].set_ylabel("$y$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "ood_failure_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(out, PAPER_FIGURES / out.name)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["dataset", "seed", "mlp_test_mse", "eml_kan_test_mse", "mlp_ood_mse", "eml_kan_ood_mse", "eml_over_mlp_ood"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def render_latex_figure() -> str:
    return "\n".join(
        [
            r"\begin{figure*}[t]",
            r"\centering",
            r"\includegraphics[width=\textwidth]{ood_failure_curves.png}",
            r"\caption{Representative one-dimensional synthetic OOD failure curves. Points show train, test, and OOD samples; solid black is the target function, dashed blue is MLP, and orange is EML-KAN. These cases are selected from the largest dataset-level OOD ratios recorded in the supplementary failure-case CSV; they illustrate that shorter checkpoint DAGs and favorable interpolation do not imply safe extrapolation.}",
            r"\label{fig:ood-failure-curves}",
            r"\Description{Three line plots compare target, MLP, and EML-KAN predictions on train, test, and OOD intervals for inverse-quadratic, cosine-plus-linear, and polynomial synthetic functions.}",
            r"\end{figure*}",
            "",
        ]
    )


def main() -> None:
    rows = plot_cases(CASES, seed=0)
    write_csv(RESULTS / "ood_failure_curve_metrics.csv", rows)
    (PAPER / "figure_ood_failure_curves.tex").write_text(render_latex_figure(), encoding="utf-8")
    print(f"ood failure curves cases={len(rows)} output={FIGURES / 'ood_failure_curves.png'}")


if __name__ == "__main__":
    main()
