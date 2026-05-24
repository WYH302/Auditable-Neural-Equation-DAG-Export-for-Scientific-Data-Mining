from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("data/equation_native")
SEEDS = [0, 1, 2]
N_TRAIN = 1200
N_VAL = 800
N_TEST = 1000
N_OOD = 1000


def sample_box(rng: np.random.Generator, n: int, d: int, low: float, high: float) -> np.ndarray:
    return rng.uniform(low, high, size=(n, d)).astype(np.float64)


def save_split(name: str, seed: int, split: str, x: np.ndarray, y: np.ndarray) -> None:
    columns = [f"x{i + 1}" for i in range(x.shape[1])]
    frame = pd.DataFrame(x, columns=columns)
    frame["y"] = y.astype(np.float64)
    path = OUT_DIR / name / f"seed_{seed}" / f"{split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def make_dataset(name: str, d: int, func, train_range=(0.15, 2.0), ood_range=(0.05, 3.0)) -> None:
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        splits = {
            "train": sample_box(rng, N_TRAIN, d, *train_range),
            "val": sample_box(rng, N_VAL, d, *train_range),
            "test": sample_box(rng, N_TEST, d, *train_range),
            "ood": sample_box(rng, N_OOD, d, *ood_range),
        }
        for split, x in splits.items():
            save_split(name, seed, split, x, func(x))
    print(f"Saved {name}")


FUNCTIONS = {
    "exp_decay": (
        1,
        lambda x: 1.7 * np.exp(-0.8 * x[:, 0]) + 0.25,
        "1.7*exp(-0.8*x1)+0.25",
    ),
    "log_response": (
        1,
        lambda x: 1.2 * np.log(1.5 * x[:, 0] + 0.7) - 0.15,
        "1.2*log(1.5*x1+0.7)-0.15",
    ),
    "power_law": (
        1,
        lambda x: 0.8 * np.exp(1.35 * np.log(x[:, 0] + 0.2)) + 0.1,
        "0.8*exp(1.35*log(x1+0.2))+0.1",
    ),
    "arrhenius_like": (
        1,
        lambda x: 2.0 * np.exp(-1.15 / (x[:, 0] + 0.6)),
        "2.0*exp(-1.15/(x1+0.6))",
    ),
    "michaelis_menten": (
        1,
        lambda x: 1.8 * x[:, 0] / (0.45 + x[:, 0]),
        "1.8*x1/(0.45+x1)",
    ),
    "multiplicative_power": (
        2,
        lambda x: 1.1 * np.exp(0.7 * np.log(x[:, 0] + 0.15) - 0.45 * np.log(x[:, 1] + 0.25)),
        "1.1*exp(0.7*log(x1+0.15)-0.45*log(x2+0.25))",
    ),
    "mixed_exp_log": (
        1,
        lambda x: 0.9 * np.exp(0.55 * x[:, 0]) + 0.65 * np.log(1.2 * x[:, 0] + 0.8),
        "0.9*exp(0.55*x1)+0.65*log(1.2*x1+0.8)",
    ),
    "damped_log_response": (
        1,
        lambda x: 1.4 * np.exp(-0.55 * x[:, 0]) * np.log(1.7 * x[:, 0] + 0.9),
        "1.4*exp(-0.55*x1)*log(1.7*x1+0.9)",
    ),
    "exp_saturation": (
        1,
        lambda x: 1.3 * (1.0 - np.exp(-1.1 * x[:, 0])) + 0.05,
        "1.3*(1-exp(-1.1*x1))+0.05",
    ),
    "two_var_decay": (
        2,
        lambda x: 0.75 * x[:, 0] * np.exp(-0.65 * x[:, 1]) + 0.2,
        "0.75*x1*exp(-0.65*x2)+0.2",
    ),
}


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_name, (dim, function, _) in FUNCTIONS.items():
        make_dataset(dataset_name, dim, function)

