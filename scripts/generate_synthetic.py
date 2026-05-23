from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("data/synthetic")
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN = 1000
N_VAL = 1000
N_TEST = 1000
N_OOD = 1000


def sample_uniform(rng, n, d, low, high):
    return rng.uniform(low, high, size=(n, d))


def save_split(name, seed, split, x, y):
    columns = [f"x{i + 1}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=columns)
    df["y"] = y
    path = OUT_DIR / name / f"seed_{seed}" / f"{split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_dataset(name, d, func):
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        splits = {
            "train": sample_uniform(rng, N_TRAIN, d, -1, 1),
            "val": sample_uniform(rng, N_VAL, d, -1, 1),
            "test": sample_uniform(rng, N_TEST, d, -1, 1),
            "ood": sample_uniform(rng, N_OOD, d, -2, 2),
        }
        for split, x in splits.items():
            save_split(name, seed, split, x, func(x))
    print(f"Saved {name}")


FUNCTIONS = {
    "poly_1d_x2_x_1": (1, lambda x: x[:, 0] ** 2 + x[:, 0] + 1),
    "poly_2d_x2_2y_1": (2, lambda x: x[:, 0] ** 2 + 2 * x[:, 1] + 1),
    "exp_1d": (1, lambda x: np.exp(x[:, 0])),
    "log_1d_x_plus_2": (1, lambda x: np.log(x[:, 0] + 2.1)),
    "exp_log_1d": (1, lambda x: np.exp(x[:, 0]) - np.log(x[:, 0] + 2.1)),
    "exp_log_2d": (2, lambda x: np.exp(x[:, 0]) - np.log(x[:, 1] + 2.1)),
    "kinetic_energy": (2, lambda x: 0.5 * (x[:, 0] + 2.1) * x[:, 1] ** 2),
    "inverse_quadratic": (1, lambda x: 1.0 / (x[:, 0] ** 2 + 1.0)),
    "sin_weak_case": (1, lambda x: np.sin(x[:, 0])),
    "cos_plus_x_weak_case": (1, lambda x: np.cos(x[:, 0]) + x[:, 0]),
}


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_name, (dim, function) in FUNCTIONS.items():
        make_dataset(dataset_name, dim, function)
