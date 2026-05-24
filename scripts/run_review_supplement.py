from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2" / "review_supplement"
sys.path.append(str(ROOT / "experiments"))
sys.path.append(str(ROOT / "scripts"))
from generate_equation_native import FUNCTIONS  # noqa: E402
from run_tabular import build_model, load_synthetic_dataset, save_checkpoint, set_seed, train_one, write_result  # noqa: E402


DEFAULT_NOISE_DATASETS = [
    "exp_decay",
    "log_response",
    "mixed_exp_log",
    "power_law",
    "michaelis_menten",
]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_NOISE_LEVELS = [0.01, 0.05, 0.10]
DEFAULT_NOISE_METHODS = ["mlp", "kan", "stable_eml", "eml_kan"]


def add_label_noise(y: list[float] | np.ndarray, noise_frac: float, seed: int) -> np.ndarray:
    values = np.asarray(y, dtype=np.float32).copy()
    if noise_frac <= 0:
        return values
    scale = float(np.std(values))
    if not math.isfinite(scale) or scale <= 0:
        return values
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=values.shape).astype(np.float32)
    noise_std = float(np.std(noise))
    if noise_std <= 0 or not math.isfinite(noise_std):
        return values
    noise = noise / noise_std * (noise_frac * scale)
    return values + noise.astype(np.float32)


def config_for(method: str, large_mlp: bool = False) -> tuple[int, int]:
    if method == "mlp":
        return (64, 3) if large_mlp else (16, 2)
    if method in {"kan", "stable_eml", "eml_kan"}:
        return 8, 2
    raise ValueError(f"Unknown method: {method}")


def make_train_args(base: argparse.Namespace, width: int, depth: int, result_root: Path) -> argparse.Namespace:
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
        edge_variant="full",
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
        result_root=str(result_root),
        save_model=True,
        device=base.device,
        verbose=base.verbose,
    )


def device_from_arg(name: str) -> torch.device:
    device = torch.device(name if name == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def existing_json(out_dir: Path, run_name: str, dataset: str, method: str, seed: int, suffix: str = "") -> Path | None:
    pattern = f"{run_name}{suffix}_{dataset}_{method}_seed{seed}_*.json"
    matches = sorted(out_dir.glob(pattern))
    return matches[-1] if matches else None


def run_mlp_large(args: argparse.Namespace) -> None:
    out_dir = RESULTS / "mlp_large" / "equation_native"
    device = device_from_arg(args.device)
    datasets = args.mlp_large_datasets or list(FUNCTIONS.keys())
    for dataset in datasets:
        for seed in args.seeds:
            if existing_json(out_dir, "review_mlp_large", dataset, "mlp", seed):
                print(f"[skip mlp-large] dataset={dataset} seed={seed}", flush=True)
                continue
            splits = load_synthetic_dataset(dataset, seed)
            in_dim = splits["train"][0].shape[1]
            width, depth = config_for("mlp", large_mlp=True)
            print(f"[mlp-large] dataset={dataset} seed={seed} width={width} depth={depth}", flush=True)
            set_seed(seed)
            model = build_model("mlp", in_dim, width, depth, 2.0, 0.35)
            train_args = make_train_args(args, width, depth, RESULTS / "mlp_large")
            metrics = train_one(model, splits, train_args, device)
            record = {
                "run_name": "review_mlp_large",
                "task": "equation_native",
                "dataset": dataset,
                "seed": seed,
                "model": "mlp",
                "variant": "mlp_large_w64_d3",
                "target_expression": FUNCTIONS[dataset][2],
                "args": vars(train_args),
                "metrics": metrics,
            }
            path = write_result(out_dir, record)
            save_checkpoint(path, model, record)


def noisy_splits(base_splits: dict, noise_frac: float, seed: int) -> dict:
    splits = {}
    for split, (x, y) in base_splits.items():
        y_out = add_label_noise(y, noise_frac, seed) if split == "train" else np.asarray(y, dtype=np.float32)
        splits[split] = (np.asarray(x, dtype=np.float32), y_out)
    return splits


def run_noise_robustness(args: argparse.Namespace) -> None:
    out_dir = RESULTS / "noise_robustness"
    device = device_from_arg(args.device)
    datasets = args.noise_datasets or DEFAULT_NOISE_DATASETS
    methods = args.noise_methods or DEFAULT_NOISE_METHODS
    for dataset in datasets:
        for seed in args.seeds:
            base_splits = load_synthetic_dataset(dataset, seed)
            in_dim = base_splits["train"][0].shape[1]
            for noise_frac in args.noise_levels:
                suffix = f"_noise{noise_frac:g}".replace(".", "p")
                splits = noisy_splits(base_splits, noise_frac, seed + int(noise_frac * 10000))
                for method in methods:
                    if existing_json(out_dir, "review_noise", dataset, method, seed, suffix=suffix):
                        print(f"[skip noise] dataset={dataset} seed={seed} noise={noise_frac:g} method={method}", flush=True)
                        continue
                    width, depth = config_for(method)
                    print(
                        f"[noise] dataset={dataset} seed={seed} noise={noise_frac:g} "
                        f"method={method} width={width} depth={depth}",
                        flush=True,
                    )
                    set_seed(seed)
                    model = build_model(method, in_dim, width, depth, 2.0, 0.35)
                    train_args = make_train_args(args, width, depth, RESULTS / "noise_robustness")
                    metrics = train_one(model, splits, train_args, device)
                    record = {
                        "run_name": f"review_noise{suffix}",
                        "task": "equation_native_noise",
                        "dataset": dataset,
                        "seed": seed,
                        "noise_frac": noise_frac,
                        "model": method,
                        "target_expression": FUNCTIONS[dataset][2],
                        "args": vars(train_args),
                        "metrics": metrics,
                    }
                    path = write_result(out_dir, record)
                    save_checkpoint(path, model, record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "mlp-large", "noise"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--mlp-large-datasets", nargs="+", default=None)
    parser.add_argument("--noise-datasets", nargs="+", default=None)
    parser.add_argument("--noise-methods", nargs="+", default=None)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-lambda", type=float, default=1e-5)
    parser.add_argument("--gate-warmup-frac", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"all", "mlp-large"}:
        run_mlp_large(args)
    if args.mode in {"all", "noise"}:
        run_noise_robustness(args)


if __name__ == "__main__":
    main()
