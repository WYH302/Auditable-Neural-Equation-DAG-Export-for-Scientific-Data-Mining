import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from run_tabular import build_model, set_seed, train_one, write_result, save_checkpoint


TRIG_TOKENS = ("sin", "cos", "tan", "asin", "acos", "atan", "arcsin", "arccos", "arctan")


def detect_ncols(path):
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                return len(stripped.split())
    return 0


def read_table(path, max_rows, max_abs_value):
    values = np.loadtxt(path, dtype=np.float32, max_rows=max_rows)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    finite = np.isfinite(values).all(axis=1)
    bounded = np.max(np.abs(values), axis=1) <= max_abs_value
    values = values[finite & bounded]
    if values.shape[0] < 100:
        raise RuntimeError(f"Too few usable rows in {path}: {values.shape[0]}")
    return values


def split_table(values, seed, train_rows, val_rows, test_rows):
    rng = np.random.default_rng(seed)
    indices = np.arange(values.shape[0])
    rng.shuffle(indices)
    need = train_rows + val_rows + test_rows
    indices = indices[: min(need, len(indices))]
    values = values[indices]

    n_train = min(train_rows, max(1, int(0.7 * len(values))))
    remaining = len(values) - n_train
    n_val = min(val_rows, max(1, remaining // 2))
    n_test = min(test_rows, len(values) - n_train - n_val)
    if n_test <= 0:
        n_test = max(1, len(values) - n_train - n_val)

    train = values[:n_train]
    val = values[n_train : n_train + n_val]
    test = values[n_train + n_val : n_train + n_val + n_test]

    def xy(block):
        return block[:, :-1].astype(np.float32), block[:, [-1]].astype(np.float32)

    return {"train": xy(train), "val": xy(val), "test": xy(test)}


def load_metadata(data_dir):
    data_dir = Path(data_dir)
    official = data_dir.parent
    meta_path = official / "FeynmanEquations.csv"
    if not meta_path.exists():
        return {}
    df = pd.read_csv(meta_path)
    return {
        row["Filename"]: {
            "formula": row.get("Formula", ""),
            "variables": (
                None
                if pd.isna(row.get("# variables", None))
                else float(row.get("# variables", None))
            ),
            "output": row.get("Output", ""),
        }
        for _, row in df.iterrows()
    }


def select_files(args):
    data_dir = Path(args.data_dir)
    metadata = load_metadata(data_dir)
    rows = []
    for path in sorted(data_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        ncols = detect_ncols(path)
        if ncols < 2:
            continue
        input_dim = ncols - 1
        if input_dim > args.max_input_dim:
            continue
        formula = str(metadata.get(path.name, {}).get("formula", ""))
        if args.exclude_trig and any(token in formula.lower() for token in TRIG_TOKENS):
            continue
        rows.append(
            {
                "name": path.name,
                "path": path,
                "ncols": ncols,
                "input_dim": input_dim,
                "bytes": path.stat().st_size,
                "formula": formula,
                "metadata_variables": metadata.get(path.name, {}).get("variables"),
            }
        )
    if args.files:
        wanted = set(args.files)
        rows = [row for row in rows if row["name"] in wanted]
    rows = rows[args.offset :]
    if args.limit_files is not None:
        rows = rows[: args.limit_files]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/feynman/official/Feynman_without_units",
    )
    parser.add_argument("--result-root", default="results_v2/feynman_lowdim")
    parser.add_argument("--models", nargs="+", default=["mlp", "eml_kan"])
    parser.add_argument("--files", nargs="*")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--max-input-dim", type=int, default=4)
    parser.add_argument("--exclude-trig", action="store_true")
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--train-rows", type=int, default=12000)
    parser.add_argument("--val-rows", type=int, default=4000)
    parser.add_argument("--test-rows", type=int, default=4000)
    parser.add_argument("--max-abs-value", type=float, default=1e8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.35)
    parser.add_argument("--gate-lambda", type=float, default=1e-4)
    parser.add_argument("--coeff-lambda", type=float, default=1e-5)
    parser.add_argument("--const-lambda", type=float, default=1e-6)
    parser.add_argument("--early-stop-metric", choices=["val", "val_ood"], default="val")
    parser.add_argument("--ood-weight", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Using device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}", flush=True)

    files = select_files(args)
    print(f"Selected {len(files)} Feynman files", flush=True)
    for row in files:
        print(
            f"[selected] {row['name']} input_dim={row['input_dim']} bytes={row['bytes']} "
            f"formula={row['formula']}",
            flush=True,
        )

    for row in files:
        values = read_table(row["path"], args.max_rows, args.max_abs_value)
        for seed in args.seeds:
            splits = split_table(
                values,
                seed=seed,
                train_rows=args.train_rows,
                val_rows=args.val_rows,
                test_rows=args.test_rows,
            )
            in_dim = splits["train"][0].shape[1]
            for model_name in args.models:
                print(f"[feynman] file={row['name']} seed={seed} model={model_name}", flush=True)
                set_seed(seed)
                model = build_model(
                    model_name, in_dim, args.width, args.depth, args.tau, args.eta
                )
                start = time.time()
                metrics = train_one(model, splits, args, device)
                record = {
                    "run_name": "feynman_lowdim",
                    "task": "feynman_lowdim",
                    "dataset": row["name"],
                    "seed": seed,
                    "model": model_name,
                    "args": vars(args),
                    "data_meta": {
                        "filename": row["name"],
                        "input_dim": row["input_dim"],
                        "ncols": row["ncols"],
                        "formula": row["formula"],
                        "metadata_variables": row["metadata_variables"],
                        "loaded_rows": int(values.shape[0]),
                        "train_rows": int(splits["train"][0].shape[0]),
                        "val_rows": int(splits["val"][0].shape[0]),
                        "test_rows": int(splits["test"][0].shape[0]),
                    },
                    "metrics": metrics,
                    "wall_sec_including_load": time.time() - start,
                }
                result_path = write_result(Path(args.result_root) / "runs", record)
                if args.save_model:
                    save_checkpoint(result_path, model, record)


if __name__ == "__main__":
    main()
