from __future__ import annotations

import argparse
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "experiments"))
from run_tabular import (  # noqa: E402
    build_model,
    load_synthetic_dataset,
    save_checkpoint,
    set_seed,
    train_one,
    write_result,
)


BUDGET_TARGETS = {
    "small": 500,
    "medium": 2500,
    "large": 10000,
}

METHOD_LABELS = ["mlp", "stable_eml", "kan", "eml_kan"]
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


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def candidate_configs(method: str) -> list[dict]:
    if method == "mlp":
        return [
            {"width": width, "depth": depth}
            for width in [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
            for depth in [2, 3, 4]
        ]
    if method == "stable_eml":
        return [
            {"width": width, "depth": depth}
            for width in [4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256]
            for depth in [1, 2, 3, 4]
        ]
    if method == "kan":
        return [{"width": width, "depth": 2} for width in [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]]
    if method == "eml_kan":
        return [
            {"width": width, "depth": depth}
            for width in [4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256]
            for depth in [2, 3, 4]
        ]
    raise ValueError(method)


def select_config(method: str, in_dim: int, budget: str) -> dict:
    target = BUDGET_TARGETS[budget]
    best = None
    for cfg in candidate_configs(method):
        model = build_model(
            method,
            in_dim,
            cfg["width"],
            cfg["depth"],
            tau=2.0,
            eta=0.35,
            exp_mode="bounded_tanh",
            clip_m=8.0,
            w0_init="normal",
        )
        params = count_params(model)
        score = abs(math.log(params / target))
        item = {**cfg, "params": params, "target_params": target, "score": score}
        if best is None or item["score"] < best["score"]:
            best = item
    assert best is not None
    return best


def existing_keys(out_dir: Path) -> set[tuple[str, int, str, str]]:
    keys = set()
    for path in sorted(out_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        keys.add(
            (
                payload.get("dataset"),
                int(payload.get("seed", -1)),
                payload.get("model"),
                payload.get("budget"),
            )
        )
    return keys


def make_args(base: argparse.Namespace, width: int, depth: int) -> Namespace:
    return Namespace(
        task="synthetic",
        datasets=["core"],
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
        grad_clip=base.grad_clip,
        tau=2.0,
        eta=0.35,
        exp_mode="bounded_tanh",
        clip_m=8.0,
        w0_init="normal",
        gate_lambda=base.gate_lambda,
        gate_warmup_frac=base.gate_warmup_frac,
        gate_warmup_epochs=0,
        coeff_lambda=base.coeff_lambda,
        const_lambda=base.const_lambda,
        early_stop_metric="val",
        ood_weight=0.1,
        train_rows=100000,
        eval_rows=10000,
        expected_cols=5,
        max_abs_value=1e6,
        result_root=base.result_root,
        save_model=True,
        device=base.device,
        verbose=False,
    )


def write_config_audit(rows: list[dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "in_dim",
                "budget",
                "method",
                "target_params",
                "params",
                "width",
                "depth",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="results_v2/pareto_budget")
    parser.add_argument("--datasets", nargs="+", default=SYNTHETIC_DATASETS)
    parser.add_argument("--methods", nargs="+", default=METHOD_LABELS)
    parser.add_argument("--budgets", nargs="+", default=["small", "medium", "large"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--patience", type=int, default=90)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gate-lambda", type=float, default=1e-5)
    parser.add_argument("--gate-warmup-frac", type=float, default=0.25)
    parser.add_argument("--coeff-lambda", type=float, default=1e-5)
    parser.add_argument("--const-lambda", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Using device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}", flush=True)
    out_dir = ROOT / args.result_root / "synthetic"
    seen = existing_keys(out_dir)
    config_rows = []
    completed = 0
    skipped = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            splits = load_synthetic_dataset(dataset, seed)
            in_dim = splits["train"][0].shape[1]
            for budget in args.budgets:
                for method in args.methods:
                    cfg = select_config(method, in_dim, budget)
                    config_rows.append(
                        {
                            "dataset": dataset,
                            "in_dim": in_dim,
                            "budget": budget,
                            "method": method,
                            "target_params": cfg["target_params"],
                            "params": cfg["params"],
                            "width": cfg["width"],
                            "depth": cfg["depth"],
                        }
                    )
                    key = (dataset, seed, method, budget)
                    if key in seen:
                        skipped += 1
                        continue
                    print(
                        f"[pareto] dataset={dataset} seed={seed} budget={budget} "
                        f"method={method} params={cfg['params']} width={cfg['width']} depth={cfg['depth']}",
                        flush=True,
                    )
                    set_seed(seed)
                    train_args = make_args(args, cfg["width"], cfg["depth"])
                    model = build_model(
                        method,
                        in_dim,
                        cfg["width"],
                        cfg["depth"],
                        tau=2.0,
                        eta=0.35,
                        exp_mode="bounded_tanh",
                        clip_m=8.0,
                        w0_init="normal",
                    )
                    metrics = train_one(model, splits, train_args, device)
                    record = {
                        "run_name": "synthetic_pareto",
                        "task": "synthetic",
                        "dataset": dataset,
                        "seed": seed,
                        "budget": budget,
                        "target_params": cfg["target_params"],
                        "selected_params": cfg["params"],
                        "model": method,
                        "args": vars(train_args),
                        "metrics": metrics,
                    }
                    result_path = write_result(out_dir, record)
                    save_checkpoint(result_path, model, record)
                    completed += 1
    write_config_audit(config_rows, ROOT / args.result_root / "pareto_config_audit.csv")
    print(f"completed={completed} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
