import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CORE_SYNTHETIC = [
    "poly_1d_x2_x_1",
    "exp_1d",
    "log_1d_x_plus_2",
    "exp_log_1d",
    "kinetic_energy",
    "inverse_quadratic",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Standardizer:
    def __init__(self, values):
        values64 = values.astype(np.float64)
        self.mean = values64.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = values64.std(axis=0, keepdims=True).astype(np.float32)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        self.std = np.where(np.isfinite(self.std), self.std, 1.0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse_y(self, values):
        return values * float(self.std.reshape(-1)[0]) + float(self.mean.reshape(-1)[0])


class MLP(nn.Module):
    def __init__(self, in_dim, width=64, depth=3):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StableEMLBlock(nn.Module):
    def __init__(self, width, tau=2.0, eta=0.35):
        super().__init__()
        self.u = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.tau = tau
        self.eta = eta

    def forward(self, h):
        u = self.u(h)
        v = self.v(h)
        exp_branch = torch.exp(self.tau * torch.tanh(u))
        log_branch = torch.log1p(torch.nn.functional.softplus(v) + 1e-6)
        z = exp_branch - log_branch
        proposal = torch.tanh(self.out(z))
        return self.norm((1.0 - self.eta) * h + self.eta * proposal)


class StableEMLNet(nn.Module):
    def __init__(self, in_dim, width=64, depth=3, tau=2.0, eta=0.35):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, width), nn.SiLU())
        self.blocks = nn.ModuleList(
            [StableEMLBlock(width, tau=tau, eta=eta) for _ in range(depth)]
        )
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class RBFKANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, grid_size=16, grid_min=-3.0, grid_max=3.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        grid = torch.linspace(grid_min, grid_max, grid_size)
        self.register_buffer("grid", grid)
        spacing = (grid_max - grid_min) / max(grid_size - 1, 1)
        self.gamma = 1.0 / max(spacing, 1e-6)
        self.base_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.spline_weight = nn.Parameter(torch.empty(in_dim, out_dim, grid_size))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.03)

    def forward(self, x):
        base = x @ self.base_weight
        basis = torch.exp(-((x.unsqueeze(-1) - self.grid) * self.gamma).square())
        spline = torch.einsum("big,iog->bo", basis, self.spline_weight)
        return base + spline + self.bias

    def regularization_loss(self):
        zero = self.spline_weight.new_tensor(0.0)
        coeff_l1 = self.spline_weight.abs().mean()
        const_l2 = self.base_weight.square().mean()
        return zero, coeff_l1, const_l2


class RBFKAN(nn.Module):
    def __init__(self, in_dim, width=32, grid_size=16):
        super().__init__()
        self.layer1 = RBFKANLayer(in_dim, width, grid_size=grid_size)
        self.layer2 = RBFKANLayer(width, 1, grid_size=grid_size)

    def forward(self, x):
        h = self.layer1(x)
        return self.layer2(h)

    def regularization_loss(self):
        losses = [self.layer1.regularization_loss(), self.layer2.regularization_loss()]
        return tuple(sum(parts) for parts in zip(*losses))


class StableEMLEdgeLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        edge_depth=3,
        tau=2.0,
        exp_mode="bounded_tanh",
        clip_m=8.0,
        w0_init="normal",
        edge_variant="full",
    ):
        super().__init__()
        valid_variants = {"full", "no_residual", "no_log", "only_log", "no_gate"}
        if edge_variant not in valid_variants:
            raise ValueError(f"Unknown edge_variant: {edge_variant}")
        shape = (edge_depth, in_dim, out_dim)
        self.edge_depth = edge_depth
        self.tau = tau
        self.exp_mode = exp_mode
        self.clip_m = clip_m
        self.w0_init = w0_init
        self.edge_variant = edge_variant
        self.w0 = nn.Parameter(torch.empty(in_dim, out_dim))
        self.w = nn.Parameter(torch.empty(*shape))
        self.gate_logits = nn.Parameter(torch.empty(*shape))
        self.a = nn.Parameter(torch.empty(*shape))
        self.b = nn.Parameter(torch.empty(*shape))
        self.c = nn.Parameter(torch.empty(*shape))
        self.d = nn.Parameter(torch.empty(*shape))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        if self.w0_init == "identity":
            nn.init.zeros_(self.w0)
            with torch.no_grad():
                if self.w0.shape[0] == self.w0.shape[1]:
                    self.w0.fill_diagonal_(1.0)
                elif self.w0.shape[0] == 1:
                    self.w0.fill_(1.0 / max(self.w0.shape[1], 1))
                elif self.w0.shape[1] == 1:
                    self.w0.fill_(1.0 / max(self.w0.shape[0], 1))
                else:
                    scale = 1.0 / max(self.w0.shape[0], self.w0.shape[1])
                    self.w0.normal_(mean=0.0, std=scale)
        else:
            nn.init.normal_(self.w0, mean=0.0, std=0.15)
        nn.init.normal_(self.w, mean=0.0, std=0.05)
        nn.init.constant_(self.gate_logits, -1.5)
        nn.init.normal_(self.a, mean=0.7, std=0.05)
        nn.init.normal_(self.b, mean=0.0, std=0.05)
        nn.init.normal_(self.c, mean=0.1, std=0.03)
        nn.init.normal_(self.d, mean=0.5, std=0.03)

    def stable_exp_input(self, u):
        if self.exp_mode == "raw":
            return u
        if self.exp_mode == "bounded_tanh":
            return self.tau * torch.tanh(u)
        if self.exp_mode == "clip":
            return torch.clamp(u, -self.clip_m, self.clip_m)
        if self.exp_mode == "softclip":
            m = max(float(self.clip_m), 1e-6)
            return m * torch.tanh(u / m)
        raise ValueError(f"Unknown exp_mode: {self.exp_mode}")

    def forward(self, x):
        z = x.unsqueeze(-1)
        residual = z * self.w0.unsqueeze(0)
        acc = torch.zeros_like(residual) if self.edge_variant == "no_residual" else residual
        for t in range(self.edge_depth):
            u = self.a[t].unsqueeze(0) * z + self.b[t].unsqueeze(0)
            v = self.c[t].unsqueeze(0) * z + self.d[t].unsqueeze(0)
            exp_branch = torch.exp(self.stable_exp_input(u))
            log_branch = torch.log1p(torch.nn.functional.softplus(v) + 1e-6)
            if self.edge_variant == "no_log":
                z = exp_branch
            elif self.edge_variant == "only_log":
                z = -log_branch
            else:
                z = exp_branch - log_branch
            if self.edge_variant == "no_gate":
                gate = torch.ones_like(self.gate_logits[t]).unsqueeze(0)
            else:
                gate = torch.sigmoid(self.gate_logits[t]).unsqueeze(0)
            acc = acc + gate * self.w[t].unsqueeze(0) * z
        return acc.sum(dim=1) + self.bias

    def regularization_loss(self):
        if self.edge_variant == "no_gate":
            gate_l1 = self.gate_logits.new_tensor(0.0)
        else:
            gate_l1 = torch.sigmoid(self.gate_logits).mean()
        coeff_l1 = self.w.abs().mean()
        const_l2 = (
            self.a.square().mean()
            + self.b.square().mean()
            + self.c.square().mean()
            + self.d.square().mean()
        )
        return gate_l1, coeff_l1, const_l2


class EMLKAN(nn.Module):
    def __init__(
        self,
        in_dim,
        width=32,
        edge_depth=3,
        tau=2.0,
        exp_mode="bounded_tanh",
        clip_m=8.0,
        w0_init="normal",
        edge_variant="full",
    ):
        super().__init__()
        self.layer1 = StableEMLEdgeLayer(
            in_dim,
            width,
            edge_depth=edge_depth,
            tau=tau,
            exp_mode=exp_mode,
            clip_m=clip_m,
            w0_init=w0_init,
            edge_variant=edge_variant,
        )
        self.layer2 = StableEMLEdgeLayer(
            width,
            1,
            edge_depth=edge_depth,
            tau=tau,
            exp_mode=exp_mode,
            clip_m=clip_m,
            w0_init=w0_init,
            edge_variant=edge_variant,
        )

    def forward(self, x):
        h = self.layer1(x)
        return self.layer2(h)

    def regularization_loss(self):
        losses = [self.layer1.regularization_loss(), self.layer2.regularization_loss()]
        return tuple(sum(parts) for parts in zip(*losses))


def build_model(
    model_name,
    in_dim,
    width,
    depth,
    tau,
    eta,
    exp_mode="bounded_tanh",
    clip_m=8.0,
    w0_init="normal",
    edge_variant="full",
):
    if model_name == "mlp":
        return MLP(in_dim, width=width, depth=depth)
    if model_name == "stable_eml":
        return StableEMLNet(in_dim, width=width, depth=depth, tau=tau, eta=eta)
    if model_name == "kan":
        return RBFKAN(in_dim, width=width, grid_size=16)
    if model_name == "eml_edge":
        return StableEMLEdgeLayer(
            in_dim,
            1,
            edge_depth=depth,
            tau=tau,
            exp_mode=exp_mode,
            clip_m=clip_m,
            w0_init=w0_init,
            edge_variant=edge_variant,
        )
    if model_name == "eml_kan":
        return EMLKAN(
            in_dim,
            width=width,
            edge_depth=depth,
            tau=tau,
            exp_mode=exp_mode,
            clip_m=clip_m,
            w0_init=w0_init,
            edge_variant=edge_variant,
        )
    if model_name == "eml_kan_raw":
        return EMLKAN(
            in_dim,
            width=width,
            edge_depth=depth,
            tau=tau,
            exp_mode="raw",
            clip_m=clip_m,
            w0_init=w0_init,
            edge_variant=edge_variant,
        )
    if model_name == "eml_kan_clip":
        return EMLKAN(
            in_dim,
            width=width,
            edge_depth=depth,
            tau=tau,
            exp_mode="clip",
            clip_m=clip_m,
            w0_init=w0_init,
            edge_variant=edge_variant,
        )
    if model_name == "eml_kan_softclip":
        return EMLKAN(
            in_dim,
            width=width,
            edge_depth=depth,
            tau=tau,
            exp_mode="softclip",
            clip_m=clip_m,
            w0_init=w0_init,
            edge_variant=edge_variant,
        )
    if model_name == "eml_kan_identity":
        return EMLKAN(
            in_dim,
            width=width,
            edge_depth=depth,
            tau=tau,
            exp_mode=exp_mode,
            clip_m=clip_m,
            w0_init="identity",
            edge_variant=edge_variant,
        )
    raise ValueError(f"Unknown model: {model_name}")


def load_synthetic_dataset(dataset, seed):
    base = Path("data/synthetic") / dataset / f"seed_{seed}"
    splits = {}
    for split in ["train", "val", "test", "ood"]:
        df = pd.read_csv(base / f"{split}.csv")
        x = df.drop(columns=["y"]).to_numpy(dtype=np.float32)
        y = df[["y"]].to_numpy(dtype=np.float32)
        splits[split] = (x, y)
    return splits


def read_srsd_split(path, max_rows, expected_cols=None, max_abs_value=1e6):
    xs = []
    ys = []
    skipped = 0
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = row["text"].split()
            if expected_cols is not None and len(values) != expected_cols:
                skipped += 1
                continue
            nums = [float(v) for v in values]
            if (
                not np.all(np.isfinite(nums))
                or max(abs(v) for v in nums) > max_abs_value
            ):
                skipped += 1
                continue
            xs.append(nums[:-1])
            ys.append([nums[-1]])
            if len(xs) >= max_rows:
                break
    if not xs:
        raise RuntimeError(f"No SRSD rows parsed from {path}")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), skipped


def load_srsd_dataset(train_rows, eval_rows, expected_cols=5, max_abs_value=1e6):
    train_x, train_y, train_skipped = read_srsd_split(
        "data/srsd_feynman_hard/train.csv", train_rows, expected_cols, max_abs_value
    )
    val_x, val_y, val_skipped = read_srsd_split(
        "data/srsd_feynman_hard/validation.csv", eval_rows, expected_cols, max_abs_value
    )
    test_x, test_y, test_skipped = read_srsd_split(
        "data/srsd_feynman_hard/test.csv", eval_rows, expected_cols, max_abs_value
    )
    return {
        "train": (train_x, train_y),
        "val": (val_x, val_y),
        "test": (test_x, test_y),
        "meta": {
            "expected_cols": expected_cols,
            "train_rows": len(train_x),
            "val_rows": len(val_x),
            "test_rows": len(test_x),
            "skipped": {
                "train": train_skipped,
                "val": val_skipped,
                "test": test_skipped,
            },
            "max_abs_value": max_abs_value,
        },
    }


def make_loader(x, y, batch_size, shuffle):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate(model, x, y, y_standardizer, device, batch_size):
    model.eval()
    preds = []
    target = []
    for bx, by in make_loader(x, y, batch_size=batch_size, shuffle=False):
        bx = bx.to(device, non_blocking=True)
        pred = model(bx).detach().cpu().numpy()
        preds.append(pred)
        target.append(by.numpy())
    pred = np.vstack(preds)
    true = np.vstack(target)
    pred_raw = y_standardizer.inverse_y(pred)
    true_raw = y_standardizer.inverse_y(true)
    mse = float(np.mean((pred_raw - true_raw) ** 2))
    rmse = math.sqrt(mse)
    var = float(np.var(true_raw))
    r2 = float(1.0 - mse / max(var, 1e-12))
    return {"mse": mse, "rmse": rmse, "r2": r2}


def train_one(model, splits, args, device):
    train_x, train_y = splits["train"]
    val_x, val_y = splits["val"]

    x_std = Standardizer(train_x)
    y_std = Standardizer(train_y)
    norm_splits = {}
    for name, split_value in splits.items():
        if name == "meta":
            continue
        x, y = split_value
        norm_splits[name] = (
            x_std.transform(x).astype(np.float32),
            y_std.transform(y).astype(np.float32),
        )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()
    loader = make_loader(
        norm_splits["train"][0], norm_splits["train"][1], args.batch_size, shuffle=True
    )

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    nan_steps = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = loss_fn(pred, by)
            if hasattr(model, "regularization_loss"):
                gate_l1, coeff_l1, const_l2 = model.regularization_loss()
                gate_warmup_epochs = int(getattr(args, "gate_warmup_epochs", 0) or 0)
                gate_warmup_frac = float(getattr(args, "gate_warmup_frac", 0.0) or 0.0)
                if gate_warmup_epochs <= 0 and gate_warmup_frac > 0:
                    gate_warmup_epochs = int(args.epochs * gate_warmup_frac)
                gate_lambda = 0.0 if epoch <= gate_warmup_epochs else args.gate_lambda
                loss = (
                    loss
                    + gate_lambda * gate_l1
                    + args.coeff_lambda * coeff_l1
                    + args.const_lambda * const_l2
                )
            if not torch.isfinite(loss):
                nan_steps += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * bx.shape[0]
            total_count += bx.shape[0]

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(
                model,
                norm_splits["val"][0],
                norm_splits["val"][1],
                y_std,
                device,
                args.eval_batch_size,
            )
            val_mse = val_metrics["mse"]
            score = val_mse
            ood_mse = None
            if args.early_stop_metric == "val_ood" and "ood" in norm_splits:
                ood_metrics = evaluate(
                    model,
                    norm_splits["ood"][0],
                    norm_splits["ood"][1],
                    y_std,
                    device,
                    args.eval_batch_size,
                )
                ood_mse = ood_metrics["mse"]
                score = val_mse + args.ood_weight * ood_mse
            if score < best_val:
                best_val = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += args.eval_every
            if args.verbose:
                train_loss = total_loss / max(total_count, 1)
                ood_text = f" ood_mse={ood_mse:.4e}" if ood_mse is not None else ""
                print(
                    f"epoch={epoch} train_loss={train_loss:.4e} "
                    f"val_mse={val_mse:.4e}{ood_text} best_score={best_val:.4e}",
                    flush=True,
                )
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "best_epoch": best_epoch,
        "best_score": best_val,
        "early_stop_metric": args.early_stop_metric,
        "nan_steps": nan_steps,
        "runtime_sec": time.time() - start,
        "x_mean": x_std.mean.reshape(-1).tolist(),
        "x_std": x_std.std.reshape(-1).tolist(),
        "y_mean": y_std.mean.reshape(-1).tolist(),
        "y_std": y_std.std.reshape(-1).tolist(),
    }
    for split_name, (x, y) in norm_splits.items():
        if split_name == "train":
            continue
        metrics[split_name] = evaluate(
            model, x, y, y_std, device, args.eval_batch_size
        )
    return metrics


def write_result(out_dir, record):
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parts = [
        record.get("run_name", "run"),
        record.get("dataset", record.get("task", "task")),
        record.get("model", "model"),
    ]
    if "seed" in record:
        parts.append(f"seed{record['seed']}")
    safe_name = "_".join(str(part).replace("/", "-") for part in parts)
    path = out_dir / f"{safe_name}_{stamp}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    return path


def save_checkpoint(result_path, model, record):
    checkpoint_path = result_path.with_suffix(".pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "record": record,
        },
        checkpoint_path,
    )
    print(f"Wrote {checkpoint_path}", flush=True)


def run_synthetic(args, device):
    datasets = CORE_SYNTHETIC if args.datasets == ["core"] else args.datasets
    records = []
    for dataset in datasets:
        for seed in args.seeds:
            splits = load_synthetic_dataset(dataset, seed)
            in_dim = splits["train"][0].shape[1]
            for model_name in args.models:
                print(
                    f"[synthetic] dataset={dataset} seed={seed} model={model_name}",
                    flush=True,
                )
                set_seed(seed)
                model = build_model(
                    model_name,
                    in_dim,
                    args.width,
                    args.depth,
                    args.tau,
                    args.eta,
                    args.exp_mode,
                    args.clip_m,
                    args.w0_init,
                    args.edge_variant,
                )
                metrics = train_one(model, splits, args, device)
                record = {
                    "run_name": "synthetic",
                    "task": "synthetic",
                    "dataset": dataset,
                    "seed": seed,
                    "model": model_name,
                    "args": vars(args),
                    "metrics": metrics,
                }
                records.append(record)
                result_path = write_result(Path(args.result_root) / "synthetic", record)
                if args.save_model:
                    save_checkpoint(result_path, model, record)
    return records


def run_srsd(args, device):
    splits = load_srsd_dataset(
        train_rows=args.train_rows,
        eval_rows=args.eval_rows,
        expected_cols=args.expected_cols,
        max_abs_value=args.max_abs_value,
    )
    in_dim = splits["train"][0].shape[1]
    for seed in args.seeds:
        for model_name in args.models:
            print(f"[srsd] seed={seed} model={model_name} in_dim={in_dim}", flush=True)
            set_seed(seed)
            model = build_model(
                model_name,
                in_dim,
                args.width,
                args.depth,
                args.tau,
                args.eta,
                args.exp_mode,
                args.clip_m,
                args.w0_init,
                args.edge_variant,
            )
            metrics = train_one(model, splits, args, device)
            record = {
                "run_name": "srsd",
                "task": "srsd",
                "dataset": "srsd_feynman_hard",
                "seed": seed,
                "model": model_name,
                "args": vars(args),
                "data_meta": splits.get("meta", {}),
                "metrics": metrics,
            }
            result_path = write_result(Path(args.result_root) / "srsd", record)
            if args.save_model:
                save_checkpoint(result_path, model, record)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["synthetic", "srsd"], required=True)
    parser.add_argument("--datasets", nargs="+", default=["core"])
    parser.add_argument("--models", nargs="+", default=["mlp", "stable_eml"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.35)
    parser.add_argument(
        "--exp-mode",
        choices=["bounded_tanh", "clip", "softclip"],
        default="bounded_tanh",
    )
    parser.add_argument("--clip-m", type=float, default=8.0)
    parser.add_argument(
        "--w0-init",
        choices=["normal", "identity"],
        default="normal",
    )
    parser.add_argument(
        "--edge-variant",
        choices=["full", "no_residual", "no_log", "only_log", "no_gate"],
        default="full",
    )
    parser.add_argument("--gate-lambda", type=float, default=1e-4)
    parser.add_argument("--gate-warmup-frac", type=float, default=0.0)
    parser.add_argument("--gate-warmup-epochs", type=int, default=0)
    parser.add_argument("--coeff-lambda", type=float, default=1e-5)
    parser.add_argument("--const-lambda", type=float, default=1e-6)
    parser.add_argument("--early-stop-metric", choices=["val", "val_ood"], default="val")
    parser.add_argument("--ood-weight", type=float, default=0.1)
    parser.add_argument("--train-rows", type=int, default=100000)
    parser.add_argument("--eval-rows", type=int, default=10000)
    parser.add_argument("--expected-cols", type=int, default=5)
    parser.add_argument("--max-abs-value", type=float, default=1e6)
    parser.add_argument("--result-root", default="results")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Using device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}", flush=True)
    if args.task == "synthetic":
        run_synthetic(args, device)
    else:
        run_srsd(args, device)


if __name__ == "__main__":
    main()
