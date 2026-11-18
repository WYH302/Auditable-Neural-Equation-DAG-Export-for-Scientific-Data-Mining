"""Extended baselines (Spline-KAN, Cheby-KAN), component ablations, and 5-fold CV.

Executes deterministically in IEEE-754 float64 using the exact benchmark generators,
training loop, and lexical DAG tokenizer of study.py and graph.py.
"""
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy.stats as stats
import torch
from torch import nn

from datasets import FUNCTIONS
from graph import cse, expand_primitives, fmt, measured, token_count
from models import EMLKAN, MLP, RBFKAN
from study import make_data, normalize

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT_PATH = PROJECT / "results_v2" / "journal_followup" / "ablation_and_cv_summary.json"


class ChebyKANLayer(nn.Module):
    """Chebyshev polynomial KAN layer on tanh-normalized inputs (Shukla et al., 2024)."""
    def __init__(self, in_dim, out_dim, degree=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree
        self.base_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.cheby_weight = nn.Parameter(torch.empty(in_dim, out_dim, degree + 1))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.cheby_weight, mean=0.0, std=1.0 / (in_dim * (degree + 1)))

    def forward(self, x):
        base = x @ self.base_weight
        u = torch.tanh(x)
        polys = [torch.ones_like(u), u]
        for _ in range(2, self.degree + 1):
            polys.append(2.0 * u * polys[-1] - polys[-2])
        basis = torch.stack(polys, dim=-1)  # (B, in_dim, degree+1)
        cheby = torch.einsum("bik,iok->bo", basis, self.cheby_weight)
        return base + cheby + self.bias

    def regularization_loss(self):
        zero = self.cheby_weight.new_tensor(0.0)
        return zero, self.cheby_weight.abs().mean(), self.base_weight.square().mean()


class ChebyKAN(nn.Module):
    def __init__(self, in_dim, width=4, degree=8):
        super().__init__()
        self.layer1 = ChebyKANLayer(in_dim, width, degree=degree)
        self.layer2 = ChebyKANLayer(width, 1, degree=degree)

    def forward(self, x):
        return self.layer2(self.layer1(x))

    def regularization_loss(self):
        l1, l2 = self.layer1.regularization_loss(), self.layer2.regularization_loss()
        return tuple(a + b for a, b in zip(l1, l2))


class BSplineKANLayer(nn.Module):
    """Cubic B-spline KAN layer with SiLU residual (Liu et al., ICLR 2025)."""
    def __init__(self, in_dim, out_dim, grid_size=8, k=3, grid_min=-3.0, grid_max=3.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.k = k
        h = (grid_max - grid_min) / grid_size
        knots = torch.arange(-k, grid_size + k + 1, dtype=torch.float64) * h + grid_min
        self.register_buffer("knots", knots)
        self.num_bases = grid_size + k
        self.base_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.spline_weight = nn.Parameter(torch.empty(in_dim, out_dim, self.num_bases))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.05)

    def b_splines(self, x):
        # x: (B, in_dim)
        x_u = x.unsqueeze(-1)  # (B, in_dim, 1)
        t = self.knots  # (G + 2k + 1,)
        bases = ((x_u >= t[:-1]) & (x_u < t[1:])).to(x.dtype)
        for p in range(1, self.k + 1):
            denom1 = t[p:-1] - t[:-p - 1]
            denom2 = t[p + 1:] - t[1:-p]
            term1 = torch.where(denom1 > 0, (x_u - t[:-p - 1]) / denom1, torch.zeros_like(x_u)) * bases[..., :-1]
            term2 = torch.where(denom2 > 0, (t[p + 1:] - x_u) / denom2, torch.zeros_like(x_u)) * bases[..., 1:]
            bases = term1 + term2
        return bases  # (B, in_dim, G + k)

    def forward(self, x):
        base = torch.nn.functional.silu(x) @ self.base_weight
        basis = self.b_splines(x)
        spline = torch.einsum("bik,iok->bo", basis, self.spline_weight)
        return base + spline + self.bias

    def regularization_loss(self):
        zero = self.spline_weight.new_tensor(0.0)
        return zero, self.spline_weight.abs().mean(), self.base_weight.square().mean()


class BSplineKAN(nn.Module):
    def __init__(self, in_dim, width=4, grid_size=8, k=3):
        super().__init__()
        self.layer1 = BSplineKANLayer(in_dim, width, grid_size=grid_size, k=k)
        self.layer2 = BSplineKANLayer(width, 1, grid_size=grid_size, k=k)

    def forward(self, x):
        return self.layer2(self.layer1(x))

    def regularization_loss(self):
        l1, l2 = self.layer1.regularization_loss(), self.layer2.regularization_loss()
        return tuple(a + b for a, b in zip(l1, l2))


class AblationEMLLayer(nn.Module):
    """Configurable EML operator-edge layer for isolating stabilization mechanisms."""
    def __init__(self, in_dim, out_dim, edge_depth=2, tau=2.0, mode="full"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_depth = edge_depth
        self.tau = tau
        self.mode = mode
        shape = (edge_depth, in_dim, out_dim)
        self.w0 = nn.Parameter(torch.empty(in_dim, out_dim))
        self.w = nn.Parameter(torch.empty(*shape))
        self.gate_logits = nn.Parameter(torch.empty(*shape))
        self.a = nn.Parameter(torch.empty(*shape))
        self.b = nn.Parameter(torch.empty(*shape))
        self.c = nn.Parameter(torch.empty(*shape))
        self.d = nn.Parameter(torch.empty(*shape))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.normal_(self.w0, mean=0.0, std=0.15)
        nn.init.normal_(self.w, mean=0.0, std=0.05)
        nn.init.constant_(self.gate_logits, -1.5)
        nn.init.normal_(self.a, mean=0.7, std=0.05)
        nn.init.normal_(self.b, mean=0.0, std=0.05)
        nn.init.normal_(self.c, mean=0.1, std=0.03)
        nn.init.normal_(self.d, mean=0.5, std=0.03)

    def forward(self, x):
        z = x.unsqueeze(-1)
        acc = torch.zeros_like(z * self.w0.unsqueeze(0)) if self.mode == "no_bypass" else z * self.w0.unsqueeze(0)
        for t in range(self.edge_depth):
            u = self.a[t].unsqueeze(0) * z + self.b[t].unsqueeze(0)
            v = self.c[t].unsqueeze(0) * z + self.d[t].unsqueeze(0)
            if self.mode in ("raw_eml_eql", "unbounded_exp"):
                exp_branch = torch.exp(u)
            else:
                exp_branch = torch.exp(self.tau * torch.tanh(u))
            if self.mode == "raw_eml_eql":
                log_branch = torch.log(v)
            elif self.mode == "abs_log":
                log_branch = torch.log(torch.abs(v) + 1e-6)
            else:
                log_branch = torch.log1p(torch.nn.functional.softplus(v) + 1e-6)
            z = exp_branch - log_branch
            if self.mode == "no_gate":
                gate = 1.0
            else:
                gate = torch.sigmoid(self.gate_logits[t]).unsqueeze(0)
            acc = acc + gate * self.w[t].unsqueeze(0) * z
        return acc.sum(dim=1) + self.bias

    def regularization_loss(self):
        gate_l1 = torch.zeros((), dtype=self.w.dtype) if self.mode == "no_gate" else torch.sigmoid(self.gate_logits).mean()
        coeff_l1 = self.w.abs().mean()
        const_l2 = self.a.square().mean() + self.b.square().mean() + self.c.square().mean() + self.d.square().mean()
        return gate_l1, coeff_l1, const_l2


class AblationEMLKAN(nn.Module):
    def __init__(self, in_dim, width=4, edge_depth=2, tau=2.0, mode="full"):
        super().__init__()
        self.layer1 = AblationEMLLayer(in_dim, width, edge_depth=edge_depth, tau=tau, mode=mode)
        self.layer2 = AblationEMLLayer(width, 1, edge_depth=edge_depth, tau=tau, mode=mode)

    def forward(self, x):
        return self.layer2(self.layer1(x))

    def regularization_loss(self):
        l1, l2 = self.layer1.regularization_loss(), self.layer2.regularization_loss()
        return tuple(a + b for a, b in zip(l1, l2))


def export_cheby(model, scales):
    d = len(scales["x_mean"])
    lines = [f"x{i+1}_norm = (x{i+1} - ({fmt(scales['x_mean'][i])}))/({fmt(scales['x_std'][i])})" for i in range(d)]
    h = [f"x{i+1}_norm" for i in range(d)]
    for lid, layer in enumerate((model.layer1, model.layer2)):
        for i, inp in enumerate(h):
            u_name = f"u{lid}_{i}"
            lines.append(f"{u_name} = tanh({inp})")
            lines.append(f"T{lid}_{i}_0 = 1.0")
            lines.append(f"T{lid}_{i}_1 = {u_name}")
            for k in range(2, layer.degree + 1):
                lines.append(f"T{lid}_{i}_{k} = 2.0*({u_name})*(T{lid}_{i}_{k-1}) - (T{lid}_{i}_{k-2})")
        outputs = []
        for j in range(layer.out_dim):
            name = "y_norm" if lid == 1 else f"h{lid}_{j}"
            terms = [f"({fmt(layer.bias[j])})"]
            for i, inp in enumerate(h):
                terms.append(f"({fmt(layer.base_weight[i,j])})*({inp})")
                for k in range(layer.degree + 1):
                    terms.append(f"({fmt(layer.cheby_weight[i,j,k])})*(T{lid}_{i}_{k})")
            lines.append(f"{name} = " + " + ".join(terms))
            outputs.append(name)
        h = outputs
    lines.append(f"y = ({fmt(scales['y_std'])})*y_norm + ({fmt(scales['y_mean'])})")
    raw_tok = token_count(";\n".join(lines))
    cse_lines = cse(lines)
    cse_tok = token_count(";\n".join(cse_lines))
    return raw_tok, cse_tok


def export_bspline_tokens(model, scales):
    """Count exact scalar DAG tokens for piecewise cubic B-spline Cox-de Boor unrolling."""
    d = len(scales["x_mean"])
    lines = [f"x{i+1}_norm = (x{i+1} - ({fmt(scales['x_mean'][i])}))/({fmt(scales['x_std'][i])})" for i in range(d)]
    h = [f"x{i+1}_norm" for i in range(d)]
    for lid, layer in enumerate((model.layer1, model.layer2)):
        t = layer.knots.detach().cpu().numpy()
        for i, inp in enumerate(h):
            # Degree-0 indicator intervals + Cox-de Boor recursion for p=1..k
            for m in range(len(t) - 1):
                lines.append(f"B{lid}_{i}_0_{m} = step({inp} - ({fmt(t[m])})) - step({inp} - ({fmt(t[m+1])}))")
            for p in range(1, layer.k + 1):
                for m in range(len(t) - 1 - p):
                    d1, d2 = t[m + p] - t[m], t[m + p + 1] - t[m + 1]
                    p1 = f"(({inp} - ({fmt(t[m])}))/({fmt(d1)}))*(B{lid}_{i}_{p-1}_{m})" if d1 > 0 else "0.0"
                    p2 = f"((({fmt(t[m+p+1])}) - {inp})/({fmt(d2)}))*(B{lid}_{i}_{p-1}_{m+1})" if d2 > 0 else "0.0"
                    lines.append(f"B{lid}_{i}_{p}_{m} = {p1} + {p2}")
        outputs = []
        for j in range(layer.out_dim):
            name = "y_norm" if lid == 1 else f"h{lid}_{j}"
            terms = [f"({fmt(layer.bias[j])})"]
            for i, inp in enumerate(h):
                terms.append(f"({fmt(layer.base_weight[i,j])})*(({inp})/(1 + exp(-({inp}))))")
                for m in range(layer.num_bases):
                    terms.append(f"({fmt(layer.spline_weight[i,j,m])})*(B{lid}_{i}_{layer.k}_{m})")
            lines.append(f"{name} = " + " + ".join(terms))
            outputs.append(name)
        h = outputs
    lines.append(f"y = ({fmt(scales['y_std'])})*y_norm + ({fmt(scales['y_mean'])})")
    raw_tok = token_count(";\n".join(lines))
    # Shared knot differences and Silu subexpressions eliminated via SymPy CSE
    import sympy as sp
    fn_map = {"exp": sp.Function("exp"), "step": sp.Function("step")}
    names, exprs = [], []
    for line in lines:
        lhs, rhs = line.split("=", 1)
        names.append(lhs.strip())
        exprs.append(sp.sympify(rhs.strip(), locals=fn_map))
    reps, reduced = sp.cse(exprs, symbols=sp.numbered_symbols("common_"), order="canonical")
    cse_str = ";\n".join([f"{n} = {sp.sstr(e, full_prec=True)}" for n, e in reps] +
                         [f"{n} = {sp.sstr(e, full_prec=True)}" for n, e in zip(names, reduced)])
    cse_tok = token_count(cse_str)
    return raw_tok, cse_tok


def fit_model(model, norm, lr, max_steps=600):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x, y = norm["train"]
    vx, vy = norm["val"]
    best_score, best_state, best_step = float("inf"), None, 0
    nonfinite = False
    for step in range(1, max_steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        if hasattr(model, "regularization_loss"):
            g, c, k = model.regularization_loss()
            loss = loss + 1e-4 * g + 1e-5 * c + 1e-6 * k
        if not bool(torch.isfinite(loss)):
            nonfinite = True
            break
        loss.backward()
        if not all(p.grad is None or bool(torch.all(torch.isfinite(p.grad))) for p in model.parameters()):
            nonfinite = True
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                val_loss = float(torch.mean((model(vx) - vy) ** 2))
            if not math.isfinite(val_loss):
                nonfinite = True
                break
            if val_loss < best_score:
                best_score, best_step = val_loss, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if step - best_step >= 120:
                break
    if nonfinite or best_state is None:
        return None
    model.load_state_dict(best_state)
    return best_score


def eval_splits(model, data, norm, scales):
    model.eval()
    out = {}
    with torch.no_grad():
        for split, (_, y) in data.items():
            pred = model(norm[split][0]).numpy().reshape(-1) * scales["y_std"] + scales["y_mean"]
            truth = y.reshape(-1)
            mse = float(np.mean((pred - truth) ** 2))
            var = max(float(np.var(truth)), 1e-15)
            out[split] = {"nmse": mse / var, "r2": 1.0 - mse / var}
    return out


def run_baseline_item(item):
    torch.set_num_threads(1)
    dataset, seed, method, width, lr = item
    data = make_data(dataset, seed)
    norm, scales = normalize(data)
    dim = data["train"][0].shape[1]
    torch.manual_seed(seed)
    if method == "cheby_kan":
        model = ChebyKAN(dim, width=width, degree=8).double()
    elif method == "bspline_kan":
        model = BSplineKAN(dim, width=width, grid_size=8, k=3).double()
    else:
        raise ValueError(method)
    score = fit_model(model, norm, lr)
    if score is None:
        return {"dataset": dataset, "seed": seed, "method": method, "width": width, "lr": lr, "status": "failed"}
    metrics = eval_splits(model, data, norm, scales)
    return {
        "dataset": dataset, "seed": seed, "method": method, "width": width, "lr": lr,
        "status": "ok", "metrics": metrics,
        "params": sum(p.numel() for p in model.parameters()),
    }


def compute_selected_tokens(item):
    torch.set_num_threads(1)
    dataset, seed, method, width, lr = item
    data = make_data(dataset, seed)
    norm, scales = normalize(data)
    dim = data["train"][0].shape[1]
    torch.manual_seed(seed)
    if method == "cheby_kan":
        model = ChebyKAN(dim, width=width, degree=8).double()
        fit_model(model, norm, lr)
        raw_t, cse_t = export_cheby(model, scales)
    elif method == "bspline_kan":
        model = BSplineKAN(dim, width=width, grid_size=8, k=3).double()
        fit_model(model, norm, lr)
        raw_t, cse_t = export_bspline_tokens(model, scales)
    return (dataset, seed, method), (raw_t, cse_t)


def run_ablation_item(item):
    torch.set_num_threads(1)
    dataset, seed, variant, lr = item
    data = make_data(dataset, seed)
    norm, scales = normalize(data)
    dim = data["train"][0].shape[1]
    torch.manual_seed(seed)
    if variant == "depth_T1":
        model = AblationEMLKAN(dim, width=4, edge_depth=1, tau=2.0, mode="full").double()
    elif variant == "depth_T3":
        model = AblationEMLKAN(dim, width=4, edge_depth=3, tau=2.0, mode="full").double()
    elif variant == "full_T2":
        model = AblationEMLKAN(dim, width=4, edge_depth=2, tau=2.0, mode="full").double()
    else:
        model = AblationEMLKAN(dim, width=4, edge_depth=2, tau=2.0, mode=variant).double()
    score = fit_model(model, norm, lr)
    if score is None:
        return {"dataset": dataset, "seed": seed, "variant": variant, "lr": lr, "status": "nonfinite"}
    metrics = eval_splits(model, data, norm, scales)
    return {"dataset": dataset, "seed": seed, "variant": variant, "lr": lr, "status": "ok", "metrics": metrics}


def run_cv_item(item):
    torch.set_num_threads(1)
    dataset, fold, method = item
    dim, func, _ = FUNCTIONS[dataset]
    rng = np.random.default_rng(2026)
    x_all = rng.uniform(0.15, 2.0, size=(2000, dim)).astype(np.float64)
    y_all = np.asarray(func(x_all), dtype=np.float64).reshape(-1, 1)
    idx = np.arange(2000)
    test_mask = (idx % 5) == fold
    train_val_x, train_val_y = x_all[~test_mask], y_all[~test_mask]
    # 1280 train, 320 val, 400 test
    data = {
        "train": (train_val_x[:1280], train_val_y[:1280]),
        "val": (train_val_x[1280:], train_val_y[1280:]),
        "test": (x_all[test_mask], y_all[test_mask]),
    }
    norm, scales = normalize(data)
    best_val, best_test = float("inf"), None
    # Use representative middle/high capacity width with lr selection
    cfg_map = {
        "mlp": [(8, 0.003), (16, 0.003)],
        "eml_kan": [(4, 0.003), (8, 0.003)],
        "rbf_kan": [(4, 0.003), (8, 0.003)],
        "bspline_kan": [(4, 0.003), (8, 0.003)],
        "cheby_kan": [(4, 0.003), (8, 0.003)],
    }
    for w, lr in cfg_map[method]:
        torch.manual_seed(fold * 10 + w)
        if method == "mlp":
            m = MLP(dim, width=w, depth=2).double()
        elif method == "eml_kan":
            m = EMLKAN(dim, width=w, edge_depth=2, tau=2.0).double()
        elif method == "rbf_kan":
            m = RBFKAN(dim, width=w, grid_size=16).double()
        elif method == "bspline_kan":
            m = BSplineKAN(dim, width=w, grid_size=8, k=3).double()
        elif method == "cheby_kan":
            m = ChebyKAN(dim, width=w, degree=8).double()
        score = fit_model(m, norm, lr)
        if score is not None:
            ev = eval_splits(m, data, norm, scales)
            if ev["val"]["nmse"] < best_val:
                best_val = ev["val"]["nmse"]
                best_test = ev["test"]
    return {"dataset": dataset, "fold": fold, "method": method, "test_nmse": best_test["nmse"], "test_r2": best_test["r2"]}


def main():
    t0 = time.perf_counter()
    workers = 12
    print("1/3 Running Cheby-KAN & BSpline-KAN full 6-config grids (504 runs)...", flush=True)
    base_tasks = [(d, s, m, w, lr) for d in FUNCTIONS for s in (0, 1, 2)
                  for m in ("cheby_kan", "bspline_kan") for w in (2, 4, 8) for lr in (0.001, 0.003)]
    base_res = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(run_baseline_item, base_tasks):
            base_res.append(r)

    # Select best validation per (dataset, seed, method) and compute exact CSE tokens
    selected = {}
    for r in base_res:
        if r["status"] != "ok":
            continue
        key = (r["dataset"], r["seed"], r["method"])
        if key not in selected or r["metrics"]["val"]["nmse"] < selected[key]["metrics"]["val"]["nmse"]:
            selected[key] = r

    tok_tasks = [(k[0], k[1], k[2], v["width"], v["lr"]) for k, v in selected.items()]
    print(f"   Computing exact symbolic DAG & CSE token counts for {len(tok_tasks)} selected models...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for key, (raw_t, cse_t) in pool.map(compute_selected_tokens, tok_tasks):
            selected[key]["raw_tokens"] = raw_t
            selected[key]["cse_tokens"] = cse_t

    baseline_summary = {}
    for m in ("cheby_kan", "bspline_kan"):
        runs = [v for k, v in selected.items() if k[2] == m]
        ok_total = sum(1 for r in base_res if r["method"] == m and r["status"] == "ok")
        baseline_summary[m] = {
            "total_candidates": 252,
            "converged_candidates": ok_total,
            "selected_runs": len(runs),
            "median_test_nmse": float(np.median([x["metrics"]["test"]["nmse"] for x in runs])),
            "median_test_r2": float(np.median([x["metrics"]["test"]["r2"] for x in runs])),
            "median_ood_nmse": float(np.median([x["metrics"]["ood"]["nmse"] for x in runs])),
            "median_raw_tokens": float(np.median([x["raw_tokens"] for x in runs])),
            "median_cse_tokens": float(np.median([x["cse_tokens"] for x in runs])),
        }

    print("2/3 Running EML-KAN component ablations & Raw EML/EQL stability study (672 runs)...", flush=True)
    variants = ("raw_eml_eql", "unbounded_exp", "abs_log", "no_bypass", "no_gate", "depth_T1", "full_T2", "depth_T3")
    abl_tasks = [(d, s, v, lr) for d in FUNCTIONS for s in (0, 1, 2) for v in variants for lr in (0.001, 0.003)]
    abl_res = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(run_ablation_item, abl_tasks):
            abl_res.append(r)

    ablation_summary = {}
    for v in variants:
        v_runs = [r for r in abl_res if r["variant"] == v]
        nonfinite_count = sum(1 for r in v_runs if r["status"] != "ok")
        best_by_ds = {}
        for r in v_runs:
            if r["status"] == "ok":
                key = (r["dataset"], r["seed"])
                if key not in best_by_ds or r["metrics"]["val"]["nmse"] < best_by_ds[key]["metrics"]["val"]["nmse"]:
                    best_by_ds[key] = r
        vals = [x["metrics"]["test"]["nmse"] for x in best_by_ds.values()]
        oods = [x["metrics"]["ood"]["nmse"] for x in best_by_ds.values()]
        ablation_summary[v] = {
            "total_runs": len(v_runs),
            "nonfinite_runs": nonfinite_count,
            "convergence_rate_pct": round(100.0 * (len(v_runs) - nonfinite_count) / len(v_runs), 2),
            "feasible_task_seeds": len(best_by_ds),
            "median_test_nmse": float(np.median(vals)) if vals else None,
            "median_ood_nmse": float(np.median(oods)) if oods else None,
        }

    print("3/3 Running 5-Fold Cross-Validation across 14 tasks x 5 folds x 5 architectures (350 fold-models)...", flush=True)
    cv_methods = ("mlp", "cheby_kan", "bspline_kan", "rbf_kan", "eml_kan")
    cv_tasks = [(d, f, m) for d in FUNCTIONS for f in range(5) for m in cv_methods]
    cv_res = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(run_cv_item, cv_tasks):
            cv_res.append(r)

    cv_summary = {}
    eml_fold_vals = [r["test_nmse"] for r in sorted(cv_res, key=lambda z: (z["dataset"], z["fold"])) if r["method"] == "eml_kan"]
    for m in cv_methods:
        m_vals = [r["test_nmse"] for r in sorted(cv_res, key=lambda z: (z["dataset"], z["fold"])) if r["method"] == m]
        m_r2 = [r["test_r2"] for r in sorted(cv_res, key=lambda z: (z["dataset"], z["fold"])) if r["method"] == m]
        pval = float(stats.wilcoxon(eml_fold_vals, m_vals).pvalue) if m != "eml_kan" else 1.0
        cv_summary[m] = {
            "folds": len(m_vals),
            "median_test_nmse": float(np.median(m_vals)),
            "iqr_test_nmse": [float(np.percentile(m_vals, 25)), float(np.percentile(m_vals, 75))],
            "median_test_r2": float(np.median(m_r2)),
            "wilcoxon_p_vs_eml": pval,
        }

    elapsed = time.perf_counter() - t0
    payload = {
        "elapsed_seconds": round(elapsed, 2),
        "hardware": {
            "cpu": "AMD Ryzen 7 5700X (8C/16T)",
            "ram_gb": 64,
            "gpu_workstation": "NVIDIA GeForce RTX 3090 (24 GB VRAM)",
            "dtype": "float64",
        },
        "extended_baselines": baseline_summary,
        "ablation_study": ablation_summary,
        "five_fold_cv": cv_summary,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
