from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "experiments"))
from run_tabular import EMLKAN, MLP, RBFKAN, StableEMLNet, build_model  # noqa: E402


@dataclass
class ExportResult:
    expr: sp.Expr
    kind: str
    notes: str = ""


def f(value: float, digits: int = 6) -> sp.Float:
    return sp.Float(float(value), digits)


def standardize_symbols(x_symbols: list[sp.Symbol], x_mean: list[float], x_std: list[float]) -> list[sp.Expr]:
    return [(sym - f(mu)) / f(std) for sym, mu, std in zip(x_symbols, x_mean, x_std)]


def inverse_y_expr(y_norm: sp.Expr, y_mean: float, y_std: float) -> sp.Expr:
    return f(y_std) * y_norm + f(y_mean)


def export_eml_edge_layer(layer, inputs: list[sp.Expr], gate_threshold: float = 0.0) -> tuple[list[sp.Expr], int, int]:
    edge_variant = getattr(layer, "edge_variant", "full")
    w0 = layer.w0.detach().cpu().numpy()
    w = layer.w.detach().cpu().numpy()
    if edge_variant == "no_gate":
        gates = np.ones_like(layer.gate_logits.detach().cpu().numpy(), dtype=float)
    else:
        gates = torch.sigmoid(layer.gate_logits).detach().cpu().numpy()
    a = layer.a.detach().cpu().numpy()
    b = layer.b.detach().cpu().numpy()
    c = layer.c.detach().cpu().numpy()
    d = layer.d.detach().cpu().numpy()
    bias = layer.bias.detach().cpu().numpy()
    edge_depth, in_dim, out_dim = w.shape
    outputs: list[sp.Expr] = []
    active_terms = 0
    active_edges = 0
    for j in range(out_dim):
        out = f(bias[j])
        edge_has_active = False
        for i in range(in_dim):
            z = inputs[i]
            if edge_variant != "no_residual":
                out += f(w0[i, j]) * inputs[i]
            for t in range(edge_depth):
                u = f(a[t, i, j]) * z + f(b[t, i, j])
                v = f(c[t, i, j]) * z + f(d[t, i, j])
                exp_branch = sp.exp(f(layer.tau) * sp.tanh(u))
                log_branch = sp.log(1 + sp.log(1 + sp.exp(v)) + f(1e-6))
                if edge_variant == "no_log":
                    z = exp_branch
                elif edge_variant == "only_log":
                    z = -log_branch
                else:
                    z = exp_branch - log_branch
                gate = float(gates[t, i, j])
                if gate >= gate_threshold:
                    out += f(gate) * f(w[t, i, j]) * z
                    active_terms += 1
                    edge_has_active = True
        if edge_has_active:
            active_edges += 1
        outputs.append(out)
    return outputs, active_edges, active_terms


def export_eml_kan(model: EMLKAN, x_symbols: list[sp.Symbol], payload: dict, gate_threshold: float = 0.0) -> ExportResult:
    metrics = payload["metrics"]
    inputs = standardize_symbols(x_symbols, metrics["x_mean"], metrics["x_std"])
    h, active_edges_1, active_terms_1 = export_eml_edge_layer(model.layer1, inputs, gate_threshold)
    y_norm, active_edges_2, active_terms_2 = export_eml_edge_layer(model.layer2, h, gate_threshold)
    expr = inverse_y_expr(y_norm[0], float(metrics["y_mean"][0]), float(metrics["y_std"][0]))
    notes = f"active_edges={active_edges_1 + active_edges_2}; active_terms={active_terms_1 + active_terms_2}"
    return ExportResult(expr=expr, kind="edge exp-log", notes=notes)


def linear_expr(weight: np.ndarray, bias: np.ndarray, inputs: list[sp.Expr]) -> list[sp.Expr]:
    outputs = []
    for j in range(weight.shape[0]):
        value = f(bias[j])
        for i, x in enumerate(inputs):
            value += f(weight[j, i]) * x
        outputs.append(value)
    return outputs


def export_mlp(model: MLP, x_symbols: list[sp.Symbol], payload: dict) -> ExportResult:
    metrics = payload["metrics"]
    inputs = standardize_symbols(x_symbols, metrics["x_mean"], metrics["x_std"])
    h = inputs
    for module in model.net:
        if isinstance(module, torch.nn.Linear):
            h = linear_expr(
                module.weight.detach().cpu().numpy(),
                module.bias.detach().cpu().numpy(),
                h,
            )
        elif isinstance(module, torch.nn.SiLU):
            h = [v / (1 + sp.exp(-v)) for v in h]
        else:
            raise TypeError(f"Unsupported MLP module: {type(module)}")
    expr = inverse_y_expr(h[0], float(metrics["y_mean"][0]), float(metrics["y_std"][0]))
    return ExportResult(expr=expr, kind="dense SiLU formula", notes=f"hidden_units={len(model.net)}")


def export_rbf_layer(layer, inputs: list[sp.Expr]) -> list[sp.Expr]:
    base_weight = layer.base_weight.detach().cpu().numpy()
    spline_weight = layer.spline_weight.detach().cpu().numpy()
    bias = layer.bias.detach().cpu().numpy()
    grid = layer.grid.detach().cpu().numpy()
    gamma = float(layer.gamma)
    outputs = []
    for j in range(layer.out_dim):
        out = f(bias[j])
        for i, x in enumerate(inputs):
            out += f(base_weight[i, j]) * x
            for k, center in enumerate(grid):
                out += f(spline_weight[i, j, k]) * sp.exp(-((x - f(center)) * f(gamma)) ** 2)
        outputs.append(out)
    return outputs


def export_kan(model: RBFKAN, x_symbols: list[sp.Symbol], payload: dict) -> ExportResult:
    metrics = payload["metrics"]
    inputs = standardize_symbols(x_symbols, metrics["x_mean"], metrics["x_std"])
    h = export_rbf_layer(model.layer1, inputs)
    y = export_rbf_layer(model.layer2, h)
    expr = inverse_y_expr(y[0], float(metrics["y_mean"][0]), float(metrics["y_std"][0]))
    terms = model.layer1.in_dim * model.layer1.out_dim * model.layer1.grid_size
    terms += model.layer2.in_dim * model.layer2.out_dim * model.layer2.grid_size
    return ExportResult(expr=expr, kind="RBF basis expansion", notes=f"basis_terms={terms}")


def export_stable_eml(model: StableEMLNet, x_symbols: list[sp.Symbol], payload: dict) -> ExportResult:
    metrics = payload["metrics"]
    h = standardize_symbols(x_symbols, metrics["x_mean"], metrics["x_std"])
    first = model.in_proj[0]
    h = linear_expr(first.weight.detach().cpu().numpy(), first.bias.detach().cpu().numpy(), h)
    h = [v / (1 + sp.exp(-v)) for v in h]
    for block in model.blocks:
        u = linear_expr(block.u.weight.detach().cpu().numpy(), block.u.bias.detach().cpu().numpy(), h)
        v = linear_expr(block.v.weight.detach().cpu().numpy(), block.v.bias.detach().cpu().numpy(), h)
        z = [
            sp.exp(f(block.tau) * sp.tanh(ui)) - sp.log(1 + sp.log(1 + sp.exp(vi)) + f(1e-6))
            for ui, vi in zip(u, v)
        ]
        proposal = linear_expr(block.out.weight.detach().cpu().numpy(), block.out.bias.detach().cpu().numpy(), z)
        proposal = [sp.tanh(p) for p in proposal]
        mixed = [(f(1.0 - block.eta) * old + f(block.eta) * new) for old, new in zip(h, proposal)]
        weight = block.norm.weight.detach().cpu().numpy()
        bias = block.norm.bias.detach().cpu().numpy()
        # Exact LayerNorm would introduce a dense square-root expression across all hidden units.
        # For export auditing, keep its affine part and mark the normalization approximation.
        h = [f(weight[i]) * mixed[i] + f(bias[i]) for i in range(len(mixed))]
    y = linear_expr(model.head.weight.detach().cpu().numpy(), model.head.bias.detach().cpu().numpy(), h)
    expr = inverse_y_expr(y[0], float(metrics["y_mean"][0]), float(metrics["y_std"][0]))
    return ExportResult(expr=expr, kind="block exp-log formula", notes="LayerNorm exported by affine parameters only")


def build_from_payload(payload: dict):
    args = payload.get("args", {})
    in_dim = len(payload["metrics"]["x_mean"])
    return build_model(
        payload["model"],
        in_dim,
        int(args.get("width", 8)),
        int(args.get("depth", 2)),
        float(args.get("tau", 2.0)),
        float(args.get("eta", 0.35)),
        args.get("exp_mode", "bounded_tanh"),
        float(args.get("clip_m", 8.0)),
        args.get("w0_init", "normal"),
        args.get("edge_variant", "full"),
    )


def load_model_from_json(json_path: Path):
    import json

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(json_path.with_suffix(".pt"), map_location="cpu")
    model = build_from_payload(payload)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, payload


def export_model(model, payload: dict, gate_threshold: float = 0.0) -> ExportResult:
    in_dim = len(payload["metrics"]["x_mean"])
    symbols = [sp.Symbol(f"x{i + 1}") for i in range(in_dim)]
    method = payload["model"]
    if method == "eml_kan":
        return export_eml_kan(model, symbols, payload, gate_threshold)
    if method == "mlp":
        return export_mlp(model, symbols, payload)
    if method == "kan":
        return export_kan(model, symbols, payload)
    if method == "stable_eml":
        return export_stable_eml(model, symbols, payload)
    raise ValueError(f"Unsupported model for export: {method}")


def torch_predict(model, payload: dict, x_raw: np.ndarray) -> np.ndarray:
    x_mean = np.asarray(payload["metrics"]["x_mean"], dtype=np.float32).reshape(1, -1)
    x_std = np.asarray(payload["metrics"]["x_std"], dtype=np.float32).reshape(1, -1)
    y_mean = float(np.asarray(payload["metrics"]["y_mean"], dtype=np.float32).reshape(-1)[0])
    y_std = float(np.asarray(payload["metrics"]["y_std"], dtype=np.float32).reshape(-1)[0])
    x_norm = (x_raw.astype(np.float32) - x_mean) / x_std
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_norm), 4096):
            bx = torch.from_numpy(x_norm[start : start + 4096])
            preds.append(model(bx).detach().cpu().numpy())
    return np.vstack(preds).reshape(-1) * y_std + y_mean


def sympy_predict(expr: sp.Expr, x_raw: np.ndarray) -> np.ndarray:
    symbols = [sp.Symbol(f"x{i + 1}") for i in range(x_raw.shape[1])]
    fn = sp.lambdify(symbols, expr, modules=["numpy"])
    values = fn(*[x_raw[:, i] for i in range(x_raw.shape[1])])
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape == ():
        arr = np.full(len(x_raw), float(arr))
    return arr.reshape(-1)

