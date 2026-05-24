from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GraphExport:
    text: str
    kind: str
    tokens: int
    ast_nodes: int
    constants: int
    operators: int
    exp_log_ops: int
    scientific_operator_ratio: float
    tree_depth: int
    piecewise_segments: int
    basis_terms: int
    coefficient_count: int
    active_edges: int = 0
    active_terms: int = 0


def fmt(value: float) -> str:
    return f"{float(value):.8g}"


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+\.\d+|\d+|[+\-*/^(),=;]", text))


def make_stats(
    text: str,
    kind: str,
    constants: int,
    operators: int,
    exp_log_ops: int,
    scientific_ops: int,
    tree_depth: int,
    basis_terms: int = 0,
    active_edges: int = 0,
    active_terms: int = 0,
) -> GraphExport:
    tokens = token_count(text)
    return GraphExport(
        text=text,
        kind=kind,
        tokens=tokens,
        ast_nodes=max(tokens, operators + constants),
        constants=constants,
        operators=operators,
        exp_log_ops=exp_log_ops,
        scientific_operator_ratio=scientific_ops / operators if operators else 0.0,
        tree_depth=tree_depth,
        piecewise_segments=0,
        basis_terms=basis_terms,
        coefficient_count=constants,
        active_edges=active_edges,
        active_terms=active_terms,
    )


def export_mlp_graph(model, variable_names: list[str]) -> GraphExport:
    lines = []
    inputs = variable_names
    constants = 0
    operators = 0
    sci_ops = 0
    activation_count = 0
    layer_id = 0
    h = inputs
    for module in model.net:
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy()
            next_h = []
            for j in range(weight.shape[0]):
                name = "y_norm" if weight.shape[0] == 1 and module is model.net[-1] else f"h{layer_id}_{j}"
                terms = [fmt(bias[j])]
                constants += 1
                for i, prev in enumerate(h):
                    terms.append(f"{fmt(weight[j, i])}*{prev}")
                    constants += 1
                    operators += 1
                    sci_ops += 1
                lines.append(f"{name} = " + " + ".join(terms))
                operators += max(len(terms) - 1, 0)
                sci_ops += max(len(terms) - 1, 0)
                next_h.append(name)
            h = next_h
            layer_id += 1
        elif isinstance(module, torch.nn.SiLU):
            h = [f"silu({name})" for name in h]
            activation_count += len(h)
            operators += len(h)
    text = ";\n".join(lines) + ";\ny = y_std*y_norm + y_mean"
    constants += 2
    operators += 2
    sci_ops += 2
    return make_stats(
        text=text,
        kind="dense SiLU computation graph",
        constants=constants,
        operators=operators,
        exp_log_ops=0,
        scientific_ops=sci_ops,
        tree_depth=layer_id + activation_count // max(len(variable_names), 1),
    )


def export_kan_graph(model, variable_names: list[str]) -> GraphExport:
    lines = []
    constants = 0
    operators = 0
    sci_ops = 0
    exp_log_ops = 0
    basis_terms = 0

    def layer_lines(layer, inputs, prefix):
        nonlocal constants, operators, sci_ops, exp_log_ops, basis_terms
        weight = layer.base_weight.detach().cpu().numpy()
        spline = layer.spline_weight.detach().cpu().numpy()
        bias = layer.bias.detach().cpu().numpy()
        grid = layer.grid.detach().cpu().numpy()
        outputs = []
        for j in range(layer.out_dim):
            name = f"{prefix}_{j}" if layer.out_dim > 1 else prefix
            terms = [fmt(bias[j])]
            constants += 1
            for i, prev in enumerate(inputs):
                terms.append(f"{fmt(weight[i, j])}*{prev}")
                constants += 1
                operators += 1
                sci_ops += 1
                for k, center in enumerate(grid):
                    terms.append(f"{fmt(spline[i, j, k])}*rbf({prev};{fmt(center)},{fmt(layer.gamma)})")
                    constants += 3
                    operators += 4
                    sci_ops += 3
                    exp_log_ops += 1
                    basis_terms += 1
            lines.append(f"{name} = " + " + ".join(terms))
            operators += max(len(terms) - 1, 0)
            sci_ops += max(len(terms) - 1, 0)
            outputs.append(name)
        return outputs

    h = layer_lines(model.layer1, variable_names, "h")
    y = layer_lines(model.layer2, h, "y_norm")
    text = ";\n".join(lines) + ";\ny = y_std*y_norm + y_mean"
    constants += 2
    operators += 2
    sci_ops += 2
    return make_stats(
        text=text,
        kind="RBF edge basis expansion",
        constants=constants,
        operators=operators,
        exp_log_ops=exp_log_ops,
        scientific_ops=sci_ops,
        tree_depth=4,
        basis_terms=basis_terms,
    )


def export_eml_kan_graph(model, variable_names: list[str], gate_threshold: float = 0.0) -> GraphExport:
    lines = []
    constants = 0
    operators = 0
    sci_ops = 0
    exp_log_ops = 0
    active_edges = 0
    active_terms = 0

    def layer_lines(layer, inputs, prefix):
        nonlocal constants, operators, sci_ops, exp_log_ops, active_edges, active_terms
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
        outputs = []
        edge_depth, in_dim, out_dim = w.shape
        for j in range(out_dim):
            out_name = f"{prefix}_{j}" if out_dim > 1 else prefix
            terms = [fmt(bias[j])]
            constants += 1
            for i, prev in enumerate(inputs):
                if edge_variant != "no_residual":
                    terms.append(f"{fmt(w0[i, j])}*{prev}")
                    constants += 1
                    operators += 1
                    sci_ops += 1
                z_prev = prev
                edge_active = False
                for t in range(edge_depth):
                    z_name = f"{prefix}z_{i}_{j}_{t}"
                    exp_text = f"exp({fmt(layer.tau)}*tanh({fmt(a[t, i, j])}*{z_prev}+{fmt(b[t, i, j])}))"
                    log_text = f"log(1+softplus({fmt(c[t, i, j])}*{z_prev}+{fmt(d[t, i, j])}))"
                    if edge_variant == "no_log":
                        rhs = exp_text
                        constants += 3
                        operators += 4
                        sci_ops += 3
                        exp_log_ops += 1
                    elif edge_variant == "only_log":
                        rhs = f"-{log_text}"
                        constants += 3
                        operators += 4
                        sci_ops += 3
                        exp_log_ops += 1
                    else:
                        rhs = f"{exp_text} - {log_text}"
                        constants += 6
                        operators += 8
                        sci_ops += 6
                        exp_log_ops += 2
                    lines.append(f"{z_name} = {rhs}")
                    if float(gates[t, i, j]) >= gate_threshold:
                        terms.append(f"{fmt(gates[t, i, j])}*{fmt(w[t, i, j])}*{z_name}")
                        constants += 2
                        operators += 2
                        sci_ops += 2
                        active_terms += 1
                        edge_active = True
                    z_prev = z_name
                if edge_active:
                    active_edges += 1
            lines.append(f"{out_name} = " + " + ".join(terms))
            operators += max(len(terms) - 1, 0)
            sci_ops += max(len(terms) - 1, 0)
            outputs.append(out_name)
        return outputs

    h = layer_lines(model.layer1, variable_names, "h")
    _ = layer_lines(model.layer2, h, "y_norm")
    text = ";\n".join(lines) + ";\ny = y_std*y_norm + y_mean"
    constants += 2
    operators += 2
    sci_ops += 2
    return make_stats(
        text=text,
        kind="edge exp-log equation DAG",
        constants=constants,
        operators=operators,
        exp_log_ops=exp_log_ops,
        scientific_ops=sci_ops,
        tree_depth=2 * model.layer1.edge_depth + 2,
        active_edges=active_edges,
        active_terms=active_terms,
    )


def export_stable_eml_graph(model, variable_names: list[str]) -> GraphExport:
    lines = []
    constants = 0
    operators = 0
    sci_ops = 0
    exp_log_ops = 0

    def linear_lines(weight, bias, inputs, prefix):
        nonlocal constants, operators, sci_ops, lines
        outputs = []
        for j in range(weight.shape[0]):
            name = f"{prefix}_{j}"
            terms = [fmt(bias[j])]
            constants += 1
            for i, prev in enumerate(inputs):
                terms.append(f"{fmt(weight[j, i])}*{prev}")
                constants += 1
                operators += 1
                sci_ops += 1
            lines.append(f"{name} = " + " + ".join(terms))
            operators += max(len(terms) - 1, 0)
            sci_ops += max(len(terms) - 1, 0)
            outputs.append(name)
        return outputs

    first = model.in_proj[0]
    h = linear_lines(
        first.weight.detach().cpu().numpy(),
        first.bias.detach().cpu().numpy(),
        variable_names,
        "h0pre",
    )
    h = [f"silu({name})" for name in h]
    operators += len(h)
    for block_id, block in enumerate(model.blocks):
        u = linear_lines(block.u.weight.detach().cpu().numpy(), block.u.bias.detach().cpu().numpy(), h, f"b{block_id}u")
        v = linear_lines(block.v.weight.detach().cpu().numpy(), block.v.bias.detach().cpu().numpy(), h, f"b{block_id}v")
        z = []
        for i, (ui, vi) in enumerate(zip(u, v)):
            name = f"b{block_id}z_{i}"
            lines.append(f"{name} = exp(2*tanh({ui})) - log(1+softplus({vi}))")
            constants += 2
            operators += 5
            sci_ops += 4
            exp_log_ops += 2
            z.append(name)
        p = linear_lines(block.out.weight.detach().cpu().numpy(), block.out.bias.detach().cpu().numpy(), z, f"b{block_id}p")
        next_h = []
        weight = block.norm.weight.detach().cpu().numpy()
        bias = block.norm.bias.detach().cpu().numpy()
        for i, pi in enumerate(p):
            name = f"h{block_id + 1}_{i}"
            lines.append(f"{name} = LayerNormAffine({fmt(1.0 - block.eta)}*{h[i]} + {fmt(block.eta)}*tanh({pi}); {fmt(weight[i])}, {fmt(bias[i])})")
            constants += 4
            operators += 4
            sci_ops += 2
            next_h.append(name)
        h = next_h
    head = linear_lines(model.head.weight.detach().cpu().numpy(), model.head.bias.detach().cpu().numpy(), h, "y_norm")
    lines.append("y = y_std*y_norm_0 + y_mean")
    constants += 2
    operators += 2
    sci_ops += 2
    text = ";\n".join(lines)
    return make_stats(
        text=text,
        kind="block exp-log neural circuit",
        constants=constants,
        operators=operators,
        exp_log_ops=exp_log_ops,
        scientific_ops=sci_ops,
        tree_depth=len(model.blocks) * 4 + 2,
    )


def export_graph(model, method: str, variable_names: list[str], gate_threshold: float = 0.0) -> GraphExport:
    if method == "mlp":
        return export_mlp_graph(model, variable_names)
    if method == "kan":
        return export_kan_graph(model, variable_names)
    if method == "stable_eml":
        return export_stable_eml_graph(model, variable_names)
    if method == "eml_kan":
        return export_eml_kan_graph(model, variable_names, gate_threshold)
    raise ValueError(method)
