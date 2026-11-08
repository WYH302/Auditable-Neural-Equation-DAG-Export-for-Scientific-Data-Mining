"""Exact scalar exports, explicit lexical tokens, and dependency-valid CSE.

The follow-up uses its own stated protocol and does not overwrite historical
export statistics. In particular, numeric literals include scientific notation,
and the model's 1e-6 log-input offset is retained.
"""
import ast
import math
import re

import numpy as np
import sympy as sp
import torch


NUMBER = r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
TOKEN = re.compile(NUMBER + r"|[A-Za-z_][A-Za-z_0-9]*|\*\*|[+\-*/(),=;]")


def token_count(text):
    compact = re.sub(r"\s+", "", text)
    tokens = TOKEN.findall(text)
    if "".join(tokens) != compact:
        raise ValueError("Unsupported character in tokenized DAG")
    return len(tokens)


def fmt(value):
    if hasattr(value, "detach"):
        value = value.detach()
    return format(float(value), ".17g")


def softplus(x):
    return np.logaddexp(0.0, x)


def silu(x):
    with np.errstate(over="ignore"):
        return x / (1.0 + np.exp(-x))


def rbf(x, center, gamma):
    return np.exp(-((x - center) * gamma) ** 2)


FUNCTIONS = {"exp": np.exp, "log": np.log, "log1p": np.log1p,
             "tanh": np.tanh, "softplus": softplus, "silu": silu,
             "rbf": rbf, "logaddexp": np.logaddexp}


def evaluate(lines, inputs, trace=False):
    env = dict(FUNCTIONS)
    x = np.asarray(inputs, dtype=np.float64)
    for i in range(x.shape[1]):
        env[f"x{i + 1}"] = x[:, i]
    values = {}
    permitted = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
                 ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
                 ast.Pow, ast.USub, ast.UAdd)
    for line in lines:
        lhs, rhs = line.split("=", 1)
        name = lhs.strip()
        if not name.isidentifier():
            raise ValueError(name)
        tree = ast.parse(rhs.strip(), mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, permitted):
                raise ValueError(type(node).__name__)
            if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name)
                or node.func.id not in FUNCTIONS or node.keywords
            ):
                raise ValueError("Unsupported primitive")
        value = eval(compile(tree, "<exported-dag>", "eval"), {"__builtins__": {}}, env)
        env[name] = value
        values[name] = np.broadcast_to(np.asarray(value), (x.shape[0],)).copy()
    if "y" not in values:
        raise ValueError("Missing output assignment")
    return values if trace else values["y"]


def export(model, method, scales):
    """Use the original scalar operation order and 17-digit coefficients."""
    d = len(scales["x_mean"])
    lines = [f"x{i+1}_norm = (x{i+1} - ({fmt(scales['x_mean'][i])}))/({fmt(scales['x_std'][i])})"
             for i in range(d)]
    core_start = len(lines)
    inputs = [f"x{i+1}_norm" for i in range(d)]
    if method == "mlp":
        layer_id = 0
        h = inputs
        modules = list(model.net)
        for module in modules:
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.detach().cpu().numpy()
                bias = module.bias.detach().cpu().numpy()
                outputs = []
                for j in range(weight.shape[0]):
                    name = "y_norm" if module is modules[-1] else f"h{layer_id}_{j}"
                    rhs = f"({fmt(bias[j])})" + "".join(
                        f" + ({fmt(weight[j,i])})*({v})" for i, v in enumerate(h))
                    lines.append(f"{name} = {rhs}")
                    outputs.append(name)
                h = outputs
                layer_id += 1
            elif isinstance(module, torch.nn.SiLU):
                h = [f"silu({v})" for v in h]
            else:
                raise TypeError(type(module).__name__)
    elif method == "eml_kan":
        h = inputs
        for layer_id, layer in enumerate((model.layer1, model.layer2)):
            outputs = []
            for j in range(layer.w0.shape[1]):
                name = "y_norm" if layer_id == 1 else f"h{layer_id}_{j}"
                terms = [f"({fmt(layer.bias[j])})"]
                for i, inp in enumerate(h):
                    terms.append(f"({fmt(layer.w0[i,j])})*({inp})")
                    prev = inp
                    for t in range(layer.edge_depth):
                        zname = f"z{layer_id}_{i}_{j}_{t}"
                        a, b, c, dd = (fmt(getattr(layer, n)[t,i,j]) for n in ("a","b","c","d"))
                        lines.append(f"{zname} = exp(({fmt(layer.tau)})*tanh(({a})*({prev}) + ({b}))) - log1p(softplus(({c})*({prev}) + ({dd})) + 1e-6)")
                        gate = float(torch.sigmoid(layer.gate_logits[t,i,j]).detach())
                        terms.append(f"({fmt(gate)})*({fmt(layer.w[t,i,j])})*{zname}")
                        prev = zname
                lines.append(f"{name} = " + " + ".join(terms))
                outputs.append(name)
            h = outputs
    elif method == "kan":
        h = inputs
        for layer_id, layer in enumerate((model.layer1, model.layer2)):
            outputs = []
            for j in range(layer.out_dim):
                name = "y_norm" if layer_id == 1 else f"h{layer_id}_{j}"
                terms = [f"({fmt(layer.bias[j])})"]
                for i, inp in enumerate(h):
                    terms.append(f"({fmt(layer.base_weight[i,j])})*({inp})")
                    for k, center in enumerate(layer.grid):
                        terms.append(f"({fmt(layer.spline_weight[i,j,k])})*rbf({inp},{fmt(center)},{fmt(layer.gamma)})")
                lines.append(f"{name} = " + " + ".join(terms))
                outputs.append(name)
            h = outputs
    else:
        raise ValueError(method)
    core_end = len(lines)
    lines.append(f"y = ({fmt(scales['y_std'])})*y_norm + ({fmt(scales['y_mean'])})")
    return lines, lines[core_start:core_end]


class ExpandPrimitives(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        f = node.func.id
        args = node.args
        if f == "silu":
            return ast.BinOp(
                left=args[0],op=ast.Div(),right=ast.BinOp(
                    left=ast.Constant(1),op=ast.Add(),right=ast.Call(func=ast.Name(id="exp",ctx=ast.Load()),args=[ast.UnaryOp(op=ast.USub(),operand=args[0])],keywords=[])))
        if f == "softplus":
            return ast.Call(func=ast.Name(id="logaddexp",ctx=ast.Load()),args=[ast.Constant(0),args[0]],keywords=[])
        if f == "rbf":
            inside = ast.BinOp(left=ast.BinOp(left=args[0],op=ast.Sub(),right=args[1]),op=ast.Mult(),right=args[2])
            power = ast.BinOp(left=inside,op=ast.Pow(),right=ast.Constant(2))
            return ast.Call(func=ast.Name(id="exp",ctx=ast.Load()),args=[ast.UnaryOp(op=ast.USub(),operand=power)],keywords=[])
        return node


def expand_primitives(lines):
    out = []
    for line in lines:
        lhs, rhs = line.split("=", 1)
        tree = ExpandPrimitives().visit(ast.parse(rhs.strip(), mode="eval"))
        ast.fix_missing_locations(tree)
        out.append(lhs.strip()+" = "+ast.unparse(tree.body))
    return out


def cse(lines):
    """CSE followed by dependency sorting; validates all emitted definitions."""
    function_map = {k: sp.Function(k) for k in FUNCTIONS}
    names, exprs = [], []
    for line in lines:
        lhs, rhs = line.split("=", 1)
        names.append(lhs.strip())
        exprs.append(sp.sympify(rhs.strip(),locals=function_map))
    replacements, reduced = sp.cse(exprs, symbols=sp.numbered_symbols("common_"), order="canonical")
    pending = [(str(n),e) for n,e in replacements] + list(zip(names,reduced))
    if len({n for n,_ in pending}) != len(pending):
        raise ValueError("CSE name collision")
    produced = {n for n,_ in pending}
    out, available = [], set()
    while pending:
        next_pending = []
        for name, expr in pending:
            deps = {str(s) for s in expr.free_symbols} & produced
            if deps <= available:
                out.append(name+" = "+sp.sstr(expr,full_prec=True))
                available.add(name)
            else:
                next_pending.append((name,expr))
        if len(next_pending) == len(pending):
            raise ValueError("Cyclic CSE dependency")
        pending = next_pending
    return out


def measured(model, method, scales, x, y_checkpoint):
    lines, core = export(model, method, scales)
    expanded = expand_primitives(lines)
    optimized = cse(expanded)
    traces = {k:evaluate(v,x) for k,v in dict(raw=lines,expanded=expanded,cse=optimized).items()}
    abs_errors = {k:float(np.max(np.abs(v-y_checkpoint))) for k,v in traces.items()}
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(y_checkpoint))))
    if not all(math.isfinite(v) and v <= tolerance for v in abs_errors.values()):
        raise ValueError(f"Export fidelity failure: {abs_errors}, tolerance {tolerance}")
    return dict(raw_tokens=token_count(";\n".join(lines)),core_tokens=token_count(";\n".join(core)),
                expanded_tokens=token_count(";\n".join(expanded)),cse_tokens=token_count(";\n".join(optimized)),
                assignments=len(lines),cse_assignments=len(optimized),max_abs_errors=abs_errors,
                fidelity_tolerance=tolerance), dict(raw=lines,expanded=expanded,cse=optimized)
