# Model definitions copied verbatim from experiments/run_tabular.py.
import torch
from torch import nn


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
    ):
        super().__init__()
        shape = (edge_depth, in_dim, out_dim)
        self.edge_depth = edge_depth
        self.tau = tau
        self.exp_mode = exp_mode
        self.clip_m = clip_m
        self.w0_init = w0_init
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
        acc = z * self.w0.unsqueeze(0)
        for t in range(self.edge_depth):
            u = self.a[t].unsqueeze(0) * z + self.b[t].unsqueeze(0)
            v = self.c[t].unsqueeze(0) * z + self.d[t].unsqueeze(0)
            exp_branch = torch.exp(self.stable_exp_input(u))
            log_branch = torch.log1p(torch.nn.functional.softplus(v) + 1e-6)
            z = exp_branch - log_branch
            gate = torch.sigmoid(self.gate_logits[t]).unsqueeze(0)
            acc = acc + gate * self.w[t].unsqueeze(0) * z
        return acc.sum(dim=1) + self.bias

    def regularization_loss(self):
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
        )
        self.layer2 = StableEMLEdgeLayer(
            width,
            1,
            edge_depth=edge_depth,
            tau=tau,
            exp_mode=exp_mode,
            clip_m=clip_m,
            w0_init=w0_init,
        )

    def forward(self, x):
        h = self.layer1(x)
        return self.layer2(h)

    def regularization_loss(self):
        losses = [self.layer1.regularization_loss(), self.layer2.regularization_loss()]
        return tuple(sum(parts) for parts in zip(*losses))
