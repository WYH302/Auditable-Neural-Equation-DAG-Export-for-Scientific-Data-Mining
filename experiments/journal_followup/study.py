"""Resumable validation-selected capacity study using original model classes.

All numerical work is local. Each trial commits a checkpoint before export,
and commits a result only after all export variants pass numerical checks.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

from datasets import FUNCTIONS
from models import EMLKAN, MLP, RBFKAN
from graph import measured


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = PROJECT / "results_v2/journal_followup"
METHODS = ("mlp", "eml_kan", "kan")
WIDTHS = {"mlp": (4, 8, 16), "eml_kan": (2, 4, 8), "kan": (2, 4, 8)}
LEARNING_RATES = (0.001, 0.003)
SIZES = dict(train=1200, val=800, test=1000, ood=1000)
PROTOCOL = dict(
    version=1, seeds=[0, 1, 2], methods=list(METHODS), widths=WIDTHS,
    learning_rates=LEARNING_RATES, maximum_full_batch_steps=600,
    validation_every=20, early_stopping_patience_steps=120,
    weight_decay=0.0001, gate_penalty=0.0001,
    coefficient_penalty=0.00001, constant_penalty=0.000001,
    gradient_clip=1.0, recurrence_depth=2, mlp_hidden_depth=2,
    tau=2.0, dtype="float64", device="cpu", splits=SIZES,
    search_candidates_per_method_task_seed=6,
    selection="Validation NMSE only; test and OOD are never used for selection.",
    comparison="Complete candidate frontier; fixed validation ceilings 1e-2, 1e-3, 1e-4; nearest-validation-NMSE pairs within factor 1.25.",
    numeric_literals="17 significant digits; signed exponent is part of a numeric literal",
    primary_graph="Common expanded primitive vocabulary, followed by dependency-sorted global CSE.",
    primary_primitives=["exp", "log1p", "tanh", "logaddexp", "+", "-", "*", "/", "**"],
    results_scope="New follow-up study; not a reconstruction of historical GPU checkpoints or historical token counts.",
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint():
    h = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode())
    h.update((HERE / "models.py").read_bytes())
    h.update((HERE / "datasets.py").read_bytes())
    return h.hexdigest()


def make_data(dataset, seed):
    dimension, function, _ = FUNCTIONS[dataset]
    rng = np.random.default_rng(seed)
    data = {}
    for split, n in SIZES.items():
        low, high = (0.05, 3.0) if split == "ood" else (0.15, 2.0)
        x = rng.uniform(low, high, size=(n, dimension)).astype(np.float64)
        y = np.asarray(function(x), dtype=np.float64).reshape(-1, 1)
        data[split] = (x, y)
    return data


def create_model(method, dimension, width, seed):
    torch.manual_seed(seed)
    if method == "mlp":
        return MLP(dimension, width=width, depth=2).double()
    if method == "eml_kan":
        return EMLKAN(dimension, width=width, edge_depth=2, tau=2.0).double()
    if method == "kan":
        return RBFKAN(dimension, width=width, grid_size=16).double()
    raise ValueError(method)


def normalize(data):
    tx, ty = data["train"]
    xm, xs = tx.mean(axis=0), tx.std(axis=0)
    ym, ys = float(ty.mean()), float(ty.std())
    xs = np.maximum(xs, 1e-12)
    ys = max(ys, 1e-12)
    normalized = {name: (torch.from_numpy((x-xm)/xs), torch.from_numpy((y-ym)/ys))
                  for name, (x,y) in data.items()}
    return normalized, dict(x_mean=xm.tolist(),x_std=xs.tolist(),y_mean=ym,y_std=ys)


def id_for(dataset, seed, method, width, lr):
    return f"{dataset}__s{seed}__{method}__w{width}__lr{lr:g}"


def train(model, normalized, scales, max_steps=600):
    optimizer = torch.optim.AdamW(model.parameters(), lr=train.learning_rate,
                                 weight_decay=PROTOCOL["weight_decay"])
    x, y = normalized["train"]
    vx, vy = normalized["val"]
    best_score, best_state, best_step = float("inf"), None, 0
    start = time.perf_counter()
    nonfinite = 0
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x) - y) ** 2)
        if hasattr(model, "regularization_loss"):
            gate, coeff, const = model.regularization_loss()
            loss = loss + 1e-4*gate + 1e-5*coeff + 1e-6*const
        if not bool(torch.isfinite(loss)):
            nonfinite += 1
            break
        loss.backward()
        if not all(p.grad is None or bool(torch.all(torch.isfinite(p.grad))) for p in model.parameters()):
            nonfinite += 1
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                value = float(torch.mean((model(vx)-vy)**2))
            if value < best_score:
                best_score, best_step = value, step
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if step - best_step >= 120:
                break
            if time.perf_counter() - start > 120:
                raise TimeoutError("One training trial exceeded 120 seconds")
    if nonfinite or best_state is None:
        raise ValueError("Nonfinite loss/gradient or no valid validation checkpoint")
    model.load_state_dict(best_state)
    return dict(best_step=best_step,steps_completed=step,training_seconds=time.perf_counter()-start,
                nonfinite_steps=nonfinite,best_normalized_validation_mse=best_score)


@torch.no_grad()
def predictions(model, normalized, scales):
    model.eval()
    return {name:model(x).numpy().reshape(-1)*scales["y_std"]+scales["y_mean"]
            for name,(x,_) in normalized.items()}


def run_trial(item, pilot_steps=None):
    torch.set_num_threads(1)
    dataset, seed, method, width, lr = item
    identifier = id_for(*item)
    result_path = OUT / "trials" / (identifier + ".json")
    checkpoint_path = OUT / "checkpoints" / (identifier + ".pt")
    export_hash = hashlib.sha256((HERE / "graph.py").read_bytes()).hexdigest()
    if pilot_steps is None and result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("protocol_sha256") == fingerprint() and saved.get("export_sha256") == export_hash:
            return dict(id=identifier, cached=True, status=saved["status"])
    data = make_data(dataset, seed)
    norm, scales = normalize(data)
    model = create_model(method, data["train"][0].shape[1], width, seed)
    cfg = dict(dataset=dataset,seed=seed,method=method,width=width,learning_rate=lr)
    base = dict(id=identifier,configuration=cfg,protocol_sha256=fingerprint(),export_sha256=export_hash)
    try:
        if pilot_steps is None and checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if ckpt["protocol_sha256"] != fingerprint():
                raise ValueError("Checkpoint belongs to a different protocol")
            model.load_state_dict(ckpt["state_dict"])
            fit = ckpt["training"]
        else:
            train.learning_rate = lr
            fit = train(model, norm, scales, max_steps=pilot_steps or 600)
            if pilot_steps is None:
                checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
                temp = checkpoint_path.with_suffix(".partial")
                torch.save(dict(state_dict=model.state_dict(),configuration=cfg,scales=scales,
                                protocol_sha256=fingerprint(),training=fit),temp)
                temp.replace(checkpoint_path)
        pred = predictions(model,norm,scales)
        metrics = {}
        for split,(_,y) in data.items():
            truth = y.reshape(-1)
            error = float(np.mean((pred[split]-truth)**2))
            variance = max(float(np.var(truth)),1e-15)
            metrics[split] = dict(mse=error,nmse=error/variance,r2=1-error/variance)
        check_x = np.vstack([data["test"][0],data["ood"][0]])
        check_pred = np.concatenate([pred["test"],pred["ood"]])
        start = time.perf_counter()
        graph_metrics, graphs = measured(model,method,scales,check_x,check_pred)
        graph_metrics["export_and_cse_seconds"] = time.perf_counter()-start
        result = dict(**base,status="ok",training=fit,metrics=metrics,graph=graph_metrics,
                      parameters=sum(p.numel() for p in model.parameters()),scales=scales,
                      target_expression=FUNCTIONS[dataset][2],evaluation_rows=len(check_x))
        if pilot_steps is None:
            dump(result_path,result)
            dump(OUT/"graphs"/(identifier+".json"),graphs)
        return dict(id=identifier,status="ok",seconds=fit["training_seconds"],
                    validation_nmse=metrics["val"]["nmse"],test_nmse=metrics["test"]["nmse"],
                    raw_tokens=graph_metrics["raw_tokens"],cse_tokens=graph_metrics["cse_tokens"],
                    export_seconds=graph_metrics["export_and_cse_seconds"])
    except Exception as exc:
        failure = dict(**base,status="failed",error=repr(exc))
        if pilot_steps is None:
            dump(result_path,failure)
        return failure


def pilot():
    output = []
    for method in METHODS:
        result = run_trial(("mixed_exp_log",0,method,8,0.003),pilot_steps=40)
        output.append(result)
        print(json.dumps(result),flush=True)
    dump(OUT/"pilot.json",output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol = dict(PROTOCOL,protocol_sha256=fingerprint(),datasets=list(FUNCTIONS),
                    numpy_version=np.__version__,torch_version=torch.__version__)
    existing = OUT/"protocol.json"
    if existing.exists() and json.loads(existing.read_text())["protocol_sha256"] != fingerprint():
        raise RuntimeError("An incompatible study already exists at the output path")
    dump(existing,protocol)
    for dataset in FUNCTIONS:
        for seed in (0,1,2):
            data = make_data(dataset,seed)
            p = OUT/"data"/(f"{dataset}__s{seed}.npz")
            p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists():
                np.savez_compressed(p,**{f"{split}_{kind}":v for split,(x,y) in data.items()
                                       for kind,v in (("x",x),("y",y))})
    if args.pilot:
        pilot()
        return
    from concurrent.futures import ProcessPoolExecutor,as_completed
    tasks = [(d,s,m,w,lr) for d in FUNCTIONS for s in (0,1,2) for m in METHODS
             for w in WIDTHS[m] for lr in LEARNING_RATES]
    start = time.perf_counter()
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_trial,task) for task in tasks]
        for i,f in enumerate(as_completed(futures),1):
            result = f.result()
            failures += int(result["status"] != "ok")
            if i % 6 == 0 or result["status"] != "ok":
                print(json.dumps(dict(completed=i,total=len(tasks),failures=failures,
                                      elapsed_seconds=round(time.perf_counter()-start,1),last=result)),flush=True)
    dump(OUT/"completion.json",dict(completed=len(tasks),failed=failures,
                                   elapsed_seconds=time.perf_counter()-start))


if __name__ == "__main__":
    main()
