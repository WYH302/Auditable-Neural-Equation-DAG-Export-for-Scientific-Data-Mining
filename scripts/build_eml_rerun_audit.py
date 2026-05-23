from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2"
PAPER = ROOT / "paper_elscas"


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        mantissa, exp = f"{value:.2e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exp)}}}"
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.5f}"


def texnum(value: float) -> str:
    text = fmt(value)
    return f"${text}$" if r"\times" in text else text


def count_params_from_record(payload: dict) -> int:
    import sys
    sys.path.append(str(ROOT / "experiments"))
    from run_tabular import build_model

    args = payload["args"]
    in_dim = len(payload["metrics"]["x_mean"])
    model = build_model(
        payload["model"],
        in_dim,
        int(args["width"]),
        int(args["depth"]),
        float(args.get("tau", 2.0)),
        float(args.get("eta", 0.35)),
        args.get("exp_mode", "bounded_tanh"),
        float(args.get("clip_m", 8.0)),
        args.get("w0_init", "normal"),
    )
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def srsd_strong_summary() -> dict:
    paths = sorted((RESULTS / "eml_rerun_strong_srsd_w80" / "srsd").glob("*.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    mses = [p["metrics"]["test"]["mse"] for p in payloads]
    r2s = [p["metrics"]["test"]["r2"] for p in payloads]
    params = [count_params_from_record(p) for p in payloads]
    return {
        "experiment": "SRSD full",
        "setting": "EML-KAN stronger matched rerun",
        "params": mean(params),
        "n": len(payloads),
        "test_mse": mean(mses),
        "test_mse_std": stdev(mses) if len(mses) > 1 else 0.0,
        "ood_mse": math.nan,
        "r2": mean(r2s),
        "interpretation": "Capacity-matched rerun improves EML-KAN but RBF-KAN remains stronger.",
    }


def collect_rows() -> list[dict]:
    rows = []
    fairness = pd.read_csv(RESULTS / "fairness_audit_summary_20260509.csv")
    main = pd.read_csv(RESULTS / "paper_unified_table_20260509.csv")
    pareto = pd.read_csv(RESULTS / "pareto_budget_v2" / "pareto_summary.csv")

    old_srsd_table = pd.read_csv(RESULTS / "srsd_grouped" / "srsd_full_by_model.csv")
    old_srsd = old_srsd_table[old_srsd_table["model"] == "eml_kan"].iloc[0]
    old_payload_path = sorted((RESULTS / "srsd_grouped" / "srsd").glob("*eml_kan*.json"))[0]
    old_srsd_params = count_params_from_record(json.loads(old_payload_path.read_text(encoding="utf-8")))
    rows.append(
        {
            "experiment": "SRSD full",
            "setting": "EML-KAN original formal run",
            "params": old_srsd_params,
            "n": old_srsd["n"],
            "test_mse": old_srsd["test_mse_mean"],
            "test_mse_std": old_srsd["test_mse_std"],
            "ood_mse": math.nan,
            "r2": old_srsd["test_r2_mean"],
            "interpretation": "Original EML-KAN used fewer parameters than the RBF-KAN reference.",
        }
    )
    rows.append(srsd_strong_summary())
    kan_srsd = main[(main["benchmark"] == "SRSD full") & (main["metric"] == "test") & (main["method"] == "kan")].iloc[0]
    kan_params = fairness[(fairness["benchmark"] == "SRSD full") & (fairness["method"] == "kan")].iloc[0]["params_mean"]
    rows.append(
        {
            "experiment": "SRSD full",
            "setting": "RBF-KAN reference",
            "params": kan_params,
            "n": kan_srsd["n"],
            "test_mse": kan_srsd["mse_mean"],
            "test_mse_std": kan_srsd["mse_std"],
            "ood_mse": math.nan,
            "r2": kan_srsd["r2_mean"],
            "interpretation": "Accuracy boundary after rerun.",
        }
    )

    syn_small = main[(main["benchmark"] == "Synthetic final") & (main["metric"] == "ood") & (main["method"] == "eml_kan")].iloc[0]
    syn_small_params = fairness[(fairness["benchmark"] == "Synthetic final") & (fairness["method"] == "eml_kan")].iloc[0]["params_mean"]
    rows.append(
        {
            "experiment": "Synthetic OOD",
            "setting": "EML-KAN small formal run",
            "params": syn_small_params,
            "n": syn_small["n"],
            "test_mse": main[(main["benchmark"] == "Synthetic final") & (main["metric"] == "test") & (main["method"] == "eml_kan")].iloc[0]["mse_mean"],
            "test_mse_std": main[(main["benchmark"] == "Synthetic final") & (main["metric"] == "test") & (main["method"] == "eml_kan")].iloc[0]["mse_std"],
            "ood_mse": syn_small["mse_mean"],
            "r2": syn_small["r2_mean"],
            "interpretation": "Small-budget EML-KAN is not an OOD winner.",
        }
    )
    syn_large = pareto[(pareto["budget"] == "large") & (pareto["method"] == "eml_kan")].iloc[0]
    rows.append(
        {
            "experiment": "Synthetic OOD",
            "setting": "EML-KAN large matched rerun",
            "params": syn_large["params_mean"],
            "n": syn_large["n"],
            "test_mse": syn_large["test_mse_mean"],
            "test_mse_std": syn_large["test_mse_std"],
            "ood_mse": syn_large["ood_mse_mean"],
            "r2": math.nan,
            "interpretation": "Expansion improves OOD but does not close the MLP gap.",
        }
    )
    mlp_large = pareto[(pareto["budget"] == "large") & (pareto["method"] == "mlp")].iloc[0]
    rows.append(
        {
            "experiment": "Synthetic OOD",
            "setting": "MLP large reference",
            "params": mlp_large["params_mean"],
            "n": mlp_large["n"],
            "test_mse": mlp_large["test_mse_mean"],
            "test_mse_std": mlp_large["test_mse_std"],
            "ood_mse": mlp_large["ood_mse_mean"],
            "r2": math.nan,
            "interpretation": "OOD boundary after fair expansion.",
        }
    )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict], path: Path) -> None:
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Targeted rerun audit for EML-KAN weak spots. SRSD is rerun with a stronger parameter-matched EML-KAN configuration, and Synthetic OOD is checked against the existing large-budget Pareto rerun. The reruns improve EML-KAN but do not overturn the reported accuracy boundaries.}",
        r"\label{tab:eml-rerun-audit}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Experiment & Setting & Params & $n$ & Test MSE $\downarrow$ & OOD MSE $\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['experiment']} & {row['setting']} & {texnum(row['params'])} & "
            f"{int(row['n'])} & ${fmt(row['test_mse'])}\\pm{fmt(row['test_mse_std'])}$ & "
            f"{texnum(row['ood_mse'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    write_csv(rows, RESULTS / "eml_rerun_audit.csv")
    write_latex(rows, PAPER / "table_eml_rerun_audit.tex")
    print(f"rerun audit rows={len(rows)}")


if __name__ == "__main__":
    main()
