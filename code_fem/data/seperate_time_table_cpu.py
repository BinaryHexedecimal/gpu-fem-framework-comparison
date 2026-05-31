
import pandas as pd
import json
from pathlib import Path

import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from data.TABLE_PARAMETER import (
    FRAMEWORK_DISPLAY,
    DTYPE_DISPLAY,
)


def create_metric_table(metric_key, framework, dtype):

    files = list(
        Path("./raw_data").glob(
            f"{framework}_{dtype}_realtime_cpu_res.json"
        )
    )

    rows_pre = []
    rows_run = []

    for f in files:
        with open(f) as file:
            data = json.load(file)
        for entry in data:
            base = {
                "framework": entry["framework"],
                "load": entry["load"],
                "version": entry["version"],
                "n_elems": entry["n_elems"],
            }


            for pre_id, pre_time in entry["precompute_runs"].items():
                rows_pre.append({
                    **base,
                    "run_id": int(pre_id),
                    "precompute_time": pre_time,
                })

            for run_id, run_data in entry["interactive_runs"].items():
                rows_run.append({
                    **base,
                    "run_id": int(run_id),
                    "solve_stage_time":
                        run_data["solve_stage_time"],
                    "first_frame_latency":
                        run_data["first_frame_latency"],
                    "iterations":
                        run_data["iterations"],
                    "force":
                        run_data["force"],
                })

    df_pre = pd.DataFrame(rows_pre)
    df_run = pd.DataFrame(rows_run)

    if dtype == "fp32":
        df_pre = df_pre[df_pre["version"] != "K"]
        df_run = df_run[df_run["version"] != "K"]


    # discard run 1
    if not df_pre.empty:
        df_pre = df_pre[df_pre["run_id"] >= 2].copy()

    if not df_run.empty:
        df_run = df_run[df_run["run_id"] >= 2].copy()


    n_pre_runs = (
        df_pre["run_id"].nunique()
        if not df_pre.empty else 0
    )

    n_run_runs = (
        df_run["run_id"].nunique()
        if not df_run.empty else 0
    )


    load_order = [
        "gravity",
        "traction",
        "compression"
    ]

    if dtype == "fp32":
        version_order = ["B", "G"]
    else:
        version_order = [ "K", "B", "G"]


    for df in [df_pre, df_run]:
        if not df.empty:
            df["load"] = pd.Categorical(
                df["load"],
                categories=load_order,
                ordered=True
            )
            df["version"] = pd.Categorical(
                df["version"],
                categories=version_order,
                ordered=True
            )

    def combine(mean, std):
        if pd.isna(mean):
            return "--"
        if pd.isna(std):
            std = 0.0
        return (
            f"\\makecell{{"
            f"{mean:.3f}"
            f"\\\\[-4pt]"
            f"({std:.3f})"
            f"}}"
        )


    if metric_key == "precompute":
        metric_col = "precompute_time"
        stats = df_pre.groupby(
            ["load", "version", "n_elems"],
            observed=True
        ).agg({
            metric_col: ["mean", "std"]
        }).reset_index()


    elif metric_key == "solver":
        metric_col = "solve_stage_time"
        stats = df_run.groupby(
            ["load", "version", "n_elems"],
            observed=True
        ).agg({
            metric_col: ["mean", "std"]
        }).reset_index()

    elif metric_key == "latency":
        metric_col = "first_frame_latency"
        stats = df_run.groupby(
            ["load", "version", "n_elems"],
            observed=True
        ).agg({
            metric_col: ["mean", "std"]
        }).reset_index()

    else:
        raise ValueError(
            f"Unknown metric: {metric_key}"
        )


    stats.columns = [
        "_".join(col).strip("_")
        for col in stats.columns
    ]

    stats["value"] = stats.apply(
        lambda r: combine(
            r[f"{metric_col}_mean"],
            r[f"{metric_col}_std"]
        ),
        axis=1
    )

    table = stats.pivot_table(
        index=["n_elems"],
        columns=["load", "version"],
        values="value",
        aggfunc="first",
        observed=False,
    )

    ordered_cols = []
    for load in load_order:
        versions_present = sorted(
            set(
                v
                for (l, v) in table.columns
                if l == load
            ),
            key=lambda x:
                version_order.index(x)
        )

        for version in versions_present:
            ordered_cols.append(
                (load, version)
            )

    table = table[ordered_cols]

    table.index = [
        f"{int(mesh):,}".replace(",", r"\,")
        for mesh in table.index
    ]

    table.index.name = None
    table.columns.names = [None, None]

    nr_versions = len(set(c[1] for c in table.columns))

    column_format = (
        '>{\\centering\\arraybackslash}m{1.2cm}|'
        + '|'.join(
            [
                '>{\\centering\\arraybackslash}m{1.25cm}'
                * nr_versions
                for _ in load_order
            ]
        )
    )


    latex_str = table.to_latex(
        multirow=True,
        multicolumn=True,
        multicolumn_format='|c',
        na_rep="--",
        escape=False,
        column_format=column_format,
    )

    framework_name = FRAMEWORK_DISPLAY[framework]
    dtype_name = DTYPE_DISPLAY[dtype]

    if metric_key == "precompute":
        caption = (
            f"Average precomputation times (in seconds) "
            f"for different load types, precomputation variants "
            f"and mesh sizes using {framework_name} with "
            f"{dtype_name} precision. "
            f"Results are averaged over {n_pre_runs} steady-state runs "
            f"(run 1 discarded as cold start), with standard "
            f"deviations shown in parentheses."
        )

    elif metric_key == "solver":
        caption = (
            f"Average solver-stage runtimes (in seconds) "
            f"for different load types, computation variants "
            f"and mesh sizes using {framework_name} with "
            f"{dtype_name} precision. "
            f"Results are averaged over {n_run_runs} steady-state runs "
            f"(run 1 discarded as cold start), with standard "
            f"deviations shown in parentheses."
        )

    elif metric_key == "latency":
        caption = (
            f"Average first-frame latency times (in seconds) "
            f"for different load types, computation variants "
            f"and mesh sizes using {framework_name} with "
            f"{dtype_name} precision and CPU-based rendering "
            f"data transfer. "
            f"Results are averaged over {n_run_runs} steady-state runs "
            f"(run 1 discarded as cold start), with standard "
            f"deviations shown in parentheses."
        )


    label = (f"tab:{framework}_{metric_key}_{dtype}")

    latex_str = latex_str.replace(
        r"\begin{tabular}",
        r"\begin{table}[H]\centering\begin{tabular}"
    )

    latex_str = latex_str.replace(
        r"\end{tabular}",
        r"\end{tabular}"
        + r"\caption{" + caption + r"}"
        + r"\label{" + label + r"}"
        + r"\end{table}"
    )

    latex_str = latex_str.replace(
        r" & \multicolumn{3}{|c}{gravity}",
        r"elements & \multicolumn{3}{|c}{gravity}"
    )
    latex_str = latex_str.replace(
        r" & \multicolumn{2}{|c}{gravity}",
        r"elements & \multicolumn{2}{|c}{gravity}"
    )

    latex_str = latex_str.replace(
        r"\multirow[t]",
        r"\multirow"
    )

    return latex_str



framework_list = [
    "pytorch",
    "jax",
    "warp",
]

fp_list = [
    "fp32",
    "fp64",
]

metric_configs = [
    ("precompute", "Precomputation"),
    ("solver", "Solver"),
    ("latency", "Latency"),
]

save_dir = Path("./analysis_res")
save_dir.mkdir(parents=True, exist_ok=True)

for dtype in fp_list:
    for metric_key, metric_title in metric_configs:
        latex_all = ""
        for framework in framework_list:
            latex_all += create_metric_table(
                metric_key,
                framework,
                dtype,
            )
            latex_all += "\n\n"
        save_path = (save_dir/ f"{metric_key}_tables_{dtype}.tex")
        with open(save_path, "w") as f:
            f.write(latex_all)
        print(f"Saved: {save_path}")