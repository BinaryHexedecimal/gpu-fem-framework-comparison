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


def create_solver_overhead_table(dtype, framework):
    files = list(
        Path("./raw_data").glob(
            f"{framework}_{dtype}_realtime_cpu_res.json"
        )
    )
    rows = []

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

                for run_id, run_data in entry["interactive_runs"].items():
                    row = base.copy()
                    row["run_id"] = int(run_id)
                    row.update(run_data)
                    rows.append(row)

    df = pd.DataFrame(rows)
    if dtype == "fp32":
        df = df[df["version"] != "K"]


    load_order = ["gravity", "traction", "compression"]
    if dtype == "fp32":
        version_order = ["B", "G"]
    else:
        version_order = ["K", "B", "G"]

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


    # Split runs
    df_run1 = df[df["run_id"] == 1]
    df_steady = df[
        df["run_id"]>=2
    ]
    n_steady = df_steady["run_id"].nunique()


    # Steady-state mean
    steady_mean = df_steady.groupby(
        ["load", "version", "n_elems"],
        observed=True
    )["solve_stage_time"].mean().reset_index()

    merged = pd.merge(
        df_run1,
        steady_mean,
        on=["load", "version", "n_elems"],
        suffixes=("_run1", "_steady")
    )

    merged["overhead"] = (
        merged["solve_stage_time_run1"]
        - merged["solve_stage_time_steady"]
    )

    merged["overhead_str"] = merged[
        "overhead"
    ].apply(
        lambda x: f"{x:.3f}"
    )

    table = merged.pivot_table(
        index=["n_elems"],
        columns=["load", "version"],
        values="overhead_str",
        aggfunc="first",
        observed=False,
    )

    new_cols = []
    for load in load_order:
        for version in version_order:
            if (load, version) in table.columns:
                new_cols.append((load, version))

    table = table[new_cols]


    table.index = [
        f"{int(mesh):,}"
        for mesh in table.index
    ]

    table.columns = pd.MultiIndex.from_tuples([
        ( load, version ) for load, version in table.columns
    ])

    table.index.names = [None]
    table.columns.names = [None, None]

    nr_versions = len(table.columns.levels[1])

    column_format = (
        '>{\\centering\\arraybackslash}m{1.5cm}|'
        + '|'.join(
            [
                '>{\\centering\\arraybackslash}m{1.2cm}'
                * nr_versions
                for _ in load_order
            ]
        )
    )

    # -----------------------------------
    # LATEX
    # -----------------------------------
    latex_str = table.to_latex(
        multirow=True,
        multicolumn=True,
        multicolumn_format='|c',
        na_rep="--",
        escape=False,
        column_format=column_format,
    )

    latex_str = latex_str.replace(
        r"\multirow[t]",
        r"\multirow"
    )

    # Caption
    framework_name = FRAMEWORK_DISPLAY.get(
        framework,
        framework
    )

    dtype_name = DTYPE_DISPLAY.get(
        dtype,
        dtype
    )
    
    caption = (
        "Estimated first-run overhead "
        "(in seconds) for different load "
        f"types and mesh sizes using "
        f"{framework_name} under {dtype_name} precision. "
        "The overhead is computed as the "
        "difference between the first "
        "execution time and the average "
        "steady-state runtime over the "
        f"subsequent {n_steady} runs. This metric "
        "captures framework-specific "
        "initialization costs, including "
        "compilation (where applicable), "
        "kernel setup, memory allocation, "
        "and cache warm-up effects."
    )

    label = f"tab:{framework}_overhead_{dtype}"

    latex_str = latex_str.replace(
        r"\begin{tabular}",
        r"\begin{table}[H]\centering\begin{tabular}"
    )

    latex_str = latex_str.replace(
        r" & \multicolumn{2}{|c}{gravity}",
        r"elements & \multicolumn{2}{|c}{gravity}"
    )

    latex_str = latex_str.replace(
        r"\end{tabular}",
        r"\end{tabular}"
        + r"\caption{" + caption + r"}"
        + r"\label{" + label + r"}"
        + r"\end{table}"
    )

    return latex_str







def create_precompute_overhead_table(dtype, framework):
    files = list(
        Path("./raw_data").glob(
            f"{framework}_{dtype}_realtime_cpu_res.json"
        )
    )
    rows = []

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
            for run_id, pre_time in entry["precompute_runs"].items():
                rows.append({
                    **base,
                    "run_id": int(run_id),
                    "precompute_time": pre_time,
                })

    df = pd.DataFrame(rows)
    if dtype == "fp32":
        df = df[df["version"] != "K"]

    load_order = [
        "gravity",
        "traction",
        "compression"
    ]

    if dtype == "fp32":
        version_order = ["B", "G"]
    else:
        version_order = ["K", "B", "G"]


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

    df_run1 = df[
        df["run_id"] == 1
    ]

    df_steady = df[
        df["run_id"] >= 2
    ]


    steady_mean = df_steady.groupby(
        ["load", "version", "n_elems"],
        observed=True
    )["precompute_time"].mean().reset_index()

    merged = pd.merge(
        df_run1,
        steady_mean,
        on=["load", "version", "n_elems"],
        suffixes=("_run1", "_steady")
    )

    merged["overhead"] = (
        merged["precompute_time_run1"]
        - merged["precompute_time_steady"]
    )

    merged["overhead_str"] = merged["overhead"].apply(
        lambda x: f"{x:.3f}"
    )


    table = merged.pivot_table(
        index=["n_elems"],
        columns=["load", "version"],
        values="overhead_str",
        aggfunc="first",
        observed=False,
    )

    new_cols = []
    for load in load_order:
        for version in version_order:
            if (load, version) in table.columns:
                new_cols.append(
                    (load, version)
                )
    table = table[new_cols]

    table.index = [
        f"{int(mesh):,}"
        for mesh in table.index
    ]

    table.columns = pd.MultiIndex.from_tuples([
        (load, version)
        for load, version in table.columns
    ])

    table.index.names = [None]
    table.columns.names = [None, None]

    nr_versions = len(table.columns.levels[1])

    column_format = (
        '>{\\centering\\arraybackslash}m{1.5cm}|'
        + '|'.join(
            [
                '>{\\centering\\arraybackslash}m{1.2cm}'
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

    latex_str = latex_str.replace(
        r"\multirow[t]",
        r"\multirow"
    )

    framework_name = FRAMEWORK_DISPLAY.get(
        framework,
        framework
    )

    dtype_name = DTYPE_DISPLAY.get(
        dtype,
        dtype
    )

    n_steady = df_steady["run_id"].nunique()

    caption = (
        "Estimated first-run precomputation overhead "
        "(in seconds) for different load types and mesh "
        f"sizes using {framework_name} under "
        f"{dtype_name} precision. "
        "The overhead is computed as the difference "
        "between the first precomputation time and "
        f"the average steady-state precomputation "
        f"time over the subsequent {n_steady} runs. "
        "This metric captures framework initialization "
        "costs such as compilation, kernel generation, "
        "memory allocation, caching, and warm-up effects."
    )

    label = (
        f"tab:{framework}_"
        f"precompute_overhead_{dtype}"
    )

    latex_str = latex_str.replace(
        r"\begin{tabular}",
        r"\begin{table}[H]\centering\begin{tabular}"
    )

    latex_str = latex_str.replace(
        r" & \multicolumn{2}{|c}{gravity}",
        r"elements & \multicolumn{2}{|c}{gravity}"
    )

    latex_str = latex_str.replace(
        r"\end{tabular}",
        r"\end{tabular}"
        + r"\caption{" + caption + r"}"
        + r"\label{" + label + r"}"
        + r"\end{table}"
    )

    return latex_str





dtype_list = ["fp32", "fp64"]
framework_lst = ["pytorch","jax","warp"]

for dtype in dtype_list:
    # ---------------solver-------------------- #
    solver_overhead_table_string = ""
    for framework in framework_lst:
        solver_overhead_table_string += create_solver_overhead_table(
                dtype,
                framework,
            )
    save_path = (
        Path("./analysis_res")
        / f"solver_overhead_table_{dtype}.tex"
    )
    with open(save_path, "w") as f:
        f.write(solver_overhead_table_string)
    print(f"Saved: {save_path}")

    # ---------------precomputation-------------------- #
    precompute_overhead_table_string = ""
    for framework in framework_lst:
        precompute_overhead_table_string += create_precompute_overhead_table(
                dtype,
                framework,
            )
    save_path = (
        Path("./analysis_res")
        / f"precompute_overhead_table_{dtype}.tex"
    )
    with open(save_path, "w") as f:
        f.write(precompute_overhead_table_string)
    print(f"Saved: {save_path}")