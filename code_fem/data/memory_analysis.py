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


def create_memory_table(dtype, framework):
    rows = []
    files = list(
        Path("./raw_data").glob(
            f"{framework}_{dtype}_realtime_cpu_res.json"
        )
    )

    for f in files:
        with open(f) as file:
            data = json.load(file)
            for entry in data:
                rows.append({
                    "framework": framework,
                    "load": entry["load"],
                    "version": entry["version"],
                    "n_elems": entry["n_elems"],
                    "peak_mem_MB": entry["peak_mem"],
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
        version_order = [ "K", "B", "G"]

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


    # df["peak_mem_MB"] = df[
    #     "peak_mem_MB"
    # ].apply(
    #     lambda x: f"{x:.2f}"
    # )
    df["peak_mem_MB"] = df["peak_mem_MB"].astype(int).astype(str)


    # Pivot
    table = df.pivot_table(
        index=["n_elems"],
        columns=["load", "version"],
        values="peak_mem_MB",
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
        f"{int(mesh):,}".replace(",", r"\,")
        for mesh in table.index
    ]


    table.columns = pd.MultiIndex.from_tuples([
        (load,version)
        for load, version in table.columns
    ])

    table.index.names = [None]
    table.columns.names = [None, None]

    nr_versions = len(
        table.columns.levels[1]
    )

    column_format = (
        '>{\\centering\\arraybackslash}m{1.35cm}|'
        + '|'.join(
            [
                '>{\\centering\\arraybackslash}m{1.0cm}'
                * nr_versions
                for _ in load_order
            ]
        )
    )

    latex_str = table.to_latex(
        multirow=True,
        multicolumn=True,
        na_rep="--",
        escape=False,
        multicolumn_format='|c',
        column_format=column_format,
    )

    latex_str = latex_str.replace(
        r'\multirow[t]',
        r'\multirow'
    )


    framework_name = FRAMEWORK_DISPLAY.get(
        framework,
        framework
    )

    dtype_name = DTYPE_DISPLAY.get(
        dtype,
        dtype
    )

    caption = (
        "Peak memory usage (MB) for different "
        "load types, computation variants and mesh sizes using "
        f"\\textbf{{{framework_name}}} with "
        f"{dtype_name} precision. "
        "Each value represents the maximum "
        "memory usage observed over the "
        "entire pipeline, including "
        "repeated precomputation and solver "
        "executions, and rendering via CPU."
    )

    label = f"tab:{framework}_memory_{dtype}"

    latex_str = latex_str.replace(
        r"\begin{tabular}",
        r"\begin{table}[H]\centering\begin{tabular}"
    )
    latex_str = latex_str.replace(
        r" & \multicolumn{3}{|c}{gravity}",
        r"elements & \multicolumn{3}{|c}{gravity}"
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
framework_lst = [ "pytorch", "jax", "warp"]
for dtype in dtype_list:
    memory_table_string = ""
    for framework in framework_lst:     
        memory_table_string += create_memory_table(
            dtype,
            framework
        )

    save_path = (
        Path("./analysis_res")
        / f"memory_tables_{dtype}.tex"
    )
    with open(save_path, "w") as f:
        f.write(memory_table_string)
    print(f"Saved: {save_path}")





