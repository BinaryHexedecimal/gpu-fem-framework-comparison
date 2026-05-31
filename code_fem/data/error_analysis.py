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


def create_error_table(dtype, framework):

    files = list(
        Path("./raw_data").glob(
            f"{framework}_{dtype}_numerical_res.json"
        )
    )

    rows = []

    # Load + flatten nested JSON
    for f in files:
        with open(f) as file:
            data = json.load(file)

        for entry in data:
            base_info = {
                "framework": entry["framework"],
                "version": entry["version"],
                "load": entry["load"],
                "fp": entry["fp"],
                "n_elems": entry["n_elems"],
            }

            # flatten runs
            for run_id, run_data in entry["runs"].items():
                row = {
                    **base_info,
                    "run": int(run_id),

                    "global_rel_error_interp":
                        run_data["global_rel_error_interp"],

                    "global_rel_error_same_mesh":
                        run_data["global_rel_error_same_mesh"],

                    "cg_iteration":
                        run_data["cg_iteration"],
                }
                rows.append(row)

    df = pd.DataFrame(rows)


    load_order = ["gravity", "traction", "compression"]
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

    # Aggregate mean/std
    stats = df.groupby(
        ["n_elems", "load", "version"],
        observed=True
    ).agg({
        "global_rel_error_interp": ["mean", "std"],
        "global_rel_error_same_mesh": ["mean", "std"],
    }).reset_index()

    stats.columns = [
        "_".join(col).strip("_")
        for col in stats.columns
    ]


    def format_cell(mean, std):

        if pd.isna(mean):
            return "--"

        if pd.isna(std):
            std = 0.0

        return (
            f"\\makecell{{"
            f"{mean:.2e} \\\\[-4pt] "
            f"({std:.1e})"
            f"}}"
        )


    stats["interp_fmt"] = stats.apply(
        lambda r: format_cell(
            r["global_rel_error_interp_mean"],
            r["global_rel_error_interp_std"]
        ),
        axis=1
    )

    stats["same_fmt"] = stats.apply(
        lambda r: format_cell(
            r["global_rel_error_same_mesh_mean"],
            r["global_rel_error_same_mesh_std"]
        ),
        axis=1
    )


    interp_df = stats[
        ["n_elems", "load", "version", "interp_fmt"]
    ].copy()

    interp_df["error_type"] = (
        r"\makecell{\scriptsize disc.\\ \scriptsize err.}"
    )

    interp_df = interp_df.rename(
        columns={"interp_fmt": "value"}
    )

    same_df = stats[
        ["n_elems", "load", "version", "same_fmt"]
    ].copy()

    same_df["error_type"] = (
        r"\makecell{\scriptsize impl.\\ \scriptsize err.}"
    )

    same_df = same_df.rename(
        columns={"same_fmt": "value"}
    )

    combined = pd.concat(
        [interp_df, same_df],
        ignore_index=True
    )


    # Pivot
    table = combined.pivot_table(
        index=["n_elems", "error_type"],
        columns=["load", "version"],
        values="value",
        aggfunc="first",
        observed=False,
    )


    table = table.sort_index(axis=0)
    table = table.sort_index(axis=1)


    table.index = pd.MultiIndex.from_tuples([
        (
            rf"\makecell{{\scriptsize {int(mesh):,} \\ \scriptsize elem.}}".replace(",", r"\,"),
            err
        ) for mesh, err in table.index

    ])

    table.index.names = [None, None]
    table.columns.names = [None, None]

    # Latex
    nr_versions = df["version"].nunique()

    latex_str = table.to_latex(

        multirow=True,
        multicolumn=True,
        na_rep="--",
        multicolumn_format='|c',
        escape=False,

        column_format=(
            'm{0.75cm}|m{0.5cm}|'
            + '|'.join([
                '>{\\centering\\arraybackslash}m{1.3cm}' * nr_versions
                for _ in range(table.shape[1] // nr_versions)
            ])
        )
    )

    latex_str = latex_str.replace(
        r"\multirow[t]",
        r"\multirow"
    )


    # Caption
    runtimes = df["run"].nunique()

    framework_name = FRAMEWORK_DISPLAY.get(
        framework,
        framework
    )

    dtype_name = DTYPE_DISPLAY.get(
        dtype,
        dtype
    )

    caption = (
        f"Global relative displacement errors for different "
        f"load cases and mesh resolutions using {framework_name} "
        f"with {dtype_name} precision. "
        f"Two types of errors are reported, corresponding to the comparison "
        f"methods described in Section~\\ref{{sec:fenics_as_benchmark}}. "
        f"The implementation error (impl.~err.) is obtained by solving "
        f"the problem on identical meshes using both the implemented solver "
        f"and FEniCS. "
        f"The discretization error (disc.~err.) is computed by comparing "
        f"the solution obtained with the implemented solver on a given mesh "
        f"to a reference (ground-truth) solution. "
        f"The reference solution is computed in FEniCS on a highly refined mesh "
        f"($360 \\times 45 \\times 45 \\times 6$ elements) and subsequently "
        f"interpolated onto the mesh under consideration. "
        f"In all cases, errors are measured using the Frobenius norm of the "
        f"difference between displacement fields, normalized by the reference "
        f"solution norm. "
        f"The reported values are averaged over {runtimes} independent runs, "
        f"with standard deviations shown in parentheses."
    )

    label = f"tab:{framework}_numerical_error_{dtype}"

    latex_str = latex_str.replace(
        r"\end{tabular}",
        r"\end{tabular}"
        + r"\caption{" + caption + r"}"
        + r"\label{" + label + r"}"
        + r"\end{table}"
    )

    latex_str = latex_str.replace(
        r"\begin{tabular}",
        r"\begin{table}[H]\center\begin{tabular}"
    )

    return latex_str






framework_lst = [
    "pytorch",
    "jax",
    "warp",
]

dtype_list = [
    "fp64",
    "fp32",
]

save_dir = Path("./analysis_res")
save_dir.mkdir(parents=True, exist_ok=True)

for dtype in dtype_list:
    error_table_string = ""
    for framework in framework_lst:
        error_table_string += create_error_table(
            dtype,
            framework
        )
        error_table_string += "\n\n"

    save_path = save_dir / f"error_table_{dtype}.tex"
    with open(save_path, "w") as f:
        f.write(error_table_string)
    print(f"Saved: {save_path}")