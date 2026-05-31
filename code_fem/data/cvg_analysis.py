import pandas as pd
import json
from pathlib import Path

import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from data.TABLE_PARAMETER import FRAMEWORK_DISPLAY, DTYPE_DISPLAY




def create_cg_table(fp, framework_lst):

    rows = []
    # Load + flatten nested JSON
    for framework in framework_lst:
        files = list(
            Path("./raw_data").glob(
                f"{framework}_{fp}_numerical_res.json"
            )
        )

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
                        "cg_iteration":
                            run_data["cg_iteration"],
                        "global_rel_error_interp":
                            run_data["global_rel_error_interp"],
                        "global_rel_error_same_mesh":
                            run_data["global_rel_error_same_mesh"],
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)

    # remove K formulation under FP32
    if fp == "fp32":
        df = df[df["version"] != "K"]


    # Ordering
    load_order = ["gravity", "traction", "compression"]
    
    if fp == "fp32":
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


    # Aggregate statistics
    stats = df.groupby(
        ["n_elems", "framework", "load", "version"],
        observed=True
    ).agg({
        "cg_iteration": ["mean", "std"]
    }).reset_index()

    stats.columns = [
        "_".join(col).strip("_")
        for col in stats.columns
    ]


    # Formatting
    def format_cell(mean, std):
        if pd.isna(mean):
            return "--"
        if pd.isna(std):
            std = 0.0
        return (
            f"\\makecell{{"
            f"{mean:.0f} \\\\[-4pt] "
            f"({std:.1f})"
            f"}}"
        )

    stats["value"] = stats.apply(
        lambda r: format_cell(
            r["cg_iteration_mean"],
            r["cg_iteration_std"]
        ),
        axis=1
    )

    # Pivot table
    table = stats.pivot_table(
        index=["n_elems", "framework"],
        columns=["load", "version"],
        values="value",
        aggfunc="first",
        observed=True
    )

    # Ensure all frameworks exist
    framework_order = ["pytorch", "warp", "jax"]
    n_elems_order = sorted(df["n_elems"].unique())
    table = table.reindex(
        index=pd.MultiIndex.from_product(
            [n_elems_order, framework_order]
        )
    )

    # Rename framework display
    table.index = pd.MultiIndex.from_tuples([
        (
            f"{int(mesh):,}",
            FRAMEWORK_DISPLAY.get(framework, framework)
        )
        for mesh, framework in table.index
    ])


    table.index.names = [None, None]
    table.columns.names = [None, None]


    # LaTeX
    nr_versions = df["version"].nunique()
    latex_str = table.to_latex(
        multirow=True,
        multicolumn=True,
        na_rep="--",
        multicolumn_format='|c',
        escape=False,
        column_format=(
            '>{\\centering\\arraybackslash}m{1.0cm}|m{1.1cm}|'
            + '|'.join(
                ['>{\\centering\\arraybackslash}m{1.0cm}' * nr_versions
                 for _ in range(table.shape[1] // nr_versions)]
            )
        )
    )

    latex_str = latex_str.replace(
        r"\multirow[t]",
        r"\multirow"
    )

    latex_str = latex_str.replace(
        r"PyTorch \\",
        r"PyTorch \\ \cline{2-10}"
    )

    latex_str = latex_str.replace(
        r"Warp \\",
        r"Warp \\ \midrule"
    )

    runtimes = df["run"].nunique()

    dtype_name = DTYPE_DISPLAY[fp]

    caption = (
        f"Number of iterations in the conjugate gradient (CG) solver "
        f"using {dtype_name} precision. "
        f"The reported values are averaged over {runtimes} independent runs, "
        f"with standard deviations shown in parentheses. "
        f"In the Warp framework, the K solver is not implemented. "
        f"Therefore, the corresponding entries are not available. "
        f"In the JAX framework, a built-in CG solver is used, "
        f"which does not directly provide the iteration count. "
        f"Instead, the number of iterations is estimated by "
        f"prescribing iteration limits and verifying convergence, "
        f"resulting in an interval of possible values."
    )

    label = f"tab:cg_iteration_{fp}"

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


    save_path = Path("./analysis_res")
    save_path.mkdir(parents=True, exist_ok=True)

    out_file = save_path / f"cg_table_{fp}.tex"

    with open(out_file, "w") as f:
        f.write(latex_str)

    print(f"Saved: {out_file}")




fp_list = ["fp32", "fp64"]

framework_lst = [
    "pytorch",
    "jax",
    "warp",
]

for fp in fp_list:
    create_cg_table(fp, framework_lst)