
import json
import pandas as pd
from pathlib import Path


MESH_DISPLAY = {
    "coarse": r"24\ 576",
    "m1": r"63\ 888",
    "m2": r"162\ 000",
    "fine": r"444\ 528",
}


def sci(x, digits=2):
    return f"{x:.{digits}e}"


def create_condition_table():

    source_path = Path("./raw_data")

    with open(source_path / "kappa_results.json") as f:
        data = json.load(f)

    mesh_order = [
        "coarse",
        "m1",
        "m2",
        "fine"
    ]

    load_order = [
        "gravity",
        "traction",
        "compression"
    ]

    quantity_order = [
        "kappa",
        "kappa_eff",
        "lambda_max",
        "lambda_min",
        "lambda_min_eff",
    ]

    quantity_display = {
        "kappa":
            r"$\kappa(K_{ff})$",

        "kappa_eff":
            r"$\kappa_{\mathrm{eff}}(K_{ff})$",

        "lambda_max":
            r"$\lambda_{\max}$",

        "lambda_min":
            r"$\lambda_{\min}$",

        "lambda_min_eff":
            r"$\lambda_{\min}^{+}$",
    }


    rows = []
    for mesh in mesh_order:
        for quantity in quantity_order:
            row = {
                "mesh": mesh,
                "quantity": quantity_display[quantity]
            }

            for load in load_order:
                value = data[mesh][load].get(quantity)
                if value is None:
                    row[load] = "--"
                else:
                    row[load] = sci(value)

            rows.append(row)


    df = pd.DataFrame(rows)

    df["mesh"] = pd.Categorical(
        df["mesh"],
        categories=mesh_order,
        ordered=True
    )

    table = df.pivot_table(
        index=["mesh", "quantity"],
        values=load_order,
        aggfunc="first",
        observed=False
    )

    table = table[load_order]

    table.index = pd.MultiIndex.from_tuples([
        (
            rf"\makecell{{{MESH_DISPLAY[mesh]} \\ elem}}",
            quantity
        )
        for mesh, quantity in table.index
    ])

    table.index.names = [None, None]
    table.columns.name = None


    # latex
    column_format = (
        'm{1.5cm}|m{2.3cm}|'
        + 'm{2.2cm}' * len(load_order)
    )
    latex_str = table.to_latex(
        multirow=True,
        escape=False,
        na_rep="--",
        column_format=column_format,
    )

    latex_str = latex_str.replace(
        r"\multirow[t]",
        r"\multirow"
    )

    caption = (
        "Condition numbers and extreme eigenvalues of the reduced "
        "global stiffness matrix $K_{ff}$ for different mesh "
        "resolutions and loading conditions. "
        r"$\lambda_{\min}^{+}$ denotes the smallest nonzero eigenvalue, "
        "used to define the effective condition number "
        r"$\kappa_{\mathrm{eff}}(K_{ff})$ for the compression model, "
        "where the stiffness matrix contains a rigid-body mode."
        "Note that the header naming is somewhat misleading. "
        "Strictly speaking, the applied load itself does not affect "
        "the conditioning of the stiffness matrix. "
        "Rather, the conditioning is primarily influenced by the boundary conditions "
        "and the set of constrained degrees of freedom associated with each loading model."
    )

    label = "tab:conditioning"

    latex_str = latex_str.replace(
        r"\begin{tabular}",
        r"\begin{table}[H]\centering\small\begin{tabular}"
    )

    latex_str = latex_str.replace(
        r"\end{tabular}",
        r"\end{tabular}"
        + r"\caption{" + caption + r"}"
        + r"\label{" + label + r"}"
        + r"\end{table}"
    )

    return latex_str







latex_table = create_condition_table()

save_path = Path("./analysis_res") / "conditioning_table.tex"

with open(save_path, "w") as f:
    f.write(latex_table)
print(f"Saved: {save_path}")