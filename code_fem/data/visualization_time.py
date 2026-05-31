import pandas as pd
import json
from pathlib import Path

import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from data.TABLE_PARAMETER import DTYPE_DISPLAY


def create_rendering_overhead_table(dtype):
    rows = []

    # CPU-based
    framework_files = {
        "pytorch_cpu": f"pytorch_{dtype}_realtime_cpu_res.json",
        "jax_cpu": f"jax_{dtype}_realtime_cpu_res.json",
        "warp_cpu": f"warp_{dtype}_realtime_cpu_res.json",
    }
    for label, filename in framework_files.items():
        path = Path("./raw_data") / filename
        if not path.exists():
            continue
        with open(path) as file:
            data = json.load(file)
        for entry in data:
            for run_id, run_data in entry["interactive_runs"].items():
                rows.append({
                    "method": label,
                    "n_elems": entry["n_elems"],
                    "run_id": int(run_id),
                    "version": entry["version"],
                    "vis_time":
                        run_data["first_frame_latency"]
                        - run_data["solve_stage_time"]
                })


    # Warp, GPU / zero-copy 
    transfer_files = {
        "warp_gpu": f"warp_{dtype}_realtime_gpu_res.json",
        "warp_zero": f"warp_{dtype}_realtime_zero_res.json",
    }

    for label, filename in transfer_files.items():
        path = Path("./raw_data") / filename
        if not path.exists():
            continue
        with open(path) as file:
            data = json.load(file)
        for entry in data:
            for run_id, run_data in entry["interactive_runs"].items():
                rows.append({
                    "method": label,
                    "n_elems": entry["n_elems"],
                    "run_id": int(run_id),
                    "version": entry["version"],
                    "vis_time":
                        run_data["first_frame_latency"]
                        - run_data["solve_stage_time"]
                })

    df = pd.DataFrame(rows)
    if dtype == "fp32":
        df = df[df["version"] != "K"]

    df = df[df["run_id"] >= 2].copy()

    method_order = [
        "pytorch_cpu",
        "jax_cpu",
        "warp_cpu",
        "warp_gpu",
        "warp_zero",
    ]

    df["method"] = pd.Categorical(
        df["method"],
        categories=method_order,
        ordered=True
    )

    stats = df.groupby(
            ["n_elems", "method"],
            observed=True
        ).agg({"vis_time": ["mean", "std"]}
    ).reset_index()

    stats.columns = [
        "_".join(col).strip("_")
        for col in stats.columns
    ]


    def combine(mean, std):
        return (
            f"\\makecell{{"
            f"{mean:.3f} \\\\[-4pt] "
            f"({std:.3f})"
            f"}}"
        )

    stats["value"] = stats.apply(
        lambda r: combine(
            r["vis_time_mean"],
            r["vis_time_std"]
        ),
        axis=1
    )

    table = stats.pivot_table(
        index=["n_elems"],
        columns=["method"],
        values="value",
        aggfunc="first",
        observed=False,
    )

    cols = []
    for method in method_order:
        if method in table.columns:
            cols.append(method)
    table = table[cols]


    table.index = [
        f"{int(mesh):,}".replace(",", r"\,")
        for mesh in table.index
    ]

    display_map = {
        "pytorch_cpu":
            r"PyTorch\\CPU",
        "jax_cpu":
            r"JAX\\CPU",
        "warp_cpu":
            r"Warp\\CPU",
        "warp_gpu":
            r"Warp\\GPU",
        "warp_zero":
            r"Warp\\Zero-copy",
    }


    table.columns = [
        rf"\makecell{{{display_map[col]}}}"
        for col in table.columns
    ]

    table.index.name = None
    table.columns.name = None

    column_format = (
        '>{\\centering\\arraybackslash}m{2.0cm}|'
        + '>{\\centering\\arraybackslash}m{1.9cm}'
        * len(table.columns)
    )


    latex_str = table.to_latex(
        multicolumn=True,
        multicolumn_format='c',
        na_rep="--",
        escape=False,
        column_format=column_format,
    )

    latex_str = latex_str.replace(
        r"\begin{tabular}",
        r"\begin{table}[H]\centering\begin{tabular}"
    )
    latex_str = latex_str.replace(
        r" & \makecell{PyTorch",
        r"elements & \makecell{PyTorch"
    )


    if dtype == "fp64":
        caption = (
            f"Rendering overhead (in seconds) for different mesh sizes "
            f"using {DTYPE_DISPLAY.get(dtype, dtype)} precision. "
            "The table compares CPU-based rendering transfer in "
            "PyTorch, JAX, and Warp, together with Warp-specific "
            "GPU-copy and zero-copy transfer strategies. "
            "Rendering overhead is defined as the elapsed time "
            "between completion of the solver computation and "
            "rendering of the deformed geometry on screen. "
            "Results are averaged over steady-state runs "
            "(run 1 discarded as warm-up), across all considered "
            "load types and solver variants. "
            "For PyTorch and JAX, each reported value is averaged "
            "over 63 measurements per mesh size "
            "(7 runs × 3 load types × 3 solver variants). "
            "For Warp, each reported value is averaged over "
            "42 measurements per mesh size "
            "(7 runs × 3 load types × 2 solver variants), "
            "since the K variant is not implemented. "
            "Standard deviations are shown in parentheses."
        )
    else:
        caption = (
            f"Rendering overhead (in seconds) for different mesh sizes "
            f"using {DTYPE_DISPLAY.get(dtype, dtype)} precision. "
            "The table compares CPU-based rendering transfer in "
            "PyTorch, JAX, and Warp, together with Warp-specific "
            "GPU-copy and zero-copy transfer strategies. "
            "Rendering overhead is defined as the elapsed time "
            "between completion of the solver computation and "
            "rendering of the deformed geometry on screen. "
            "Results are averaged over steady-state runs "
            "(run 1 discarded as warm-up), across all considered "
            "load types and solver variants. "
            "Each reported value is averaged "
            "over 42 measurements per mesh size "
            "(7 runs × 3 load types × 2 solver variants). "
            "Standard deviations are shown in parentheses."
        )

    label = f"tab:rendering_overhead_combined_{dtype}"
    latex_str = latex_str.replace(
        r"\end{tabular}",
        r"\end{tabular}"
        + r"\caption{" + caption + r"}"
        + r"\label{" + label + r"}"
        + r"\end{table}"
    )

    return latex_str



for dtype in ["fp32", "fp64"]:
    latex_table = create_rendering_overhead_table(dtype)
    save_path = (
        Path("./analysis_res")
        / f"rendering_overhead_combined_{dtype}.tex"
    )
    with open(save_path, "w") as f:
        f.write(latex_table)
    print(f"Saved: {save_path}")