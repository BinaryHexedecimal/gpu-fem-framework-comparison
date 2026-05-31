import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
#import numpy as np


def plot_precision_ratio(framework_lst):
    # Load fp32 + fp64
    rows = []
    for dtype in ["fp32", "fp64"]:
        for framework in framework_lst:
            files = list(
                Path("./raw_data").glob(
                    f"{framework}_{dtype}_realtime_cpu_res.json"
                )
            )
            for f in files:
                with open(f) as file:
                    data = json.load(file)
                    for entry in data:
                        base = {
                            "framework": framework,
                            "dtype": dtype,
                            "load": entry["load"],
                            "version": entry["version"],
                            "n_elems": entry["n_elems"],
                        }

                        for run_id, run_data in entry["interactive_runs"].items():
                            row = base.copy()
                            row["run_id"] = int(run_id)
                            row["solver_time"] = run_data["solve_stage_time"]
                            rows.append(row)

    df = pd.DataFrame(rows)

    # discard warm-up
    df = df[df["run_id"].between(2, 9)].copy()

    # mean runtime
    stats = df.groupby(
            ["framework", "dtype", "load", "version", "n_elems"],
            observed=True
        ).agg({"solver_time": "mean"}
    ).reset_index()


    # remove FP32-K 
    stats = stats[
        ~(
            (stats["dtype"] == "fp32") & (stats["version"] == "K")
        )
    ].copy()


    pivot = stats.pivot_table(
        index=["framework", "load", "version", "n_elems"],
        columns="dtype",
        values="solver_time",
        observed=False,
    ).reset_index()
    
    pivot = pivot.dropna(subset=["fp32", "fp64"])
    pivot["ratio"] = ( pivot["fp32"] / pivot["fp64"] )

    loads = ["gravity", "traction", "compression"]
    color_map = {
        "jax": "blue",
        "pytorch": "red",
        "warp": "green"
    }

    linestyle_map = {
        #"K": "-",
        "B": "--",
        "G": ":"
    }

    for load in loads:
        plt.figure(figsize=(7, 5))
        sub_load = pivot[pivot["load"] == load]
        for (framework, version), sub in sub_load.groupby(
            ["framework", "version"],
            sort=True,
            observed=True,
        ):
            sub = sub.sort_values("n_elems")
            x = sub["n_elems"].values
            y = sub["ratio"].values
            plt.plot(
                x,
                y,
                marker='o',
                color=color_map[framework],
                linestyle=linestyle_map[version],
                label=f"{framework}-{version}"
            )

        # formatting
        plt.xscale("log")
        all_x = sorted(pivot["n_elems"].unique())
        plt.xticks(
            all_x,
            [f"{int(v):,}" for v in all_x],
            rotation=20
        )
        plt.xlabel("Number of elements (log scale)")
        plt.ylabel(r"$t_{FP32} / t_{FP64}$")
        plt.title(f"FP32 / FP64 solver runtime ratio, {load}")
        plt.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.5
        )
        plt.axhline(
            y=1.0,
            color='black',
            linestyle='--',
            linewidth=1.3,
            alpha=0.7,
        )
        plt.ylim(-0.02, 2.1)
        plt.legend(fontsize=8)
        plt.tight_layout()
        save_path = ( Path("./analysis_res") / f"precision_ratio_{load}.png")
        plt.savefig(save_path, dpi=300)
        plt.show()




framework_lst = [ "pytorch", "jax", "warp" ]
plot_precision_ratio(framework_lst)