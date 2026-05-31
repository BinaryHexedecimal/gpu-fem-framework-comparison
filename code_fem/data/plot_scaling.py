import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_scaling_all_frameworks(framework, dtype):

    rows = []
    for framework in framework_lst:
        files = list(Path("./raw_data").glob(f"{framework}_{dtype}_realtime_cpu_res.json"))
        for f in files:
            with open(f) as file:
                data = json.load(file)
                for entry in data:
                    base = {
                        "framework": framework,
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
    df = df[df["run_id"]>=2].copy()

    # aggregate
    stats = df.groupby(
                ["framework", "load", "version", "n_elems"], observed=True
            ).agg({"solver_time": "mean"}
        ).reset_index()


    # remove K for FP32
    if dtype == "fp32":
        stats = stats[stats["version"] != "K"].copy()


    loads = ["gravity", "traction", "compression"]
    color_map = {
        "jax": "blue",
        "pytorch": "red",
        "warp": "green"
    }

    linestyle_map = {
        "K": "-",
        "B": "--",
        "G": ":"
    }
    stats["version"] = pd.Categorical(
        stats["version"],
        categories=["K", "B", "G"],
        ordered=True
    )
    for load in loads:
        plt.figure()
        sub_load = stats[stats["load"] == load]
        
        # loop over framework + version
        for (framework, version), sub in sub_load.groupby(
                ["framework", "version"],
                sort=True,
                observed=True,
            ):
            sub = sub.sort_values("n_elems")

            x = sub["n_elems"].values
            #y = sub["solver_time_seconds"].values
            y = sub["solver_time"].values
            coef = np.polyfit(np.log(x), np.log(y), 1)
            slope = coef[0]

            framework_display = {
                "pytorch": "PyTorch",
                "jax": "JAX",
                "warp": "Warp",
            }

            label=f"{framework_display[framework]} ({version}, slope={slope:.2f})"
            plt.plot(
                x, y,
                color=color_map[framework],
                linestyle=linestyle_map[version],
                marker='o',
                label=label,
            )

        # log-log scaling
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(0.01, 10)
        all_x = sorted(stats["n_elems"].unique())
        plt.xticks(
            all_x,
            [f"{int(v):,}" for v in all_x],
            rotation=20
        )
        plt.xlabel("Number of elements (log scale)")
        plt.ylabel("Solver time (s, log scale)")
        plt.title(f"Scaling behavior ({load}, precision: {dtype})")

        plt.legend(fontsize=8)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        save_path = Path("./analysis_res") / f"scaling_{load}_{dtype}.png"
        plt.savefig(save_path, dpi=300)
        plt.show()


dtype_list = ["fp32", "fp64"]
framework_lst = [ "pytorch", "jax", "warp"]

for dtype in dtype_list:
    plot_scaling_all_frameworks(framework_lst, dtype)