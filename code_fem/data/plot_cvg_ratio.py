import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_cg_ratio(framework_lst):

    # Load FP32 + FP64
    rows = []
    for fp in ["fp32", "fp64"]:
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
                    for run_id, run_data in entry["runs"].items():
                        rows.append({
                            "framework":
                                entry["framework"],
                            "dtype":
                                entry["fp"],
                            "load":
                                entry["load"],
                            "version":
                                entry["version"],
                            "n_elems":
                                entry["n_elems"],
                            "run":
                                int(run_id),
                            "cg_iteration":
                                run_data["cg_iteration"],
                        })

    df = pd.DataFrame(rows)

    # Remove invalid vaules
    df = df[
        df["cg_iteration"] > 0
    ].copy()

    df = df[df["version"] != "K"]


    # Avg. over runs
    stats = df.groupby(
            [
                "framework",
                "dtype",
                "load",
                "version",
                "n_elems"
            ],
            observed=True
        ).agg({"cg_iteration": ["mean", "std"]}
    ).reset_index()

    stats.columns = [
        "_".join(col).strip("_")
        for col in stats.columns
    ]


    pivot = stats.pivot_table(
        index=[
            "framework",
            "load",
            "version",
            "n_elems"
        ],
        columns="dtype",
        values="cg_iteration_mean",
        observed=True,
    ).reset_index()

    # keep only valid fp32/fp64 pairs
    pivot = pivot.dropna(
        subset=["fp32", "fp64"]
    )
    pivot["ratio"] = (
        pivot["fp32"]
        / pivot["fp64"]
    )

    loads = [
        "gravity",
        "traction",
        "compression"
    ]

    color_map = {
        "pytorch": "red",
        "jax": "blue",
        "warp": "green",
    }

    linestyle_map = {
       # "K": "-",
        "B": "--",
        "G": ":"
    }

    framework_display = {
        "pytorch": "PyTorch",
        "jax": "JAX",
        "warp": "Warp",
    }

    save_dir = Path("./analysis_res")
    save_dir.mkdir(parents=True, exist_ok=True)

    for load in loads:
        plt.figure(figsize=(7, 5))
        sub_load = pivot[
            pivot["load"] == load
        ]
        for (framework, version), sub in sub_load.groupby(
            ["framework", "version"],
            observed=True,
            sort=True,
        ):
            sub = sub.sort_values("n_elems")
            x = sub["n_elems"].values
            y = sub["ratio"].values
            plt.plot(
                x,
                y,
                marker='o',
                color=color_map.get(framework, "black"),
                linestyle=linestyle_map.get(version, "-"),
                label=f"{framework_display.get(framework, framework)}-{version}"
            )

        # format
        plt.xscale("log")
        all_x = sorted(pivot["n_elems"].unique())
        plt.xticks(
            all_x,
            [f"{int(v):,}" for v in all_x],
            rotation=20
        )
        plt.xlabel("Number of elements (log scale)")
        plt.ylabel(r"$iter_{FP32} / iter_{FP64}$")
        plt.title(f"FP32 / FP64 CG iteration ratio ({load})")
        
        # y = 1 reference line
        plt.axhline(
            y=1.0,
            color='black',
            linestyle='--',
            linewidth=1.0,
            alpha=0.7,
        )
        plt.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.5
        )
        plt.legend(fontsize=8)
        plt.ylim(0.95, 3)
        plt.tight_layout()
        save_path = (save_dir / f"cg_ratio_{load}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.show()



framework_lst = [
    "pytorch",
    "jax",
    "warp",
]
plot_cg_ratio(framework_lst)