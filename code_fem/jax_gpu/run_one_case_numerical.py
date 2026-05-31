import jax
import argparse
import json

from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utility.GEOMETRI import MESHES, COMMON_PARAMETERS


framework="jax"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mesh", type=str, default = "coarse")
    parser.add_argument("--model", type=str, default = "gravity")
    parser.add_argument("--version", type=str, default = "G")
    parser.add_argument("--dtype_name", type=str, default = "fp64")
    parser.add_argument("--numerical_runtimes", type=int, default = 1)


    return parser.parse_args()



                    
def main():

    args = parse_args()


    if args.dtype_name == "fp64":
        jax.config.update("jax_enable_x64", True)

    elif args.dtype_name == "fp32":
        jax.config.update("jax_enable_x64", False)

    else:
        raise ValueError
    

    print("JAX devices:", jax.devices(), file=sys.stderr)
    print("Default backend:", jax.default_backend(), file=sys.stderr)


    import jax.numpy as jnp

    if args.dtype_name == "fp64":
        dtype = jnp.float64
    elif args.dtype_name == "fp32":
        dtype = jnp.float32
    else:
        raise ValueError


    from jax_lib.jax_utils import (
        build_dofs_left_face_fix_x, 
        build_dofs_left_face_clamped, 

    )

    from jax_lib.jax_utils_fem_calculation import run_simulation_fem

    from jax_lib.jax_B_ops import solver_B, precompute_B
    from jax_lib.jax_G_ops import solver_G, precompute_G
    from jax_lib.jax_K_ops import solver_K, precompute_K


    version_dict = {
        "K": {"precompute": precompute_K, "solver": solver_K},
        "B": {"precompute": precompute_B, "solver": solver_B},
        "G": {"precompute": precompute_G, "solver": solver_G},
    }

    model_parameters = {
        "gravity": {
            "force_type": "body",
            "theoretical_res": "uz=1.23", 
            "force_val" : 0.05,
            "precalculate_dofs_func": build_dofs_left_face_clamped,
        },    

        "traction": {
            "force_type": "surface",
            "theoretical_res": "uz = 0.82", 
            "force_val" : 0.1,
            "precalculate_dofs_func": build_dofs_left_face_clamped
        },
        
        "compression": {
            "force_type": "surface",
            "theoretical_res": "ux = -0.64",
            "force_val" : -20,
            "precalculate_dofs_func": build_dofs_left_face_fix_x
        },

    }

    res = run_simulation_fem (
                    nx = MESHES[args.mesh]["nx"], 
                    ny = MESHES[args.mesh]["ny"], 
                    nz = MESHES[args.mesh]["nz"], 

                    L = COMMON_PARAMETERS["L"], 
                    W = COMMON_PARAMETERS["W"], 
                    H = COMMON_PARAMETERS["H"], 
                    tet = COMMON_PARAMETERS["tet"],
                    lam = COMMON_PARAMETERS["lam"], 
                    mu = COMMON_PARAMETERS["mu"],

                    precompute_func = version_dict[args.version]["precompute"],
                    solver_func =  version_dict[args.version]["solver"],


                    precalculate_dofs_func =  model_parameters[args.model]["precalculate_dofs_func"],
                    force_type = model_parameters[args.model]["force_type"], 
                    force_val = model_parameters[args.model]["force_val"],


                    load = args.model,
                    framework = framework,
                    version = args.version,
                    theoretical_res = model_parameters[args.model]["theoretical_res"],

                    rtol = COMMON_PARAMETERS["rtol"],
                    maxiter = COMMON_PARAMETERS["maxiter"],

                    correctness_check = True,

                    plot_pyvista = False,
                    plot_by_openGL = False,

                    dtype = dtype,
                    num_runtimes = args.numerical_runtimes,

                )
    print(json.dumps(res))


if __name__ == "__main__":
    main()
