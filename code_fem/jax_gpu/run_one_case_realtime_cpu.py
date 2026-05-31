import jax
import argparse
import json

from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utility.GEOMETRI import MESHES, COMMON_PARAMETERS






def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mesh", type=str, default = "coarse")
    parser.add_argument("--model", type=str, default = "gravity")
    parser.add_argument("--version", type=str, default = "B")
    parser.add_argument("--dtype_name", type=str, default = "fp64")

    parser.add_argument("--real_runtimes", type=int, default=1)
    parser.add_argument("--precompute_runtimes", type=int, default=1)
    parser.add_argument("--auto_force_input", action="store_true")

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
        #run_simulation,
    )

    from jax_lib.jax_utils_realtime_via_cpu import simulation_user_driven_via_CPU

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



    framework="jax"


    if args.auto_force_input:
        auto_force = [ model_parameters[args.model]["force_val"] for _ in range(args.real_runtimes) ]
    else:
        auto_force = None

    res = simulation_user_driven_via_CPU (
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

                    load = args.model,
                    framework = framework,
                    version = args.version,

                    rtol = COMMON_PARAMETERS["rtol"],
                    maxiter = COMMON_PARAMETERS["maxiter"],

                    precompute_times = args.precompute_runtimes,
                    monitor_memory = True,
                    inputs = auto_force,

                    dtype = dtype,

                )
    print(json.dumps(res))

if __name__ == "__main__":
    main()



