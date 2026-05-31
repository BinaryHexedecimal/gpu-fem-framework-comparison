from utility.mesh_generation import (
    detect_right_boundary_faces_from_nodes,
)

import jax.numpy as jnp
import numpy as np

from utility.measurement import summarize_solution

from jax_lib.jax_utils import (
    prepare_mesh_jax,
    precalculate_body_F_jax,
    precalculate_surface_F_jax,
    )

from OpenGL.GL import *

# Inside a @jit function you can only use:
# jax.numpy operations
# JAX primitives
# static Python control flow



def run_simulation_fem(
        nx, ny, nz, 
        L, W, H, tet,
        lam, mu,

        precompute_func,      
        solver_func,           

        precalculate_dofs_func,
        force_type, 
        force_val,

        load, 
        framework, 
        version,
        theoretical_res,

        rtol, 
        maxiter,

        correctness_check,

        plot_pyvista, 
        plot_by_openGL,

        dtype,
        num_runtimes,
       
):

    data_records = {
        "framework": framework,
        "version": version,
        "load": load,
        "fp": "fp64" if dtype == jnp.float64 else "fp32",
        "n_elems": (nx-1)*(ny-1)*(nz-1)*tet,
        "runs": {},
    }

    lam = dtype(lam)
    mu  = dtype(mu)

    for run in range(1, num_runtimes+1):

        elements, nodes, elements_np, nodes_np = prepare_mesh_jax(nx, ny, nz, L, W, H, tet, dtype)

        # variant precomputation
        precomp = precompute_func(nodes, elements, lam, mu)

        dofs_all, dofs_fix = precalculate_dofs_func(nodes, elements, W, H, ny, nz)

        # ----------------------------------------------- #
        # ------------------ solver --------------------- #
        # ----------------------------------------------- #

        if load == "compression":
            force_vec = jnp.array(
                [ dtype(force_val), dtype(0.0), dtype(0.0),],
                dtype=dtype
            )
        elif load in ("gravity", "traction"):
            force_vec = jnp.array(
                [dtype(0.0), dtype(0.0), dtype(force_val)],
                dtype=dtype
            )
        else:
            raise ValueError("Unknown load")

        if force_type == "body":
            F = precalculate_body_F_jax(nodes, dofs_all, force_vec, precomp["V_all"])
            F = F.at[dofs_fix].set(0.0)
        elif force_type == "surface":
            faces_np = detect_right_boundary_faces_from_nodes(elements_np, nodes_np, L)
            faces = jnp.array(faces_np, dtype=jnp.int32)
            F = precalculate_surface_F_jax(nodes, faces, force_vec)
            F = F.at[dofs_fix].set(0.0)
        else:
            raise ValueError(f"Unknown force_type: {force_type}")


        assert force_vec.dtype == dtype
        assert nodes.dtype == dtype

        u, rel_res = solver_func(precomp, dofs_all, dofs_fix, lam, mu, F, rtol, maxiter)
        assert u.dtype == dtype


        # print(f"Relative residual: {rel_res:.3e}, rtol: {rtol:.3e}")
        # if rel_res <= rtol:
        #     print("Converged")
        # else:
        #     print("NOT converged")


        # ----------------------------------------------- #
        # -----------------measure----------------------- #
        # ----------------------------------------------- #
        if correctness_check:
            u_np = np.array(u)

            numerical_res = summarize_solution(
                u_np, nodes_np, elements_np,
                nx, ny, nz,
                load = load,
                theoretical_res = theoretical_res,
                framework = framework,
                version = version,
                plot_pyvista = plot_pyvista,
                plot_by_openGL = plot_by_openGL,
            )

            numerical_res["cg_iteration"] = 1 if rel_res <= rtol else -1
            data_records["runs"][run] = numerical_res

    
    return data_records







