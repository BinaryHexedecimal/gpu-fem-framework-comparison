# pyright: reportInvalidTypeForm=false

import sys

import warp_lib.warp_config as wcfg
import warp as wp

from OpenGL.GL import *


from utility.mesh_generation import (
    detect_right_boundary_faces_from_nodes,
)
from utility.measurement import summarize_solution

from warp_lib.warp_utils import (
    prepare_mesh_warp,
    precalculate_body_F_warp,
    precalculate_surface_F_warp,
    enforce_dirichlet_on_F_kernel,

)





def run_simulation_fem(
        nx, ny, nz, 
        L, W, H, tet,
        lam, mu,

        precompute_func,  
        solver_func,          

        precalculate_fixed_func,  
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

        num_runtimes,
        
):
    

    n_elem = (nx-1)*(ny-1)*(nz-1)*tet
    data_records = {
        "framework": framework,
        "version": version,
        "load": load,
        "fp": "fp64" if wcfg.USE_FP64 else "fp32",
        "n_elems": n_elem,
        "runs": {},
    }

    lam = wcfg.scalar(lam)
    mu  = wcfg.scalar(mu)
    
    for run in range(1, num_runtimes+1):
        elements_wp, nodes_wp, elements_np, nodes_np = prepare_mesh_warp(nx, ny, nz, L, W, H, tet)

        n_nodes = nodes_np.shape[0]

        fixed_mask = precalculate_fixed_func(nodes_np, W, H, ny, nz)
        precomp = precompute_func(nodes_wp, elements_wp)


        if load == "compression":
            force_vec = wcfg.vec3(
                    wcfg.scalar(force_val),
                    wcfg.scalar(0.0),
                    wcfg.scalar(0.0)
                )
        elif load in ("gravity", "traction"):
            force_vec = wcfg.vec3(
                    wcfg.scalar(0.0),
                    wcfg.scalar(0.0),
                    wcfg.scalar(force_val)
                )
        else:
            raise ValueError("Unknown load")


        if force_type == "body":
            F = precalculate_body_F_warp(n_nodes, elements_wp, force_vec, precomp["V_all"])
            # enforce Dirichlet on RHS
            wp.launch(enforce_dirichlet_on_F_kernel,
                    dim=n_nodes,
                    inputs=[F, fixed_mask])
        elif force_type == "surface":
            faces_np = detect_right_boundary_faces_from_nodes(elements_np, nodes_np, L)
            faces_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32)
            F = precalculate_surface_F_warp(nodes_wp, faces_wp, force_vec)
            wp.launch(
                enforce_dirichlet_on_F_kernel,
                dim=n_nodes,
                inputs=[F, fixed_mask]
            )
        else:
            raise ValueError(f"Unknown force_type: {force_type}")


        assert type(force_vec) == wcfg.vec3
        assert nodes_wp.dtype == wcfg.vec3
        
        u, iters = solver_func(precomp, fixed_mask, elements_wp, F, n_nodes, lam, mu, rtol, maxiter)

        assert u.dtype == wcfg.vec3
        
            
        print(f"CG iterations: {iters}", file=sys.stderr)
        if iters == maxiter:
            print("CG reached max iterations", file=sys.stderr)

        
        # ----------------------------------------------- #
        # ----------------measure error------------------ #
        # ----------------------------------------------- #
        if correctness_check:
            u_np = u.numpy()

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

            numerical_res["cg_iteration"] = iters
            data_records["runs"][run] = numerical_res

    return data_records





