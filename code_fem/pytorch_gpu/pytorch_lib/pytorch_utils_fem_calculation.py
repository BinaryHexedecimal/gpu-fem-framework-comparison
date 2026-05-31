import torch

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
target_folder = project_root / "data"
target_folder.mkdir(parents=True, exist_ok=True)


from utility.mesh_generation import detect_right_boundary_faces_from_nodes
from utility.measurement import summarize_solution 

from pytorch_lib.pytorch_utils import (
    prepare_mesh_pytorch,
    build_body_F_pytorch,
    build_surface_F_pytorch,
)

from OpenGL.GL import *



def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


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

        dtype, 
        device,

        correctness_check,

        plot_cvg,
        plot_pyvista, 
        plot_by_openGL,

        num_runtimes,

):
    # ----------------------------
    # Memory monitor start
    # ----------------------------
    torch.set_grad_enabled(False)
    n_elem = (nx-1)*(ny-1)*(nz-1)*tet

    data_records = {
        "framework": framework,
        "version": version,
        "load": load,
        "fp": "fp64" if dtype == torch.float64 else "fp32",
        "n_elems": n_elem,
        "runs": {},
    }

    for run in range(1, num_runtimes+1):

        # ----------------------------------------------- #
        # --------------Pre-computation------------------ #
        # ----------------------------------------------- #
        
        elements, nodes, elements_np, nodes_np = prepare_mesh_pytorch(nx, ny, nz, L, W, H, tet, dtype, device)
        precomp = precompute_func(nodes, elements, lam, mu, dtype, device)
        n_nodes = nodes_np.shape[0]

        dofs_all, dofs_fix = precalculate_dofs_func(nodes, elements, device, W, H, ny, nz)

        if load == "compression":
            force_vec = torch.tensor([force_val, 0.0, 0.0], dtype=dtype, device=device)
        elif load in ("gravity", "traction"):
            force_vec = torch.tensor([0.0, 0.0, force_val], dtype=dtype, device=device)
        else:
            raise ValueError("Unknown load")

        if force_type == "body":
            F = build_body_F_pytorch(n_nodes, dofs_all, precomp["V_all"], force_vec, dtype, device)
            F[dofs_fix] = 0.0
        elif force_type == "surface":
            faces_np = detect_right_boundary_faces_from_nodes(elements_np, nodes_np, L)
            faces = torch.tensor(faces_np, dtype=torch.long, device=device)
            F=build_surface_F_pytorch(nodes, faces, force_vec, dtype, device)
            F[dofs_fix] = 0.0
        else:
            raise ValueError(f"Unknown force_type: {force_type}")
        assert force_vec.dtype == dtype
        assert nodes.dtype == dtype

        u, cnt = solver_func(precomp, dofs_all, dofs_fix,lam, mu, F, n_elem,
                    rtol, maxiter, 
                    load,
                    framework,
                    version,
                    plot_cvg
            )
            
        assert u.dtype == dtype


        if u.requires_grad:
            print("WARNING: autograd is ON", file=sys.stderr)


        # ----------------------------------------------- #
        # -----------------measure----------------------- #
        # ----------------------------------------------- #
        if correctness_check:
            u_np = u.detach().cpu().numpy()
            numerical_res = summarize_solution(u_np, nodes_np, elements_np, 
                                nx, ny, nz,
                                load, 
                                theoretical_res,
                                framework, version,
                                plot_pyvista,
                                plot_by_openGL,
                    )
            numerical_res["cg_iteration"] = cnt
            data_records["runs"][run] = numerical_res

    return data_records


