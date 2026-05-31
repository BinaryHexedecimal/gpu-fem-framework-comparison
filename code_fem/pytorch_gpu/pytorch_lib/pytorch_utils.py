import torch
import sys

from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
target_folder = project_root / "data"
target_folder.mkdir(parents=True, exist_ok=True)



from utility.mesh_generation import (
    generate_5tet_tetrahedra,
    generate_6tet_tetrahedra, 
    generate_grid_3d, 
    left_boundary_nodes_center,
    left_boundary_nodes_free,
    left_boundary_nodes,
)


from utility.measurement import plot_convergency

from OpenGL.GL import *




# -------------------------------------------------
# Dofs_all and dofs_fix
# -------------------------------------------------

# The standard GPU trick:
# Instead of removing DOFs, build a mask.
# Another trick in FEM solvers:
# Modify stiffness matrix rows: by enforcesing u[dofs_fix] = 0 without deleting DOFs.



# Build element DOF mapping and fixed DOFs for a clamped left face.
# All displacement components (ux, uy, uz) are fixed on selected boundary nodes.
def build_dofs_left_face_clamped(nodes, elements, device, W, H, ny, nz):
    n_elem= elements.shape[0]

    lb_nodes = torch.tensor(
        left_boundary_nodes(nodes.cpu().numpy()),
        dtype=torch.long,
        device=device
    )
    comp = torch.arange(3, device=device)
    dofs_fix = (3*lb_nodes[:,None] + comp).reshape(-1)

    # DOF mapping (vectorized)
    dofs_all = (3 * elements[:, :, None] + comp).reshape(n_elem, 12)

    return dofs_all, dofs_fix, 



# Build element DOF mapping and fixed DOFs for nodes on the left boundary.
# Partially constrained left face
def build_dofs_left_face_fix_x(nodes, elements, device, W, H, ny, nz):
    n_elem= elements.shape[0]
    lb_center_nodes = torch.tensor(
        left_boundary_nodes_center(nodes.cpu().numpy(), W, H, ny, nz),
        dtype=torch.long,
        device=device
    )
    lb_free_nodes = torch.tensor(
        left_boundary_nodes_free(nodes.cpu().numpy(), W, H, ny, nz),
        dtype=torch.long,
        device=device
    )

    comp = torch.arange(3, device=device)
    lb_center_dofs = (3*lb_center_nodes[:,None] + comp).reshape(-1)
    lb_free_dofs = 3*lb_free_nodes.reshape(-1)

    dofs_fix = torch.cat([lb_center_dofs, lb_free_dofs], dim=0)

    # DOF mapping (vectorized)
    dofs_all = (3 * elements[:, :, None] + comp).reshape(n_elem, 12)

    return dofs_all, dofs_fix






# -------------------------------------------------
# Surface force
# -------------------------------------------------

def build_surface_F_pytorch(nodes, faces, force_vec, dtype, device):

    n_nodes=nodes.shape[0]
    F = torch.zeros(3 * n_nodes, dtype=dtype, device=device)

    # Triangle vertices
    p0 = nodes[faces[:, 0]]
    p1 = nodes[faces[:, 1]]
    p2 = nodes[faces[:, 2]]

    # Triangle areas
    cross_prod = torch.cross(p1 - p0, p2 - p0, dim=1)
    area = 0.5 * torch.linalg.norm(cross_prod, dim=1)

    # Nodal force per triangle node
    nodal_force = force_vec * (area.unsqueeze(1) / 3.0)   # (n_faces,3)

    # Scatter to global vector
    for i in range(3):
        node_ids = faces[:, i]
        F.index_add_(0, 3*node_ids + 0, nodal_force[:,0])
        F.index_add_(0, 3*node_ids + 1, nodal_force[:,1])
        F.index_add_(0, 3*node_ids + 2, nodal_force[:,2])

    return F




# -------------------------------------------------
# Body force
# -------------------------------------------------

def build_body_F_pytorch(n_nodes, dofs_all, V_all, force_vec, dtype, device):

    F = torch.zeros(3*n_nodes, dtype=dtype, device=device)
    scale = (V_all / 4.0)[:, None, None]
    scale = scale.repeat(1, 4, 1)

    Fe_all = (scale * force_vec).reshape(-1, 12)

    F.index_add_(0, dofs_all.reshape(-1), Fe_all.reshape(-1))

    return F




# -------------------------------------------------
# Mesh
# -------------------------------------------------


def prepare_mesh_pytorch(nx, ny, nz, L, W, H, tet, dtype, device):
    
    if tet == 6:
        elements_np = generate_6tet_tetrahedra(nx, ny, nz)
    elif tet == 5:
        elements_np = generate_5tet_tetrahedra(nx, ny, nz)
    else:
        raise ValueError("Only 5 tet and 6 tet can be accepted")
    #print(f"{tet} tetrehedras per cubic.")
    nodes_np = generate_grid_3d(nx, ny, nz, L, W, H)
    nodes = torch.tensor(
        nodes_np,
        dtype=dtype,
        device=device
    )
    elements = torch.tensor(
        elements_np,
        dtype = torch.long, #dtype=dtype,
        device=device
    )
    return elements, nodes, elements_np, nodes_np



# -------------------------------------------------
# generic CG solver
# -------------------------------------------------

def solver_core(A_func, F, rtol, maxiter, record):
    """
    Generic Conjugate Gradient solver for Ax = F.
    A_func(x) computes the matrix-vector product Ax.
    """
    if record:
        it_history = []
        relres_history = []

    # Initial guess
    x = torch.zeros_like(F) 

    # Initial residual
    r = F - A_func(x)
    p = r.clone()

    rr_old = torch.dot(r, r)
    r0_norm = torch.sqrt(rr_old).item()

    if r0_norm == 0:
        print("Initial residual is zero.")
        return x
    
    for i in range(maxiter):
        Ap = A_func(p)
        denom = torch.dot(p, Ap).item()
        if abs(denom) < 1e-20:
            print("CG breakdown (denominator too small)")
            i = -1
            break

        alpha = rr_old / denom

        x = x + alpha * p
        r = r - alpha * Ap

        rr_new = torch.dot(r, r)
        r_norm = torch.sqrt(rr_new).item()


        if record:
            if i % 10 == 0:
                it_history.append(i)
                relres_history.append(r_norm / r0_norm)

        # Relative stopping criterion
        if r_norm < rtol * r0_norm:
            break

        beta = rr_new / rr_old
        p = r + beta * p

        rr_old = rr_new

    if record:
        return x, i, it_history, relres_history 
    else :
        return x, i



def cg_solver(A, F, n_elem,
                 rtol, maxiter, 
                 load,
                 framework,
                 version,
                 plot_cvg
        ):
    if plot_cvg: 
        u, cnt, it_history, relres_history = solver_core(A, F, rtol, maxiter, True)
    else:
        u, cnt =  solver_core(A, F, rtol, maxiter, False)


    #---control convergency -----
    if cnt == maxiter - 1:
        print("CG reached max iterations", file=sys.stderr)
        print("CG did not converge", file=sys.stderr)
    else:
        print(f"CG converged in {cnt} iterations", file=sys.stderr)

    if plot_cvg:
        # ---- detect precision ----
        if u.dtype == torch.float32:
            precision = "fp32"
        elif u.dtype == torch.float64:
            precision = "fp64"
        else:
            precision = str(u.dtype)
        title = f"CG Convergence, {load}, {framework}, {n_elem} elements, {version}, {precision}"
        filename = f"Cvg_{n_elem}_elem_{load}_{version}_{precision}.png"
        plot_convergency(it_history, relres_history, title, filename)

    return u, cnt


















