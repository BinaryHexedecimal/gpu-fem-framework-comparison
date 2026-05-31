# -------------------------------------------------
# Kappa
# -------------------------------------------------
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from pytorch_lib.pytorch_K_ops import precompute_K
from pytorch_lib.pytorch_utils import (
    prepare_mesh_pytorch, 
    build_body_F_pytorch, 
    build_surface_F_pytorch,

)
from utility.mesh_generation import detect_right_boundary_faces_from_nodes


import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


import torch
import matplotlib.pyplot as plt

def assemble_global_K(dofs_all, Ke_all, n_dofs):

    n_elem = dofs_all.shape[0]
    n_loc  = dofs_all.shape[1]   # 12

    # Expand DOF indices
    rows = dofs_all.unsqueeze(2).expand(n_elem, n_loc, n_loc)
    cols = dofs_all.unsqueeze(1).expand(n_elem, n_loc, n_loc)

    rows = rows.reshape(-1).cpu().numpy()
    cols = cols.reshape(-1).cpu().numpy()

    vals = Ke_all.reshape(-1).cpu().numpy()

    # assemble sparse matrix
    K = sp.coo_matrix((vals, (rows, cols)), shape=(n_dofs, n_dofs))

    # Convert to CSR (required for eigensolvers)
    K = K.tocsr()

    return K




def apply_dirichlet_bc(K, dofs_fix):

    n_dofs = K.shape[0]

    mask = np.ones(n_dofs, dtype=bool)
    mask[dofs_fix.cpu().numpy()] = False

    free_dofs = np.where(mask)[0]

    K_ff = K[free_dofs][:, free_dofs]

    return K_ff



def apply_dirichlet_bc_F(K, F, dofs_fix):

    n_dofs = K.shape[0]

    mask = np.ones(n_dofs, dtype=bool)
    mask[dofs_fix.cpu().numpy()] = False

    free_dofs = np.where(mask)[0]

    K_ff = K[free_dofs][:, free_dofs]
    F_f  = F[free_dofs]
    return K_ff, F_f



def build_reduced_stiffness_matrix(
                    nx, ny, nz, L, W, H, tet,
                    lam, mu,
                    load,
                    precalculate_dofs_func,   

                    dtype, 
                    device,
                   ):
    print(f"Load: {load}")
    print(f"The number of elements: {nx-1}*{ny-1}*{nz-1}*{tet}")
    
    elements, nodes, elements_np, nodes_np = prepare_mesh_pytorch(nx, ny, nz, L, W, H, tet, dtype, device)
    print(f"Nr. of element: {elements_np.shape[0]}")
    res = precompute_K(nodes, elements, lam, mu, dtype, device)
    Ke_all = res["K_all"]
    dofs_all, dofs_fix = precalculate_dofs_func(nodes, elements, device)

    n_dofs = int(dofs_all.max().item() + 1)

    print("Assembling global stiffness matrix...")
    K = assemble_global_K(dofs_all, Ke_all, n_dofs)

    print("Applying boundary conditions...")
    K_ff = apply_dirichlet_bc(K, dofs_fix)

    return K_ff


    
def calculate_projection(nx, ny, nz, L, W, H, tet, 
                lam, mu,
                force_type,
                precalculate_dofs_func,
                force_vec,
                load,
                dtype, device):
    elements, nodes, elements_np, nodes_np = prepare_mesh_pytorch(nx, ny, nz, L, W, H, tet, dtype, device)
    res = precompute_K(nodes, elements, lam, mu, dtype, device)
    Ke_all = res["K_all"]
    V_all = res["V_all"]
    dofs_all, dofs_fix = precalculate_dofs_func(nodes, elements, device, W, H, ny, nz)

    n_nodes = nodes_np.shape[0]
    n_elem = elements_np.shape[0]
    dofs_all, dofs_fix = precalculate_dofs_func(nodes, elements, device, W, H, ny, nz)
    n_dofs = int(dofs_all.max().item() + 1)

    if force_type == "body":
        F = build_body_F_pytorch(n_nodes, dofs_all, V_all, force_vec, dtype, device)
        F[dofs_fix] = 0.0
    elif force_type == "surface":
        faces_np = detect_right_boundary_faces_from_nodes(elements_np, nodes_np, L)
        faces = torch.tensor(faces_np, dtype=torch.long, device=device)
        F=build_surface_F_pytorch(nodes, faces, dofs_fix, force_vec, dtype, device)
        F[dofs_fix] = 0.0
    else:
        raise ValueError(f"Unknown force_type: {force_type}")
    

    print(f"F størrelse {F.shape}")

    print("Assembling global stiffness matrix...")
    K = assemble_global_K(dofs_all, Ke_all, n_dofs)
    print(f"K has shape {K.shape}")

    print("Applying boundary conditions...")
    K_ff, F_ff = apply_dirichlet_bc_F(K, F, dofs_fix)
    print(f"K_ff has shape {K_ff.shape}")

    print("Computing eigenvalues...")

    # Compute smallest eigenpairs
    K_dense = K_ff.toarray()   
    eigvals, eigvecs = np.linalg.eigh(K_dense)

    # Convert F to numpy
    F_np = F_ff.cpu().numpy().reshape(-1)

    # Compute projections 
    projections = eigvecs.T @ F_np

    # Energy contribution:
    energy = (projections**2)/eigvals

    mask = eigvals > 0 #1e-40
    eigvals = eigvals[mask]


    # Eigenvalue distribution
    plt.figure()
    plt.hist(eigvals, bins=50)
    #plt.yscale('log')  # optional, if counts vary a lot

    plt.xlabel(r'Eigenvalues $\lambda_i$')
    plt.ylabel('Count')
    plt.title(f'Eigenvalue distribution, {load}, {n_elem} elements')

    plt.savefig(f"figures/{load}_{n_elem}_eigen_hist.png", dpi=300)
    plt.show()


    # Energy distribution
    plt.figure()

    # Avoid zeros for log scale
    energy_positive = energy[energy > 0] #1e-50]
    plt.scatter(range(len(energy_positive)), energy_positive, s=5)
    plt.yscale('log')


    plt.ylabel(r'Energy contribution ')
    plt.xlabel('Eigenmode i')
    plt.title(r'Energy contribution $\frac{1}{\lambda_i}(v_i^T F)^2$,' + f' {n_elem} elements, {load}')

    plt.savefig(f"figures/{load}_{n_elem}_energy.png", dpi=300)
    plt.show()
    print(f"save for {n_elem} elements, {load}")








def analyze_spectrum(K_ff, tol=1e-8):

    # largest eigenvalue
    lambda_max = eigsh(
        K_ff,
        k=1,
        which='LM',
        return_eigenvectors=False
    )[0]

    # several smallest eigenvalues
    eigvals = eigsh(
        K_ff,
        k=6,
        which='SM',
        return_eigenvectors=False
    )

    eigvals = np.sort(eigvals)

    lambda_min = eigvals[0]

    # effective smallest eigenvalue
    nonzero = eigvals[eigvals > tol]

    if len(nonzero) > 0:
        lambda_min_eff = nonzero[0]
        kappa_eff = lambda_max / lambda_min_eff
    else:
        lambda_min_eff = np.nan
        kappa_eff = np.nan

    kappa = lambda_max / lambda_min

    return {
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "kappa": kappa,
        "lambda_min_eff": lambda_min_eff,
        "kappa_eff": kappa_eff,
        "small_eigs": eigvals.tolist(),
    }