
import torch
from pytorch_lib.pytorch_utils import cg_solver 



def precompute_B(nodes, elems, lam, mu, dtype, device):

    n_elem = elems.shape[0]

    # Gather node coordinates for all elements
    p0 = nodes[elems[:, 0]]
    p1 = nodes[elems[:, 1]]
    p2 = nodes[elems[:, 2]]
    p3 = nodes[elems[:, 3]]

    # Build Jacobians (n_elem, 3, 3)
    J = torch.stack([p1 - p0, p2 - p0, p3 - p0], dim=2)

    detJ = torch.det(J)
    V_all = torch.abs(detJ) / 6.0
    invJ = torch.linalg.inv(J)

    grad_hat = torch.tensor([
        [-1., -1., -1.],
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.]
    ], dtype=dtype, device=device)


    grads = grad_hat @ invJ 

    # build B matrices in batch
    B_all = torch.zeros((n_elem, 6, 12), dtype=dtype, device=device)

    for i in range(4):
        gx = grads[:, i, 0]
        gy = grads[:, i, 1]
        gz = grads[:, i, 2]
        col = 3 * i

        B_all[:, 0, col+0] = gx
        B_all[:, 1, col+1] = gy
        B_all[:, 2, col+2] = gz

        B_all[:, 3, col+0] = gy
        B_all[:, 3, col+1] = gx
        B_all[:, 4, col+1] = gz
        B_all[:, 4, col+2] = gy
        B_all[:, 5, col+0] = gz
        B_all[:, 5, col+2] = gx

    D = torch.tensor([
        [lam+2*mu, lam, lam, 0, 0, 0],
        [lam, lam+2*mu, lam, 0, 0, 0],
        [lam, lam, lam+2*mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu],
    ], dtype=dtype, device=device)

    return {"B_all": B_all, "V_all": V_all, "D": D}



def apply_K_B(u, dofs_all, dofs_fix, B_all, V_all, D):

    # Gather local displacements
    u_e = u[dofs_all]    # (n_elem,12)
    # strain = B * u
    strain = torch.einsum("eij,ej->ei", B_all, u_e)
    # stress = D * strain
    stress = torch.einsum("ij,ej->ei", D, strain)
    # internal force = V * B^T * stress
    fe_all = torch.einsum("eji,ej->ei", B_all, stress)
    fe_all = fe_all * V_all[:,None]

    # Scatter-add
    result = torch.zeros_like(u)
    result.index_add_(0, dofs_all.reshape(-1), fe_all.reshape(-1))

    # Re-enforce BC
    result[dofs_fix] = 0.0 

    return result





def solver_B(precomp, dofs_all, dofs_fix, lam, mu, F, n_elem,
                 rtol, maxiter, 
                 load,
                 framework,
                 version,
                 plot_cvg
        ):


    def A(x):
        return apply_K_B(x, dofs_all, dofs_fix, precomp["B_all"], 
                         precomp["V_all"], precomp["D"])

    u, cnt  = cg_solver(A, F, n_elem,
                 rtol, maxiter, 
                 load,
                 framework,
                 version,
                 plot_cvg
        )
    return u, cnt 



