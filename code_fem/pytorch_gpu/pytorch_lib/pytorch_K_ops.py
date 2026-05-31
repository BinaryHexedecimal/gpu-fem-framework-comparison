import torch
from pytorch_lib.pytorch_utils import cg_solver 


def precompute_K(nodes, elems, lam, mu, dtype, device):

    n_elem = elems.shape[0]

    # Gather node coordinates
    p0 = nodes[elems[:,0]]
    p1 = nodes[elems[:,1]]
    p2 = nodes[elems[:,2]]
    p3 = nodes[elems[:,3]]

    # Jacobians
    J = torch.stack([p1-p0, p2-p0, p3-p0], dim=2)

    detJ = torch.det(J)
    V_all = torch.abs(detJ) / 6.0
    invJ = torch.linalg.inv(J)

    grad_hat = torch.tensor([
        [-1.,-1.,-1.],
        [ 1., 0., 0.],
        [ 0., 1., 0.],
        [ 0., 0., 1.]
    ], dtype=dtype, device=device)

    # shape gradients
    grads = torch.einsum("ij,ejk->eik", grad_hat, invJ)   # (n_elem,4,3)

    B_all = torch.zeros((n_elem, 6,12), dtype=dtype, device=device)

    for i in range(4):
        gx = grads[:,i,0]
        gy = grads[:,i,1]
        gz = grads[:,i,2]

        c = 3*i

        B_all[:,0,c+0] = gx
        B_all[:,1,c+1] = gy
        B_all[:,2,c+2] = gz

        B_all[:,3,c+0] = gy
        B_all[:,3,c+1] = gx

        B_all[:,4,c+1] = gz
        B_all[:,4,c+2] = gy

        B_all[:,5,c+0] = gz
        B_all[:,5,c+2] = gx


    D = torch.tensor([
        [lam+2*mu, lam, lam, 0,0,0],
        [lam, lam+2*mu, lam, 0,0,0],
        [lam, lam, lam+2*mu, 0,0,0],
        [0,0,0, mu,0,0],
        [0,0,0, 0,mu,0],
        [0,0,0, 0,0,mu],
    ], dtype=dtype, device=device)

    # Ke = V * B^T D B
    K_all = torch.einsum(
        "eji,jk,ekl,e->eil",
        B_all,
        D,
        B_all,
        V_all
    )

    return {"K_all": K_all, "V_all": V_all}



def apply_K_K(u, dofs_all, dofs_fix, K_all):

    # gather local displacements
    u_e = u[dofs_all]   # (n_elem,12)

    # fe = Ke * u
    fe_all = torch.einsum("eij,ej->ei", K_all, u_e)

    result = torch.zeros_like(u)

    result.index_add_(0, dofs_all.reshape(-1), fe_all.reshape(-1))

    result[dofs_fix] = 0.0

    return result





def solver_K(precomp, dofs_all, dofs_fix, lam, mu, F, n_elem,
                 rtol, maxiter, 
                 load,
                 framework,
                 version,
                 plot_cvg
        ):
    def A(x):
        return apply_K_K(x, dofs_all, dofs_fix, precomp["K_all"])

    
    u, cnt = cg_solver(A, F, n_elem,
                 rtol, maxiter, 
                 load,
                 framework,
                 version,
                 plot_cvg
        )
    return u, cnt










