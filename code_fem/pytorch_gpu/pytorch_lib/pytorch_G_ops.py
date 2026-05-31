
import torch

from pytorch_lib.pytorch_utils import cg_solver #, solver_generator


def precompute_G(nodes, elems, lam, mu, dtype, device):

    p0 = nodes[elems[:,0]]
    p1 = nodes[elems[:,1]]
    p2 = nodes[elems[:,2]]
    p3 = nodes[elems[:,3]]

    J = torch.stack([p1-p0, p2-p0, p3-p0], dim=2)

    detJ = torch.det(J)
    V_all = torch.abs(detJ) / 6.0

    invJ = torch.linalg.inv(J)

    grad_hat = torch.tensor([
        [-1.,-1.,-1.],
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.]
    ], dtype=dtype, device=device)

    grads_all = torch.matmul(
        grad_hat.unsqueeze(0),
        invJ
    )  # (n_elem,4,3)

    return {"grads_all": grads_all, "V_all": V_all}



def apply_K_grad(u, dofs_all, dofs_fix, grads_all, V_all, lam, mu):

    n_elem = grads_all.shape[0]

    u_e = u[dofs_all].reshape(n_elem,4,3)

    g0 = grads_all[:,0]
    g1 = grads_all[:,1]
    g2 = grads_all[:,2]
    g3 = grads_all[:,3]

    u0 = u_e[:,0]
    u1 = u_e[:,1]
    u2 = u_e[:,2]
    u3 = u_e[:,3]

    du_dx = (
        torch.einsum("ni,nj->nij", u0, g0) +
        torch.einsum("ni,nj->nij", u1, g1) +
        torch.einsum("ni,nj->nij", u2, g2) +
        torch.einsum("ni,nj->nij", u3, g3)
    )

    eps = 0.5*(du_dx + du_dx.transpose(1,2))

    tr = eps[:,0,0] + eps[:,1,1] + eps[:,2,2]

    I = torch.eye(3, dtype=u.dtype, device=u.device)

    sigma = lam*tr[:,None,None]*I + 2*mu*eps

    f0 = V_all[:,None]*(sigma @ g0.unsqueeze(-1)).squeeze(-1)
    f1 = V_all[:,None]*(sigma @ g1.unsqueeze(-1)).squeeze(-1)
    f2 = V_all[:,None]*(sigma @ g2.unsqueeze(-1)).squeeze(-1)
    f3 = V_all[:,None]*(sigma @ g3.unsqueeze(-1)).squeeze(-1)

    fe = torch.stack([f0,f1,f2,f3], dim=1).reshape(n_elem,12)

    result = torch.zeros_like(u)
    result.index_add_(0, dofs_all.reshape(-1), fe.reshape(-1))

    result[dofs_fix] = 0

    return result




def solver_G(precomp, dofs_all, dofs_fix,
                 lam, mu, F, n_elem,
                 rtol, maxiter, 
                 load,
                 framework,
                 version,
                 plot_cvg
        ):

    def A(x):
        return apply_K_grad(x, dofs_all, dofs_fix, precomp["grads_all"], precomp["V_all"], lam, mu)

    u, cnt  = cg_solver(A, F, n_elem,
                 rtol, maxiter, 
                 load,
                 framework,
                 version,
                 plot_cvg
        )
    return u, cnt 

