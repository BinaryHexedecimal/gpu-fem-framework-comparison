import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.sparse.linalg import cg



def element_geom(elem_nodes):
    dtype = elem_nodes.dtype
    p0, p1, p2, p3 = elem_nodes
    J = jnp.column_stack((p1 - p0, p2 - p0, p3 - p0))
    detJ = jnp.linalg.det(J)
    six = jnp.array(6.0, dtype=dtype)
    V = jnp.abs(detJ) / six

    grad_hat = jnp.array([
            [-1.0, -1.0, -1.0],
            [ 1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0,  1.0]
        ], dtype=dtype
    )

    invJ = jnp.linalg.inv(J)
    grads = grad_hat @ invJ

    return grads, V

@jit
def precompute_G(nodes, elements, lam, mu):
    #dtype = nodes.dtype
    element_nodes = nodes[elements]  # (n_elem, 4, 3)
    grads_all, V_all = vmap(element_geom)(element_nodes)
    return {"grads_all": grads_all, "V_all": V_all}


# the explicit version
# fewer temporary tensors
# less tensor algebra
# simpler kernel fusion
# easier for XLA optimizer
@jit
def apply_A_grads(x, precomp, dofs_fix, dofs_all, lam, mu):
    # u -> strain -> stress -> internal force
    # Then scatter-add the element force fe to the global vector.

    result = jnp.zeros_like(x)

    ue = x[dofs_all]           # (n_elem,12)
    ue = ue.reshape(-1,4,3)    # (n_elem,4,3)
    g_all = precomp["grads_all"]
    gx = g_all[...,0]
    gy = g_all[...,1]
    gz = g_all[...,2]

    ux = ue[...,0]
    uy = ue[...,1]
    uz = ue[...,2]

    # strain components
    exx = jnp.sum(gx * ux, axis=1)
    eyy = jnp.sum(gy * uy, axis=1)
    ezz = jnp.sum(gz * uz, axis=1)

    exy = jnp.sum(gy * ux + gx * uy, axis=1)
    eyz = jnp.sum(gz * uy + gy * uz, axis=1)
    ezx = jnp.sum(gz * ux + gx * uz, axis=1)
    trace = exx + eyy + ezz

    # stress
    sxx = lam*trace + 2*mu*exx
    syy = lam*trace + 2*mu*eyy
    szz = lam*trace + 2*mu*ezz
    sxy = mu*exy
    syz = mu*eyz
    szx = mu*ezx

    # internal forces
    v_all = precomp["V_all"]
    fx = v_all[:,None]*(sxx[:,None]*gx + sxy[:,None]*gy + szx[:,None]*gz)
    fy = v_all[:,None]*(sxy[:,None]*gx + syy[:,None]*gy + syz[:,None]*gz)
    fz = v_all[:,None]*(szx[:,None]*gx + syz[:,None]*gy + szz[:,None]*gz)

    fe = jnp.stack([fx, fy, fz], axis=-1)  # (n_elem,4,3)
    fe = fe.reshape(-1,12)
    result = result.at[dofs_all].add(fe)
    result = result.at[dofs_fix].set(0)

    return result

"""
## ---------------tensor version ------------------- ##
@jit
def apply_A_grads(x, precomp, dofs_fix, dofs_all, lam, mu,  I):
    # gather element displacement
    ue = x[dofs_all].reshape(-1,4,3)   # (n_elem,4,3) 

    # displacement gradient 
    du_dx = jnp.einsum("eki,ekj->eij", ue, precomp["grads_all"])

    # symmetric strain
    eps = 0.5 * (du_dx + jnp.swapaxes(du_dx,1,2))

    # trace
    trace = jnp.trace(eps, axis1=1, axis2=2)

    # stress 
    sigma = lam * trace[:,None,None] * I + 2 * mu * eps

    # internal force
    fe = precomp["V_all"][:,None,None] * jnp.einsum("eij,ekj->eki", sigma, precomp["grads_all"])

    fe = fe.reshape(-1,12)

    result = jnp.zeros_like(x)
    result = result.at[dofs_all].add(fe) 
    result = result.at[dofs_fix].set(0)

    return result

"""



@jit
def solver_G(precomp, dofs_all, dofs_fix, lam, mu, F, rtol, maxiter):
    #I = jnp.eye(3, dtype=F.dtype)
    A = lambda x: apply_A_grads(x, precomp, dofs_fix, dofs_all, lam, mu)
    u, info = cg(
         A, 
         F, 
         tol=rtol, 
         maxiter=maxiter
         ) 
    r = F - A(u)
    rel_res = jnp.linalg.norm(r) / jnp.linalg.norm(F)
    return u, rel_res