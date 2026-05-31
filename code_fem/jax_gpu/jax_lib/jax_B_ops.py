import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.sparse.linalg import cg


# -----------------------------------------------------------------------------
# In this version, we precompute per-element geometric data:
# 1. B matrices (strain-displacement operators)
# 2. V (element volumes)
# These quantities depend only on the mesh geometry and material parameters,
# and therefore remain constant throughout the CG iterations.

def element_stiffness_BV(p0, p1, p2, p3):
    dtype = p0.dtype
    #maps reference tetrahedron -> physical tetrahedron.
    J = jnp.column_stack((p1 - p0, p2 - p0, p3 - p0)) 
    detJ = jnp.linalg.det(J)
    six = jnp.array(6.0, dtype=dtype)
    V = jnp.abs(detJ) / six

    grad_hat = jnp.array([
        [-1.0, -1.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0]
        ], dtype=dtype)  #Fixed constants.

    invJ = jnp.linalg.inv(J)
    grads = grad_hat @ invJ 

    B = jnp.zeros((6, 12), dtype=dtype)

    def fill_B(i, B):
        gx, gy, gz = grads[i]
        col = 3 * i

        B = B.at[0, col+0].set(gx)
        B = B.at[1, col+1].set(gy)
        B = B.at[2, col+2].set(gz)

        B = B.at[3, col+0].set(gy)
        B = B.at[3, col+1].set(gx)

        B = B.at[4, col+1].set(gz)
        B = B.at[4, col+2].set(gy)

        B = B.at[5, col+0].set(gz)
        B = B.at[5, col+2].set(gx)

        return B 

    B = jax.lax.fori_loop(0, 4, fill_B, B)
    return B, V


@jit
def precompute_B(nodes, elements, lam, mu):
    dtype = nodes.dtype

    element_nodes = nodes[elements]   # (n_elem, 4, 3)

    def element_action(elem_nodes):
        p0, p1, p2, p3 = elem_nodes
        return element_stiffness_BV(p0, p1, p2, p3)

    # Each element uses small dense linear algebra
    # vmap parallelizes across elements
    B_all, V_all = vmap(element_action)(element_nodes)

    z = jnp.array(0.0, dtype=dtype)

    D = jnp.array([
        [lam+2*mu, lam, lam, z, z, z],
        [lam, lam+2*mu, lam, z, z, z],
        [lam, lam, lam+2*mu, z, z, z],
        [z, z, z, mu, z, z],
        [z, z, z, z, mu, z],
        [z, z, z, z, z, mu],
        ], dtype=dtype
    )
    return {"B_all": B_all, "V_all": V_all, "D": D}



# During each CG iteration, we avoid storing the full element stiffness matrix Ke.
# Instead, we compute fe = V B^T D B ue  on-the-fly 
# This trades additional arithmetic for reduced memory traffic,
# which may improve performance on GPUs where memory bandwidth is often the bottleneck.

# Here, B is strain-displacement matrix
# so then in solver:
# strain = B u
# stress = D strain
# force  = V B^T stress

#version from d. 23 maj
@jit
def apply_A_B(u, dofs_fix, dofs_all, B_all, V_all, D):

    #result is f = Ku, accumulated from all elements.
    result = jnp.zeros_like(u)

    # gather local displacements
    ue_all = u[dofs_all]  # (n_elem, 12)

    # Step 1: compute strain = B u_e
    strain = jnp.matmul(B_all, ue_all[..., None])  # (n_elem, 6, 1)

    # Step 2: stress = D strain
    stress = jnp.matmul(D, strain)  # (n_elem, 6, 1)

    # Step 3: internal force = V B^T stress
    fe_all = V_all[:, None, None] * jnp.matmul(
        jnp.transpose(B_all, (0, 2, 1)),
        stress
    )

    fe_all = fe_all.squeeze(-1)  # (n_elem, 12)
    result = result.at[dofs_all].add(fe_all)
    # apply Dirichlet BC
    result = result.at[dofs_fix].set(0)

    return result


"""
# version before 22 maj
@jit
def apply_A_B(x, dofs_fix, dofs_all, B_all, V_all, D):

    # gather local displacements
    ue_all = x[dofs_all]  # (n_elem,12)

    # strain = B * u
    strain = jnp.einsum("eij,ej->ei", B_all, ue_all)

    # stress = D * strain
    stress = jnp.einsum("ij,ej->ei", D, strain)

    # internal force = V * B^T * stress
    fe_all = jnp.einsum("eji,ej->ei", B_all, stress)

    fe_all = fe_all * V_all[:, None]

    # assemble
    result = jnp.zeros_like(x)
    result = result.at[dofs_all].add(fe_all)

    # apply Dirichlet BC
    result = result.at[dofs_fix].set(0)

    return result
"""


@jit
def solver_B(precomp, dofs_all, dofs_fix, lam, mu, F, rtol, maxiter):
    A = lambda x: apply_A_B(x, dofs_fix, dofs_all, precomp["B_all"], precomp["V_all"], precomp["D"])

    #jax.debug.print("F:", F.dtype)
    
    u, info = cg(A, F, tol=rtol, maxiter=maxiter) 
    #jax.debug.print("u:", u.dtype)
    r = F - A(u)
    #jax.debug.print("CG solver info: {}", info)
    #print("r, shape:", r.shape)
    #print("info: {}", info)

    rel_res = jnp.linalg.norm(r) / jnp.linalg.norm(F)

    return u, rel_res


