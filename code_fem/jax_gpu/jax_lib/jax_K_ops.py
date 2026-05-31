
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.sparse.linalg import cg
import jax





# In this variant, we precompute Ke and V for all elements once, 
# and store them in Ke_all and V_all.
# During CG iterations, we only compute Ke @ u_e using the precomputed Ke


# version d. 22 maj
def element_stiffness_K(p0, p1, p2, p3, lam, mu):
    #maps reference tetrahedron -> physical tetrahedron.
    dtype = p0.dtype
    J = jnp.column_stack((p1 - p0, p2 - p0, p3 - p0)) 
    detJ = jnp.linalg.det(J)

    six = jnp.array(6.0, dtype=dtype)
    V = jnp.abs(detJ) / six
    #V = jnp.abs(detJ) / 6.0

    grad_hat = jnp.array([
        [-1.0, -1.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0]
    ], dtype=dtype) #Fixed constants.

    invJ = jnp.linalg.inv(J)
    grads = grad_hat @ invJ 

    B = jnp.zeros((6, 12), dtype=dtype)

    def fill_B(i, B):
        gx, gy, gz = grads[i]
        col = 3*i

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
    D = jnp.array(
                    [
                        [lam+2*mu, lam, lam, 0, 0, 0],
                        [lam, lam+2*mu, lam, 0, 0, 0],
                        [lam, lam, lam+2*mu, 0, 0, 0],
                        [0, 0, 0, mu, 0, 0],
                        [0, 0, 0, 0, mu, 0],
                        [0, 0, 0, 0, 0, mu],
                    ]
                    , dtype=dtype
                )

    Ke = V * (B.T @ D @ B) # 12x12

    return Ke, V

@jit
def precompute_K(nodes, elements, lam, mu):
    element_nodes = nodes[elements]  # (n_elem,4,3)

    def element_action(elem_nodes):
        p0, p1, p2, p3 = elem_nodes
        return element_stiffness_K(p0, p1, p2, p3, lam, mu)

    K_all, V_all = vmap(element_action)(element_nodes)
    return {"K_all": K_all, "V_all": V_all}





@jit
def apply_A_K(u, dofs_fix, dofs_all, K_all):
    result = jnp.zeros_like(u)
    ue = u[dofs_all]  # (n_elem,12)

    # fe = Ke * ue
    fe_all = jnp.einsum("eij,ej->ei", K_all, ue)
    result = result.at[dofs_all].add(fe_all)
    result = result.at[dofs_fix].set(0)

    return result



@jit
def solver_K(precomp, dofs_all, dofs_fix, lam, mu, F, rtol, maxiter):
    
    A = lambda x: apply_A_K(x, dofs_fix, dofs_all, precomp["K_all"])

    u, info = cg(A, F, tol=rtol, maxiter=maxiter) 
    # ---------------------------------------------------------------------#
    # ------Even though cg solver return tuple u, info --------------------#
    # ------info always is None!!!!!---------------------------------------#
    # ------Therefore, I add a code snippet to check the real convergency--#
    # ---------------------------------------------------------------------#

    r = F - A(u)
    rel_res = jnp.linalg.norm(r) / jnp.linalg.norm(F)

    return u, rel_res



"""
# the version before d. 20 maj
# the explicit version avoids many temporary tensor constructions and complicated tensor algebra.

def element_stiffness_K(p0, p1, p2, p3, lam, mu):
    
    dtype = p0.dtype
    # Jacobian
    J = jnp.column_stack((p1 - p0, p2 - p0, p3 - p0))
    detJ = jnp.linalg.det(J)
    #V = jnp.abs(detJ) / 6.0
    six = jnp.array(6.0, dtype=dtype)
    V = jnp.abs(detJ) / six

    grad_hat = jnp.array([
        [-1., -1., -1.],
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.]
    ], dtype=dtype)

    invJ = jnp.linalg.inv(J)

    # shape gradients
    grads = jnp.einsum("ij,jk->ik", grad_hat, invJ)  # (4,3)

    # build B
    B = jnp.zeros((6,12), dtype=dtype)

    for i in range(4):
        gx, gy, gz = grads[i]
        c = 3*i

        B = B.at[0,c+0].set(gx)
        B = B.at[1,c+1].set(gy)
        B = B.at[2,c+2].set(gz)

        B = B.at[3,c+0].set(gy)
        B = B.at[3,c+1].set(gx)

        B = B.at[4,c+1].set(gz)
        B = B.at[4,c+2].set(gy)

        B = B.at[5,c+0].set(gz)
        B = B.at[5,c+2].set(gx)

    D = jnp.array(
        [
            [lam+2*mu, lam, lam, 0,0,0],
            [lam, lam+2*mu, lam, 0,0,0],
            [lam, lam, lam+2*mu, 0,0,0],
            [0,0,0, mu,0,0],
            [0,0,0, 0,mu,0],
            [0,0,0, 0,0,mu],
        ] , 
        dtype=p0.dtype
    )

    # Ke = V * B^T D B
    K = V * jnp.einsum("ki,kl,lj->ij", B, D, B)

    return K, V

"""
