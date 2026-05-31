# pyright: reportInvalidTypeForm=false
import warp as wp

import warp_lib.warp_config as wcfg

from warp_lib.warp_utils import cg_solver




# ----------------------------------- #
# ---------- main kernels------------ #
# ----------------------------------- #
# In this variant, Warp never constructs Ke
# Instead it stores: gradients of shape functions
# grads.shape = (n_elem * 4, wcfg.vec3)


@wp.kernel
def precompute_GV_kernel(
        nodes: wp.array(dtype=wcfg.vec3),
        elements: wp.array(dtype=wp.int32), #(n_elem*4) 
        grads_all: wp.array(dtype=wcfg.vec3), #(n_elem*4, ) , each node has a grad, which is wcfg.vec3d
        V_all: wp.array(dtype=wcfg.scalar),
    ):
    e = wp.tid() #elements

    i0 = elements[e*4 + 0]
    i1 = elements[e*4 + 1]
    i2 = elements[e*4 + 2]
    i3 = elements[e*4 + 3]

    x0 = nodes[i0]
    x1 = nodes[i1]
    x2 = nodes[i2]
    x3 = nodes[i3]

    # Jacobian
    J = wp.matrix_from_cols(x1 - x0, x2 - x0, x3 - x0)
    detJ = wp.determinant(J)
    V = wp.abs(detJ) / wcfg.scalar(6.0)
    invJ = wp.transpose(wp.inverse(J))

    # Reference gradients
    # g0 : 3 vector     (wp.wcfg.vec3d)
    g0 = wcfg.vec3(
        wcfg.scalar(-1.0),
        wcfg.scalar(-1.0),
        wcfg.scalar(-1.0)
    )

    g1 = wcfg.vec3(
        wcfg.scalar(1.0),
        wcfg.scalar(0.0),
        wcfg.scalar(0.0)
    )

    g2 = wcfg.vec3(
        wcfg.scalar(0.0),
        wcfg.scalar(1.0),
        wcfg.scalar(0.0)
    )

    g3 = wcfg.vec3(
        wcfg.scalar(0.0),
        wcfg.scalar(0.0),
        wcfg.scalar(1.0)
    )

    # Transform to physical space
    #grad_i = (gx_i, gy_i, gz_i)
    grads_all[e*4 + 0] = invJ * g0
    grads_all[e*4 + 1] = invJ * g1
    grads_all[e*4 + 2] = invJ * g2
    grads_all[e*4 + 3] = invJ * g3

    V_all[e] = V



@wp.kernel 
def apply_K_grad_kernel(
    elements: wp.array(dtype=wp.int32), #(n_elem*4 )
    grads_all: wp.array(dtype=wcfg.vec3), #(n_elem*4, )
    V_all: wp.array(dtype=wcfg.scalar), #(n_elem )
    I : wcfg.mat33,
    u: wp.array(dtype=wcfg.vec3), #(n_nodes, )
    f: wp.array(dtype=wcfg.vec3), #(n_nodes, )
    lam: wcfg.scalar,
    mu: wcfg.scalar,
):

    e = wp.tid() # elements

    i0 = elements[e*4 + 0] # idx of node
    i1 = elements[e*4 + 1]
    i2 = elements[e*4 + 2]
    i3 = elements[e*4 + 3]

    g0 = grads_all[e*4 + 0]
    g1 = grads_all[e*4 + 1]
    g2 = grads_all[e*4 + 2]
    g3 = grads_all[e*4 + 3]

    V = V_all[e]

    u0 = u[i0]
    u1 = u[i1]
    u2 = u[i2]
    u3 = u[i3]

    # displacement gradient
    # wp.outer(u0, g0)  -> outer product
    du_dx = (
        wp.outer(u0, g0) +
        wp.outer(u1, g1) +
        wp.outer(u2, g2) +
        wp.outer(u3, g3)
    ) 

    eps = wcfg.scalar(0.5) * (du_dx + wp.transpose(du_dx))
    # stress:  
    tr = eps[0,0] + eps[1,1] + eps[2,2]
    sigma = lam * tr * I + wcfg.scalar(2.0) * mu * eps # sigma : 3×3 matrix   (wp.wcfg.mat33d)
    

    f0 = V * (sigma @ g0) #In Python, @ is the matrix multiplication operator.
    f1 = V * (sigma @ g1)
    f2 = V * (sigma @ g2)
    f3 = V * (sigma @ g3)

    wp.atomic_add(f, i0, f0)
    wp.atomic_add(f, i1, f1)
    wp.atomic_add(f, i2, f2)
    wp.atomic_add(f, i3, f3)




def precompute_G(nodes, elements):
    n_elem = elements.shape[0] // 4
    V_all = wp.zeros(n_elem, dtype=wcfg.scalar)
    grads_all = wp.zeros(n_elem*4, dtype=wcfg.vec3)

    wp.launch(precompute_GV_kernel,
          dim=n_elem,
          inputs=[nodes, elements, grads_all, V_all])
    return {"grads_all":grads_all, "V_all": V_all}




def solver_G(
        precomp, fixed, elements, F, 
        n_nodes, lam, mu, 
        rtol, maxiter
):
    n_elem = elements.shape[0] // 4
    I = wcfg.mat33( wcfg.scalar(1.0), wcfg.scalar(0.0), wcfg.scalar(0.0), 
                  wcfg.scalar(0.0), wcfg.scalar(1.0), wcfg.scalar(0.0), 
                  wcfg.scalar(0.0), wcfg.scalar(0.0), wcfg.scalar(1.0) )

    def apply_A_grad(p, Ap):
        Ap.zero_()
        wp.launch(
            apply_K_grad_kernel,
            dim=n_elem,
            inputs=[elements, precomp["grads_all"], precomp["V_all"], I, p, Ap, lam, mu]
        )

    return cg_solver(
        apply_A_grad,
        fixed,
        n_nodes,
        maxiter,
        rtol,
        F
    )








