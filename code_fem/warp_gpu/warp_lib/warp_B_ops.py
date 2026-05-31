# pyright: reportInvalidTypeForm=false
import warp as wp
import warp_lib.warp_config as wcfg
from warp_lib.warp_utils import cg_solver


@wp.kernel
def precompute_BV_kernel(
    nodes: wp.array(dtype=wcfg.vec3),
    elements: wp.array(dtype=wp.int32), #(n_elem*4) 
    B_all: wp.array(dtype=wcfg.scalar), #(n_elem*72) 
    V_all: wp.array(dtype=wcfg.scalar),  #(n_elem) 
):
    e = wp.tid() #elements

    # node indices
    i0 = elements[e*4+0]
    i1 = elements[e*4+1]
    i2 = elements[e*4+2]
    i3 = elements[e*4+3]

    x0 = nodes[i0]
    x1 = nodes[i1]
    x2 = nodes[i2]
    x3 = nodes[i3]

    # Jacobian
    J = wp.matrix_from_cols(x1-x0, x2-x0, x3-x0)
    detJ = wp.determinant(J)

    V = wp.abs(detJ) / wcfg.scalar(6.0)

    invJ = wp.transpose(wp.inverse(J))

    # shape gradients
    g0 = invJ * wcfg.vec3(wcfg.scalar(-1.0),wcfg.scalar(-1.0),wcfg.scalar(-1.0))
    g1 = invJ * wcfg.vec3(wcfg.scalar(1.0),wcfg.scalar(0.0),wcfg.scalar(0.0))
    g2 = invJ * wcfg.vec3(wcfg.scalar(0.0),wcfg.scalar(1.0),wcfg.scalar(0.0))
    g3 = invJ * wcfg.vec3(wcfg.scalar(0.0),wcfg.scalar(0.0),wcfg.scalar(1.0))

    # construct B
    offset = e * 72

    # loop over 4 nodes
    for a in range(4):
        if a == 0:
            g = g0
        elif a == 1:
            g = g1
        elif a == 2:
            g = g2
        else:
            g = g3
        gx = g[0]
        gy = g[1]
        gz = g[2]

        col = a * 3

        # xx
        B_all[offset + 0*12 + col + 0] = gx
        # yy
        B_all[offset + 1*12 + col + 1] = gy
        # zz
        B_all[offset + 2*12 + col + 2] = gz
        # xy
        B_all[offset + 3*12 + col + 0] = gy
        B_all[offset + 3*12 + col + 1] = gx
        # yz
        B_all[offset + 4*12 + col + 1] = gz
        B_all[offset + 4*12 + col + 2] = gy
        # zx
        B_all[offset + 5*12 + col + 0] = gz
        B_all[offset + 5*12 + col + 2] = gx

    V_all[e] = V

# Because Warp does not support
# mat63, vec12, mat66, etc....
# So we must expand the algebra manually.
# implementation: very verbose !!
@wp.kernel
def apply_K_B_kernel(
    elements: wp.array(dtype=wp.int32),
    B_all: wp.array(dtype=wcfg.scalar),
    V_all: wp.array(dtype=wcfg.scalar),
    u: wp.array(dtype=wcfg.vec3),
    f: wp.array(dtype=wcfg.vec3),
    lam: wcfg.scalar,
    mu: wcfg.scalar,
):

    e = wp.tid()
    offset = e * 72
    V = V_all[e]

    # node indices
    i0 = elements[e*4+0]
    i1 = elements[e*4+1]
    i2 = elements[e*4+2]
    i3 = elements[e*4+3]

    # element displacement vector (12)
    ue = wp.zeros(12, dtype=wcfg.scalar)

    u0 = u[i0]; u1 = u[i1]; u2 = u[i2]; u3 = u[i3]

    ue[0] = u0[0]
    ue[1] = u0[1]
    ue[2] = u0[2]

    ue[3] = u1[0]
    ue[4] = u1[1]
    ue[5] = u1[2]

    ue[6] = u2[0]
    ue[7] = u2[1]
    ue[8] = u2[2]

    ue[9]  = u3[0]
    ue[10] = u3[1]
    ue[11] = u3[2]


    # ---------Version 1: naive / (direct global memory access)--------- #
  
    # strain = B @ u_e
    strain = wp.zeros(6, dtype=wcfg.scalar)

    for i in range(6):
        s = wcfg.scalar(0.0)
        for j in range(12):
            s += B_all[offset + i*12 + j] * ue[j]
        strain[i] = s

    exx = strain[0]
    eyy = strain[1]
    ezz = strain[2]
    exy = strain[3]
    eyz = strain[4]
    ezx = strain[5]

    # stress
    trace = exx + eyy + ezz

    sxx = lam*trace + wcfg.scalar(2.0)*mu*exx
    syy = lam*trace + wcfg.scalar(2.0)*mu*eyy
    szz = lam*trace + wcfg.scalar(2.0)*mu*ezz

    sxy = mu*exy
    syz = mu*eyz
    szx = mu*ezx

    # fe = V BT sigma
    fe = wp.zeros(12, dtype=wcfg.scalar)

    for j in range(12):
        s = (
            B_all[offset + 0*12 + j]*sxx +
            B_all[offset + 1*12 + j]*syy +
            B_all[offset + 2*12 + j]*szz +
            B_all[offset + 3*12 + j]*sxy +
            B_all[offset + 4*12 + j]*syz +
            B_all[offset + 5*12 + j]*szx
        )
        fe[j] = V * s

    
    # -----Version 2: Optimized implementation (local caching of element matrix)--------- #

    """
    # Preload B_e
    Be = wp.zeros(72, dtype=wcfg.scalar)

    for k in range(72):
        Be[k] = B_all[offset + k]

    # strain computation
    strain = wp.zeros(6, dtype=wcfg.scalar)
    for i in range(6):
        s = wcfg.scalar(0.0)
        base = i*12
        for j in range(12):
            s += Be[base + j] * ue[j]
        strain[i] = s
    exx = strain[0]
    eyy = strain[1]
    ezz = strain[2]
    exy = strain[3]
    eyz = strain[4]
    ezx = strain[5]

    # stress
    trace = exx + eyy + ezz

    sxx = lam*trace + wcfg.scalar(2.0)*mu*exx
    syy = lam*trace + wcfg.scalar(2.0)*mu*eyy
    szz = lam*trace + wcfg.scalar(2.0)*mu*ezz

    sxy = mu*exy
    syz = mu*eyz
    szx = mu*ezx

    # force
    fe = wp.zeros(12, dtype=wcfg.scalar)
    for j in range(12):
        s = (
            Be[0*12 + j]*sxx +
            Be[1*12 + j]*syy +
            Be[2*12 + j]*szz +
            Be[3*12 + j]*sxy +
            Be[4*12 + j]*syz +
            Be[5*12 + j]*szx
        )
        fe[j] = V * s
    
    
    # ------------------------------------------------------- #
    # -----------------------slut---------------------------- #
    # ------------------------------------------------------- #
    """


    wp.atomic_add(f, i0, wcfg.vec3(fe[0],fe[1],fe[2]))
    wp.atomic_add(f, i1, wcfg.vec3(fe[3],fe[4],fe[5]))
    wp.atomic_add(f, i2, wcfg.vec3(fe[6],fe[7],fe[8]))
    wp.atomic_add(f, i3, wcfg.vec3(fe[9],fe[10],fe[11]))




def precompute_B(nodes, elements):
    n_elem = elements.shape[0]//4
    B_all = wp.zeros(n_elem * 72, dtype=wcfg.scalar)
    V_all = wp.zeros(n_elem, dtype=wcfg.scalar)

    wp.launch(precompute_BV_kernel,
          dim=n_elem,
          inputs=[nodes, elements, B_all, V_all])
    
    return {"B_all":B_all, "V_all": V_all}




def solver_B(
        precomp, fixed, elements, F, 
        n_nodes, lam, mu, rtol, maxiter
):
    # Ap=Ku
    # Ap=f -->F

    def apply_A_grad(p, Ap):

        Ap.zero_()
        wp.launch(apply_K_B_kernel, 
                dim=n_elem, 
                inputs=[elements, precomp["B_all"], precomp["V_all"], p, Ap, lam, mu]
        )

    n_elem = elements.shape[0] // 4

    return cg_solver(
        apply_A_grad,
        fixed,
        n_nodes,
        maxiter,
        rtol,
        F
    )

