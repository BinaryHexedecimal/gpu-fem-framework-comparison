# pyright: reportInvalidTypeForm=false


import warp as wp
import numpy as np
from utility.mesh_generation import (
    generate_6tet_tetrahedra, 
    generate_5tet_tetrahedra, 
    generate_grid_3d,
    left_boundary_nodes,
    left_boundary_nodes_center,
    left_boundary_nodes_free,

)

import warp_lib.warp_config as wcfg


#Warp has several predefined vector and matrix types,

# wp.vec3d is a 3-component vector type with double precision.
# (x,y,z)
# d:  double precision (float64)
# wp.vec3. : float 32
# wp.vec3d. : float 64

# vector type:
# wp.vec2, wp.vec3, wp.vec4.  
# Warp vectors are meant for graphics/math primitives, 
# Larger vectors are rare in GPU kernels

# matrix type
# wp.mat33: 3×3 matrix
# wp.mat22, wp.mat33, wp.mat44 (and wp.mat22d, etc.)


# wp.array(dtype=wp.vec3d)
# This defines an array where each element is a vec3d vector.
# Conceptually: N × 3 double matrix
# f.eks.
# points = wp.array(
#     [
#         wp.vec3(0.0, 0.0, 0.0),
#         wp.vec3(1.0, 2.0, 3.0),
#         wp.vec3(4.0, 5.0, 6.0),
#     ],
#     dtype=wp.vec3
# )



# ----------------------------------- #
# ------- utility kernels------------ #
# ----------------------------------- #
# In Warp:
# Kernels compute things.
# Arrays live on device.
# Arithmetic must be inside kernels.
# No implicit broadcasting.
# No Python-level math on device arrays.


# @wp.kernel 
# register this function as a Warp GPU kernel. 
# @ is called a decorator.
# so @wp.kernel tells Warp (NVIDIA) that the function should be compiled 
# into a GPU/parallel kernel instead of running as normal Python code.

@wp.kernel
def vec_dot_kernel(
    a: wp.array(dtype=wcfg.vec3),
    b: wp.array(dtype=wcfg.vec3),
    out: wp.array(dtype=wcfg.scalar)
):
    i = wp.tid() # thread id
    wp.atomic_add(out, 0, wp.dot(a[i], b[i]))


@wp.kernel
def vec_sub_kernel(
    a: wp.array(dtype=wcfg.vec3),
    b: wp.array(dtype=wcfg.vec3),
    out: wp.array(dtype=wcfg.vec3),
):
    i = wp.tid()
    out[i] = a[i] - b[i]

@wp.kernel # copy element
def copy_kernel(
    src: wp.array(dtype=wcfg.vec3), 
    dst: wp.array(dtype=wcfg.vec3), 
):
    i = wp.tid()          
    dst[i] = src[i]      

@wp.kernel
def sqrt_kernel(x: wp.array(dtype=wcfg.scalar),
                out: wp.array(dtype=wcfg.scalar)):
    out[0] = wp.sqrt(x[0])


@wp.kernel
def max_norm_kernel(
    u_wp: wp.array(dtype=wcfg.vec3),
    out_max: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    u = u_wp[i]
    mag = wp.float32(
        wp.sqrt(
        u[0]*u[0] +
        u[1]*u[1] +
        u[2]*u[2]
        )
    )
    wp.atomic_max(out_max, 0, mag)



# update p
# p = r + beta p
@wp.kernel
def update_p_kernel(
    r: wp.array(dtype=wcfg.vec3),
    beta: wcfg.scalar,
    p: wp.array(dtype=wcfg.vec3)
):
    i = wp.tid()
    p[i] = r[i] + beta * p[i]


# Update u
# u=u+alpha p
# or
# update r
# r = r - alpha Ap
@wp.kernel
def axpy_kernel(
    alpha: wcfg.scalar, 
    p: wp.array(dtype=wcfg.vec3),
    x: wp.array(dtype=wcfg.vec3),
): 
    i = wp.tid()
    x[i] = x[i] + alpha*p[i]




@wp.kernel
def mark_fixed_kernel(
    fix_nodes: wp.array(dtype=wp.int32),
    fixed_mask: wp.array(dtype=wp.int32), 
    flag: int
):
    # constraint mask per node
    # 0 = free
    # 1 = fully fixed
    # 2 = x direction is fixed
    i = wp.tid()
    node = fix_nodes[i]
    fixed_mask[node] = flag



@wp.kernel
def enforce_dirichlet_on_F_kernel(
    F: wp.array(dtype=wcfg.vec3),
    fixed_mask: wp.array(dtype=wp.int32)
):
    # flag:
    # 0 = free
    # 1 = fully fixed
    # 2 = x direction is fixed
    i = wp.tid()
    flag = fixed_mask[i]
    if flag == 1:
        F[i][0] = wcfg.scalar(0.0)
        F[i][1] = wcfg.scalar(0.0)
        F[i][2] = wcfg.scalar(0.0)

    elif flag == 2:
        F[i][0] = wcfg.scalar(0.0)


# We set K[i,i] = 1, K[i,j] = 0,
# In theory we should also set K[j,i] = 0 to keep the matrix SPD, which CG requires.
# However, since we enforce p[i] = 0 for fixed DOFs, the term K[j,i] * p[i] = 0.
# Therefore the column contribution vanishes automatically 
# and we do not need to explicitly set K[j,i] = 0 in this matrix-free implementation.

@wp.kernel
def apply_dirichlet_kernel(
    x: wp.array(dtype=wcfg.vec3),
    fixed_mask: wp.array(dtype=wp.int32),
    result: wp.array(dtype=wcfg.vec3)
):
    i = wp.tid()
    flag = fixed_mask[i]
    if flag == 1:
        result[i] = x[i]     # because diagonal in K = 1
    
    elif flag == 2:
        result[i][0] = x[i][0]


# -------------------------------------------------
# Body force
# -------------------------------------------------

@wp.kernel
def build_body_force_kernel(
    # Note: 1 element -> 4 int (idx of node)
    elements: wp.array(dtype=wp.int32),      # (n_elem*4) 
    volumes: wp.array(dtype=wcfg.scalar),     # (n_elem)
    f_vec: wcfg.vec3,    # body force per unit volume
    F: wp.array(dtype=wcfg.vec3)          # (n_nodes)
):
    e = wp.tid()  # element

    # element node indices
    i0 = elements[e*4 + 0]
    i1 = elements[e*4 + 1]
    i2 = elements[e*4 + 2]
    i3 = elements[e*4 + 3]

    # distribute equally to 4 nodes
    fe = (volumes[e] / wcfg.scalar(4.0)) * f_vec  
    # force assembly
    wp.atomic_add(F, i0, fe)
    wp.atomic_add(F, i1, fe)
    wp.atomic_add(F, i2, fe)
    wp.atomic_add(F, i3, fe)



def precalculate_body_F_warp(
        n_nodes, 
        elements, 
        force_vec, 
        volumes,
):
    F = wp.zeros(n_nodes, dtype=wcfg.vec3)
    n_elem = elements.shape[0]//4
    wp.launch(build_body_force_kernel,
            dim=n_elem,
            inputs=[elements, volumes, force_vec, F])

    return F
    



# -------------------------------------------------
# Surface force
# -------------------------------------------------

@wp.kernel
def build_surface_force_kernel(
    nodes: wp.array(dtype=wcfg.vec3),
    # Note: 1 face -> 3 int(i.e.  3 idx of node)
    faces: wp.array(dtype=wp.int32),       # (n_faces*3).     
    f_vec: wcfg.vec3,  # f_vec is force/unit area
    F: wp.array(dtype=wcfg.vec3)    # (n_nodes)
):
    face = wp.tid() # face

    i0 = faces[face*3 + 0] #index
    i1 = faces[face*3 + 1]
    i2 = faces[face*3 + 2]

    x0 = nodes[i0] # wcfg.vec3d
    x1 = nodes[i1]
    x2 = nodes[i2]

    # triangle area
    e1 = x1 - x0
    e2 = x2 - x0
    A = wp.length(wp.cross(e1, e2)) * wcfg.scalar(0.5)

    # distribute traction
    fe = (A / wcfg.scalar(3.0)) * f_vec # force on a node from one face

    wp.atomic_add(F, i0, fe)
    wp.atomic_add(F, i1, fe)
    wp.atomic_add(F, i2, fe)



def precalculate_surface_F_warp(nodes, faces, f_vec):
    
    n_nodes=nodes.shape[0]
    n_faces = faces.shape[0] // 3

    F = wp.zeros(n_nodes, dtype=wcfg.vec3)

    wp.launch(
        build_surface_force_kernel,
        dim=n_faces,
        inputs=[nodes, faces, f_vec, F]
    )

    return F




# -------------------------------------------------
# Dofs_all and dofs_fix
# -------------------------------------------------

# Build element DOF mapping and fixed DOFs for a clamped left face.
# All displacement components (ux, uy, uz) are fixed on selected boundary nodes.
def build_dofs_left_face_clamped(nodes_np, W, H, ny, nz):
    # ------------------------------- #
    # constraint mask per node (a flag per node)
    # 0 = free
    # 1 = fully fixed
    # 2 = x direction is fixed
    # ------------------------------- #
    n_nodes = nodes_np.shape[0]
    lb_nodes_np = left_boundary_nodes(nodes_np)
    lb_nodes = wp.array(lb_nodes_np, dtype=wp.int32)
    fixed_mask = wp.zeros(n_nodes, dtype=wp.int32)   

    wp.launch(
        mark_fixed_kernel, 
        dim=len(lb_nodes_np),
        inputs=[lb_nodes, fixed_mask, 1]
    )
    return fixed_mask 



# Build element DOF mapping and fixed DOFs for nodes on the left boundary.
# Partially constrained left face
def build_dofs_left_face_fix_x(nodes_np, W, H, ny, nz):
    # ------------------------------- #
    # constraint mask per node (a flag per node)
    # 0 = free
    # 1 = fully fixed
    # 2 = x direction is fixed
    # ------------------------------- #
    n_nodes = nodes_np.shape[0]

    lb_center_nodes_np = left_boundary_nodes_center(nodes_np, W, H, ny, nz)
    lb_free_nodes_np = left_boundary_nodes_free(nodes_np, W, H, ny, nz)

    lb_center_nodes = wp.array(lb_center_nodes_np, dtype=wp.int32)
    lb_free_nodes = wp.array(lb_free_nodes_np, dtype=wp.int32)

    fixed_mask = wp.zeros(n_nodes, dtype=wp.int32)

    wp.launch(
        mark_fixed_kernel,
        dim=len(lb_center_nodes_np),
        inputs=[lb_center_nodes, fixed_mask, 1]
    )

    wp.launch(
        mark_fixed_kernel,
        dim=len(lb_free_nodes_np),
        inputs=[lb_free_nodes, fixed_mask, 2]
    )

    return fixed_mask







# -------------------------------------------------
# Mesh
# -------------------------------------------------
def prepare_mesh_warp(nx, ny, nz, L, W, H, tet):

    nodes_np= generate_grid_3d(nx, ny, nz, L, W, H)
    if tet == 6:
        elements_np = generate_6tet_tetrahedra(nx, ny, nz)
    elif tet == 5:
        elements_np = generate_5tet_tetrahedra(nx, ny, nz)
    else:
        raise ValueError("Only 5 tet and 6 tet can be accepted")

    # GPU allocations
    nodes_warp = wp.array(nodes_np, dtype=wcfg.vec3)
    #shape becomes (n_elem * 4,)
    elements_warp = wp.array(elements_np.reshape(-1), dtype=wp.int32)
    #print(f"{tet} tetrehedras per cubic.")
    return elements_warp, nodes_warp, elements_np, nodes_np







# -------------------------------------------------
# cg solver (Matrix-Free, Relative Tol)
# --------------------------------------------------
# Warp does NOT have built-in CG.
# u: displacement vector, the unknown we want to solve for.
# At convergency, Ku=f
# r: residual, how wrong the current solution is. if r = 0, the equation is solved
# r = F - Ku
# p: search direction, the direction in which we update the solution.
# In gradient descent you would move along the gradient.
# In CG you move along conjugate directions.
# p[k+1]​=r[k+1]​+βp[k]
# Ap: matrix-vector product, Ap = K*p,  stiffness response to direction
# Ap is just the result of applying the stiffness operator to the direction

# steps:
#1. compute operator Ap=Kp
#2. compute step length alpha (in which we use Ap, p, r), alpha = xxx
#3. update solution, u=u+alpha*p
#4. update residual, r=r−alpha*(Kp)
#5. update direction p, p=r+beta*p (to get beta, we use r)



def cg_solver(
        A_func,     # function that computes Ap = A(p)
        fixed,
        n_nodes,
        maxiter,
        rtol,
        F
):

    converged_iter = maxiter

    # vectors
    #initial guess
    u  = wp.zeros(n_nodes, dtype=wcfg.vec3)
    r  = wp.zeros(n_nodes, dtype=wcfg.vec3)  # residual 
    p  = wp.zeros(n_nodes, dtype=wcfg.vec3)
    Ap = wp.zeros(n_nodes, dtype=wcfg.vec3)

    rr_old = wp.zeros(1, dtype=wcfg.scalar)
    rr_new = wp.zeros(1, dtype=wcfg.scalar)
    denom  = wp.zeros(1, dtype=wcfg.scalar)
    r0_norm = wp.zeros(1, dtype=wcfg.scalar)
    
    A_func(u, Ap) # Ap = A(u)

    wp.launch(vec_sub_kernel, dim=n_nodes, inputs=[F, Ap, r])

    #Initial Search Direction, CG initialization.
    wp.launch(copy_kernel, dim=n_nodes, inputs=[r, p])

    rr_old.zero_()
    wp.launch(vec_dot_kernel, dim=n_nodes, inputs=[r, r, rr_old])
    wp.launch(sqrt_kernel, dim=1, inputs=[rr_old, r0_norm])
    r0_norm_val = r0_norm.numpy()[0]
    rr_old_val = rr_old.numpy()[0]

    for i in range(maxiter):

        Ap.zero_()
        A_func(p, Ap)

        # overwrite boundary rows
        wp.launch(apply_dirichlet_kernel,
                  dim=n_nodes,
                  inputs=[p, fixed, Ap])

        denom.zero_()
        wp.launch(vec_dot_kernel, dim=n_nodes, inputs=[p, Ap, denom])
        
        # ONE sync here
        denom_val = denom.numpy()[0]
        if abs(denom_val) < 1e-20:
            print("CG breakdown: denom ~ 0")
            break

        alpha = rr_old_val / denom_val

        # u = u + alpha p
        wp.launch(axpy_kernel, dim=n_nodes, inputs=[alpha, p, u])

        # r = r - alpha Ap
        wp.launch(axpy_kernel, dim=n_nodes, inputs=[-alpha, Ap, r])

        #update p, p = r + beta p
        rr_new.zero_()
        wp.launch(vec_dot_kernel, dim=n_nodes, inputs=[r, r, rr_new])

        # ONE sync here
        # Copy rr_new back to Python
        # Python checks convergence
        # If not converged -> launch next GPU kernels
        # So every CG iteration involves a tiny CPU-GPU sync.
        # It is usually not the bottleneck.
        rr_new_val = rr_new.numpy()[0]
        
        
        if np.sqrt(rr_new_val) < rtol * r0_norm_val:
            converged_iter = i
            break

        beta = rr_new_val / rr_old_val
        wp.launch(update_p_kernel, dim=n_nodes, inputs=[r, beta, p])

        rr_old_val = rr_new_val

    return u, converged_iter












@wp.kernel
def build_vertices_kernel(
    nodes_render_wp: wp.array2d(dtype=wp.float32),
    u_wp: wp.array(dtype=wcfg.vec3),
    surfaces_wp: wp.array(dtype=wp.int32),

    anchor_x: wp.float32,
    center_y: wp.float32,
    center_z: wp.float32,
    scale: wp.float32,
    wall_pos: wp.float32,
    coefficient: wp.float32,


    max_val: wp.array(dtype=wp.float32),
    out_vertices: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    node_id = surfaces_wp[tid]
    # position + displacement
    ux = wp.float32(u_wp[node_id][0]) #cast, FP32
    uy = wp.float32(u_wp[node_id][1])
    uz = wp.float32(u_wp[node_id][2])
    
    px = nodes_render_wp[node_id, 0] + ux
    py = nodes_render_wp[node_id, 1] + uy
    pz = nodes_render_wp[node_id, 2] + uz

    # anchor / center
    px = px - anchor_x
    py = py - center_y
    pz = pz - center_z

    # normalize + scale
    px = px / scale * coefficient
    py = py / scale * coefficient
    pz = pz / scale * coefficient

    # screen shift
    px = px + wall_pos

    # write vertex
    base = tid * 6
    out_vertices[base + 0] = px
    out_vertices[base + 1] = py
    out_vertices[base + 2] = pz

    # color
    mag = wp.sqrt(ux * ux + uy * uy + uz * uz)

    val = mag / (max_val[0] + wp.float32(1e-8))
    val = wp.min(val, wp.float32(1.0))

    out_vertices[base + 3] = val
    out_vertices[base + 4] = wp.float32(0.0)
    out_vertices[base + 5] = wp.float32(1.0) - val
