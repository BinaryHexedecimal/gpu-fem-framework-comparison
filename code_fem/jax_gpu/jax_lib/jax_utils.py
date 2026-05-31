# jax, Implicit float32

from utility.mesh_generation import (
    generate_6tet_tetrahedra, 
    generate_5tet_tetrahedra,
    generate_grid_3d, 
    left_boundary_nodes_center,
    left_boundary_nodes_free,
    left_boundary_nodes,
)

import jax.numpy as jnp
from jax import jit #, vmap

from OpenGL.GL import *






# -------------------------------------------------
# Dofs_all and dofs_fix
# -------------------------------------------------

# dofs are indices of displacement variables u in the global vector.
# each node has 3 unknowns: u_x, u_y, u_z
# u = [u0x, u0y, u0z,    u1x, u1y, u1z,    u2x, u2y, u2z,    ...]
# each entry in this vector has an index 
# that index is the degree of freedom.
# Therefore, dofs is one-dimensional, with size = 3 * n_nodes
# Mapping rule: dof(node i, component k) = 3*i + k
# k = 0 for x
# k = 1 for y
# k = 2 for z


# fx

# (1) dofs_fix
# lb_nodes = [2,5]
# node 2 -> DOFs [6,7,8]
# node 5 -> DOFs [15,16,17]
# dofs_fix = [6,7,8,15,16,17],  fixed displacement components.

# (2) dofs_all
# elements[e] = [5,8,2,9]
# node 5 -> [15,16,17]
# node 8 -> [24,25,26]
# node 2 -> [6,7,8]
# node 9 -> [27,28,29]
# dofs_all[e] =
#     [15,16,17,
#     24,25,26,
#     6,7,8,
#     27,28,29]
# dofs_all has shape (n_elem, 12) , where 12 = 4 nodes × 3 DOF 



# Build element DOF mapping and fixed DOFs for a clamped left face.
# All displacement components (ux, uy, uz) are fixed on selected boundary nodes.
def build_dofs_left_face_clamped(nodes, elements, W, H, ny, nz):
    lb_nodes = jnp.array(left_boundary_nodes(nodes),  dtype=jnp.int32)
    dofs_fix = (3 * lb_nodes[:, None] + jnp.arange(3)).reshape(-1)
    dofs_all = (3 * elements[..., None] + jnp.arange(3)).reshape(elements.shape[0], 12)
    return dofs_all, dofs_fix 


# Build element DOF mapping and fixed DOFs for nodes on the left boundary.
# Partially constrained left face
def build_dofs_left_face_fix_x(nodes, elements, W, H, ny, nz):
    lb_center_nodes = jnp.array(left_boundary_nodes_center(nodes, W, H, ny, nz),  dtype=jnp.int32)
    lb_free_nodes = jnp.array(left_boundary_nodes_free(nodes, W, H, ny, nz),  dtype=jnp.int32)
    
    lb_center_dofs = (3 * lb_center_nodes[:, None] + jnp.arange(3)).reshape(-1)
    lb_free_dofs = (3 * lb_free_nodes[:, None]).reshape(-1)
    dofs_fix = jnp.concatenate([lb_center_dofs, lb_free_dofs])
    
    dofs_all = (3 * elements[..., None] + jnp.arange(3)).reshape(elements.shape[0], 12)
    return dofs_all, dofs_fix 









# -------------------------------------------------
# Mesh
# -------------------------------------------------

def prepare_mesh_jax(nx, ny, nz, W, L, H, tet, dtype):
    nodes_np = generate_grid_3d(nx, ny, nz, W, L, H)
    nodes_jax = jnp.array(nodes_np, dtype=dtype)
    if tet == 6:       
        elements_np = generate_6tet_tetrahedra(nx, ny, nz)
    elif tet == 5:
        elements_np = generate_5tet_tetrahedra(nx, ny, nz)
    else:
        raise ValueError("Only 5 tet and 6 tet can be accepted")
    #print(f"{tet} tetrehedras per cubic.")
    elements_jax = jnp.array(elements_np)
    return elements_jax, nodes_jax, elements_np, nodes_np


# -------------------------------------------------
# Surface force
# -------------------------------------------------
@jit
def build_surface_force(F, nodes, faces, force_vec):

    # triangle vertices
    p0 = nodes[faces[:, 0]]
    p1 = nodes[faces[:, 1]]
    p2 = nodes[faces[:, 2]]

    # triangle area
    cross_prod = jnp.cross(p1 - p0, p2 - p0)
    half = cross_prod.dtype.type(0.5)
    area = half * jnp.linalg.norm(cross_prod, axis=1)

    # distribute force equally to nodes
    nodal_force = force_vec * (area[:, None] / 3.0)

    for i in range(3):
        node_ids = faces[:, i]
        F = F.at[3*node_ids + 0].add(nodal_force[:, 0])
        F = F.at[3*node_ids + 1].add(nodal_force[:, 1])
        F = F.at[3*node_ids + 2].add(nodal_force[:, 2])

    return F

@jit
def precalculate_surface_F_jax(nodes, faces, force_vec):

    n_nodes = nodes.shape[0]
    
    F = jnp.zeros(3 * n_nodes, dtype=force_vec.dtype)

    F = build_surface_force(F, nodes, faces, force_vec)

    return F


# -------------------------------------------------
# Body force
# -------------------------------------------------

@jit
def build_body_force(F, force_vec, dofs_all, V_all):

    # expand V to (n_elem, 4, 1)
    scale = (V_all / V_all.dtype.type(4.0))[:, None, None]

    # repeat 4 times along node axis
    scale = jnp.repeat(scale, 4, axis=1)

    Fe_all = scale * force_vec   # (n_elem, 4, 3)
    Fe_all = Fe_all.reshape(-1, 12)
    F = F.at[dofs_all].add(Fe_all)

    return F


@jit
def precalculate_body_F_jax(nodes, dofs_all, force_vec, V_all):
    n_nodes = nodes.shape[0]

    F = jnp.zeros(3 * n_nodes, dtype=force_vec.dtype)
    F = build_body_force(F, force_vec, dofs_all, V_all)
    return F
