import numpy as np


# Generate structured 3D grid
def generate_grid_3d(nx, ny, nz, L, W, H):
    xs = np.linspace(0, L, nx) # we want nx-1 elements, so we need nx nodes
    ys = np.linspace(0, W, ny)
    zs = np.linspace(0, H, nz)
    nodes = []
    for z in zs:
        for y in ys:
            for x in xs:
                nodes.append([x, y, z])
    #print("Generated", len(nodes), "nodes.")
    return np.array(nodes)


def node_id(i, j, k, nx, ny):
    return k * (nx * ny) + j * nx + i


def generate_5tet_tetrahedra(nx, ny, nz):
    elements = []

    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):

                n0 = node_id(i, j, k, nx, ny)
                n1 = node_id(i+1, j, k, nx, ny)
                n2 = node_id(i+1, j+1, k, nx, ny)
                n3 = node_id(i, j+1, k, nx, ny)
                n4 = node_id(i, j, k+1, nx, ny)
                n5 = node_id(i+1, j, k+1, nx, ny)
                n6 = node_id(i+1, j+1, k+1, nx, ny)
                n7 = node_id(i, j+1, k+1, nx, ny)

                # Split cube into 5 tetrahedra
                elements += [
                    [n0, n1, n3, n4],
                    [n1, n2, n3, n6],
                    [n1, n3, n4, n6],
                    [n1, n5, n4, n6],
                    [n3, n4, n6, n7],
                ]
    #print("Generated", len(elements), "elements.")
    return np.array(elements)


def generate_6tet_tetrahedra(nx, ny, nz):
    elements = []

    for k in range(nz -1):
        for j in range(ny -1):
            for i in range(nx -1):

                n0 = node_id(i,   j,   k, nx, ny)
                n1 = node_id(i+1, j,   k, nx, ny)
                n2 = node_id(i+1, j+1, k, nx, ny)
                n3 = node_id(i,   j+1, k, nx, ny)
                n4 = node_id(i,   j,   k+1, nx, ny)
                n5 = node_id(i+1, j,   k+1, nx, ny)
                n6 = node_id(i+1, j+1, k+1, nx, ny)
                n7 = node_id(i,   j+1, k+1, nx, ny)

                # 6-tet symmetric split using body diagonal n0 -> n6
                elements += [
                    [n0, n1, n2, n6],
                    [n0, n2, n3, n6],
                    [n0, n3, n7, n6],
                    [n0, n7, n4, n6],
                    [n0, n4, n5, n6],
                    [n0, n5, n1, n6],
                ]
    #print("Generated", len(elements), "elements.")
    return np.array(elements)



def left_boundary_nodes(nodes):
    return np.where(nodes[:, 0] == 0)[0]


def right_boundary_nodes(nodes, L):
    return np.where(np.isclose(nodes[:, 0], L))[0]


def left_boundary_nodes_free(nodes, W, H, ny, nz):

    y_mid = W * np.floor((ny - 1) / 2) / (ny - 1)
    z_mid = H * np.floor((nz - 1) / 2) / (nz - 1)
    x_mask = np.isclose(nodes[:, 0], 0.0)
    center_mask = (
        np.isclose(nodes[:, 1], y_mid) &
        np.isclose(nodes[:, 2], z_mid)
    )
    mask = x_mask & (~center_mask)
    return np.where(mask)[0]




def left_boundary_nodes_center(nodes, W, H, ny, nz):

    y_mid = W * np.floor((ny - 1) / 2) / (ny - 1)
    z_mid = H * np.floor((nz - 1) / 2) / (nz - 1)

    mask = (
        np.isclose(nodes[:, 0], 0.0) &
        np.isclose(nodes[:, 1], y_mid) &
        np.isclose(nodes[:, 2], z_mid)
    )

    return np.where(mask)[0]



def detect_boundary_faces_from_nodes(elements_np, n_nodes, boundary_nodes):

    boundary_nodes = np.asarray(boundary_nodes, dtype=np.int32) #list to np

    # mask: node -> is boundary
    boundary_mask = np.zeros(n_nodes, dtype=bool)
    boundary_mask[boundary_nodes] = True

    face_pattern = np.array([
        [0,1,2],
        [0,1,3],
        [0,2,3],
        [1,2,3]
    ])

    # collect all faces
    faces_np = elements_np[:, face_pattern]   # (n_elem, 4, 3) # one elem has 4 nodes, each node has one row [node_i, node_j, node_k]
    faces_np = faces_np.reshape(-1,3) # (n_elem * 4, 3)

    # check boundary condition
    mask = boundary_mask[faces_np].all(axis=1)
    
    faces_np = faces_np[mask]
    faces_np = np.unique(np.sort(faces_np, axis=1), axis=0) #remove duplicates
    return faces_np




def detect_right_boundary_faces_from_nodes(elements_np, nodes_np, L):
    boundary_nodes=right_boundary_nodes(nodes_np, L) 
    n_nodes = nodes_np.shape[0]
    right_faces_np = detect_boundary_faces_from_nodes(elements_np, n_nodes, boundary_nodes)
    return right_faces_np  #(n_faces, 3)