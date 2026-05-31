# theoretical max ux = 0.64

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile



def validate_prism_compression(L, H, W,  nx, ny, nz, lam, mu, compression, rtol, max_iter):
    # nx, ny, nz: the number of cells (elements) per direction
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]),
        np.array([L, H, W])],
        [nx-1, ny-1, nz-1],
        cell_type=mesh.CellType.tetrahedron,
    )

    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lam * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2 * mu * epsilon(u)


    def left_boundary(x):
        return np.isclose(x[0], 0.0)


    def center_point(x):
        iy = (ny-1) // 2
        iz = (nz-1) // 2

        y_target = W * iy / (ny - 1)
        z_target = H * iz / (nz - 1)

        return np.logical_and.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[1], y_target),
            np.isclose(x[2], z_target)
        ))


    
    # The center point at x=0 is fixed.    
    Vy, map_y = V.sub(1).collapse()
    dofs_y = fem.locate_dofs_geometrical(Vy, center_point)
    parent_y = map_y[dofs_y]
    parent_y = np.array(parent_y, dtype=np.int32)
    bc_y = fem.dirichletbc(PETSc.ScalarType(0.0),
                        parent_y,
                        V.sub(1))

    Vz, map_z = V.sub(2).collapse()
    dofs_z = fem.locate_dofs_geometrical(Vz, center_point)
    parent_z = map_z[dofs_z]
    parent_z = np.array(parent_z, dtype=np.int32)
    bc_z = fem.dirichletbc(PETSc.ScalarType(0.0),
                        parent_z,
                        V.sub(2))

    # The entire face at x=0 is fixed in the x-direction.
    # Collapse subspace
    Vx, submap = V.sub(0).collapse()

    # Locate DOFs in collapsed scalar space
    dofs_x = fem.locate_dofs_geometrical(Vx, left_boundary)

    # Map to parent space
    parent_dofs = submap[dofs_x]

    parent_dofs = np.array(parent_dofs, dtype=np.int32)

    bc = fem.dirichletbc(PETSc.ScalarType(0.0), parent_dofs, V.sub(0))


    def right_boundary(x):
        return np.isclose(x[0], L)

    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)

    right_facets = mesh.locate_entities_boundary(domain, fdim, right_boundary)



    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

    compression = fem.Constant(domain, PETSc.ScalarType((compression, 0.0, 0)))


    right_facets = np.sort(right_facets)

    mt = mesh.meshtags(domain, fdim, right_facets,
                    np.full(len(right_facets), 1, dtype=np.int32))

    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)
    
    # The weak form Neumann condition
    Lform = ufl.dot(compression, v) * ds(1)


    problem = LinearProblem(
        a,
        Lform,
        bcs=[bc, bc_y, bc_z],
        petsc_options_prefix="elasticity",
        petsc_options={
            "ksp_type": "cg",
            "ksp_rtol": rtol,
            #"ksp_atol": 0.0,
            "ksp_max_it": max_iter,
            "pc_type": "none"
        }
    )

    uh = problem.solve()


    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    print("Number of tetrahedra:", num_cells)


    coords = domain.geometry.x
    u_vals = uh.x.array.reshape(-1,3)

    np.save(f"data/fenics_compression_{nx}_{ny}_{nz}_coords.npy", coords)
    np.save(f"data/fenics_compression_{nx}_{ny}_{nz}_u.npy", u_vals)


    mask_ = center_point(coords.T)
    print("Center point(s):")
    print(coords[mask_])


    u_mag = np.linalg.norm(u_vals, axis=1)

    local_max_idx = np.argmax(u_mag)
    local_max_val = u_mag[local_max_idx]
    local_max_coord = coords[local_max_idx]
    local_ux, local_uy, local_uz = u_vals[local_max_idx]


    comm = MPI.COMM_WORLD

    all_max_vals = comm.gather(local_max_val, root=0)
    all_max_coords = comm.gather(local_max_coord, root=0)
    all_max_u = comm.gather((local_ux, local_uy, local_uz), root=0)

    if comm.rank == 0:
        global_rank_idx = np.argmax(all_max_vals)

        print("Max displacement magnitude:", all_max_vals[global_rank_idx])
        print("Coordinates:", all_max_coords[global_rank_idx])
        print("ux =", all_max_u[global_rank_idx][0])
        print("uy =", all_max_u[global_rank_idx][1])
        print("uz =", all_max_u[global_rank_idx][2])



    uh.name = "u"

    with XDMFFile(domain.comm, "data/prism_compression_solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)





L = 8.0
H = 1.0
W = 1.0


lam = 100.0
mu = 100.0

rtol = 1e-6
max_iter = 10000


compression = -20.0 # positive value means tension, negative means compression

mesh_list = [
    (65, 9, 9),
    (89, 12, 12),
    (121, 16, 16),
    (169, 22, 22),
]



for (nx, ny, nz) in mesh_list:
    validate_prism_compression(L, H, W, nx, ny, nz, lam, mu, 
                           compression, 
                           rtol, max_iter)