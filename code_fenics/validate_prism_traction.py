# theoretical max uz = 0.82
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile



def validate_prism_traction(L, H, W,  nx, ny, nz, lam, mu, traction, rtol, max_iter):

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

    
    bc_dofs = fem.locate_dofs_geometrical(V, left_boundary)


    bc = fem.dirichletbc(PETSc.ScalarType((0.0, 0.0, 0.0)), bc_dofs, V)


    def right_boundary(x):
        return np.isclose(x[0], L)

    fdim = domain.topology.dim - 1
    right_facets = mesh.locate_entities_boundary(domain, fdim, right_boundary)



    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

    traction = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, traction)))

    mt = mesh.meshtags(domain, fdim, right_facets, np.ones(len(right_facets), dtype=np.int32))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

    Lform = ufl.dot(traction, v) * ds(1)

    problem = LinearProblem(
        a,
        Lform,
        bcs=[bc],
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

    np.save(f"data/fenics_traction_{nx}_{ny}_{nz}_coords.npy", coords)
    np.save(f"data/fenics_traction_{nx}_{ny}_{nz}_u.npy", u_vals)


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
        # find global maximum among ranks
        global_rank_idx = np.argmax(all_max_vals)

        print("Max displacement magnitude:", all_max_vals[global_rank_idx])
        print("Coordinates:", all_max_coords[global_rank_idx])
        print("ux =", all_max_u[global_rank_idx][0])
        print("uy =", all_max_u[global_rank_idx][1])
        print("uz =", all_max_u[global_rank_idx][2])


        
    uh.name = "u"

    with XDMFFile(domain.comm, "data/prism_traction_solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)




traction = 0.1

L = 8.0
H = 1.0
W = 1.0


lam = 100.0
mu = 100.0

rtol = 1e-6
max_iter = 10000


mesh_list = [
    (65, 9, 9),
    (89, 12, 12),
    (121, 16, 16),
    (169, 22, 22),
]




for (nx, ny, nz) in mesh_list:
    validate_prism_traction(L, H, W, nx, ny, nz, lam, mu, 
                        traction, 
                        rtol, max_iter)