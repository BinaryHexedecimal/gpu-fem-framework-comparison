import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem



def gen_groundtruth_prism_compression(L, H, W, nx, ny, nz, lam, mu, 
                               compression, 
                               rtol, 
                               max_iter):

    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]),
        np.array([L, H, W])],
        [nx, ny, nz],
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
        iy = ny // 2
        iz = nz // 2

        y_target = W * iy / ny
        z_target = H * iz / nz

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


    Vx, submap = V.sub(0).collapse()

    dofs_x = fem.locate_dofs_geometrical(Vx, left_boundary)

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


    compression_value = compression

    compression_force = fem.Constant(
        domain,
        PETSc.ScalarType((compression_value, 0.0, 0.0))
    )
    

    right_facets = np.sort(right_facets)

    mt = mesh.meshtags(domain, fdim, right_facets,
                    np.full(len(right_facets), 1, dtype=np.int32))

    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)
    

    Lform = ufl.dot(compression_force, v) * ds(1)

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
    compression_str = f"{compression_value:.2f}".replace(".", "p")
    np.save(
        f"data/{nx}_{ny}_{nz}_compression_{compression_str}_solution.npy",
        uh.x.array
    )

   









def gen_groundtruth_prism_gravity(L, H, W, nx, ny, nz, lam, mu, 
                                gravity,                               
                                rtol, 
                                max_iter):
    
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]),
        np.array([L, H, W])],
        [nx, ny, nz],
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

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    gravity_value = gravity

    f = fem.Constant(
        domain,
        PETSc.ScalarType((0.0, 0.0, gravity_value))
    )

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    Lform = ufl.dot(f, v) * ufl.dx

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
    gravity_str = f"{gravity_value:.2f}".replace(".", "p")
    np.save(
        f"data/{nx}_{ny}_{nz}_gravity_{gravity_str}_solution.npy",
        uh.x.array
    )






def gen_groundtruth_prism_traction(L, H, W, nx, ny, nz, lam, mu, 
                                 traction, 
                                 rtol, max_iter):

    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]),
        np.array([L, H, W])],
        [nx, ny, nz],
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

    traction_value = traction

    traction_force = fem.Constant(
        domain,
        PETSc.ScalarType((0.0, 0.0, traction_value))
    )

    mt = mesh.meshtags(domain, fdim, right_facets, np.ones(len(right_facets), dtype=np.int32))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

    Lform = ufl.dot(traction_force, v) * ds(1)

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
    traction_str = f"{traction_value:.2f}".replace(".", "p")
    np.save(
        f"data/{nx}_{ny}_{nz}_traction_{traction_str}_solution.npy",
        uh.x.array
    )





L = 8.0
H = 1.0
W = 1.0


lam = 100.0
mu = 100.0


rtol = 1e-6
max_iter = 80000

nx = 360
ny = nz = 45


compression = -20.0 # positive value means tension, negative means compression
gravity = 0.05
traction = 0.1



gen_groundtruth_prism_compression(L, H, W, nx, ny, nz, lam, mu, 
                               compression, 
                               rtol, 
                               max_iter)
gen_groundtruth_prism_gravity(L, H, W,  nx, ny, nz, lam, mu, 
                            gravity, 
                            rtol, 
                            max_iter)
gen_groundtruth_prism_traction(L, H, W, nx, ny, nz, lam, mu, 
                                 traction, 
                                 rtol, 
                                 max_iter)