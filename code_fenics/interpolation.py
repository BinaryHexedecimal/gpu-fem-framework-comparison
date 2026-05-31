
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry




def interpolation_prism(L, H, W, mesh_list, force_dict):

    for force_name, force_value in force_dict.items():
        force_str = f"{force_value:.2f}".replace(".", "p")
        domain = mesh.create_box(
            MPI.COMM_WORLD,
            [
                np.array([0.0, 0.0, 0.0]),
                np.array([L, H, W])
            ],
            [360, 45, 45],
            cell_type=mesh.CellType.tetrahedron,
        )
        V = fem.functionspace(
            domain,
            ("Lagrange", 1, (3,))
        )

        uh = fem.Function(V)

        uh.x.array[:] = np.load(
            f"data/360_45_45_{force_name}_{force_str}_solution.npy",
        )

        for (nx, ny, nz) in mesh_list:
            nodes_np = np.load(f"data/fenics_{force_name}_{nx}_{ny}_{nz}_coords.npy")
            points = nodes_np.astype(np.float64)

            # Build bounding box tree
            bb_tree = geometry.bb_tree(domain, domain.topology.dim)

            # Find candidate cells for each point
            cell_candidates = geometry.compute_collisions_points(bb_tree, points)

            # For each point, find the actual containing cell
            colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

            # Prepare output
            u_interp = np.zeros((points.shape[0], 3))

            missing_count = 0

            for i, point in enumerate(points):
                cells = colliding_cells.links(i)

                if len(cells) == 0:
                    missing_count += 1
                    continue

                cell = cells[0]

                # Evaluate FEM solution at this point
                #u_interp[i] = uh.eval(point, cell)
                u_interp[i] = uh.eval(point, cell).reshape(-1)

            print("missing points:", missing_count)
            print("total points:", len(points))
            print("missing ratio:", missing_count / len(points))

            np.save(f"data/fenics_{force_name}_{nx}_{ny}_{nz}_u_interp.npy", u_interp)





L = 8.0
H = 1.0
W = 1.0



force_dict = {
    "compression": -20.0,
    "traction": 0.1,
    "gravity": 0.05,
}      

mesh_list = [
    (65, 9, 9),
    (89, 12, 12),
    (121, 16, 16),
    (169, 22, 22),
]

interpolation_prism(L, H, W, mesh_list, force_dict)