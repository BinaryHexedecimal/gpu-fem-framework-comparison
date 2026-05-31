import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pathlib import Path
import gc
from .render_static import render_openGL_static


base_dir = Path(__file__).resolve().parent
folder_des = base_dir.parent/ "data/fenics_benchmark/"



def summarize_solution(u_np, nodes_np, elements_np, 
                       nx, ny, nz,
                       load, theoretical_res,
                       framework, version,
                       plot_pyvista,
                       plot_by_openGL,
                    ):

    n_elem = elements_np.shape[0]
    # print("================================================")
    # print("---------------Solution summary-----------------")
    # print("================================================")

    coords_fenics_same_mesh = np.load(folder_des / f"fenics_{load}_{nx}_{ny}_{nz}_coords.npy")
    tree = cKDTree(coords_fenics_same_mesh)
    dist, idx = tree.query(nodes_np)


    # Global relative error compared with FEniCS interpolation
    u_nodewise = u_np.reshape(-1,3)

    u_fenics_nodewise_interp = np.load(folder_des / f"fenics_{load}_{nx}_{ny}_{nz}_u_interp.npy")
    #print(f"Shape of u: {u_nodewise.shape}, Shape of FEniCS interpolated u: {u_fenics_nodewise_interp.shape}")
    assert u_nodewise.shape == u_fenics_nodewise_interp.shape, "Shape of u mismatch!"
    
    u_fenics_nodewise_interp_reordered = u_fenics_nodewise_interp[idx]

    rel_error_global_interp = np.linalg.norm(u_nodewise - u_fenics_nodewise_interp_reordered) / np.linalg.norm(u_fenics_nodewise_interp_reordered)

    #print(f"Global relative displacement error, interpolation: {rel_error_global_interp:.5e}",  file=sys.stderr)

    # -----------------------------------
    # Global relative error compared with FEniCS, the same mesh
    # -----------------------------------
    
    u_fenics_same_mesh = np.load(folder_des / f"fenics_{load}_{nx}_{ny}_{nz}_u.npy")

    u_fenics_reordered = u_fenics_same_mesh[idx]

    rel_error_global_same_mesh = np.linalg.norm(u_nodewise - u_fenics_reordered) / np.linalg.norm(u_fenics_reordered)

    #print(f"Global relative displacement error, same mesh: {rel_error_global_same_mesh:.5e}", file=sys.stderr)
    #print(f"Theoretical max displacement ({load} case): {theoretical_res}")
    #max_abs_norm_node = np.max(np.linalg.norm(u_nodewise, axis=1))
    #print(f"Max displacement magnitude (solver): {max_abs_norm_node:.5f}")
    #print(f"Max displacement magnitude (solver): {np.max(np.linalg.norm(u_nodewise, axis=1)):.5f}", file=sys.stderr)

    # -----------------------------------
    # Plot deformation
    # -----------------------------------
    if plot_pyvista:
        filename = f"{framework}_{load}_{n_elem}_{version}"
        render_pyvista(nodes_np, elements_np, u_np, filename=filename, scale=1.0)
    

    if plot_by_openGL:
        render_openGL_static(nodes_np, elements_np, u_np)
    

    res =  {
        "global_rel_error_interp": float(rel_error_global_interp),
        "global_rel_error_same_mesh": float(rel_error_global_same_mesh)
    }
    return res



def render_pyvista(nodes, elements, u, filename, scale):

    u_vec = u.reshape(-1, 3)
    # --- compute magnitude ---
    u_mag = np.linalg.norm(u_vec, axis=1)

    # Build original grid
    cells = []
    for tet in elements:
        cells.extend([4, *tet])
    cells = np.array(cells)

    celltypes = np.full(len(elements), pv.CellType.TETRA)

    grid_original = pv.UnstructuredGrid(cells, celltypes, nodes)


    # Build deformed grid
    nodes_deformed = nodes + scale * u_vec

    grid_deformed = pv.UnstructuredGrid(cells, celltypes, nodes_deformed)
    grid_deformed["|u|"] = u_mag

    surface_def = grid_deformed.extract_surface()
    surface_org = grid_original.extract_surface()


    # Plot
    plotter = pv.Plotter(off_screen=True)
    #plotter = pv.Plotter()
    # Original outline
    plotter.add_mesh(surface_org,
                     style="wireframe",
                     color="black",
                     line_width=1)

    # Deformed surface (colored)
    plotter.add_mesh(surface_def,
                     scalars="|u|",
                     cmap="viridis",
                     show_edges=False)


    plotter.show(
        screenshot="figures/" + filename + ".png",
        window_size=[1400, 1000],
        auto_close=True
    )
    plotter.close()
    del plotter, grid_original, grid_deformed, surface_org, surface_def
    gc.collect()



def plot_convergency(it_history, relres_history, title, filename):
    plt.figure()
    plt.semilogy(it_history, relres_history)
    plt.xlabel("Iteration")
    plt.ylabel("Relative Residual")
    plt.grid(True)
    plt.title(title)
    plt.savefig("figures/" + filename)
    #plt.show()
    plt.close()
