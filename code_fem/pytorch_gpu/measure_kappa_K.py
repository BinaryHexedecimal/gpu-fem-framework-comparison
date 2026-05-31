import sys


from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
target_folder = project_root / "data/raw_data/"

from utility.GEOMETRI import MESHES, COMMON_PARAMETERS
import torch
from functools import partial

from pytorch_lib.pytorch_utils import build_dofs_left_face_fix_x, build_dofs_left_face_clamped
from pytorch_lib.condition_nr import build_reduced_stiffness_matrix, analyze_spectrum

import json

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

print("Torch CUDA:", torch.cuda.is_available())
print("Device:", device)



L = COMMON_PARAMETERS["L"]
W = COMMON_PARAMETERS["W"]
H = COMMON_PARAMETERS["H"]
tet = COMMON_PARAMETERS["tet"]

lam = COMMON_PARAMETERS["lam"]
mu = COMMON_PARAMETERS["mu"]




print("-----------------------------------------------------------")
print("---------------------calculate kappa-----------------------")
print("-----------------------------------------------------------")

results =  {}
for mesh, geo in  MESHES.items():
    nx = geo["nx"]         
    ny = geo["ny"]
    nz = geo["nz"]

    build_dofs_left_face_fix_x_prebind = partial(build_dofs_left_face_fix_x, W=W, H=H, ny=ny, nz=nz)

    build_dofs_left_face_clamped_prebind = partial(build_dofs_left_face_clamped, W=W, H=H, ny=ny, nz=nz)
    
    model_parameters = {
        "compression": {"load": "compression", 
                        "force_type": "surface",
                        "precalculate_dofs_func": build_dofs_left_face_fix_x_prebind
                        },
        "traction": {"load": "traction", 
                        "force_type": "surface",
                        "precalculate_dofs_func": build_dofs_left_face_clamped_prebind
                        },
        "gravity": {"load": "gravity", 
                        "force_type": "body",
                        "precalculate_dofs_func": build_dofs_left_face_clamped_prebind
                        },             

    }
    results[mesh] = {}

    for model, data in model_parameters.items():
        K_ff = build_reduced_stiffness_matrix(
            nx, ny, nz, L, W, H, tet,
            lam, mu,
            data["load"],
            data["precalculate_dofs_func"],   
            dtype, 
            device,
        )
        spec = analyze_spectrum(K_ff)

        results[mesh][model] = spec

# save json
file_name = f"kappa_results.json"
file_path = target_folder / file_name
with open(file_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Saved to {file_name}")