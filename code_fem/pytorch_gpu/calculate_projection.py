import sys
import torch

from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
target_folder = project_root / "data/raw_data/"

from utility.GEOMETRI import MESHES, COMMON_PARAMETERS
from pytorch_lib.pytorch_utils import build_dofs_left_face_fix_x, build_dofs_left_face_clamped
from pytorch_gpu.pytorch_lib.condition_nr import calculate_projection 


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

print("Torch CUDA:", torch.cuda.is_available())
print("Device:", device)


# geometry
L = COMMON_PARAMETERS["L"]
W = COMMON_PARAMETERS["W"]
H = COMMON_PARAMETERS["H"]
tet = COMMON_PARAMETERS["tet"]

lam = COMMON_PARAMETERS["lam"]
mu = COMMON_PARAMETERS["mu"]

nx = MESHES["coarse"]["nx"]
ny = MESHES["coarse"]["ny"]
nz = MESHES["coarse"]["nz"]



# -------------------------------
# Compression
# -------------------------------
force_value_compression = -20
force_vec_compression = torch.tensor([force_value_compression, 0, 0], dtype=dtype, device=device)

# -------------------------------
# Traction
# -------------------------------
force_value_traction = 0.1
force_vec_traction = torch.tensor([0, 0, force_value_traction], dtype=dtype, device=device)

# -------------------------------
# Gravity
# -------------------------------
force_value_gravity = 0.05
force_vec_gravity = torch.tensor([0, 0, force_value_gravity], dtype=dtype, device=device)


model_parameters = {
    "gravity": {"load": "gravity", 
                    "force_type": "body",
                    "theoretical_res": "uz=1.23", 
                    "force_vec" : force_vec_gravity,
                    "precalculate_dofs_func": build_dofs_left_face_clamped,
                    },    

    "traction": {"load": "traction", 
                    "force_type": "surface",
                    "theoretical_res": "uz = 0.82", 
                    "force_vec" : force_vec_traction,
                    "precalculate_dofs_func": build_dofs_left_face_clamped
                    },
    
    "compression": {"load": "compression", 
                    "force_type": "surface",
                    "theoretical_res": "ux = -0.64",
                    "force_vec" : force_vec_compression,
                    "precalculate_dofs_func": build_dofs_left_face_fix_x
                    },
         

}

for model, data in model_parameters.items():
    calculate_projection(nx, ny, nz, L, W, H, tet, 
                lam, mu,
                data["force_type"],
                data["precalculate_dofs_func"],
                data["force_vec"],
                data["load"],
                dtype, device
    )