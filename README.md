# GPU FEM Framework Comparison

This repository contains the code developed for the master's thesis:

**"Performance Comparison of GPU-Based FEM Frameworks: PyTorch, JAX, and NVIDIA Warp"**

The code implements finite element solvers for the three-dimensional linear elasticity equation using PyTorch, JAX, and NVIDIA Warp, and compares their performance, memory usage, initialization overhead, and scalability.

All three implementations are integrated with OpenGL for real-time visualization through host-mediated data transfer. The Warp implementation additionally supports fully GPU-resident visualization pipelines using both GPU-to-GPU copy and zero-copy interoperability.

## Repository Structure

gpu-fem-framework-comparison/
├── code_fem/                          # GPU FEM implementations
│   ├── data/                          # Benchmark results and processed data
│   │   ├── analysis_res/              # Analysis outputs (generated figures and tables)
│   │   ├── fenics_benchmark/          # FEniCS reference benchmark data
│   │   ├── raw_data/                  # Raw benchmark measurements
│   │   └── *.py                       # Data processing and analysis scripts
│   │
│   ├── pytorch_gpu/                   # PyTorch implementation
│   │   ├── pytorch_lib/               # PyTorch-specific solver, rendering, and utility modules
│   │   ├── figures/                   # Generated figures
│   │   └── *.py                       # Benchmark and visualization scripts
│   │
│   ├── warp_gpu/                      # NVIDIA Warp implementation
│   │   ├── warp_lib/                  # Warp-specific solver, rendering, and utility modules
│   │   ├── figures/                   # Generated figures
│   │   └── *.py                       # Benchmark and visualization scripts
│   │
│   ├── jax_gpu/                       # JAX implementation
│   │   ├── jax_lib/                   # JAX-specific solver, rendering, and utility modules
│   │   ├── figures/                   # Generated figures
│   │   └── *.py                       # Benchmark and visualization scripts
│   │
│   ├── utility/                       # Framework-independent utilities
│   │    ├── *.cpp                     # CUDA/OpenGL interoperability code
│   │    └── *.py                      # Shared utilities and helper modules
│   │
│   └── requirements_gpu.txt
│
├── code_fenics/                       # FEniCS reference implementation
│   ├── data/                          # FEniCS benchmark results
│   └── *.py                           # Benchmark scripts
│
└── README.md                          # Project documentation


## Installation

The repository contains two independent implementations:

- **code_fem/**: GPU-based FEM solvers implemented using PyTorch, JAX, and NVIDIA Warp.
- **code_fenics/**: Reference implementation based on FEniCSx.

These components require different software environments and should be installed separately.

### GPU FEM Environment (`code_fem`)

#### Requirements

- Python 3.12+
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x
- OpenGL 4.6 compatible driver

Create and activate a virtual environment:

```bash
cd code_fem

python -m venv fem-env
source fem-env/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements_gpu.txt
```

### Building the CUDA–OpenGL Interoperability Module

The GPU-resident visualization pipeline relies on a custom C++ extension module that provides CUDA–OpenGL interoperability through pybind11.

Before building the module, ensure that:

- CUDA Toolkit 12.x is installed
- OpenGL development libraries are available
- pybind11 is installed

Install pybind11 if it is not already installed:

```bash
pip install pybind11
```

Build the extension from the `utility/` directory:

```bash
cd code_fem/utility

python setup_ubuntu.py build_ext --inplace
```

This generates the Python extension module:

```text
cuda_gl_interop*.so
```

which can then be imported directly from Python:

```python
import cuda_gl_interop
```

**Notes**

1. The provided build script assumes a standard Linux CUDA installation located at:

```text
/usr/local/cuda
```

If CUDA is installed elsewhere, update the `cuda_path` variable in `setup_ubuntu.py` accordingly.

2. The CUDA–OpenGL interoperability module is only required for the GPU-resident visualization pipeline and currently supports Linux systems with CUDA installed. Without this module, all three GPU FEM implementations can still be executed and visualized using the standard CPU-mediated data transfer pipeline.


### FEniCS Environment (code_fenics)

The FEniCS reference implementation is executed inside a Docker container.

Pull the official FEniCSx image:

```bash
docker pull dolfinx/dolfinx:stable
```

Start the container:

```bash
docker run -ti \
    -v $(pwd):/root/shared \
    dolfinx/dolfinx:stable
```

The FEniCS scripts can then be executed from within the container.

## Usage

### Activate the Environment

```bash
cd code_fem
source fem-env/bin/activate
```

### Running Individual Benchmarks

Each framework has its own implementation directory:

```text
pytorch_gpu/
jax_gpu/
warp_gpu/
```

Navigate to the desired framework directory:

```bash
cd pytorch_gpu
```

(or `jax_gpu` or `warp_gpu`).

---

### Numerical Validation

Run:

```bash
python run_one_case_numerical.py
```

Available options:

```text
--mesh                Mesh resolution:
                      coarse, m1, m2, fine
                      (default: coarse)
--model               Loading :
                      gravity, traction, compression
                      (default: gravity)
--version             Solver formulation:
                      K, B, G
                      (Warp supports only B and G)
                      (default: G)
--dtype_name          Floating-point precision:
                      fp32, fp64
                      (default: fp64)
--numerical_runtimes  Number of repeated solution measurement
                      (default: 1)
```

Example:

```bash
python run_one_case_numerical.py \
    --mesh fine \
    --dtype_name fp32 \
    --numerical_runtimes 10
```

---

### Real-Time Visualization (CPU-Mediated Data Transfer)

Run:

```bash
python run_one_case_realtime_cpu.py
```

This launches an interactive visualization window. During execution, users can modify the applied force through keyboard input. Instructions and feedback are displayed in the window title bar.

Available options:

```text
--mesh                 Mesh resolution:
                       coarse, m1, m2, fine
                       (default: coarse)
--model                Loading scenario:
                       gravity, traction, compression
                       (default: gravity)
--version              Solver formulation:
                       K, B, G
                       (Warp supports only B and G)
                       (default: G)
--dtype_name           Floating-point precision:
                       fp32, fp64
                       (default: fp64)
--real_runtimes        Number of repeated visualization runtime measurements  
                       (default: 1)
--precompute_runtimes  Number of repeated precomputation runtime measurements  
                       (default: 1)
--auto_force_input     Automatically apply predefined force inputs,
                       eliminating the need for manual user interaction
```

Example:

```bash
python run_one_case_realtime_cpu.py \
    --mesh fine \
    --dtype_name fp32 \
    --real_runtimes 5
```

---

### Real-Time Visualization (GPU-Resident Data Transfer)

The Warp implementation additionally supports GPU-resident visualization pipelines.

GPU-to-GPU copy:

```bash
python run_one_case_realtime_gpu.py
```

Zero-copy interoperability:

```bash
python run_one_case_realtime_zero.py
```

These scripts evaluate CUDA–OpenGL interoperability using GPU-to-GPU copy and zero-copy data transfer, respectively.

---

### Running Batch Benchmarks

To execute all benchmark configurations for a framework:

```bash
python pytorch_batch_coldstart.py
```

Available options:

```text
--fp                   Floating-point precision:
                       32 or 64
                       (default: 64)
--real_runtimes        Number of repeated visualization runtime measurements
                       performed after a single invocation of the script
                       (default: 9)
--precompute_runtimes  Number of repeated precomputation runtime measurements
                       performed after a single invocation of the script
                       (default: 9)
--numerical_runtimes   Number of repeated solution measurement
                       performed after a single invocation of the script
                       (default: 4)
--save                 Save benchmark results
                       (default: disabled)
--auto_force_input     Automatically apply predefined force inputs
                       without user interaction
                       (default: disabled)
```

Example:

```bash
python pytorch_batch_coldstart.py \
    --fp 32 \
    --real_runtimes 20 \
    --save
```

---

### Output Files

When result saving is enabled, benchmark outputs are written to:

Numerical validation:

```text
data/raw_data/{framework}_{precision}_numerical_res.json
```

CPU-mediated visualization:

```text
data/raw_data/{framework}_{precision}_realtime_cpu_res.json
```

Warp GPU-to-GPU visualization:

```text
data/raw_data/{framework}_{precision}_realtime_gpu_res.json
```

Warp zero-copy visualization:

```text
data/raw_data/{framework}_{precision}_realtime_zero_res.json
```
### Run FEniCS Benchmarks

Start the Docker container and enter the container environment.

Navigate to the FEniCS implementation directory:

```
cd code_fenics
```

#### Generate Reference Solutions

Run:

```
python validate_prism_{load}.py
```

where `{load}` is one of:

```text 
gravity
traction
compression
```

This generates FEniCS reference solutions for the same mesh configurations used by the GPU FEM implementations.

#### Generate Ground Truth Solution

Run:

```bash 
python gen_groundtruth.py
```

This solves the problem on a significantly finer mesh.

#### Interpolate Solutions onto the Ground-Truth Mesh

Run:

```bash 
python interpolation.py
```
This script generates interpolation reference data for the four benchmark meshes (coarse, m1, m2, and fine) using the ground-truth solution obtained on the finest mesh. The generated data are subsequently used by the GPU FEM implementations to evaluate discretization errors.





