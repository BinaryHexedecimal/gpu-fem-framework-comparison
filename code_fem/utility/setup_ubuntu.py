
# to compile a C++ extension module that connects:
# Python, CUDA, OpenGL, pybind11
# into a native compiled Python module called "cuda_gl_interop"

from setuptools import setup, Extension
# lets C++ functions/classes appear as normal Python objects.
import pybind11
import os

cuda_path = "/usr/local/cuda" # standard Linux CUDA installation location.

# Creates one compiled module
ext_modules = [
    # defines a compiled C/C++ extension module
    Extension(
        "cuda_gl_interop", # module name
        ["interop.cpp"],  #source files, the actual C++ implementation file.
        # tell the compiler where header files live.
        include_dirs=[
            # pybind11 Headers, find headers like #include <pybind11/pybind11.h>
            pybind11.get_include(),
            # CUDA headers 
            # Provides access to:
            #include <cuda_runtime.h>
            #include <cuda_gl_interop.h>
            # they define cuda APIs
            os.path.join(cuda_path, "include"), 

        ],
        # where compiled CUDA libraries (.so) exist
        library_dirs=[
            os.path.join(cuda_path, "lib64"),
        ],
        # linked during compilation.
        libraries=[
            "cudart", #link CUDA runtime library (needed for cudaMemcpy, etc.)
            "GL",   #link OpenGL library (needed for GL/gl.h functions)
        ],
        extra_compile_args=["-O3"], ## Optimization flag (makes compiled code faster)
        language="c++",
    ),
]

# defines how the package builds
setup(
    name="cuda_gl_interop",
    ext_modules=ext_modules,
)

