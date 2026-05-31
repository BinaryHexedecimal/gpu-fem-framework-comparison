# pyright: reportInvalidTypeForm=false

import importlib

def check(pkg_name, import_name=None):
    try:
        module = importlib.import_module(import_name or pkg_name)
        version = getattr(module, "__version__", "unknown")
        print(f"{pkg_name}: {version}")
    except ImportError:
        print(f"{pkg_name}: NOT INSTALLED")

check("warp", "warp")
check("torch", "torch")
check("jax", "jax")
