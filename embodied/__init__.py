__version__ = "2.0.0"

try:
    import colored_traceback

    colored_traceback.add_hook(colors="terminal")
except ImportError:
    pass

# Import .core first to make Agent base class available before jax module loads
# The import order is critical: .core must be imported before .jax to avoid circular import
# ruff: noqa: I001
from .core import *  # Must be first - provides embodied.Agent base class
from . import envs, jax, run  # Requires embodied.Agent to be defined
