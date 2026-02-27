"""
src_py – GYRAZE surrogate model package
========================================
Public re-exports for convenient top-level access:

    from src_py import surrogate_model, muvec           # basic NN+SVM
    from src_py import surrogate_model_proj              # extended (boundary search)
    from src_py import generate_c_code                   # C export

Individual modules can still be imported directly:
    from src_py.gyraze_surrogate import surrogate_model
    from src_py.export_to_c import generate_c_code
"""

from .gyraze_surrogate import surrogate_model, muvec
from .surrogate_proj import surrogate_model as surrogate_model_proj
from .export_to_c import generate_c_code

__all__ = [
    "surrogate_model",
    "surrogate_model_proj",
    "muvec",
    "generate_c_code",
]
