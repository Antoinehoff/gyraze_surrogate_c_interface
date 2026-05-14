import sys
import os

# Add project root to path so `src_py` is importable as a package
_root = os.path.normpath(os.path.join(os.path.abspath(''), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Generate KANN version of the surrogate model
from src_py import to_kann
to_kann.write_kann(
    model_path="/Users/ahoffman/gkeyll_sheath_ai/model/nn_model_conv_MPE.pth",
    norm_path="/Users/ahoffman/gkeyll_sheath_ai/model/normalization_conv_MPE.npz",
    output_path="/Users/ahoffman/gkeyll_sheath_ai/model",
)