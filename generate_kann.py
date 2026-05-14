import sys
import os
import numpy as np
rng = np.random.default_rng(42)
# Add project root to path so `src_py` is importable as a package
_root = os.path.normpath(os.path.join(os.path.abspath(''), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)
    
# .pth and .npz inputs:
model_pth = "/Users/ahoffman/gkeyll_sheath_ai/model/nn_model_conv_MPE.pth"
norm_npz = "/Users/ahoffman/gkeyll_sheath_ai/model/normalization_conv_MPE.npz"

# .kann output:
output_kann = "/Users/ahoffman/gkeyll_sheath_ai/model/nn_model_conv_MPE.kann"

# Generate KANN version of the surrogate model
from src_py import to_kann
to_kann.write_kann(
    model_path=model_pth,
    norm_path=norm_npz,
    output_path=output_kann,
)

x_raw = rng.standard_normal(3).astype('float32') * 2.5 + 4.0

to_kann.verify_kann_numpy(
    kann_path=output_kann,
    model_path=model_pth,
    norm_path=norm_npz,
    x_input=x_raw,
)