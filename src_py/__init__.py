# Import the grid
from .gyraze_conv_surrogate import muvec
# Vanilla surrogate model trained on convergent GYRAZE cases
from .gyraze_conv_surrogate import surrogate_model as surrogate_conv_model, muvec
# Surrogate only for the unconvergent cases, trained on projected unconverged data.
from .gyraze_unconv_surrogate import surrogate_model as surrogate_unconv_model
# Double surrogate, one for the converged cases and one for the unconverged cases, with a SVM classifier to predict convergence.
from .gyraze_conv_unconv_surrogate import surrogate_model as surrogate_conv_unconv_model
# Vanilla surrogate but with a Python wrapper that checks convergence and project the parameters if needed.
from .gyraze_surrogate_proj import surrogate_model as surrogate_model_proj
# Surrogate that should be equivalent to the previous one but with the projection done internally by the NN.
from .gyraze_full_surrogate import surrogate_model as surrogate_full_model
# Surrogate like above but using the MPE data.
from .gyraze_full_MPE_surrogate import surrogate_model as surrogate_full_MPE_model
# Surrogate trained on the MPE data but only converged GYRAZE
from .gyraze_conv_MPE_surrogate import surrogate_model as surrogate_conv_MPE_model
# SVM classifier for convergence prediction
from .gyraze_surrogate_proj import svm_predict
# C code export function
from .export_to_c import generate_c_code

__all__ = [
    "surrogate_conv_model",
    "surrogate_unconv_model",
    "surrogate_conv_unconv_model",
    "surrogate_model_proj",
    "surrogate_full_model",
    "surrogate_full_MPE_model",
    "surrogate_conv_MPE_model",
    "svm_predict",
    "muvec",
    "generate_c_code",
]
