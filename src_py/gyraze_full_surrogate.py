
# ============================================================
#  NONCONVERGENT PROJECTION ALGORITHM
# ============================================================

import torch
from torch import nn
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

_MODEL_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "model"))

clf = joblib.load(os.path.join(_MODEL_DIR, "svm_model.pkl"))

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=3, output_dim=20, width=70, depth=3, activation='silu'):
        super().__init__()
        act_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),
            'sigmoid': nn.Sigmoid(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'softplus': nn.Softplus()
        }[activation]
        layers = [nn.Linear(input_dim, width), act_fn]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act_fn]
        layers += [nn.Linear(width, output_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


model = NeuralNetwork(input_dim=3, output_dim=20, width=70, depth=3, activation='silu')
model.load_state_dict(torch.load(os.path.join(_MODEL_DIR, "nn_model_full.pth"), map_location='cpu'))
model.eval()

# Load normalization parameters
norms = np.load(os.path.join(_MODEL_DIR, "normalization_full.npz"))
X_mu, X_sigma = torch.tensor(norms["X_mu"]), torch.tensor(norms["X_sigma"])
Y_mu, Y_sigma = torch.tensor(norms["Y_mu"]), torch.tensor(norms["Y_sigma"])

def normX(X): return (X - X_mu) / X_sigma
def denormy(Y): return Y * Y_sigma + Y_mu

# Fixed mu-grid
vvec = np.array([
    0.000000, 0.020000, 0.080000, 0.180000, 0.320000, 0.500000, 0.720000,
    0.980000, 1.280000, 1.620000, 2.000000, 2.420000, 2.880000, 3.380000,
    3.920000, 4.500000, 5.120000, 5.780000, 6.480000, 7.220000
])

# ============================================================
#  FUNCTION TO EVALUATE THE SURROGATE
# ============================================================

def surrogate_model(mu: float, alpha: float, gamma: float, phi: float):
    """
    NN model for both convergent and unconvergent GYRAZE cases.
    The model was trained on the normal converged GYRAZE data, but also on projected parameters for the unconverged cases.
    It should be equivalent to `gyraze_surrogate_proj.py` model but uses only a signle NN.
    """
    params = [alpha, gamma, phi]

    # --- NN regression prediction ---
    with torch.no_grad():
        x_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
        Y_pred = model(normX(x_tensor))
        Y_pred_denorm = denormy(Y_pred).cpu().numpy().flatten()

    interp = np.interp(mu, vvec, Y_pred_denorm)

    return interp