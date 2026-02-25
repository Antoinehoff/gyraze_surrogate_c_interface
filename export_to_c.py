"""
export_to_c.py
==============
Exports the GYRAZE surrogate (SVM + NN) to a self-contained C file.

Dependencies:
    pip install m2cgen

Usage (from Python / notebook):
    from export_to_c import generate_c_code
    generate_c_code(
        nn_model    = "nn_model.pth",
        svm_model   = "svm_model.pkl",
        normalization = "normalization.npz",
        output      = "surrogate"        # produces surrogate.c + surrogate.h
    )
"""

import os
import torch
from torch import nn
import numpy as np
import joblib
import m2cgen as m2c


# ── Model definition ─────────────────────────────────────────────────────────

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=3, output_dim=20, width=75, depth=3, activation='silu'):
        super().__init__()
        act_fn = {
            'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'silu': nn.SiLU(),
            'sigmoid': nn.Sigmoid(), 'selu': nn.SELU(),
            'elu': nn.ELU(), 'softplus': nn.Softplus()
        }[activation]
        layers = [nn.Linear(input_dim, width), act_fn]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act_fn]
        layers += [nn.Linear(width, output_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _c_array(values, name, comment=""):
    """Format a Python list as a static C float array initialiser."""
    body = ", ".join(f"{v:.8f}f" for v in values)
    s  = f"/* {comment} */\n" if comment else ""
    s += f"static const float {name}[{len(values)}] = {{{body}}};\n"
    return s


def _dense_layer_c(i, in_dim, out_dim, activation):
    """Emit C code for one dense layer with optional SiLU activation."""
    lines = [f"    /* --- layer {i} --- */"]
    for o in range(out_dim):
        acc = f"b{i}[{o}]"
        for k in range(in_dim):
            acc += f" + W{i}[{o}*{in_dim}+{k}]*h{i}[{k}]"
        lines.append(f"    h{i+1}[{o}] = {acc};")
    if activation == "silu":
        lines.append(f"    for (int _j = 0; _j < {out_dim}; _j++)")
        lines.append(f"        h{i+1}[_j] = h{i+1}[_j] / (1.0f + expf(-h{i+1}[_j]));")
    lines.append("")
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_c_code(
    nn_model: str      = "nn_model.pth",
    svm_model: str     = "svm_model.pkl",
    normalization: str = "normalization.npz",
    output_dir: str    = ".",
    output_name: str   = "surrogate",
):
    """
    Export the GYRAZE surrogate to C.

    Parameters
    ----------
    nn_model      : path to the PyTorch weights file (.pth)
    svm_model     : path to the scikit-learn SVM file (.pkl)
    normalization : path to the normalisation arrays (.npz)
    output_dir    : directory where output files are written (default: '.')
    output_name   : base name for output files (default: 'surrogate')
                    produces <output_dir>/<output_name>.c and .h
    """
    # ── Load models ───────────────────────────────────────────────────────────
    clf = joblib.load(svm_model)

    model = NeuralNetwork(input_dim=3, output_dim=20, width=75, depth=3, activation='silu')
    model.load_state_dict(torch.load(nn_model, map_location='cpu'))
    model.eval()

    norms   = np.load(normalization)
    X_mu    = norms["X_mu"].tolist()
    X_sigma = norms["X_sigma"].tolist()
    Y_mu    = norms["Y_mu"].tolist()
    Y_sigma = norms["Y_sigma"].tolist()

    # ── Extract NN weights ────────────────────────────────────────────────────
    linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    layer_dims    = [(l.in_features, l.out_features) for l in linear_layers]
    n_layers      = len(linear_layers)

    weight_arrays = ""
    for i, layer in enumerate(linear_layers):
        W = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy()
        weight_arrays += _c_array(W.flatten(), f"W{i}",
                                  f"Layer {i} weights ({layer.out_features} x {layer.in_features})")
        weight_arrays += _c_array(b, f"b{i}", f"Layer {i} biases ({layer.out_features})")
        weight_arrays += "\n"

    # ── Build NN forward-pass C code ──────────────────────────────────────────
    buf_decls = "    float h0[3];\n"
    for i, (_, out_d) in enumerate(layer_dims):
        buf_decls += f"    float h{i+1}[{out_d}];\n"

    norm_code = "    /* input normalisation */\n"
    for k in range(3):
        norm_code += f"    h0[{k}] = (x[{k}] - {X_mu[k]:.8f}f) / {X_sigma[k]:.8f}f;\n"

    layer_code = ""
    for i, (in_d, out_d) in enumerate(layer_dims):
        layer_code += _dense_layer_c(i, in_d, out_d, "silu" if i < n_layers - 1 else None)

    out_dim     = layer_dims[-1][1]
    denorm_code = (
        "    /* output denormalisation */\n"
        f"    for (int _j = 0; _j < {out_dim}; _j++)\n"
        f"        out[_j] = h{n_layers}[_j] * Y_sigma[_j] + Y_mu[_j];\n"
    )

    # ── SVM via m2cgen ────────────────────────────────────────────────────────
    svm_c_code = m2c.export_to_c(clf)
    svm_c_code = svm_c_code.replace("double score(", "static double svm_score(")

    os.makedirs(output_dir, exist_ok=True)
    header_name = output_name + ".h"

    # ── Assemble .c ───────────────────────────────────────────────────────────
    c_source = (
        f"/*\n"
        f" * {output_name}.c  –  GYRAZE surrogate model (auto-generated)\n"
        f" *\n"
        f" * Public API:\n"
        f" *   int   gyraze_converged(double alpha, double gamma, double phi);\n"
        f" *   void  gyraze_predict  (float alpha, float gamma, float phi, float out[{out_dim}]);\n"
        f" */\n"
        f"#include <math.h>\n"
        f"#include <stddef.h>\n"
        f'#include "{header_name}"\n\n'
        "/* ── Normalisation constants ─────────────────────────────────────────── */\n" +
        _c_array(X_mu,    "X_mu",    "Input normalisation mean") +
        _c_array(X_sigma, "X_sigma", "Input normalisation std-dev") +
        _c_array(Y_mu,    "Y_mu",    "Output denormalisation mean") +
        _c_array(Y_sigma, "Y_sigma", "Output denormalisation std-dev") + "\n" +
        "/* ── Network weights & biases ───────────────────────────────────────────── */\n" +
        weight_arrays +
        f"/* ── Neural-network forward pass ────────────────────────────────────────── */\n"
        f"void gyraze_predict(float alpha, float gamma, float phi, float out[{out_dim}])\n"
        f"{{\n"
        f"    float x[3] = {{alpha, gamma, phi}};\n"
        f"{buf_decls}\n"
        f"{norm_code}\n"
        f"{layer_code}\n"
        f"{denorm_code}"
        f"}}\n\n"
        "/* ── SVM (generated by m2cgen) ──────────────────────────────────────────── */\n" +
        svm_c_code + "\n"
        "/* ── SVM convergence classifier ─────────────────────────────────────────── */\n"
        "int gyraze_converged(double alpha, double gamma, double phi)\n"
        "{\n"
        "    double input[3] = {alpha, gamma, phi};\n"
        "    return (svm_score(input) >= 0.5) ? 1 : 0;\n"
        "}\n"
    )

    # ── Assemble .h ───────────────────────────────────────────────────────────
    guard = output_name.upper() + "_H"
    h_source = (
        f"/* {header_name}  –  GYRAZE surrogate model public API (auto-generated) */\n"
        f"#ifndef {guard}\n"
        f"#define {guard}\n\n"
        f"/* Returns 1 if GYRAZE is predicted to converge, 0 otherwise. */\n"
        f"int  gyraze_converged(double alpha, double gamma, double phi);\n\n"
        f"/* Runs the NN regression; writes {out_dim} predicted values into out[]. */\n"
        f"void gyraze_predict(float alpha, float gamma, float phi, float out[{out_dim}]);\n\n"
        f"#endif /* {guard} */\n"
    )

    # ── Assemble test_surrogate.c ─────────────────────────────────────────────
    test_source = (
        f'#include <stdio.h>\n'
        f'#include "{header_name}"\n\n'
        f'int main(void)\n'
        f'{{\n'
        f'    double alpha = 4.0, gamma = 1.0, phi = 2.5;\n'
        f'    float out[{out_dim}];\n\n'
        f'    if (!gyraze_converged(alpha, gamma, phi)) {{\n'
        f'        printf("Did not converge\\n");\n'
        f'        return 1;\n'
        f'    }}\n\n'
        f'    gyraze_predict((float)alpha, (float)gamma, (float)phi, out);\n\n'
        f'    for (int i = 0; i < {out_dim}; i++)\n'
        f'        printf("out[%2d] = %.6f\\n", i, out[i]);\n\n'
        f'    return 0;\n'
        f'}}\n'
    )

    # ── Assemble Makefile ─────────────────────────────────────────────────────
    makefile_source = (
        f"CC      = gcc\n"
        f"CFLAGS  = -O2 -Wall\n"
        f"LIBS    = -lm\n\n"
        f"TARGET  = test_surrogate\n"
        f"SRCS    = test_surrogate.c {output_name}.c\n\n"
        f"all: $(TARGET)\n\n"
        f"$(TARGET): $(SRCS)\n"
        f"\t$(CC) $(CFLAGS) -o $@ $^ $(LIBS)\n\n"
        f"clean:\n"
        f"\trm -f $(TARGET)\n"
    )

    # ── Write files ───────────────────────────────────────────────────────────
    c_path    = os.path.join(output_dir, output_name + ".c")
    h_path    = os.path.join(output_dir, output_name + ".h")
    test_path = os.path.join(output_dir, "test_surrogate.c")
    make_path = os.path.join(output_dir, "Makefile")

    with open(c_path, "w") as f:
        f.write(c_source)
    with open(h_path, "w") as f:
        f.write(h_source)
    with open(test_path, "w") as f:
        f.write(test_source)
    with open(make_path, "w") as f:
        f.write(makefile_source)

    print(f"Generated {c_path}, {h_path}, {test_path}, {make_path}")
    print(f"Build: cd {output_dir} && make")


# ── Allow running directly: python export_to_c.py ────────────────────────────

if __name__ == "__main__":
    generate_c_code()
