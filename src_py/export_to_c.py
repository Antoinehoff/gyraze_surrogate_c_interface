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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))


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
    """Format a Python list as a static C double array initialiser."""
    body = ", ".join(f"{v:.8f}f" for v in values)
    s  = f"/* {comment} */\n" if comment else ""
    s += f"static const double {name}[{len(values)}] = {{{body}}};\n"
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
    nn_model: str      = os.path.join(_ROOT, "model", "nn_model.pth"),
    svm_model: str     = os.path.join(_ROOT, "model", "svm_model.pkl"),
    normalization: str = os.path.join(_ROOT, "model", "normalization.npz"),
    output_dir: str    = os.path.join(_ROOT, "generated_c_code"),
    output_name: str   = "surrogate",
    gkeyll_dir: str    = None,
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
    gkeyll_dir    : directory for the Gkeyll kernel files (default: same as output_dir)
                    produces gk_gyraze_surrogate.c and gkyl_gk_gyraze_surrogate.h
    """
    # ── Mu-grid ───────────────────────────────────────────────────────────────
    try:
        from .gyraze_surrogate import muvec as _muvec
    except ImportError:
        from gyraze_surrogate import muvec as _muvec
    mu_grid = _muvec.tolist()

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
    buf_decls = "    double h0[3];\n"
    for i, (_, out_d) in enumerate(layer_dims):
        buf_decls += f"    double h{i+1}[{out_d}];\n"

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
    # remove redundent "#include <math.h>" from svm_c_code since it's already included in the header
    svm_c_code = svm_c_code.replace("#include <math.h>\n", "")

    os.makedirs(output_dir, exist_ok=True)
    header_name = output_name + ".h"

    # ── Assemble .c ───────────────────────────────────────────────────────────
    c_source = (
        f"/*\n"
        f" * {output_name}.c  –  GYRAZE surrogate model generated from {_HERE}\n"
        f" */\n"
        f'#include "{header_name}"\n'
        +
        # _c_array(X_mu,    "X_mu",    "Input normalisation mean") +
        # _c_array(X_sigma, "X_sigma", "Input normalisation std-dev") +
        _c_array(Y_mu,    "Y_mu",    "Output denormalisation mean") +
        _c_array(Y_sigma, "Y_sigma", "Output denormalisation std-dev") + "\n" +
        weight_arrays +
        _c_array(mu_grid, "MU_GRID", f"Fixed evaluation mu-grid ({len(mu_grid)} points)") + "\n" +
        f"void srgrz_predict(double alpha, double gamma, double phi, double out[{out_dim}])\n"
        f"{{\n"
        f"    double x[3] = {{alpha, gamma, phi}};\n"
        f"{buf_decls}\n"
        f"{norm_code}\n"
        f"{layer_code}\n"
        f"{denorm_code}"
        f"}}\n\n"
        
        + svm_c_code + "\n"
        "int srgrz_converged(double alpha, double gamma, double phi)\n"
        "{\n"
        "    double input[3] = {alpha, gamma, phi};\n"
        "    return (svm_score(input) >= 0.5) ? 1 : 0;\n"
        "}\n\n"
        f"double *srgrz_grid(double *out)\n"
        f"{{\n"
        f"    for (int i = 0; i < SRGRZ_N_MU; i++) {{\n"
        f"        out[i] = MU_GRID[i];\n"
        f"    }}\n"
        f"    return out;\n"
        f"}}\n\n"
        
        f"void srgrz_interp(const double *vcut, const double *mu_new, int n, double *out)\n"
        f"{{\n"
        f"    int ng = SRGRZ_N_MU;\n"
        f"    for (int i = 0; i < n; i++) {{\n"
        f"        double mu = mu_new[i];\n"
        f"        if (mu <= MU_GRID[0])          {{ out[i] = vcut[0];          continue; }}\n"
        f"        if (mu >= MU_GRID[ng - 1])     {{ out[i] = vcut[ng - 1];     continue; }}\n"
        f"        /* binary search for the bracketing interval */\n"
        f"        int lo = 0, hi = ng - 1;\n"
        f"        while (hi - lo > 1) {{\n"
        f"            int mid = (lo + hi) >> 1;\n"
        f"            if (MU_GRID[mid] <= mu) lo = mid; else hi = mid;\n"
        f"        }}\n"
        f"        double t = (mu - MU_GRID[lo]) / (MU_GRID[hi] - MU_GRID[lo]);\n"
        f"        out[i] = vcut[lo] + t * (vcut[hi] - vcut[lo]);\n"
        f"    }}\n"
        f"}}\n\n"
        
        f"void srgrz_eval(const double *mu_new, int n, double alpha, double gamma, double phi, double *out)\n"
        f"{{\n"
        f"    double vcut[SRGRZ_N_MU];\n"
        f"    srgrz_predict(alpha, gamma, phi, vcut);\n"
        f"    srgrz_interp(vcut, mu_new, n, out);\n"
        f"}}\n\n"
        
        f"void srgrz_eval_vcut_fact(const double *mu_new, int n, double alpha, double gamma, double phi, double *out)\n"
        f"{{\n"
        f"    double vcut[SRGRZ_N_MU];\n"
        f"    srgrz_predict(alpha, gamma, phi, vcut);\n"
        f"    double vcut_const = sqrt(SRGRZ_ELEM_CHARGE * (phi - 0.0) / 1.0);\n"
        f"    for (int i = 0; i < n; i++) {{\n"
        f"        out[i] = vcut[i]/vcut_const;\n"
        f"    }}\n"
        f"}}\n\n"

        f"void srgrz_eval_physical(\n"
        f"    const double *mu_new,  int n,\n"
        f"    double phi,            double phi_wall,\n"
        f"    double dens_e,         double temp_e,\n"
        f"    double m_e,            double bmag,\n"
        f"    double bimpact_angle,\n"
        f"    double *out)\n"
        f"{{\n"
        f"    double *munorm = (double*)malloc(n * sizeof(double));\n"
        f"    for (int i = 0; i < n; i++) {{\n"
        f"        munorm[i] = mu_new[i] * bmag / temp_e;\n"
        f"    }}\n"
        f"    double gamma   = (1.0 / bmag) * sqrt(m_e * dens_e / SRGRZ_EPS0);\n"
        f"    double phinorm = SRGRZ_ELEM_CHARGE * (phi - phi_wall) / temp_e;\n"
        f"    double alpha = bimpact_angle * 180.0 / M_PI;\n"
        f"    srgrz_eval(munorm, n, alpha, gamma, phinorm, out);\n"
        f"    free(munorm);\n"
        f"}}\n"
        
        f"void srgrz_eval_physical_vcut_fact(\n"
        f"    const double *mu_new,  int n,\n"
        f"    double phi,            double phi_wall,\n"
        f"    double dens_e,         double temp_e,\n"
        f"    double m_e,            double bmag,\n"
        f"    double bimpact_angle,\n"
        f"    double *out)\n"
        f"{{\n"
        f"    double vcut_const = sqrt(SRGRZ_ELEM_CHARGE * (phi - phi_wall) /temp_e);\n"
        f"    srgrz_eval_physical(mu_new, n, phi, phi_wall, dens_e, temp_e, m_e, bmag, bimpact_angle, out);\n"
        f"    for (int i = 0; i < n; i++) {{\n"
        f"        out[i] = out[i]/vcut_const;\n"
        f"    }}\n"
        f"}}\n"
    )

    # ── Assemble .h ───────────────────────────────────────────────────────────
    guard = output_name.upper() + "_H"
    n_mu = len(mu_grid)
    h_source = (
        f"/* {header_name}  –  GYRAZE surrogate model public API generated from {_HERE} */\n"
        f"#ifndef {guard}\n"
        f"#define {guard}\n\n"
        f"#include <math.h>\n"
        f"#include <stdlib.h>\n"
        f"#include <stddef.h>\n\n"
        f"#ifndef M_PI\n"
        f"#define M_PI 3.14159265358979323846\n"
        f"#endif\n\n"
        f"static const double SRGRZ_ELEM_CHARGE = 1.602176634e-19;  /* Elementary charge (C) */\n"
        f"static const double SRGRZ_EPS0        = 8.8541878128e-12; /* Vacuum permittivity (F/m) */\n\n"
        f"/* Number of points in the fixed mu-grid. */\n"
        f"#define SRGRZ_N_MU {n_mu}\n\n"
        f"/* Returns 1 if GYRAZE is predicted to converge, 0 otherwise. */\n"
        f"int  srgrz_converged(double alpha, double gamma, double phi);\n\n"
        f"/* Runs the NN regression; writes SRGRZ_N_MU predicted v_par_cut values into out[]. */\n"
        f"void srgrz_predict(double alpha, double gamma, double phi, double out[SRGRZ_N_MU]);\n\n"
        f"/* Copies the SRGRZ_N_MU-element mu-grid into out[] and returns out. */\n"
        f"double *srgrz_grid(double *out);\n\n"
        f"/* Linear interpolation of vcut[SRGRZ_N_MU] (on the fixed mu-grid) onto\n"
        f" * mu_new[n]; results are written into out[n]. Clamps at the grid boundaries. */\n"
        f"void srgrz_interp(const double *vcut, const double *mu_new, int n, double *out);\n\n"
        f"/* Returns the prediction of a custom mu grid of size n */\n"
        f"void srgrz_eval(const double *mu_new, int n, double alpha, double gamma, double phi, double *out);\n\n"
        f"/* Converts from physical parameters and evaluates on a custom mu grid.\n"
        f" * Conversion formulas:\n"
        f" *   gamma   = (1/Bmag) * sqrt(m_e * dens_e / eps0)\n"
        f" *   phinorm = e * phi / temp_e\n"
        f" */\n"
        f"void srgrz_eval_physical(\n"
        f"    const double *mu_new, int n,\n"
        f"    double phi, double phi_wall, double dens_e, double temp_e, double m_e, double bmag, double bimpact_angle,\n"
        f"    double *out);\n\n"
        f"/* Same as srgrz_eval_physical, but normalises output by sqrt(e * (phi - phi_wall) / T_e) */\n"
        f"void srgrz_eval_physical_vcut_fact(\n"
        f"    const double *mu_new, int n,\n"
        f"    double phi, double phi_wall, double dens_e, double temp_e, double m_e, double bmag, double bimpact_angle,\n"
        f"    double *out);\n\n"
        f"#endif /* {guard} */\n"
    )

    # ── Assemble test_surrogate.c ─────────────────────────────────────────────
    test_source = (
        f'#include <stdio.h>\n'
        f'#include <stdlib.h>\n'
        f'#include <string.h>\n'
        f'#include "{header_name}"\n\n'
        f'int main(int argc, char *argv[])\n'
        f'{{\n'
        f'    double alpha = 4.0, gamma = 1.0, phi = 2.5, mu = 1.0;\n'
        f'    double phi_wall = 0.0, dens_e = 1e19, temp_e = 100.0;\n'
        f'    double m_e = 9.109e-31, bmag = 1.0, bimpact_angle = 0.0;\n'
        f'    double out[{out_dim}];\n\n'
        f'    if (argc < 2) {{\n'
        f'        fprintf(stderr, "Usage: %s {{predict|eval|physical|physical_vcut_fact}} [args...]\\n", argv[0]);\n'
        f'        fprintf(stderr, "\\n  predict [alpha [gamma [phi]]]:\\n");\n'
        f'        fprintf(stderr, "    Evaluate full surrogate at (alpha, gamma, phi)\\n");\n'
        f'        fprintf(stderr, "    Returns {out_dim} values (one for each mu grid point)\\n");\n'
        f'        fprintf(stderr, "\\n  eval mu [alpha [gamma [phi]]]:\\n");\n'
        f'        fprintf(stderr, "    Evaluate surrogate at single mu point with (alpha, gamma, phi)\\n");\n'
        f'        fprintf(stderr, "    mu: scalar value (not a grid)\\n");\n'
        f'        fprintf(stderr, "\\n  physical mu [phi [phi_wall [dens_e [temp_e [m_e [bmag [bimpact_angle]]]]]]]:\\n");\n'
        f'        fprintf(stderr, "    Evaluate at single mu using physical parameters\\n");\n'
        f'        fprintf(stderr, "\\n  physical_vcut_fact mu [phi [phi_wall [dens_e [temp_e [m_e [bmag [bimpact_angle]]]]]]]:\\n");\n'
        f'        fprintf(stderr, "    Evaluate at single mu using physical parameters and normalise by sqrt(e * (phi - phi_wall) / T_e)\\n");\n'
        f'        return 1;\n'
        f'    }}\n\n'
        f'    if (strcmp(argv[1], "predict") == 0) {{\n'
        f'        /* Parse optional arguments */\n'
        f'        if (argc > 2) alpha = atof(argv[2]);\n'
        f'        if (argc > 3) gamma = atof(argv[3]);\n'
        f'        if (argc > 4) phi = atof(argv[4]);\n\n'
        f'        srgrz_predict(alpha, gamma, phi, out);\n\n'
        f'        for (int i = 0; i < {out_dim}; i++)\n'
        f'            printf("out[%2d] = %.6f\\n", i, out[i]);\n\n'
        f'    }}\n'
        f'    else if (strcmp(argv[1], "eval") == 0) {{\n'
        f'        double result;\n\n'
        f'        /* Parse arguments */\n'
        f'        if (argc > 2) mu = atof(argv[2]);\n'
        f'        if (argc > 3) alpha = atof(argv[3]);\n'
        f'        if (argc > 4) gamma = atof(argv[4]);\n'
        f'        if (argc > 5) phi = atof(argv[5]);\n\n'
        f'        srgrz_eval(&mu, 1, alpha, gamma, phi, &result);\n\n'
        f'        printf("out = %.6f\\n", result);\n\n'
        f'    }}\n'
        f'    else if (strcmp(argv[1], "physical") == 0) {{\n'
        f'        double result;\n\n'
        f'        /* Parse arguments */\n'
        f'        if (argc > 2) mu = atof(argv[2]);\n'
        f'        if (argc > 3) phi = atof(argv[3]);\n'
        f'        if (argc > 4) phi_wall = atof(argv[4]);\n'
        f'        if (argc > 5) dens_e = atof(argv[5]);\n'
        f'        if (argc > 6) temp_e = atof(argv[6]);\n'
        f'        if (argc > 7) m_e = atof(argv[7]);\n'
        f'        if (argc > 8) bmag = atof(argv[8]);\n'
        f'        if (argc > 9) bimpact_angle = atof(argv[9]);\n\n'
        f'        srgrz_eval_physical(&mu, 1, phi, phi_wall, dens_e, temp_e, m_e, bmag, bimpact_angle, &result);\n\n'
        f'        printf("out = %.6f\\n", result);\n\n'
        f'    }}\n'
        f'    else if (strcmp(argv[1], "physical_vcut_fact") == 0) {{\n'
        f'        double result;\n\n'
        f'        /* Parse arguments */\n'
        f'        if (argc > 2) mu = atof(argv[2]);\n'
        f'        if (argc > 3) phi = atof(argv[3]);\n'
        f'        if (argc > 4) phi_wall = atof(argv[4]);\n'
        f'        if (argc > 5) dens_e = atof(argv[5]);\n'
        f'        if (argc > 6) temp_e = atof(argv[6]);\n'
        f'        if (argc > 7) m_e = atof(argv[7]);\n'
        f'        if (argc > 8) bmag = atof(argv[8]);\n'
        f'        if (argc > 9) bimpact_angle = atof(argv[9]);\n\n'
        f'        srgrz_eval_physical_vcut_fact(&mu, 1, phi, phi_wall, dens_e, temp_e, m_e, bmag, bimpact_angle, &result);\n\n'
        f'        printf("out = %.6f\\n", result);\n\n'
        f'    }}\n'
        f''
        f'    else {{\n'
        f'        fprintf(stderr, "Unknown mode: %s\\n", argv[1]);\n'
        f'        fprintf(stderr, "Use \'predict\', \'eval\', or \'physical\'\\n");\n'
        f'        return 1;\n'
        f'    }}\n\n'
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

    # ── Assemble Gkeyll kernel ────────────────────────────────────────────────
    gk_dir         = gkeyll_dir if gkeyll_dir is not None else output_dir
    gk_c_name      = "bc_sheath_gyrokinetic_gyraze_surrogate.c"
    gk_h_name      = "gkyl_bc_sheath_gyrokinetic_gyraze_surrogate.h"

    gk_h_source = (
        f"#pragma once\n\n"
        f"#include <math.h>\n"
        f"#include <gkyl_const.h>\n"
        f"#include <gkyl_util.h>\n\n"
        
        f"/* Number of points in the fixed mu-grid. */\n"
        f"#define SRGRZ_N_MU {n_mu}\n\n"
        f"EXTERN_C_BEG\n\n"
        
        f"/**\n"
        f" * Returns 1 if GYRAZE is predicted to converge, 0 otherwise.\n"
        f" *\n"
        f" * @param alpha: impact angle in degrees\n"
        f" * @param gamma: normalised plasma density parameter\n"
        f" * @param phi:   normalised sheath potential drop (e * (phi - phi_wall) / T_e)\n"
        f" */\n"
        f"GKYL_CU_DH int  bc_sheath_gyrokinetic_srgrz_converged(double alpha, double gamma, double phi);\n\n"
        
        f"/**\n"
        f" * Runs the NN regression; writes SRGRZ_N_MU predicted v_par_cut values into out[].\n"
        f" *\n"
        f" * @param alpha: impact angle in degrees\n"
        f" * @param gamma: normalised plasma density parameter\n"
        f" * @param phi:   normalised sheath potential drop (e * (phi - phi_wall) / T_e)\n"
        f" */\n"
        f"GKYL_CU_DH void bc_sheath_gyrokinetic_srgrz_predict(double alpha, double gamma, double phi, double out[SRGRZ_N_MU]);\n\n"
        
        f"/**\n"
        f" * Copies the SRGRZ_N_MU-element mu-grid into out[] and returns out.\n"
        f" *\n"
        f" * @param out: output array of size SRGRZ_N_MU\n"
        f" */\n"
        f"GKYL_CU_DH double *bc_sheath_gyrokinetic_srgrz_grid(double *out);\n\n"
        
        f"/**\n"
        f" * Linear interpolation of vcut[SRGRZ_N_MU] (on the fixed mu-grid) onto\n"
        f" * mu_new[n]; results are written into out[n]. Clamps at the grid boundaries.\n"
        f" *\n"
        f" * @param vcut:    input array of size SRGRZ_N_MU containing values at the fixed mu-grid\n"
        f" * @param mu_new:  input array of size n containing the new mu points\n"
        f" * @param n:       number of points in mu_new and out\n"
        f" * @param out:     output array of size n where interpolated values are written\n"
        f" */\n"
        f"GKYL_CU_DH void bc_sheath_gyrokinetic_srgrz_interp(const double *vcut, const double *mu_new, int n, double *out);\n\n"
        
        f"/**\n"
        f" * Returns the prediction of a custom mu grid of size n\n"
        f" *\n"
        f" * @param mu_new:  input array of size n containing the new mu points\n"
        f" * @param n:       number of points in mu_new and out\n"
        f" * @param alpha:   impact angle in degrees\n"
        f" * @param gamma:   normalised plasma density parameter\n"
        f" * @param phi:     normalised sheath potential drop (e * (phi - phi_wall) / T_e)\n"
        f" * @param out:     output array of size n where interpolated values are written\n"
        f" */\n"
        f"GKYL_CU_DH void bc_sheath_gyrokinetic_srgrz_eval(const double *mu_new, int n, double alpha, double gamma, double phi, double *out);\n\n"
        
        f"/**\n"
        f" * Converts from physical parameters and evaluates on a custom mu grid.\n"
        f" * Conversion formulas:\n"
        f" *   munorm  = mu*Bmag / temp_e\n"
        f" *   gamma   = (1/Bmag) * sqrt(m_e * dens_e / eps0)\n"
        f" *   phinorm = e * phi / temp_e\n"
        f" *\n"
        f" * @param mu_new:  input array of size n containing the new mu points\n"
        f" * @param n:       number of points in mu_new and out\n"
        f" * @param phi:     sheath potential (V)\n"
        f" * @param phi_wall: wall potential (V)\n"
        f" * @param dens_e:  electron density (m^-3)\n"
        f" * @param temp_e:  electron temperature (eV)\n"
        f" * @param m_e:     electron mass (kg)\n"
        f" * @param bmag:    magnetic field strength (T)\n"
        f" * @param bimpact_angle: magnetic impact angle (radians)\n"
        f" */\n"
        f"GKYL_CU_DH void bc_sheath_gyrokinetic_srgrz_eval_physical(const double *mu_new, int n, double phi, double phi_wall,\n"
        f"    double dens_e, double temp_e, double m_e, double bmag, double bimpact_angle, double *out);\n\n"
        
        f"/**\n"
        f" * Same as srgrz_eval_physical, but normalises output by sqrt(2 * e * (phi - phi_wall) / m_e)\n"
        f" *\n"
        f" * @param mu_new:  input array of size n containing the new mu points\n"
        f" * @param n:       number of points in mu_new and out\n"
        f" * @param phi:     sheath potential (V)\n"
        f" * @param phi_wall: wall potential (V)\n"
        f" * @param dens_e:  electron density (m^-3)\n"
        f" * @param temp_e:  electron temperature (eV)\n"
        f" * @param m_e:     electron mass (kg)\n"
        f" * @param bmag:    magnetic field strength (T)\n"
        f" * @param bimpact_angle: magnetic impact angle (radians)\n"
        f" */\n"
        f"GKYL_CU_DH void bc_sheath_gyrokinetic_srgrz_eval_physical_vcut_fact(const double *mu_new, int n, double phi, double phi_wall,\n"
        f"    double dens_e, double temp_e, double m_e, double bmag, double bimpact_angle, double *out);\n\n"
        
        f"/**\n"
        f" * Same as srgrz_eval_physical_vcut_fact, but normalises return 0 if gyraze is not converging.\n"
        f" *\n"
        f" * @param mu_new:  input array of size n containing the new mu points\n"
        f" * @param n:       number of points in mu_new and out\n"
        f" * @param phi:     sheath potential (V)\n"
        f" * @param phi_wall: wall potential (V)\n"
        f" * @param dens_e:  electron density (m^-3)\n"
        f" * @param temp_e:  electron temperature (eV)\n"
        f" * @param m_e:     electron mass (kg)\n"
        f" * @param bmag:    magnetic field strength (T)\n"
        f" * @param bimpact_angle: magnetic impact angle (radians)\n"
        f" */\n"
        f"GKYL_CU_DH void bc_sheath_gyrokinetic_srgrz_eval_physical_vcut_fact_converged(const double *mu_new, int n, double phi, double phi_wall,\n"
        f"    double dens_e, double temp_e, double m_e, double bmag, double bimpact_angle, double *out);\n\n"
    )

    gk_c_source = (
        f"/*\n"
        f" * {gk_c_name}  --  GYRAZE surrogate model, Gkeyll kernel generated from {_HERE}\n"
        f" */\n"
        f'#include "{gk_h_name}"\n\n'
        
        "/*  Normalisation constants   */\n" +
        # _c_array(X_mu,    "X_mu",    "Input normalisation mean") +
        # _c_array(X_sigma, "X_sigma", "Input normalisation std-dev") +
        _c_array(Y_mu,    "Y_mu",    "Output denormalisation mean") +
        _c_array(Y_sigma, "Y_sigma", "Output denormalisation std-dev") + "\n" +
        "/*  Network weights & biases   */\n" +
        weight_arrays +
        "/*   Mu-grid */\n" +
        _c_array(mu_grid, "MU_GRID", f"Fixed evaluation mu-grid ({len(mu_grid)} points)") + "\n" +
        f"/*  Neural-network forward pass  */\n"
        f"GKYL_CU_DH void bc_sheath_gyrokinetic_srgrz_predict(double alpha, double gamma, double phi, double out[{out_dim}])\n"
        f"{{\n"
        f"    double x[3] = {{alpha, gamma, phi}};\n"
        f"{buf_decls}\n"
        f"{norm_code}\n"
        f"{layer_code}\n"
        f"{denorm_code}"
        f"}}\n\n"
        + svm_c_code + "\n"
        
        "GKYL_CU_DH int bc_sheath_gyrokinetic_srgrz_converged(double alpha, double gamma, double phi)\n"
        "{\n"
        "    double input[3] = {alpha, gamma, phi};\n"
        "    return (svm_score(input) >= 0.5) ? 1 : 0;\n"
        "}\n\n"
        
        f"GKYL_CU_DH double *srgrz_grid(double *out)\n"
        f"{{\n"
        f"    for (int i = 0; i < SRGRZ_N_MU; i++) {{\n"
        f"        out[i] = MU_GRID[i];\n"
        f"    }}\n"
        f"    return out;\n"
        f"}}\n\n"
        
        f"GKYL_CU_DH void bc_sheath_gyrokinetic_srgrz_interp(const double *vcut, const double *mu_new, int n, double *out)\n"
        f"{{\n"
        f"    int ng = SRGRZ_N_MU;\n"
        f"    for (int i = 0; i < n; i++) {{\n"
        f"        double mu = mu_new[i];\n"
        f"        if (mu <= MU_GRID[0])          {{ out[i] = vcut[0];          continue; }}\n"
        f"        if (mu >= MU_GRID[ng - 1])     {{ out[i] = vcut[ng - 1];     continue; }}\n"
        f"        /* binary search for the bracketing interval */\n"
        f"        int lo = 0, hi = ng - 1;\n"
        f"        while (hi - lo > 1) {{\n"
        f"            int mid = (lo + hi) >> 1;\n"
        f"            if (MU_GRID[mid] <= mu) lo = mid; else hi = mid;\n"
        f"        }}\n"
        f"        double t = (mu - MU_GRID[lo]) / (MU_GRID[hi] - MU_GRID[lo]);\n"
        f"        out[i] = vcut[lo] + t * (vcut[hi] - vcut[lo]);\n"
        f"    }}\n"
        f"}}\n\n"
        
        f"GKYL_CU_DH void bc_sheath_gyrokinetic_srgrz_eval(const double *mu_new, int n, double alpha, double gamma, double phi, double *out)\n"
        f"{{\n"
        f"    double vcut[SRGRZ_N_MU];\n"
        f"    bc_sheath_gyrokinetic_srgrz_predict(alpha, gamma, phi, vcut);\n"
        f"    bc_sheath_gyrokinetic_srgrz_interp(vcut, mu_new, n, out);\n"
        f"}}\n\n"
        
        "GKYL_CU_DH void bc_sheath_gyrokinetic_srgrz_eval_physical(const double *mu_new, int n, double phi, double phi_wall, double dens_e,\n"
        "    double temp_e, double m_e, double bmag, double bimpact_angle, double *out)\n"
        "{\n"
        "    double munorm[n];\n"
        "    for (int i = 0; i < n; i++) {\n"
        "        munorm[i] = mu_new[i] * bmag / temp_e;\n"
        "    }\n"
        "    double gamma   = (1.0 / bmag) * sqrt(m_e * dens_e / GKYL_EPSILON0);\n"
        "    double phinorm = (GKYL_ELEMENTARY_CHARGE * phi) / temp_e;\n"
        "    double alpha = bimpact_angle * 180/M_PI;\n"
        "    bc_sheath_gyrokinetic_srgrz_eval(munorm, n, alpha, gamma, phinorm, out);\n"
        "}\n"
    
        "GKYL_CU_DH void bc_sheath_gyrokinetic_srgrz_eval_physical_vcut_fact(const double *mu_new,  int n, double phi, double phi_wall,\n"
        "    double dens_e, double temp_e, double m_e, double bmag, double bimpact_angle, double *out)\n"
        "{\n"
        "    double vcut_const = sqrt(GKYL_ELEMENTARY_CHARGE * (phi - phi_wall) /temp_e);\n"
        "    bc_sheath_gyrokinetic_srgrz_eval_physical(mu_new, n, phi, phi_wall, dens_e, temp_e, m_e, bmag, bimpact_angle, out);\n"
        "    for (int i = 0; i < n; i++) {\n"
        "        out[i] = out[i]/vcut_const;\n"
        "    }\n"
        "}\n"
        
        "GKYL_CU_DH void bc_sheath_gyrokinetic_srgrz_eval_physical_vcut_fact_converged(const double *mu_new,  int n, double phi, double phi_wall,\n"
        "    double dens_e, double temp_e, double m_e, double bmag, double bimpact_angle, double *out)\n"
        "{\n"
        "    double gamma   = (1.0 / bmag) * sqrt(m_e * dens_e / GKYL_EPSILON0);\n"
        "    double phinorm = (GKYL_ELEMENTARY_CHARGE * phi) / temp_e;\n"
        "    double alpha = bimpact_angle * 180/M_PI;\n"
        "    int converged = bc_sheath_gyrokinetic_srgrz_converged(alpha, gamma, phinorm);\n"
        "    if (converged) {\n"
        "        bc_sheath_gyrokinetic_srgrz_eval_physical(mu_new, n, phi, phi_wall, dens_e, temp_e, m_e, bmag, bimpact_angle, out);\n"
        "    } else {\n"
        "        for (int i = 0; i < n; i++) {\n"
        "            out[i] = 0.0;\n"
        "        }\n"
        "    }\n"
        "}\n"
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

    os.makedirs(gk_dir, exist_ok=True)
    gk_c_path = os.path.join(gk_dir, gk_c_name)
    gk_h_path = os.path.join(gk_dir, gk_h_name)

    with open(gk_c_path, "w") as f:
        f.write(gk_c_source)
    with open(gk_h_path, "w") as f:
        f.write(gk_h_source)

    print(f"Generated {c_path}, {h_path}, {test_path}, {make_path}")
    print(f"Generated Gkeyll kernel: {gk_c_path}, {gk_h_path}")
    print(f"Build: cd {output_dir} && make")


# ── Allow running directly: python export_to_c.py ────────────────────────────

if __name__ == "__main__":
    generate_c_code()
