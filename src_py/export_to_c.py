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
REPO_INFO = f"{os.path.basename(_ROOT)} @ {os.popen('git -C ' + _ROOT + ' rev-parse --short HEAD').read().strip()}"


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


def _dense_layer_c(i, in_dim, out_dim, activation, weights_ptr="w"):
    """Emit C code for one dense layer with optional SiLU activation."""
    pfx = f"{weights_ptr}->" if weights_ptr else ""
    lines = [f"    /* --- layer {i} --- */"]
    for o in range(out_dim):
        acc = f"{pfx}b{i}[{o}]"
        for k in range(in_dim):
            acc += f" + {pfx}W{i}[{o}*{in_dim}+{k}]*h{i}[{k}]"
        lines.append(f"    h{i+1}[{o}] = {acc};")
    if activation == "silu":
        lines.append(f"    for (int _j = 0; _j < {out_dim}; _j++)")
        lines.append(f"        h{i+1}[_j] = h{i+1}[_j] / (1.0f + expf(-h{i+1}[_j]));")
    lines.append("")
    return "\n".join(lines)


def _c_struct_def(layer_dims, out_dim, n_mu, type_name):
    """Generate the typedef struct for weight storage."""
    lines = ["/* --- weight storage type --- **/", f"typedef struct {{"]
    for i, (in_d, out_d) in enumerate(layer_dims):
        lines.append(f"  double W{i}[{in_d * out_d}], b{i}[{out_d}];")
    lines.append(f"  double Y_mu[{out_dim}], Y_sigma[{out_dim}];")
    lines.append(f"  double MU_GRID[{n_mu}];")
    lines.append(f"}} {type_name};\n")
    return "\n".join(lines) + "\n"


def _c_struct_fields(linear_layers, Y_mu, Y_sigma, mu_grid):
    """Generate the designated initializer fields for the weight struct."""
    lines = []
    for i, layer in enumerate(linear_layers):
        W = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy()
        W_body = ", ".join(f"{v:.8f}f" for v in W.flatten())
        b_body = ", ".join(f"{v:.8f}f" for v in b.flatten())
        lines.append(f"  /* Layer {i} weights ({layer.out_features} x {layer.in_features}) */")
        lines.append(f"  .W{i} = {{{W_body}}},")
        lines.append(f"  .b{i} = {{{b_body}}},")
    Y_mu_body    = ", ".join(f"{v:.8f}f" for v in Y_mu)
    Y_sigma_body = ", ".join(f"{v:.8f}f" for v in Y_sigma)
    mu_body      = ", ".join(f"{v:.8f}f" for v in mu_grid)
    lines.append(f"  .Y_mu    = {{{Y_mu_body}}},")
    lines.append(f"  .Y_sigma = {{{Y_sigma_body}}},")
    lines.append(f"  .MU_GRID = {{{mu_body}}}")
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
    n_mu = len(mu_grid)
    
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

    struct_fields = _c_struct_fields(linear_layers, Y_mu, Y_sigma, mu_grid)

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
        f"        out[_j] = h{n_layers}[_j] * w->Y_sigma[_j] + w->Y_mu[_j];\n"
    )
    filter_negative_code = (
        "    /* enforce non-negativity of the output (physically vcut cannot be negative) */\n"
        f"    for (int _j = 0; _j < {out_dim}; _j++)\n"
        f"        if (out[_j] < 0.0f) out[_j] = 0.0f;\n"
    )

    # ── SVM via m2cgen ────────────────────────────────────────────────────────
    svm_c_code = m2c.export_to_c(clf)
    svm_c_code = svm_c_code.replace("double score(", "double svm_score(")
    # remove redundent "#include <math.h>" from svm_c_code since it's already included in the header
    svm_c_code = svm_c_code.replace("#include <math.h>\n", "")

    # -- Generate both a test version and a Gkeyll-compatible version of the C code --        
    for code_gen in ['test','gkyl']:
        if code_gen == 'test':
            c_fname = f"{output_name}.c"
            h_fname = f"{output_name}.h"
            outdir = output_dir
            
            cu_flag = ""
            func_prefix = "srgrz"
            cons_prefix = "SRGRZ"
            header_head = (
                f"#ifndef {output_name.upper()}_H\n"
                f"#define {output_name.upper()}_H\n\n"
                f"#include <math.h>\n"
                f"#include <stdlib.h>\n"
                f"#include <stddef.h>\n\n"
                f"#include <stdio.h>\n\n"
                f"#ifndef {cons_prefix}_PI\n"
                f"#define {cons_prefix}_PI 3.14159265358979323846\n"
                f"#endif\n\n"
                f"static const double {cons_prefix}_ELEMENTARY_CHARGE = 1.602176634e-19;  /* Elementary charge (C) */\n"
                f"static const double {cons_prefix}_ELECTRON_MASS = 9.10938356e-31; /* Electron mass (kg) */\n"
                f"static const double {cons_prefix}_EPSILON0 = 8.8541878128e-12; /* Vacuum permittivity (F/m) */\n\n"
            )
            header_tail = f"#endif /* {output_name.upper()}_H */\n"
            extern_c_beg = ""
        else:
            c_fname = "bc_sheath_gyrokinetic_gyraze_surrogate.c"
            h_fname = "gkyl_bc_sheath_gyrokinetic_gyraze_surrogate.h"
            outdir = gkeyll_dir if gkeyll_dir is not None else output_dir
            
            cu_flag = "GKYL_CU_DH "
            func_prefix = "bc_sheath_gyrokinetic_srgrz"
            cons_prefix = "GKYL"
            header_head = (
                f"#pragma once\n\n"
                f"#include <math.h>\n"
                f"#include <gkyl_const.h>\n"
                f"#include <gkyl_util.h>\n\n"
            )
            header_tail = f"EXTERN_C_END\n\n"
            extern_c_beg = "EXTERN_C_BEG\n\n"
        type_name = "srgrz_weights_t"
        inst_name = "srgrz_w"
        struct_def = _c_struct_def(layer_dims, out_dim, n_mu, type_name)
        if code_gen == 'gkyl':
            struct_instances = (
                f"/* host copy – visible in the __host__ pass */\n"
                f"static const {type_name} {inst_name}_h = {{\n{struct_fields}\n}};\n\n"
                f"/* device copy – visible in the __device__ pass */\n"
                f"#ifdef GKYL_HAVE_CUDA\n"
                f"__device__ static const {type_name} {inst_name}_d = {{\n{struct_fields}\n}};\n"
                f"#endif\n\n"
                f"/* Select the right copy based on compilation context */\n"
                f"#ifdef __CUDA_ARCH__\n"
                f"#  define SRGRZ_WEIGHTS {inst_name}_d\n"
                f"#else\n"
                f"#  define SRGRZ_WEIGHTS {inst_name}_h\n"
                f"#endif\n\n"
            )
        else:
            struct_instances = (
                f"/* host copy */\n"
                f"static const {type_name} {inst_name}_h = {{\n{struct_fields}\n}};\n"
                f"#define SRGRZ_WEIGHTS {inst_name}_h\n\n"
            )
        # ── Assemble .c ───────────────────────────────────────────────────────────
        c_source = (
            f"/*\n"
            f" * {c_fname}  –  GYRAZE surrogate model generated from {REPO_INFO}\n"
            f" */\n"
            f'#include "{h_fname}"\n\n'
            + struct_instances +
            f"{cu_flag}void {func_prefix}_predict(double alpha, double gamma, double phi, double out[{out_dim}])\n"
            f"{{\n"
            f"    const {type_name} *w = &SRGRZ_WEIGHTS;\n"
            f"    double x[3] = {{alpha, gamma, phi}};\n"
            f"{buf_decls}\n"
            f"{norm_code}\n"
            f"{layer_code}\n"
            f"{denorm_code}"
            f"{filter_negative_code}"
            f"}}\n\n"
            + cu_flag + svm_c_code + "\n"
            
            f"{cu_flag}int {func_prefix}_converged(double alpha, double gamma, double phi)\n"
            "{\n"
            f"    double input[3] = {{alpha, gamma, phi}};\n"
            f"    return (svm_score(input) >= 0.0) ? 1 : 0;\n"
            "}\n\n"

            f"/* Finite-difference gradient of svm_score(x) */\n"
            f"{cu_flag}static void _srgrz_svm_grad(const double *x, double h, double *g)\n"
            f"{{\n"
            f"    double xp[3], xm[3];\n"
            f"    for (int i = 0; i < 3; i++) {{\n"
            f"        xp[0]=x[0]; xp[1]=x[1]; xp[2]=x[2];\n"
            f"        xm[0]=x[0]; xm[1]=x[1]; xm[2]=x[2];\n"
            f"        xp[i] += h; xm[i] -= h;\n"
            f"        g[i] = (svm_score(xp) - svm_score(xm)) / (2.0*h);\n"
            f"    }}\n"
            f"}}\n\n"

            f"/* Project x0 onto the convergent region by minimising\n"
            f" * f(x) = svm_score(x)^2 + lam*||x-x0||^2 using L-BFGS\n"
            f" * with Armijo backtracking. Mirrors find_nearest() in surrogate_proj.py.\n"
            f" * Returns 1 if svm_score(x_out) >= 0.0, 0 otherwise. */\n"
            f"{cu_flag} int _srgrz_project(const double *x0, double *x_out,\n"
            f"                             double lam, int maxiter, double tol)\n"
            f"{{\n"
            f"    const int N = 3;\n"
            f"    const int M = 5; /* L-BFGS memory depth */\n"
            f"    double x[3] = {{x0[0], x0[1], x0[2]}};\n"
            f"    const double h = 1e-5;\n"
            f"    double g_prev[3], s_mem[5][3], y_mem[5][3], rho[5];\n"
            f"    double alpha_ls, q[3], r[3], a_coeff[5];\n"
            f"    int k = 0, m_stored = 0;\n"
            f"\n"
            f"    /* Evaluate initial objective and gradient */\n"
            f"    double s = svm_score(x);\n"
            f"    double obj = s*s;\n"
            f"    double sg[3], g[3];\n"
            f"    _srgrz_svm_grad(x, h, sg);\n"
            f"    for (int i = 0; i < N; i++) {{\n"
            f"        obj += lam*(x[i]-x0[i])*(x[i]-x0[i]);\n"
            f"        g[i] = 2.0*s*sg[i] + 2.0*lam*(x[i]-x0[i]);\n"
            f"    }}\n"
            f"\n"
            f"    for (int iter = 0; iter < maxiter; iter++) {{\n"
            f"        /* L-BFGS two-loop recursion to get search direction r = -H*g */\n"
            f"        for (int i = 0; i < N; i++) q[i] = g[i];\n"
            f"        int bound = m_stored;\n"
            f"        for (int j = bound - 1; j >= 0; j--) {{\n"
            f"            int idx = (k - 1 - (bound - 1 - j) + M * M) % M;\n"
            f"            double dot = 0;\n"
            f"            for (int i = 0; i < N; i++) dot += s_mem[idx][i] * q[i];\n"
            f"            a_coeff[j] = rho[idx] * dot;\n"
            f"            for (int i = 0; i < N; i++) q[i] -= a_coeff[j] * y_mem[idx][i];\n"
            f"        }}\n"
            f"        /* Scale: H0 = (s'y)/(y'y) * I */\n"
            f"        double gamma_lbfgs = 1.0;\n"
            f"        if (m_stored > 0) {{\n"
            f"            int idx = (k - 1 + M) % M;\n"
            f"            double sy = 0, yy = 0;\n"
            f"            for (int i = 0; i < N; i++) {{\n"
            f"                sy += s_mem[idx][i] * y_mem[idx][i];\n"
            f"                yy += y_mem[idx][i] * y_mem[idx][i];\n"
            f"            }}\n"
            f"            if (yy > 0) gamma_lbfgs = sy / yy;\n"
            f"        }}\n"
            f"        for (int i = 0; i < N; i++) r[i] = gamma_lbfgs * q[i];\n"
            f"        for (int j = 0; j < bound; j++) {{\n"
            f"            int idx = (k - bound + j + M * M) % M;\n"
            f"            double dot = 0;\n"
            f"            for (int i = 0; i < N; i++) dot += y_mem[idx][i] * r[i];\n"
            f"            double beta = rho[idx] * dot;\n"
            f"            for (int i = 0; i < N; i++) r[i] += s_mem[idx][i] * (a_coeff[j] - beta);\n"
            f"        }}\n"
            f"        /* Search direction = -r */\n"
            f"        for (int i = 0; i < N; i++) r[i] = -r[i];\n"
            f"\n"
            f"        /* Armijo backtracking line search */\n"
            f"        double dg = 0;\n"
            f"        for (int i = 0; i < N; i++) dg += g[i] * r[i];\n"
            f"        alpha_ls = 1.0;\n"
            f"        for (int ls = 0; ls < 40; ls++) {{\n"
            f"            double xn[3] = {{x[0]+alpha_ls*r[0], x[1]+alpha_ls*r[1], x[2]+alpha_ls*r[2]}};\n"
            f"            double sn = svm_score(xn);\n"
            f"            double obj_n = sn*sn;\n"
            f"            for (int i = 0; i < N; i++) obj_n += lam*(xn[i]-x0[i])*(xn[i]-x0[i]);\n"
            f"            if (obj_n <= obj + 1e-4*alpha_ls*dg) break;\n"
            f"            alpha_ls *= 0.5;\n"
            f"        }}\n"
            f"\n"
            f"        /* Update x and store L-BFGS vectors */\n"
            f"        for (int i = 0; i < N; i++) g_prev[i] = g[i];\n"
            f"        double step2 = 0.0;\n"
            f"        int sidx = k % M;\n"
            f"        for (int i = 0; i < N; i++) {{\n"
            f"            double dx = alpha_ls * r[i];\n"
            f"            s_mem[sidx][i] = dx;\n"
            f"            x[i] += dx;\n"
            f"            step2 += dx*dx;\n"
            f"        }}\n"
            f"        if (step2 < tol*tol) break;\n"
            f"\n"
            f"        /* Recompute objective and gradient at new x */\n"
            f"        s = svm_score(x);\n"
            f"        obj = s*s;\n"
            f"        _srgrz_svm_grad(x, h, sg);\n"
            f"        for (int i = 0; i < N; i++) {{\n"
            f"            obj += lam*(x[i]-x0[i])*(x[i]-x0[i]);\n"
            f"            g[i] = 2.0*s*sg[i] + 2.0*lam*(x[i]-x0[i]);\n"
            f"        }}\n"
            f"\n"
            f"        /* y = g_new - g_old */\n"
            f"        double sy = 0;\n"
            f"        for (int i = 0; i < N; i++) {{\n"
            f"            y_mem[sidx][i] = g[i] - g_prev[i];\n"
            f"            sy += s_mem[sidx][i] * y_mem[sidx][i];\n"
            f"        }}\n"
            f"        if (sy > 1e-20) {{\n"
            f"            rho[sidx] = 1.0 / sy;\n"
            f"            k++;\n"
            f"            if (m_stored < M) m_stored++;\n"
            f"        }}\n"
            f"    }}\n"
            f"    x_out[0]=x[0]; x_out[1]=x[1]; x_out[2]=x[2];\n"
            f"    return (svm_score(x_out) >= 0.0) ? 1 : 0;\n"
            f"}}\n\n"

            f"{cu_flag}double *{func_prefix}_grid(double *out)\n"
            f"{{\n"
            f"    const {type_name} *w = &SRGRZ_WEIGHTS;\n"
            f"    for (int i = 0; i < SRGRZ_N_MU; i++) {{\n"
            f"        out[i] = w->MU_GRID[i];\n"
            f"    }}\n"
            f"    return out;\n"
            f"}}\n\n"
            
            f"{cu_flag}void {func_prefix}_interp(const double *vcut, const double *mu_new, int n, double mu_ref, double *out)\n"
            f"{{\n"
            f"    const {type_name} *w = &SRGRZ_WEIGHTS;\n"
            f"    int ng = SRGRZ_N_MU;\n"
            f"    for (int i = 0; i < n; i++) {{\n"
            f"        double mu = mu_new[i]/mu_ref;\n"
            f"        if (mu <= w->MU_GRID[0])          {{ out[i] = vcut[0];          continue; }}\n"
            f"        if (mu >= w->MU_GRID[ng - 1])     {{ out[i] = vcut[ng - 1];     continue; }}\n"
            f"        /* binary search for the bracketing interval */\n"
            f"        int lo = 0, hi = ng - 1;\n"
            f"        while (hi - lo > 1) {{\n"
            f"            int mid = (lo + hi) >> 1;\n"
            f"            if (w->MU_GRID[mid] <= mu) lo = mid; else hi = mid;\n"
            f"        }}\n"
            f"        double t = (mu - w->MU_GRID[lo]) / (w->MU_GRID[hi] - w->MU_GRID[lo]);\n"
            f"        out[i] = vcut[lo] + t * (vcut[hi] - vcut[lo]);\n"
            f"    }}\n"
            f"}}\n\n"
            
            f"{cu_flag} int {func_prefix}_project(double alpha, double gamma, double phi, double *alpha_proj, double *gamma_proj, double *phi_proj)\n"
            f"{{\n"
            f"    /* If already converged, return original point unchanged (matches Python). */\n"
            f"    if ({func_prefix}_converged(alpha, gamma, phi)) {{\n"
            f"        *alpha_proj = alpha;\n"
            f"        *gamma_proj = gamma;\n"
            f"        *phi_proj = phi;\n"
            f"        return 1;\n"
            f"    }}\n"
            f"    double x0[3] = {{alpha, gamma, phi}};\n"
            f"    double xp[3];\n"
            f"    int converged = _srgrz_project(x0, xp, 1e-3, 500, 1e-6);\n"
            f"    *alpha_proj = xp[0];\n"
            f"    *gamma_proj = xp[1];\n"
            f"    *phi_proj = xp[2];\n"
            f"    return converged;\n"
            f"}}\n\n"

            f"{cu_flag}void {func_prefix}_eval(const double *mu_new, int n, double mu_ref, double alpha, double gamma, double phi, double *out)\n"
            f"{{\n"
            f"    double vcut[SRGRZ_N_MU];\n"
            f"    {func_prefix}_predict(alpha, gamma, phi, vcut);\n"
            f"    {func_prefix}_interp(vcut, mu_new, n, mu_ref, out);\n"
            f"}}\n\n"

            f"{cu_flag}void {func_prefix}_proj_eval(const double *mu_new, int n, double mu_ref,\n"
            f"    double alpha, double gamma, double phi, double *out)\n"
            f"{{\n"
            f"    double xp[3];\n"
            f"    {func_prefix}_project(alpha, gamma, phi, &xp[0], &xp[1], &xp[2]);\n"
            f"    {func_prefix}_eval(mu_new, n, mu_ref, xp[0], xp[1], xp[2], out);\n"
            f"}}\n\n"

            f"{cu_flag}void {func_prefix}_eval_physical(const double *mu_new, int n, double phi, double phi_wall, double density,\n"
            f"    double temperature, double bmag, double impact_angle, double *out)\n"
            f"{{\n"
            f"    double muref = temperature / bmag;\n"
            f"    double gamma   = (1.0 / bmag) * sqrt({cons_prefix}_ELECTRON_MASS * density / {cons_prefix}_EPSILON0);\n"
            f"    double phinorm = ({cons_prefix}_ELEMENTARY_CHARGE * (phi - phi_wall)) / temperature;\n"
            f"    double alpha = impact_angle * 180/{cons_prefix}_PI;\n"
            f"    {func_prefix}_eval(mu_new, n, muref, alpha, gamma, phinorm, out);\n"
            f"}}\n"
        
            f"{cu_flag}void {func_prefix}_eval_physical_vcut_fact(const double *mu_new,  int n, double phi, double phi_wall,\n"
            f"    double density, double temperature, double q2Dm, double bmag, double impact_angle, double *out)\n"
            f"{{\n"
            f"    double muref = temperature / bmag;\n"
            f"    double gamma   = (1.0 / bmag) * sqrt({cons_prefix}_ELECTRON_MASS * density / {cons_prefix}_EPSILON0);\n"
            f"    double phinorm = ({cons_prefix}_ELEMENTARY_CHARGE * (phi - phi_wall)) / temperature;\n"
            f"    double alpha = impact_angle * 180/{cons_prefix}_PI;\n\n"
            f"    if (phinorm > 0.05) {{\n"
            f"      {func_prefix}_eval(mu_new, n, muref, alpha, gamma, phinorm, out);\n\n"
            
            f"      double vcut_const_sq = -q2Dm * (phi - phi_wall);\n"
            f"      double vte_sq = temperature / {cons_prefix}_ELECTRON_MASS;\n"
            f"      for (int i = 0; i < n; i++)\n"
            f"        out[i] = pow(out[i],2) / (2*phinorm);\n"
            f"    }} else {{\n"
            f"      for (int i = 0; i < n; i++)\n"
            f"        out[i] = 1.0;\n"
            f"    }}\n"
            f"}}\n"
            
            f"{cu_flag}void {func_prefix}_eval_proj_physical_vcut_fact(const double *mu_new,  int n, double phi, double phi_wall,\n"
            f"    double density, double temperature, double q2Dm, double bmag, double impact_angle, double *out)\n"
            f"{{\n"
            f"    double gamma = (1.0 / bmag) * sqrt({cons_prefix}_ELECTRON_MASS * density / {cons_prefix}_EPSILON0);\n"
            f"    double phinorm = ({cons_prefix}_ELEMENTARY_CHARGE * (phi - phi_wall)) / temperature;\n"
            f"    double alpha = impact_angle * 180/{cons_prefix}_PI;\n"
            f"    double muref = temperature / bmag;\n\n"
            f"    if (phinorm > 0.05) {{\n"
            f"      if (!{func_prefix}_converged(alpha, gamma, phinorm)) {{\n"
            f"        double xp[3];\n"
            f"        {func_prefix}_project(alpha, gamma, phinorm, &xp[0], &xp[1], &xp[2]);\n"
            f"        alpha = xp[0];\n"
            f"        gamma = xp[1];\n"
            f"        phinorm = xp[2];\n"
            f"      }}\n"
            
            f"      {func_prefix}_eval(mu_new, n, muref, alpha, gamma, phinorm, out);\n\n"
            
            f"      double vcut_const_sq = -q2Dm * (phi - phi_wall);\n"
            f"      double vte_sq = temperature / {cons_prefix}_ELECTRON_MASS;\n"
            f"      for (int i = 0; i < n; i++)\n"
            f"        out[i] = pow(out[i],2) / (2*phinorm);\n"
            f"    }} else {{\n"
            f"      for (int i = 0; i < n; i++)\n"
            f"        out[i] = 1.0;\n"
            f"    }}\n"
            f"}}\n"        
        )

        # ── Assemble .h ───────────────────────────────────────────────────────────               
        h_source = (
            f"/* {h_fname}  –  GYRAZE surrogate model public API generated from {REPO_INFO} */\n"
            
            f"{header_head}"
            
            f"/* Number of points in the fixed mu-grid. */\n"
            f"#define SRGRZ_N_MU {n_mu}\n\n"

            f"{struct_def}"
            f"{extern_c_beg}"
            
            f"/**\n"
            f" * Returns 1 if GYRAZE is predicted to converge, 0 otherwise.\n"
            f" *\n"
            f" * @param alpha: impact angle in degrees\n"
            f" * @param gamma: normalised plasma density parameter\n"
            f" * @param phi:   normalised sheath potential drop (e * (phi - phi_wall) / T_e)\n"
            f" */\n"
            f"{cu_flag}int {func_prefix}_converged(double alpha, double gamma, double phi);\n\n"
            
            f"/**\n"
            f" * Runs the NN regression; writes SRGRZ_N_MU predicted v_par_cut values into out[].\n"
            f" *\n"
            f" * @param alpha: impact angle in degrees\n"
            f" * @param gamma: normalised plasma density parameter\n"
            f" * @param phi:   normalised sheath potential drop (e * (phi - phi_wall) / T_e)\n"
            f" */\n"            
            f"{cu_flag}void {func_prefix}_predict(double alpha, double gamma, double phi, double out[SRGRZ_N_MU]);\n\n"
            
            f"/**\n"
            f" * Copies the SRGRZ_N_MU-element mu-grid into out[] and returns out.\n"
            f" *\n"
            f" * @param out: output array of size SRGRZ_N_MU\n"
            f" */\n"            
            f"{cu_flag}double *{func_prefix}_grid(double *out);\n\n"
            
            f"/**\n"
            f" * Linear interpolation of vcut[SRGRZ_N_MU] (on the fixed mu-grid) onto\n"
            f" * mu_new[n]; results are written into out[n]. Clamps at the grid boundaries.\n"
            f" *\n"
            f" * @param vcut:    input array of size SRGRZ_N_MU containing values at the fixed mu-grid\n"
            f" * @param mu_new:  input array of size n containing the new mu points\n"
            f" * @param n:       number of points in mu_new and out\n"
            f" * @param mu_ref:  reference mu value for normalisation (e.g. temperature / Bmag)\n"
            f" * @param out:     output array of size n where interpolated values are written\n"
            f" */\n"
            f"{cu_flag}void {func_prefix}_interp(const double *vcut, const double *mu_new, int n, double mu_ref, double *out);\n\n"
            
            f"/**\n"
            f" * Projects (alpha, gamma, phi) onto the nearest convergent point in parameter space.\n"
            f" * The projection minimises svm_score(x)^2 + 1e-3*||x-x0||^2 via gradient descent\n"
            f" * with Armijo backtracking (mirrors find_nearest() in surrogate_proj.py).\n"
            f" * Returns 1 if the projected point is convergent, 0 otherwise.\n"
            f" *\n"
            f" * @param alpha: impact angle in degrees\n"
            f" * @param gamma: normalised plasma density parameter\n"
            f" * @param phi:   normalised sheath potential drop (e * (phi - phi_wall) / T_e)\n"
            f" * @param alpha_proj: pointer to where the projected impact angle (degrees) is written\n"
            f" * @param gamma_proj: pointer to where the projected gamma is written\n"
            f" * @param phi_proj:   pointer to where the projected phi is written\n"
            f" */\n"
            f"{cu_flag}int {func_prefix}_project(double alpha, double gamma, double phi,\n"
            f"                             double *alpha_proj, double *gamma_proj, double *phi_proj);\n\n"
            
            f"/**\n"
            f" * Returns the prediction of a custom mu grid of size n\n"
            f" *\n"
            f" * @param mu_new:  input array of size n containing the new mu points\n"
            f" * @param n:       number of points in mu_new and out\n"
            f" * @param mu_ref:  reference mu value for normalisation (e.g. temperature / Bmag)\n"
            f" * @param alpha:   impact angle in degrees\n"
            f" * @param gamma:   normalised plasma density parameter\n"
            f" * @param phi:     normalised sheath potential drop (e * (phi - phi_wall) / T_e)\n"
            f" * @param out:     output array of size n where interpolated values are written\n"
            f" */\n"
            f"{cu_flag}void {func_prefix}_eval(const double *mu_new, int n, double mu_ref, double alpha, double gamma, double phi, double *out);\n\n"

            f"/**\n"
            f" * Like {func_prefix}_eval, but projects (alpha, gamma, phi) onto the nearest\n"
            f" * convergent point in parameter space when GYRAZE is predicted not to converge.\n"
            f" * The projection minimises svm_score(x)^2 + 1e-3*||x-x0||^2 via gradient\n"
            f" * descent with Armijo backtracking (mirrors find_nearest() in surrogate_proj.py).\n"
            f" *\n"
            f" * @param mu_new:  input array of size n containing the new mu points\n"
            f" * @param n:       number of points in mu_new and out\n"
            f" * @param mu_ref:  reference mu value for normalisation (e.g. temperature / Bmag)\n"
            f" * @param alpha:   impact angle in degrees\n"
            f" * @param gamma:   normalised plasma density parameter\n"
            f" * @param phi:     normalised sheath potential drop (e * (phi - phi_wall) / T_e)\n"
            f" * @param out:     output array of size n where interpolated values are written\n"
            f" */\n"
            f"{cu_flag}void {func_prefix}_proj_eval(const double *mu_new, int n, double mu_ref,\n"
            f"    double alpha, double gamma, double phi, double *out);\n\n"

            f"/**\n"
            f" * Converts from physical parameters and evaluates on a custom mu grid.\n"
            f" * Conversion formulas:\n"
            f" *   munorm  = mu*Bmag / temperature\n"
            f" *   gamma   = (1/Bmag) * sqrt(m_e * density / eps0)\n"
            f" *   phinorm = e * phi / temperature\n"
            f" *\n"
            f" * @param mu_new:  input array of size n containing the new mu points\n"
            f" * @param n:       number of points in mu_new and out\n"
            f" * @param phi:     sheath potential (V)\n"
            f" * @param phi_wall: wall potential (V)\n"
            f" * @param density:  electron density (m^-3)\n"
            f" * @param temperature:  electron temperature (eV)\n"
            f" * @param bmag:    magnetic field strength (T)\n"
            f" * @param impact_angle: magnetic impact angle (radians)\n"
            f" */\n"
            f"{cu_flag}void {func_prefix}_eval_physical(const double *mu_new, int n, double phi, double phi_wall,\n"
            f"    double density, double temperature, double bmag, double impact_angle, double *out);\n\n"
            
            f"/**\n"
            f" * Same as {func_prefix}_eval_physical, but normalises output by sqrt(2 * e * (phi - phi_wall) / mass)\n"
            f" *\n"
            f" * @param mu_new:  input array of size n containing the new mu points\n"
            f" * @param n:       number of points in mu_new and out\n"
            f" * @param phi:     sheath potential (V)\n"
            f" * @param phi_wall: wall potential (V)\n"
            f" * @param density:  electron density (m^-3)\n"
            f" * @param temperature:  electron temperature (eV)\n"
            f" * @param q2Dm:     2 x charge-to-mass ratio (C/kg)\n"
            f" * @param bmag:    magnetic field strength (T)\n"
            f" * @param impact_angle: magnetic impact angle (radians)\n"
            f" */\n"
            f"{cu_flag}void {func_prefix}_eval_physical_vcut_fact(const double *mu_new, int n, double phi, double phi_wall,\n"
            f"    double density, double temperature, double q2Dm, double bmag, double impact_angle, double *out);\n\n"
            
            f"/**\n"
            f" * Same as srgrz_eval_physical_vcut_fact, but normalises return 0 if gyraze is not converging.\n"
            f" *\n"
            f" * @param mu_new:  input array of size n containing the new mu points\n"
            f" * @param n:       number of points in mu_new and out\n"
            f" * @param phi:     sheath potential (V)\n"
            f" * @param phi_wall: wall potential (V)\n"
            f" * @param density:  electron density (m^-3)\n"
            f" * @param temperature:  electron temperature (eV)\n"
            f" * @param q2Dm:     2 x charge-to-mass ratio (C/kg)\n"
            f" * @param bmag:    magnetic field strength (T)\n"
            f" * @param impact_angle: magnetic impact angle (radians)\n"
            f" */\n"
            f"{cu_flag}void {func_prefix}_eval_proj_physical_vcut_fact(const double *mu_new, int n, double phi, double phi_wall,\n"
            f"    double density, double temperature, double q2Dm, double bmag, double impact_angle, double *out);\n\n"
            f"{header_tail}"
        )
        os.makedirs(outdir, exist_ok=True)
        c_path = os.path.join(outdir, c_fname)
        h_path = os.path.join(outdir, h_fname)
        with open(c_path, "w") as f:
            f.write(c_source)
        with open(h_path, "w") as f:
            f.write(h_source)
        
        print(f"Generated {c_path} and {h_path}")

    # ── Assemble test_surrogate.c ─────────────────────────────────────────────
    test_source = (
        f'#include <stdio.h>\n'
        f'#include <stdlib.h>\n'
        f'#include <string.h>\n'
        f'#include "{output_name}.h"\n\n'
        f'int main(int argc, char *argv[])\n'
        f'{{\n'
        f'    double alpha = 4.0, gamma = 1.0, phi = 2.5, mu = 1.0;\n'
        f'    double phi_wall = 0.0, density = 1e19, temperature = 100.0;\n'
        f'    double q2Dm = -2*SRGRZ_ELEMENTARY_CHARGE/SRGRZ_ELECTRON_MASS, bmag = 1.0, impact_angle = 0.0;\n'
        f'    double out[{out_dim}];\n\n'
        f'    if (argc < 2) {{\n'
        f'        fprintf(stderr, "Usage: %s {{predict|eval|physical|physical_vcut_fact|physical_proj_vcut_fact}} [args...]\\n", argv[0]);\n'
        f'        fprintf(stderr, "\\n  predict [alpha [gamma [phi]]]:\\n");\n'
        f'        fprintf(stderr, "    Evaluate full surrogate at (alpha, gamma, phi)\\n");\n'
        f'        fprintf(stderr, "    Returns {out_dim} values (one for each mu grid point)\\n");\n'
        f'        fprintf(stderr, "\\n  eval mu [alpha [gamma [phi]]]:\\n");\n'
        f'        fprintf(stderr, "    Evaluate surrogate at single mu point with (alpha, gamma, phi)\\n");\n'
        f'        fprintf(stderr, "    mu: scalar value (not a grid)\\n");\n'
        f'        fprintf(stderr, "\\n  physical mu [phi [phi_wall [density [temperature [q2Dm [bmag [impact_angle]]]]]]]:\\n");\n'
        f'        fprintf(stderr, "    Evaluate at single mu using physical parameters\\n");\n'
        f'        fprintf(stderr, "\\n  physical_vcut_fact mu [phi [phi_wall [density [temperature [q2Dm [bmag [impact_angle]]]]]]]:\\n");\n'
        f'        fprintf(stderr, "    Evaluate at single mu using physical parameters and normalise by sqrt(e * (phi - phi_wall) / T_e)\\n");\n'
        f'        fprintf(stderr, "\\n  proj_physical_vcut_fact mu [phi [phi_wall [density [temperature [q2Dm [bmag [impact_angle]]]]]]]:\\n");\n'
        f'        fprintf(stderr, "    Same as physical_vcut_fact but projects (alpha, gamma, phi) onto nearest convergent point if not converging\\n");\n'
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
        f'        srgrz_eval(&mu, 1, 1.0, alpha, gamma, phi, &result);\n\n'
        f'        printf("out = %.6f\\n", result);\n\n'
        f'    }}\n'
        f'    else if (strcmp(argv[1], "physical") == 0) {{\n'
        f'        double result;\n\n'
        f'        /* Parse arguments */\n'
        f'        if (argc > 2) mu = atof(argv[2]);\n'
        f'        if (argc > 3) phi = atof(argv[3]);\n'
        f'        if (argc > 4) phi_wall = atof(argv[4]);\n'
        f'        if (argc > 5) density = atof(argv[5]);\n'
        f'        if (argc > 6) temperature = atof(argv[6]);\n'
        f'        if (argc > 7) bmag = atof(argv[7]);\n'
        f'        if (argc > 8) impact_angle = atof(argv[8]);\n\n'
        f'        srgrz_eval_physical(&mu, 1, phi, phi_wall, density, temperature, bmag, impact_angle, &result);\n\n'
        f'        printf("out = %.6f\\n", result);\n\n'
        f'    }}\n'
        f'    else if (strcmp(argv[1], "physical_vcut_fact") == 0) {{\n'
        f'        double result;\n\n'
        f'        /* Parse arguments */\n'
        f'        if (argc > 2) mu = atof(argv[2]);\n'
        f'        if (argc > 3) phi = atof(argv[3]);\n'
        f'        if (argc > 4) phi_wall = atof(argv[4]);\n'
        f'        if (argc > 5) density = atof(argv[5]);\n'
        f'        if (argc > 6) temperature = atof(argv[6]);\n'
        f'        if (argc > 7) q2Dm = atof(argv[7]);\n'
        f'        if (argc > 8) bmag = atof(argv[8]);\n'
        f'        if (argc > 9) impact_angle = atof(argv[9]);\n\n'
        f'        srgrz_eval_physical_vcut_fact(&mu, 1, phi, phi_wall, density, temperature, q2Dm, bmag, impact_angle, &result);\n\n'
        f'        printf("out = %.6f\\n", result);\n\n'
        f'    }} else if (strcmp(argv[1], "proj_physical_vcut_fact") == 0) {{\n'
        f'        double result;\n\n'
        f'        /* Parse arguments */\n'
        f'        if (argc > 2) mu = atof(argv[2]);\n'
        f'        if (argc > 3) phi = atof(argv[3]);\n'
        f'        if (argc > 4) phi_wall = atof(argv[4]);\n'
        f'        if (argc > 5) density = atof(argv[5]);\n'
        f'        if (argc > 6) temperature = atof(argv[6]);\n'
        f'        if (argc > 7) q2Dm = atof(argv[7]);\n'
        f'        if (argc > 8) bmag = atof(argv[8]);\n'
        f'        if (argc > 9) impact_angle = atof(argv[9]);\n\n'
        f'        srgrz_eval_proj_physical_vcut_fact(&mu, 1, phi, phi_wall, density, temperature, q2Dm, bmag, impact_angle, &result);\n\n'
        f'        printf("out = %.6f\\n", result);\n\n'
        f'    }} else {{\n'
        f'        fprintf(stderr, "Unknown mode: %s\\n", argv[1]);\n'
        f'        fprintf(stderr, "Use \'predict\', \'eval\', \'physical\', \'physical_vcut_fact\', or \'proj_physical_vcut_fact\'\\n");\n'
        f'        return 1;\n'
        f'    }}\n\n'
        f'    return 0;\n'
        f'}}\n'
    )
    
    test_proj_source = (
        f'#include <stdio.h>\n'
        f'#include <stdlib.h>\n'
        f'#include <string.h>\n'
        f'#include "{output_name}.h"\n\n'
        f'int main(int argc, char *argv[])\n'
        f'{{\n'
        f'    double alpha = 4.0, gamma = 1.0, phi = 2.5;\n'
        f'    if (argc < 2) {{\n'
        f'        fprintf(stderr, "Usage: %s {{project}} [args...]\\n", argv[0]);\n'
        f'        fprintf(stderr, "\\n  project [alpha [gamma [phi]]]:\\n");\n'
        f'        fprintf(stderr, "    Project (alpha, gamma, phi) onto nearest convergent point and evaluate at the projection\\n");\n'
        f'        return 1;\n'
        f'    }}\n\n'
        f'    if (strcmp(argv[1], "project") == 0) {{\n'
        f'        /* Parse optional arguments */\n'
        f'        if (argc > 2) alpha = atof(argv[2]);\n'
        f'        if (argc > 3) gamma = atof(argv[3]);\n'
        f'        if (argc > 4) phi = atof(argv[4]);\n\n'
        f"        double xp[3];\n"
        f"        int converged = srgrz_project(alpha, gamma, phi, &xp[0], &xp[1], &xp[2]);\n"
        f"        if (!converged) {{\n"
        f"            fprintf(stderr, \"Warning: Projection did not converge for (alpha, gamma, phi) = (%f, %f, %f)\\n\", alpha, gamma, phi);\n"
        f"        }}\n"
        f"        printf(\"Projected (alpha, gamma, phi) = (%f, %f, %f)\\n\", xp[0], xp[1], xp[2]);\n"
        f'    }} else {{\n'
        f'        fprintf(stderr, "Unknown mode: %s\\n", argv[1]);\n'
        f'        fprintf(stderr, "Use \'project\'\\n");\n'
        f'        return 1;\n'
        f'    }}\n\n'
        f'    return 0;\n'
        f'}}\n'
    )

    # ── Assemble Makefile ─────────────────────────────────────────────────────
    makefile_source = (
        f"CC = gcc\n"
        f"CFLAGS = -O3 -Wall -Wextra\n"
        f"LDLIBS = -lm\n\n"
        f"SURROGATE_SRC = {output_name}.c\n"
        f"SURROGATE_OBJ = $(SURROGATE_SRC:.c=.o)\n"
        f"TEST_SRGRZ_OBJ = test_surrogate.o\n"
        f"TEST_PROJ_OBJ = test_projection.o\n\n"
        f"EXECS = test_surrogate test_projection\n\n"
        f"all: $(EXECS)\n\n"
        f"test_surrogate: $(TEST_SRGRZ_OBJ) $(SURROGATE_OBJ)\n"
        f"\t$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)\n\n"
        f"test_projection: $(TEST_PROJ_OBJ) $(SURROGATE_OBJ)\n"
        f"\t$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)\n\n"
        f"%.o: %.c {output_name}.h\n"
        f"\t$(CC) $(CFLAGS) -c $<\n\n"
        f"clean:\n"
        f"\trm -f $(TEST_SRGRZ_OBJ) $(TEST_PROJ_OBJ) $(SURROGATE_OBJ) $(EXECS)\n"
    )

    # ── Write test code ───────────────────────────────────────────────────────────
    test_path = os.path.join(output_dir, "test_surrogate.c")
    test_proj_path = os.path.join(output_dir, "test_projection.c")
    make_path = os.path.join(output_dir, "Makefile")

    with open(test_path, "w") as f:
        f.write(test_source)
    with open(test_proj_path, "w") as f:
        f.write(test_proj_source)
    with open(make_path, "w") as f:
        f.write(makefile_source)

    print(f"Generated {test_path}, {test_proj_path}, {make_path}")
    print(f"Build: cd {output_dir} && make")


# ── Allow running directly: python export_to_c.py ────────────────────────────

if __name__ == "__main__":
    generate_c_code()
