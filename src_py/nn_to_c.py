
"""
nn_to_c.py
==========
Encapsulates everything needed to lower one trained PyTorch NN into C code:
weights, normalization, and the generated forward-pass source fragments.

One instance = one NN.  Pass two instances to generate_c_code to produce the
"total surrogate" that routes converged / non-converged queries to different
networks, mirroring gyraze_total_surrogate.py.
"""

import os
import torch
from torch import nn
import numpy as np


class NeuralNetwork(nn.Module):
    """Fully-connected network whose architecture is driven by layer_sizes."""
    def __init__(self, layer_sizes: list, activation: str = "silu"):
        super().__init__()
        act_map = {
            "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU,
            "sigmoid": nn.Sigmoid, "selu": nn.SELU,
            "elu": nn.ELU, "softplus": nn.Softplus,
        }
        act_cls = act_map[activation]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # no activation after the output layer
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class nnToC:
    """
    Load a single trained NN + normalisation file and produce all C code
    fragments needed to embed it in the generated surrogate library.

    Parameters
    ----------
    nn_model_path      : path to a PyTorch .pth weights file
    normalization_path : path to a .npz file with keys X_mu, X_sigma, Y_mu, Y_sigma
    mu_grid            : list/array of µ values for the fixed output grid
    struct_type_name   : C typedef name for the weight struct  (e.g. "srgrz_weights_t")
    weights_var_name   : C variable name for the struct instance (e.g. "srgrz_w")
    """

    def __init__(self, nn_model_path: str, normalization_path: str,
                 mu_grid: list, struct_type_name: str, weights_var_name: str,
                 activation: str = "silu"):

        self.nn_model_path      = nn_model_path
        self.normalization_path = normalization_path
        self.mu_grid            = list(mu_grid)
        self.n_mu               = len(self.mu_grid)
        self.struct_type_name   = struct_type_name
        self.weights_var_name   = weights_var_name
        self.activation         = activation

        self._load()
        self._build_fragments()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        state_dict = torch.load(self.nn_model_path, map_location="cpu")

        # Infer layer sizes from the weight tensor shapes in the state dict.
        # Weight matrices are stored as [out_features, in_features].
        weight_keys = sorted(
            (k for k in state_dict if k.endswith(".weight")),
            key=lambda k: int(k.split(".")[1]),
        )
        layer_sizes = [state_dict[weight_keys[0]].shape[1]] + [
            state_dict[k].shape[0] for k in weight_keys
        ]

        model = NeuralNetwork(layer_sizes=layer_sizes, activation=self.activation)
        model.load_state_dict(state_dict)
        model.eval()

        norms = np.load(self.normalization_path)
        self.X_mu    = norms["X_mu"].tolist()
        self.X_sigma = norms["X_sigma"].tolist()
        self.Y_mu    = norms["Y_mu"].tolist()
        self.Y_sigma = norms["Y_sigma"].tolist()

        self.linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
        self.layer_dims    = [(l.in_features, l.out_features) for l in self.linear_layers]
        self.n_layers      = len(self.linear_layers)
        self.out_dim       = self.layer_dims[-1][1]

    # ── C fragment builders ───────────────────────────────────────────────────

    def _build_fragments(self):
        """Pre-compute every C source fragment that callers will need."""

        tn   = self.struct_type_name
        vn   = "w"
        nd   = self.n_layers
        od   = self.out_dim
        ld   = self.layer_dims
        n_mu = self.n_mu

        # --- struct typedef ---
        lines = ["/* --- weight storage type --- **/", f"typedef struct {{"]
        for i, (in_d, out_d) in enumerate(ld):
            lines.append(f"  double W{i}[{in_d * out_d}], b{i}[{out_d}];")
        lines.append(f"  double Y_mu[{od}], Y_sigma[{od}];")
        lines.append(f"  double MU_GRID[{n_mu}];")
        lines.append(f"}} {tn};\n")
        self.struct_def = "\n".join(lines) + "\n"

        # --- struct initialiser fields ---
        field_lines = []
        for i, layer in enumerate(self.linear_layers):
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            field_lines.append(f"  /* Layer {i} weights ({layer.out_features} x {layer.in_features}) */")
            field_lines.append(f"  .W{i} = {{{', '.join(f'{v:.8f}f' for v in W.flatten())}}},")
            field_lines.append(f"  .b{i} = {{{', '.join(f'{v:.8f}f' for v in b.flatten())}}},")
        field_lines.append(f"  .Y_mu    = {{{', '.join(f'{v:.8f}f' for v in self.Y_mu)}}},")
        field_lines.append(f"  .Y_sigma = {{{', '.join(f'{v:.8f}f' for v in self.Y_sigma)}}},")
        field_lines.append(f"  .MU_GRID = {{{', '.join(f'{v:.8f}f' for v in self.mu_grid)}}}")
        self._struct_fields = "\n".join(field_lines)

        # --- local variable declarations for the forward pass ---
        buf = f"    const {self.struct_type_name} *{vn} = &{self.weights_var_name.upper()};\n"
        buf +=f"    double x[3] = {{alpha, gamma, phi}};\n"
        buf +=f"    double h0[3];\n"
        for i, (_, out_d) in enumerate(ld):
            buf += f"    double h{i+1}[{out_d}];\n"
        self.loc_decls = buf

        # --- input normalisation ---
        norm = "    /* input normalisation */\n"
        for k in range(3):
            norm += f"    h0[{k}] = (x[{k}] - {self.X_mu[k]:.8f}f) / {self.X_sigma[k]:.8f}f;\n"
        self.norm_code = norm

        # --- dense layers ---
        layer_code = ""
        for i, (in_d, out_d) in enumerate(ld):
            pfx = f"{vn}->"
            layer_lines = [f"    /* --- layer {i} --- */"]
            for o in range(out_d):
                acc = f"{pfx}b{i}[{o}]"
                for k in range(in_d):
                    acc += f" + {pfx}W{i}[{o}*{in_d}+{k}]*h{i}[{k}]"
                layer_lines.append(f"    h{i+1}[{o}] = {acc};")
            is_hidden = (i < self.n_layers - 1)
            if is_hidden:
                layer_lines.append(f"    for (int _j = 0; _j < {out_d}; _j++)")
                layer_lines.append(f"        h{i+1}[_j] = h{i+1}[_j] / (1.0f + expf(-h{i+1}[_j]));")
            layer_lines.append("")
            layer_code += "\n".join(layer_lines)
        self.layer_code = layer_code

        # --- output denormalisation ---
        self.denorm_code = (
            "    /* output denormalisation */\n"
            f"    for (int _j = 0; _j < {od}; _j++)\n"
            f"        out[_j] = h{nd}[_j] * {vn}->Y_sigma[_j] + {vn}->Y_mu[_j];\n"
        )

        # --- non-negativity clamp ---
        self.filter_negative_code = (
            "    /* enforce non-negativity (vcut cannot be negative) */\n"
            f"    for (int _j = 0; _j < {od}; _j++)\n"
            f"        if (out[_j] < 0.0f) out[_j] = 0.0f;\n"
        )

    # ── Public helpers ────────────────────────────────────────────────────────

    def struct_instances_c(self, for_gkyl: bool = False) -> str:
        """Return the C static struct instance(s) initialised with all weights."""
        tn = self.struct_type_name
        vn = self.weights_var_name
        f  = self._struct_fields
        if for_gkyl:
            return (
                f"/* host copy – visible in the __host__ pass */\n"
                f"static const {tn} {vn}_h = {{\n{f}\n}};\n\n"
                f"/* device copy – visible in the __device__ pass */\n"
                f"#ifdef GKYL_HAVE_CUDA\n"
                f"__device__ static const {tn} {vn}_d = {{\n{f}\n}};\n"
                f"#endif\n\n"
                f"#ifdef __CUDA_ARCH__\n"
                f"#  define {vn.upper()} {vn}_d\n"
                f"#else\n"
                f"#  define {vn.upper()} {vn}_h\n"
                f"#endif\n\n"
            )
        else:
            return (
                f"static const {tn} {vn}_h = {{\n{f}\n}};\n"
                f"#define {vn.upper()} {vn}_h\n\n"
            )

    def predict_body_c(self) -> str:
        """
        Return the C body of a predict function using this NN.
        The caller is responsible for the function signature and braces.
        Expects a local  `double x[3]`  and writes into  `double out[]`.
        Also expects a local pointer  `const <type> *<vn>`  pointing
        to the correct weight struct instance.
        """
        return (
            self.loc_decls + "\n"
            + self.norm_code + "\n"
            + self.layer_code + "\n"
            + self.denorm_code
            + self.filter_negative_code
        )