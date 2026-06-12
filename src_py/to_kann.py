"""
to_kann.py
==========
Convert a trained PyTorch neural network (.pth) into a native KANN binary
file (.kann) compatible with Gkeyll's existing ``kann.h`` / ``kann_load()``.

No external dependencies beyond numpy and torch.  No C compiler required.

Native KANN Binary Layout (little-endian, matches Gkeyll core/minus/kann.c)
---------------------------------------------------------------------------
  char[4]        magic     = "KAN\\x01"
  int32          n_node
  Per-node records (see kad_save1 in kautodiff.c):
    For leaf nodes (n_child == 0):
      int32 ext_label, uint32 ext_flag, uint8 flag, int32 n_child(=0),
      uint8 n_d, int32[n_d] dimensions
    For op nodes (n_child > 0):
      int32 ext_label, uint32 ext_flag, uint8 flag, int32 n_child,
      uint16 op, int32[n_child] child_indices, int32 pre(-1=none),
      int32 ptr_size, bytes[ptr_size] ptr_data
  float32[n_var]   variable values (weights + biases, in node order)
  float32[n_const] constant values (none for feedforward nets)

KANN flag constants:
  KAD_VAR=0x1 (node flag), KAD_CONST=0x2 (node flag)
  KANN_F_IN=0x1, KANN_F_OUT=0x2, KANN_F_TRUTH=0x4, KANN_F_COST=0x8 (ext_flag)

KANN op codes used:
  kad_add=1, kad_mul=2, kad_cmul=3, kad_sigm=6, kad_mse=29

Network graph for a 4-layer MLP with SiLU activation (3->75->75->75->20):
  25 nodes total; weights written as float32 (NOT float64 as in old format).
  Normalization is NOT embedded – the calling C code handles it manually.

Usage
-----
    python src_py/to_kann.py \\
        --model   model/nn_model_conv_MPE.pth \\
        --output  model/nn_model_conv_MPE.kann

    # Or from Python:
    from src_py.to_kann import write_kann
    write_kann("model/nn_model_conv_MPE.pth", output_path="model/nn_model_conv_MPE.kann")
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Native KANN constants (must match core/minus/kautodiff.h and kann.h)
# ---------------------------------------------------------------------------

# Node flag byte (kad_node_t.flag)
KAD_VAR   = 0x1
KAD_CONST = 0x2

# ext_flag (kad_node_t.ext_flag) – KANN-level labels
KANN_F_IN    = 0x1
KANN_F_OUT   = 0x2
KANN_F_TRUTH = 0x4
KANN_F_COST  = 0x8

# Operator codes (kad_op_list index in kautodiff.c)
KAD_OP_ADD  = 1
KAD_OP_MUL  = 2
KAD_OP_CMUL = 3
KAD_OP_SIGM = 6
KAD_OP_TANH = 7
KAD_OP_RELU = 8
KAD_OP_MSE  = 29

KANN_MAGIC = b"KAN\x01"

SUPPORTED_ACTIVATIONS = ("relu", "tanh", "silu", "sigmoid")


# ---------------------------------------------------------------------------
# Minimal model definition – mirrors nn_to_c.py so the same .pth files load
# ---------------------------------------------------------------------------

class _NeuralNetwork(nn.Module):
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
            if i < len(layer_sizes) - 2:
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _infer_layer_sizes(state_dict: dict) -> list:
    weight_keys = sorted(
        (k for k in state_dict if k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    if not weight_keys:
        raise ValueError("No linear weight tensors found in state_dict")
    return [state_dict[weight_keys[0]].shape[1]] + [
        state_dict[k].shape[0] for k in weight_keys
    ]


# ---------------------------------------------------------------------------
# Native KANN binary serialisation helpers
# ---------------------------------------------------------------------------

def _write_leaf(f, ext_label: int, ext_flag: int, flag: int, dims: list) -> None:
    """Write a leaf node (n_child=0) record as per kad_save1."""
    f.write(struct.pack("<iIBi", ext_label, ext_flag, flag, 0))  # n_child=0
    f.write(struct.pack("<B", len(dims)))
    for d in dims:
        f.write(struct.pack("<i", d))


def _write_op(f, ext_label: int, ext_flag: int, flag: int,
              op: int, children: list, pre: int = -1, ptr_data: bytes = b"") -> None:
    """Write an operator node (n_child>0) record as per kad_save1."""
    f.write(struct.pack("<iIBi", ext_label, ext_flag, flag, len(children)))
    f.write(struct.pack("<H", op))                           # uint16 op
    for c in children:
        f.write(struct.pack("<i", c))                        # child index
    f.write(struct.pack("<i", pre))                          # pre index
    f.write(struct.pack("<i", len(ptr_data)))                # ptr_size
    if ptr_data:
        f.write(ptr_data)


def _build_and_write_graph(f, linear_layers: list, activation: str,
                           norm: dict = None) -> list:
    """
    Build a feedforward KANN computational graph and write it to *f*.

    When *norm* is provided (dict with float32 arrays X_mu, X_sigma, Y_mu,
    Y_sigma), input and output normalisation are baked into the graph as
    KAD_CONST leaf nodes.  The caller passes *raw* physical inputs to
    ``kann_apply1`` and receives *raw* physical outputs directly.

    Without *norm*, the graph accepts pre-normalised inputs and returns
    normalised outputs (legacy behaviour).

    Graph layout with normalization (SiLU, 3→75→75→75→20):

      0  : input feed  (KANN_F_IN,  raw physical input)
      1  : const -X_mu [n_in]
      2  : const 1/X_sigma [n_in]
      3  : add(0,1)  → x - X_mu
      4  : mul(3,2)  → (x-X_mu)/X_sigma
      5  : W0  var [75,3];   6: b0 var [75]
      7  : cmul(4,5);  8: add(7,6)
      9  : sigm(8);  10: mul(8,9)   ← SiLU
      ... (repeated for hidden layers) ...
      -8 : W3 var [20,75];  -7: b3 var [20]
      -6 : cmul(-9,-8);  -5: add(-6,-7)  ← raw NN output
      -4 : const Y_sigma [n_out]
      -3 : const Y_mu    [n_out]
      -2 : mul(-5,-4);  -1: add(-2,-3, KANN_F_OUT)  ← physical output
      -??: truth feed (KANN_F_TRUTH);  MSE cost (KANN_F_COST)

    Returns a list of float32 numpy arrays (one per KAD_CONST node, in
    node-index order) to be appended to the file after variable values.
    """
    nodes = []          # list of node descriptor tuples
    const_arrays = []   # float32 numpy arrays for KAD_CONST leaves, in order

    def leaf(ext_label, ext_flag, flag, dims, const_data=None):
        nodes.append(("leaf", ext_label, ext_flag, flag, dims))
        if const_data is not None:
            const_arrays.append(np.asarray(const_data, dtype=np.float32).flatten())

    def op(ext_label, ext_flag, flag, opcode, children):
        nodes.append(("op", ext_label, ext_flag, flag, opcode, children))

    n_in = linear_layers[0].weight.shape[1]
    n_out_final = linear_layers[-1].weight.shape[0]

    leaf(0, KANN_F_IN, 0, [1, n_in])   # raw physical input
    current = 0

    # Input normalisation: x_norm = (x - X_mu) / X_sigma
    if norm is not None:
        neg_mu_idx = len(nodes)
        leaf(0, 0, KAD_CONST, [n_in], -norm["X_mu"])

        inv_sigma_idx = len(nodes)
        leaf(0, 0, KAD_CONST, [n_in], 1.0 / norm["X_sigma"])

        center_idx = len(nodes)
        op(0, 0, KAD_VAR, KAD_OP_ADD, [current, neg_mu_idx])

        norm_idx = len(nodes)
        op(0, 0, KAD_VAR, KAD_OP_MUL, [center_idx, inv_sigma_idx])

        current = norm_idx

    for i, layer in enumerate(linear_layers):
        n_in_l  = layer.weight.shape[1]
        n_out_l = layer.weight.shape[0]
        is_last = (i == len(linear_layers) - 1)

        w_idx = len(nodes)
        leaf(0, 0, KAD_VAR, [n_out_l, n_in_l])   # W
        b_idx = len(nodes)
        leaf(0, 0, KAD_VAR, [n_out_l])            # b

        cmul_idx = len(nodes)
        op(0, 0, KAD_VAR, KAD_OP_CMUL, [current, w_idx])

        # Mark KANN_F_OUT on the last layer only if no output denorm follows
        add_ext = 0 if (is_last and norm is not None) else (KANN_F_OUT if is_last else 0)
        add_idx = len(nodes)
        op(0, add_ext, KAD_VAR, KAD_OP_ADD, [cmul_idx, b_idx])

        if is_last:
            current = add_idx
        elif activation == "silu":
            sigm_idx = len(nodes)
            op(0, 0, KAD_VAR, KAD_OP_SIGM, [add_idx])
            silu_idx = len(nodes)
            op(0, 0, KAD_VAR, KAD_OP_MUL,  [add_idx, sigm_idx])
            current = silu_idx
        elif activation == "tanh":
            act_idx = len(nodes)
            op(0, 0, KAD_VAR, KAD_OP_TANH, [add_idx])
            current = act_idx
        elif activation == "relu":
            act_idx = len(nodes)
            op(0, 0, KAD_VAR, KAD_OP_RELU, [add_idx])
            current = act_idx
        elif activation == "sigmoid":
            act_idx = len(nodes)
            op(0, 0, KAD_VAR, KAD_OP_SIGM, [add_idx])
            current = act_idx
        else:
            raise ValueError(f"Unsupported activation for KANN export: {activation}")

    # Output denormalisation: y_phys = y_norm * Y_sigma + Y_mu
    if norm is not None:
        y_sigma_idx = len(nodes)
        leaf(0, 0, KAD_CONST, [n_out_final], norm["Y_sigma"])

        y_mu_idx = len(nodes)
        leaf(0, 0, KAD_CONST, [n_out_final], norm["Y_mu"])

        scale_idx = len(nodes)
        op(0, 0, KAD_VAR, KAD_OP_MUL, [current, y_sigma_idx])

        shift_idx = len(nodes)
        op(0, KANN_F_OUT, KAD_VAR, KAD_OP_ADD, [scale_idx, y_mu_idx])

        current = shift_idx

    truth_idx = len(nodes)
    leaf(0, KANN_F_TRUTH, 0, [1, n_out_final])

    op(0, KANN_F_COST, KAD_VAR, KAD_OP_MSE, [current, truth_idx])

    # Write n_node then all node records
    f.write(struct.pack("<i", len(nodes)))
    for node in nodes:
        if node[0] == "leaf":
            _, ext_label, ext_flag, flag, dims = node
            _write_leaf(f, ext_label, ext_flag, flag, dims)
        else:
            _, ext_label, ext_flag, flag, opcode, children = node
            _write_op(f, ext_label, ext_flag, flag, opcode, children)

    return const_arrays


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_kann(
    model_path: str,
    output_path: str,
    activation: str = "silu",
    norm_path: str = None,
) -> None:
    """
    Load *model_path* (.pth) and write *output_path* in native KANN format
    readable by Gkeyll's ``kann_load()`` from ``core/minus/kann.h``.

    When *norm_path* (.npz) is supplied the input/output normalisation is
    baked into the graph as ``KAD_CONST`` nodes.  The resulting file is
    fully self-contained: ``kann_apply1()`` accepts raw physical inputs and
    returns raw physical outputs.  No normalization constants need to be
    hardcoded in the calling C code.

    Without *norm_path*, the graph accepts pre-normalised inputs and the
    caller must handle normalisation manually.

    Parameters
    ----------
    model_path  : Path to the PyTorch weights file (.pth).
    output_path : Destination .kann file (or a directory).
    activation  : Hidden-layer activation (default "silu").
    norm_path   : Path to .npz normalization file (keys: X_mu, X_sigma,
                  Y_mu, Y_sigma).  When given, normalization is embedded
                  into the graph; otherwise omitted.
    """
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Activation '{activation}' is not supported for native KANN export. "
            f"Choose from: {SUPPORTED_ACTIVATIONS}"
        )

    # Resolve output path
    abs_out = os.path.abspath(output_path)
    if os.path.isdir(abs_out) or output_path.rstrip("/\\") in (".", ".."):
        stem = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(abs_out, stem + ".kann")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Load PyTorch model
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    layer_sizes = _infer_layer_sizes(state_dict)
    model = _NeuralNetwork(layer_sizes=layer_sizes, activation=activation)
    model.load_state_dict(state_dict)
    model.eval()

    linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]

    # Load normalisation if requested
    norm = None
    if norm_path is not None:
        norms = np.load(norm_path)
        missing = {"X_mu", "X_sigma", "Y_mu", "Y_sigma"} - set(norms.keys())
        if missing:
            raise KeyError(f"normalization file missing keys: {missing}")
        norm = {k: norms[k].astype(np.float32).flatten()
                for k in ("X_mu", "X_sigma", "Y_mu", "Y_sigma")}

    with open(output_path, "wb") as f:
        # 1. Magic
        f.write(KANN_MAGIC)

        # 2. Computational graph (n_node + per-node records)
        const_arrays = _build_and_write_graph(f, linear_layers, activation, norm=norm)

        # 3. Variable values: float32, in node-index order (W0, b0, W1, b1, …)
        for layer in linear_layers:
            W = layer.weight.detach().numpy().astype(np.float32)
            b = layer.bias.detach().numpy().astype(np.float32)
            f.write(W.flatten(order="C").tobytes())
            f.write(b.flatten().tobytes())

        # 4. Constant values (normalization, in KAD_CONST node order)
        for arr in const_arrays:
            f.write(arr.tobytes())

    n_var   = sum(layer.weight.numel() + layer.bias.numel() for layer in linear_layers)
    n_const = sum(a.size for a in const_arrays)
    print(f"Wrote {output_path}")
    print(f"  layers    : {layer_sizes}")
    print(f"  activation: {activation}")
    print(f"  norm baked: {'yes' if norm is not None else 'no'}")
    print(f"  n_var     : {n_var},  n_const: {n_const}")
    print(f"  file size : {os.path.getsize(output_path)} bytes")


def verify_kann_numpy(model_path: str, kann_path: str,
                      x_input: np.ndarray, activation: str = "silu",
                      norm_path: str = None) -> np.ndarray:
    """
    Evaluate the .kann file with NumPy and compare against PyTorch.

    *x_input* is the raw physical input when *norm_path* is provided
    (normalization baked into the graph), or a pre-normalised input otherwise.

    Returns the numpy output of the graph.
    Raises AssertionError if the max absolute error exceeds 1e-4.
    """
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    layer_sizes = _infer_layer_sizes(state_dict)
    model = _NeuralNetwork(layer_sizes=layer_sizes, activation=activation)
    model.load_state_dict(state_dict)
    model.eval()

    norm = None
    if norm_path is not None:
        norms = np.load(norm_path)
        norm = {k: norms[k].astype(np.float32).flatten()
                for k in ("X_mu", "X_sigma", "Y_mu", "Y_sigma")}

    # PyTorch reference output
    with torch.no_grad():
        if norm is not None:
            x_n = (x_input.astype(np.float32) - norm["X_mu"]) / norm["X_sigma"]
            y_n = model(torch.tensor(x_n).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
            y_torch = y_n * norm["Y_sigma"] + norm["Y_mu"]
        else:
            y_torch = model(torch.tensor(x_input.astype(np.float32)).unsqueeze(0))
            y_torch = y_torch.squeeze(0).numpy().astype(np.float32)

    act_fns = {
        "silu":    lambda x: x * (1.0 / (1.0 + np.exp(-x))),
        "tanh":    np.tanh,
        "relu":    lambda x: np.maximum(x, 0.0),
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    }
    act_fn = act_fns[activation]

    # Parse .kann file: separate var-leaf and const-leaf shapes
    with open(kann_path, "rb") as f:
        magic = f.read(4)
        if magic != KANN_MAGIC:
            raise ValueError(f"Not a native KANN file (magic={magic!r})")
        n_node = struct.unpack("<i", f.read(4))[0]

        var_shapes   = []   # (dims, size) for KAD_VAR leaves
        const_shapes = []   # (dims, size) for KAD_CONST leaves
        for _ in range(n_node):
            ext_label = struct.unpack("<i", f.read(4))[0]
            ext_flag  = struct.unpack("<I", f.read(4))[0]
            flag      = struct.unpack("<B", f.read(1))[0]
            n_child   = struct.unpack("<i", f.read(4))[0]
            if n_child == 0:
                n_d  = struct.unpack("<B", f.read(1))[0]
                dims = list(struct.unpack(f"<{n_d}i", f.read(4 * n_d)))
                size = 1
                for d in dims:
                    size *= d
                if flag & KAD_VAR:
                    var_shapes.append((dims, size))
                elif flag & KAD_CONST:
                    const_shapes.append((dims, size))
            else:
                f.read(2)            # op (uint16)
                f.read(4 * n_child)  # child indices
                f.read(4)            # pre
                ptr_size = struct.unpack("<i", f.read(4))[0]
                if ptr_size > 0:
                    f.read(ptr_size)

        wb = []
        for dims, size in var_shapes:
            wb.append(np.frombuffer(f.read(4 * size), dtype=np.float32).reshape(dims))

        consts = []
        for dims, size in const_shapes:
            consts.append(np.frombuffer(f.read(4 * size), dtype=np.float32).reshape(dims))

    # NumPy evaluation
    # If normalization is baked in, the first 2 consts are -X_mu and 1/X_sigma;
    # the last 2 are Y_sigma and Y_mu.
    h = x_input.astype(np.float32)
    if consts and norm is not None:
        neg_mu, inv_sigma = consts[0].flatten(), consts[1].flatten()
        h = (h + neg_mu) * inv_sigma   # == (h - X_mu) / X_sigma
        y_sigma, y_mu = consts[2].flatten(), consts[3].flatten()

    n_layers = len(wb) // 2
    for i in range(n_layers):
        W = wb[2 * i]        # [n_out, n_in]
        b = wb[2 * i + 1]    # [n_out]
        h = W @ h + b
        if i < n_layers - 1:
            h = act_fn(h)

    if consts and norm is not None:
        h = h * y_sigma + y_mu

    max_err = float(np.abs(h - y_torch).max())
    print(f"  verify: max |numpy(.kann) - PyTorch| = {max_err:.2e}")
    assert max_err < 1e-4, f"Verification FAILED: max error {max_err:.2e}"
    print("  Verification PASSED")
    return h


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert a PyTorch .pth model to a native Gkeyll KANN binary file "
            "(readable by kann_load() from core/minus/kann.h)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",      "-m", required=True,
                   help="Path to .pth weights file")
    p.add_argument("--output",     "-o", required=True,
                   help="Output .kann file path (or directory)")
    p.add_argument("--activation", "-a", default="silu",
                   choices=list(SUPPORTED_ACTIVATIONS),
                   help="Hidden-layer activation function")
    p.add_argument("--verify",     "-v", action="store_true",
                   help="Run a NumPy round-trip check after writing")
    p.add_argument("--norm",       "-n", default=None,
                   help="Path to .npz normalization file (X_mu/X_sigma/Y_mu/Y_sigma); "
                        "when supplied, normalization is baked into the graph")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    write_kann(args.model, args.output, activation=args.activation,
               norm_path=args.norm)
    if args.verify:
        print("Verifying…")
        state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
        layer_sizes = _infer_layer_sizes(state_dict)
        rng = np.random.default_rng(42)
        # Use raw physical-scale input when norm is baked in, else normalised
        x_input = rng.standard_normal(layer_sizes[0]).astype(np.float32)
        out_path = args.output
        if os.path.isdir(out_path):
            stem = os.path.splitext(os.path.basename(args.model))[0]
            out_path = os.path.join(out_path, stem + ".kann")
        verify_kann_numpy(args.model, out_path, x_input,
                          activation=args.activation, norm_path=args.norm)

def generate(model_path, output_path, norm_path, activation="silu", verify=True):
    write_kann(model_path, output_path, activation=activation, norm_path=norm_path)
    if verify:
        print("Verifying…")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        layer_sizes = _infer_layer_sizes(state_dict)
        rng = np.random.default_rng(42)
        x_input = rng.standard_normal(layer_sizes[0]).astype(np.float32)
        verify_kann_numpy(model_path, output_path, x_input,
                          activation=activation, norm_path=norm_path)

if __name__ == "__main__":
    main()
