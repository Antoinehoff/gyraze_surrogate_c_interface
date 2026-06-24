"""
to_kann_grid.py
===============
Same as ``to_kann.py`` (convert a trained PyTorch ``.pth`` into a native
Gkeyll KANN binary ``.kann`` file) **but additionally bakes a fixed grid into
the model as extra static outputs**.

Why
---
The surrogate networks output their values sampled on a fixed ``mu``-grid
(``muvec`` in ``src_py/gyraze_conv_surrogate.py``).  Today that grid lives only
in Python and the C side has to hardcode it.  This script embeds the grid
straight into the ``.kann`` file so ``kann_load()`` / ``kann_apply1()`` hands
it back automatically — no duplication, no risk of drift.

How (no new ops required)
-------------------------
KANN serialises every weight and bias, and the op set understood by the
existing ``kann_load()`` here is only add/mul/cmul/sigm/tanh/relu/mse — there
is no concatenation op.  So instead of building a parallel branch and
concatenating it, we **augment the final linear layer**:

      out_features:  n_out  ->  n_out + n_grid
      W_last:        [n_out, width]      ->  [[W_last      ],   (real rows)
                                              [0 ... 0      ]]   (grid rows, zero)
      b_last:        [n_out]             ->  [b_last, grid]

Because the grid rows have zero weights, those outputs are constant and equal
to their bias (= the grid values), regardless of the input.  The resulting
single output node has size ``n_out + n_grid`` and looks like::

      [ y_0 ... y_{n_out-1} , g_0 ... g_{n_grid-1} ]
        \___ NN outputs ___/   \____ grid ______/

When normalisation is baked in (``norm_path`` given), the output
denormalisation ``y_phys = y_norm * Y_sigma + Y_mu`` would also touch the grid
rows, so we extend the constants with ``Y_sigma = 1`` and ``Y_mu = 0`` on the
grid rows — they pass through untouched and come out exactly as supplied.

This reuses ``to_kann._build_and_write_graph`` verbatim; only the weights and
normalisation arrays are augmented before writing.

Usage
-----
    python src_py/to_kann_grid.py \\
        --model  model/nn_model_conv_MPE.pth \\
        --output model/nn_model_conv_MPE_grid.kann \\
        --norm   model/normalization_conv_MPE.npz \\
        --grid   0,0.02,0.08,...          # or a .npy/.npz/.txt file
        --verify

    # Or from Python:
    from src_py.to_kann_grid import write_kann_grid
    from src_py import muvec
    write_kann_grid("model/nn_model_conv_MPE.pth",
                    "model/nn_model_conv_MPE_grid.kann",
                    grid=muvec,
                    norm_path="model/normalization_conv_MPE.npz")
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch
from torch import nn

# Reuse the building blocks from to_kann so the two stay in lock-step.
# Works both as a package module (``from src_py.to_kann_grid import …``) and as
# a standalone script (``python src_py/to_kann_grid.py``).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from src_py.to_kann import (
        KANN_MAGIC, KAD_VAR, KAD_CONST, SUPPORTED_ACTIVATIONS,
        _NeuralNetwork, _infer_layer_sizes, _build_and_write_graph,
    )
except ImportError:
    from to_kann import (
        KANN_MAGIC, KAD_VAR, KAD_CONST, SUPPORTED_ACTIVATIONS,
        _NeuralNetwork, _infer_layer_sizes, _build_and_write_graph,
    )


# ---------------------------------------------------------------------------
# Grid loading
# ---------------------------------------------------------------------------

def _load_grid(grid) -> np.ndarray:
    """
    Normalise *grid* into a 1-D float32 numpy array.

    Accepts an array-like, or a path to a ``.npy`` / ``.txt`` / ``.csv`` file,
    or a ``.npz`` file (first array, or key ``grid``/``muvec``/``vvec``), or a
    comma-separated string of floats.
    """
    if isinstance(grid, str):
        ext = os.path.splitext(grid)[1].lower()
        if ext == ".npy":
            arr = np.load(grid)
        elif ext == ".npz":
            npz = np.load(grid)
            for key in ("grid", "muvec", "vvec"):
                if key in npz:
                    arr = npz[key]
                    break
            else:
                arr = npz[npz.files[0]]
        elif ext in (".txt", ".csv", ".dat"):
            arr = np.loadtxt(grid, delimiter="," if ext == ".csv" else None)
        else:
            # treat as a comma/space separated literal list
            arr = np.fromstring(grid.replace(",", " "), sep=" ")
    else:
        arr = np.asarray(grid)

    arr = np.asarray(arr, dtype=np.float32).flatten()
    if arr.size == 0:
        raise ValueError("grid is empty")
    return arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_kann_grid(
    model_path: str,
    output_path: str,
    grid,
    activation: str = "silu",
    norm_path: str = None,
) -> np.ndarray:
    """
    Like ``to_kann.write_kann`` but appends *grid* to the network output.

    The written model produces an output vector of length ``n_out + n_grid``:
    the first ``n_out`` entries are the (optionally denormalised) NN outputs,
    the trailing ``n_grid`` entries are the static grid values.

    Parameters
    ----------
    model_path  : Path to the PyTorch weights file (.pth).
    output_path : Destination .kann file (or a directory).
    grid        : Array-like, or path to .npy/.npz/.txt, or comma-separated
                  string of grid values to embed as extra outputs.
    activation  : Hidden-layer activation (default "silu").
    norm_path   : Optional .npz normalization file (X_mu/X_sigma/Y_mu/Y_sigma);
                  when given, normalization is baked into the graph and the grid
                  rows are made pass-through (Y_sigma=1, Y_mu=0).

    Returns the float32 grid array that was embedded.
    """
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Activation '{activation}' is not supported for native KANN export. "
            f"Choose from: {SUPPORTED_ACTIVATIONS}"
        )

    grid = _load_grid(grid)
    n_grid = int(grid.size)

    # Resolve output path (same convention as to_kann.write_kann)
    abs_out = os.path.abspath(output_path)
    if os.path.isdir(abs_out) or output_path.rstrip("/\\") in (".", ".."):
        stem = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(abs_out, stem + "_grid.kann")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Load PyTorch model
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    layer_sizes = _infer_layer_sizes(state_dict)
    model = _NeuralNetwork(layer_sizes=layer_sizes, activation=activation)
    model.load_state_dict(state_dict)
    model.eval()

    linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]

    # ---- Augment the final linear layer with the grid rows ----
    last = linear_layers[-1]
    width = last.weight.shape[1]
    n_out = last.weight.shape[0]

    aug = nn.Linear(width, n_out + n_grid)
    with torch.no_grad():
        aug.weight.zero_()
        aug.weight[:n_out].copy_(last.weight)        # real NN weights
        aug.bias.zero_()                              # grid rows -> zero weights
        aug.bias[:n_out].copy_(last.bias)            # real NN biases
        aug.bias[n_out:].copy_(torch.from_numpy(grid))  # grid values as bias
    linear_layers[-1] = aug

    # ---- Load + augment normalisation so grid rows pass through ----
    norm = None
    if norm_path is not None:
        norms = np.load(norm_path)
        missing = {"X_mu", "X_sigma", "Y_mu", "Y_sigma"} - set(norms.keys())
        if missing:
            raise KeyError(f"normalization file missing keys: {missing}")
        X_mu    = norms["X_mu"].astype(np.float32).flatten()
        X_sigma = norms["X_sigma"].astype(np.float32).flatten()
        Y_mu    = norms["Y_mu"].astype(np.float32).flatten()
        Y_sigma = norms["Y_sigma"].astype(np.float32).flatten()
        # Pass-through denorm on the grid rows: y*1 + 0 = grid
        Y_mu    = np.concatenate([Y_mu,    np.zeros(n_grid, dtype=np.float32)])
        Y_sigma = np.concatenate([Y_sigma, np.ones (n_grid, dtype=np.float32)])
        norm = {"X_mu": X_mu, "X_sigma": X_sigma, "Y_mu": Y_mu, "Y_sigma": Y_sigma}

    with open(output_path, "wb") as f:
        f.write(KANN_MAGIC)
        const_arrays = _build_and_write_graph(f, linear_layers, activation, norm=norm)
        # Variable values: float32, in node order (W0, b0, W1, b1, …)
        for layer in linear_layers:
            W = layer.weight.detach().numpy().astype(np.float32)
            b = layer.bias.detach().numpy().astype(np.float32)
            f.write(W.flatten(order="C").tobytes())
            f.write(b.flatten().tobytes())
        # Constant values (normalization), in KAD_CONST node order
        for arr in const_arrays:
            f.write(arr.tobytes())

    n_var   = sum(l.weight.numel() + l.bias.numel() for l in linear_layers)
    n_const = sum(a.size for a in const_arrays)
    print(f"Wrote {output_path}")
    print(f"  layers     : {layer_sizes}  (last layer -> {n_out + n_grid} outputs)")
    print(f"  activation : {activation}")
    print(f"  norm baked : {'yes' if norm is not None else 'no'}")
    print(f"  grid points: {n_grid}  (outputs [{n_out}:{n_out + n_grid}])")
    print(f"  n_var      : {n_var},  n_const: {n_const}")
    print(f"  file size  : {os.path.getsize(output_path)} bytes")
    return grid


def verify_kann_grid_numpy(model_path: str, kann_path: str,
                           x_input: np.ndarray, grid,
                           activation: str = "silu",
                           norm_path: str = None) -> np.ndarray:
    """
    Evaluate the grid-augmented .kann file with NumPy and check that

      * output[:n_out]   matches the PyTorch model (max abs err < 1e-4), and
      * output[n_out:]   matches the embedded *grid* exactly (< 1e-5).

    *x_input* is the raw physical input when *norm_path* is provided, or a
    pre-normalised input otherwise.  Returns the full numpy output.
    """
    grid = _load_grid(grid)
    n_grid = int(grid.size)

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    layer_sizes = _infer_layer_sizes(state_dict)
    n_out = layer_sizes[-1]
    model = _NeuralNetwork(layer_sizes=layer_sizes, activation=activation)
    model.load_state_dict(state_dict)
    model.eval()

    norm = None
    if norm_path is not None:
        norms = np.load(norm_path)
        norm = {k: norms[k].astype(np.float32).flatten()
                for k in ("X_mu", "X_sigma", "Y_mu", "Y_sigma")}

    # PyTorch reference output (first n_out entries)
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

        var_shapes   = []
        const_shapes = []
        for _ in range(n_node):
            f.read(4)                                # ext_label
            f.read(4)                                # ext_flag
            flag = struct.unpack("<B", f.read(1))[0]
            n_child = struct.unpack("<i", f.read(4))[0]
            if n_child == 0:
                n_d  = struct.unpack("<B", f.read(1))[0]
                dims = list(struct.unpack(f"<{n_d}i", f.read(4 * n_d)))
                size = int(np.prod(dims)) if dims else 1
                if flag & KAD_VAR:
                    var_shapes.append((dims, size))
                elif flag & KAD_CONST:
                    const_shapes.append((dims, size))
            else:
                f.read(2)                            # op
                f.read(4 * n_child)                  # children
                f.read(4)                            # pre
                ptr_size = struct.unpack("<i", f.read(4))[0]
                if ptr_size > 0:
                    f.read(ptr_size)

        wb = [np.frombuffer(f.read(4 * size), dtype=np.float32).reshape(dims)
              for dims, size in var_shapes]
        consts = [np.frombuffer(f.read(4 * size), dtype=np.float32).reshape(dims)
                  for dims, size in const_shapes]

    # NumPy forward pass
    h = x_input.astype(np.float32)
    if consts and norm is not None:
        neg_mu, inv_sigma = consts[0].flatten(), consts[1].flatten()
        h = (h + neg_mu) * inv_sigma
        y_sigma, y_mu = consts[2].flatten(), consts[3].flatten()

    n_layers = len(wb) // 2
    for i in range(n_layers):
        W = wb[2 * i]
        b = wb[2 * i + 1]
        h = W @ h + b
        if i < n_layers - 1:
            h = act_fn(h)

    if consts and norm is not None:
        h = h * y_sigma + y_mu

    nn_err   = float(np.abs(h[:n_out] - y_torch).max())
    grid_err = float(np.abs(h[n_out:n_out + n_grid] - grid).max())
    print(f"  verify NN   : max |numpy(.kann) - PyTorch| = {nn_err:.2e}")
    print(f"  verify grid : max |numpy(.kann) - grid|    = {grid_err:.2e}")
    assert nn_err < 1e-4,  f"NN verification FAILED: max error {nn_err:.2e}"
    assert grid_err < 1e-5, f"grid verification FAILED: max error {grid_err:.2e}"
    print("  Verification PASSED")
    return h


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert a PyTorch .pth model to a native Gkeyll KANN binary file "
            "with a fixed grid embedded as extra static outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",      "-m", required=True,
                   help="Path to .pth weights file")
    p.add_argument("--output",     "-o", required=True,
                   help="Output .kann file path (or directory)")
    p.add_argument("--grid",       "-g", default=None,
                   help="Grid values: comma-separated floats, or path to a "
                        ".npy/.npz/.txt file. Defaults to src_py.muvec.")
    p.add_argument("--activation", "-a", default="silu",
                   choices=list(SUPPORTED_ACTIVATIONS),
                   help="Hidden-layer activation function")
    p.add_argument("--norm",       "-n", default=None,
                   help="Path to .npz normalization file (X_mu/X_sigma/Y_mu/Y_sigma); "
                        "when supplied, normalization is baked into the graph")
    p.add_argument("--verify",     "-v", action="store_true",
                   help="Run a NumPy round-trip check after writing")
    return p


def _resolve_grid_arg(grid_arg):
    """Return *grid_arg*, or fall back to the package's ``muvec``."""
    if grid_arg is not None:
        return grid_arg
    try:
        from src_py import muvec
    except Exception:
        try:
            from gyraze_conv_surrogate import muvec
        except Exception as exc:
            raise SystemExit(
                "No --grid supplied and could not import default 'muvec'. "
                f"({exc})"
            )
    return muvec


def main(argv=None):
    args = _build_parser().parse_args(argv)
    grid = _resolve_grid_arg(args.grid)
    write_kann_grid(args.model, args.output, grid,
                    activation=args.activation, norm_path=args.norm)
    if args.verify:
        print("Verifying…")
        state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
        layer_sizes = _infer_layer_sizes(state_dict)
        rng = np.random.default_rng(42)
        x_input = rng.standard_normal(layer_sizes[0]).astype(np.float32)
        out_path = args.output
        if os.path.isdir(out_path):
            stem = os.path.splitext(os.path.basename(args.model))[0]
            out_path = os.path.join(out_path, stem + "_grid.kann")
        verify_kann_grid_numpy(args.model, out_path, x_input, grid,
                               activation=args.activation, norm_path=args.norm)


def generate(model_path, output_path, norm_path, grid=None, activation="silu", verify=True):
    """Notebook-friendly entry point mirroring ``to_kann.generate``."""
    grid = _resolve_grid_arg(grid)
    write_kann_grid(model_path, output_path, grid,
                    activation=activation, norm_path=norm_path)
    if verify:
        print("Verifying…")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        layer_sizes = _infer_layer_sizes(state_dict)
        rng = np.random.default_rng(42)
        x_input = rng.standard_normal(layer_sizes[0]).astype(np.float32)
        verify_kann_grid_numpy(model_path, output_path, x_input, grid,
                               activation=activation, norm_path=norm_path)


if __name__ == "__main__":
    main()
