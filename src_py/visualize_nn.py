"""
visualize_nn.py — Draw a neural network architecture diagram from a .pth state dict.

Usage:
    python visualize_nn.py model/nn_model_conv.pth
    python visualize_nn.py model/nn_model_full.pth --output diagram.png
    python visualize_nn.py model/nn_model_conv.pth --max-neurons 12
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_layer_sizes(state_dict: dict) -> list[int]:
    """
    Walk the state dict keys and collect Linear layer sizes in order.
    Works for both `net.0.weight` style (Sequential wrapper) and
    `layers.0.weight` style flat dicts.
    """
    linear_weights = {}
    for key, tensor in state_dict.items():
        if key.endswith(".weight") and tensor.dim() == 2:
            # Extract the numeric index from the key so we can sort layers
            parts = key.split(".")
            # Build a sortable index from all numeric tokens
            idx = tuple(int(p) for p in parts if p.isdigit())
            linear_weights[idx] = tensor

    if not linear_weights:
        raise ValueError("No Linear layer weights found in the state dict.")

    sorted_keys = sorted(linear_weights)
    sizes: list[int] = []
    for i, k in enumerate(sorted_keys):
        w = linear_weights[k]
        out_features, in_features = w.shape
        if i == 0:
            sizes.append(in_features)
        sizes.append(out_features)
    return sizes


def layer_label(idx: int, sizes: list[int]) -> str:
    if idx == 0:
        return f"Input\n({sizes[0]})"
    if idx == len(sizes) - 1:
        return f"Output\n({sizes[-1]})"
    return f"Hidden\n({sizes[idx]})"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_network(
    sizes: list[int],
    ax: plt.Axes,
    max_neurons: int = 8,
    node_radius: float = 0.15,
    h_spacing: float = 1.0,
    v_spacing: float = 0.4,
    cmap_name: str = "tab10",
    input_labels: list[str] | None = None,
    output_labels: list[str] | None = None,
    fontsize_params: int = 18,
    fontsize_layer: int = 12,
) -> None:
    """
    Draw a neural-network diagram on *ax*.

    Parameters
    ----------
    sizes         : list of neuron counts per layer.
    max_neurons   : maximum neurons to draw per layer (extras become "...").
    input_labels  : optional per-node labels drawn to the left of the input layer.
    output_labels : optional per-node labels drawn to the right of the output layer.
    """
    n_layers = len(sizes)
    cmap = plt.get_cmap(cmap_name)

    # Hidden layers are drawn with up to max_neurons nodes.
    # Input/output are capped at (hidden_draw_max - 2) so they always appear
    # visually smaller, preserving a sense of scale.
    hidden_draw_max = max_neurons
    io_draw_max = max(1, hidden_draw_max - 2)
    draw_limits = [
        io_draw_max if (li == 0 or li == n_layers - 1) else hidden_draw_max
        for li in range(n_layers)
    ]

    # ---- compute node positions ----------------------------------------
    # Each layer is a column; nodes are centred vertically.
    positions: list[list[tuple[float, float]]] = []
    for li, n in enumerate(sizes):
        x = li * h_spacing
        n_draw = min(n, draw_limits[li])
        total_height = (n_draw - 1) * v_spacing
        ys = [total_height / 2 - j * v_spacing for j in range(n_draw)]
        positions.append([(x, y) for y in ys])

    # ---- draw edges (connections) first so nodes sit on top -------------
    edge_color = "#000000"
    edge_alpha = 0.2
    for li in range(n_layers - 1):
        src_pos = positions[li]
        dst_pos = positions[li + 1]
        for sx, sy in src_pos:
            for dx, dy in dst_pos:
                ax.plot(
                    [sx + node_radius, dx - node_radius],
                    [sy, dy],
                    color=edge_color,
                    lw=1,
                    alpha=edge_alpha,
                    zorder=1,
                )

    # ---- draw nodes -------------------------------------------------------
    for li, (n, pos_list) in enumerate(zip(sizes, positions)):
        color = cmap(li / max(n_layers - 1, 1))
        n_draw = min(n, draw_limits[li])
        truncated = n > draw_limits[li]

        for ni, (x, y) in enumerate(pos_list):
            # If truncated and this is the last drawn node, show "..." instead
            is_ellipsis = truncated and ni == n_draw - 1
            face = "#f5f5f5" if is_ellipsis else color
            circle = plt.Circle(
                (x, y),
                node_radius,
                color=face,
                ec="#333333",
                lw=0.8,
                zorder=3,
            )
            ax.add_patch(circle)
            if is_ellipsis:
                ax.text(
                    x, y, r"$\cdots$",
                    ha="center", va="center",
                    fontsize=fontsize_layer, zorder=4, color="#555555",
                )

            # ---- per-node labels on input / output layers -----------------
            label_offset = node_radius + 0.08
            if li == 0 and input_labels is not None and ni < len(input_labels) and not is_ellipsis:
                ax.text(
                    x - label_offset, y,
                    input_labels[ni],
                    ha="right", va="center",
                    fontsize=fontsize_params, zorder=5, color="#111111",
                )
            if li == n_layers - 1 and output_labels is not None and ni < len(output_labels) and not is_ellipsis:
                ax.text(
                    x + label_offset, y,
                    output_labels[ni],
                    ha="left", va="center",
                    fontsize=fontsize_params, zorder=5, color="#111111",
                )

        # Layer label below the column
        max_y = pos_list[0][1]
        min_y = pos_list[-1][1]
        label_y = min_y - v_spacing * 0.5
        ax.text(
            pos_list[0][0], label_y,
            layer_label(li, sizes),
            ha="center", va="top",
            fontsize=fontsize_layer,
            color="#222222",
        )

    # ---- axes formatting --------------------------------------------------
    all_x = [x for col in positions for x, y in col]
    all_y = [y for col in positions for x, y in col]
    left_margin  = 0.4 if input_labels else 0.3
    right_margin = 1.0 if output_labels else 0.3
    ax.set_xlim(min(all_x) - left_margin, max(all_x) + right_margin)
    ax.set_ylim(min(all_y) - v_spacing * 1.8, max(all_y) + 0.25)
    ax.set_aspect("equal")
    ax.axis("off")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a PyTorch neural network from a .pth state dict."
    )
    parser.add_argument("pth_file", help="Path to the .pth model file.")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output image path (e.g. diagram.png). Defaults to <model_name>_diagram.png.",
    )
    parser.add_argument(
        "--max-neurons", "-m",
        type=int, default=10,
        help="Maximum number of neurons to draw per layer (default: 10).",
    )
    parser.add_argument(
        "--dpi",
        type=int, default=150,
        help="Resolution of the saved image (default: 150).",
    )
    args = parser.parse_args()

    pth_path = args.pth_file
    if not os.path.isfile(pth_path):
        sys.exit(f"Error: file not found: {pth_path}")

    # ---- Load state dict --------------------------------------------------
    state_dict = torch.load(pth_path, map_location="cpu")
    # Support both bare state dicts and checkpoint dicts
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif not isinstance(state_dict, dict):
        sys.exit("Error: .pth file does not contain a state dict.")

    # ---- Infer architecture -----------------------------------------------
    try:
        sizes = infer_layer_sizes(state_dict)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    n_params = sum(t.numel() for t in state_dict.values())
    print(f"Model file : {pth_path}")
    print(f"Layer sizes: {' → '.join(str(s) for s in sizes)}")
    print(f"Parameters : {n_params:,}")

    # ---- Draw -------------------------------------------------------------
    # Scale figure width with number of layers
    fig_w = max(6, len(sizes) * 1.2 + 2.5)
    fig_h = max(4, args.max_neurons * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # ---- Default node labels (match the physical meaning of this surrogate) --
    # Input:  α, γ̂, φ̂
    # Output: v_{∥,cut}(μ_i) for i = 1 … n_out
    n_in = sizes[0]
    n_out = sizes[-1]
    default_input_labels = [
        r"$\alpha$",
        r"$\hat{\gamma}$",
        r"$\hat{\phi}$",
    ][:n_in]
    default_output_labels = [
        # r"$v_{\parallel,\mathrm{cut}}(\mu_{" + str(i + 1) + r"})$"
        r"$v_{\parallel,\mathrm{cut}}^{\mu_{" + str(i + 1) + r"}}$"
        for i in range(n_out)
    ]

    draw_network(
        sizes, ax,
        max_neurons=args.max_neurons,
        input_labels=default_input_labels,
        output_labels=default_output_labels,
    )

    model_name = os.path.splitext(os.path.basename(pth_path))[0]
    title = f"{model_name}   [{' → '.join(str(s) for s in sizes)}]   ({n_params:,} params)"
    # ax.set_title(title, fontsize=16, pad=8, color="#222222")

    # ---- Legend for layer types -------------------------------------------
    cmap = plt.get_cmap("tab10")
    n_layers = len(sizes)
    legend_handles = []
    for li, n in enumerate(sizes):
        color = cmap(li / max(n_layers - 1, 1))
        label = layer_label(li, sizes).replace("\n", " ")
        legend_handles.append(mpatches.Patch(color=color, label=label))
    # ax.legend(
    #     handles=legend_handles,
    #     loc="upper right",
    #     fontsize=7,
    #     framealpha=0.8,
    #     title="Layers",
    #     title_fontsize=7,
    # )

    fig.tight_layout()

    output_path = args.output or f"{model_name}_diagram.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", transparent=True)
    print(f"Saved      : {output_path}")


if __name__ == "__main__":
    main()
