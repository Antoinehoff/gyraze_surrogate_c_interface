import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium", app_title="GYRAZE Surrogate Explorer")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sys, os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Make src_py importable from the notebook directory
    _root = os.path.normpath(os.path.join(os.path.abspath(""), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    # Public helpers
    from src_py import muvec as srg_grid
    from src_py import surrogate_model_proj as srg_proj
    from src_py import surrogate_unconv_model as srg_unconv
    from src_py import surrogate_full_model as srg_full

    # Low-level NN objects so we can evaluate the network bypassing the SVM
    import torch
    from src_py.gyraze_conv_surrogate import model, normX, denormy
    def srg_conv(alpha, gamma, phi):
        """Evaluate the NN directly, ignoring the SVM convergence check."""
        params = [alpha, gamma, phi]
        with torch.no_grad():
            x = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
            return denormy(model(normX(x))).cpu().numpy().flatten()

    return np, plt, srg_grid, srg_conv, srg_proj, srg_unconv, srg_full


# Checkboxes — defined here, displayed in the header cell below
@app.cell
def _(mo):
    show_conv   = mo.ui.checkbox(label="NN raw", value=True)
    show_proj   = mo.ui.checkbox(label="NN + projection", value=True)
    show_unconv = mo.ui.checkbox(label="Unconv surrogate", value=True)
    show_full   = mo.ui.checkbox(label="Full surrogate", value=True)
    return show_conv, show_proj, show_unconv, show_full


# All parameter sliders
@app.cell
def _(mo):
    sl_alpha = mo.ui.slider(1.0, 30.0, value=5.0,  step=0.5, label="α (deg)")
    sl_gamma = mo.ui.slider(0.0, 10.0, value=2.5,  step=0.1, label="γ")
    sl_phi   = mo.ui.slider(0.0, 10.0, value=3.0,  step=0.1, label="φ (norm.)")
    sl_npts  = mo.ui.slider(8,   64,   value=32,   step=4,   label="µ-grid points")
    sl_mumax = mo.ui.slider(1.0, 20.0, value=10.0, step=0.5, label="µ max")
    return sl_alpha, sl_gamma, sl_phi, sl_npts, sl_mumax


# Header: description table only
@app.cell
def _(mo):
    mo.md(
        r"""
        # GYRAZE Surrogate Explorer
        | Curve | Description |
        |---|---|
        | **NN raw** | Neural network output — SVM check|
        | **NN + projection** | Standard surrogate + projection of the parameters when unconverged |
        | **Unconv surrogate** | Trained on unconverged case that has been projected |
        | **Full surrogate** | Trained everywhere and internalizes projection |

        The vertical dotted line marks $\sqrt{\phi}$, the constant-cutoff reference.
        """
    )
    return


# Convergence map — re-runs when α changes (grid) or γ/φ change (marker)
@app.cell
def _(np, plt, srg_conv, sl_alpha, sl_gamma, sl_phi):
    from matplotlib.colors import ListedColormap as _LCM
    from src_py import svm_predict
    _NG, _NP = 35, 35
    _gv = np.linspace(0.0, 10.0, _NG)
    _pv = np.linspace(0.0, 10.0, _NP)
    _z  = np.array([[svm_predict(sl_alpha.value, g, p) for g in _gv] for p in _pv])

    _fig_map, _ax = plt.subplots(figsize=(3.5,3.5))
    _ax.pcolormesh(_gv, _pv, _z, 
                   cmap=_LCM(["#d9534f", "#5cb85c"]), 
                   vmin=0, vmax=1
                   )
    _ax.plot(sl_gamma.value, sl_phi.value, "w+", ms=14, mew=2.5)
    _ax.set_xlabel("γ")
    _ax.set_ylabel("φ")
    _ax.set_title(f"Convergence map  (α={sl_alpha.value:.1f}°)", fontsize=10)
    plt.tight_layout()
    map_fig = _fig_map
    return (map_fig,)


# Computation: evaluate surrogate and build the plot figure
@app.cell
def _(mo, np, plt, sl_alpha, sl_gamma, sl_phi, sl_npts, sl_mumax,
      show_conv, show_proj, show_unconv, show_full,
      srg_conv, srg_grid, srg_proj, srg_unconv, srg_full):
    alpha = sl_alpha.value
    gamma = sl_gamma.value
    phi   = sl_phi.value
    npts  = sl_npts.value
    mumax = sl_mumax.value

    mu_grid = np.linspace(0, mumax, npts)

    conv_vcut   = srg_conv( alpha, gamma, phi)
    proj_vcut  = srg_proj(mu_grid, alpha, gamma, phi)
    unconv_vcut = srg_unconv(mu_grid, alpha, gamma, phi)
    total_vcut = srg_full(mu_grid, alpha, gamma, phi)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    if show_conv.value:
        ax.plot(conv_vcut, srg_grid, color="C0", linestyle="-", marker="o",
                ms=6, label="Conv. surrogate")

    if show_proj.value:
        ax.plot(proj_vcut, mu_grid, color="C1", linestyle="--", marker="x",
                ms=6, label="Proj. Conv. surrogate")

    if show_unconv.value:
        ax.plot(unconv_vcut, mu_grid, color="C2", linestyle="-.", marker=".",
                ms=4, label="Unconv. surrogate")

    if show_full.value:
        ax.plot(total_vcut, mu_grid, color="C3", linestyle="-", marker="d",
                ms=5, label="Total surrogate")

    if phi > 0:
        ax.axvline(np.sqrt(phi), color="k", linestyle=":", lw=1.2,
                   label=r"$\sqrt{\phi}$")

    ax.set_xlabel(r"$v_{\mathrm{cut}}$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(rf"α = {alpha:.2f}°,  γ = {gamma:.3f},  φ = {phi:.2f}")
    ax.legend(fontsize=9)
    plt.tight_layout()

    return fig


# Two-column controls: sliders | curve toggles
@app.cell
def _(mo, sl_alpha, sl_gamma, sl_phi, sl_npts, sl_mumax,
      show_conv, show_proj, show_unconv, show_full):
    _sliders = mo.vstack([sl_alpha, sl_gamma, sl_phi, sl_npts, sl_mumax])
    _toggles = mo.vstack([mo.md("**Show curves:**"),
                          show_conv, show_proj, show_unconv, show_full])
    mo.hstack([_sliders, _toggles], justify="start")
    return


# side-by-side plots
@app.cell
def _(mo, fig, map_fig):
    mo.vstack([mo.hstack([fig, map_fig], justify="start")])
    return
    

if __name__ == "__main__":
    app.run()
