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

    # Make src_py importable — add project root (cwd when running marimo)
    # and also the notebook's own parent, for when the cwd differs
    _cwd  = os.path.abspath("")
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = _cwd if os.path.isdir(os.path.join(_cwd, "src_py")) else os.path.normpath(os.path.join(_here, ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    # Public helpers
    from src_py import muvec as srg_grid
    from src_py import surrogate_model_proj as srg_proj
    from src_py import surrogate_unconv_model as srg_unconv
    from src_py import surrogate_full_model as srg_full
    from src_py import surrogate_full_MPE_model as srg_full_MPE
    from src_py import surrogate_conv_MPE_model as srg_conv_MPE

    # Low-level NN objects so we can evaluate the network bypassing the SVM
    import torch
    from src_py.gyraze_conv_surrogate import model, normX, denormy
    def srg_conv(alpha, gamma, phi):
        """Evaluate the NN directly, ignoring the SVM convergence check."""
        params = [alpha, gamma, phi]
        with torch.no_grad():
            x = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
            return denormy(model(normX(x))).cpu().numpy().flatten()

    return np, plt, srg_grid, srg_conv, srg_proj, srg_unconv, srg_full, srg_full_MPE, srg_conv_MPE


# Checkboxes — defined here, displayed in the header cell below
@app.cell
def _(mo):
    show_conv     = mo.ui.checkbox(label="NN raw", value=False)
    show_proj     = mo.ui.checkbox(label="NN + projection", value=False)
    show_unconv   = mo.ui.checkbox(label="Unconv surrogate", value=False)
    show_full     = mo.ui.checkbox(label="Full surrogate", value=False)
    show_full_MPE = mo.ui.checkbox(label="Full MPE surrogate", value=True)
    show_conv_MPE = mo.ui.checkbox(label="Conv MPE surrogate", value=True)
    return show_conv, show_proj, show_unconv, show_full, show_full_MPE, show_conv_MPE


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
        | **Full MPE surrogate** | Full surrogate trained with mean percentage error loss |
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
      show_conv, show_proj, show_unconv, show_full, show_full_MPE, show_conv_MPE,
      srg_conv, srg_grid, srg_proj, srg_unconv, srg_full, srg_full_MPE,
      srg_conv_MPE):
    alpha = sl_alpha.value
    gamma = sl_gamma.value
    phi   = sl_phi.value
    npts  = sl_npts.value
    mumax = sl_mumax.value

    mu_grid = np.linspace(0, mumax, npts)

    conv_vcut      = srg_conv(alpha, gamma, phi)
    proj_vcut      = srg_proj(mu_grid, alpha, gamma, phi)
    unconv_vcut    = srg_unconv(mu_grid, alpha, gamma, phi)
    full_vcut     = srg_full(mu_grid, alpha, gamma, phi)
    full_MPE_vcut = srg_full_MPE(mu_grid, alpha, gamma, phi)
    conv_MPE_vcut  = srg_conv_MPE(mu_grid, alpha, gamma, phi)

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
        ax.plot(full_vcut, mu_grid, color="C3", linestyle="-", marker="d",
                ms=5, label="Full surrogate")

    if show_full_MPE.value:
        ax.plot(full_MPE_vcut, mu_grid, color="C4", linestyle="--", marker="s",
                ms=5, label="Full MPE surrogate")

    if show_conv_MPE.value:
        ax.plot(conv_MPE_vcut, mu_grid, color="C5", linestyle="-", marker="o",
                ms=5, label="Conv MPE surrogate")

    if phi > 0:
        ax.axvline(np.sqrt(2 * phi), color="k", linestyle=":", lw=1.2,
                   label=r"$\sqrt{2\phi} (Gkeyll)$")

    ax.set_xlabel(r"$v_{\mathrm{cut}}$")
    ax.set_ylabel(r"$\mu$")
    ax.set_xlim(0, 5.0)
    ax.set_title(rf"α = {alpha:.2f}°,  γ = {gamma:.3f},  φ = {phi:.2f}")
    ax.legend(fontsize=9)
    plt.tight_layout()

    return fig


# Two-column controls: sliders | curve toggles
@app.cell
def _(mo, sl_alpha, sl_gamma, sl_phi, sl_npts, sl_mumax,
      show_conv, show_proj, show_unconv, show_full, show_full_MPE, show_conv_MPE):
    _sliders = mo.vstack([sl_alpha, sl_gamma, sl_phi, sl_npts, sl_mumax])
    _toggles = mo.vstack([mo.md("**Show curves:**"),
                          show_conv, show_proj, show_unconv, show_full, show_full_MPE, show_conv_MPE])
    mo.hstack([_sliders, _toggles], justify="start")
    return


# side-by-side plots
@app.cell
def _(mo, fig, map_fig):
    mo.vstack([mo.hstack([fig, map_fig], justify="start")])
    return
    

if __name__ == "__main__":
    app.run()
