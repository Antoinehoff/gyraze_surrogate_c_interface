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
    from src_py import surrogate_model as srg, muvec as srg_grid
    from src_py import surrogate_model_proj as srg_proj
    from src_py import surrogate_model_total as srg_total

    # Low-level NN objects so we can evaluate the network bypassing the SVM
    import torch
    from src_py.gyraze_surrogate import model, normX, denormy

    def nn_raw(alpha, gamma, phi):
        """Evaluate the NN directly, ignoring the SVM convergence check."""
        params = [alpha, gamma, phi]
        with torch.no_grad():
            x = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
            return denormy(model(normX(x))).cpu().numpy().flatten()

    return nn_raw, np, plt, srg, srg_grid, srg_proj, srg_total


# Checkboxes — defined here, displayed in the header cell below
@app.cell
def _(mo):
    show_raw   = mo.ui.checkbox(label="Show NN raw (SVM bypassed)", value=True)
    show_svm   = mo.ui.checkbox(label="Show SVM + NN (may be None)", value=True)
    show_proj  = mo.ui.checkbox(label="Show Projection", value=True)
    show_total = mo.ui.checkbox(label="Show Total surrogate", value=True)
    return show_raw, show_svm, show_proj, show_total


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

        Move the sliders to explore the three surrogate outputs across the $\mu$-grid:

        | Curve | Description |
        |---|---|
        | **NN raw** | Neural network output — SVM check bypassed |
        | **SVM + NN** | Standard surrogate: `None` when SVM predicts non-convergence |
        | **Projection** | Projects $(α,γ,φ)$ to the converged boundary when needed |
        | **Total surrogate** | Uses projection internally, and always returns a curve |

        The vertical dotted line marks $\sqrt{\phi}$, the constant-cutoff reference.
        """
    )
    return


# Convergence map — re-runs when α changes (grid) or γ/φ change (marker)
@app.cell
def _(np, plt, srg, sl_alpha, sl_gamma, sl_phi):
    from matplotlib.colors import ListedColormap as _LCM
    _NG, _NP = 35, 35
    _gv = np.linspace(0.0, 10.0, _NG)
    _pv = np.linspace(0.0, 10.0, _NP)
    _z  = np.array([[1.0 if srg(sl_alpha.value, g, p) is not None else 0.0
                     for g in _gv] for p in _pv])

    _fig_map, _ax = plt.subplots(figsize=(3.5,3.5))
    _ax.pcolormesh(_gv, _pv, _z, cmap=_LCM(["#d9534f", "#5cb85c"]), vmin=0, vmax=1)
    _ax.plot(sl_gamma.value, sl_phi.value, "w+", ms=14, mew=2.5)
    _ax.set_xlabel("γ")
    _ax.set_ylabel("φ")
    _ax.set_title(f"Convergence map  (α={sl_alpha.value:.1f}°)", fontsize=10)
    plt.tight_layout()
    map_fig = _fig_map
    return (map_fig,)


# Computation: evaluate surrogate and build the plot figure
@app.cell
def _(mo, nn_raw, np, plt, sl_alpha, sl_gamma, sl_phi, sl_npts, sl_mumax,
      show_raw, show_svm, show_proj, show_total,
      srg, srg_grid, srg_proj, srg_total):
    alpha = sl_alpha.value
    gamma = sl_gamma.value
    phi   = sl_phi.value
    npts  = sl_npts.value
    mumax = sl_mumax.value

    mu_grid = np.linspace(0, mumax, npts)

    raw_vcut   = nn_raw(alpha, gamma, phi)
    srg_vcut   = srg(alpha, gamma, phi)
    proj_vcut  = srg_proj(mu_grid, alpha, gamma, phi)
    total_vcut = srg_total(mu_grid, alpha, gamma, phi)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    if show_raw.value:
        ax.plot(raw_vcut, srg_grid, color="C0", linestyle="-", marker="o",
                ms=6, label="NN raw (SVM bypassed)")

    if show_svm.value:
        if srg_vcut is not None:
            ax.plot(srg_vcut, srg_grid, color="C1", linestyle="--", marker="x",
                    ms=6, label="SVM + NN (converged)")
        else:
            ax.axvline(x=0, color="C1", linestyle="--", alpha=0,
                    label="SVM + NN → None (non-converged)")

    if show_proj.value:
        ax.plot(proj_vcut, mu_grid, color="C2", linestyle="-.", marker=".",
                ms=4, label="Projection")

    if show_total.value:
        ax.plot(total_vcut, mu_grid, color="C3", linestyle="-", marker="d",
                ms=5, label="Total surrogate (projection internally)")

    if phi > 0:
        ax.axvline(np.sqrt(phi), color="k", linestyle=":", lw=1.2,
                   label=r"$\sqrt{\phi}$")

    ax.set_xlabel(r"$v_{\mathrm{cut}}$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(rf"α = {alpha:.2f}°,  γ = {gamma:.3f},  φ = {phi:.2f}")
    ax.legend(fontsize=9)
    plt.tight_layout()

    converged = srg_vcut is not None
    status = mo.callout(
        mo.md(f"**SVM: converged** — NN output available at (α={alpha:.2f}, γ={gamma:.3f}, φ={phi:.2f}).") if converged
        else mo.md(f"**SVM: non-converged** — NN raw shown but SVM+NN returns `None`. Projection uses boundary point."),
        kind="success" if converged else "warn",
    )
    return fig, status


# Two-column controls: sliders | curve toggles
@app.cell
def _(mo, sl_alpha, sl_gamma, sl_phi, sl_npts, sl_mumax,
      show_raw, show_svm, show_proj, show_total):
    _sliders = mo.vstack([sl_alpha, sl_gamma, sl_phi, sl_npts, sl_mumax])
    _toggles = mo.vstack([mo.md("**Show curves:**"),
                          show_raw, show_svm, show_proj, show_total])
    mo.hstack([_sliders, _toggles], justify="start")
    return


# Status + side-by-side plots
@app.cell
def _(mo, fig, map_fig, status):
    mo.vstack([mo.hstack([fig, map_fig], justify="start"), status])
    return
    

if __name__ == "__main__":
    app.run()
