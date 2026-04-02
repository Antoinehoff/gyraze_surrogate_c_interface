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

    # Low-level NN objects so we can evaluate the network bypassing the SVM
    import torch
    from src_py.gyraze_surrogate import model, normX, denormy

    def nn_raw(alpha, gamma, phi):
        """Evaluate the NN directly, ignoring the SVM convergence check."""
        params = [alpha, gamma, phi]
        with torch.no_grad():
            x = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
            return denormy(model(normX(x))).cpu().numpy().flatten()

    return nn_raw, np, plt, srg, srg_grid, srg_proj


@app.cell
def _(mo):
    mo.md(
        r"""
        # GYRAZE Surrogate Explorer

        Move the sliders to explore the three surrogate outputs across the $\mu$-grid:

        | Curve | Description |
        |---|---|
        | **NN raw** (blue) | Neural network output — SVM check bypassed |
        | **SVM + NN** (green) | Standard surrogate: `None` when SVM predicts non-convergence |
        | **Projection** (red) | Projects $(α,γ,φ)$ to the converged boundary when needed |

        The vertical dotted line marks $\sqrt{\phi}$, the constant-cutoff reference.
        """
    )
    return


@app.cell
def _(mo):
    sl_alpha = mo.ui.slider(1.0, 30.0, value=5.0,  step=0.25, label="α (deg)")
    sl_gamma = mo.ui.slider(0.1,  5.0, value=1.0,  step=0.05, label="γ")
    sl_phi   = mo.ui.slider(0.0, 10.0, value=1.0,  step=0.1,  label="φ (norm.)")
    sl_npts  = mo.ui.slider(8,   64,   value=32,   step=4,    label="µ-grid points")
    sl_mumax = mo.ui.slider(1.0, 20.0, value=10.0, step=0.5,  label="µ max")

    mo.vstack([
        mo.hstack([sl_alpha, sl_gamma, sl_phi], justify="start"),
        mo.hstack([sl_npts, sl_mumax], justify="start"),
    ])
    return sl_alpha, sl_gamma, sl_mumax, sl_npts, sl_phi


@app.cell
def _(mo, nn_raw, np, plt, sl_alpha, sl_gamma, sl_mumax, sl_npts, sl_phi,
      srg, srg_grid, srg_proj):
    alpha = sl_alpha.value
    gamma = sl_gamma.value
    phi   = sl_phi.value
    npts  = sl_npts.value
    mumax = sl_mumax.value

    mu_grid = np.linspace(0, mumax, npts)

    # --- Three curves ---
    # 1. Raw NN (always available)
    raw_vcut = nn_raw(alpha, gamma, phi)   # shape (20,) on the fixed srg_grid

    # 2. SVM-gated surrogate (may be None)
    srg_vcut = srg(alpha, gamma, phi)

    # 3. Projection (always available, projects if needed)
    proj_vcut = srg_proj(mu_grid, alpha, gamma, phi)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    ax.plot(raw_vcut, srg_grid, color="steelblue",   linestyle="-",  marker=".",
            ms=6, label="NN raw (SVM bypassed)")

    if srg_vcut is not None:
        ax.plot(srg_vcut, srg_grid, color="green", linestyle="--", marker=".",
                ms=6, label="SVM + NN (converged)")
    else:
        ax.axvline(x=0, color="green", linestyle="--", alpha=0,
                   label="SVM + NN → None (non-converged)")

    ax.plot(proj_vcut, mu_grid, color="crimson", linestyle="-",  marker=".",
            ms=4, label="Projection")

    if phi > 0:
        ax.axvline(np.sqrt(phi), color="k", linestyle=":", lw=1.2,
                   label=r"$\sqrt{\phi}$")

    ax.set_xlabel(r"$v_{\mathrm{cut}}$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(rf"α = {alpha:.2f}°,  γ = {gamma:.3f},  φ = {phi:.2f}")
    ax.legend(fontsize=9)
    plt.tight_layout()

    # --- Status badge ---
    converged = srg_vcut is not None
    status = mo.callout(
        mo.md(f"**SVM: converged** — NN output available at (α={alpha:.2f}, γ={gamma:.3f}, φ={phi:.2f}).") if converged
        else mo.md(f"**SVM: non-converged** — NN raw shown but SVM+NN returns `None`. Projection uses boundary point."),
        kind="success" if converged else "warn",
    )

    mo.vstack([status, fig])
    return


if __name__ == "__main__":
    app.run()
