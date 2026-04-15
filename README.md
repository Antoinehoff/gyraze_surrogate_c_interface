# GYRAZE Surrogate Model

A set of Neural Network (NN) surrogates alongside a nonlinear Support Vector Machine (SVM) classifier that:

1. Determines whether GYRAZE converges for the given inputs (using the SVM).
2. Provides a fast prediction for `v_par_cut(mu)` over the µ-grid used in GYRAZE (using a trained NN).

Note: 
    When we talk about "convergence" here, we mean whether GYRAZE successfully converges 
    to a solution for the given input parameters within the maximum number of iterations.
    The "unconverged" model refers to a model trained on the projected boundary of 
    non-converging inputs.

Five surrogate variants are available:

| Exported name | Module | Behaviour when SVM says non-converged |
|---------------|--------|---------------------------------------|
| `surrogate_conv_model` | `gyraze_conv_surrogate` | returns `None` |
| `surrogate_model_proj` | `gyraze_surrogate_proj` | projects `(α,γ,φ)` to nearest SVM boundary via L-BFGS, evaluates conv NN there |
| `surrogate_conv_unconv_model` | `gyraze_conv_unconv_surrogate` | routes to a dedicated *unconv NN* trained on projected boundary data; always returns a result |
| `surrogate_full_model` | `gyraze_full_surrogate` | single NN trained on both converged and projected data; no SVM needed at inference |
| `surrogate_unconv_model` | `gyraze_unconv_surrogate` | unconv NN only, no SVM routing |

---

## Authors & Acknowledgements

The neural network surrogate model is developed and maintained by **Camden Warme** and **Prof. Roger G. Ghanem** at the [University of Southern California (USC)](https://www.usc.edu/), in collaboration with the **CEDA group** at the [Princeton Plasma Physics Laboratory (PPPL)](https://www.pppl.gov/), led by Prof. **Felix Para Diaz**.

---

## Files

### Source (`src_py/`)

| File | Description |
|------|-------------|
| `gyraze_conv_surrogate.py` | Conv NN + SVM; returns `None` on non-convergence |
| `gyraze_surrogate_proj.py` | Projects `(α,γ,φ)` to SVM boundary, evaluates conv NN; also exports `svm_predict` |
| `gyraze_conv_unconv_surrogate.py` | SVM-routed dual NN: conv NN or unconv NN depending on convergence |
| `gyraze_full_surrogate.py` | Single NN trained on converged + projected data; always returns a result |
| `gyraze_unconv_surrogate.py` | Unconv NN only (no SVM) |
| `nn_to_c.py` | `nnToC` class: loads a `.pth` + `.npz` and produces C code fragments |
| `export_to_c.py` | `generate_c_code()`: assembles and writes the full C surrogate library |

### Models (`model/`)

| File | Description |
|------|-------------|
| `nn_model_conv.pth` | Pretrained conv NN weights (PyTorch) |
| `nn_model_unconv.pth` | Pretrained unconv NN weights (PyTorch) |
| `nn_model_full.pth` | Pretrained full NN weights (PyTorch, width 70) |
| `normalization_conv.npz` | Input/output normalisation for the conv NN |
| `normalization_unconv.npz` | Input/output normalisation for the unconv NN |
| `normalization_full.npz` | Input/output normalisation for the full NN |
| `svm_model.pkl` | Pretrained SVM classifier (scikit-learn) |

### Notebooks (`notebook/`)

| File | Description |
|------|-------------|
| `explore_surrogate.mo.py` | Marimo app: interactive explorer for all surrogate variants |
| `test_surrogate.ipynb` | Jupyter notebook: evaluation + C export + verification |
| `retrain_models.ipynb` | Jupyter notebook: model retraining |

### Other

| Path | Description |
|------|-------------|
| `generated_c_code/` | Auto-generated C source (`surrogate.c/h`, `test_surrogate.c`, `test_projection.c`, `Makefile`) |
| `data/` | Training data (`.h5` datasets) |
| `requirements.txt` | Python dependencies |

---

## Python Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Conv surrogate** (returns `None` when SVM predicts non-convergence):
```python
import sys
sys.path.insert(0, '/path/to/gkeyll_sheath_ai')

from src_py import surrogate_conv_model

result = surrogate_conv_model(4, 0.5, 2.5)
# returns v_par_cut array (shape 20) or None if SVM predicts non-convergence
```

**Projection surrogate** (projects to SVM boundary, always returns a value):
```python
from src_py import surrogate_model_proj

result = surrogate_model_proj(mu=1.5, alpha=4, gamma=0.5, phi=2.5)
# projects (α,γ,φ) if non-converged, then evaluates conv NN; returns scalar interpolated at mu
```

**Conv/unconv surrogate** (SVM-routed dual NN, always returns a value):
```python
from src_py import surrogate_conv_unconv_model

result = surrogate_conv_unconv_model(mu=1.5, alpha=4, gamma=0.5, phi=2.5)
# routes to conv NN if converged, unconv NN otherwise; returns scalar interpolated at mu
```

**Full surrogate** (single NN, no SVM at inference):
```python
from src_py import surrogate_full_model

result = surrogate_full_model(mu=1.5, alpha=4, gamma=0.5, phi=2.5)
# single NN trained on converged + projected data; always returns a scalar
```

**SVM classifier only:**
```python
from src_py import svm_predict

label = svm_predict(alpha=4, gamma=0.5, phi=2.5)
# returns 1 (converged) or 0 (non-converged)
```

**µ-grid:**
```python
from src_py import muvec   # numpy array, shape (20,)
```

---

## Neural Network Architecture

All networks share the same fully-connected structure:

```
Input (3)  →  [Linear → SiLU] × (depth-1)  →  Linear  →  Output (20)
```

- **Input**: `(α, γ, φ)` — normalised to zero mean / unit variance
- **Hidden layers**: The width (number of neurons) and depth (number of layers) vary between models; activation is SiLU.
- **Output**: 20 values of `v_par_cut` on the fixed µ-grid

The three trained models differ in **architecture width** and **training data**:

| Exported name | Weights file | Width | Trained on |
|---------------|-------------|-------|-----------|
| `surrogate_conv_model` | `nn_model_conv.pth` | 75 | Converged GYRAZE runs |
| `surrogate_unconv_model` / `surrogate_conv_unconv_model` | `nn_model_unconv.pth` | 75 | Projected non-converged inputs (nearest SVM-boundary points) |
| `surrogate_full_model` | `nn_model_full.pth` | 70 | Converged + projected data combined |

The SVM classifier (RBF kernel) gates the conv and unconv NNs in `surrogate_conv_unconv_model` and `surrogate_model_proj`.

---

## Interactive Explorer (Marimo)

`notebook/explore_surrogate.mo.py` is a reactive [Marimo](https://marimo.io) app for visually comparing all surrogate variants:

```bash
cd notebook
marimo run explore_surrogate.mo.py
```

The app shows:
- **Header**: description table and curve toggles
- **Controls**: sliders for `α`, `γ`, `φ`, µ-grid density and µ max; a convergence map (green = converged, red = non-converged for the selected `α`) with a marker at the current `(γ, φ)` point
- **Plot**: `v_cut(µ)` curves for all active surrogates side by side with the convergence map

---

## C Code Export

The surrogate can be exported to pure C with no runtime dependencies beyond `libm`:

```python
import sys
sys.path.insert(0, '/path/to/gkeyll_sheath_ai')   # project root

from src_py import generate_c_code

generate_c_code(
    nn_model      = "model/nn_model_conv.pth",
    svm_model     = "model/svm_model.pkl",
    normalization = "model/normalization_conv.npz",
    output_dir    = "generated_c_code",   # optional, default "generated_c_code"
    output_name   = "surrogate",           # optional, default "surrogate"
    # Optional: pass both to generate the SVM-routed dual-NN variant
    nn_total_model      = "model/nn_model_unconv.pth",
    normalization_total = "model/normalization_unconv.npz",
)
```

This generates the following files:

| File | Description |
|------|-------------|
| `<output_dir>/surrogate.c` | Self-contained surrogate library |
| `<output_dir>/surrogate.h` | Public API header |
| `<output_dir>/test_surrogate.c` | CLI test program (predict / eval / physical modes) |
| `<output_dir>/test_projection.c` | CLI test program for the projection function |
| `<output_dir>/Makefile` | Build file |
| `<gkeyll_dir>/bc_sheath_gyrokinetic_gyraze_surrogate.c` | Gkeyll-compatible kernel (CUDA-aware) |
| `<gkeyll_dir>/gkyl_bc_sheath_gyrokinetic_gyraze_surrogate.h` | Gkeyll-compatible header |

**Build and run the example:**
```bash
cd generated_c_code && make
./test_surrogate predict 4.0 1.0 2.5
./test_surrogate eval 1.5 4.0 1.0 2.5
./test_surrogate physical 1.5
```

### C API

All public functions are prefixed `srgrz_` and are tagged `GKYL_CU_DH` in the Gkeyll kernel (no-op in plain C).

```c
#define SRGRZ_N_MU 20   /* number of µ-grid points */

/* Returns 1 if GYRAZE is predicted to converge, 0 otherwise. */
int srgrz_converged(double alpha, double gamma, double phi);

/* Runs the NN; writes SRGRZ_N_MU predicted v_par_cut values into out[]. */
void srgrz_predict(double alpha, double gamma, double phi, double out[SRGRZ_N_MU]);

/* Copies the SRGRZ_N_MU µ-grid into out[] and returns out. */
double *srgrz_grid(double *out);

/* Linear interpolation of vcut[SRGRZ_N_MU] onto mu_new[n]; clamps at boundaries. */
void srgrz_interp(const double *vcut, const double *mu_new, int n, double mu_ref, double *out);

/* Projects (α,γ,φ) onto nearest convergent point via L-BFGS; returns 1 if successful. */
int srgrz_project(double alpha, double gamma, double phi,
                  double *alpha_proj, double *gamma_proj, double *phi_proj);

/* Predict on a custom µ-grid using normalised inputs. */
void srgrz_eval_norm(const double *mu_new, int n, double mu_ref,
                     double alpha, double gamma, double phi, double *out);

/* Like srgrz_eval_norm but projects non-converged inputs first. */
void srgrz_proj_eval_norm(const double *mu_new, int n, double mu_ref,
                          double alpha, double gamma, double phi, double *out);

/* Predict on a custom µ-grid using physical parameters. */
void srgrz_eval(const double *mu_new, int n, double phi, double phi_wall,
                double density, double temperature, double bmag,
                double impact_angle, double *out);

/* Like srgrz_eval but returns vcut^2 / (2*phi_norm). */
void srgrz_eval_fact(const double *mu_new, int n, double phi, double phi_wall,
                     double density, double temperature, double q2Dm,
                     double bmag, double impact_angle, double *out);

/* Like srgrz_eval_fact but returns 1.0 for non-converged inputs. */
void srgrz_conv_eval_fact(const double *mu_new, int n, double phi, double phi_wall,
                          double density, double temperature, double q2Dm,
                          double bmag, double impact_angle, double *out);

/* Like srgrz_eval_fact but projects non-converged inputs first. */
void srgrz_proj_eval_fact(const double *mu_new, int n, double phi, double phi_wall,
                          double density, double temperature, double q2Dm,
                          double bmag, double impact_angle, double *out);
```

---

## Training Domain

Input data was sampled uniformly from:

| Parameter | Range |
|-----------|-------|
| `alpha` | 2 – 10 |
| `gamma_MPE` | ~0.5 – ~4 |
| `phi_wall` | 1 – 10 |

Fixed simulation settings:
- `type_distfunc_entrance` = ADHOC
- `mi/me` = 3600
- `Ti/Te` = `ni/ne` = `n_spec` = 1
- Set gamma at DS = FALSE
- Set current = 0
