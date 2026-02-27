# GYRAZE Surrogate Model

A Neural Network (NN) regression function alongside a nonlinear Support Vector Machine (SVM) classifier which:

1. Determines whether GYRAZE converges for the given inputs (using the SVM).
2. If GYRAZE converges, provides a fast prediction for `v_par_cut(mu)` over the mu-grid used in GYRAZE (using the trained NN).

---

## Authors & Acknowledgements

The neural network surrogate model is developed and maintained by **Camden Warme** and **Prof. Roger G. Ghanem** at the [University of Southern California (USC)](https://www.usc.edu/), in collaboration with the **CEDA group** at the [Princeton Plasma Physics Laboratory (PPPL)](https://www.pppl.gov/), led by Prof. **Felix Para Diaz**.

---

## Files

| File | Description |
|------|-------------|
| `src_py/gyraze_surrogate.py` | Python surrogate model (NN + SVM evaluation) |
| `src_py/surrogate_proj.py` | Extended surrogate with nearest-boundary search |
| `src_py/export_to_c.py` | Exports the surrogate to self-contained C code |
| `model/nn_model.pth` | Pretrained NN weights (PyTorch) |
| `model/svm_model.pkl` | Pretrained SVM classifier (scikit-learn 1.6.1) |
| `model/normalization.npz` | Input/output normalisation parameters |
| `notebook/test_surrogate.ipynb` | Jupyter notebook: evaluation + C export + verification |
| `notebook/retrain_models.ipynb` | Jupyter notebook: model retraining |
| `generated_c_code/` | Auto-generated C source (surrogate.c/h, Makefile) |
| `data/` | Data files (training data, etc.) |
| `requirements.txt` | Python dependencies |

---

## Python Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Evaluate the surrogate:**
```python
import sys
sys.path.insert(0, '/path/to/gkeyll_sheath_ai')   # project root

from src_py import surrogate_model

surrogate_model(4, 0.5, 2.5)
# predicts v_par_cut(mu) for alpha=4, gamma=0.5, phi_wall=2.5
# returns None if SVM predicts non-convergence
```

---

## C Code Export

The surrogate can be exported to pure C with no runtime dependencies beyond `libm`:

```python
import sys
sys.path.insert(0, '/path/to/gkeyll_sheath_ai')   # project root

from src_py import generate_c_code

generate_c_code(
    nn_model      = "model/nn_model.pth",
    svm_model     = "model/svm_model.pkl",
    normalization = "model/normalization.npz",
    output_dir    = "generated_c_code",   # optional, default "generated_c_code"
    output_name   = "surrogate",           # optional, default "surrogate"
)
```

This generates the following files in `<output_dir>/`:

| File | Description |
|------|-------------|
| `surrogate.c` | Self-contained surrogate library |
| `surrogate.h` | Public API header |
| `test_surrogate.c` | Minimal example program |
| `Makefile` | Build file |

**Build and run the example:**
```bash
cd generated_c_code && make
./test_surrogate
```

### C API

```c
#define GYRAZE_N_MU 20   /* number of mu-grid points */

// Returns 1 if GYRAZE is predicted to converge, 0 otherwise.
int gyraze_converged(double alpha, double gamma, double phi);

// Runs the NN; writes GYRAZE_N_MU predicted v_par_cut values into out[].
void gyraze_predict(double alpha, double gamma, double phi, double out[GYRAZE_N_MU]);

// Returns a read-only pointer to the GYRAZE_N_MU-element mu-grid.
const double *gyraze_grid(void);

// Linear interpolation of vcut[GYRAZE_N_MU] (on the fixed mu-grid) onto
// mu_new[n]; results written into out[n]. Clamps at grid boundaries.
void gyraze_interp(const double *vcut, const double *mu_new, int n, double *out);
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
