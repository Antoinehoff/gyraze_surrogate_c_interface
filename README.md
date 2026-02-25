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
| `gyraze_surrogate.py` | Python surrogate model (NN + SVM evaluation) |
| `export_to_c.py` | Exports the surrogate to self-contained C code |
| `nn_model.pth` | Pretrained NN weights (PyTorch) |
| `svm_model.pkl` | Pretrained SVM classifier (scikit-learn 1.6.1) |
| `normalization.npz` | Input/output normalisation parameters |
| `test_surrogate.ipynb` | Jupyter notebook: evaluation + C export + verification |
| `retrain_models.ipynb` | Jupyter notebook: model retraining |
| `requirements.txt` | Python dependencies |

---

## Python Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Evaluate the surrogate:**
```python
from gyraze_surrogate import surrogate_model

surrogate_model(4, 0.5, 2.5)
# predicts v_par_cut(mu) for alpha=4, gamma=0.5, phi_wall=2.5
# returns None if SVM predicts non-convergence
```

---

## C Code Export

The surrogate can be exported to pure C with no runtime dependencies beyond `libm`:

```python
from export_to_c import generate_c_code

generate_c_code(
    nn_model      = "nn_model.pth",
    svm_model     = "svm_model.pkl",
    normalization = "normalization.npz",
    output_dir    = "generated_c_code",   # optional, default "."
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
// Returns 1 if GYRAZE is predicted to converge, 0 otherwise.
int gyraze_converged(double alpha, double gamma, double phi);

// Runs the NN and writes 20 predicted v_par_cut values into out[].
void gyraze_predict(float alpha, float gamma, float phi, float out[20]);
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
