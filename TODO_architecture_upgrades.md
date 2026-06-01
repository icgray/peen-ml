# peen-ml Architecture Upgrade TODO

---

## ✅ Completed

| # | Item | Notes |
|---|------|-------|
| 7 | **Material Awareness** — `materials.py` library, material CLI args in `native_dataset_gen.py`, material conditioning on all three CNN architectures (`mat_dim=7`), GUI material selectors in Generate and Evaluate tabs, per-simulation `[material]` block in `simulation_params.txt` | Done 2026-05-31 |
| 8 | **CI Linting + Coverage** — fixed all pylint/flake8/black failures; workflows green on Python 3.9/3.10/3.11; tests expanded from 13 → 35; coverage raised from 26% → 41% | Done 2026-06-01 |

---

## 🟡 Next: Increase Code Coverage

**Priority: High** — CI is now green. Coverage sits at **41%** across `src/`.
Target: push to **≥ 70%** before implementing new architectures so regressions are caught early.

### Coverage by file (as of 2026-06-01)

Run locally to reproduce:
```bash
# from shotpeeningML/
PYTHONPATH=$(pwd)/src/peen-ml MPLBACKEND=Agg \
  pytest tests/ --cov=src --cov-branch --cov-report=term-missing --cov-config=.coveragerc
```

| File | Coverage | Uncovered blocks |
|---|---|---|
| `src/peen-ml/__init__.py` | 100% | — |
| `src/peen-ml/data_viz.py` | 43% | visualize_mesh (136–164), visualize_stress_field (169–228), visualize_deformation (238–257), visualize_all body (264–272), error branches in compute_deformed_mesh (129–131) |
| `src/peen-ml/model.py` | 40% | create_data_loaders (337–363), train_model (413–481), evaluate_model (516–556), main() (574–613), train_save_gui (640–679), create_test_loader (696–707), evaluate_model_gui (723–780), load_and_evaluate_model_gui (804–823) |

### What to test next — per file

#### `src/peen-ml/data_viz.py`

Remaining uncovered functions all call `plt.show()`. Use `unittest.mock.patch("matplotlib.pyplot.show")` (already established in `test_data_viz.py`) and pass synthetic numpy arrays directly.

- **`visualize_mesh`** (lines 134–164) — create small `node_coords`, `deformed_coords` (shape `(N,3)`), and `element_nodes` (list of index lists); call with `patch("matplotlib.pyplot.show")`. No file I/O required.
- **`visualize_stress_field`** (lines 167–228) — needs a `simulation_folder` with `stresses.npy` (shape `(N,6)`) and `stress_element_labels.npy`. Add a new `tmpdir` fixture that writes these files, then call with `patch("matplotlib.pyplot.show")`.
- **`visualize_deformation`** (lines 228–257) — call directly with synthetic `deformed_coords`, `element_nodes`, and `aligned_displacements`; patch `plt.show()`. The first argument (`simulation_folder`) is unused (`_` parameter).
- **`visualize_all` success path** (lines 262–295) — create a fully-populated `tmpdir` with all required `.npy` files (see `create_simulation` fixture) and call `visualize_all(folder, scale_factor=1)` with `plt.show()` patched.

#### `src/peen-ml/model.py`

Training/evaluation functions require real data loaders. Use the existing `tests/test_simulations/` fixtures (2 simulations, 10×10 checkerboard, 10 nodes). Note: `DisplacementPredictor` FC layer is hardcoded to `128 * 5 * 5 = 3200` inputs, so it only works with **5×5 checkerboard input**. The existing test fixtures use 10×10; use synthetic data in tests rather than the fixture files.

- **`create_data_loaders`** (lines 337–363) — call with `base_folder="./tests/test_simulations"`. Returns train/val/test loaders. Assert all three are non-None and that batches have the right shapes.
- **`train_model`** (lines 413–481) — create a tiny model (5×5 input, `num_nodes=10`), build minimal train/val loaders from synthetic data (2 samples each), run for 1 epoch. Assert losses list is non-empty and loss is finite. **Patch `plt.show` and `plt.ion`** to suppress the interactive plot.
- **`evaluate_model`** (lines 516–556) — same tiny model + a test loader; call `evaluate_model(model, test_loader, criterion)`. Assert the returned MSE is a positive float.
- **`create_test_loader`** (lines 696–707) — call with `base_folder="./tests/test_simulations"`. Assert the loader yields `(checkerboard, displacement)` tuples.
- **`evaluate_model_gui`** (lines 723–780) — call with a tiny model, test loader, `nn.MSELoss()`, and a `tmpdir` for `pred_save_dir`. Assert `.npy` and `.csv` files are written to `pred_save_dir/Simulation_0/`.
- **`load_and_evaluate_model_gui`** (lines 804–823) — save the tiny model to a temp file with `torch.save`, then call `load_and_evaluate_model_gui(model_path, data_path, pred_save_dir)`. Assert predictions are saved.
- **`main()`** (lines 574–613) — the hardcoded Windows path makes this untestable without heavy mocking. **Skip** — mark with `# pragma: no cover` or add to `.coveragerc` omit list.
- **`train_save_gui`** (lines 640–679) — same as `train_model` but also assert the saved model file appears in `data_path/saved_model/`.

### Suggested test additions

Add these to existing test files:

**`tests/test_model.py`**:
```python
# Helper to build a tiny loader from synthetic data (5x5 checkerboard, 10 nodes)
def _tiny_loader(n_samples=4, batch_size=2):
    cb = np.random.rand(n_samples, 5, 5)
    disp = np.random.rand(n_samples, 10, 3)
    ds = CheckerboardDataset(cb, disp)
    norm = NormalizedDataset(ds)
    return DataLoader(norm, batch_size=batch_size)

def test_train_model_one_epoch():
    model = DisplacementPredictor(input_channels=1, num_nodes=10)
    loader = _tiny_loader()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.ion"):
        train_losses, val_losses = train_model(
            model, loader, loader, criterion, optimizer, scheduler,
            epochs=1, patience=5
        )
    assert len(train_losses) == 1
    assert not np.isnan(train_losses[0])

def test_evaluate_model_returns_mse():
    model = DisplacementPredictor(input_channels=1, num_nodes=10)
    loader = _tiny_loader(n_samples=2, batch_size=1)
    mse = evaluate_model(model, loader, nn.MSELoss())
    assert mse >= 0.0

def test_create_data_loaders():
    train, val, test, data = create_data_loaders("./tests/test_simulations")
    assert train is not None and val is not None and test is not None
    assert "checkerboard" in data
```

**`tests/test_data_viz.py`**:
```python
def test_visualize_mesh_basic():
    node_coords = np.random.rand(6, 3)
    deformed_coords = node_coords + 0.01
    element_nodes = [[0, 1, 2], [3, 4, 5]]
    with patch("matplotlib.pyplot.show"):
        visualize_mesh(node_coords, deformed_coords, element_nodes)

def test_visualize_deformation_basic():
    deformed_coords = np.random.rand(6, 3)
    element_nodes = [[0, 1, 2], [3, 4, 5]]
    aligned_displacements = np.random.rand(6, 3)
    with patch("matplotlib.pyplot.show"):
        visualize_deformation(None, deformed_coords, element_nodes, aligned_displacements)

def test_visualize_all_success(tmpdir):
    # Full set of required files
    n = 6
    np.save(str(tmpdir.join("checkerboard.npy")), np.random.rand(5, 5))
    np.save(str(tmpdir.join("node_coords.npy")), np.random.rand(n, 3))
    np.save(str(tmpdir.join("node_labels.npy")), np.arange(n))
    np.save(str(tmpdir.join("displacements.npy")), np.random.rand(n, 3))
    np.save(str(tmpdir.join("disp_node_labels.npy")), np.arange(n))
    np.save(str(tmpdir.join("element_connectivity.npy")), np.array([[0,1,2],[3,4,5]]))
    with patch("matplotlib.pyplot.show"):
        visualize_all(str(tmpdir), scale_factor=1)
```

### Acceptance criteria
- [ ] `pytest --cov=src --cov-branch` reports ≥ 70% total coverage
- [ ] No new test marks `# pragma: no cover` except `main()` in `model.py` (hardcoded data path)
- [ ] All existing 35 tests continue to pass

---

## 🔴 ~~Fix CI Linting Failures~~ ✅ COMPLETE (2026-06-01)

**Was:** CI failing with 2 failing + 3 cancelled jobs due to:
1. Double comma `C0116,,C0103` in `pylint.yml` `--disable` list (pylint parse error)
2. `Pillow` missing from pip install → `E0401` import errors for PIL
3. `model_gui_test_case.py` not ignored (hardcoded Windows paths + module-level execution)
4. `.coveragerc` at `.github/workflows/.coveragerc` but referenced as `.coveragerc` from root
5. `tk` listed as a pip package (does not exist — should be `python3-tk` via apt)

**Fixed files (2026-06-01):**

| File | Change |
|---|---|
| `.github/workflows/pylint.yml` | Fixed double comma; added `Pillow`; added `model_gui_test_case.py` to ignore; added `black` + `flake8` steps; expanded matrix to 3.9/3.10/3.11 |
| `.github/workflows/test_code.yml` | Added `python3-tk` apt step; replaced `tk` with `Pillow pandas requests`; set `MPLBACKEND=Agg`; fixed `.coveragerc` path |
| `.coveragerc` | Moved from `.github/workflows/` to repo root; switched to forward-slash paths |
| `.flake8` | New file — sets `max-line-length=120`, ignores `E402,E203,E226,W503`, excludes dataset/test-case scripts |
| `shotpeen_gui.py` | Added `self.splash_image = None` in `__init__` (W0201); fixed E302 blank lines; removed dead commented-out code; applied `black` |
| `src/peen-ml/data_viz.py` | Fixed E302 blank line before `visualize_all`; replaced `range(len())` with `enumerate` (C0200); stripped trailing whitespace in docstring; applied `black` |
| `src/peen-ml/model.py` | Fixed `{epoch+1}` → `{epoch + 1}` in f-strings (E226); applied `black` |
| `tests/test_model.py` | Expanded from 4 → 12 tests: added `NormalizedDataset`, `SpatialAttention`, `DisplacementPredictor` forward, `smape`, `create_model`, skip-missing branch |
| `tests/test_data_viz.py` | Expanded from 5 → 10 tests: added `visualize_checkerboard` smoke, missing element_connectivity, `visualize_all` abort path |
| `tests/test_shotpeen_gui.py` | Expanded from 4 → 13 tests: added `check_file_in_folder`, `get_file_path`, `num_of_simulations`, `check_install` paths |

**How to verify locally:**
```bash
# from shotpeeningML/
PYTHONPATH=$(pwd)/src/peen-ml MPLBACKEND=Agg pytest tests/ -q          # 35 passed
python -m black --line-length=120 --check $(git ls-files '*.py' | grep -v dataset)
python -m flake8 $(git ls-files '*.py')
```

---

## Context

The current pipeline has two model architectures:

- **DisplacementPredictor** — CNN encoder → `Linear(512, N×3)` FC output. Scales linearly
  with node count; causes CUDA OOM at N=1,002,001 (full Dataset_Gaussian).
- **ConvDecoderPredictor** — same encoder → convolutional decoder → `(3, H, W)` field →
  bilinear sample at node coords. 178× fewer parameters, node-count-agnostic. Currently
  the best available architecture (val MSE ~2.3e-05 after 50 epochs on Dataset_Gaussian_2601).

### Key dataset facts
- Checkerboard input: 20×20, **78% zeros** (sparse — shots hit a small patch)
- Displacement output: 2601 nodes (subsampled) or 1,002,001 (full), **84% zeros**
- Nodes lie on a regular 51×51 grid (X outer, Y inner) in [0,1]×[0,1]
- Training set: 200 simulations, 140 train / 30 val / 30 test

### Current file locations
```
src/peen-ml/model.py              — DisplacementPredictor, ConvDecoderPredictor,
                                    train_save_gui, train_save_conv_gui,
                                    curved_surface_inference (Layers 1-3)
src/peen-ml/stl_surface.py        — STLSurface: normals, KDTree, checkerboard
src/peen-ml/nozzle_trajectory.py  — raster_scan, spiral_scan, zigzag_scan, from_csv
src/peen-ml/curved_surface_sim.py — run_curved_surface_sim
src/peen-ml/data_viz.py           — visualize_stl_deformation, visualize_stl_stress
shotpeen_gui.py                   — GUI: Generate / Train / Load+Evaluate tabs
Dataset_Gaussian_2601/            — 200 sims, 2601 nodes (51×51 subsampled), G=20
Dataset_Gaussian_2601/saved_model/           — trained DisplacementPredictor
Dataset_Gaussian_2601/saved_model_conv/      — trained ConvDecoderPredictor
```

---

## Upgrade Candidates

Ranked by expected impact vs implementation effort.

---

### 1. Fourier Neural Operator (FNO)
**Priority: High**

**Why:** Designed specifically for PDE-governed physics fields (elasticity, fluid flow).
Learns a mapping between function spaces in the frequency domain rather than point-to-point.
Shot peening deformation is governed by elastic PDEs — FNO is a principled fit.

**Key properties:**
- Operates via FFT → learned spectral filter → IFFT
- Captures global interactions in O(N log N) vs O(N²) for attention
- Resolution-invariant: same operator works at 20×20, 51×51, or 1001×1001
- Proven on Navier-Stokes, elasticity benchmarks

**Implementation plan:**
1. Install `neuraloperator` (`pip install neuraloperator`) or implement a minimal FNO block
2. Replace the ConvDecoder decoder with FNO layers:
   - Encoder output: (B, 128, 20, 20)
   - FNO layers: spectral conv + pointwise conv residual, keep spatial dims
   - Upsample to (H, W) and project to 3 channels
3. Train on Dataset_Gaussian_2601 with same `train_save_conv_gui` entry point
4. Compare val MSE vs ConvDecoderPredictor baseline (2.3e-05)

**Files to create/modify:**
- `src/peen-ml/model.py` — add `FNODecoderPredictor` class
- `src/peen-ml/fno_layers.py` — SpectralConv2d, FNOBlock (optional separate file)
- `tests/test_model.py` — add FNO shape and forward-pass tests

---

### 2. Implicit Neural Representation (INR / SIREN)
**Priority: High**

**Why:** Represents the displacement field as a continuous function f(x,y) → (ux,uy,uz)
conditioned on a latent code from the checkerboard. Solves N-scaling completely —
the INR MLP has fixed size (~10K params) regardless of mesh resolution.
Naturally smooth (SIREN's sin activations match elastic field behaviour).

**Implementation plan:**
1. CNN encoder(checkerboard) → latent z (e.g. 256-dim vector via global average pool)
2. INR MLP: `[x, y, z_latent] → [ux, uy, uz]` with sin activations (SIREN)
   - 4 hidden layers × 256 units ≈ 200K params total
3. Training: for each sample, randomly sample K node positions per forward pass
   (e.g. K=512), evaluate INR at those coords, compute MSE vs ground truth
   — no need to evaluate all 2601 nodes every step
4. Inference: evaluate INR at all N node coords (any mesh, any N)

**Key design decisions to test:**
- Latent z dimension: 64 vs 128 vs 256
- SIREN (sin) vs ReLU activations in INR
- Random node subsampling during training vs full grid
- Whether to condition via concatenation or FiLM (feature-wise linear modulation)

**Files to create/modify:**
- `src/peen-ml/model.py` — add `SIRENPredictor`, `INRDecoder` classes
- `tests/test_model.py` — forward pass, coordinate sampling, shape tests

---

### 3. VAE Latent Compression
**Priority: Medium**

**Why:** Compresses the (N,3) displacement field to a small latent z (e.g. 32-dim).
CNN then predicts z instead of the full field (32 outputs vs N×3). Also gives
uncertainty estimation by sampling z ~ N(μ, σ).

**Implementation plan:**
1. Train a VAE offline on the displacement fields:
   - Encoder: (2601,3) → reshape (3,51,51) → conv layers → (μ, log σ) ∈ R^32
   - Decoder: z ∈ R^32 → deconv layers → (3,51,51) → (2601,3)
   - Loss: MSE reconstruction + KL divergence
2. Freeze the VAE decoder
3. Train CNN to predict z from checkerboard: Linear(128*G*G, 32)
4. At inference: CNN → z → VAE decoder → displacement field

**Key metrics:**
- VAE reconstruction loss (lower bound on achievable accuracy)
- CNN→z→decode pipeline MSE vs direct ConvDecoder

**Files to create/modify:**
- `src/peen-ml/model.py` — add `DisplacementVAE`, `VAEConditionedPredictor`
- `tests/test_model.py` — VAE encode/decode roundtrip, latent dim tests

---

### 4. RAFT-style Iterative Refinement (Optical Flow)
**Priority: Medium**

**Why:** Optical flow = per-pixel displacement vector field from an image. Structurally
identical to checkerboard → displacement field. RAFT's iterative refinement (coarse
prediction + repeated residual corrections) could improve accuracy on the impact zone
boundary where the displacement field has steep gradients.

**Key ideas to borrow:**
- **Correlation volume**: cross-correlate encoder features at different spatial offsets
  to explicitly model where shots landed vs where deformation appears
- **Iterative refinement**: predict base field, add small residuals over T iterations
  (each iteration is cheap — only the residual is predicted)
- **ConvGRU**: maintain a hidden state across refinement iterations

**Implementation plan:**
1. Encode checkerboard → feature map F (B, 128, 20, 20)
2. Initialize displacement field d_0 = zeros (B, 3, H, W)
3. For t in range(T=4):
   - Look up features at current displaced coordinates
   - ConvGRU update: hidden, Δd = gru(hidden, F, d_t)
   - d_{t+1} = d_t + Δd
4. Return d_T

**Files to create/modify:**
- `src/peen-ml/model.py` — add `RAFTStylePredictor`, `ConvGRU`

---

### 5. Sparse Convolutions (Encoder)
**Priority: Low-Medium**

**Why:** Checkerboard is 78% zeros. Standard dense convolutions compute on all 400 cells
even though only ~88 are non-zero. Sparse convolutions only compute at non-zero locations.

**Implementation plan:**
1. Install `spconv` or `MinkowskiEngine`
2. Replace `conv1/conv2/conv3` in the encoder with sparse conv equivalents
3. Convert checkerboard to sparse tensor before forward pass
4. Keep the decoder (ConvDecoder or FNO) unchanged — densify after encoder

**Caveat:** adds a non-trivial C++ dependency. Evaluate if the 78% sparsity
actually gives meaningful speedup on a 20×20 input (only 400 cells — may not be
worth the complexity at this resolution).

**Files to modify:**
- `src/peen-ml/model.py` — add `SparseEncoder` as an optional encoder swap

---

### 6. Patch Tokenization / Perceiver IO
**Priority: Low**

**Why:** Transformer-based approach. Divide 20×20 checkerboard into patches (e.g. 4×4),
produce 25 tokens. Cross-attend from a set of learned query points (or mesh node coords)
to checkerboard tokens. Perceiver IO makes the output size arbitrary.

**Useful if:** the dataset grows large enough to benefit from transformer-scale training.
With 200 training samples, attention may not have enough data to outperform the conv
baseline.

**Revisit when:** dataset size > 1000 simulations.

---

## Suggested Experimental Order

1. **FNO** — highest physics motivation, cleanest swap for the ConvDecoder's decoder half
2. **INR/SIREN** — solves N-scaling most elegantly, enables inference at the full 1M-node mesh
3. **VAE** — good for uncertainty quantification, complementary to any of the above
4. **RAFT refinement** — if FNO/INR accuracy on impact-zone boundaries is insufficient
5. **Sparse encoder** — last, only if profiling shows encoder compute is a bottleneck

---

## Baseline to Beat

| Metric | Value |
|---|---|
| Architecture | ConvDecoderPredictor |
| Parameters | 170,458 |
| GPU memory (weights + Adam) | ~2 MB |
| Val MSE (Dataset_Gaussian_2601, 50 epochs) | 2.34e-05 |
| Test MSE (Simulation_199, held-out) | 1.18e-04 |
| STL inference method | Bilinear sampling (no RBF needed) |
| Training time (50 epochs, RTX 4090 Laptop) | ~3 min |

Any new architecture should be compared against these numbers on the same
Dataset_Gaussian_2601 train/val/test split (seed=2024) before considering it an improvement.

---

## ~~Material Properties — GUI, Dataset Generation, and Inference~~ ✅ COMPLETE

**Priority: High** ~~(impacts simulation accuracy more than architecture choice)~~
**Completed: 2026-05-31** — see Completed table at top of this file.

### Problem

Material properties are largely hardcoded and invisible to the user.  Changing
the shot or workpiece material changes every number the physics engine produces —
yet there is no UI surface for it and no link to trusted property sources.

**Current state by location:**

| Where | Shot properties | Workpiece properties | GUI-exposed? |
|---|---|---|---|
| `impact_sim.py:96–105` (`ShotPeenParams`) | `E_s=210e9, nu_s=0.30, rho_s=7800` (steel) | `E_b=113.8e9, nu_b=0.34, sigma_yield=276e6, c=3e9` (Ti-6Al-4V) | **No** |
| `multi_shot_sim.py:129` (`MultiShotParams`) | via `base_params` (ShotPeenParams) | same | **No** |
| `curved_surface_sim.py:149–153` (`CurvedSurfaceSimParams`) | `shot_material="steel", D=0.0005` | same Ti-6Al-4V defaults | **No** |
| `native_dataset_gen.py` | randomises V, D only | yield stress range 200–400 MPa | V, D via GUI |
| `gaussian_nozzle_dataset_gen.py:231–233` | 4 shot types (dict) | yield range 200–800 MPa | **No** |
| `shotpeen_gui.py` | V, D sliders only | nothing | partially |

The four shot-material dicts in `gaussian_nozzle_dataset_gen.py:143–149`
(`SHOT_MATERIALS`) are the only structured property store in the codebase.
Everything else is a scalar literal.

---

### What to build

#### 7a. In-code material property library

Create `src/peen-ml/materials.py` with a `MATERIALS` dict covering both shot
and workpiece types.  Each entry should include provenance (source URL/DOI):

```python
WORKPIECE_MATERIALS = {
    "Ti-6Al-4V":    {"E": 113.8e9, "nu": 0.34, "sigma_yield": 880e6, "c": 3.0e9,
                     "source": "MatWeb / Granta MI — AMS 4928"},
    "316L-SS":      {"E": 193e9,   "nu": 0.27, "sigma_yield": 170e6, "c": 5.0e9,
                     "source": "NIST SRD 171 / Matweb"},
    "4340-Steel":   {"E": 205e9,   "nu": 0.29, "sigma_yield": 470e6, "c": 7.0e9,
                     "source": "ASM Handbook vol 2"},
    "Al-7075-T6":   {"E":  71.7e9, "nu": 0.33, "sigma_yield": 503e6, "c": 2.5e9,
                     "source": "MIL-HDBK-5J / MMPDS"},
    "Inconel-718":  {"E": 200e9,   "nu": 0.29, "sigma_yield": 1034e6,"c": 8.0e9,
                     "source": "Special Metals datasheet"},
}

SHOT_MATERIALS = {                        # ← move from gaussian_nozzle_dataset_gen.py
    "steel":     {"rho_s": 7800, "E_s": 210e9, "nu_s": 0.30,
                  "source": "ISO 11124-2 / SAE J827"},
    "ceramic":   {"rho_s": 6000, "E_s": 380e9, "nu_s": 0.22,
                  "source": "CoorsTek ZrO2 datasheet"},
    "glass":     {"rho_s": 2500, "E_s":  70e9, "nu_s": 0.22,
                  "source": "ASTM B851 / Potters Industries"},
    "cast_iron": {"rho_s": 7300, "E_s": 170e9, "nu_s": 0.26,
                  "source": "ASM Handbook vol 1"},
}
```

Open-source / freely accessible property databases to draw from:
- **[MatWeb](https://www.matweb.com)** — free searchable database, 175 000+ alloys
- **[NIST SRD 171 (Structural Materials)](https://www.nist.gov/srd/nist-standard-reference-database-171)** — Fe, Ni, Ti alloys with uncertainty bounds
- **[MMPDS / MIL-HDBK-5J](https://www.metallic-materials.net)** — aerospace-grade handbook values (public USAF data)
- **[ASM Handbook (public abstracts)](https://www.asminternational.org)** — citation-quality sources per alloy
- **[OpenAlloy](https://openalloy.org)** (community effort) — CC-licensed property records
- **[Springer Materials (open access tier)](https://materials.springer.com)** — DOI-linked property records

Each entry in `MATERIALS` must include a `"source"` string so that:
1. Users can verify the value before a production run
2. The provenance is logged in `simulation_params.txt` automatically

#### 7b. GUI material selector — Generate tabs

In `shotpeen_gui.py`, in both the Native and Gaussian Nozzle generator dialogs,
add a `ttk.LabelFrame` "Material Properties" section with:

- **Shot material** — `ttk.Combobox` populated from `SHOT_MATERIALS.keys()`
  (default: `"steel"`).  On selection, display `E_s`, `nu_s`, `rho_s`, and the
  source string in a read-only label beneath.
- **Workpiece material** — `ttk.Combobox` from `WORKPIECE_MATERIALS.keys()`
  (default: `"Ti-6Al-4V"`).  Display `E`, `nu`, `sigma_yield`, `c`, and source.
- **"Custom…" option** in both combos — expands text fields allowing the user to
  enter arbitrary values (with a visible warning: "Custom values have no
  provenance check").

Pass the selected material to the generator CLI/Python call so material
identity ends up in `simulation_params.txt`.

#### 7c. GUI material selector — Inference tab

In the Curved Surface Inference tab (Load Model dialog), show the same
two material combos.  The chosen properties are forwarded to
`CurvedSurfaceSimParams` (or the equivalent SIREN path) so the inference
simulation uses physically correct contact mechanics rather than the
hardcoded Ti-6Al-4V defaults.

#### 7d. Native dataset generator: propagate material CLI args

Add to `native_dataset_gen.py`:
```
--workpiece_material  [name from WORKPIECE_MATERIALS or "custom"]
--E_b FLOAT           # override Young's modulus (Pa)
--nu_b FLOAT          # override Poisson's ratio
--sigma_yield FLOAT   # override yield stress (Pa)
--c FLOAT             # override bilinear hardening modulus (Pa)
--shot_material       [name from SHOT_MATERIALS or "custom"]
--E_s FLOAT  --nu_s FLOAT  --rho_s FLOAT
```

If `--workpiece_material` is a named entry, load from `MATERIALS` dict.
If `"custom"`, require all four override flags.
All resolved values logged to `simulation_params.txt`.

#### 7e. Output: `simulation_params.txt` must log material provenance

Each simulation folder already writes `simulation_params.txt`.  Extend it to
include the resolved material block:
```
workpiece_material : Ti-6Al-4V
  E_b              : 113800000000 Pa
  nu_b             : 0.34
  sigma_yield      : 880000000 Pa
  c                : 3000000000 Pa
  source           : MatWeb / Granta MI — AMS 4928
shot_material      : steel
  E_s              : 210000000000 Pa
  nu_s             : 0.30
  rho_s            : 7800 kg/m³
  source           : ISO 11124-2 / SAE J827
```

---

### Files to create / modify

| File | Change |
|---|---|
| `src/peen-ml/materials.py` | **Create** — `WORKPIECE_MATERIALS`, `SHOT_MATERIALS` dicts with provenance |
| `src/peen-ml/impact_sim.py` | Import from materials.py; `ShotPeenParams` defaults look up library |
| `src/peen-ml/curved_surface_sim.py` | `CurvedSurfaceSimParams` defaults from library |
| `src/peen-ml/native_dataset_gen.py` | Add `--workpiece_material` / `--shot_material` CLI args; log provenance |
| `src/peen-ml/gaussian_nozzle_dataset_gen.py` | Replace local `SHOT_MATERIALS` dict with import from materials.py |
| `shotpeen_gui.py` | Material combos in Generate tabs and Inference tab |
| `tests/test_materials.py` | **Create** — verify all entries have required keys + valid value ranges |

---

### Testing checklist

- [ ] `WORKPIECE_MATERIALS` and `SHOT_MATERIALS` each have `≥ 5` entries
- [ ] Every entry has `E`, `nu`, `sigma_yield`/`E_s`/`nu_s`/`rho_s` and `source` keys
- [ ] `source` string is non-empty for every entry (provenance required)
- [ ] `simulation_params.txt` contains material section after dataset gen
- [ ] GUI combo correctly populates from the dict at runtime
- [ ] Selecting a different workpiece changes the output of `impact_sim.py`
- [ ] Custom override values flow through without being overwritten by defaults

---

## Testing Checklist for Each New Architecture

- [ ] Forward pass produces correct output shape
- [ ] Trains without NaN loss for 5 epochs
- [ ] Val MSE lower than ConvDecoder baseline (2.34e-05) or within 2× with a compensating
      benefit (fewer params, uncertainty, resolution-free)
- [ ] `curved_surface_inference()` works with new model (add `isinstance` branch in model.py)
- [ ] `load_and_evaluate_model_gui()` works with new model
- [ ] GUI Preview Deformation renders without error
- [ ] At least 3 pytest tests added to `tests/test_model.py`
