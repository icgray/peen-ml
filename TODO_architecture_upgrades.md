# peen-ml Architecture Upgrade TODO

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

## Testing Checklist for Each New Architecture

- [ ] Forward pass produces correct output shape
- [ ] Trains without NaN loss for 5 epochs
- [ ] Val MSE lower than ConvDecoder baseline (2.34e-05) or within 2× with a compensating
      benefit (fewer params, uncertainty, resolution-free)
- [ ] `curved_surface_inference()` works with new model (add `isinstance` branch in model.py)
- [ ] `load_and_evaluate_model_gui()` works with new model
- [ ] GUI Preview Deformation renders without error
- [ ] At least 3 pytest tests added to `tests/test_model.py`
