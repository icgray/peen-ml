# JOSS Paper Publication TODO — peen-ml

**Target journal:** [Journal of Open Source Software (JOSS)](https://joss.theoj.org/)
**JOSS submission docs:** [joss.readthedocs.io/en/latest/submitting.html](https://joss.readthedocs.io/en/latest/submitting.html)
**Paper format spec:** [joss.readthedocs.io/en/latest/paper.html](https://joss.readthedocs.io/en/latest/paper.html)
**Reviewer checklist:** [joss.readthedocs.io/en/latest/review_checklist.html](https://joss.readthedocs.io/en/latest/review_checklist.html)

> **Note on format:** JOSS requires `paper.md` (Markdown) + `paper.bib` (BibTeX/BibLaTeX), NOT LaTeX.
> If you are maintaining a `.tex` version (e.g. for a conference proceedings alongside JOSS),
> use this checklist for both — the required sections and figures are identical.
> Keep `paper.md` as the submission artifact. The `.tex` file can mirror it.

---

## SESSION NOTES — last updated 2026-06-06 (influence field pressure test)

### What was done this session

**New architecture: node-resolution influence fields (4-channel ConvDecoder)**
- Added `compute_influence_fields()` to `multi_shot_sim.py`: 4 physics-kernel channels at FEM node resolution (51×51): Ch0 Hertz depth, Ch1 shot KDE, Ch2 lateral-force Fx, Ch3 Fy
- Added `InfluenceFieldDataset`, `create_influence_field_loaders`, `train_influence_field_model` to `model.py`
- Backfilled all 7 datasets (200/300/2000-sim; 6900 total sims, 0 failures)
- Trained and evaluated influence field ConvDecoder on all 5 key material combos
- Fixed 4 bugs: MT displacement denorm, MT input channel routing (physics CB vs density CB vs influence fields), multi-channel figure save, evaluate_on_dataset `isinstance(dict)` for MultiTaskPredictor

**Influence field results vs checkerboard baseline:**

| Material | D ux r (CB) | I ux r (IF) | D uz r (CB) | I uz r (IF) | ux RMSE gain |
|---|---|---|---|---|---|
| Ti+steel | 0.374 | **0.780** | 0.135 | **0.585** | 13.9 → 6.1 µm |
| 316L+ceramic | 0.593 | **0.652** | 0.078 | **0.442** | 52.5 → 31.4 µm |
| Inconel+tungsten | 0.434 | **0.570** | 0.115 | **0.451** | 22.8 → 18.7 µm |
| Al+glass | 0.371 | **0.796** | 0.163 | **0.563** | 7.5 → 3.5 µm |
| 4340+cast_iron | 0.521 | **0.675** | 0.078 | **0.424** | 35.6 → 22.9 µm |

**Phased training fix for MultiTaskPredictor** (`train_model_multitask`):
- Added `warmup_disp_epochs=20` (displacement-only for first 20 epochs)
- Reduced λ_s from 0.05 → 0.005
- `stress_components=2` (S11+S22 only; S33/S12 zero in biaxial peening)
- Cupping r=0.82 confirmed on 200 Ti+steel sims (MT model, existing weights)
- MT displacement head not yet retrained with new settings (still shows r=0.31)

**Cross-material generalization** (test: train on A, evaluate on B):
- Ti+steel → Al+glass: r=0.74 (good spatial transfer, low RMSE)
- Ti+steel → 316L+ceramic: r=0.86 pattern / RMSE=58µm (pattern transfers, scale doesn't)
- Al+glass → Ti+steel: r=0.83 (best cross-material result)
- 316L+ceramic → Ti+steel: r=0.49 (asymmetric; high-V training biases model)

---

### HOLES FOUND — pressure test findings (priority ordered)

#### HOLE 1 — Influence field normalization loses V/D amplitude information [HIGH]

**Root cause:** Ch0 (Hertz depth) and Ch1 (KDE) are normalized per-sim to [0,1]. A sim with V=10 m/s (tiny dents, r_p≈0.07mm) and a sim with V=80 m/s (large dents, r_p≈1.38mm) both get `max=1.0` in Ch0. The model cannot distinguish them from the spatial pattern alone.

**Evidence:**
- Ti+steel V<20 m/s: mean ux r=0.60 vs V>60 m/s: mean ux r=0.88 (gap = 0.28)
- 316L+ceramic V<20 m/s: mean ux r=0.42 vs V>60: r=0.74 (gap = 0.32)
- a_p varies 18× across Ti+steel sims (0.017–0.312 mm, CV=54%)
- r_p varies 18× (0.074–1.380 mm); both are set by V, which is randomized across [10,80] m/s

**Fix options:**
- [ ] Add V, D, r_p, delta_p as scalar conditioning inputs appended to the ConvDecoder latent (similar to mat_dim conditioning already in ConvDecoderPredictor). Expected: closes V<20 gap from 0.18 to ~0.05.
- [ ] OR: store the unnormalized Hertz field in Ch0 (physical units: meters) and normalize globally across the training set. Then the model sees absolute dent depth, not relative pattern. Simpler but requires recomputing influence_fields.npy.

#### HOLE 2 — 316L+ceramic has 5000× ux dynamic range; disp_scale suppresses 46% of sims [HIGH]

**Root cause:** `disp_scale = global max(|disp|)` = 2.245e-3 m. For low-V/few-shot 316L sims, ux_std ≈ 0.5–2 µm → normalized target ≈ 2e-4. These 92/200 sims (46%) contribute near-zero MSE and effectively don't train the model. During evaluation, these sims have near-zero GT and the model predicts near-zero → r is undefined or negative due to tiny numerics.

**Evidence:**
- 316L ux_std: min=0.21 µm, p5=0.92 µm, median=24.8 µm, p95=430 µm, max=1122 µm
- 46% of sims have ux_std < 1% of disp_scale (signal lost in normalization)
- 19/200 sims have negative predicted r (model anti-correlated with GT)
- Same pattern in Inconel+W (19 neg-r) and 4340+cast_iron (10 neg-r)
- Ti+steel and Al+glass have 0 neg-r sims → their dynamic range is smaller

**Fix options:**
- [ ] Per-sim normalization: divide each sim's displacements by that sim's `max(|disp|)` before stacking. Save the per-sim scales for inference denormalization. Training signal is uniform. Main cost: model can no longer predict absolute amplitude from the pattern alone — must add V/D conditioning for that.
- [ ] Log-amplitude weighting in the loss: weight each sim's MSE contribution by `1/sim_disp_std` so low-amplitude sims are not swamped. Keeps global disp_scale but rebalances gradients.
- [ ] Filter training set: exclude sims where `ux_std / disp_scale < threshold` (e.g., 0.005). Removes ~46% of 316L data but makes the model honest about its operating range.

#### HOLE 3 — MT displacement head still running old settings, uz not retrained [MEDIUM]

**Root cause:** MT model in `LargeScaleRun1/Models/MT_MultiTask_Ti_Steel` was trained before the phased-warmup fix. Uses λ_s=0.05 with no warmup. MT ux r=0.31 vs single-task r=0.37 on same data.

**Fix:**
- [ ] Retrain MT model with phased warmup using:
  ```bash
  python large_scale_train.py --phase train --skip-gen --output LargeScaleRun1 \
      --n_sims 200 --no-multimat --no-hires --no-siren --epochs 100
  ```
  This will use the new `warmup_disp_epochs=20`, `loss_weights=(1.0, 0.005, 0.01)`, `stress_components=2`.
- Expected: MT ux r 0.31 → 0.37+ (matches single-task), retains cupping r=0.82.

#### HOLE 4 — ConvDecoder boundary: no physics-informed boundary condition [LOW-MEDIUM]

**Root cause:** ConvDecoder uses `nn.Upsample(bilinear)` as its decoder, which has no knowledge of plate boundary conditions. Real physics: free edges have zero traction (ux/uy displacement is unconstrained, but no force applied at boundary). The bilinear upsampler doesn't enforce this.

**Evidence:** Relative edge error = 26.1% vs center error = 32.6% — edges are slightly better than center, so boundary isn't a systematic failure. But the asymmetry (corner bias = +0.64 µm in Ti+steel) suggests the decoder has learned an implicit correction, not a physics-consistent one.

**Fix:**
- [ ] Pad influence fields at boundaries with zeros (already the case for shots outside domain) — low cost, may reduce corner artifacts.
- [ ] Add a boundary-loss term during training: enforce that predicted displacement at the 4 clamped-edge nodes matches the constraint (zero displacement at a fixed node, if any). Requires knowing which nodes are constrained.
- [ ] Lower priority: flag for camera-ready but not blocking.

#### HOLE 5 — Cross-material RMSE misleading: scale mismatch on unseen materials [MEDIUM]

**Root cause:** Cross-material r is high (0.74–0.89) because spatial pattern transfers well. But cross-material RMSE is large (58–69 µm) because the model was trained with one material's disp_scale and predicts in normalized space for the new material. The normalization_stats.npy stores the training material's disp_scale.

**Evidence:**
- Ti+steel → 316L: ux r=0.86 but RMSE=58.5 µm (316L disp_scale is 4.9× larger)
- Al+glass → 316L: ux r=0.89 but RMSE=69.3 µm

**Fix:**
- [ ] For cross-material deployment, apply the target material's disp_scale at inference. Document this explicitly in the API.
- [ ] Add material conditioning (V, D as scalars) so one model can generalize to multiple materials with correct amplitude calibration.
- [ ] For the paper: report cross-material r separately from cross-material RMSE. Explain that the r measures spatial pattern fidelity and the RMSE measures absolute amplitude accuracy — they require separate discussion.

#### HOLE 6 — 2000-sim and HighRes influence field models not yet trained [LOW]

**Context:** All datasets have been backfilled:
- `Dataset_Al_7075_T6__glass_2000`: 2000 sims, influence_fields.npy present in all
- `Dataset_Ti_6Al_4V__steel_2000`: 2000 sims, influence_fields.npy present in all
- `Dataset_HighRes_Ti_Steel_300`: 300 sims at Nx=100 (101×101 grid), influence_fields present

**Fix:**
- [ ] Train influence field model on Al+glass 2000: expected to show whether scaling from 200→2000 sims improves r with the better encoding (previous checkerboard models showed flat scaling).
- [ ] Train on Ti+steel 2000 similarly.
- [ ] Train high-res model: ConvDecoder(input_channels=4, out_H=101, out_W=101) on HighRes dataset — higher output resolution may improve uz RMSE.

#### HOLE 7 — sMAPE metric: ~150% throughout, meaningless, should be dropped [LOW]

**Root cause:** sMAPE = 2|y-ŷ|/(|y|+|ŷ|) is undefined when both y=0 and ŷ=0 (returns NaN via clamping to 0/1e-8 = 0), and is ~200% when y≈0 but ŷ≠0 (or vice versa). Since the displacement field has large near-zero regions (nodes far from all shots), sMAPE is dominated by these undefined cases.

**Evidence:** All trained models show sMAPE 141–163% regardless of Pearson r improvement from 0.37 to 0.78.

**Fix:**
- [ ] Remove sMAPE from `large_scale_results.csv`, `run_eval.py`, and all paper tables.
- [ ] Replace with: Pearson r (primary), RMSE (µm), and relative RMSE = RMSE/max(|GT|) (normalized comparison across materials with different scales).

---

## SESSION NOTES — last updated 2026-06-05

### What was done this session
- Created `stress_test_benchmark.py`: generates all 25 material-combo datasets + 5 special configs, trains DisplacementPredictor on each, detects anomalies automatically.
- Created `tests/test_stress.py`: 187 pytest tests across all 25 material combos, checkerboard patterns, edge cases, physics plausibility. All pass in ~15 s.
- Ran comprehensive accuracy analysis against the Shen-Atluri ground truth on `Dataset_bench_final` (500 sims, 75 test cases).

### What needs to be fixed next session (and why it is not good enough)

#### 1. MODEL ACCURACY IS TOO WEAK FOR PUBLICATION — highest priority

The current ConvDecoderPredictor accuracy on the test set (500-sim dataset, 31×31 nodes, 10×10 checkerboard) is:

| Metric | Node-level | Cell-averaged |
|---|---|---|
| RMSE | 25 µm (median 22.7) | 15.6 µm (median 12.9) |
| Pearson r | **0.197 mean / 0.118 median** | 0.332 mean / 0.303 median |
| Relative RMSE | ~20% of peak | ~11% of peak |

Per-component breakdown (>5% threshold):
- ux / uy (in-plane): r = 0.72 — **good pattern capture**
- uz (out-of-plane): r = 0.156 mean / **0.118 median** — **too weak**

**Why this is not good enough:** r = 0.12 for the primary output (uz, surface depth) is barely above random. A reviewer will reject the paper or benchmark section on these numbers alone. The model predicts a spatially smooth uz field while the ground truth has fine-grained shot-placement variation that the 10×10 input simply cannot encode. The cell-averaged correlation (r ≈ 0.33) is the correct figure of merit for this input resolution, but even that is modest.

**What to fix:**
- [ ] Increase checkerboard resolution: try G=20×20 or G=30×30 (match mesh resolution better). More spatial information in → more spatial information out.
- [ ] Train on a larger dataset: 500 sims is marginal. Target 2000–5000 sims for the benchmark. The stress_test_benchmark.py script generates them; just run with `--n_sims 200` per combo.
- [ ] Use the material-conditioned path (`load_material_features=True`, `mat_dim=7`): the current benchmark never turns this on. Including material features may improve generalisation across the 25 combos.
- [ ] Report cell-averaged r as the primary metric in the paper (not node-level). Be explicit that the input resolution limits node-level accuracy.
- [ ] Consider: train separate models per component (ux/uy vs. uz) since they have different physical scaling — uz is dominated by contact mechanics while ux/uy are dominated by coverage pattern.

#### 2. pred_vs_gt.png IS MISLEADING — fix before paper submission

`images/pred_vs_gt.png` currently shows **Simulation_498**, which is the **99th-percentile case** (r = 0.617). The median test sim has r ≈ 0.12. A reviewer who runs the benchmark will immediately notice the discrepancy.

**What to fix:**
- [ ] Regenerate `pred_vs_gt.png` to show the **median-r test simulation** (sort test sims by Pearson r on uz, pick the p50 case). The `train_bench.py` `med_idx` logic picks by array position, not by r value — change it to `idx = np.argsort(r_vals)[len(r_vals)//2]`.
- [ ] Show a 4-panel figure: best case / median case / worst case / scatter plot of r vs. peak displacement — so readers can calibrate expectations honestly.
- [ ] Caption must state the metric is computed on affected nodes (>5% of peak) and report the **distribution** (mean ± std or p10/p50/p90), not a single case.

#### 3. THREE CODE BUGS — must fix before submission

Found by the stress test suite this session:

**Bug A — `NormalizedDataset` crashes with IndexError when n_sims < 7**
- Location: `model.py`, `NormalizedDataset.__init__`, line `self._has_mat = len(base_dataset[0]) == 3`
- When `n_sims < 7`, the 15% val split rounds to 0 elements. `base_dataset[0]` on an empty Subset raises `IndexError` inside `create_data_loaders`.
- Fix: guard with `if len(base_dataset) > 0` before accessing element 0. Raise a clear error (`ValueError: val split has 0 samples — need at least 7 simulations`) instead of a cryptic IndexError.
- Test that documents the bug: `tests/test_stress.py::TestDegenerateDatasetSplit`

**Bug B — `DisplacementPredictor.forward(x, None)` crashes when `mat_dim > 0`**
- Location: `model.py`, `DisplacementPredictor.forward`, FC layer shape mismatch
- If a material-conditioned model (`mat_dim=7`) receives `mat=None`, the FC layer gets `128*G*G` inputs but expects `128*G*G + 7`. RuntimeError, no informative message.
- Fix: add a guard in `forward()`: `if mat is None and self.mat_dim > 0: raise ValueError(f"Model was built with mat_dim={self.mat_dim} but mat=None was passed.")`.
- Test: `tests/test_stress.py::TestModelArchitectureVariants::test_material_conditioning_none_raises_hole`

**Bug C — `smape()` returns NaN for zero displacements**
- Location: `model.py`, `smape()` function
- When both y_true and y_pred are near zero (happens with sparse checkerboards or low-velocity shots on hard materials), `|y_true - y_pred| / ((|y_true| + |y_pred|) / 2)` = 0/0 = NaN. This silently corrupts the sMAPE metric in every evaluation that includes near-zero nodes.
- Fix: add `denominator = denominator.clamp(min=1e-8)` before the division, so smape returns 0.0 for zero-displacement nodes instead of NaN.
- Test: `tests/test_stress.py::TestSmapeNaNHole`

#### 4. BENCHMARK SCRIPT SHOULD RUN THE FULL 25-COMBO SWEEP

`stress_test_benchmark.py` exists but has only been smoke-tested with 1–2 configs and 8 simulations. Before the next accuracy table is written:
- [ ] Run the full sweep: `python stress_test_benchmark.py --n_sims 50 --epochs 40` to get baseline numbers for all 25 material combos.
- [ ] Check whether any material combo systematically fails or shows anomalous accuracy.
- [ ] Record the output `BenchmarkResults/benchmark_report.csv` and commit it — this becomes the evidence for the Research Impact section.

#### 5. MINIMUM n_sims GUARD IS MISSING FROM THE GUI

The GUI calls `train_save_gui(data_path)`, which calls `create_data_loaders`, which crashes with IndexError if the user generates < 7 simulations (Bug A above). There is no user-facing validation. After fixing Bug A, add a check in `train_save_gui`:
```python
sim_folders = [d for d in os.listdir(data_path) if d.startswith("Simulation_")]
if len(sim_folders) < 7:
    raise ValueError(f"Dataset has only {len(sim_folders)} simulations. "
                     "Minimum 7 required for a non-empty validation split.")
```

---

## 0. Pre-Submission Repository Gates (BLOCKING — must pass before submitting)

- [ ] Repository has been **public for at least 6 months** with commits distributed across that period
  - *Current state: check `git log --format="%ad" --date=short | tail -1` to verify first commit date*
- [ ] Development history shows **iterative commits** over months — not a single burst
- [ ] **Zenodo DOI** created: archive a tagged release at [zenodo.org](https://zenodo.org) and link the DOI badge in README
- [ ] **Tagged release** exists (`git tag v0.1.0` or similar) with changelog/release notes
- [ ] `CONTRIBUTING.md` file added — must describe how to contribute, file issues, and seek support
- [ ] Issue tracker is publicly accessible and allows external users to open issues
- [ ] `LICENSE` file is a **plain-text OSI-approved license file** in the repo root (MIT already exists — confirm it is not just mentioned in README)
- [ ] Evidence of **community engagement**: external issues, PRs, citations, or adoptions

---

## 1. Paper File Requirements

- [ ] Create `paper.md` at the repo root (JOSS submission file)
- [ ] Create `paper.bib` (BibTeX bibliography)
- [ ] Add all figure files to repo alongside source code
- [ ] Paper length: **750–1750 words** (excluding references and figures)
  - Papers over 1750 words will be asked to cut down
  - Target ~1000–1400 words for comfortable fit

### 1.1 YAML Metadata Header (required at top of `paper.md`)

```yaml
---
title: 'peen-ml: A Machine Learning Surrogate for Shot Peening Deformation Prediction'
tags:
  - Python
  - shot peening
  - machine learning
  - surrogate model
  - manufacturing simulation
  - convolutional neural network
  - finite element analysis
authors:
  - name: Harshavardhan Sameer Raje
    orcid: 0000-XXXX-XXXX-XXXX
    affiliation: 1
  - name: Onest Rexhepi
    orcid: 0000-XXXX-XXXX-XXXX
    affiliation: 1
  - name: Jiachen Zhong
    orcid: 0000-XXXX-XXXX-XXXX
    affiliation: 1
  - name: Xuanyu Shen
    orcid: 0000-XXXX-XXXX-XXXX
    affiliation: 1
affiliations:
  - name: Paul G. Allen School of Computer Science & Engineering, University of Washington, USA
    index: 1
    ror: 00cvxb145
date: 04 June 2026
bibliography: paper.bib
---
```

- [ ] Each author: obtain ORCID iD at [orcid.org](https://orcid.org) — free registration
- [ ] Confirm UW ROR ID (`00cvxb145`) or look up at [ror.org](https://ror.org)
- [ ] Tags: 5–8 keywords covering domain and technique

---

## 2. Required Paper Sections

### 2.1 Summary
- [ ] 1–2 paragraphs, ~150–200 words
- [ ] Written for a **non-specialist, diverse audience** — minimize jargon
- [ ] State: what peen-ml does, what problem it solves, who it is for
- [ ] Example angle: "peen-ml is a Python library that replaces multi-hour finite element analyses of shot peening with CNN-based surrogate predictions that complete in seconds, enabling engineers to rapidly explore process parameters without FEA software."

### 2.2 Statement of Need
- [ ] Clearly state: what problems the software solves
- [ ] Identify target users: process engineers, manufacturing researchers, aerospace/automotive industry
- [ ] Quantify the pain: typical FEA shot peening simulation time vs. peen-ml inference time
- [ ] Contrast with existing approaches:
  - Commercial FEA (Abaqus, LS-DYNA): accurate but slow, expensive license, no rapid iteration
  - Analytical models (Hertz contact, Shen & Atluri): fast but single-shot, no spatial field output
  - Existing ML surrogates: mostly academic, no open GUI, no curved surface support
- [ ] Cite the Shen & Atluri (2006) model that underpins the simulator `[@shen2006]`

### 2.3 State of the Field
- [ ] Survey competing or related tools (cite each):
  - Abaqus/LS-DYNA scripted FEA — gold standard, no open source equivalent at same fidelity
  - strucscan `[@strucscan2022]` — materials simulation framework (Python), different domain
  - SMT Surrogate Modeling Toolbox — general surrogates, no shot peening physics
  - OpenFOAM — CFD/shot peening fluid dynamics, not deformation field prediction
- [ ] Answer the "build vs. contribute" question: why a new tool vs. contributing to an existing one
- [ ] Highlight unique capabilities: integrated GUI + physics simulator + three CNN architectures + STL curved surface support + material library — no single existing tool combines these

### 2.4 Software Design
- [ ] Describe the **three ML architectures** and the design trade-offs:
  - `DisplacementPredictor` — 30 M params, fixed mesh, FC output; baseline
  - `ConvDecoderPredictor` — 170 K params, resolution-agnostic, **178× fewer parameters**; recommended for production
  - `SIRENPredictor` — implicit neural representation, memory-safe for very large meshes
- [ ] Describe the **three-layer interpolation system** (bilinear resize → TPS-RBF → Rodrigues rotation)
- [ ] Describe the **physics simulator** (Shen & Atluri elastic-plastic impact model, ~2 s/simulation, no Abaqus required)
- [ ] Describe the **material library** (`materials.py`) and material conditioning (`mat_dim=7`)
- [ ] Describe the **STL/curved surface pipeline** and nozzle trajectory planner
- [ ] Mention the **GUI** as the primary access point for non-ML users
- [ ] Note: do NOT include API docs here — those belong in code documentation

### 2.5 Research Impact Statement
- [ ] Provide evidence of realized impact OR credible near-term impact:
  - If cited: list any papers that use peen-ml, with DOIs
  - If adopted: list groups/organizations using it
  - If neither yet: provide benchmark showing speedup (FEA hours → peen-ml seconds), reproducible numbers, and describe specific research workflows it enables
  - Note any workshop presentations, course adoption, or industrial interest
- [ ] Aspirational statements alone are insufficient — need at least one concrete, verifiable claim
- [ ] Consider: add a reproducible benchmark table (speedup, MSE, sMAPE) as the impact evidence

### 2.6 AI Usage Disclosure
- [ ] Required section (added by JOSS in 2026)
- [ ] Disclose: which generative AI tools were used (if any), where (code generation, documentation, paper authoring), nature/scope of assistance, and confirmation that all AI-assisted outputs were reviewed by a human author
- [ ] If no AI tools were used: state that explicitly

### 2.7 Acknowledgements
- [ ] Acknowledge: University of Washington CSE 583 course and instructors
- [ ] Acknowledge: any financial support, grants, or compute resources
- [ ] Acknowledge: Shen & Atluri (2006) foundational model

### 2.8 References (`paper.bib`)
- [ ] Shen & Atluri (2006) — CMC: Computers, Materials & Continua
- [ ] Cite PyTorch (for CNN training)
- [ ] Cite NumPy, SciPy, Matplotlib, trimesh (software dependencies used in the research)
- [ ] Cite at least 2–3 FEA/shot peening literature references
- [ ] Cite relevant competing tools (strucscan, SMT toolbox, or others compared in State of Field)
- [ ] Use **full journal names** (not abbreviations) in all references
- [ ] Verify BibTeX syntax compiles with Pandoc

---

## 3. Figures and Plots

JOSS has minimal hard figure specs but reviewers expect publication-quality visuals.
All figures must live in the repo alongside `paper.md`.

### 3.1 Format Guidelines
- [ ] **Preferred formats:** PDF (vector, best for line plots/diagrams) or PNG at ≥300 DPI (for raster)
- [ ] **Do not use:** JPEG for scientific figures (lossy compression causes artifacts on sharp lines)
- [ ] Figure files: place in `paper/figures/` or `images/` alongside `paper.md`
- [ ] Reference in Markdown: `![Caption text.](figures/fig1.png){ width=80% }` (width is optional)
- [ ] Each figure must appear **alone in its own paragraph**
- [ ] Captions: must be **informative and self-contained** — a reader should understand the figure without reading the body text
- [ ] Labels for cross-referencing: `![Caption](fig.png){#fig:label}`, referenced as `@fig:label`

### 3.2 Suggested Figures for peen-ml Paper (aim for 3–5 total)

- [ ] **Figure 1 — System Overview Diagram**
  - Schematic showing the full pipeline: shot peening parameters → physics simulator → dataset → CNN training → inference → deformation field
  - Format: vector diagram (draw in draw.io or matplotlib, export PDF/SVG)
  - Size: full column width (~3.5 in wide for single column JOSS layout)

- [ ] **Figure 2 — Architecture Comparison**
  - Side-by-side diagram of the three CNN architectures (DisplacementPredictor, ConvDecoderPredictor, SIRENPredictor)
  - Or: table/plot comparing parameter counts (30 M vs. 170 K vs. 2 M) and inference accuracy
  - Include attention block schematic if space allows

- [ ] **Figure 3 — Prediction vs. Ground Truth**
  - Side-by-side color maps: ground truth FEA displacement field vs. CNN prediction
  - Use `data_viz.py` to generate; export at 300 DPI PNG or as PDF
  - Include colorbar with units (mm or µm displacement)
  - Caption should include the MSE or sMAPE value for the shown case

- [ ] **Figure 4 — Curved Surface / STL Deformation**
  - 3D STL surface colored by predicted displacement magnitude (from `visualize_stl_deformation`)
  - Shows the unique STL capability not present in other tools
  - Export: PNG at 300 DPI, use a white or transparent background

- [ ] **Figure 5 — Training/Benchmark Results (optional but strongly recommended)**
  - Loss curves (train vs. validation) for each architecture
  - Or: benchmark table/bar chart — FEA simulation time vs. peen-ml inference time, and accuracy (MSE/sMAPE) for each architecture
  - This directly supports the Research Impact section

### 3.3 Plot Style Checklist (for all figures)
- [ ] Font size: minimum 9 pt for axis labels, 10 pt for titles (legible after scaling to column width)
- [ ] Use colorblind-safe colormaps: `viridis`, `plasma`, `cividis` (avoid pure `jet`/rainbow)
- [ ] Include units on all axes and colorbars
- [ ] Remove chartjunk: no unnecessary gridlines, no 3D bar charts for 2D data
- [ ] All text in figures must be editable (not rasterized text in PNG) — use vector export when possible
- [ ] DPI: set `plt.savefig("fig.png", dpi=300, bbox_inches="tight")` in Matplotlib

---

## 4. Software/Repository Checklist

These are reviewed independently of the paper by JOSS reviewers.

### 4.1 Documentation
- [ ] `README.md` provides a high-level overview (already exists — good)
- [ ] Installation instructions include automated package management (`pip install .` — already exists)
- [ ] Example usage covers a real-world problem end-to-end (GUI walkthrough exists — good)
- [ ] API-level documentation: add docstrings to public functions in `model.py`, `materials.py`, `data_viz.py`, `nozzle_trajectory.py`

### 4.2 Tests
- [ ] Automated test suite exists (pytest — already exists — good)
- [ ] CI runs tests on push (add test runner to `.github/workflows/`)
  - Currently have Pylint CI; add a `pytest` workflow
- [ ] Code coverage badge in README (already configured with codecov — good)
- [ ] All tests pass on a clean install

### 4.3 Community Guidelines
- [ ] Create `CONTRIBUTING.md` with:
  - How to file a bug report (link to GitHub Issues)
  - How to propose a feature
  - How to submit a pull request
  - Code style guide (PEP 8 — already noted in README)
  - How to seek support (e.g. GitHub Discussions or email)

### 4.4 Release & Archive
- [ ] Create a `CHANGELOG.md` or GitHub Releases entry for v0.1.0
- [ ] Create a Zenodo archive: go to [zenodo.org](https://zenodo.org), link GitHub repo, trigger a release
- [ ] Add Zenodo DOI badge to `README.md`
- [ ] Record the Zenodo DOI for the `paper.md` YAML metadata

---

## 5. Similar Published JOSS Papers (Reference for Content and Style)

Study these papers for structure, figure style, and tone:

| Paper | DOI | Why relevant |
|-------|-----|--------------|
| strucscan — Python framework for high-throughput material simulation | [10.21105/joss.04719](https://joss.theoj.org/papers/10.21105/joss.04719) | Python simulation tool for materials; similar domain and audience |
| Foundry-ML — ML datasets in materials science | [10.21105/joss.05467](https://joss.theoj.org/papers/10.21105/joss.05467) | ML + materials science Python tool; good Statement of Need model |
| ParMOO — parallel multiobjective simulation optimization | [10.21105/joss.04468](https://joss.theoj.org/papers/10.21105/joss.04468) | Optimization tool with simulation interface; Software Design section model |
| GWSurrogate — Python surrogate models | [10.21105/joss.07073](https://joss.theoj.org/papers/10.21105/joss.07073) | Surrogate model package; directly analogous software category |

**What to look for in each:** length of each section, how they quantify impact, how figures are captioned, how they handle the "build vs. contribute" justification.

---

## 6. Submission Steps (in order)

- [ ] Complete all sections above
- [ ] Run `pandoc paper.md --bibliography paper.bib --citeproc -o paper.pdf` locally to verify PDF compiles
- [ ] Create a tagged release (e.g. `v0.1.0`) on GitHub
- [ ] Archive on Zenodo and record the DOI
- [ ] Submit at [joss.theoj.org](https://joss.theoj.org) — no submission fee
  - Required fields: title, description, GitHub repo URL, paper branch/path, programming language, software version, editor suggestions (optional)
- [ ] **Response timeline after submission:**
  - Respond to reviewer feedback within **2 weeks**
  - Complete all requested changes within **4–6 weeks**
  - Allow editors **1 week** before following up

---

## 7. No Hard Deadline

JOSS operates on a rolling basis with no submission deadlines. Review timelines are typically 1–3 months from submission to acceptance, depending on reviewer availability and revision cycles.

---

## Sources

- [JOSS Paper Format](https://joss.readthedocs.io/en/latest/paper.html)
- [JOSS Submission Guide](https://joss.readthedocs.io/en/latest/submitting.html)
- [JOSS Review Checklist](https://joss.readthedocs.io/en/latest/review_checklist.html)
- [JOSS Review Criteria](https://joss.readthedocs.io/en/latest/review_criteria.html)
- [JOSS Reviewer Guidelines](https://joss.readthedocs.io/en/latest/reviewer_guidelines.html)
- [strucscan JOSS Paper](https://joss.theoj.org/papers/10.21105/joss.04719)
- [Foundry-ML JOSS Paper](https://joss.theoj.org/papers/10.21105/joss.05467)
- [GWSurrogate JOSS Paper](https://joss.theoj.org/papers/10.21105/joss.07073)
