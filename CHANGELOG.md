# Changelog

All notable changes to peen-ml are documented here.

## [0.1.0] — 2026-06-06

Initial public release.

### Added
- Physics simulator (`impact_sim.py`, `multi_shot_sim.py`): Python-native Shen & Atluri (2006)
  elastic-plastic impact model; ~2 s per simulation on a laptop CPU, no commercial FEA required.
- Three CNN surrogate architectures in `model.py`:
  - `DisplacementPredictor` — fully connected decoder, baseline (~30 M parameters).
  - `ConvDecoderPredictor` — lightweight convolutional decoder with FiLM material conditioning
    (~170 K parameters, 178× fewer than FC baseline, resolution-agnostic inference).
  - `SIRENPredictor` — sinusoidal implicit neural representation for million-node meshes.
- `InfluenceFieldDataset` and `train_influence_field_model`: physics-derived 4-channel input
  (Hertz depth, shot KDE, lateral forces Fx/Fy) achieving Pearson r = 0.95–0.97 in-plane.
- `ImprovedDisplacementPredictor` with per-simulation displacement normalization.
- Material library (`materials.py`): 5 aerospace workpiece alloys × 5 shot media, 25 combinations.
- 10-dimensional material + process conditioning vector (V, D, n scalars + 7 material properties).
- Three-layer mesh interpolation: bilinear resize → TPS-RBF → Rodrigues rotation for STL surfaces.
- STL curved-surface pipeline (`stl_surface.py`, `curved_surface_sim.py`).
- Nozzle trajectory planner (`nozzle_trajectory.py`): raster, spiral, zigzag, and custom CSV paths.
- Multi-task predictor (`MultiTaskPredictor`) with displacement + cupping + stress heads.
- Tkinter GUI (`shotpeen_gui.py`): Generate Dataset / Train Model / Load & Evaluate panels.
- Large-scale training pipeline (`large_scale_train.py`) with AMP, per-sim normalization,
  and multi-material conditioning (MatCond model, 25 combos, 5000 simulations).
- Dataset generation: `native_dataset_gen.py` with multiprocessing parallelism.
- Visualization: `data_viz.py`, `make_pred_gt_fig.py` for publication-quality figures.
- Test suite (`tests/`) with 187+ pytest tests; CI via GitHub Actions (Pylint + pytest-cov).
- `CONTRIBUTING.md` with contribution guidelines and code style notes.
- `paper.md` + `paper.bib` for JOSS submission.

### Fixed
- Per-simulation normalization added to `InfluenceFieldDataset` and `ConvDecoder` loaders,
  replacing global-max normalization that suppressed low-amplitude simulations.
- `train_conv_decoder` RMSE diagnostic corrected (was scaling by 1e6 on pre-normalized values).
- `mat_dim` conditioning vector extended from 7 to 10 to include V, D, n_shots.
- Headless Tkinter and Agg matplotlib backend configured for CI environments.
