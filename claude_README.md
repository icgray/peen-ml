# PeenML — Claude Reference README

**Project:** CSE583_ShotPeenWithML
**Version:** 0.1.0
**Authors:** Onest Rexhepi, Harshavardhan Sameer Raje, Jiachen Zhong, Xuanyu Shen
**Repo:** https://github.com/onestr1/CSE583_ShotPeenWithML
**Last documented:** 2026-04-04

---

## Project Overview

PeenML is a machine learning pipeline that predicts sheet-metal deformation caused by shot peening — a manufacturing process where a surface is blasted with small pellets to induce compressive residual stress and controlled bending. The goal is to replace expensive Abaqus FEA dynamic simulations with a fast CNN inference step.

The high-level workflow is:

1. **Dataset generation** — Abaqus scripts run hundreds of FEA simulations, each with a different checkerboard pattern of peening intensity (thermal expansion coefficient), and save the resulting nodal displacements and stresses as `.npy`/`.csv` files.
2. **Model training** — A CNN with channel and spatial attention (`DisplacementPredictor`) is trained to map a checkerboard input image to a flat displacement field over all mesh nodes.
3. **Inference & visualization** — A Tkinter GUI lets users train a new model or load a saved one, run inference on new checkerboard inputs, and visualize deformed meshes and stress fields.

---

## Repository Structure

```
peen-ml/
├── shotpeen_gui.py              # Main entry point — Tkinter GUI application
├── pyproject.toml               # Build configuration and project metadata
├── requirements.txt             # Pinned conda/pip environment (Windows, wide-character encoded)
├── src/
│   └── peen-ml/
│       ├── __init__.py          # Package init (empty)
│       ├── model.py             # CNN model definition, training, and evaluation
│       ├── data_viz.py          # Mesh and field visualization utilities
│       ├── impact_sim.py        # Python-native shot-peen impact simulation (Shen & Atluri 2006)
│       ├── taichi_impact_sim.py # MLS-MPM numerical physics simulation (Taichi backend)
│       ├── dataset1_script.py   # Abaqus script — Dataset 1 (random checkerboard, 2000 sims)
│       ├── dataset2_script.py   # Abaqus script — Dataset 2 (single-square, 100 sims)
│       ├── model_gui_test_case.py  # Standalone evaluation test harness
│       ├── model_notebook_v2.ipynb # Jupyter exploration notebook (v2)
│       └── model_notebook_v3.ipynb # Jupyter exploration notebook (v3)
└── tests/
    ├── __init__.py
    ├── conftest.py              # Shared pytest fixtures (session-scoped, 5×5 checkerboards)
    ├── test_model.py            # Unit tests for model.py  (~70 tests)
    ├── test_data_viz.py         # Unit tests for data_viz.py  (~35 tests)
    ├── test_shotpeen_gui.py     # Unit tests for shotpeen_gui.py  (~40 tests)
    ├── test_impact_sim.py       # Unit tests for impact_sim.py  (~110 tests)
    └── test_taichi_impact_sim.py # Unit tests for taichi_impact_sim.py  (~80 tests)
```

---

## Script Reference

---

### `shotpeen_gui.py`

**Location:** `shotpeen_gui.py` (project root)
**Role:** Main entry point. Launches the Tkinter GUI that wraps the training and evaluation pipeline for non-technical users.
**Run with:** `python shotpeen_gui.py`

**Dependencies:**
- Standard library: `tkinter`, `sys`, `subprocess`, `shutil`, `os`, `threading`
- Third-party: `torch`, `PIL` (Pillow)
- Internal: `model.py` (`train_save_gui`, `load_and_evaluate_model_gui`, `create_data_loaders`, `create_model`, `train_model`), `data_viz.py` (`visualize_checkerboard`, `visualize_all`)

**How it works:**

The script appends `src/peen-ml` to `sys.path` at runtime so that `model.py` and `data_viz.py` can be imported without installing the package. A splash image (`src/peen-ml/bullet_bill.png`) is displayed on the main menu. The `App` class manages all GUI state.

**Standalone functions:**

| Function | Signature | Description |
|---|---|---|
| `check_install` | `(package_id: str)` | Attempts to import a package and, if missing, installs it via `pip` then falls back to `conda`. Currently commented out at startup. |

**`App` class — method reference:**

| Method | Key Args | Description |
|---|---|---|
| `__init__` | `root_tk` | Sets window title/size to `800×600`, initializes state variables, calls `main_menu()`. |
| `main_menu` | — | Clears all widgets, renders splash image and two buttons: **Train Model** → `train_model_dialog`, **Load Model** → `load_model_dialog`. |
| `get_file_path` | `relative_path` | Returns absolute path compatible with both development mode and PyInstaller `.exe` (uses `sys._MEIPASS` if present). |
| `train_model_dialog` | — | Opens a `Toplevel` dialog with a folder-picker for training data, a scrollable log `Text` widget, a `Progressbar`, and a **Train** button that calls `train_save_gui(data_folder)` from `model.py`. |
| `load_model_dialog` | — | Opens a `Toplevel` dialog with pickers for: model `.pth` file, peen-intensity folder (checkerboard `.npy` files), and output directory. Buttons trigger preview, evaluate, and deformation visualization. |
| `browse_file` | `variable` | Opens `filedialog.askopenfilename()` and stores the result in the `StringVar`. |
| `browse_directory` | `variable` | Opens `filedialog.askdirectory()` and stores the result in the `StringVar`. |
| `preview_file` | `folder_path` | Validates the folder exists and is non-empty, then spawns a thread calling `run_preview()`. Shows `messagebox` errors on failure. |
| `run_preview` | `geometry_folder_path` | Calls `visualize_checkerboard(geometry_folder_path)` from `data_viz.py` on a background thread. |
| `preview_deformation` | `test_folder_path, deformation_folder_path` | Copies `node_coords.npy`, `node_labels.npy`, and `disp_node_labels.npy` from the test folder into the output folder, then calls `visualize_all()` if `displacements.npy` is already present (i.e., evaluate was run first). |
| `check_file_in_folder` | `folder_path, file_name` | Returns `True`/`False` for whether `file_name` exists inside `folder_path`. |
| `train_model` | `data_folder` | Programmatic (non-GUI) training path: calls `create_data_loaders`, `create_model`, then `train_model` from `model.py` with hardcoded `num_nodes=5202`, `lr=0.001`. Shows a completion `messagebox`. |
| `start_training` | `log_widget, progress_bar` | Writes "Training started…" to the log and starts an indeterminate `Progressbar`. Schedules `finish_training()` after 3 s (simulated — production training is triggered directly via `train_save_gui`). |
| `finish_training` | `log_widget, progress_bar` | Stops the `Progressbar` and writes "Training completed!" to the log. |
| `num_of_simulations` | `folder_path` | Counts subdirectories whose names match the pattern `Simulation_<integer>`. Returns the count. |

**Entry point:**
```python
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
```

---

### `src/peen-ml/model.py`

**Location:** `src/peen-ml/model.py`
**Role:** Core ML module. Defines the dataset classes, attention modules, CNN architecture, and all training/evaluation/saving utilities.
**Author:** Jiachen Zhong (Dec 10, 2024)

**Dependencies:**
- Standard library: `os`, `pathlib`
- Third-party: `numpy`, `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`, `matplotlib`

---

#### Data Loading

**`load_all_npy_files(base_folder, load_files, skip_missing)`**

Scans `base_folder` for subdirectories named `Simulation_<N>` (sorted numerically), loads the requested `.npy` files from each, and returns a dict of stacked numpy arrays.

| Arg | Type | Default | Notes |
|---|---|---|---|
| `base_folder` | `str` | — | Path containing `Simulation_*` subdirs |
| `load_files` | `tuple` | `("checkerboard", "displacements")` | File stems to load (without `.npy`) |
| `skip_missing` | `bool` | `True` | If `False`, raises `FileNotFoundError` on missing files |

Returns: `dict` mapping file stem → `np.ndarray` of shape `(N_sims, ...)`.

---

#### Dataset Classes

**`CheckerboardDataset(Dataset)`**

Wraps paired numpy arrays of checkerboard patterns and displacements. `__getitem__` adds a channel dimension to the checkerboard tensor (`(1, H, W)`) and returns `(checkerboard_tensor, displacement_tensor)`.

**`NormalizedDataset(Dataset)`**

Wraps a `CheckerboardDataset` and applies min-max normalization to the checkerboard channel using the global min/max computed at construction time. Displacements are returned unchanged.

---

#### Attention Modules

**`ChannelAttention(nn.Module)`**

Implements squeeze-and-excitation style channel attention. Uses both global average pooling and global max pooling, passed through two 1×1 convolutions and a sigmoid gate, then multiplies the input feature map.

| Constructor Arg | Default | Notes |
|---|---|---|
| `channels` | — | Number of input feature channels |
| `reduction` | `16` | Channel compression ratio |

**`SpatialAttention(nn.Module)`**

Concatenates channel-wise average and max projections along the channel axis, passes through a 7×7 conv, and applies a sigmoid gate to weight spatial positions.

---

#### CNN Model

**`DisplacementPredictor(nn.Module)`**

Three-stage convolutional encoder with CBAM-style attention at each stage, followed by two fully connected layers for regression.

| Constructor Arg | Description |
|---|---|
| `input_channels` | Number of input channels (1 for grayscale checkerboard) |
| `num_nodes` | Number of FEA mesh nodes (typically `5202`) |

Architecture summary:
- **Conv1:** 1→32 channels, 3×3, BatchNorm, ReLU → ChannelAttention(32) → SpatialAttention
- **Conv2:** 32→64 channels, 3×3, BatchNorm, ReLU → ChannelAttention(64) → SpatialAttention
- **Conv3:** 64→128 channels, 3×3, BatchNorm, ReLU → ChannelAttention(128) → SpatialAttention
- **FC:** `128×5×5 → 512 → num_nodes×3`
- **Output shape:** `(batch_size, num_nodes, 3)` — (x, y, z) displacement per node

---

#### Utility Functions

**`create_model(input_channels, num_nodes)`**
Instantiates and returns a `DisplacementPredictor`.

**`create_data_loaders(base_folder, load_files, skip_missing, batch_size)`**
Loads data, creates a `CheckerboardDataset`, splits 70/15/15 (train/val/test), wraps each split in `NormalizedDataset`, and returns `(train_loader, val_loader, test_loader, loaded_data)`. Seeds numpy and torch to `2024` for reproducibility. Default `batch_size=15`.

**`train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience)`**
Training loop with early stopping. Updates a live matplotlib plot of train/val loss each epoch. Returns `(train_losses, val_losses)`.

| Arg | Default | Notes |
|---|---|---|
| `epochs` | `10` | Max training epochs |
| `patience` | `5` | Early-stop patience (epochs without val improvement) |

**`smape(y_true, y_pred)`**
Computes the Symmetric Mean Absolute Percentage Error as a scalar tensor. Used alongside MSE during evaluation.

**`evaluate_model(model, test_loader, criterion)`**
Runs inference on the test set, prints MSE and sMAPE, and prints a sample of predicted vs ground-truth displacements for the first batch.

**`train_save_gui(data_path)`**
GUI-facing training entry point. Mirrors `main()` but skips the test evaluation and saves the trained model to `<data_path>/saved_model/trained_displacement_predictor_full_model.pth` using `torch.save(model, ...)`.

**`create_test_loader(test_data_path, load_files, batch_size)`**
Creates a `DataLoader` over the entire dataset in `test_data_path` (no split), normalized, `batch_size=1` by default.

**`evaluate_model_gui(model, test_loader, criterion, pred_save_dir)`**
Saves predictions for each simulation to `<pred_save_dir>/Simulation_<N>/pred_displacements.npy` and `.csv`. Reports MSE and sMAPE.

**`load_and_evaluate_model_gui(model_path, test_data_path, pred_save_dir)`**
GUI-facing evaluation entry point. Loads a saved `.pth` model with `torch.load`, creates the test loader, and calls `evaluate_model_gui`. This is what the **Evaluate Model** button in the GUI calls.

---

### `src/peen-ml/data_viz.py`

**Location:** `src/peen-ml/data_viz.py`
**Role:** Visualization module. Loads simulation `.npy` output files and renders checkerboard patterns, FEA meshes, stress fields, and deformation magnitudes via matplotlib.

**Dependencies:**
- Standard library: `os`
- Third-party: `numpy`, `matplotlib`, `matplotlib.collections`

**Run standalone:** `python data_viz.py` — executes `main()` which reads from `tests/simulation_0/`.

---

#### Function Reference

**`load_data(file_path, description="")`**
Loads a `.npy` file with `np.load`. Returns `None` and prints an error if loading fails (broad exception catch). Used as a safe wrapper throughout this module.

**`visualize_checkerboard(simulation_folder)`**
Loads `checkerboard.npy` from the given folder and renders it as a 6×6 inch `imshow` with the `viridis` colormap, labeled with expansion coefficient values.

**`compute_deformed_mesh(simulation_folder, scale_factor=1)`**
Loads `node_coords.npy`, `node_labels.npy`, `displacements.npy`, `disp_node_labels.npy`, and `element_connectivity.npy`. Aligns displacements to node indices by label, computes deformed coordinates as `node_coords + scale_factor * displacements`, and converts element connectivity to 0-indexed lists.

Returns: `(node_coords, deformed_coords, element_nodes)` or `(None, None, None)` on error.

**`visualize_mesh(node_coords, deformed_coords, element_nodes)`**
Renders the undeformed (gray) and deformed (blue) mesh as `LineCollection` objects on a 10×10 inch plot, drawing each element's edges in 2D (X-Y plane).

**`visualize_stress_field(simulation_folder, deformed_coords, element_nodes)`**
Loads `stresses.npy` and `stress_element_labels.npy`. Computes Von Mises stress as `sqrt(S11² - S11·S22 + S22² + 3·S12²)` and colors each element polygon using a `PolyCollection` with the `jet` colormap.

**`visualize_deformation(_, deformed_coords, element_nodes, aligned_displacements)`**
Computes per-node displacement magnitude via `np.linalg.norm`, averages over each element, and renders the result as a `PolyCollection` on the deformed mesh using the `plasma` colormap.

**`visualize_all(folder_path, scale_factor)`**
Convenience wrapper that runs steps 1–5 in sequence: checkerboard → deformed mesh computation → mesh visualization → stress field → deformation magnitude. Called by the GUI's **Predicted Deformation Preview** button.

**`main()`**
Calls `visualize_all` on `tests/simulation_0/` with `scale_factor=1`. Entry point for standalone use.

---

### `src/peen-ml/dataset1_script.py`

**Location:** `src/peen-ml/dataset1_script.py`
**Role:** Abaqus CAE Python script for generating **Dataset 1** — 2000 FEA simulations with randomly assigned checkerboard peening patterns on a 5×5 grid (25 cells, 4 possible expansion coefficients each).
**Execution environment:** Must be run inside Abaqus CAE (imports `abaqus`, `abaqusConstants`, `caeModules`, `driverUtils`, `visualization`). Not runnable as a standalone Python script.
**Output directory:** `U:\Shot Peening\Checkerboard\Dataset1_Random_Board\`

**What each simulation does:**

1. Creates a new Abaqus model database (`Mdb()`).
2. Sketches a 1m×1m square and creates two shell parts: **Sheet** (5mm thick aluminum, no expansion) and **Peen** (0.2mm thick, expansion coefficient varies by cell).
3. Partitions the Peen part into a 5×5 checkerboard grid (cell size 0.2m).
4. Randomly shuffles the 25 cells and cyclically assigns one of four expansion sections (`Peen-005`, `Peen-01`, `Peen-015`, `Peen-02`) — corresponding to coefficients 0.005, 0.01, 0.015, 0.02.
5. Saves the expansion coefficient grid as `checkerboard.npy` and `.csv`.
6. Creates assembly instances, applies a **Tie** constraint between Sheet-top and Peen-bottom, and applies displacement boundary conditions (fully fixed bottom-left corner; roll constraints at top-left and bottom-right).
7. Applies a uniform temperature field of 1.0 on the Peen, which drives thermal expansion and simulates peening.
8. Seeds and meshes both parts (element size 0.02m).
9. Submits and waits for the Abaqus job.
10. Opens the `.odb` results file, extracts nodal displacements (`U`), element stresses (`S`), node coordinates, and element connectivity.
11. Saves all extracted data as `.npy` and `.csv` in `Simulation_<idx>/`.
12. Moves the `.odb` file into the simulation folder and clears the model database.

**Output files per simulation:**

| File | Shape | Description |
|---|---|---|
| `checkerboard.npy` | `(5, 5)` | Expansion coefficient grid |
| `displacements.npy` | `(num_nodes, 3)` | Nodal displacements (U1, U2, U3) |
| `disp_node_labels.npy` | `(num_nodes,)` | Node labels matching displacements |
| `stresses.npy` | `(num_elements, 6)` | Element stresses (S11, S22, S33, S12, S13, S23) |
| `stress_element_labels.npy` | `(num_elements,)` | Element labels matching stresses |
| `node_labels.npy` | `(num_nodes,)` | All node labels |
| `node_coords.npy` | `(num_nodes, 3)` | All node coordinates |
| `element_labels.npy` | `(num_elements,)` | All element labels |
| `element_connectivity.npy` | `(num_elements, nodes_per_elem)` | Element-to-node connectivity |
| `<job>.cae` | — | Saved Abaqus model |
| `<job>.odb` | — | Abaqus output database |

**Material properties (both datasets):**
Aluminum: E = 68 GPa, ν = 0.36. Sheet has no expansion. Peen cells vary.

---

### `src/peen-ml/dataset2_script.py`

**Location:** `src/peen-ml/dataset2_script.py`
**Role:** Abaqus CAE Python script for generating **Dataset 2** — 100 FEA simulations exhaustively covering every combination of a single activated cell × 4 expansion coefficients on a 5×5 grid (4 coefficients × 25 cells = 100).
**Execution environment:** Same Abaqus requirement as `dataset1_script.py`.
**Output directory:** `U:\Shot Peening\Checkerboard\Method2\Dataset1_Single_Square\`

**Key differences from `dataset1_script.py`:**

- `num_simulations = 100` (do not change — exactly covers all combinations).
- All cells are first assigned `Peen-Zero` (expansion = 0.0), then only the target cell `(i, j)` is overridden with the active coefficient.
- Outer loops iterate over `expansion_data` (4 coefficients) × each cell `(i, j)` in the 5×5 grid.
- A zero-expansion material (`Aluminum-Zero`) is created and used as the default.
- `sample_idx` is a manually incremented counter (not the loop variable).
- The scratch directory for the job is set to `simulation_folder` (instead of `C:\temp`).

The output files per simulation have the same format and naming as Dataset 1.

---

### `src/peen-ml/model_gui_test_case.py`

**Location:** `src/peen-ml/model_gui_test_case.py`
**Role:** Lightweight standalone test harness for the GUI evaluation path (`load_and_evaluate_model_gui`). Useful for verifying a trained model works end-to-end without launching the full GUI.

**Usage:** Edit the three path constants and run with `python model_gui_test_case.py` from inside the `src/peen-ml/` directory (or after appending the correct path).

**Constants:**

| Constant | Description |
|---|---|
| `DATA_PATH` | Folder containing `Simulation_*` subdirs with test checkerboard/displacement data |
| `MODEL_PATH` | Path to a saved `.pth` model file |
| `PRED_SAVE_DIR` | Directory where predicted displacements will be written |

Calls `load_and_evaluate_model_gui(MODEL_PATH, DATA_PATH, PRED_SAVE_DIR)` directly.

---

### `src/peen-ml/__init__.py`

**Location:** `src/peen-ml/__init__.py`
**Role:** Package init file. Currently empty — marks the directory as a Python package for setuptools discovery.

---

## Test Suite

Tests use **pytest** and are in the `tests/` directory. Run with `pytest` from the project root.

### Running tests

```bash
# All tests
pytest tests/ -v

# Single file
pytest tests/test_model.py -v

# Specific class or test
pytest tests/test_model.py::TestSmape -v
pytest tests/test_model.py::TestSmape::test_identical_inputs_return_zero -v

# Show output for passing tests too
pytest tests/ -v -s
```

---

### `tests/conftest.py` *(shared fixtures)*

Shared pytest fixtures used across all three test files. All fixtures that involve disk I/O or model creation use `scope="session"` so they are built once per test run.

| Fixture | Scope | What it provides |
|---|---|---|
| `sim_folder` | session | A complete simulation folder with every `.npy` file the pipeline can read (5×5 checkerboard, 10 nodes, 4 quad elements, stresses). Dimensionally correct synthetic data. |
| `multi_sim_folder` | session | Base folder with 3 `Simulation_N` subdirectories, each with `checkerboard.npy` (5×5) and `displacements.npy`. Used by loader and training tests. |
| `shuffled_labels_sim_folder` | session | Simulation folder where `disp_node_labels` are in *reversed* order relative to `node_labels` — specifically for testing label-alignment correctness in `compute_deformed_mesh`. |
| `tiny_model` | session | A `DisplacementPredictor(input_channels=1, num_nodes=10)` in eval mode. |
| `tiny_loaders` | session | `(train_loader, val_loader, test_loader, loaded_data)` built from `multi_sim_folder`, batch size 1. |

---

### `tests/test_model.py`

**Tests:** `model.py` — all data, dataset, attention, model, training, and evaluation functions.

#### `TestLoadAllNpyFiles`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_runs_without_error` | Smoke | Loads the pre-built `test_simulations/` fixtures without crashing |
| `test_loads_correct_number_of_simulations` | One-shot | Stacked axis-0 size == 2 for a folder with 2 sims |
| `test_simulations_sorted_numerically` | One-shot | Folders are loaded in Simulation_0,1,2 order |
| `test_returns_none_for_empty_key` | One-shot | Empty base folder → value is `None` |
| `test_skip_missing_true_does_not_raise` | One-shot | Absent file is skipped gracefully |
| `test_skip_missing_false_raises_on_absent_file` | Edge | `skip_missing=False` raises `FileNotFoundError` |
| `test_loads_single_file_key` | Edge | Requesting one key works, other key absent from result |

#### `TestCheckerboardDataset`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_instantiation` | Smoke | Dataset creates without error |
| `test_length` | One-shot | `__len__` returns number of samples |
| `test_checkerboard_shape` | One-shot | Tensor has leading channel dim → `(1, H, W)` |
| `test_displacement_shape` | One-shot | Displacement tensor shape `(num_nodes, 3)` |
| `test_tensor_dtypes` | One-shot | Both tensors are `float32` |
| `test_single_sample_dataset` | Edge | Dataset of size 1 works |

#### `TestNormalizedDataset`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_instantiation` | Smoke | Wraps base dataset without error |
| `test_length_preserved` | One-shot | Same length as base dataset |
| `test_checkerboard_in_unit_range` | One-shot | All normalised values ∈ [0, 1] |
| `test_displacement_unchanged` | One-shot | Displacements pass through unmodified |
| `test_shape_preserved` | One-shot | Output shapes match base dataset |
| `test_constant_checkerboard_does_not_raise` | Edge | Uniform checkerboard (min==max) doesn't crash |

#### `TestChannelAttention`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_forward` | Smoke | Module forward pass without error |
| `test_output_shape_unchanged` | One-shot | Shape `(B, C, H, W)` is preserved |
| `test_custom_reduction_ratio` | One-shot | `reduction=4` still preserves shape |
| `test_output_values_bounded` | Property | Sigmoid gate → `|output| ≤ |input|` |
| `test_single_batch_element` | Edge | Batch size 1 works |

#### `TestSpatialAttention`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_forward` | Smoke | Forward pass without error |
| `test_output_shape_unchanged` | One-shot | All dimensions preserved |
| `test_works_with_varying_channel_counts` | One-shot | Channel-agnostic (no channel constructor arg) |

#### `TestDisplacementPredictor`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_forward` | Smoke | Forward pass on `(1,1,5,5)` input |
| `test_output_shape` | One-shot | Output is `(batch, num_nodes, 3)` |
| `test_output_dtype` | One-shot | Output is `float32` |
| `test_deterministic_in_eval_mode` | One-shot | Same input → same output in eval mode |
| `test_batch_independence` | Property | Per-item outputs match individual forward passes |
| `test_gradients_flow` | Property | `loss.backward()` produces non-zero gradients |
| `test_single_node_output` | Edge | `num_nodes=1` → `(1, 1, 3)` output |

#### `TestCreateModel`
| Test | Category | What it checks |
|---|---|---|
| `test_returns_displacement_predictor` | One-shot | Returns a `DisplacementPredictor` instance |
| `test_model_has_trainable_parameters` | Property | At least one parameter requires grad |

#### `TestCreateDataLoaders`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_returns_four_values` | Smoke | Returns 4-tuple without error |
| `test_loader_types` | One-shot | All three loaders are `DataLoader` instances |
| `test_batch_checkerboard_shape` | One-shot | Batched checkerboard is `(B, 1, 5, 5)` |
| `test_batch_displacement_shape` | One-shot | Batched displacement is `(B, num_nodes, 3)` |
| `test_loaded_data_dict_keys` | One-shot | Dict contains `checkerboard` and `displacements` |
| `test_train_split_is_largest` | Property | Train set ≥ val and test |
| `test_total_samples_preserved` | Property | train + val + test == total simulations |
| `test_batch_size_one` | Edge | `batch_size=1` produces single-sample batches |

#### `TestSmape`
| Test | Category | What it checks |
|---|---|---|
| `test_identical_inputs_return_zero` | One-shot | `smape(x, x) == 0` |
| `test_known_value` | One-shot | Handcrafted `(1→3)` → sMAPE = 1.0 |
| `test_returns_scalar` | One-shot | Output is 0-dimensional tensor |
| `test_symmetry` | Property | `smape(a,b) == smape(b,a)` |
| `test_non_negative` | Property | sMAPE ≥ 0 always |

#### `TestEvaluateModel`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_runs` | Smoke | Runs without crashing |
| `test_returns_float` | One-shot | Returns Python `float` |
| `test_mse_is_non_negative` | Property | MSE ≥ 0 |
| `test_perfect_prediction_gives_zero_mse` | Property | Identity model → MSE < 1e-10 |

#### `TestTrainModel`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_one_epoch` | Smoke | Training 1 epoch completes |
| `test_returns_loss_lists` | One-shot | Returns two lists of equal length |
| `test_early_stopping_fires` | One-shot | `patience=1` halts well before `max_epochs=20` |

#### `TestTrainSaveGui`
| Test | Category | What it checks |
|---|---|---|
| `test_model_file_created` | One-shot | `.pth` written to `<data_path>/saved_model/` |
| `test_saved_model_is_loadable` | Property | `torch.load` on the file returns a `DisplacementPredictor` |

#### `TestCreateTestLoader`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_runs` | Smoke | Runs without error |
| `test_returns_data_loader` | One-shot | Returns `DataLoader` instance |
| `test_all_data_included` | One-shot | Dataset size == total simulations (no split) |
| `test_batch_shape` | One-shot | First batch has correct shapes |

#### `TestEvaluateModelGui`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_runs` | Smoke | Runs without error |
| `test_creates_simulation_subdirectories` | One-shot | `Simulation_N/` dir created per batch |
| `test_npy_prediction_file_created` | One-shot | `pred_displacements.npy` exists per sim |
| `test_csv_prediction_file_created` | One-shot | `pred_displacements.csv` exists per sim |
| `test_prediction_shape_in_npy` | Property | Saved array has shape `(1, num_nodes, 3)` |
| `test_returns_float_mse` | One-shot | Returns float ≥ 0 |

#### `TestLoadAndEvaluateModelGui`
| Test | Category | What it checks |
|---|---|---|
| `test_end_to_end` | One-shot | Full train → save → load → evaluate → files written |
| `test_loaded_model_is_in_eval_mode` | Property | Model is in `.eval()` mode after loading |

---

### `tests/test_data_viz.py`

**Tests:** `data_viz.py` — data loading and all visualization functions.

All tests that open matplotlib figures patch `data_viz.plt.show` to prevent GUI windows during CI.

#### `TestLoadData`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_load_real_file` | Smoke | Loads a valid `.npy` from the shared fixture |
| `test_returns_correct_array` | One-shot | Loaded array `allclose` to saved array |
| `test_returns_none_for_nonexistent_file` | One-shot | Missing path → `None` (no exception) |
| `test_description_param_does_not_affect_return` | One-shot | Description is logging-only |
| `test_returns_none_for_empty_file` | Edge | Zero-byte file → `None` |
| `test_returns_none_for_corrupt_file` | Edge | Garbage bytes → `None` |
| `test_returns_correct_shape` | Property | Shape matches what was saved |

#### `TestVisualizeCheckerboard`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_runs_without_error` | Smoke | Runs on fixture; `plt.show` called once |
| `test_uses_real_simulation_folder` | One-shot | Runs on `tests/simulation_0/` |
| `test_missing_checkerboard_file_does_not_crash` | Edge | Absent file → silent return, no `plt.show` |
| `test_nonexistent_folder_does_not_crash` | Edge | Bad path → silent return |

#### `TestComputeDeformedMesh`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_returns_three_values` | Smoke | Returns 3-tuple |
| `test_returns_correct_node_coords` | One-shot | Returned coords match saved file |
| `test_deformed_coords_equals_coords_plus_displacements` | One-shot | `deformed = node_coords + disp` (aligned labels) |
| `test_scale_factor_doubles_deformation` | One-shot | `scale=2` → twice the displacement as `scale=1` |
| `test_scale_factor_zero_returns_original_coords` | One-shot | `scale=0` → `deformed == node_coords` |
| `test_element_nodes_is_list` | One-shot | Returns a Python list |
| `test_element_nodes_count_matches_connectivity` | Property | `len(element_nodes) == len(element_connectivity)` |
| `test_shuffled_label_alignment` | Property | Reversed disp labels are correctly re-indexed |
| `test_correct_data_with_real_simulation_0` | One-shot | Regression guard using `tests/simulation_0/` |
| `test_missing_node_coords_returns_none_triple` | Edge | Absent `node_coords.npy` → `(None,None,None)` |
| `test_missing_element_connectivity_returns_none_triple` | Edge | Absent connectivity → `(None,None,None)` |
| `test_empty_folder_returns_none_triple` | Edge | Completely empty folder → `(None,None,None)` |

#### `TestVisualizeMesh`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_runs_without_error` | Smoke | Runs on fixture data |
| `test_calls_plt_show` | One-shot | `plt.show` called exactly once |
| `test_identical_coords_does_not_crash` | Edge | Zero displacement (same coords both args) |

#### `TestVisualizeStressField`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_runs_without_error` | Smoke | Runs on fixture data |
| `test_smoke_with_real_simulation_0` | Smoke | Runs on `tests/simulation_0/` |
| `test_missing_stress_file_does_not_crash` | Edge | Absent stress files → silent return |

#### `TestVisualizeDeformation`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_runs_without_error` | Smoke | Runs on fixture data |
| `test_zero_displacement_does_not_crash` | One-shot | All-zero displacement renders without error |
| `test_large_displacement_values_do_not_crash` | Edge | Values of `1e30` don't overflow/crash |

#### `TestVisualizeAll`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_runs_without_error` | Smoke | Full pipeline on fixture folder |
| `test_smoke_with_real_simulation_0` | Smoke | Full pipeline on `tests/simulation_0/` |
| `test_calls_show_multiple_times` | One-shot | `plt.show` called ≥ 3 times |
| `test_scale_factor_two_does_not_crash` | One-shot | Non-unit scale factor accepted |
| `test_missing_required_file_exits_early` | Edge | Incomplete folder exits before crashing |

---

### `tests/test_shotpeen_gui.py`

**Tests:** `shotpeen_gui.py` — GUI app and all `App` methods.

#### `TestAppInitialisation`
| Test | Category | What it checks |
|---|---|---|
| `test_smoke_initialisation_does_not_crash` | Smoke | App creates without error |
| `test_window_title_set` | One-shot | Title is `"Model GUI"` |
| `test_window_geometry_set` | One-shot | Geometry is `"800x600"` |
| `test_window_size_attribute` | One-shot | `self.window_size == "800x600"` |
| `test_initial_data_path_is_empty_string` | One-shot | `test_train_data_path` starts empty |
| `test_parent_process_initially_none` | One-shot | `parent_process is None` at init |
| `test_missing_bullet_bill_raises_file_not_found` | Edge | Absent splash image → `FileNotFoundError` |

#### `TestGetFilePath`
| Test | Category | What it checks |
|---|---|---|
| `test_returns_string` | One-shot | Always returns `str` |
| `test_path_contains_relative_component` | One-shot | Result ends with the requested component |
| `test_uses_meipass_when_available` | One-shot | `sys._MEIPASS` used as base when present |
| `test_falls_back_to_cwd_without_meipass` | One-shot | Falls back to `os.path.abspath(".")` |

#### `TestCheckFileInFolder`
| Test | Category | What it checks |
|---|---|---|
| `test_returns_true_when_file_exists` | One-shot | `True` for a file that is present |
| `test_returns_false_when_file_absent` | One-shot | `False` for missing file |
| `test_returns_false_for_nonexistent_folder` | Edge | Non-existent folder → `False` |
| `test_case_sensitive_filename` | Property | Case-sensitive on Linux/macOS |

#### `TestNumOfSimulations`
| Test | Category | What it checks |
|---|---|---|
| `test_counts_correct_simulation_dirs` | One-shot | Counts exactly 3 `Simulation_N` dirs |
| `test_ignores_non_matching_directories` | One-shot | Non-matching dirs not counted |
| `test_empty_folder_returns_zero` | Edge | Empty folder → 0 |
| `test_mixed_digit_and_non_digit_suffixes` | Edge | `Simulation_1a` and `Simulation_` ignored |
| `test_large_index` | Property | `Simulation_9999` counted correctly |

#### `TestBrowseFile`
| Test | Category | What it checks |
|---|---|---|
| `test_sets_variable_on_selection` | One-shot | `StringVar.set` called with selected path |
| `test_does_not_update_variable_on_cancel` | Edge | Empty return → `set` not called |
| `test_dialog_is_called_once` | Property | Exactly one file dialog opened |

#### `TestBrowseDirectory`
| Test | Category | What it checks |
|---|---|---|
| `test_sets_variable_on_selection` | One-shot | `StringVar.set` called with selected path |
| `test_does_not_update_variable_on_cancel` | Edge | Empty return → `set` not called |
| `test_dialog_called_once` | Property | Exactly one directory dialog opened |

#### `TestTrainModel`
| Test | Category | What it checks |
|---|---|---|
| `test_nonexistent_directory_shows_error` | One-shot | `showerror` fires with correct message |
| `test_existing_directory_does_not_show_error` | One-shot | Existence error not shown for valid path |

#### `TestPreviewFile`
| Test | Category | What it checks |
|---|---|---|
| `test_nonexistent_path_shows_error` | One-shot | `showerror` fires for missing path |
| `test_file_path_instead_of_dir_shows_error` | Edge | File passed instead of directory → `showerror` |
| `test_empty_directory_shows_warning` | Edge | Empty dir → `showwarning` |
| `test_valid_directory_starts_thread` | One-shot | Non-empty dir starts background thread |

#### `TestPreviewDeformation`
| Test | Category | What it checks |
|---|---|---|
| `test_shows_error_when_displacement_file_missing` | One-shot | Absent `displacements.npy` → `showerror` |
| `test_copies_required_files_to_output` | One-shot | Required `.npy` files are copied to output folder |

#### `TestTrainingLogHelpers`
| Test | Category | What it checks |
|---|---|---|
| `test_start_training_logs_started_message` | One-shot | "started" appears in log text |
| `test_start_training_starts_progressbar` | One-shot | `progress.start()` called |
| `test_finish_training_stops_progressbar` | One-shot | `progress.stop()` called |
| `test_finish_training_logs_completed_message` | One-shot | "completed" appears in log text |

#### `TestCheckInstall`
| Test | Category | What it checks |
|---|---|---|
| `test_installed_package_does_not_call_subprocess` | One-shot | Existing package → no subprocess call |
| `test_missing_package_attempts_pip_install` | One-shot | Missing package → pip subprocess called |

---

## Key Data File Formats

All simulation output files follow a consistent schema across both datasets:

- **`checkerboard.npy`** — `float64 (5, 5)` — expansion coefficient for each of the 25 cells in the 5×5 peen pattern grid.
- **`displacements.npy`** — `float64 (num_nodes, 3)` — U1, U2, U3 nodal displacements from Abaqus.
- **`disp_node_labels.npy`** — `int (num_nodes,)` — Abaqus node labels aligned with `displacements.npy`.
- **`node_coords.npy`** — `float64 (num_nodes, 3)` — X, Y, Z coordinates of each mesh node.
- **`node_labels.npy`** — `int (num_nodes,)` — Abaqus node labels for the full mesh.
- **`element_connectivity.npy`** — `int (num_elements, nodes_per_element)` — node label indices per element.
- **`stresses.npy`** — `float64 (num_elements, 6)` — stress components (S11, S22, S33, S12, S13, S23).
- **`stress_element_labels.npy`** — `int (num_elements,)` — Abaqus element labels aligned with stresses.

---

### `impact_sim.py`

**Location:** `src/peen-ml/impact_sim.py`
**Role:** Python-native single-shot impact simulation. Implements the Shen & Atluri (2006) analytical model to compute residual stress depth profiles, plastic zone geometry, energy partitioning, and surface deformation — all without Abaqus. Produces `.npy` files in the same schema as the Abaqus datasets so `data_viz.py` can consume them directly.

**Reference:** Shen, S. & Atluri, S.N. (2006). "An Analytical Model for Shot-Peening Induced Residual Stresses." *CMC: Computers, Materials & Continua*, vol. 4, no. 2, pp. 75–85.

**Run with:**
```bash
python impact_sim.py --output ./Simulation_0 --V 35.9 --D 0.0005 --Nx 10 --Ny 10 --plot
```

**Dependencies:** `numpy`, `matplotlib` (standard Python math for all physics equations)

---

#### Physics Model Summary

The module follows the Shen & Atluri model in three stages:

**Stage 1 — Elastic contact (Hertz)**

Using the equivalent modulus $E^* = \left(\frac{1-\nu_s}{E_s} + \frac{1-\nu_b}{E_b}\right)^{-1}$ (Eq 3), the elastic contact radius is:
$$a_e = R \left(\frac{5\pi k \rho_s V_n^2}{4 E^*}\right)^{1/5}  \quad \text{(Eq 2)}$$

The Hertz pressure distribution $p(r) = p_0\sqrt{1-(r/a_e)^2}$ drives elastic stress components (Eq 4–8) at depth $z$ via dimensionless coefficients $A = 1/(1+\bar{z}^2)$, $B = 1 - \bar{z}\arctan(1/\bar{z})$ (Eqs 5a–5b), where $\bar{z} = z/a_e$.

**Stage 2 — Elastic-plastic loading (bilinear hardening)**

The material model uses bilinear hardening with slope $c = \frac{2}{3}E_p$:

- Loading plastic strain (Eq 20): $\varepsilon^p_x = \frac{\sqrt{\sigma_y^2 + \frac{3c}{2E_b}\sigma_{eq}^2} - \sigma_y}{3c}$
- Deviatoric loading stress (Eq 21): $S_{xl} = \frac{\sigma_y}{3} + c\,\varepsilon^p_x$
- Unloading plastic strain (Eq 26): $\varepsilon^p_{xu} = \frac{\sqrt{4\sigma_y^2 + \frac{3c}{2E_b}\sigma_{eq}^2} - 2\sigma_y}{3c}$
- Deviatoric unloading stress (Eq 23): $S_{xu} = S_{xl} - \frac{2}{3}\sigma_y - c\,\varepsilon^p_{xu}$

**Stage 3 — Plastic zone geometry and residual stress**

- Dent radius (Eq 44): $a_p = D\left(\frac{\rho_s V_n^2}{18\sigma_y}\right)^{1/4}$
- Plastic zone radius (Eq 43): $r_p = a_p\left(\frac{2E_b}{3\sigma_y}\right)^{1/3}$
- Mean plastic strain (Eq 45): $\varepsilon_{Mp} = \frac{9}{4}\frac{a_p^4}{D\,r_p^3}$
- Plastic zone volume (Eq 42): $V_p = \frac{2\pi}{3}r_p^3$
- Total plastic work (Eq 41): $W_t = V_p\,\sigma_y\,\varepsilon_{Mp}$

The residual stress (Eq 35) after biaxial relaxation:
$$\sigma_R = \frac{S_{xs}(1+\nu_b)/(1-\nu_b) - E_b\,\bar{\varepsilon}^p/(1-\nu_b)}{2}$$

where $S_{xs}$ and $\bar{\varepsilon}^p$ are piecewise functions of the von-Mises loading stress (Eqs 22, 47a–b).

**Energy note:** The Shen & Atluri model is energy-consistent by derivation — $W_t = \frac{\pi}{12}D^3\rho_s V_n^2 \equiv \text{KE}_\text{initial}$ for normal impact. This is an intentional upper bound (all KE absorbed as plastic work). The COR in the energy balance therefore equals zero for the normal velocity component, which is the model's implicit assumption.

---

#### Public API

| Function / Class | Signature | Description |
|---|---|---|
| `ShotPeenParams` | `dataclass` | Container for all material, shot, and impact parameters with derived properties `R`, `Ms`, `Vn`. |
| `generate_mesh` | `(Lx, Ly, Lz, Nx, Ny, Nz) → dict` | Structured quad (Nz=1) or hex (Nz>1) mesh. Returns `node_labels`, `node_coords`, `element_labels`, `element_connectivity`, `impact_center`. |
| `compute_contact_params` | `(params) → dict` | Hertzian contact: `E_eq`, `ae`, `p0`, `F`, `t`, `delta`. |
| `compute_stress_field` | `(contact, params) → dict` | Through-depth stress arrays: `Z`, `Z_bar`, `sigma_xe`, `sigma_ze`, `sigma_eqe`, `eps_load_p`, `eps_unload_p`, `Sxl`, `Sxu`, `eps_avg`, `sxs`, `sR`. |
| `compute_plastic_zone` | `(params) → dict` | Plastic geometry: `a_p`, `r_p`, `epsilon_Mp`, `V_p`, `W_t`. |
| `compute_energy_balance` | `(params, contact, plastic) → dict` | Energy partition: `KE_initial`, `W_plastic`, `KE_rebound`, `W_wave`, `e` (COR). |
| `map_displacements` | `(mesh, contact, plastic, params, impact_center) → (labels, disp)` | Maps permanent dent profile + radial bulge onto mesh nodes. Returns `(node_labels, displacements)` as `float32 (N, 3)`. |
| `map_stresses` | `(mesh, stress_field, plastic, params, impact_center) → (labels, stresses)` | Assigns biaxial residual stress to each element centroid using depth interpolation + Gaussian radial attenuation. Returns `float32 (E, 4)` as `[S11, S22, S33=0, S12=0]`. |
| `run_simulation` | `(params, output_dir, Lx, Ly, Lz, Nx, Ny, Nz, save_npy, verbose) → dict` | Orchestrates all steps, saves `.npy` files, prints energy balance. Returns full results dict. |
| `plot_residual_stress` | `(results, show, save_path)` | 3-panel figure: σR depth profile, plastic strain profile, energy bar chart. |

---

#### `ShotPeenParams` fields

| Field | Default | Description |
|---|---|---|
| `E_s` | 210 GPa | Young's modulus of shot (steel S170) |
| `nu_s` | 0.3 | Poisson's ratio of shot |
| `D` | 0.5 mm | Shot diameter |
| `rho_s` | 2000 kg/m³ | Shot density |
| `E_b` | 113.8 GPa | Young's modulus of target (Ti alloy) |
| `nu_b` | 0.34 | Poisson's ratio of target |
| `sigma_yield` | 276 MPa | Yield stress of target |
| `c` | 3.0 GPa | Bilinear hardening modulus (= 2/3 E_p) |
| `V` | 35.9 m/s | Impact velocity |
| `phi` | π/2 rad | Impact angle from surface (π/2 = normal) |
| `k` | 0.8 | Elasticity factor |
| `n_depth` | 300 000 | Depth profile resolution |
| `depth_max_factor` | 8.0 | Profile depth limit = factor × ae |
| `R` | D/2 | *(derived)* Shot radius |
| `Ms` | 4/3·π·R³·ρs | *(derived)* Shot mass |
| `Vn` | V·sin(φ) | *(derived)* Normal velocity component |

---

#### Output `.npy` file schema (from `run_simulation`)

| File | Shape | dtype | Description |
|---|---|---|---|
| `node_labels.npy` | (N,) | int32 | Node IDs, 1-indexed |
| `node_coords.npy` | (N, 3) | float32 | [x, y, z] in metres |
| `element_labels.npy` | (E,) | int32 | Element IDs, 1-indexed |
| `element_connectivity.npy` | (E, 4) or (E, 8) | int32 | Node label indices per element |
| `disp_node_labels.npy` | (N,) | int32 | Same as node_labels |
| `displacements.npy` | (N, 3) | float32 | [ux, uy, uz] permanent deformation |
| `stress_element_labels.npy` | (E,) | int32 | Element IDs for stress array |
| `stresses.npy` | (E, 4) | float32 | [S11, S22, S33, S12] residual stresses |
| `sR_depth_profile.npy` | (n_depth, 2) | float64 | Columns [Z (m), σR (Pa)] |
| `sigma_eqe_profile.npy` | (n_depth, 2) | float64 | Columns [Z/ae, σeq (Pa)] |
| `energy_balance.txt` | — | text | Key=value energy, plastic, contact scalars |

---

#### Displacement field model

On the surface mesh, permanent displacements are computed as:

- **Normal dent (uz, negative = into material):**
  - Within dent (r ≤ a_p): $u_z(r) = -\delta_p(1 - r^2/a_p^2)$, paraboloid with $\delta_p = a_p^2/(2R)$
  - Transition zone (a_p < r ≤ r_p): $u_z(r) = -\delta_p(a_p/r)^2\exp(-(r-a_p)/a_p)$
  - Far field (r > r_p): $u_z \approx 0$

- **Radial bulge (ux, uy, outward plastic incompressibility):**
  $u_r(r) = \frac{2}{3}\delta_p\frac{r}{r_p}\exp(-(r/r_p)^2)$, then $u_x = u_r\cos\theta$, $u_y = u_r\sin\theta$

---

#### Stress field model

Each element's centroid position is used to:
1. Interpolate $\sigma_R$ from the depth profile at depth $|z_c|$.
2. Apply a Gaussian radial kernel: $G(r) = \exp(-r^2/(2(r_p/2)^2))$, zeroed for $r > 3r_p$.
3. Assign $S_{11} = S_{22} = \sigma_R(z_c)\cdot G(r)$; $S_{33} = S_{12} = 0$ (biaxial, normal impact).

---

#### Verified default outputs (Ti alloy / S170, V = 35.9 m/s, D = 0.5 mm, φ = 90°)

| Quantity | Value |
|---|---|
| $a_e$ (elastic contact radius) | 37.3 µm |
| $p_0$ (peak Hertz pressure) | 13.6 GPa |
| $a_p$ (dent radius) | 75.5 µm |
| $r_p$ (plastic zone radius) | 490.7 µm |
| $\varepsilon_{Mp}$ (mean plastic strain) | 0.00124 |
| $W_t$ (plastic work) | 84.4 µJ |
| $\sigma_R^\text{min}$ (peak compressive residual) | −90.7 MPa |

These agree with the values produced by `ResidualStress.py` for the same parameter set.

---

### `tests/test_impact_sim.py`

**Location:** `tests/test_impact_sim.py`
**Role:** Comprehensive pytest suite for `impact_sim.py`. ~110 tests across 12 test classes covering every public function, physics invariants, I/O correctness, and numerical reproducibility.

**Test classes:**

| Class | Coverage |
|---|---|
| `TestShotPeenParams` | Derived properties (`R`, `Ms`, `Vn`), custom overrides, Poisson ratio bounds |
| `TestGenerateMeshSurface` | Node/element counts, connectivity validity, coordinate ranges, dtype |
| `TestGenerateMeshVolume` | 3-D hex mesh counts, negative z-coordinates for depth layers |
| `TestComputeContactParams` | ae sign/scale, p0 sign, F sign, delta formula, velocity/size monotonicity, harmonic mean bound |
| `TestComputeStressField` | Array key presence, Z monotonicity, sign conventions (compressive near surface), loading ≥ unloading plastic strain, sR compressive somewhere, deep decay to zero |
| `TestComputePlasticZone` | r_p > a_p, V_p formula, W_t = V_p·σy·εMp, monotonicity with V and σy |
| `TestComputeEnergyBalance` | KE_initial formula, energy conservation identity, COR in [0,1], W_wave ≥ 0, restitution formula |
| `TestMapDisplacements` | Output shapes/dtype, centre node max |uz|, uz < 0 at centre, far field uz ≈ 0, custom impact centre, radial symmetry |
| `TestMapStresses` | Output shapes/dtype, S33 = S12 = 0, S11 = S22, centre > edge stress, far-field zero stress, label consistency |
| `TestRunSimulation` | Required keys, all 11 .npy files created, shape consistency, `save_npy=False`, verbose toggle, custom params, 3-D mesh, reproducibility |
| `TestPlotResidualStress` | Runs without error, saves file, `show=False` suppresses plt.show, `show=True` calls it |
| `TestPhysicsSanity` | ae < 10R, r_p micron scale, W_t formula, surface compressive stress, equivalent stress decays at depth, energy fractions sum to 1.0, COR range, Vn ≤ V, mesh covers plastic zone |

**Run with:** `pytest tests/test_impact_sim.py -v`

---

### `taichi_impact_sim.py`

**Location:** `src/peen-ml/taichi_impact_sim.py`
**Role:** Numerical "ground-truth" counterpart to `impact_sim.py`. Implements a full 3-D Moving Least Squares Material Point Method (MLS-MPM) simulation of a single shot-peen impact, producing deformation fields, stress fields, and energy histories that can be directly compared against the Shen & Atluri analytical model.

**Install prerequisite:** `pip install taichi`

**Run with:**
```bash
python taichi_impact_sim.py --arch cpu --n_grid 48 --steps 600 --compare --plot
```

**Dependencies:** `numpy`, `matplotlib`, `taichi` (optional — module imports cleanly without it; `MPMShotPeenSolver` raises `ImportError` at construction time only)

---

#### Why MPM for shot peening?

The Material Point Method uses Lagrangian particles (to track material history: plastic strain, stress) mapped onto a background Eulerian grid (to solve momentum equations and enforce boundary conditions). Contact between the rigid shot and deformable target is handled naturally — particles from different bodies interact only through the shared grid, so no explicit contact detection is needed. This makes MPM particularly well-suited to high-velocity impact where the contact geometry evolves rapidly.

---

#### Constitutive model

**Elastic:** Hencky (logarithmic) strain formulation of linear elasticity in principal strain space:
$$\tau = 2\mu\,\varepsilon_H + \lambda\,\mathrm{tr}(\varepsilon_H)\,\mathbf{I}$$
where $\varepsilon_H = \log(\Sigma)$ (log of the SVD singular values of the elastic deformation gradient $F_e$).

**Plastic:** Von Mises yield function with isotropic bilinear hardening:
$$f = \|\mathbf{s}\|_F - \sqrt{\tfrac{2}{3}}\,(\sigma_y + H\,\varepsilon_p^{eq})$$

Radial return mapping gives:
$$\Delta\gamma = \frac{f^{\text{trial}}}{2\mu + \tfrac{2}{3}H}$$
$$\mathbf{s}^{\text{corrected}} = \left(1 - \frac{2\mu\,\Delta\gamma}{\|\mathbf{s}^{\text{trial}}\|}\right)\mathbf{s}^{\text{trial}}$$
$$\varepsilon_p^{eq,\,n+1} = \varepsilon_p^{eq,\,n} + \sqrt{\tfrac{2}{3}}\,\Delta\gamma$$

**Hardening slope:** H = (3/2)·c — directly mirrors the `c` parameter in `ShotPeenParams` / `impact_sim.py`, making the comparison exact in material terms.

---

#### Rigid shot model

The shot is not discretised into MPM particles. Instead it acts as a moving signed-distance-function obstacle. At every grid update step, any grid node found inside the sphere has its velocity modified to prevent interpenetration:
- Normal relative velocity $v_r = (\mathbf{v}_{node} - \mathbf{v}_{shot})\cdot\hat{\mathbf{n}}$
- If $v_r < 0$ (approaching): apply impulse $\Delta\mathbf{v} = -v_r\,\hat{\mathbf{n}}$ to grid node
- Reaction (Newton's 3rd law): accumulate impulse on shot → update shot velocity
- COR emerges naturally from the momentum exchange, not prescribed

---

#### MLS-MPM algorithm (one substep)

1. **P→G** (`_p2g`): transfer mass $m$ and APIC momentum $(m\mathbf{v} + \mathbf{A}\Delta\mathbf{x})$ to grid using quadratic B-spline weights; simultaneously apply Kirchhoff stress divergence as grid force.
2. **Grid ops** (`_grid_ops`): normalise to velocity; enforce rigid sphere BC with impulse accumulation; apply plate boundary conditions (fixed bottom, free lateral).
3. **G→P** (`_g2p`): interpolate grid velocity back to particles; update particle velocity, affine matrix $C$, position $x$, and deformation gradient $F = (I + dt\,C)\,F_n$; run SVD + von Mises return mapping; update plastic strain $\varepsilon_p^{eq}$ and Cauchy stress $\sigma = \tau/J$.

---

#### Public API

| Class / Function | Description |
|---|---|
| `MPMShotPeenSolver(params, Lx, Ly, Lz, n_grid, ppc, rho_target, arch, use_f64, verbose)` | Main solver class. All Taichi fields allocated in `__init__`. |
| `solver.initialize()` | Place particles in a regular lattice; reset shot state. |
| `solver.run(n_steps, record_every)` | Run simulation; update shot position/velocity from accumulated impulse. Stops early if shot rebounds. |
| `solver.extract_results(output_dir, Nx_out, Ny_out, save_npy)` | IDW-interpolate particle data to structured mesh; save `.npy` files; return results dict. |
| `solver.plot_energy_history(show, save_path)` | Plot KE of target + shot vs time, and shot $v_z(t)$. |
| `compare_results(mpm, analytical, show, save_path)` | 2×2 comparison figure: σR(z), ε_p(z), energy bar chart, shot velocity history. |
| `run_mpm_simulation(params, output_dir, n_grid, n_steps, arch, verbose)` | One-liner convenience wrapper; same interface as `impact_sim.run_simulation()`. |

---

#### `MPMShotPeenSolver` key attributes

| Attribute | Formula / Value | Description |
|---|---|---|
| `dx` | `max(Lx,Ly,Lz) / n_grid` | Uniform grid cell size |
| `dt` | `0.3·dx / (c_p + V)` | CFL-stable timestep |
| `mu`, `la` | Lamé constants from E_b, nu_b | Elastic moduli |
| `H_hard` | `(3/2)·c` | Isotropic hardening slope |
| `n_particles` | `(nx_p·ny_p·nz_p)` particles | Total target particles |
| `p_vol` | `(dx/ppc_1d)³` | Particle volume |
| `p_mass_val` | `ρ·p_vol` | Particle mass |
| `shot_center` | numpy (3,) | Shot centre (physical z) |
| `shot_vel` | numpy (3,) | Shot velocity (updated each step) |
| `z_offset` | `nz_surface·dx` | Grid→physical z conversion |

---

#### Output `.npy` schema

Same as `impact_sim.py` so `data_viz.py` and the ML pipeline consume both without modification:

| File | Shape | Description |
|---|---|---|
| `node_labels.npy` | (N,) int32 | 1-indexed surface mesh nodes |
| `node_coords.npy` | (N, 3) float32 | [x, y, z=0] surface nodes |
| `element_labels.npy` | (E,) int32 | Quad element IDs |
| `element_connectivity.npy` | (E, 4) int32 | Node label indices |
| `displacements.npy` | (N, 3) float32 | IDW-interpolated [ux, uy, uz] |
| `stresses.npy` | (E, 4) float32 | [S11, S22, S33, S12] |
| `sR_depth_profile.npy` | (M, 2) float64 | [depth (m), σR (Pa)] from impact column |
| `eps_depth_profile.npy` | (M, 2) float64 | [depth (m), ε_p_eq] |
| `energy_history.npy` | (T, 3) float64 | [t (s), KE_target (J), KE_shot (J)] |
| `energy_balance.txt` | text | Key=value energy scalars |

---

#### How to compare MPM vs analytical

```python
from taichi_impact_sim import MPMShotPeenSolver, compare_results
from impact_sim import ShotPeenParams, run_simulation

params = ShotPeenParams()                          # same params for both

# Numerical (MPM)
solver = MPMShotPeenSolver(params, n_grid=48, arch="cpu")
solver.initialize()
solver.run(n_steps=600)
mpm_res = solver.extract_results("./mpm_output")

# Analytical (Shen & Atluri)
ana_res = run_simulation(params, output_dir="./analytical_output", Nx=20, Ny=20)

# Side-by-side comparison
compare_results(mpm_res, ana_res, save_path="./comparison.png")
```

---

#### Known limitations and modelling choices

| Item | Choice | Rationale |
|---|---|---|
| Shot rigidity | Rigid sphere (kinematic BC) | Steel E=210 GPa >> Ti E=114 GPa; error < 5% in contact radius |
| Particle lattice | Regular grid, no jitter | Avoids mode-locking artefacts common in random packing |
| Grid resolution | `n_grid=48` default | Shot diameter spans ~10 cells; adequate for capturing contact zone |
| Air gap | 3×R above surface | Allows shot to start in free flight before contact |
| P-wave speed (Ti alloy) | √((λ+2μ)/ρ) ≈ 6240 m/s | Standard Ti-6Al-4V value used for CFL |
| Gravity | Omitted | g×t² < 1 nm for a 140 ns impact — negligible |
| 2D vs 3D | Full 3D | Normal impact is axisymmetric in theory; 3D allows oblique angles later |

---

### `tests/test_taichi_impact_sim.py`

**Location:** `tests/test_taichi_impact_sim.py`
**Role:** ~80 tests across 10 test classes. Tests are split into two groups:

- `@pytest.mark.no_taichi` (default): pure-Python tests that run even without Taichi installed. Cover import, `compare_results()`, energy physics formulae, and the NumPy mirror of the von Mises return mapping.
- `@requires_taichi`: functional tests that exercise the full solver. Skipped automatically when `taichi` is absent.

**Test classes:**

| Class | Coverage | Taichi required? |
|---|---|---|
| `TestModuleImport` | `_TAICHI_AVAILABLE` flag, all public names importable, docstring | No |
| `TestShotPeenParamsInMPM` | R, Ms, Vn, angle variants | No |
| `TestMPMSolverConstruction` | dx, dt, CFL, grid dims, field allocation, `raises_without_taichi` | Yes |
| `TestMPMInitialization` | Particles in domain, zero velocity, identity F, zero Jp | Yes |
| `TestMPMAfterRun` | Histories populated, KE ≥ 0, plastic strain ≥ 0, deformation occurred | Yes |
| `TestExtractResults` | Required keys, dtypes, shapes, energy formula, npy files, connectivity validity | Yes |
| `TestCompareResults` | Runs without error, saves file, `show` flag, empty results degrade gracefully | No |
| `TestEnergyPhysics` | KE formula, V² scaling, COR definition, H=3/2·c, Lamé constants, CFL bound | No |
| `TestVonMisesReturnMapping` | Elastic unchanged, plastic reduces stress, Jp increases, consistency on yield surface, hydrostatic invariance, hardening monotonicity | No |
| `TestPlotEnergyHistory` | Runs, saves file, `show` flag | Yes |

**Run all tests:**    `pytest tests/test_taichi_impact_sim.py -v`
**Run without Taichi:** `pytest tests/test_taichi_impact_sim.py -v -m "not requires_taichi"`

---

## Dependencies

Core runtime dependencies (from `pyproject.toml`):

| Package | Min Version |
|---|---|
| requests | 2.25.1 |
| numpy | 1.20.0 |
| matplotlib | 3.4.0 |
| pandas | 1.3.0 |
| torch | 1.9.0 |
| torchvision | 0.10.0 |
| pillow | any |

Python >= 3.7 required.

The `requirements.txt` is a pinned conda environment export (wide-character/Windows encoded) and lists the exact versions used in development, including `torch==2.5.0`, `numpy==2.2.0`, and `matplotlib==3.9.2`.

`impact_sim.py` and `taichi_impact_sim.py` (pure-Python path) use only `numpy` and `matplotlib`. The MPM solver additionally requires `taichi` (`pip install taichi`). Taichi — no heavy dependencies. The previously used `scipy.integrate.cumulative_trapezoid` was removed from the module (it was only needed for exploratory plotting in `ResidualStress.py`).

---

## Taichi 1.7.4 Compatibility Fixes

The following bugs were identified and fixed when running `taichi_impact_sim.py` against Taichi 1.7.4:

**1. `@ti.func` rejects all Python type annotations (arguments 1–2 of `_von_mises_return`)**

Taichi 1.7.4's `@ti.func` decorator does not accept any Python-style type annotations on method parameters. Only `@ti.kernel` supports annotated arguments (and only `ti.template()` for field handles). All annotations were removed:
```python
# Before (broken):
def _von_mises_return(self, tau_trial_p: ti.template(), Jp_n: float):
# After (fixed):
def _von_mises_return(self, tau_trial_p, Jp_n):
```

**2. `int(ti.Vector)` / `float(ti.Vector)` invalid inside kernels (`_p2g`, `_g2p`)**

Python's built-in `int()` and `float()` only work on Python scalars. Inside `@ti.kernel` / `@ti.func`, vector types must use Taichi's `.cast()` method:
```python
# Before (broken):
base = int(Xp - 0.5)
fx   = Xp - float(base)
dpos = (float(offset) - fx) / inv_dx
# After (fixed):
base = (Xp - 0.5).cast(int)
fx   = Xp - base.cast(float)
dpos = (offset.cast(float) - fx) / inv_dx
```
This fix applies to both the `_p2g` and `_g2p` kernels.

**3. `ti.atomic_add` requires a Taichi field element, not a Python local (`_grid_ops`)**

`ti.atomic_add(var, val)` requires its first argument to be a Taichi field element (e.g. `field[index]`). Python-scope local floats are not valid targets. The fix resets the output fields at kernel start and accumulates directly into them:
```python
# Before (broken — imp_x/y/z are Python locals):
imp_x = 0.0
ti.atomic_add(imp_x, -dm * dv_x)
imp_x_out[None] = imp_x   # assignment at end

# After (fixed — reset fields first, then accumulate directly):
imp_x_out[None] = 0.0     # reset at kernel start
ti.atomic_add(imp_x_out[None], -dm * dv_x)   # atomic on field element
```
