# peen-ml

[![Pylint](https://github.com/onestr1/peen-ml/actions/workflows/pylint.yml/badge.svg)](https://github.com/onestr1/peen-ml/actions/workflows/pylint.yml)
[![codecov](https://codecov.io/gh/onestr1/peen-ml/branch/main/graph/badge.svg)](https://codecov.io/gh/onestr1/peen-ml)

**Machine learning solution to predict deformation from shot peening as an alternative to dynamic simulation.** This repository provides tools and workflows to input shot peening parameters and geometry, train machine learning (ML) models to predict resulting deformations, visualize outcomes, and interact with the models through a user-friendly GUI. It aims to streamline the process of exploring and optimizing shot peening recipes, reducing reliance on time-consuming finite element analysis (FEA) simulations.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Core Components](#core-components)
- [User Roles and Use Cases](#user-roles-and-use-cases)
- [Installation and Setup](#installation-and-setup)
- [GUI Walkthrough](#gui-walkthrough)
  - [Step 1 — Launch and Main Menu](#step-1--launch-and-main-menu)
  - [Step 2 — Generate a Dataset](#step-2--generate-a-dataset)
  - [Step 3 — Train a Model](#step-3--train-a-model)
  - [Step 4 — Load and Evaluate](#step-4--load-and-evaluate)
- [Material Properties Library](#material-properties-library)
- [Command-Line Usage](#command-line-usage)
- [Curved Surface and STL Support](#curved-surface-and-stl-support)
- [Data Visualization](#data-visualization)
- [Testing](#testing)
- [License](#license)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

---

## Overview
Shot peening is a manufacturing process used to improve material properties by bombarding a surface with small beads (shots). Predicting deformation due to shot peening typically involves complex FEA simulations. This repository provides a machine learning-driven approach to quickly approximate deformation results, enabling engineers to:
- Rapidly iterate on shot peening parameters.
- Compare multiple recipes without running time-consuming simulations.
- Visualize predicted outcomes and analyze their effects on component geometry.
- Apply predictions to arbitrary 3D curved surfaces from STL files.

![Shotpeening_brief](https://raw.githubusercontent.com/onestr1/peen-ml/refs/heads/main/images/what_is_shotpeen.png)

This project was developed as part of the CSE 583 Software Development for Data Scientists course at the University of Washington.

### Style
Code follows **PEP 8** guidelines with relaxed line-length limits where needed for readability.

### Replication
- **Demo Simulations:** `tests/simulation_0/` contains a sample simulation for exploring workflows.
- **Built-in Simulator:** `native_dataset_gen.py` generates training datasets entirely in Python — no Abaqus licence required.
- **Legacy Abaqus Scripts:** `dataset1_script.py` and `dataset2_script.py` are retained for reproducibility but are not required for normal use.

---

## Key Features
- **GUI for Ease of Use**: A graphical user interface (`shotpeen_gui.py`) to generate datasets, train models, and evaluate predictions without requiring deep ML expertise.
- **Built-in Physics Simulator**: A Python-native multi-shot simulation engine (`multi_shot_sim.py`, `native_dataset_gen.py`) based on the Shen & Atluri impact model — eliminates the Abaqus dependency entirely.
- **Material Property Library**: Centralised `materials.py` library with sourced properties for 5 workpiece alloys and 5 shot media. Material selection is exposed in both the Generate and Evaluate dialogs, enabling material-conditioned model training.
- **Three ML Architectures**:
  - *DisplacementPredictor* — CNN with channel and spatial attention, dense FC output.
  - *ConvDecoderPredictor* — Same encoder but a convolutional decoder that predicts a full spatial displacement field. **178× fewer parameters** and evaluates at any mesh resolution without retraining.
  - *SIRENPredictor* — Implicit neural representation; memory-safe for very large meshes.
- **Curved Surface / STL Support**: Full inference pipeline for arbitrary 3D shells loaded from STL files, including nozzle trajectory planning (raster, spiral, zigzag or custom waypoints) and per-vertex normal-frame rotation of predicted displacements.
- **Three-Layer Mesh Interpolation**:
  - *Layer 1* — Bilinear resize of checkerboard to match trained grid size.
  - *Layer 2* — Thin-plate-spline RBF interpolation from training mesh to evaluation mesh (FC model), or exact bilinear sampling (convolutional decoder).
  - *Layer 3* — Rodrigues rotation of flat-plate predictions into local STL surface normals.
- **Data Visualization Tools**: 2D mesh plots (`data_viz.py`) and 3D STL surface plots with Matplotlib.
- **Continuous Integration**: GitHub Actions workflow for style checking (`.github/workflows/pylint.yml`).

---

## Repository Structure
```
[Repository Root]
├─ .github/workflows/pylint.yml      # CI pipeline (Pylint checks)
├─ blueprint/                        # Use-case and component docs
├─ images/                           # Screenshots and diagrams used in README
├─ src/peen-ml/
│  ├─ materials.py                   # Centralised material property library
│  ├─ model.py                       # ML models, training & evaluation
│  ├─ data_viz.py                    # 2D/3D visualization tools
│  ├─ multi_shot_sim.py              # Python-native shot peening simulator
│  ├─ native_dataset_gen.py          # Dataset generator (no Abaqus needed)
│  ├─ gaussian_nozzle_dataset_gen.py # Gaussian nozzle profile generator
│  ├─ impact_sim.py                  # Per-impact stress/displacement model
│  ├─ stl_surface.py                 # STL geometry: normals, KDTree, checkerboard
│  ├─ nozzle_trajectory.py           # Nozzle scan patterns & waypoint loading
│  └─ curved_surface_sim.py          # Physics simulation on curved STL surfaces
├─ tests/
│  ├─ simulation_0/                  # Example data for test cases
│  ├─ test_materials.py              # Material library tests (13 tests)
│  ├─ test_model.py                  # Model training, evaluation & interpolation tests
│  └─ ...                            # (other test files)
├─ README.md
├─ pyproject.toml
└─ shotpeen_gui.py                   # Main GUI application
```

---

## Core Components

- **Physics Simulation Engine** (`multi_shot_sim.py`, `native_dataset_gen.py`, `impact_sim.py`):
  Python-native multi-shot simulation based on the Shen & Atluri elastic-plastic impact model. Generates training datasets (~2 s/simulation) without any FEA software.

- **Material Library** (`materials.py`):
  Single source of truth for workpiece and shot properties. Used by the dataset generator, model training, and inference.

- **Curved Surface Simulator** (`curved_surface_sim.py`, `stl_surface.py`, `nozzle_trajectory.py`):
  Extends the flat-plate simulator to arbitrary 3D shells.

- **Prediction Engine** (`model.py`):
  Three architectures share the same encoder (3 conv + attention blocks). All support optional material-feature conditioning (`mat_dim=7`).

- **GUI** (`shotpeen_gui.py`):
  Three workflows in one window: *Generate Dataset*, *Train Model*, *Load & Evaluate*.

---

## User Roles and Use Cases

- **Mechanical Engineer (Alex)**: Inputs geometry and recipe parameters, quickly gets deformation predictions.
- **Process Engineer (Jordan)**: Compares multiple recipes to find optimal shot peening settings.
- **Quality Control Engineer (Taylor)**: Validates predicted results against actual deformation data.
- **Data Scientist (Sam)**: Experiments with ML architectures and generates new training datasets.

---

## Installation and Setup

**Prerequisites:** Python 3.8–3.12, Git

**Dependencies:**
```
numpy >= 1.20.0
matplotlib >= 3.4.0
torch >= 2.0.0
scipy >= 1.7.0
trimesh
pillow
tkinter  (bundled with most Python distributions)
```

**Steps:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/onestr1/peen-ml.git
   cd peen-ml
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install .
   pip install trimesh              # STL support
   ```

---

## GUI Walkthrough

Launch the application from the project root:
```bash
python shotpeen_gui.py
```

The full workflow — generate data → train → evaluate — is done entirely through the GUI. Follow the four steps below in order.

---

### Step 1 — Launch and Main Menu

<img src="images/Screenshot%202026-05-31%20214853.png" alt="Main Menu" width="700">

The main window offers three entry points:

| Button | When to use |
|--------|-------------|
| **Train Model →** | You already have a dataset (or just generated one) and want to train a CNN |
| **Load Model →** | You have a trained `.pth` file and want to predict deformation on a new shot pattern |
| **Generate Dataset →** | You have no simulation data yet — start here |

**First-time users click Generate Dataset →**

---

### Step 2 — Generate a Dataset

<img src="images/Screenshot%202026-05-31%20215006.png" alt="Generate Dataset" width="700">

**Click order:**

1. **Output folder** — type a path (e.g. `./Dataset_Native`) or click **Browse…** to choose a directory. The generator creates `Simulation_0/`, `Simulation_1/`, … sub-folders here.
2. **Simulations** — number of shot-peening simulations to run (default 100; use 200–500 for training).
3. **Workers** — parallel processes. Set to the number of CPU cores for fastest generation; `1` = sequential.
4. **Mesh Nx / Ny** — finite-element mesh resolution (default 50×50).
5. **Grid G** — checkerboard size G×G (default 5). This is the input image size fed to the CNN.
6. **Shots min / max** — range for the random number of shots per simulation.
7. **Shot D min / max** — shot diameter range in mm.
8. **Velocity min / max** — impact velocity range in m/s.
9. **Material (optional)** — select a **Workpiece** and **Shot** material from the drop-downs. Leaving both blank uses physics defaults (Ti-6Al-4V + steel). Selected materials are logged to each `simulation_params.txt` and enable material-conditioned model training later. See [Material Properties Library](#material-properties-library) for the available options.
10. Click **Generate** — progress streams into the log panel below. Click **Stop** to cancel early.

The **Gaussian Nozzle** tab runs an alternative generator that models the spatial intensity profile of a real nozzle head — useful for more realistic coverage patterns.

---

### Step 3 — Train a Model

<img src="images/Screenshot%202026-05-31%20214928.png" alt="Train Model" width="700">

**Click order:**

1. **Step 1 — Browse** to the dataset folder produced in Step 2 (the folder containing `Simulation_0/`, `Simulation_1/`, …). The app reads the first simulation to auto-detect grid size G and node count N.
2. **Step 2 — Verify** the detected shape is shown automatically (G×G checkerboard, N mesh nodes). No action needed — just confirm the numbers match your dataset.
3. **Step 2b — Choose architecture:**
   - *Convolutional Decoder* **(recommended)** — 170 K parameters, works on any mesh without retraining, best for production use.
   - *FC — Legacy* — 30 M parameters, requires fixed node count; may OOM for N > 100 K.
   - *SIREN / INR* — implicit neural representation; memory-safe for very large meshes.
4. **Step 3 — Click Train**. Training runs in a background thread; the log panel streams epoch-by-epoch loss. A live loss curve appears in **Step 4** once training finishes.
5. The trained model is saved automatically to `<dataset_folder>/saved_model_conv/trained_conv_decoder_full_model.pth`.

> **Tip:** Hover over the **? Help** label in any section for detailed tooltip guidance.

---

### Step 4 — Load and Evaluate

<img src="images/Screenshot%202026-05-31%20214953.png" alt="Load and Evaluate" width="700">

**Click order:**

1. **Step 1 — Browse** to the `.pth` model file saved during training (default location: `<dataset>/saved_model_conv/trained_conv_decoder_full_model.pth`).
2. **Step 2 — Browse** to a `Simulation_N/` folder containing `checkerboard.npy` (the shot pattern to predict). Click **Preview Input Pattern** to view a colour-map of the checkerboard before running.
3. **Step 3 — Browse** to an empty output folder where `pred_displacements.npy` and `pred_displacements.csv` will be written.
4. **Optional — Material Properties** — if the model was trained with material conditioning (`mat_dim=7`), select the matching **Workpiece** and **Shot** from the drop-downs. Standard (non-conditioned) models ignore these fields.
5. **Optional — Curved Surface & Nozzle Trajectory** — to predict deformation on a 3D part instead of a flat plate:
   - Browse to an STL file.
   - Choose *Parametric scan* (set raster/spiral/zigzag pattern, speed, line spacing, standoff) or *Waypoint file* (.csv or .npy).
   - Leave blank to use the standard flat-plate mode.
6. **Step 4 — Click 1. Evaluate Model**. MSE and sMAPE vs. ground-truth are printed to the console.
7. **Click 2. Preview Deformation** — opens a matplotlib window showing the deformed mesh coloured by displacement magnitude.
8. **Click 3. Preview STL Deformation** — (STL mode only) opens a 3D surface plot coloured by displacement magnitude.

---

## Material Properties Library

`src/peen-ml/materials.py` is the single source of truth for all material data used throughout the pipeline. Properties are looked up by name and passed to the physics simulator, the ML model, and the GUI.

### Available workpiece materials

| Name | E (GPa) | ν | σ_yield (MPa) | Source |
|------|---------|---|---------------|--------|
| `Ti-6Al-4V` | 113.8 | 0.342 | 880 | ASM Aerospace Specification Metals |
| `316L-SS` | 193 | 0.265 | 290 | ASME Boiler & Pressure Vessel Code II-D |
| `4340-Steel` | 200 | 0.290 | 470 | MatWeb 4340 annealed |
| `Al-7075-T6` | 71.7 | 0.330 | 503 | MIL-HDBK-5J Table 3.7.6.0(b) |
| `Inconel-718` | 200 | 0.290 | 1100 | Special Metals datasheet SMC-045 |

### Available shot materials

| Name | ρ (kg/m³) | E (GPa) | ν | Source |
|------|-----------|---------|---|--------|
| `steel` | 7800 | 210 | 0.30 | ASM Handbook vol. 4 |
| `ceramic` | 6000 | 380 | 0.22 | Zircoa Inc. ZrO₂ datasheet |
| `glass` | 2500 | 70 | 0.22 | MIL-S-851D glass bead spec |
| `cast_iron` | 7300 | 170 | 0.26 | ASM Handbook vol. 1 |
| `tungsten` | 19300 | 411 | 0.28 | Plansee AG tungsten datasheet |

### Adding a new material — where to look and what format to expect

**Recommended source: [MatWeb](https://www.matweb.com)**

MatWeb is the most comprehensive free database for engineering material properties. Search by alloy name (e.g. "Inconel 625", "6061-T6 Aluminum") and the result page lists:

| MatWeb property name | Field in `materials.py` | Unit conversion |
|----------------------|------------------------|-----------------|
| Modulus of Elasticity | `E` (workpiece) or `E_s` (shot) | GPa → multiply by `1e9` to get Pa |
| Poisson's Ratio | `nu` / `nu_s` | dimensionless — use directly |
| Tensile Yield Strength | `sigma_yield` | MPa → multiply by `1e6` to get Pa |
| Density | `rho_s` (shot only) | g/cm³ → multiply by `1000` to get kg/m³ |

MatWeb **export format**: the free tier lets you copy the property table as plain text or HTML. A Pro/subscription account exports to **Excel (.xls)**, **comma-separated text (.csv)**, or **XML**. The CSV format has columns: `Property`, `Value`, `Units`, `Test Conditions`. The hardening modulus `c` (bilinear slope) is not directly listed — a practical estimate is `c ≈ 0.01 × E` for most structural alloys, or use the tangent modulus from a stress-strain curve if available.

**To add a new entry**, append to the relevant dict in `src/peen-ml/materials.py`:

```python
WORKPIECE_MATERIALS["My-Alloy"] = {
    "E":           200e9,     # Pa  — MatWeb "Modulus of Elasticity" × 1e9
    "nu":          0.29,      # —   — MatWeb "Poisson's Ratio"
    "sigma_yield": 500e6,     # Pa  — MatWeb "Tensile Yield Strength" × 1e6
    "c":           2.0e9,     # Pa  — ~0.01×E, or from stress-strain tangent
    "source":      "MatWeb: https://www.matweb.com/search/datasheet/...",
}
```

---

## Command-Line Usage

### Generating a dataset (no GUI)
```bash
python src/peen-ml/native_dataset_gen.py --n_sims 200 --output ./Dataset_Native --workpiece_material 316L-SS --shot_material ceramic
```

Each `Simulation_N/` folder contains `checkerboard.npy`, `displacements.npy`, `node_coords.npy`, and a `[material]` block in `simulation_params.txt`.

### Training (programmatic)
```python
from model import train_save_conv_gui
train_save_conv_gui("Dataset_Native", epochs=50)
# Saves to Dataset_Native/saved_model_conv/trained_conv_decoder_full_model.pth
```

### Architecture comparison

| | DisplacementPredictor | ConvDecoderPredictor | SIRENPredictor |
|---|---|---|---|
| Output | `Linear(512+mat, N×3)` | `(3,H,W)` field + bilinear sample | Implicit field sampled at coords |
| Parameters (N=2601, G=20) | ~30 M | ~170 K | ~2 M |
| Node-count fixed | Yes | No | No |
| Material conditioning | `mat_dim=7` concat | `mat_dim=7` FiLM bias | `mat_dim=7` latent concat |

---

## Curved Surface and STL Support

### Programmatic usage
```python
from model import curved_surface_inference
from nozzle_trajectory import ScanParams, raster_scan

traj = raster_scan(ScanParams(
    Lx=0.05, Ly=0.05, z_standoff=0.15,
    scan_speed=0.02, line_spacing=0.005, dt=0.1,
))

result = curved_surface_inference(
    model_path="Dataset_Native/saved_model_conv/trained_conv_decoder_full_model.pth",
    stl_path="my_part.stl",
    trajectory_or_checkerboard=traj,
    G=20,
    pred_save_dir="output/",
)
# result keys: displacements_on_stl (V,3), vertex_normals (V,3), checkerboard (G,G)
```

### Nozzle trajectory patterns
| Pattern | Function |
|---|---|
| Raster scan | `raster_scan(ScanParams(...))` |
| Spiral scan | `spiral_scan(ScanParams(...))` |
| Zigzag scan | `zigzag_scan(ScanParams(...))` |
| From CSV | `from_csv("waypoints.csv")` |
| From .npy | `from_npy("waypoints.npy")` |

---

## Data Visualization
```bash
python src/peen-ml/data_viz.py   # demo run on tests/simulation_0/
```

Generates plots for:
- Checkerboard patterns (shot coverage heatmap).
- Undeformed vs. deformed FEA mesh.
- Deformation magnitude on deformed mesh.
- 3D STL surface coloured by displacement magnitude (`visualize_stl_deformation`).
- 3D STL surface coloured by von Mises stress (`visualize_stl_stress`).

---

## Testing

```bash
pytest tests/
```

| Test file | What it covers |
|---|---|
| `test_materials.py` | Material library completeness, physical plausibility, accessor helpers |
| `test_model.py` | Model creation, training, evaluation, Layer 1/2 interpolation |
| `test_Shotpeen_Gui.py` | GUI widget construction and callback wiring |
| `test_data_viz.py` | Visualization utility functions |
| `test_stl_surface.py` | STL loading, vertex normals, shot projection |
| `test_nozzle_trajectory.py` | Raster/spiral/zigzag scan coverage, CSV/npy round-trips |
| `test_curved_surface_sim.py` | Curved simulation on flat STL, flat-plate equivalence |

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Authors
- [Onest Rexhepi](mailto:onestr@uw.edu)
  *Contributions:*
  - Established and managed the GitHub repository, including all documentation and replication workflows.
  - Generated data for both datasets utilized by the model.
  - Designed and implemented data visualization scripts integrated into the GUI.

- [Harshavardhan Sameer Raje](mailto:harshr@uw.edu)
  *Contributions:*
  - Developed the graphical user interface (GUI) for user-friendly interaction with the model.
  - Integrated the GUI with backend scripts for training, simulation, and visualization.
  - Extended the pipeline with STL curved-surface support, nozzle trajectory planning, the convolutional decoder architecture, and the three-layer mesh interpolation system.
  - Built the material awareness system: centralised `materials.py` library, material conditioning on all three CNN architectures, material selection in the Generate and Evaluate GUI dialogs, and per-simulation material metadata logging.
  - Linting and code coverage efforts.

- [Jiachen Zhong](mailto:jczhong@uw.edu)
  *Contributions:*
  - Designed and implemented the CNN model architecture for deformation prediction.
  - Optimized the model pipeline for efficient training and inference.

- [Xuanyu Shen](mailto:xshen20@uw.edu)
  *Contributions:*
  - Developed test cases to ensure functionality and reliability of scripts and models.

---

## Acknowledgments
- University of Washington CSE 583 course staff.
- Shen, S. & Atluri, S.N. (2006). "An Analytical Model for Shot-Peening Induced Residual Stresses." *CMC: Computers, Materials & Continua*, vol. 4, no. 2, pp. 75–85.
- The broader Python and open-source community for providing tools and libraries that made this project possible.
