# peen-ml

**Machine learning solution to predict deformation from shot peening as an alternative to dynamic simulation.** This repository provides tools and workflows to input shot peening parameters and geometry, train machine learning (ML) models to predict resulting deformations, visualize outcomes, and interact with the models through a user-friendly GUI. It aims to streamline the process of exploring and optimizing shot peening recipes, reducing reliance on time-consuming finite element analysis (FEA) simulations.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Core Components](#core-components)
- [User Roles and Use Cases](#user-roles-and-use-cases)
- [Installation and Setup](#installation-and-setup)
- [Running the GUI](#running-the-gui)
- [Training and Evaluating the ML Model](#training-and-evaluating-the-ml-model)
- [Curved Surface and STL Support](#curved-surface-and-stl-support)
- [Data Visualization](#data-visualization)
- [Testing](#testing)
- [License](#license)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## Overview
Shot peening is a manufacturing process used to improve material properties by bombarding a surface with small beads (shots). Predicting deformation due to shot peening typically involves complex FEA simulations. This repository provides a machine learning-driven approach to quickly approximate deformation results, enabling engineers to:
- Rapidly iterate on shot peening parameters.
- Compare multiple recipes without running time-consuming simulations.
- Visualize predicted outcomes and analyze their effects on component geometry.
- Apply predictions to arbitrary 3D curved surfaces from STL files.

![Shotpeening_brief](https://raw.githubusercontent.com/onestr1/peen-ml/refs/heads/main/images/what_is_shotpeen.png)

This project was developed as part of the CSE 583 Software Development for Data Scientists course at the University of Washington, aiming to demonstrate best practices in code organization, testing, documentation, and continuous integration.

### Style
To ensure clean and maintainable code, we have followed **PEP 8** guidelines and linted the code to the best of our ability. However, we decided to relax the line length limitations to maintain code readability and logical flow where necessary.

### Replication
This repository includes resources to replicate experiments and demonstrate functionality:
- **Demo Simulations:** The `tests/simulation_0` directory contains a sample simulation that can be used to explore and validate the repository's workflows.
- **Built-in Simulator:** Use `native_dataset_gen.py` to generate training datasets entirely in Python — no Abaqus licence required.
- **Legacy Abaqus Scripts:** `dataset1_script.py` and `dataset2_script.py` are retained for reproducibility but are not required for normal use.

**Note:** The Abaqus scripts are provided solely for reproducibility purposes and have not been linted or formatted for PEP 8 compliance.

## Key Features
- **GUI for Ease of Use**: A graphical user interface (`shotpeen_gui.py`) to generate datasets, train models, and evaluate predictions without requiring deep ML expertise.
- **Built-in Physics Simulator**: A Python-native multi-shot simulation engine (`multi_shot_sim.py`, `native_dataset_gen.py`) based on the Shen & Atluri impact model — eliminates the Abaqus dependency entirely.
- **Two ML Architectures**:
  - *DisplacementPredictor* — CNN with channel and spatial attention, dense FC output.
  - *ConvDecoderPredictor* — Same encoder but a convolutional decoder that predicts a full spatial displacement field. **178× fewer parameters** and evaluates at any mesh resolution without retraining.
- **Curved Surface / STL Support**: Full inference pipeline for arbitrary 3D shells loaded from STL files, including nozzle trajectory planning (raster, spiral, zigzag or custom waypoints) and per-vertex normal-frame rotation of predicted displacements.
- **Three-Layer Mesh Interpolation**:
  - *Layer 1* — Bilinear resize of checkerboard to match trained grid size.
  - *Layer 2* — Thin-plate-spline RBF interpolation from training mesh to evaluation mesh (FC model), or exact bilinear sampling (convolutional decoder).
  - *Layer 3* — Rodrigues rotation of flat-plate predictions into local STL surface normals.
- **Data Visualization Tools**: 2D mesh plots (`data_viz.py`) and 3D STL surface plots (`visualize_stl_deformation`, `visualize_stl_stress`) with Matplotlib.
- **Modular Architecture**: Separate modules for simulation, STL geometry, trajectory generation, ML models, visualization, and the GUI.
- **Continuous Integration**: GitHub Actions workflow for style checking (`.github/workflows/pylint.yml`).

## Repository Structure
```
[Repository Root]
├─ .github/
│  └─ workflows/
│     └─ pylint.yml                  # CI pipeline (Pylint checks)
├─ blueprint/
│  ├─ Components.md
│  ├─ Describing_a_usecase.md
│  └─ User_story.md
├─ dataset_sample/
│  ├─ dataset1_sample.rar
│  ├─ dataset2_sample.rar
│  └─ readme.txt.txt
├─ src/
│  └─ peen-ml/
│     ├─ model.py                    # ML models, training & evaluation
│     ├─ data_viz.py                 # 2D/3D visualization tools
│     ├─ multi_shot_sim.py           # Python-native shot peening simulator
│     ├─ native_dataset_gen.py       # Dataset generator (no Abaqus needed)
│     ├─ gaussian_nozzle_dataset_gen.py  # Gaussian nozzle profile generator
│     ├─ impact_sim.py               # Per-impact stress/displacement model
│     ├─ stl_surface.py              # STL geometry: normals, KDTree, checkerboard
│     ├─ nozzle_trajectory.py        # Nozzle scan patterns & waypoint loading
│     ├─ curved_surface_sim.py       # Physics simulation on curved STL surfaces
│     ├─ dataset1_script.py          # Legacy Abaqus dataset script
│     ├─ dataset2_script.py          # Legacy Abaqus dataset script
│     ├─ model_notebook_v2.ipynb
│     └─ model_notebook_v3.ipynb
├─ tests/
│  ├─ simulation_0/                  # Example data for test cases
│  ├─ simulation_1/                  # Example data for test cases
│  ├─ test_Shotpeen_Gui.py           # GUI tests
│  ├─ test_data_viz.py               # Visualization tests
│  ├─ test_model.py                  # Model training, evaluation & interpolation tests
│  ├─ test_stl_surface.py            # STLSurface geometry tests
│  ├─ test_nozzle_trajectory.py      # Trajectory generation tests
│  └─ test_curved_surface_sim.py     # Curved surface simulation tests
├─ .gitignore
├─ LICENSE
├─ README.md
├─ pyproject.toml
└─ shotpeen_gui.py                   # Main GUI application
```

## Core Components

- **Physics Simulation Engine** (`multi_shot_sim.py`, `native_dataset_gen.py`, `impact_sim.py`):
  Python-native multi-shot simulation based on the Shen & Atluri elastic-plastic impact model. Generates training datasets (~2 s/simulation) without any FEA software.

- **Curved Surface Simulator** (`curved_surface_sim.py`, `stl_surface.py`, `nozzle_trajectory.py`):
  Extends the flat-plate simulator to arbitrary 3D shells. Load an STL file, define a nozzle scan trajectory (parametric or from a CSV/npy waypoint file), and accumulate per-vertex displacements and residual stresses accounting for local surface normals.

- **Prediction Engine** (`model.py`):
  Two architectures share the same encoder (3 conv + attention blocks):
  - *DisplacementPredictor* — flattens to `Linear(512, N×3)`. Simple but node-count fixed.
  - *ConvDecoderPredictor* — decodes to a `(3, H, W)` spatial field, then bilinearly samples at any node coordinates. Node count never appears in model parameters; works on any mesh without retraining.

- **Three-Layer Inference Pipeline** (`model.py`):
  `curved_surface_inference()` and `load_and_evaluate_model_gui()` automatically apply:
  1. Grid resize (Layer 1) — handles checkerboard resolution mismatch.
  2. Spatial interpolation to evaluation mesh (Layer 2) — RBF for FC model, bilinear sampling for ConvDecoder.
  3. Normal-frame rotation (Layer 3) — rotates flat-plate predictions into per-vertex STL surface normals.

- **Visualization Module** (`data_viz.py`):
  - `visualize_checkerboard` — shot coverage heatmap.
  - `visualize_mesh` / `visualize_deformation` — 2D undeformed/deformed FEA mesh.
  - `visualize_stl_deformation` — 3D Poly3DCollection surface coloured by displacement magnitude.
  - `visualize_stl_stress` — 3D surface coloured by von Mises stress.

- **GUI** (`shotpeen_gui.py`):
  Three tabs: *Generate Dataset*, *Train Model*, *Load & Evaluate*. The evaluate tab includes an optional **Curved Surface & Nozzle Trajectory** section for STL-based inference.

## User Roles and Use Cases
Use cases and users are defined in `blueprint/`:

- **Mechanical Engineer (Alex)**: Inputs geometry and recipe parameters, quickly gets deformation predictions.
- **Process Engineer (Jordan)**: Compares multiple recipes to find optimal shot peening settings.
- **Quality Control Engineer (Taylor)**: Validates predicted results against actual deformation data.
- **Data Scientist (Sam)**: Experiments with ML architectures and generates new training datasets.

## Installation and Setup
**Prerequisites:**
- Python 3.8–3.12
- Git
- (Optional) Conda for environment management

**Dependencies:**
```
numpy >= 1.20.0
matplotlib >= 3.4.0
torch >= 2.0.0
scipy >= 1.7.0
trimesh
pillow
tkinter  (usually bundled with Python)
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
   pip install trimesh              # STL support (not in pyproject.toml yet)
   ```

## Running the GUI
```bash
python shotpeen_gui.py
```

![Gui_Main_Menu](https://raw.githubusercontent.com/onestr1/peen-ml/refs/heads/main/images/gui_main_menu.png)

**Tabs:**
- **Generate Dataset** — run the built-in physics simulator to produce a training dataset (no Abaqus needed).
- **Train Model** — select a dataset folder; the model architecture and node count are auto-detected.
- **Load & Evaluate** — load a trained model, select an evaluation simulation, set an output path, and click *Evaluate Model*. Optionally supply an STL file and nozzle trajectory for curved-surface prediction.

## Training and Evaluating the ML Model

### Generating a dataset
```bash
python src/peen-ml/native_dataset_gen.py   # creates Dataset_Python/ with 200 simulations
```
Each `Simulation_N/` folder contains `checkerboard.npy`, `displacements.npy`, `node_coords.npy`, and supporting mesh arrays.

### Training via GUI
<img src="https://raw.githubusercontent.com/onestr1/peen-ml/refs/heads/main/images/train_model_page.png" alt="Train Model" width="400">

1. Open the **Train Model** tab and browse to your dataset folder.
2. Click **Train**. Grid size and node count are inferred automatically.
3. The trained model and `reference_node_coords.npy` are saved to `<dataset>/saved_model/`.

### Training the convolutional decoder (programmatic)
```python
from model import train_save_conv_gui
train_save_conv_gui("Dataset_Python", epochs=50)
# Saves to Dataset_Python/saved_model_conv/trained_conv_decoder_full_model.pth
```

### Architecture comparison

| | DisplacementPredictor | ConvDecoderPredictor |
|---|---|---|
| Output | `Linear(512, N×3)` | `(3, H, W)` field + bilinear sample |
| Parameters (N=2601, G=20) | 30 M | 170 K |
| GPU memory (weights + Adam) | ~364 MB | ~2 MB |
| Node-count fixed | Yes — retrain for new mesh | No — sample at any coordinates |
| STL inference (Layer 2) | RBF interpolation | Exact bilinear sampling |

### Evaluating via GUI
<img src="https://raw.githubusercontent.com/onestr1/peen-ml/refs/heads/main/images/load_model_page.png" alt="Load Model" width="400">

1. Open the **Load & Evaluate** tab.
2. Browse to the `.pth` model file and a `Simulation_N/` eval folder.
3. Set the output path and click **Evaluate Model**.
4. Click **Preview Deformation** to view the predicted displacement field.

## Curved Surface and STL Support

The pipeline supports inference on arbitrary 3D shell geometries:

### Programmatic usage
```python
from model import curved_surface_inference
from nozzle_trajectory import ScanParams, raster_scan

traj = raster_scan(ScanParams(
    Lx=0.05, Ly=0.05, z_standoff=0.15,
    scan_speed=0.02, line_spacing=0.005, dt=0.1,
))

result = curved_surface_inference(
    model_path="Dataset_Python/saved_model_conv/trained_conv_decoder_full_model.pth",
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

### GUI usage
In the **Load & Evaluate** tab, expand the **Optional — Curved Surface & Nozzle Trajectory** section:
1. Browse to an STL file.
2. Choose *Parametric scan* (select pattern, set speed/spacing/standoff) or *Waypoint file* (CSV or .npy).
3. Click **Evaluate Model** — inference runs the full three-layer pipeline automatically.
4. Click **Preview STL Deformation** to view the 3D coloured surface.

## Data Visualization
`data_viz.py` provides functions called from the GUI and usable standalone:

```bash
python src/peen-ml/data_viz.py   # demo run
```

Generates plots for:
- Checkerboard patterns (shot coverage heatmap).
- Undeformed vs. deformed FEA mesh.
- Deformation magnitude on deformed mesh.
- 3D STL surface coloured by displacement magnitude (`visualize_stl_deformation`).
- 3D STL surface coloured by von Mises stress (`visualize_stl_stress`).

## Testing
Tests are in the `tests/` directory and cover the full stack.

```bash
pytest tests/
```

| Test file | What it covers |
|---|---|
| `test_model.py` | Model creation, training, evaluation, Layer 1/2 interpolation, batch-dim correctness |
| `test_Shotpeen_Gui.py` | GUI widget construction and callback wiring |
| `test_data_viz.py` | Visualization utility functions |
| `test_stl_surface.py` | STL loading, vertex normals, shot projection, checkerboard building |
| `test_nozzle_trajectory.py` | Raster/spiral/zigzag scan coverage, CSV/npy round-trips |
| `test_curved_surface_sim.py` | Curved simulation on flat STL, flat-plate equivalence |

## License
This project is licensed under the [MIT License](LICENSE).

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
  - Linting and code coverage efforts.

- [Jiachen Zhong](mailto:jczhong@uw.edu)
  *Contributions:*
  - Designed and implemented the CNN model architecture for deformation prediction.
  - Optimized the model pipeline for efficient training and inference.

- [Xuanyu Shen](mailto:xshen20@uw.edu)
  *Contributions:*
  - Developed test cases to ensure functionality and reliability of scripts and models.

## Acknowledgments
- University of Washington CSE 583 course staff for guidance on best practices in software development for data scientists.
- The broader Python and open-source community for providing tools and libraries that made this project possible.
