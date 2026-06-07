---
title: 'peen-ml: A Python Package for Machine Learning Surrogate Modeling of Shot Peening Deformation'
tags:
  - Python
  - shot peening
  - surrogate model
  - convolutional neural network
  - manufacturing simulation
  - finite element analysis
  - material science
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

# Summary

Shot peening is a surface treatment process used in aerospace, automotive, and power generation
manufacturing to induce compressive residual stresses that improve fatigue life and corrosion
resistance. Process engineers must select shot parameters—media size, velocity, coverage pattern,
and material—that achieve target deformation profiles without over-peening. Validating a parameter
set traditionally requires finite element analysis (FEA) simulations that can take hours per run
and require expensive commercial licenses.

`peen-ml` is a Python package that trains convolutional neural networks (CNNs) as rapid surrogates
for shot peening simulations, reducing prediction time from hours to under one second. The package
includes a built-in Python-native physics simulator (no commercial FEA software required), three
CNN architectures suited to different mesh sizes and memory constraints, a material property
library spanning five workpiece alloys and five shot media, a three-layer mesh interpolation
pipeline for arbitrary-resolution inference, full support for curved 3D geometries via STL files,
and a graphical user interface (GUI) that exposes the entire workflow to engineers without machine
learning expertise.

# Statement of Need

Shot peening is specified by manufacturing standards for safety-critical components including
turbine blades, landing gear struts, and drive shafts. Each design iteration requires evaluating
how changes in shot parameters (diameter, velocity, coverage) alter the resulting surface
displacement field. Full FEA simulations in commercial packages such as Abaqus or LS-DYNA take
hours per configuration on high-performance hardware, making rapid parameter sweeps impractical
for design exploration. Analytical models such as the Shen & Atluri (2006) elastic-plastic impact
model [@shenatluri2006] provide closed-form single-shot solutions but cannot directly predict
multi-shot spatial displacement fields over arbitrary coverage patterns.

`peen-ml` addresses this gap with three contributions. First, a Python-native multi-shot physics
simulator completes one simulation in approximately 2 seconds on a modern CPU, eliminating the
need for commercial FEA software and allowing dataset generation at scale. Second, three CNN
architectures—described in the Software Design section—perform inference on a new shot pattern in
under one second. Third, a GUI makes the full generate–train–evaluate workflow accessible to
process engineers and students who are not Python or machine learning specialists.


# State of the Field

No existing open-source Python package combines shot peening physics simulation, CNN surrogate
training, and curved-surface inference in a single tool.

General surrogate modeling toolkits such as the Surrogate Modeling Toolbox (SMT) [@bouhlel2019smt]
provide Gaussian processes and polynomial-chaos surrogates but lack shot-peening domain physics,
spatial field outputs, or a GUI. High-throughput simulation frameworks for materials science
such as strucscan [@strucscan2022] target atomic-scale density-functional-theory calculations and
do not address manufacturing deformation prediction. Commercial packages such as IACS PeenScan
and Abaqus shot-peening utilities require proprietary licenses and do not expose ML training
pipelines or Python APIs.

The `peen-ml` ConvDecoderPredictor produces a full 3D displacement field rather than scalar
or point predictions, allowing evaluation at any mesh resolution after a single training run.
The SIRENPredictor variant [@sitzmann2020siren] further supports very large meshes (millions
of nodes) through coordinate-based implicit inference, with memory cost independent of total
mesh size. Neither capability is present in any of the tools above, and combining them with
a material-aware training pipeline and an STL-based curved-surface mode constitutes the primary
novelty of `peen-ml` as an open-source tool.

# Software Design

`peen-ml` is organized into five subsystems, shown schematically in \autoref{fig:pipeline}.

![Overview of the peen-ml pipeline. Shot parameters and coverage patterns enter the physics simulator, which generates training datasets. A CNN surrogate is trained on these datasets and performs inference on new patterns in under one second. The three-layer interpolation system adapts predictions to arbitrary mesh resolutions and curved STL geometries.\label{fig:pipeline}](images/pipeline_overview.png){ width=100% }

## Physics Simulator

`impact_sim.py` implements the Shen & Atluri (2006) closed-form elastic-plastic impact model
[@shenatluri2006], covering Hertzian contact theory (Equations 1–8), bilinear hardening
plasticity (Equations 15–26), plastic zone geometry (Equations 41–45), and residual stress
depth profiles (Equations 27–36). `multi_shot_sim.py` extends this to multi-shot patterns
over a structured mesh, and `native_dataset_gen.py` parallelises generation across CPU
cores using Python's `multiprocessing` module. A 50×50-node mesh simulation completes in
approximately 2 seconds on a laptop CPU, compared to hours for equivalent FEA analyses.

## Material Library

`materials.py` provides a curated dictionary of five aerospace workpiece alloys
(Ti-6Al-4V, 316L stainless steel, 4340 steel, Al-7075-T6, Inconel-718) and five shot media
(steel, ceramic, glass, cast iron, tungsten), with sourced mechanical properties. A
10-dimensional normalised conditioning vector (seven material property scalars plus impact
velocity V, shot diameter D, and shot count n, all log-scaled) is concatenated to encoder
outputs, enabling a single trained model to generalise across all 25 material combinations
while retaining amplitude information.

## Three CNN Architectures

All three architectures share a common three-block CNN encoder with interleaved channel and
spatial attention modules [@cbam2018]. They differ in how they decode to displacement predictions:

**DisplacementPredictor** uses a fully connected decoder that maps the flattened encoder
output directly to N×3 nodal displacements. It is simple to train but requires retraining for
each mesh size and reaches approximately 30 million parameters at a 50×50 mesh.

**ConvDecoderPredictor** (recommended) uses a lightweight convolutional decoder to produce
a full (3, H, W) displacement field, then samples it at arbitrary node coordinates via bilinear
interpolation. At ~170 K parameters—178× fewer than DisplacementPredictor—it generalises to any
evaluation mesh resolution without retraining. Material conditioning is applied via Feature-wise
Linear Modulation (FiLM) [@film2018], and reflect padding enforces zero-flux boundary conditions
at plate edges.

**SIRENPredictor** [@sitzmann2020siren] conditions a sinusoidal decoder on a latent vector,
queried at explicit (x, y) coordinates. Subsampling 512 nodes per forward pass makes GPU memory
independent of total mesh size, enabling training on million-node industrial meshes (~2 M parameters).

## Three-Layer Mesh Interpolation

Mismatched evaluation meshes are handled by three sequential operations: (1) bilinear resize
of the checkerboard input; (2) TPS-RBF interpolation (FC model) or bilinear field sampling
(convolutional decoder) from training to evaluation coordinates; (3) Rodrigues rotation of
displacement vectors into the local surface normal frame of each STL vertex.

## Curved Surface and Nozzle Trajectory

`stl_surface.py` loads arbitrary STL meshes, computes per-vertex normals, and builds a
KD-tree for nearest-vertex lookup. `nozzle_trajectory.py` generates parameterised scan
patterns (raster, spiral, zigzag) or reads custom waypoints from CSV or NumPy files.
`curved_surface_sim.py` orchestrates full inference on curved surfaces by composing the
flat-plate predictor, the three-layer interpolation system, and the normal-frame rotation.

## Graphical User Interface

`shotpeen_gui.py` provides a Tkinter-based three-panel interface (Generate Dataset, Train Model,
Load & Evaluate) shown in \autoref{fig:gui}. All three CNN architectures, material selection,
STL curved-surface mode, and nozzle trajectory configuration are accessible through the GUI
without requiring Python or ML knowledge. Training runs in a background thread with epoch-by-epoch
loss streamed to a log panel.

![The peen-ml GUI. Left: Generate Dataset panel with material and shot parameter controls. Centre: Train Model panel showing architecture selection and training progress. Right: Load & Evaluate panel with STL curved-surface and nozzle trajectory options.\label{fig:gui}](images/gui_composite.png){ width=100% }

# Accuracy and Performance

\autoref{fig:pred} shows the ConvDecoderPredictor on a held-out test simulation from a
500-simulation benchmark (350 train / 75 val / 75 test; Ti-6Al-4V plate, 961 nodes,
10×10 shot-density checkerboard, V = 25–55 m/s, D = 0.4–0.9 mm, 300–1000 shots).
CNN inference runs in under one second on a laptop CPU, versus 1.4 s for the physics
simulator on 4 CPU cores—and orders of magnitude faster than FEA.

![The ConvDecoderPredictor applied to a median-accuracy held-out test simulation (Simulation\_427). Left: 10×10 shot-density checkerboard input. Centre-left: ground-truth out-of-plane displacement $u_z$ at the 31×31 node level. Centre-right: CNN prediction—the model captures the macro spatial trend but not sub-cell shot positions, which are unresolvable from the density representation. Right two panels: ground truth and prediction averaged to the 10×10 checkerboard cell level; the median cell-averaged pattern correlation across 75 test simulations is $r = 0.28$ (RMSE = 13 µm). A median-accuracy simulation is shown rather than a best-case result.\label{fig:pred}](images/pred_vs_gt.png){ width=100% }

The cell-averaged comparison (rightmost two panels of \autoref{fig:pred}) shows that the
model correctly identifies which plate regions receive the most deformation. Node-level
accuracy is lower ($r = 0.19$, RMSE = 26 µm, rel RMSE = 17%) because individual shot
crater positions within each density cell are not recoverable from the checkerboard
alone—a fundamental limit of the density representation. Higher-resolution checkerboards
or explicit shot-position inputs would reduce this gap.

| Architecture | Parameters | Inference | Cell-avg $r$ | Node rel RMSE |
|---|---|---|---|---|
| ConvDecoderPredictor | ~170 K | <1 s | 0.28 | 18% |
| DisplacementPredictor (FC) | ~30 M | <1 s | — | — |
| SIRENPredictor | ~2 M | <1 s | — | — |

: Benchmark results for the ConvDecoderPredictor on 75 held-out test simulations (Ti-6Al-4V workpiece, steel shot, 300–1000 shots/sim, 30×30 mesh, 10×10 checkerboard). Cell-avg $r$ measures spatial pattern correlation (scale-invariant). Node rel RMSE = RMSE / peak\_gt × 100% measures scale accuracy on nodes with |u_z| > 5% of the field maximum. FC and SIREN rows require retraining on the same dataset. Inference measured on an NVIDIA RTX 4090 Laptop GPU. \label{tab:results}

Training on larger datasets spanning the full parameter range (V = 10–80 m/s, 200 sims per
material pair) reveals a strong dependence on input representation. The checkerboard
architecture achieves in-plane $r = 0.33$–$0.59$ across five material pairs; the
InfluenceField ConvDecoder—using four physics-derived spatial maps (Hertz depth, shot KDE,
lateral forces $F_x$, $F_y$)—achieves $r = 0.95$–$0.97$ in-plane and $r = 0.67$–$0.71$
out-of-plane, with 8–11\% relative RMSE. See \autoref{tab:largescale}.

| Architecture | Input | Mesh | ux $r$ | ux rel RMSE | uz $r$ | RMSE ux |
|---|---|---|---|---|---|---|
| ImprovedDispl. (Ti+steel) | Checkerboard | 51×51 | 0.37 | 40% | 0.13 | 13.9 µm |
| ImprovedDispl. (316L+ceramic) | Checkerboard | 51×51 | 0.59 | 77% | 0.08 | 52.5 µm |
| ImprovedDispl. (Al+glass) | Checkerboard | 51×51 | 0.37 | 34% | 0.16 | 7.5 µm |
| MatCond (25 combos, mat+shot cond.) | Checkerboard | 51×51 | 0.67 | 21% | 0.27 | 14.7 µm |
| ConvDecoder HighRes (Ti+steel) | Checkerboard | 101×101 | 0.30 | 30% | 0.04 | 13.4 µm |
| **I-Ti+steel** | **Physics maps** | **51×51** | **0.95** | **11%** | **0.70** | **3.7 µm** |
| **I-316L+ceramic** | **Physics maps** | **51×51** | **0.97** | **8%** | **0.67** | **12.4 µm** |
| **I-Al+glass** | **Physics maps** | **51×51** | **0.95** | **11%** | **0.67** | **2.3 µm** |
| **I-Inconel+tungsten** | **Physics maps** | **51×51** | **0.96** | **9%** | **0.71** | **5.9 µm** |
| **I-4340+cast iron** | **Physics maps** | **51×51** | **0.97** | **8%** | **0.68** | **8.6 µm** |

: Large-scale results. "200 sims" rows use 200 simulations per material pair; MatCond uses 5000 total across 25 combinations. Parameter range: V = 10–80 m/s, D = 0.1–1.5 mm. ConvDecoder HighRes uses a 101×101 output grid (300 Ti+steel sims); all others use a 51×51 grid. $r$ = Pearson correlation (pattern fidelity, scale-invariant). rel RMSE = RMSE / peak\_gt $\times$ 100\% (scale accuracy, nodes with $|u| > 5\%$ of maximum). Physics maps = four per-simulation fields (Hertz depth, KDE, $F_x$, $F_y$). \label{tab:largescale}

# Limitations

Per-simulation normalization of checkerboard channels can reduce the model's ability to
distinguish different impact velocities; users training on wide velocity ranges (e.g., 10–80 m/s)
should enable the V, D, n conditioning scalars in the 10-dimensional conditioning vector.
Cross-material deployment transfers spatial patterns well (Pearson $r = 0.74$–$0.89$) but may
produce incorrect absolute displacements without material-specific rescaling; the package reports
both pattern correlation and relative RMSE to make this distinction explicit.

# Research Impact Statement

`peen-ml` lowers the computational and licensing barriers to shot peening process simulation,
enabling rapid parameter exploration on commodity hardware without commercial FEA software.
The 178-fold parameter reduction of ConvDecoderPredictor makes it practical to train and
deploy on a standard laptop, while the SIREN variant supports industrial mesh sizes.
The combination of a physics-grounded simulator, material-aware CNN training, and a
beginner-friendly GUI supports reproducible shot peening research and serves as a teaching
tool for surrogate modeling in manufacturing courses.

# AI Usage Disclosure

The authors used Claude (Anthropic) to assist with code documentation, README writing,
and paper drafting. All AI-assisted content was reviewed, edited, and verified by the human
authors. All software algorithms, architecture designs, physics model implementations, training
experiments, and reported results are the original work of the authors.

# Acknowledgements

The authors thank the instructors and teaching assistants of CSE 583 (Software Development
for Data Scientists) at the University of Washington for project guidance and feedback.
The physics simulator is grounded in the analytical model of @shenatluri2006.

# References
