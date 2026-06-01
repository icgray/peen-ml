"""
gaussian_nozzle_dataset_gen.py
================================
Gaussian nozzle shot peening dataset generator for peen-ml.

Replaces the uniform-random / synthetic-pattern shot placement in
``native_dataset_gen.py`` with a physically realistic nozzle model:

  * Impact positions follow a 2-D Gaussian centred on the nozzle footprint.
    Spatial spread  sxy = h_nozzle × tan(θ_div)  (geometry of the jet cone).
  * Shot exit velocities follow a Gaussian distribution N(V_mean, sigma_V²),
    truncated below V_min.
  * Each shot's radial offset r from the nozzle axis determines its impact
    angle  alpha = arctan(r / h_nozzle).  The effective normal velocity is
    V_n = V_exit × cos(alpha)  (Hertz contact uses the normal component only).
  * The angle is passed through ``ShotPeenParams.phi = π/2 − alpha`` so the
    existing Shen & Atluri model handles it correctly.
  * The checkerboard ML input is the normalised effective-energy density per
    cell:  Σ V_n² for shots in cell, normalised to [0, 1].  This encodes
    both spatial density and angle-corrected impact strength.

Grid-size note
--------------
Requested: 40 m × 40 m, 100 divisions per metre → Nx = Ny = 4 000,
N_nodes ≈ 16 million.  At the current analytical-superposition rate
(~0.5 µs per node × shot), 50 shots on 16 M nodes ≈ 400 s/sim.
For a 200-case dataset that is ~22 h — feasible on a workstation only with
many parallel workers.

Default chosen: Lx = Ly = 0.040 m (40 mm), Nx = Ny = 100 → 10 201 nodes.
This runs in ~3–8 s/sim on a laptop CPU.

To simulate a larger domain while staying tractable, reduce the resolution
to, e.g., Nx = Ny = 200 on a 0.5 m × 0.5 m plate (90 601 nodes;
~60 s/sim × 50 workers → feasible overnight).  Set ``Lx``, ``Ly``, ``Nx``,
``Ny`` in ``GaussianNozzleParams`` to match your hardware.

Dataset exploration axes
------------------------
Variable                Range (default)          Physical meaning
-----------             ---------------          -------------------
h_nozzle                [50–400 mm]              Standoff height
theta_div               [5–30deg]                  Jet cone half-angle
V_mean                  [25–80 m/s]              Mean exit velocity
sigma_V_frac            [0.03–0.20]              Velocity CV = sigma_V / V_mean
n_shots                 [30–200]                 Shots per peening pass
D (diameter)            [0.3–1.0 mm]             Shot size
rho_shot (density)      steel / ceramic / glass  Shot material
sigma_yield             [200–800 MPa]            Workpiece yield stress
nozzle_x, nozzle_y      [0.1–0.9] × plate size  Nozzle x,y position

Generated data format
---------------------
Each ``Simulation_<k>/`` folder contains the same .npy files as
Abaqus / multi_shot_sim.py output (compatible with model.py, data_viz.py):

    Simulation_0/
        checkerboard.npy            (G, G) float32 — normalised energy density
        displacements.npy           (N_nodes, 3) float32 — [ux, uy, uz]
        disp_node_labels.npy        (N_nodes,) int32
        node_coords.npy             (N_nodes, 3) float32
        node_labels.npy             (N_nodes,) int32
        element_connectivity.npy    (N_elems, 4) int32
        element_labels.npy          (N_elems,) int32
        stresses.npy                (N_elems, 4) float32 — [S11,S22,S33,S12]
        stress_element_labels.npy   (N_elems,) int32
        sR_depth_profile.npy        (L, 2) float32 — [depth, sigmaR]
        shot_positions.npy          (N_shots_actual, 2) — (x, y) on plate
        shot_V_normal.npy           (N_shots_actual,) — per-shot V_n (m/s)
        shot_angles.npy             (N_shots_actual,) — per-shot alpha (rad)
        nozzle_params.txt           — traceability
        energy_balance.txt

Usage
-----
    python gaussian_nozzle_dataset_gen.py --n_sims 200 --output ./Dataset_Gaussian

From Python:
    from gaussian_nozzle_dataset_gen import GaussianNozzleParams, generate_gaussian_dataset
    gp = GaussianNozzleParams(n_simulations=50, output_dir="./Dataset_Gaussian")
    generate_gaussian_dataset(gp)

References
----------
Shen & Atluri (2006) CMC vol. 4 no. 2 pp. 75-85.
Zinn & Lohman (1999) "Shot peening nozzle design", Proc. ICSP 7.
Bhuvaraghan et al. (2010) Int. J. Mech. Sci. 52(10), 1220–1229.
"""
from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional PyTorch import — used for the CUDA batched superposition kernel.
# The module works without PyTorch (falls back to the CPU shot loop).
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:          # pragma: no cover
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Local imports — same source directory as multi_shot_sim.py
# ---------------------------------------------------------------------------
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from impact_sim import (           # noqa: E402
    ShotPeenParams,
    generate_mesh,
    compute_contact_params,
    compute_stress_field,
    compute_plastic_zone,
    compute_energy_balance,
    map_displacements,
    map_stresses,
)
from multi_shot_sim import compute_coverage   # noqa: E402
from materials import (                       # noqa: E402
    SHOT_MATERIALS,
    WORKPIECE_MATERIALS,
    get_shot,
    get_workpiece,
)

__all__ = [
    "GaussianNozzleParams",
    "SHOT_MATERIALS",
    "WORKPIECE_MATERIALS",
    "sample_gaussian_nozzle_shots",
    "gaussian_density_checkerboard",
    "run_gaussian_nozzle_simulation",
    "generate_gaussian_dataset",
    "validate_gaussian_dataset",
]


# ---------------------------------------------------------------------------
# 2.  Parameter container
# ---------------------------------------------------------------------------

@dataclass
class GaussianNozzleParams:
    """Configuration for the Gaussian nozzle shot peening dataset generator.

    Nozzle geometry
    ---------------
    h_range      : (h_min, h_max) standoff height (m).
    theta_div_range : (θ_min, θ_max) jet half-angle (rad).
                      Controls Gaussian spread: sxy = h × tan(θ_div).
    nozzle_pos_frac : (frac_min, frac_max) nozzle (x, y) as fraction of
                      plate width.  Default [0.3, 0.7] keeps the Gaussian
                      footprint mostly on-plate.

    Velocity distribution
    ---------------------
    V_mean_range   : (V_min, V_max) mean exit velocity (m/s).
    sigma_V_frac_range : (cv_min, cv_max) velocity coefficient of variation
                         sigma_V = cv × V_mean.
    V_exit_min     : Hard lower bound on sampled exit velocities (m/s).

    Shots
    -----
    n_shots_range  : (n_min, n_max) total shots per simulation.
    D_range        : (D_min, D_max) shot diameter (m).
    shot_materials : List of material names to draw from each simulation.
                     Each name must be a key in SHOT_MATERIALS.

    Workpiece
    ---------
    sigma_yield_range : (sy_min, sy_max) workpiece yield stress (Pa).
    E_b               : Workpiece Young's modulus (Pa; fixed at Ti-6Al-4V default).
    nu_b              : Workpiece Poisson's ratio.

    Mesh
    ----
    Lx, Ly  : Plate dimensions (m).  Default 40 mm × 40 mm.
    Nx, Ny  : Number of quad elements per axis.  Default 100 × 100.
              → 101 × 101 = 10 201 nodes, ~3–8 s/sim on a laptop.
    Lz      : Plate thickness for mesh (m).

    Checkerboard
    ------------
    checkerboard_size : G for the G × G shot-energy-density grid saved as ML
                        input.  Default 20 (4× finer than the legacy 5 × 5).

    Output / reproducibility
    ------------------------
    output_dir    : Root directory; sims go into <output_dir>/Simulation_<k>/.
    n_simulations : Number of simulation cases.
    start_index   : Starting index (useful for resuming or distributed runs).
    workers       : Parallel worker processes (1 = sequential).
    base_seed     : Master RNG seed; each simulation gets seed = base_seed + k.
    """

    # Nozzle geometry
    h_range: Tuple[float, float] = (0.050, 0.400)        # m: 50–400 mm
    theta_div_range: Tuple[float, float] = (               # rad: 5deg–30deg
        math.radians(5.0),
        math.radians(30.0),
    )
    nozzle_pos_frac: Tuple[float, float] = (0.30, 0.70)  # fraction of Lx/Ly

    # Velocity distribution
    V_mean_range: Tuple[float, float] = (25.0, 80.0)     # m/s
    sigma_V_frac_range: Tuple[float, float] = (0.03, 0.20)
    V_exit_min: float = 5.0                               # m/s hard lower bound

    # Shots
    n_shots_range: Tuple[int, int] = (30, 200)
    D_range: Tuple[float, float] = (0.0003, 0.0010)      # m
    shot_materials: List[str] = field(
        default_factory=lambda: ["steel", "ceramic", "glass", "cast_iron"]
    )

    # Workpiece
    sigma_yield_range: Tuple[float, float] = (200e6, 800e6)  # Pa
    E_b: float = 113.8e9     # Ti-6Al-4V Young's modulus (Pa)
    nu_b: float = 0.34       # Ti-6Al-4V Poisson's ratio

    # Mesh
    Lx: float = 0.040        # m (40 mm)
    Ly: float = 0.040        # m (40 mm)
    Lz: float = 0.004        # m (plate thickness for mesh)
    Nx: int = 100             # elements per x-axis → 101 nodes
    Ny: int = 100             # elements per y-axis → 101 nodes

    # Checkerboard resolution
    checkerboard_size: int = 20   # G × G grid

    # Output / reproducibility
    output_dir: str = "./Dataset_Gaussian"
    n_simulations: int = 100
    start_index: int = 0
    workers: int = 1
    base_seed: int = 0


# ---------------------------------------------------------------------------
# 3.  Gaussian nozzle shot sampler
# ---------------------------------------------------------------------------

def sample_gaussian_nozzle_shots(
    h_nozzle: float,
    theta_div: float,
    V_mean: float,
    sigma_V: float,
    n_shots: int,
    Lx: float,
    Ly: float,
    nozzle_x: float,
    nozzle_y: float,
    V_exit_min: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample shot positions and velocities from a Gaussian nozzle model.

    The nozzle is located at (nozzle_x, nozzle_y, h_nozzle) above the
    plate surface.  Shots exit with a speed drawn from N(V_mean, sigma_V²) and
    travel in directions whose x-y spread on the plate follows
    N(0, sxy²) × N(0, sxy²)  where  sxy = h_nozzle × tan(θ_div).

    Only shots that land on the plate [0, Lx] × [0, Ly] are returned; the
    caller should ensure n_shots is large enough that the expected number of
    on-plate hits exceeds the target.  In practice, if the Gaussian foot
    print is much smaller than the plate this function samples up to
    5 × n_shots candidates before truncating.

    Parameters
    ----------
    h_nozzle  : Nozzle standoff height above plate surface (m).
    theta_div : Jet cone half-angle (rad).  sxy = h × tan(θ_div).
    V_mean    : Mean exit velocity (m/s).
    sigma_V   : Std-dev of exit velocity (m/s).
    n_shots   : Target number of shots (on-plate hits).
    Lx, Ly   : Plate dimensions (m).
    nozzle_x  : X-coordinate of nozzle centre above plate (m).
    nozzle_y  : Y-coordinate of nozzle centre above plate (m).
    V_exit_min: Minimum allowed exit velocity (m/s) after truncation.
    rng       : Numpy random generator.

    Returns
    -------
    centres  : (N, 2) float — (x, y) impact positions on plate.
    V_normal : (N,)   float — normal-component velocity V_n = V_exit × cos(alpha).
    V_exit   : (N,)   float — raw exit velocities before angle correction.
    alpha    : (N,)   float — impact angle from surface normal (rad).
               alpha = 0 → perpendicular impact;  alpha = π/2 → grazing.
    """
    sigma_xy = h_nozzle * math.tan(theta_div)     # spatial Gaussian spread (m)

    # Over-sample to account for shots landing off the plate.
    # The fraction inside the plate depends on how sxy compares to the
    # plate size; use a generous 5× safety factor with a hard cap.
    n_candidate = min(n_shots * 10, max(n_shots * 5, 2000))

    # ---- Sample exit velocities (truncated Gaussian) ----
    V_raw = rng.normal(V_mean, sigma_V, n_candidate)
    V_raw = np.clip(V_raw, V_exit_min, None)    # truncate below physical minimum

    # ---- Sample impact positions from 2-D Gaussian ----
    dx = rng.normal(0.0, sigma_xy, n_candidate)   # x offset from nozzle axis (m)
    dy = rng.normal(0.0, sigma_xy, n_candidate)   # y offset from nozzle axis (m)
    x_imp = nozzle_x + dx
    y_imp = nozzle_y + dy

    # ---- Keep only on-plate shots ----
    on_plate = (
        (x_imp >= 0.0) & (x_imp <= Lx) &
        (y_imp >= 0.0) & (y_imp <= Ly)
    )
    x_imp = x_imp[on_plate]
    y_imp = y_imp[on_plate]
    V_raw = V_raw[on_plate]

    # ---- Trim or pad to exactly n_shots ----
    if len(x_imp) >= n_shots:
        # More shots than needed — trim
        idx = rng.choice(len(x_imp), size=n_shots, replace=False)
        x_imp, y_imp, V_raw = x_imp[idx], y_imp[idx], V_raw[idx]
    elif len(x_imp) == 0:
        # Edge case: nozzle so far off-plate that no shots land on it.
        # Fall back to uniform random to avoid an empty dataset.
        x_imp = rng.uniform(0.0, Lx, n_shots)
        y_imp = rng.uniform(0.0, Ly, n_shots)
        V_raw = rng.normal(V_mean, sigma_V, n_shots).clip(V_exit_min)
    else:
        # Fewer than requested — tile and trim (preserves the distribution shape)
        repeats = math.ceil(n_shots / len(x_imp))
        x_imp = np.tile(x_imp, repeats)[:n_shots]
        y_imp = np.tile(y_imp, repeats)[:n_shots]
        V_raw = np.tile(V_raw, repeats)[:n_shots]

    # ---- Compute angle-corrected normal velocity ----
    r = np.sqrt((x_imp - nozzle_x) ** 2 + (y_imp - nozzle_y) ** 2)
    alpha = np.arctan2(r, h_nozzle)              # impact angle from normal (rad)
    V_normal = V_raw * np.cos(alpha)             # Hertz normal component (m/s)
    V_normal = np.clip(V_normal, V_exit_min, None)   # safety floor

    centres = np.stack([x_imp, y_imp], axis=1).astype(np.float32)
    return centres, V_normal.astype(np.float32), V_raw.astype(np.float32), alpha.astype(np.float32)


# ---------------------------------------------------------------------------
# 4.  Derive checkerboard from Gaussian shot distribution
# ---------------------------------------------------------------------------

def gaussian_density_checkerboard(
    centres: np.ndarray,
    V_normal: np.ndarray,
    Lx: float,
    Ly: float,
    G: int,
    mode: str = "energy",
) -> np.ndarray:
    """Build a (G, G) checkerboard from actual shot positions.

    Parameters
    ----------
    centres  : (N, 2) shot impact positions (m).
    V_normal : (N,)   normal velocity of each shot (m/s).
    Lx, Ly  : Plate dimensions (m).
    G        : Grid size — output is (G, G).
    mode     : Aggregation mode:
               'energy'   — sum V_n² per cell (default; encodes energy density)
               'count'    — raw shot count per cell
               'velocity' — mean V_n per cell

    Returns
    -------
    cb : (G, G) float32 in [0, 1].  Zero cells where no shots landed.
    """
    cb = np.zeros((G, G), dtype=np.float64)
    cell_w = Lx / G
    cell_h = Ly / G

    col_idx = np.clip((centres[:, 0] / cell_w).astype(int), 0, G - 1)
    row_idx = np.clip((centres[:, 1] / cell_h).astype(int), 0, G - 1)

    if mode == "energy":
        # Sum of V_n² per cell (proportional to kinetic energy deposited)
        for col, row, vn in zip(col_idx, row_idx, V_normal):
            cb[row, col] += vn * vn

    elif mode == "count":
        # Raw shot count per cell
        for col, row in zip(col_idx, row_idx):
            cb[row, col] += 1.0

    elif mode == "velocity":
        # Mean V_n per cell (requires a separate count array)
        counts = np.zeros((G, G), dtype=np.float64)
        for col, row, vn in zip(col_idx, row_idx, V_normal):
            cb[row, col] += vn
            counts[row, col] += 1.0
        mask = counts > 0
        cb[mask] /= counts[mask]

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'energy', 'count', or 'velocity'.")

    # Normalise to [0, 1]
    cb_max = cb.max()
    if cb_max > 0.0:
        cb /= cb_max

    return cb.astype(np.float32)


# ---------------------------------------------------------------------------
# 5b. PyTorch CUDA batched superposition kernel
# ---------------------------------------------------------------------------
#
# The CPU shot loop in section 5 is O(N_shots x N_nodes) and runs
# sequentially.  This section replaces it with three functions:
#
#   _compute_elem_centroids   — vectorised element centroid computation
#   _precompute_shot_scalars  — CPU loop that extracts one (a_p, r_p,
#                               delta_p, sR_surface) tuple per shot using
#                               the Shen & Atluri functions; also collects
#                               energy_list / sR_profiles / plastic_ref
#   _superpose_shots_batched  — PyTorch kernel that builds the full
#                               (N_shots, N_nodes) distance matrix and
#                               evaluates the piecewise displacement and
#                               Gaussian stress formulas in a single fused
#                               vectorised pass, automatically using CUDA
#                               when available and chunking over nodes to
#                               stay within GPU memory
#
# The formulas match map_displacements() / map_stresses() in impact_sim.py
# exactly so CPU and GPU paths produce numerically identical results.
# ---------------------------------------------------------------------------

def _compute_elem_centroids(mesh: Dict) -> np.ndarray:
    """Return element centroid coordinates as a (N_elems, 3) float32 array.

    Replicates the per-element centroid loop inside ``map_stresses`` but
    using NumPy fancy indexing so it runs in O(N_elems) instead of a Python
    loop.  Needed to build the element position tensor for the CUDA kernel.

    Parameters
    ----------
    mesh : dict returned by ``generate_mesh()``.

    Returns
    -------
    centroids : (N_elems, 3) float32.
    """
    node_coords  = mesh["node_coords"]          # (N_nodes, 3) float32
    node_labels  = mesh["node_labels"]          # (N_nodes,)   int32
    connectivity = mesh["element_connectivity"] # (N_elems, 4) int32

    # Build label -> 0-based index mapping (same as map_stresses)
    label_to_idx = np.empty(int(node_labels.max()) + 1, dtype=np.int64)
    label_to_idx[node_labels.astype(np.int64)] = np.arange(len(node_labels), dtype=np.int64)

    # Map each connectivity entry to a 0-based index, then average
    idx_matrix = label_to_idx[connectivity.astype(np.int64)]  # (N_elems, 4)
    centroids   = node_coords[idx_matrix].mean(axis=1)         # (N_elems, 3)
    return centroids.astype(np.float32)


def _precompute_shot_scalars(
    n_actual:     int,
    alpha:        np.ndarray,   # (N_actual,) impact angles from normal (rad)
    V_exit:       np.ndarray,   # (N_actual,) exit speeds (m/s)
    D:            float,
    mat_props:    Dict[str, float],
    sigma_yield:  float,
    E_b:          float,
    nu_b:         float,
) -> Tuple[Dict[str, np.ndarray], List, List, Dict]:
    """Pre-compute Shen & Atluri scalar parameters for every shot on the CPU.

    Runs the same per-shot physics as the existing loop (Hertz contact,
    plastic zone, stress field) but only extracts four scalars per shot:

        a_p       — dent radius (m)       [Eq 44]
        r_p       — plastic zone radius   [Eq 43]
        delta_p   — permanent indent depth = a_p^2 / (2 R)
        sR_surface— surface residual stress (Pa) = sR_profile[0]
                    (= sR_profile evaluated at z = ae/1000, the shallowest
                     depth point, used as the left-extrapolated surface value
                     — exactly what map_stresses does for z_elem = 0)

    These four scalars are all the CUDA kernel needs; the spatial loop over
    N_nodes / N_elems is offloaded to GPU.

    The function also collects energy_list and sR_profiles (needed for
    coverage reporting and the mean depth-profile plot) and returns
    plastic_ref (the first shot's plastic zone, used for Avrami coverage).

    Parameters
    ----------
    n_actual    : Number of on-plate shots.
    alpha       : Per-shot impact angle from surface normal (rad).
    V_exit      : Per-shot exit velocity (m/s).
    D           : Shot diameter (m).
    mat_props   : Dict with keys E_s, nu_s, rho_s.
    sigma_yield : Workpiece yield stress (Pa).
    E_b         : Workpiece Young's modulus (Pa).
    nu_b        : Workpiece Poisson's ratio.

    Returns
    -------
    scalars     : dict with keys 'a_p', 'r_p', 'delta_p', 'sR_surface'
                  each a (N_actual,) float64 array.
    energy_list : list of energy dicts (one per shot).
    sR_profiles : list of (L, 2) float32 arrays [[depth, sR], ...].
    plastic_ref : plastic zone dict for the first shot (used for coverage).
    """
    a_p_arr    = np.empty(n_actual, dtype=np.float64)
    r_p_arr    = np.empty(n_actual, dtype=np.float64)
    delta_p_arr= np.empty(n_actual, dtype=np.float64)
    sR_surf_arr= np.empty(n_actual, dtype=np.float64)

    energy_list: List[Dict] = []
    sR_profiles: List[np.ndarray] = []
    plastic_ref: Dict = {}

    for i in range(n_actual):
        # Impact angle from surface (ShotPeenParams convention: phi = pi/2 is normal)
        phi_i = math.pi / 2.0 - float(alpha[i])
        phi_i = max(phi_i, math.radians(5.0))   # numerical floor: avoid grazing singularity

        p_i = ShotPeenParams(
            E_s=mat_props["E_s"],
            nu_s=mat_props["nu_s"],
            D=D,
            rho_s=mat_props["rho_s"],
            E_b=E_b,
            nu_b=nu_b,
            sigma_yield=sigma_yield,
            V=float(V_exit[i]),
            phi=phi_i,
            n_depth=500,    # coarse depth resolution — only sR_surf matters here
        )

        contact_i = compute_contact_params(p_i)
        plastic_i = compute_plastic_zone(p_i)
        sf_i      = compute_stress_field(contact_i, p_i)
        energy_i  = compute_energy_balance(p_i, contact_i, plastic_i)

        a_p_arr[i]     = plastic_i["a_p"]
        r_p_arr[i]     = plastic_i["r_p"]
        delta_p_arr[i] = plastic_i["a_p"] ** 2 / (2.0 * p_i.R)
        # sR_profile[0] is the value at z = ae/1000 (shallowest computed depth).
        # map_stresses uses np.interp(z_elem=0, Z, sR, left=sR[0]) which returns
        # sR[0] because 0 < Z[0]; so this is the exact surface stress used.
        sR_surf_arr[i] = float(sf_i["sR"][0])

        energy_list.append(energy_i)
        sR_profiles.append(
            np.stack([sf_i["Z"], sf_i["sR"]], axis=1).astype(np.float32)
        )
        if i == 0:
            plastic_ref = plastic_i

    scalars = {
        "a_p":        a_p_arr,
        "r_p":        r_p_arr,
        "delta_p":    delta_p_arr,
        "sR_surface": sR_surf_arr,
    }
    return scalars, energy_list, sR_profiles, plastic_ref


def _superpose_shots_batched(
    centres:     np.ndarray,       # (N_shots, 2) float32 — (x,y) on plate
    scalars:     Dict[str, np.ndarray],
    node_xy:     np.ndarray,       # (N_nodes, 2) float32 — surface node (x,y)
    elem_xy:     np.ndarray,       # (N_elems, 2) float32 — element centroid (x,y)
    device:      "torch.device",   # pre-resolved torch.device
    chunk_nodes: int,              # nodes processed per GPU batch
) -> Tuple[np.ndarray, np.ndarray]:
    """PyTorch vectorised shot superposition — runs on CUDA when available.

    Implements the piecewise displacement formula from ``map_displacements``
    and the Gaussian radial attenuation formula from ``map_stresses`` as
    batched tensor operations over the full (N_shots, N_nodes) distance
    matrix, processing *chunk_nodes* columns at a time to bound GPU memory.

    Displacement model (Shen & Atluri / Hertz permanent dent):
        r in [0, a_p]     :  uz = -delta_p * (1 - (r/a_p)^2)
        r in (a_p, r_p]   :  uz = -delta_p * (a_p/r)^2 * exp(-(r-a_p)/a_p)
        r > r_p           :  uz = 0
        ur (radial bulge) :  ur = (2/3)*delta_p*(r/r_p)*exp(-(r/r_p)^2)
        ux = ur*cos(theta), uy = ur*sin(theta)

    Stress model (biaxial residual compression at the surface, z=0):
        sigma_r = r_p / 2
        G_r     = exp(-r^2 / (2*sigma_r^2));  G_r = 0 for r > 3*r_p
        S11 = S22 = sR_surface * G_r
        S33 = S12 = 0

    Parameters
    ----------
    centres     : (N_shots, 2) — impact positions.
    scalars     : dict of per-shot 1-D numpy arrays (from _precompute_shot_scalars).
    node_xy     : (N_nodes, 2) — node x,y coordinates.
    elem_xy     : (N_elems, 2) — element centroid x,y coordinates.
    device      : torch.device (cuda or cpu).
    chunk_nodes : Columns per chunk — controls peak GPU memory use.

    Returns
    -------
    displacements : (N_nodes, 3) float32 numpy array [ux, uy, uz].
    stresses      : (N_elems, 4) float32 numpy array [S11, S22, S33, S12].
    """
    N_shots = len(centres)
    N_nodes = len(node_xy)
    N_elems = len(elem_xy)

    # ---- Move per-shot scalars to GPU (shape (N_shots,)) ----
    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    sx  = _t(centres[:, 0])            # shot x  (N_shots,)
    sy  = _t(centres[:, 1])            # shot y
    ap  = _t(scalars["a_p"])           # dent radius
    rp  = _t(scalars["r_p"]).clamp(min=1e-15)
    dp  = _t(scalars["delta_p"])       # permanent indent depth
    sR  = _t(scalars["sR_surface"])    # surface residual stress

    nx_all = _t(node_xy[:, 0])         # node x  (N_nodes,)
    ny_all = _t(node_xy[:, 1])

    # ---- Displacement accumulator (CPU, to avoid one large GPU alloc) ----
    ux_acc = np.zeros(N_nodes, dtype=np.float64)
    uy_acc = np.zeros(N_nodes, dtype=np.float64)
    uz_acc = np.zeros(N_nodes, dtype=np.float64)

    # ---- Process nodes in chunks ----
    for k0 in range(0, N_nodes, chunk_nodes):
        k1 = min(k0 + chunk_nodes, N_nodes)
        C  = k1 - k0                   # chunk width

        # Node subset: broadcast against shots → (N_shots, C)
        nxc = nx_all[k0:k1].unsqueeze(0)   # (1, C)
        nyc = ny_all[k0:k1].unsqueeze(0)

        dxc = sx.unsqueeze(1) - nxc        # (N_shots, C)
        dyc = sy.unsqueeze(1) - nyc
        rc  = torch.sqrt(dxc * dxc + dyc * dyc)

        # Per-shot scalars broadcast: (N_shots, 1)
        ap1 = ap.unsqueeze(1).clamp(min=1e-15)
        rp1 = rp.unsqueeze(1)
        dp1 = dp.unsqueeze(1)

        # --- Normal displacement uz ---
        in_dent  = rc.le(ap1)                          # r <= a_p
        in_trans = rc.gt(ap1) & rc.le(rp1)             # a_p < r <= r_p

        # Dent region: uz = -dp * (1 - (r/ap)^2)
        uz_dent  = dp1.neg() * (1.0 - (rc / ap1).pow(2))

        # Transition: uz = -dp * (ap/r)^2 * exp(-(r-ap)/ap)
        rc_s     = rc.clamp(min=1e-15)
        uz_trans = dp1.neg() * (ap1 / rc_s).pow(2) * torch.exp((ap1 - rc) / ap1.clamp(min=1e-15))

        uz_chunk = torch.where(in_dent,  uz_dent,
                   torch.where(in_trans, uz_trans,
                                torch.zeros_like(rc)))

        # --- Radial bulge ur → decomposed into ux, uy ---
        ur_chunk = (2.0 / 3.0) * dp1 * (rc / rp1) * torch.exp(-(rc / rp1).pow(2))

        # cos/sin of azimuthal angle; zero at r=0 (shot centre)
        r_safe   = rc.clamp(min=1e-15)
        nonzero  = rc.gt(1e-15)
        cos_t    = torch.where(nonzero, dxc / r_safe, torch.zeros_like(dxc))
        sin_t    = torch.where(nonzero, dyc / r_safe, torch.zeros_like(dyc))

        ux_chunk = ur_chunk * cos_t
        uy_chunk = ur_chunk * sin_t

        # Sum over shots → (C,)
        ux_acc[k0:k1] = ux_chunk.sum(0).cpu().numpy()
        uy_acc[k0:k1] = uy_chunk.sum(0).cpu().numpy()
        uz_acc[k0:k1] = uz_chunk.sum(0).cpu().numpy()

        del dxc, dyc, rc, uz_dent, uz_trans, uz_chunk, ur_chunk, ux_chunk, uy_chunk

    # ---- Stress accumulator ----
    ex_all = _t(elem_xy[:, 0])     # (N_elems,)
    ey_all = _t(elem_xy[:, 1])
    S11_acc = np.zeros(N_elems, dtype=np.float64)

    sig_r = (rp / 2.0).clamp(min=1e-15)    # (N_shots,) Gaussian spread parameter

    for e0 in range(0, N_elems, chunk_nodes):
        e1  = min(e0 + chunk_nodes, N_elems)
        exc = ex_all[e0:e1].unsqueeze(0)   # (1, chunk_e)
        eyc = ey_all[e0:e1].unsqueeze(0)

        dxe  = sx.unsqueeze(1) - exc        # (N_shots, chunk_e)
        dye  = sy.unsqueeze(1) - eyc
        re   = torch.sqrt(dxe * dxe + dye * dye)

        sig1 = sig_r.unsqueeze(1)           # (N_shots, 1)
        rp1  = rp.unsqueeze(1)
        sR1  = sR.unsqueeze(1)

        # Gaussian radial attenuation (cutoff beyond 3*r_p)
        G_r  = torch.exp(-re.pow(2) / (2.0 * sig1.pow(2)))
        G_r  = torch.where(re.gt(3.0 * rp1), torch.zeros_like(G_r), G_r)

        S11_acc[e0:e1] = (sR1 * G_r).sum(0).cpu().numpy()

        del dxe, dye, re, G_r

    # ---- Build output arrays ----
    displacements = np.stack(
        [ux_acc, uy_acc, uz_acc], axis=1
    ).astype(np.float32)

    S11 = S11_acc.astype(np.float32)
    stresses = np.stack(
        [S11, S11.copy(),
         np.zeros(N_elems, dtype=np.float32),
         np.zeros(N_elems, dtype=np.float32)],
        axis=1,
    )                   # (N_elems, 4) — [S11, S22, S33=0, S12=0]

    return displacements, stresses


def _resolve_device_and_chunk(N_shots: int, N_nodes: int) -> Tuple["torch.device", int]:
    """Auto-detect the best PyTorch device and a safe node chunk size.

    Targets using at most ~60% of free GPU memory (or 1 GB on CPU).
    Each displacement chunk needs approximately
    ``9 * N_shots * chunk_nodes * 4`` bytes (9 float32 intermediate tensors).

    Parameters
    ----------
    N_shots : Number of shots in the simulation.
    N_nodes : Total node count (used to cap chunk at full mesh if it fits).

    Returns
    -------
    device      : torch.device ('cuda:0' if a GPU is found, else 'cpu').
    chunk_nodes : int — recommended node chunk for this device/problem size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bytes_per_node_per_shot = 9 * 4     # 9 float32 intermediate tensors
    bytes_total_per_node    = bytes_per_node_per_shot * N_shots

    if device.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            budget = int(free_bytes * 0.60)
        except Exception:
            budget = 4 * (1024 ** 3)    # 4 GB conservative fallback
    else:
        budget = 1 * (1024 ** 3)        # 1 GB on CPU

    chunk = max(4096, min(N_nodes, int(budget / max(1, bytes_total_per_node))))
    return device, chunk


# ---------------------------------------------------------------------------
# 5.  Single-simulation runner
# ---------------------------------------------------------------------------

def run_gaussian_nozzle_simulation(
    sim_index: int,
    gen_params: GaussianNozzleParams,
) -> Dict:
    """Generate and save one Gaussian-nozzle simulation case.

    Each simulation independently draws:
      - Nozzle height, divergence angle, position
      - Exit velocity distribution (mean + CV)
      - Shot diameter and material
      - Number of shots
      - Workpiece yield stress

    Parameters
    ----------
    sim_index  : Simulation number (determines output folder and RNG seed).
    gen_params : GaussianNozzleParams driving the sweep.

    Returns
    -------
    dict with keys: sim_index, output_dir, n_nodes, n_elems,
                    coverage_percent, almen_MPa, elapsed_s, success, error,
                    plus the physical parameters used.
    """
    t0 = time.perf_counter()
    seed = gen_params.base_seed + sim_index
    rng = np.random.default_rng(seed)

    out_folder = os.path.join(gen_params.output_dir, f"Simulation_{sim_index}")

    # ------------------------------------------------------------------
    # Draw randomised parameters
    # ------------------------------------------------------------------

    # Nozzle geometry
    h_nozzle  = float(rng.uniform(*gen_params.h_range))
    theta_div = float(rng.uniform(*gen_params.theta_div_range))
    nozzle_x  = float(rng.uniform(
        gen_params.Lx * gen_params.nozzle_pos_frac[0],
        gen_params.Lx * gen_params.nozzle_pos_frac[1],
    ))
    nozzle_y  = float(rng.uniform(
        gen_params.Ly * gen_params.nozzle_pos_frac[0],
        gen_params.Ly * gen_params.nozzle_pos_frac[1],
    ))

    # Velocity distribution
    V_mean   = float(rng.uniform(*gen_params.V_mean_range))
    cv       = float(rng.uniform(*gen_params.sigma_V_frac_range))
    sigma_V  = cv * V_mean

    # Shots
    n_shots  = int(rng.integers(gen_params.n_shots_range[0],
                                gen_params.n_shots_range[1] + 1))
    D        = float(rng.uniform(*gen_params.D_range))

    # Shot material — pick uniformly from allowed materials
    mat_name  = str(rng.choice(gen_params.shot_materials))
    mat_props = SHOT_MATERIALS[mat_name]

    # Workpiece
    sigma_yield = float(rng.uniform(*gen_params.sigma_yield_range))

    # ------------------------------------------------------------------
    # Sample shot positions and velocities from Gaussian nozzle
    # ------------------------------------------------------------------
    try:
        centres, V_normal, V_exit, alpha = sample_gaussian_nozzle_shots(
            h_nozzle=h_nozzle,
            theta_div=theta_div,
            V_mean=V_mean,
            sigma_V=sigma_V,
            n_shots=n_shots,
            Lx=gen_params.Lx,
            Ly=gen_params.Ly,
            nozzle_x=nozzle_x,
            nozzle_y=nozzle_y,
            V_exit_min=gen_params.V_exit_min,
            rng=rng,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {"sim_index": sim_index, "output_dir": out_folder,
                "elapsed_s": elapsed, "success": False, "error": str(exc)}

    n_actual = len(centres)

    # ------------------------------------------------------------------
    # Generate mesh
    # ------------------------------------------------------------------
    try:
        mesh = generate_mesh(
            Lx=gen_params.Lx,
            Ly=gen_params.Ly,
            Lz=gen_params.Lz,
            Nx=gen_params.Nx,
            Ny=gen_params.Ny,
            Nz=1,
        )
        N_nodes = len(mesh["node_labels"])
        N_elems = len(mesh["element_labels"])
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {"sim_index": sim_index, "output_dir": out_folder,
                "elapsed_s": elapsed, "success": False,
                "error": f"Mesh generation failed: {exc}"}

    # ------------------------------------------------------------------
    # Superpose shots analytically (Shen & Atluri)
    #
    # Two-phase dispatch:
    #   Phase 1 (always CPU) — extract per-shot scalar parameters
    #     (a_p, r_p, delta_p, sR_surface) and collect energy_list /
    #     sR_profiles needed for coverage and the depth-profile plot.
    #     Cost: O(N_shots x n_depth_coarse) — fast even for 200 shots.
    #   Phase 2 — spatial superposition:
    #     If PyTorch is available: batched (N_shots x N_nodes) GPU kernel
    #       via _superpose_shots_batched(), chunked to fit GPU memory.
    #     Otherwise: per-shot CPU loop with map_displacements/map_stresses.
    # ------------------------------------------------------------------
    try:
        # ---- Phase 1: CPU scalar extraction ----
        shot_scalars, energy_list, sR_profiles, plastic_ref = \
            _precompute_shot_scalars(
                n_actual, alpha, V_exit, D,
                mat_props, sigma_yield,
                gen_params.E_b, gen_params.nu_b,
            )

        # ---- Phase 2: spatial superposition ----
        if _TORCH_AVAILABLE:
            # Build element centroids (needed by stress kernel)
            elem_centroids = _compute_elem_centroids(mesh)

            # Resolve device and safe chunk size
            device, chunk_nodes = _resolve_device_and_chunk(n_actual, N_nodes)

            node_xy = mesh["node_coords"][:, :2].astype(np.float32)
            elem_xy = elem_centroids[:, :2].astype(np.float32)

            disp_f32, stress_f32 = _superpose_shots_batched(
                centres.astype(np.float32),
                shot_scalars,
                node_xy,
                elem_xy,
                device=device,
                chunk_nodes=chunk_nodes,
            )

        else:
            # CPU fallback — original per-shot loop using map_displacements /
            # map_stresses (identical physics, just sequential).
            disp_total   = np.zeros((N_nodes, 3), dtype=np.float64)
            stress_total = np.zeros((N_elems, 4), dtype=np.float64)

            for i in range(n_actual):
                phi_i = math.pi / 2.0 - float(alpha[i])
                phi_i = max(phi_i, math.radians(5.0))

                p_i = ShotPeenParams(
                    E_s=mat_props["E_s"],
                    nu_s=mat_props["nu_s"],
                    D=D,
                    rho_s=mat_props["rho_s"],
                    E_b=gen_params.E_b,
                    nu_b=gen_params.nu_b,
                    sigma_yield=sigma_yield,
                    V=float(V_exit[i]),
                    phi=phi_i,
                    n_depth=500,
                )
                contact_i = compute_contact_params(p_i)
                # Reuse pre-computed plastic zone and sR profile from Phase 1
                plastic_i = {
                    "a_p":        shot_scalars["a_p"][i],
                    "r_p":        shot_scalars["r_p"][i],
                    "epsilon_Mp": 0.0, "V_p": 0.0, "W_t": 0.0,
                }
                sf_i = {
                    "Z":  sR_profiles[i][:, 0].astype(np.float64),
                    "sR": sR_profiles[i][:, 1].astype(np.float64),
                }
                ic = np.array([centres[i, 0], centres[i, 1], 0.0])

                _, disp_i   = map_displacements(mesh, contact_i, plastic_i, p_i, ic)
                _, stress_i = map_stresses(mesh, sf_i, plastic_i, p_i, ic)

                disp_total   += disp_i.astype(np.float64)
                stress_total += stress_i.astype(np.float64)

            disp_f32   = disp_total.astype(np.float32)
            stress_f32 = stress_total.astype(np.float32)

        # Mean residual stress depth profile across all shots
        min_len  = min(p.shape[0] for p in sR_profiles) if sR_profiles else 1
        sR_stack = np.stack([p[:min_len, :] for p in sR_profiles], axis=0)
        sR_mean  = sR_stack.mean(axis=0)

        # Coverage (Avrami equation — mesh-resolution independent)
        coverage_info = compute_coverage(
            disp_f32, plastic_ref, mesh["node_coords"],
            gen_params.Lx, gen_params.Ly,
            centres=centres,
        )
        almen_MPa = float(np.min(stress_f32[:, 0]) / 1e6)

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {"sim_index": sim_index, "output_dir": out_folder,
                "elapsed_s": elapsed, "success": False,
                "error": f"Simulation failed: {exc}"}

    # ------------------------------------------------------------------
    # Build Gaussian-derived checkerboard (primary ML input)
    # ------------------------------------------------------------------
    checkerboard = gaussian_density_checkerboard(
        centres, V_normal,
        gen_params.Lx, gen_params.Ly,
        gen_params.checkerboard_size,
        mode="energy",          # V_n²-weighted density per cell
    )

    # Also save a raw count checkerboard for comparison
    checkerboard_count = gaussian_density_checkerboard(
        centres, V_normal,
        gen_params.Lx, gen_params.Ly,
        gen_params.checkerboard_size,
        mode="count",
    )

    # ------------------------------------------------------------------
    # Save .npy files (same schema as Abaqus / native_dataset_gen.py)
    # ------------------------------------------------------------------
    try:
        os.makedirs(out_folder, exist_ok=True)

        np.save(os.path.join(out_folder, "node_labels.npy"),
                mesh["node_labels"])
        np.save(os.path.join(out_folder, "node_coords.npy"),
                mesh["node_coords"])
        np.save(os.path.join(out_folder, "element_labels.npy"),
                mesh["element_labels"])
        np.save(os.path.join(out_folder, "element_connectivity.npy"),
                mesh["element_connectivity"])
        np.save(os.path.join(out_folder, "disp_node_labels.npy"),
                mesh["node_labels"])
        np.save(os.path.join(out_folder, "displacements.npy"),
                disp_f32)
        np.save(os.path.join(out_folder, "stress_element_labels.npy"),
                mesh["element_labels"])
        np.save(os.path.join(out_folder, "stresses.npy"),
                stress_f32)
        np.save(os.path.join(out_folder, "sR_depth_profile.npy"),
                sR_mean)
        np.save(os.path.join(out_folder, "checkerboard.npy"),
                checkerboard)
        np.save(os.path.join(out_folder, "checkerboard_count.npy"),
                checkerboard_count)
        np.save(os.path.join(out_folder, "shot_positions.npy"),
                centres)
        np.save(os.path.join(out_folder, "shot_V_normal.npy"),
                V_normal)
        np.save(os.path.join(out_folder, "shot_V_exit.npy"),
                V_exit)
        np.save(os.path.join(out_folder, "shot_angles.npy"),
                alpha)

        # Human-readable traceability files
        _write_nozzle_metadata(
            out_folder, sim_index, gen_params,
            h_nozzle, theta_div, nozzle_x, nozzle_y,
            V_mean, sigma_V, n_shots, n_actual, D,
            mat_name, sigma_yield, coverage_info, almen_MPa,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {"sim_index": sim_index, "output_dir": out_folder,
                "elapsed_s": elapsed, "success": False,
                "error": f"Save failed: {exc}"}

    elapsed = time.perf_counter() - t0
    sigma_xy_mm = h_nozzle * 1e3 * math.tan(theta_div)   # Gaussian spread in mm

    return {
        "sim_index":       sim_index,
        "output_dir":      out_folder,
        # Mesh
        "n_nodes":         N_nodes,
        "n_elems":         N_elems,
        # Nozzle
        "h_nozzle_mm":     h_nozzle * 1e3,
        "theta_div_deg":   math.degrees(theta_div),
        "sigma_xy_mm":     sigma_xy_mm,
        "nozzle_x_mm":     nozzle_x * 1e3,
        "nozzle_y_mm":     nozzle_y * 1e3,
        # Velocity
        "V_mean_mps":      V_mean,
        "sigma_V_mps":     sigma_V,
        "V_normal_mean_mps": float(np.mean(V_normal)),
        "V_normal_min_mps":  float(np.min(V_normal)),
        "alpha_mean_deg":    float(np.degrees(np.mean(alpha))),
        # Shots
        "n_shots_target":  n_shots,
        "n_shots_actual":  n_actual,
        "D_mm":            D * 1e3,
        "material":        mat_name,
        # Workpiece
        "sigma_yield_MPa": sigma_yield / 1e6,
        # Results
        "coverage_percent": coverage_info["coverage_percent"],
        "almen_MPa":       almen_MPa,
        "elapsed_s":       elapsed,
        "success":         True,
        "error":           None,
    }


def _write_nozzle_metadata(
    out_folder: str,
    sim_index: int,
    gp: GaussianNozzleParams,
    h_nozzle: float,
    theta_div: float,
    nozzle_x: float,
    nozzle_y: float,
    V_mean: float,
    sigma_V: float,
    n_shots_target: int,
    n_shots_actual: int,
    D: float,
    mat_name: str,
    sigma_yield: float,
    coverage_info: Dict,
    almen_MPa: float,
) -> None:
    """Write nozzle_params.txt (traceability) into out_folder."""
    sigma_xy_mm = h_nozzle * 1e3 * math.tan(theta_div)
    path = os.path.join(out_folder, "nozzle_params.txt")
    with open(path, "w") as fh:
        fh.write(f"sim_index:          {sim_index}\n")
        fh.write(f"generator:          gaussian_nozzle_dataset_gen.py\n")
        fh.write(f"--- Nozzle ---\n")
        fh.write(f"h_nozzle_mm:        {h_nozzle*1e3:.1f}\n")
        fh.write(f"theta_div_deg:      {math.degrees(theta_div):.2f}\n")
        fh.write(f"sigma_xy_mm:        {sigma_xy_mm:.2f}\n")
        fh.write(f"nozzle_x_mm:        {nozzle_x*1e3:.2f}\n")
        fh.write(f"nozzle_y_mm:        {nozzle_y*1e3:.2f}\n")
        fh.write(f"--- Velocity ---\n")
        fh.write(f"V_mean_mps:         {V_mean:.2f}\n")
        fh.write(f"sigma_V_mps:        {sigma_V:.3f}\n")
        fh.write(f"CV_percent:         {sigma_V/V_mean*100:.1f}\n")
        fh.write(f"--- Shots ---\n")
        fh.write(f"n_shots_target:     {n_shots_target}\n")
        fh.write(f"n_shots_actual:     {n_shots_actual}\n")
        fh.write(f"D_mm:               {D*1e3:.4f}\n")
        fh.write(f"shot_material:      {mat_name}\n")
        mat = SHOT_MATERIALS[mat_name]
        fh.write(f"rho_shot_kg_m3:     {mat['rho_s']:.0f}\n")
        fh.write(f"E_shot_GPa:         {mat['E_s']/1e9:.0f}\n")
        fh.write(f"--- Workpiece ---\n")
        fh.write(f"sigma_yield_MPa:    {sigma_yield/1e6:.1f}\n")
        fh.write(f"E_b_GPa:            {gp.E_b/1e9:.1f}\n")
        fh.write(f"nu_b:               {gp.nu_b:.3f}\n")
        fh.write(f"--- Mesh ---\n")
        fh.write(f"Lx_mm:              {gp.Lx*1e3:.1f}\n")
        fh.write(f"Ly_mm:              {gp.Ly*1e3:.1f}\n")
        fh.write(f"Nx:                 {gp.Nx}\n")
        fh.write(f"Ny:                 {gp.Ny}\n")
        fh.write(f"checkerboard_size:  {gp.checkerboard_size}\n")
        fh.write(f"--- Results ---\n")
        fh.write(f"coverage_pct:       {coverage_info['coverage_percent']:.2f}\n")
        fh.write(f"almen_MPa:          {almen_MPa:.2f}\n")


# ---------------------------------------------------------------------------
# 6.  Main dataset generator
# ---------------------------------------------------------------------------

def generate_gaussian_dataset(
    gen_params: Optional[GaussianNozzleParams] = None,
    verbose: bool = True,
) -> List[Dict]:
    """Generate a full ML training dataset using the Gaussian nozzle model.

    Parameters
    ----------
    gen_params : GaussianNozzleParams (defaults created if None).
    verbose    : Print per-simulation progress to stdout.

    Returns
    -------
    List of result dicts from ``run_gaussian_nozzle_simulation()``.
    Also writes a ``dataset_summary.csv`` to the output directory.
    """
    if gen_params is None:
        gen_params = GaussianNozzleParams()

    os.makedirs(gen_params.output_dir, exist_ok=True)

    indices = list(range(
        gen_params.start_index,
        gen_params.start_index + gen_params.n_simulations,
    ))
    n_total = len(indices)

    _log = print if verbose else (lambda *a, **k: None)
    n_nodes_expected = (gen_params.Nx + 1) * (gen_params.Ny + 1)

    # Report compute backend
    if _TORCH_AVAILABLE:
        _dev_str = ("CUDA (" + torch.cuda.get_device_name(0) + ")"
                    if torch.cuda.is_available() else "CPU (torch)")
    else:
        _dev_str = "CPU (numpy fallback -- install torch for GPU)"

    _log("=" * 66)
    _log("Gaussian Nozzle Dataset Generator -- peen-ml")
    _log("=" * 66)
    _log(f"  Output        : {gen_params.output_dir}")
    _log(f"  Cases         : {n_total}  (indices {indices[0]}-{indices[-1]})")
    _log(f"  Plate         : {gen_params.Lx*1e3:.0f} mm x {gen_params.Ly*1e3:.0f} mm")
    _log(f"  Mesh          : {gen_params.Nx} x {gen_params.Ny}  "
         f"-> {n_nodes_expected} nodes")
    _log(f"  Compute       : {_dev_str}")
    _log(f"  Workers       : {gen_params.workers}")
    _log(f"  Checkerboard  : {gen_params.checkerboard_size} x {gen_params.checkerboard_size}")
    _log(f"  Nozzle h      : {gen_params.h_range[0]*1e3:.0f}-"
         f"{gen_params.h_range[1]*1e3:.0f} mm")
    _log(f"  Jet div angle : {math.degrees(gen_params.theta_div_range[0]):.0f}deg-"
         f"{math.degrees(gen_params.theta_div_range[1]):.0f}deg")
    _log(f"  V_mean        : {gen_params.V_mean_range[0]:.0f}-"
         f"{gen_params.V_mean_range[1]:.0f} m/s")
    _log(f"  Shot mats     : {', '.join(gen_params.shot_materials)}")
    _log("=" * 66)


    t_start = time.perf_counter()
    results: List[Dict] = []
    n_ok = n_fail = 0

    if gen_params.workers <= 1:
        # ---------- Sequential ----------
        for idx, sim_idx in enumerate(indices):
            res = run_gaussian_nozzle_simulation(sim_idx, gen_params)
            results.append(res)
            if res["success"]:
                n_ok += 1
                if verbose:
                    print(
                        f"  [{idx+1:4d}/{n_total}] Sim_{sim_idx:04d}  "
                        f"h={res['h_nozzle_mm']:.0f}mm  "
                        f"sxy={res['sigma_xy_mm']:.1f}mm  "
                        f"V_n={res['V_normal_mean_mps']:.1f}m/s  "
                        f"alpha={res['alpha_mean_deg']:.1f}deg  "
                        f"cov={res['coverage_percent']:.0f}%  "
                        f"almen={res['almen_MPa']:.0f}MPa  "
                        f"({res['elapsed_s']:.1f}s)"
                    )
            else:
                n_fail += 1
                print(f"  [{idx+1:4d}/{n_total}] Sim_{sim_idx:04d}  FAILED: {res['error']}")

    else:
        # ---------- Parallel ----------
        with ProcessPoolExecutor(max_workers=gen_params.workers) as pool:
            futures = {
                pool.submit(run_gaussian_nozzle_simulation, si, gen_params): si
                for si in indices
            }
            done = 0
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                done += 1
                if res["success"]:
                    n_ok += 1
                    if verbose:
                        print(
                            f"  [{done:4d}/{n_total}] Sim_{res['sim_index']:04d}  "
                            f"h={res['h_nozzle_mm']:.0f}mm  "
                            f"cov={res['coverage_percent']:.0f}%  "
                            f"({res['elapsed_s']:.1f}s)"
                        )
                else:
                    n_fail += 1
                    print(f"  [{done:4d}/{n_total}] Sim_{res['sim_index']:04d}  "
                          f"FAILED: {res['error']}")

    elapsed = time.perf_counter() - t_start
    _write_summary_csv(
        os.path.join(gen_params.output_dir, "dataset_summary.csv"),
        results,
    )

    _log("=" * 66)
    _log(f"Done.  {n_ok}/{n_total} OK,  {n_fail} failed  "
         f"({elapsed:.0f} s total,  {elapsed/n_total:.1f} s/sim)")
    _log(f"Summary: {gen_params.output_dir}/dataset_summary.csv")
    _log("=" * 66)

    n_nodes = (gen_params.Nx + 1) * (gen_params.Ny + 1)
    _log(f"\nCNN notes:")
    _log(f"  num_nodes        = {n_nodes}")
    _log(f"  checkerboard_size= {gen_params.checkerboard_size}")
    _log(f"  FC input dim     = 128 x {gen_params.checkerboard_size} x "
         f"{gen_params.checkerboard_size} = "
         f"{128 * gen_params.checkerboard_size ** 2}")
    _log(f"  Pass these to create_model() in model.py when retraining.\n")

    return results


def _write_summary_csv(path: str, results: List[Dict]) -> None:
    """Write a CSV summary of all simulation results."""
    import csv
    fieldnames = [
        "sim_index", "success",
        "n_nodes", "n_elems",
        "h_nozzle_mm", "theta_div_deg", "sigma_xy_mm",
        "nozzle_x_mm", "nozzle_y_mm",
        "V_mean_mps", "sigma_V_mps",
        "V_normal_mean_mps", "V_normal_min_mps", "alpha_mean_deg",
        "n_shots_target", "n_shots_actual",
        "D_mm", "material", "sigma_yield_MPa",
        "coverage_percent", "almen_MPa",
        "elapsed_s", "error",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda x: x.get("sim_index", 0)):
            writer.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# 7.  Validation
# ---------------------------------------------------------------------------

def validate_gaussian_dataset(dataset_dir: str, n_check: int = 5) -> None:
    """Quick sanity-check on a generated Gaussian nozzle dataset.

    Loads ``n_check`` random simulations and verifies:
    - File existence (all required .npy files present)
    - Array shapes (nodes × 3 for displacements, G × G for checkerboard)
    - Physics plausibility (non-zero uz, V_n > 0, checkerboard is Gaussian-shaped)
    - Gaussian shape of the shot position distribution (optional visual)

    Parameters
    ----------
    dataset_dir : Root directory of the generated dataset.
    n_check     : Number of simulations to inspect.
    """
    sims = sorted([
        d for d in os.listdir(dataset_dir)
        if d.startswith("Simulation_") and
        os.path.isdir(os.path.join(dataset_dir, d))
    ], key=lambda x: int(x.split("_")[1]))

    if not sims:
        print(f"No Simulation_N/ folders found in: {dataset_dir}")
        return

    rng = np.random.default_rng(42)
    sample = list(rng.choice(sims, size=min(n_check, len(sims)), replace=False))

    print(f"\nGaussian Nozzle Dataset Validation: {dataset_dir}")
    print(f"  Total simulations found: {len(sims)}")
    print(f"  Checking {len(sample)} samples ...\n")

    required_files = [
        "checkerboard.npy", "displacements.npy",
        "node_coords.npy", "node_labels.npy",
        "element_connectivity.npy", "element_labels.npy",
        "disp_node_labels.npy",
        "shot_positions.npy", "shot_V_normal.npy", "shot_angles.npy",
    ]

    all_ok = True
    for sim_name in sample:
        sim_dir = os.path.join(dataset_dir, sim_name)
        errors: List[str] = []

        # --- File existence ---
        for fname in required_files:
            if not os.path.exists(os.path.join(sim_dir, fname)):
                errors.append(f"Missing: {fname}")

        if errors:
            all_ok = False
            print(f"  {sim_name}: FAIL")
            for e in errors:
                print(f"    -> {e}")
            continue

        # --- Load arrays ---
        cb      = np.load(os.path.join(sim_dir, "checkerboard.npy"))
        disp    = np.load(os.path.join(sim_dir, "displacements.npy"))
        centres = np.load(os.path.join(sim_dir, "shot_positions.npy"))
        V_n     = np.load(os.path.join(sim_dir, "shot_V_normal.npy"))
        alpha   = np.load(os.path.join(sim_dir, "shot_angles.npy"))

        # --- Shape checks ---
        if cb.ndim != 2:
            errors.append(f"checkerboard ndim={cb.ndim}, expected 2")
        if disp.ndim != 2 or disp.shape[1] != 3:
            errors.append(f"displacements shape={disp.shape}, expected (N,3)")
        if centres.ndim != 2 or centres.shape[1] != 2:
            errors.append(f"shot_positions shape={centres.shape}, expected (N,2)")

        # --- Physics plausibility ---
        if np.all(disp[:, 2] == 0):
            errors.append("All uz=0 -- suspect (no deformation computed)")
        if np.any(V_n <= 0):
            errors.append(f"{np.sum(V_n<=0)} shots with V_n <= 0")
        if np.any(alpha < 0) or np.any(alpha > math.pi / 2):
            errors.append("Impact angles outside [0, pi/2]")
        if cb.max() > 1.001 or cb.min() < -0.001:
            errors.append(f"Checkerboard not normalised: [{cb.min():.3f}, {cb.max():.3f}]")

        # --- Gaussian shape check ---
        # For a well-centred nozzle, the checkerboard should be brighter in the
        # centre.  Check that the centre 3×3 cells sum > corner 3×3 cells.
        G = cb.shape[0]
        if G >= 6:
            c = G // 2
            r = max(1, G // 6)
            centre_energy = cb[c-r:c+r+1, c-r:c+r+1].mean()
            corner_energy = (
                cb[:r, :r].mean() + cb[:r, -r:].mean() +
                cb[-r:, :r].mean() + cb[-r:, -r:].mean()
            ) / 4.0
            if centre_energy < corner_energy and len(centres) > 20:
                # This can legitimately happen when the nozzle is off-centre
                errors.append(
                    f"Checkerboard peak not at centre: "
                    f"centre={centre_energy:.3f} < corner={corner_energy:.3f} "
                    f"(may be OK if nozzle is off-centre)"
                )

        # Report
        status = "OK" if not errors else "WARN"
        if errors:
            all_ok = False
        print(f"  {sim_name}: {status}  "
              f"(N_shots={len(centres)}, "
              f"V_n_mean={V_n.mean():.1f} m/s, "
              f"alpha_mean={math.degrees(alpha.mean()):.1f} deg, "
              f"CB shape={cb.shape})")
        for e in errors:
            print(f"    -> {e}")

    print(f"\n{'All checks passed.' if all_ok else 'Some checks produced warnings.'}\n")


# ---------------------------------------------------------------------------
# 8.  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Gaussian nozzle shot peening dataset generator.\n"
            "Shots follow a 2-D Gaussian distribution centred below the nozzle;\n"
            "exit velocities follow a Gaussian; impact angle corrects V to V_n."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Output
    parser.add_argument("--output",   default="./Dataset_Gaussian",
                        help="Root output directory (default: ./Dataset_Gaussian)")
    parser.add_argument("--n_sims",   type=int, default=100,
                        help="Number of simulation cases (default: 100)")
    parser.add_argument("--start",    type=int, default=0,
                        help="Starting simulation index (default: 0)")
    parser.add_argument("--workers",  type=int, default=1,
                        help="Parallel worker processes (default: 1 = sequential)")
    parser.add_argument("--seed",     type=int, default=0,
                        help="Master RNG seed (default: 0)")

    # Mesh
    parser.add_argument("--Lx",  type=float, default=0.040,
                        help="Plate X-dimension in metres (default: 0.040 = 40 mm)")
    parser.add_argument("--Ly",  type=float, default=0.040,
                        help="Plate Y-dimension in metres (default: 0.040 = 40 mm)")
    parser.add_argument("--Nx",  type=int,   default=100,
                        help="Quad elements in X (default: 100 → 101 nodes)")
    parser.add_argument("--Ny",  type=int,   default=100,
                        help="Quad elements in Y (default: 100 → 101 nodes)")

    # Nozzle geometry
    parser.add_argument("--h_min",        type=float, default=0.050,
                        help="Min nozzle standoff height (m, default 0.050)")
    parser.add_argument("--h_max",        type=float, default=0.400,
                        help="Max nozzle standoff height (m, default 0.400)")
    parser.add_argument("--div_min_deg",  type=float, default=5.0,
                        help="Min jet divergence half-angle (deg, default 5)")
    parser.add_argument("--div_max_deg",  type=float, default=30.0,
                        help="Max jet divergence half-angle (deg, default 30)")

    # Velocity
    parser.add_argument("--V_min",  type=float, default=25.0,
                        help="Min mean exit velocity (m/s, default 25)")
    parser.add_argument("--V_max",  type=float, default=80.0,
                        help="Max mean exit velocity (m/s, default 80)")

    # Shots
    parser.add_argument("--n_shots_min", type=int,   default=30)
    parser.add_argument("--n_shots_max", type=int,   default=200)
    parser.add_argument("--D_min",       type=float, default=0.0003,
                        help="Min shot diameter (m, default 0.0003 = 0.3 mm)")
    parser.add_argument("--D_max",       type=float, default=0.0010,
                        help="Max shot diameter (m, default 0.0010 = 1.0 mm)")
    parser.add_argument("--materials",   nargs="+",
                        default=["steel", "ceramic", "glass", "cast_iron"],
                        choices=list(SHOT_MATERIALS.keys()),
                        help="Shot materials to randomise over")

    # Checkerboard
    parser.add_argument("--grid_size",  type=int, default=20,
                        help="Checkerboard grid resolution G (default: 20 → 20×20)")

    # Actions
    parser.add_argument("--validate", action="store_true",
                        help="Run validation checks after generation")

    args = parser.parse_args()

    gp = GaussianNozzleParams(
        output_dir=args.output,
        n_simulations=args.n_sims,
        start_index=args.start,
        workers=args.workers,
        base_seed=args.seed,
        Lx=args.Lx,
        Ly=args.Ly,
        Nx=args.Nx,
        Ny=args.Ny,
        h_range=(args.h_min, args.h_max),
        theta_div_range=(math.radians(args.div_min_deg),
                         math.radians(args.div_max_deg)),
        V_mean_range=(args.V_min, args.V_max),
        n_shots_range=(args.n_shots_min, args.n_shots_max),
        D_range=(args.D_min, args.D_max),
        shot_materials=args.materials,
        checkerboard_size=args.grid_size,
    )

    # Print computational cost estimate
    n_nodes = (gp.Nx + 1) * (gp.Ny + 1)
    n_shots_mid = sum(gp.n_shots_range) // 2
    est_s = 0.0005e-3 * n_nodes * n_shots_mid  # rough: 0.5 us per node x shot
    print(f"\nEstimated time per simulation: {est_s:.1f} s "
          f"({n_nodes} nodes x {n_shots_mid} shots x 0.5 us/node/shot)")
    print(f"Estimated total time: {gp.n_simulations * est_s / max(1, gp.workers):.0f} s "
          f"with {gp.workers} worker(s)\n")

    generate_gaussian_dataset(gp, verbose=True)

    if args.validate:
        validate_gaussian_dataset(args.output, n_check=10)
