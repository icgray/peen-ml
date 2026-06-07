"""
multi_shot_sim.py
=================
Multi-shot shot peening simulation built on the Shen & Atluri (2006)
analytical single-shot model (impact_sim.py).

Simulates realistic shot peening with N shots distributed across the target
surface and computes:
  - Superposed surface displacement field (total dent topology)
  - Superposed residual stress field (biaxial, depth-dependent)
  - Shot coverage map (fraction of area plastically deformed)
  - Almen intensity proxy (peak compressive residual stress, MPa)
  - Checkerboard representation of the shot pattern (ML model input)

Shot Distribution Modes
-----------------------
``random``
    Uniform random impact centres (Monte Carlo shot peening).

``grid``
    Regular grid with optional random jitter — deterministic, uniform coverage.

``poisson``
    Poisson-disk sampling: each shot is at least ``min_sep`` away from all
    previous shots (more realistic than uniform random).

``checkerboard``
    Derive shot positions from a user-supplied 2-D intensity array:
    each cell generates a number of shots proportional to its value.

Physics notes
-------------
Superposition of residual stresses uses the additive principle (valid while
plastic zones don't extensively overlap, i.e. Almen coverage < ~98%).
Stresses and displacements are accumulated per node/element by summing the
contributions from every individual impact.  Mutual interaction between
nearby plastic zones is not modelled — this is the same assumption used in
most analytical multi-shot models in the literature.

References
----------
Shen & Atluri (2006) CMC vol. 4 no. 2 pp. 75-85.
Rouquette et al. (2009) J. Mater. Process. Technol. 209, 3048–3055.
Miao et al. (2010) Surf. Coat. Technol. 205, 78–86.

Usage
-----
>>> from multi_shot_sim import MultiShotParams, run_multi_shot_simulation
>>> params = MultiShotParams(n_shots=60, Lx=0.010, Ly=0.010)
>>> results = run_multi_shot_simulation(params, output_dir="./multi_output")
>>> print(f"Coverage: {results['coverage_fraction']*100:.1f}%")
>>> print(f"Almen intensity (proxy): {results['almen_intensity_MPa']:.1f} MPa")

CLI
---
    python multi_shot_sim.py --n_shots 80 --V 40 --D 0.0006 --plot
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import the single-shot analytical model
# ---------------------------------------------------------------------------
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from impact_sim import (  # noqa: E402
    ShotPeenParams,
    generate_mesh,
    compute_contact_params,
    compute_stress_field,
    compute_plastic_zone,
    compute_energy_balance,
    map_displacements,
    map_stresses,
)

__all__ = [
    "MultiShotParams",
    "generate_shot_positions",
    "run_multi_shot_simulation",
    "compute_coverage",
    "checkerboard_to_shots",
    "displacements_to_checkerboard",
    "compute_physics_checkerboard",
    "compute_influence_fields",
    "compute_cupping_from_profile",
    "element_to_nodal_stress",
]


# ---------------------------------------------------------------------------
# 1.  Parameter container
# ---------------------------------------------------------------------------


@dataclass
class MultiShotParams:
    """All parameters for a multi-shot simulation campaign.

    Material / shot defaults
    ------------------------
    ``base_params`` : ShotPeenParams instance used for every shot.
        Override individual fields (V, D, sigma_yield, …) as needed.

    Shot pattern
    ------------
    n_shots          : Total number of shots across the surface.
    distribution     : 'random', 'grid', 'poisson', or 'checkerboard'.
    V_scatter        : Std-dev of impact velocity (m/s). 0 = monodisperse.
    angle_scatter    : Std-dev of impact angle from normal (rad). 0 = normal.
    D_scatter        : Std-dev of shot diameter (m). 0 = monodisperse.

    Surface geometry
    ----------------
    Lx, Ly : Target plate dimensions (m).
    Nx, Ny : Mesh resolution (number of quad elements per axis).

    Coverage / intensity
    --------------------
    min_sep_factor : Poisson-disk minimum shot separation as a multiple of
                     the shot radius (default 1.5 — shots don't overlap much).
    """

    # Single-shot parameters (shared baseline)
    base_params: ShotPeenParams = field(default_factory=ShotPeenParams)

    # Multi-shot pattern
    n_shots: int = 50
    distribution: str = "random"  # 'random' | 'grid' | 'poisson' | 'checkerboard'
    seed: Optional[int] = 42  # RNG seed for reproducibility (None = random)

    # Shot scatter (realistic process variation)
    V_scatter: float = 2.0  # m/s
    angle_scatter: float = 0.05  # rad
    D_scatter: float = 0.0  # m (0 = monodisperse)

    # Plate geometry
    Lx: float = 0.010  # m
    Ly: float = 0.010  # m

    # Mesh resolution
    Nx: int = 50
    Ny: int = 50

    # Poisson-disk minimum separation (multiple of shot radius)
    min_sep_factor: float = 1.5

    # Checkerboard grid (used when distribution == 'checkerboard')
    checkerboard_grid: int = 5  # NxN cells
    checkerboard_intensity_min: float = 0.005  # min shots-per-cell intensity
    checkerboard_intensity_max: float = 0.020  # max shots-per-cell intensity


# ---------------------------------------------------------------------------
# 2.  Shot position generators
# ---------------------------------------------------------------------------


def generate_shot_positions(
    params: MultiShotParams,
    checkerboard: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (x, y) impact centres and per-shot parameters.

    Parameters
    ----------
    params       : MultiShotParams
    checkerboard : (G, G) float array of cell intensities (only used when
                   ``params.distribution == 'checkerboard'``).
    rng          : numpy random generator (created from params.seed if None).

    Returns
    -------
    centres : (N, 2) array of impact (x, y) coordinates in [0, Lx] × [0, Ly]
    V_vec   : (N,)  per-shot impact velocity (m/s)
    D_vec   : (N,)  per-shot shot diameter (m)
    """
    if rng is None:
        rng = np.random.default_rng(params.seed)

    Lx, Ly = params.Lx, params.Ly
    bp = params.base_params
    n = params.n_shots

    dist = params.distribution.lower()

    # ----- Generate (x, y) -----
    if dist == "random":
        centres = np.stack(
            [
                rng.uniform(0, Lx, n),
                rng.uniform(0, Ly, n),
            ],
            axis=1,
        )

    elif dist == "grid":
        cols = max(1, int(math.sqrt(n * Lx / Ly)))
        rows = max(1, int(math.ceil(n / cols)))
        xs = np.linspace(Lx / (2 * cols), Lx - Lx / (2 * cols), cols)
        ys = np.linspace(Ly / (2 * rows), Ly - Ly / (2 * rows), rows)
        grid_pts = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
        # Random subset if grid has more points than requested
        if len(grid_pts) > n:
            idx = rng.choice(len(grid_pts), size=n, replace=False)
            grid_pts = grid_pts[idx]
        elif len(grid_pts) < n:
            extra = rng.uniform([0, 0], [Lx, Ly], (n - len(grid_pts), 2))
            grid_pts = np.concatenate([grid_pts, extra], axis=0)
        centres = grid_pts
        # Add small jitter (± half cell size)
        jitter_x = Lx / (2 * cols) * 0.5
        jitter_y = Ly / (2 * rows) * 0.5
        centres = centres + rng.uniform([-jitter_x, -jitter_y], [jitter_x, jitter_y], centres.shape)
        centres = np.clip(centres, [0, 0], [Lx, Ly])

    elif dist == "poisson":
        centres = _poisson_disk(Lx, Ly, bp.R * params.min_sep_factor, n, rng)

    elif dist == "checkerboard":
        if checkerboard is None:
            raise ValueError("checkerboard array must be supplied for distribution='checkerboard'.")
        centres = checkerboard_to_shots(checkerboard, Lx, Ly, n, rng)

    else:
        raise ValueError(
            f"Unknown distribution '{params.distribution}'. " "Choose 'random', 'grid', 'poisson', or 'checkerboard'."
        )

    n_actual = len(centres)

    # ----- Per-shot velocity -----
    V_base = bp.V
    if params.V_scatter > 0:
        V_vec = np.abs(rng.normal(V_base, params.V_scatter, n_actual))
    else:
        V_vec = np.full(n_actual, V_base)

    # ----- Per-shot diameter -----
    D_base = bp.D
    if params.D_scatter > 0:
        D_vec = np.abs(rng.normal(D_base, params.D_scatter, n_actual))
        D_vec = np.clip(D_vec, D_base * 0.5, D_base * 2.0)
    else:
        D_vec = np.full(n_actual, D_base)

    return centres, V_vec, D_vec


def _poisson_disk(
    Lx: float,
    Ly: float,
    min_dist: float,
    n_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bridson's algorithm for Poisson-disk sampling in a 2-D rectangle."""
    cell = min_dist / math.sqrt(2.0)
    nx_grid = max(1, int(math.ceil(Lx / cell)))
    ny_grid = max(1, int(math.ceil(Ly / cell)))
    grid: Dict[Tuple[int, int], np.ndarray] = {}

    def grid_coords(pt):
        return (int(pt[0] / cell), int(pt[1] / cell))

    def too_close(pt, existing):
        gx, gy = grid_coords(pt)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nb = (gx + dx, gy + dy)
                if nb in existing:
                    if np.hypot(*(pt - existing[nb])) < min_dist:
                        return True
        return False

    samples: List[np.ndarray] = []
    active: List[np.ndarray] = []
    k = 30  # candidates per active point

    first = rng.uniform([0, 0], [Lx, Ly])
    samples.append(first)
    active.append(first)
    grid[grid_coords(first)] = first

    while active and len(samples) < n_target:
        idx = rng.integers(len(active))
        pt = active[idx]
        found = False
        for _ in range(k):
            angle = rng.uniform(0, 2 * math.pi)
            radius = rng.uniform(min_dist, 2.0 * min_dist)
            candidate = pt + np.array([radius * math.cos(angle), radius * math.sin(angle)])
            if 0 <= candidate[0] <= Lx and 0 <= candidate[1] <= Ly and not too_close(candidate, grid):
                samples.append(candidate)
                active.append(candidate)
                grid[grid_coords(candidate)] = candidate
                found = True
                break
        if not found:
            active.pop(idx)

    # If we couldn't fill n_target via Poisson-disk, pad with uniform random
    while len(samples) < n_target:
        samples.append(rng.uniform([0, 0], [Lx, Ly]))

    return np.array(samples[:n_target])


# ---------------------------------------------------------------------------
# 3.  Checkerboard ↔ shot-positions conversion
# ---------------------------------------------------------------------------


def checkerboard_to_shots(
    checkerboard: np.ndarray,
    Lx: float,
    Ly: float,
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Convert a 2-D intensity grid to shot impact positions.

    Each cell (i, j) contributes shots in proportion to its value.
    Shots within a cell are distributed uniformly at random.

    Parameters
    ----------
    checkerboard : (G, G) float array, values in (0, 1] or any positive range.
    Lx, Ly       : Plate dimensions (m).
    n_total      : Total number of shots to distribute.
    rng          : numpy random generator.

    Returns
    -------
    centres : (n_total, 2) impact positions.
    """
    cb = np.asarray(checkerboard, dtype=float)
    if cb.ndim == 1:
        sz = int(math.sqrt(len(cb)))
        cb = cb.reshape(sz, sz)

    G_rows, G_cols = cb.shape
    total_intensity = cb.sum()
    if total_intensity <= 0:
        # Uniform fallback
        return np.stack([rng.uniform(0, Lx, n_total), rng.uniform(0, Ly, n_total)], axis=1)

    # Number of shots per cell (proportional to intensity)
    probs = cb.ravel() / total_intensity
    n_per_cell = rng.multinomial(n_total, probs)  # shape (G*G,)

    cell_w = Lx / G_cols
    cell_h = Ly / G_rows

    shots = []
    for idx, cnt in enumerate(n_per_cell):
        if cnt == 0:
            continue
        row = idx // G_cols
        col = idx % G_cols
        x_lo, x_hi = col * cell_w, (col + 1) * cell_w
        y_lo, y_hi = row * cell_h, (row + 1) * cell_h
        xs = rng.uniform(x_lo, x_hi, cnt)
        ys = rng.uniform(y_lo, y_hi, cnt)
        shots.append(np.stack([xs, ys], axis=1))

    return np.concatenate(shots, axis=0) if shots else np.zeros((0, 2))


def displacements_to_checkerboard(
    displacements: np.ndarray,
    node_coords: np.ndarray,
    Lx: float,
    Ly: float,
    grid_size: int = 5,
) -> np.ndarray:
    """Summarise nodal displacements into a coarse intensity checkerboard.

    The result is the mean |uz| deformation magnitude per grid cell,
    normalised to [0, 1].  This is the inverse operation of
    ``checkerboard_to_shots`` — useful for visualising the simulated outcome
    in the same format as the ML model input.

    Parameters
    ----------
    displacements : (N, 3) nodal displacements [ux, uy, uz].
    node_coords   : (N, 3) nodal coordinates.
    Lx, Ly        : Plate dimensions (m).
    grid_size     : Size of output checkerboard (default 5 → 5×5).

    Returns
    -------
    checkerboard : (grid_size, grid_size) float array in [0, 1].
    """
    G = grid_size
    cb = np.zeros((G, G))
    counts = np.zeros((G, G), dtype=int)

    cell_w = Lx / G
    cell_h = Ly / G

    uz = np.abs(displacements[:, 2])  # surface normal deformation

    for i, (coord, u) in enumerate(zip(node_coords, uz)):
        col = min(G - 1, int(coord[0] / cell_w))
        row = min(G - 1, int(coord[1] / cell_h))
        cb[row, col] += u
        counts[row, col] += 1

    mask = counts > 0
    cb[mask] /= counts[mask]

    # Normalise to [0, 1]
    cb_max = cb.max()
    if cb_max > 0:
        cb /= cb_max

    return cb


# ---------------------------------------------------------------------------
# 4.  Coverage estimation
# ---------------------------------------------------------------------------


def compute_coverage(
    displacements: np.ndarray,
    plastic: Dict,
    node_coords: np.ndarray,
    Lx: float,
    Ly: float,
    threshold_factor: float = 0.05,
    centres: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Estimate Almen coverage using both a node-count method and the
    analytical Avrami (Johnson-Mehl) equation.

    The Avrami estimate is mesh-resolution-independent and is the primary
    result when shot positions are available.  The node-count method is a
    secondary check (it under-counts for coarse meshes where r_p < node
    spacing).

    Parameters
    ----------
    displacements   : (N, 3) nodal displacements.
    plastic         : Dict from compute_plastic_zone() with keys 'a_p', 'r_p'.
    node_coords     : (N, 3) nodal coordinates.
    Lx, Ly          : Plate dimensions (m).
    threshold_factor: Fraction of a_p used as peened threshold.
    centres         : (M, 2) optional shot impact positions.  When supplied
                      the Avrami equation uses per-shot positions; otherwise
                      only the node-count method is returned.

    Returns
    -------
    dict with keys:
        coverage_fraction       : Avrami estimate (0–1) [primary]
        coverage_percent        : same in %
        coverage_fraction_nodes : node-count estimate (secondary)
        n_peened                : nodes above displacement threshold
        n_total                 : total surface nodes
        avrami_k                : mean single-shot coverage fraction
        threshold_m             : threshold displacement (m)
    """
    a_p = plastic.get("a_p", 1e-5)
    r_p = plastic.get("r_p", a_p * 6.0)
    threshold = threshold_factor * a_p
    A_surface = Lx * Ly

    # ---- Avrami (Johnson-Mehl) equation ----
    # C = 1 - exp(-sum_i(A_i / A_surface))
    # where A_i = pi * r_p_i^2 is the plastic zone area of shot i.
    if centres is not None and len(centres) > 0:
        A_single = math.pi * r_p**2
        k_single = A_single / A_surface  # single-shot coverage fraction
        n_shots = len(centres)
        avrami_C = 1.0 - math.exp(-k_single * n_shots)
    else:
        avrami_C = 0.0
        k_single = 0.0

    # ---- Node-count method (mesh-resolution dependent) ----
    uz = np.abs(displacements[:, 2])
    surface_mask = np.abs(node_coords[:, 2]) < 1e-9
    uz_surf = uz[surface_mask]
    n_total = len(uz_surf)
    n_peened = int(np.sum(uz_surf > threshold))
    node_C = n_peened / n_total if n_total > 0 else 0.0

    # Use Avrami as primary if centres were supplied, else fall back to nodes
    primary_C = avrami_C if centres is not None else node_C

    return {
        "coverage_fraction": primary_C,
        "coverage_percent": primary_C * 100.0,
        "coverage_fraction_nodes": node_C,
        "n_peened": n_peened,
        "n_total": n_total,
        "avrami_k": k_single,
        "threshold_m": threshold,
    }


# ---------------------------------------------------------------------------
# 5.  Main multi-shot runner
# ---------------------------------------------------------------------------


def run_multi_shot_simulation(
    params: Optional[MultiShotParams] = None,
    checkerboard: Optional[np.ndarray] = None,
    output_dir: str = "./multi_shot_output",
    save_npy: bool = True,
    verbose: bool = True,
    grid_size: int = 5,
) -> Dict:
    """Run a complete multi-shot simulation and return results.

    Orchestrates:
      1. Generate mesh.
      2. Draw shot positions (from params.distribution or checkerboard).
      3. For each shot, run the single-shot analytical model.
      4. Superpose displacements and stresses across all shots.
      5. Compute coverage and Almen intensity.
      6. Save .npy files (compatible with data_viz.py and model.py).

    Parameters
    ----------
    params       : MultiShotParams (defaults created if None).
    checkerboard : (G, G) intensity array driving shot distribution.
                   If supplied and params.distribution != 'checkerboard',
                   it is still saved as the ML model input.
    output_dir   : Directory to write .npy files.
    save_npy     : Write output files to disk.
    verbose      : Print progress.
    grid_size    : Checkerboard resolution for the auto-generated summary map.

    Returns
    -------
    dict with keys:
        mesh, centres, displacements, stresses, sR_depth_profile,
        energy (per-shot list), coverage, almen_intensity_MPa,
        checkerboard (the input pattern), output_checkerboard (summary map)
    """
    if params is None:
        params = MultiShotParams()

    _log = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(params.seed)
    bp = params.base_params

    _log("=" * 62)
    _log("Multi-Shot Peening Simulation")
    _log("=" * 62)

    # ---- 1. Mesh ----
    _log(f"[1/5] Building mesh ({params.Nx}x{params.Ny} elements) ...")
    mesh = generate_mesh(
        Lx=params.Lx,
        Ly=params.Ly,
        Lz=0.002,
        Nx=params.Nx,
        Ny=params.Ny,
        Nz=1,
    )
    N_nodes = len(mesh["node_labels"])
    N_elems = len(mesh["element_labels"])
    _log(f"      {N_nodes} nodes, {N_elems} elements")

    # ---- 2. Shot positions ----
    _log(f"[2/5] Generating {params.n_shots} shot positions " f"(mode='{params.distribution}') ...")
    centres, V_vec, D_vec = generate_shot_positions(params, checkerboard, rng)
    n_actual = len(centres)
    _log(f"      Actual shots placed: {n_actual}")
    _log(
        f"      V range: {V_vec.min():.1f}-{V_vec.max():.1f} m/s  "
        f"  D range: {D_vec.min()*1e3:.3f}-{D_vec.max()*1e3:.3f} mm"
    )

    # ---- 3. Single-shot precomputation (shared stress profile) ----
    # Compute one representative stress profile for depth superposition
    _log("[3/5] Computing representative stress depth profile ...")
    contact0 = compute_contact_params(bp)
    sf0 = compute_stress_field(contact0, bp)
    plastic0 = compute_plastic_zone(bp)

    # ---- 4. Superpose impacts ----
    _log(f"[4/5] Superposing {n_actual} impacts onto mesh ...")

    # Accumulate on nodes and elements
    disp_total = np.zeros((N_nodes, 3), dtype=np.float64)
    stress_total = np.zeros((N_elems, 4), dtype=np.float64)

    energy_list = []
    sR_profiles = []  # collect representative depth profiles
    plastic_ref = plastic0  # use representative plastic zone for coverage
    per_shot_physics = []  # collect per-shot physics for physics checkerboard

    for i, (centre, V_i, D_i) in enumerate(zip(centres, V_vec, D_vec)):
        # Build per-shot params (most fields shared, V and D can vary)
        p_i = ShotPeenParams(
            E_s=bp.E_s,
            nu_s=bp.nu_s,
            D=D_i,
            rho_s=bp.rho_s,
            E_b=bp.E_b,
            nu_b=bp.nu_b,
            sigma_yield=bp.sigma_yield,
            c=bp.c,
            V=V_i,
            phi=bp.phi,
            k=bp.k,
            n_depth=max(1000, bp.n_depth // 100),  # coarser for speed
        )

        contact_i = compute_contact_params(p_i)
        plastic_i = compute_plastic_zone(p_i)
        sf_i = compute_stress_field(contact_i, p_i)
        energy_i = compute_energy_balance(p_i, contact_i, plastic_i)

        ic = np.array([centre[0], centre[1], 0.0])

        _, disp_i = map_displacements(mesh, contact_i, plastic_i, p_i, ic)
        _, stress_i = map_stresses(mesh, sf_i, plastic_i, p_i, ic)

        disp_total += disp_i.astype(np.float64)
        stress_total += stress_i.astype(np.float64)

        energy_list.append(energy_i)
        sR_profile_i = np.stack([sf_i["Z"], sf_i["sR"]], axis=1)
        sR_profiles.append(sR_profile_i)

        # Collect physics for multi-channel checkerboard
        a_p_i = plastic_i["a_p"]
        r_p_i = plastic_i["r_p"]
        delta_p_i = a_p_i**2 / (2.0 * p_i.R) if p_i.R > 0 else 0.0
        sigma_R_surface_i = float(sf_i["sR"][0]) if len(sf_i["sR"]) > 0 else 0.0
        per_shot_physics.append(
            {
                "x": float(centre[0]),
                "y": float(centre[1]),
                "V_n": float(p_i.Vn),
                "D": float(D_i),
                "rho_s": float(bp.rho_s),
                "a_p": float(a_p_i),
                "r_p": float(r_p_i),
                "delta_p": float(delta_p_i),
                "sigma_R_surface": float(sigma_R_surface_i),
                "sR_depth": sR_profile_i,  # (K, 2): col0=depth, col1=sigma_R
            }
        )

        if verbose and (i + 1) % max(1, n_actual // 10) == 0:
            _log(f"      {i+1}/{n_actual} shots processed ...")

    disp_total_f32 = disp_total.astype(np.float32)
    stress_total_f32 = stress_total.astype(np.float32)

    # Mean depth profile across all shots (for analysis/plotting)
    min_len = min(p.shape[0] for p in sR_profiles)
    sR_stack = np.stack([p[:min_len, :] for p in sR_profiles], axis=0)  # (N, L, 2)
    sR_mean = sR_stack.mean(axis=0)  # (L, 2): col-0 = depth, col-1 = mean σR

    # ---- 5. Coverage & Almen intensity ----
    coverage_info = compute_coverage(
        disp_total_f32,
        plastic0,
        mesh["node_coords"],
        params.Lx,
        params.Ly,
        centres=centres,
    )
    # Almen intensity proxy: peak compressive residual stress in MPa
    almen_MPa = float(np.min(stress_total_f32[:, 0]) / 1e6)

    _log(f"[5/5] Coverage: {coverage_info['coverage_percent']:.1f}%  |  " f"Almen proxy: {almen_MPa:.1f} MPa")

    # ---- Auto-generate checkerboard summary if not supplied ----
    if checkerboard is None:
        cb_summary = displacements_to_checkerboard(disp_total_f32, mesh["node_coords"], params.Lx, params.Ly, grid_size)
    else:
        cb_summary = np.asarray(checkerboard, dtype=np.float32)
    # Also create a shot-density map as the ML input checkerboard
    shot_density_cb = _shots_to_density_map(centres, params.Lx, params.Ly, grid_size)

    # ---- Save ----
    if save_npy:
        os.makedirs(output_dir, exist_ok=True)
        _log(f"      Saving .npy to: {output_dir} ...")

        np.save(os.path.join(output_dir, "node_labels.npy"), mesh["node_labels"])
        np.save(os.path.join(output_dir, "node_coords.npy"), mesh["node_coords"])
        np.save(os.path.join(output_dir, "element_labels.npy"), mesh["element_labels"])
        np.save(os.path.join(output_dir, "element_connectivity.npy"), mesh["element_connectivity"])
        np.save(os.path.join(output_dir, "disp_node_labels.npy"), mesh["node_labels"])
        np.save(os.path.join(output_dir, "displacements.npy"), disp_total_f32)
        np.save(os.path.join(output_dir, "stress_element_labels.npy"), mesh["element_labels"])
        np.save(os.path.join(output_dir, "stresses.npy"), stress_total_f32)
        np.save(os.path.join(output_dir, "sR_depth_profile.npy"), sR_mean)
        np.save(os.path.join(output_dir, "shot_positions.npy"), centres)
        np.save(os.path.join(output_dir, "checkerboard.npy"), shot_density_cb)
        np.save(os.path.join(output_dir, "checkerboard_deformation.npy"), cb_summary)

        # Coverage summary text
        with open(os.path.join(output_dir, "coverage_report.txt"), "w") as fh:
            fh.write(f"n_shots: {n_actual}\n")
            fh.write(f"distribution: {params.distribution}\n")
            for k, v in coverage_info.items():
                fh.write(f"{k}: {v}\n")
            fh.write(f"almen_intensity_MPa: {almen_MPa:.3f}\n")

        _log("      Done.")

    _log("=" * 62)

    return {
        "params": params,
        "mesh": mesh,
        "centres": centres,
        "V_vec": V_vec,
        "D_vec": D_vec,
        "node_labels": mesh["node_labels"],
        "node_coords": mesh["node_coords"],
        "elem_labels": mesh["element_labels"],
        "element_connectivity": mesh["element_connectivity"],
        "displacements": disp_total_f32,
        "stresses": stress_total_f32,
        "sR_depth_profile": sR_mean,
        "energy_list": energy_list,
        "coverage": coverage_info,
        "coverage_fraction": coverage_info["coverage_fraction"],
        "almen_intensity_MPa": almen_MPa,
        "checkerboard": shot_density_cb,
        "checkerboard_deformation": cb_summary,
        "plastic_ref": plastic0,
        "contact_ref": contact0,
        "per_shot_physics": per_shot_physics,
        "E_b": bp.E_b,
    }


def _shots_to_density_map(
    centres: np.ndarray,
    Lx: float,
    Ly: float,
    G: int,
) -> np.ndarray:
    """Convert shot positions to a (G, G) shot-density checkerboard."""
    cb = np.zeros((G, G), dtype=np.float32)
    cell_w = Lx / G
    cell_h = Ly / G
    for c in centres:
        col = min(G - 1, int(c[0] / cell_w))
        row = min(G - 1, int(c[1] / cell_h))
        cb[row, col] += 1.0
    # Normalise by maximum to get [0, 1] range
    mx = cb.max()
    if mx > 0:
        cb /= mx
    return cb


# ---------------------------------------------------------------------------
# 6.  Physics-rich multi-channel sector encoding
# ---------------------------------------------------------------------------


def compute_physics_checkerboard(
    per_shot_physics: List[Dict],
    G: int,
    Lx: float,
    Ly: float,
) -> np.ndarray:
    """Build a 6-channel physics tensor (6, G, G) from per-shot impact data.

    Each cell (row i, col j) aggregates the deterministic contact-mechanics
    outcomes of every shot that landed inside it:

    Ch 0  shot_count             — number of impacts
    Ch 1  energy_density         — total KE per cell area  (J/m²)
    Ch 2  total_dent_depth       — sum of permanent indentation depths (m)
    Ch 3  surface_stress_density — sum of |σ_R(z=0)| per cell area (Pa/m²)
    Ch 4  coverage               — Avrami coverage fraction ∈ [0,1]
    Ch 5  bending_moment_density — sum of ∫σ_R·z dz per cell area (N/m)

    All channels are independently normalised to [0,1] across the plate.

    Parameters
    ----------
    per_shot_physics : list of dicts from run_multi_shot_simulation()
        Each dict must contain: x, y, V_n, D, rho_s, a_p, r_p,
        delta_p, sigma_R_surface, sR_depth (K,2) array.
    G    : grid size (G×G cells)
    Lx   : plate length in x (m)
    Ly   : plate length in y (m)

    Returns
    -------
    physics_cb : (6, G, G) float32, each channel in [0, 1]
    """
    cell_w = Lx / G
    cell_h = Ly / G
    A_cell = cell_w * cell_h

    # Raw accumulators (unnormalised)
    n_shots_grid = np.zeros((G, G), dtype=np.float64)  # ch 0
    energy_grid = np.zeros((G, G), dtype=np.float64)  # ch 1
    dent_grid = np.zeros((G, G), dtype=np.float64)  # ch 2
    stress_grid = np.zeros((G, G), dtype=np.float64)  # ch 3
    rp2_grid = np.zeros((G, G), dtype=np.float64)  # for coverage (ch 4)
    bimoment_grid = np.zeros((G, G), dtype=np.float64)  # ch 5

    for sp in per_shot_physics:
        col = min(G - 1, int(sp["x"] / cell_w))
        row = min(G - 1, int(sp["y"] / cell_h))

        n_shots_grid[row, col] += 1.0

        # KE = ½ ρ_s V_n² (π D³/6)
        vol_shot = math.pi * sp["D"] ** 3 / 6.0
        ke = 0.5 * sp["rho_s"] * sp["V_n"] ** 2 * vol_shot
        energy_grid[row, col] += ke

        dent_grid[row, col] += sp["delta_p"]
        stress_grid[row, col] += abs(sp["sigma_R_surface"])
        rp2_grid[row, col] += sp["r_p"] ** 2

        # Bending moment per shot: ∫ σ_R(z) · z dz  (trapezoidal)
        depth_prof = sp["sR_depth"]  # (K, 2): depth, sigma_R
        if depth_prof.shape[0] > 1:
            z_arr = depth_prof[:, 0]
            sR_arr = depth_prof[:, 1]
            bm = float(np.trapz(sR_arr * z_arr, z_arr))
        else:
            bm = 0.0
        bimoment_grid[row, col] += bm

    # Derive coverage from Avrami equation per cell
    coverage_grid = 1.0 - np.exp(-math.pi * rp2_grid / A_cell)

    # Normalise per-cell density quantities by cell area
    energy_grid /= A_cell
    stress_grid /= A_cell
    bimoment_grid /= A_cell

    # Stack channels
    raw = np.stack(
        [
            n_shots_grid,
            energy_grid,
            dent_grid,
            stress_grid,
            coverage_grid,
            bimoment_grid,
        ],
        axis=0,
    ).astype(
        np.float64
    )  # (6, G, G)

    # Normalise each channel independently to [0, 1]
    out = np.zeros_like(raw, dtype=np.float32)
    for c in range(6):
        ch = raw[c]
        mn, mx = ch.min(), ch.max()
        if mx > mn:
            out[c] = ((ch - mn) / (mx - mn)).astype(np.float32)
        else:
            out[c] = np.zeros_like(ch, dtype=np.float32)

    return out  # (6, G, G)


# ---------------------------------------------------------------------------
# 7.  Node-resolution influence fields from shot positions
# ---------------------------------------------------------------------------


def compute_influence_fields(
    shot_positions: np.ndarray,
    node_coords: np.ndarray,
    a_p: float,
    r_p: float,
    delta_p: float,
    Nx: int,
    Ny: int,
) -> np.ndarray:
    """Compute 4 physics-informed spatial fields at FEM node resolution.

    Each field is a (Nx+1, Ny+1) array evaluated at every mesh node.  The
    four channels encode the causal drivers of each displacement component:

    Ch 0  Hertz contact depth   Σ_j  δ_p · max(0, 1 − dist²/a_p²)
          → direct analytical proxy for |uz|; encodes where material was dented.
    Ch 1  Shot KDE              Σ_j  exp(−dist² / 2r_p²)
          → continuous density at FEM resolution; replaces coarse checkerboard.
    Ch 2  Lateral force x       Σ_j  (x_shot − x_node) · max(0, 1 − dist/r_p)
          → main causal driver of ux (shots push material laterally).
    Ch 3  Lateral force y       Σ_j  (y_shot − y_node) · max(0, 1 − dist/r_p)
          → main causal driver of uy.

    Analytical upper-bound correlations with ground-truth FEM (Ti+steel, n=20):
      Ch 0 vs uz: r ≈ 0.58    Ch 2 vs ux: r ≈ 0.65
    The current 10×10 checkerboard achieves only r ≈ 0.36 for uz.

    Parameters
    ----------
    shot_positions : (M, 2) float64  — shot x,y positions (m)
    node_coords    : (N, 3) float32  — FEM node coordinates (m), columns x,y,z
    a_p            : contact radius (m)
    r_p            : plastic zone radius (m)
    delta_p        : permanent indentation depth per shot (m) = a_p²/(2R)
    Nx, Ny         : mesh subdivision counts; grid is (Nx+1) × (Ny+1)

    Returns
    -------
    fields : (4, Nx+1, Ny+1) float32, each channel independently in [0, 1]
             (Fy channel is sign-preserved: normalised to [−1, 1] centered at 0)
    """
    if len(shot_positions) == 0:
        return np.zeros((4, Nx + 1, Ny + 1), dtype=np.float32)

    x_n = node_coords[:, 0].astype(np.float64)  # (N,)
    y_n = node_coords[:, 1].astype(np.float64)

    x_s = shot_positions[:, 0].astype(np.float64)  # (M,)
    y_s = shot_positions[:, 1].astype(np.float64)

    # (N, M) vectors from nodes to shots
    dx = x_s[None, :] - x_n[:, None]  # positive = shot is to the right of node
    dy = y_s[None, :] - y_n[:, None]
    dist2 = dx**2 + dy**2
    dist = np.sqrt(dist2)

    # Ch 0: Hertz contact depth (uz proxy) — parabolic kernel within contact radius
    hertz = delta_p * np.maximum(0.0, 1.0 - dist2 / max(a_p**2, 1e-30))
    ch0 = hertz.sum(axis=1)  # (N,)

    # Ch 1: Gaussian KDE with σ=r_p (smooth density at FEM resolution)
    ch1 = np.exp(-dist2 / max(2.0 * r_p**2, 1e-30)).sum(axis=1)  # (N,)

    # Ch 2,3: Lateral force fields — shots push nodes away (sign: shot − node)
    hat = np.maximum(0.0, 1.0 - dist / max(r_p, 1e-30))  # tent weight
    ch2 = (dx * hat).sum(axis=1)  # (N,) — Fx (positive = pushed in +x)
    ch3 = (dy * hat).sum(axis=1)  # (N,) — Fy

    def _norm_0_1(arr):
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            return ((arr - lo) / (hi - lo)).astype(np.float32)
        return np.zeros_like(arr, dtype=np.float32)

    def _norm_signed(arr):
        # Centre at 0.5 and scale by abs-max so range is within [0,1]
        scale = max(float(np.abs(arr).max()), 1e-30)
        return (arr / (2.0 * scale) + 0.5).astype(np.float32)

    # Reshape flat (N,) to (Nx+1, Ny+1) — node ordering is X-outer, Y-inner
    def _reshape(flat):
        return flat.reshape(Nx + 1, Ny + 1)

    fields = np.stack(
        [
            _reshape(_norm_0_1(ch0)),
            _reshape(_norm_0_1(ch1)),
            _reshape(_norm_signed(ch2)),
            _reshape(_norm_signed(ch3)),
        ],
        axis=0,
    ).astype(
        np.float32
    )  # (4, Nx+1, Ny+1)

    return fields


# ---------------------------------------------------------------------------
# 8.  Cupping (global Almen arc-height) from residual stress depth profile
# ---------------------------------------------------------------------------


def compute_cupping_from_profile(
    sR_depth_profile: np.ndarray,
    E_b: float,
    t_plate: float = 3e-3,
    L_plate: float = 10e-3,
) -> float:
    """Compute the Almen-style arc-height from the mean residual stress profile.

    Uses simple plate bending (Euler-Bernoulli):
        M_b  = ∫₀ᵗ σ_R(z) · z dz          [N·m/m]
        κ    = M_b / (E_b · t³/12)          [1/m]
        arc_height = κ · L² / 8             [m]

    Parameters
    ----------
    sR_depth_profile : (K, 2) array — col0 = depth z (m), col1 = σ_R (Pa).
                       z=0 is the surface; stresses are typically compressive
                       (negative) near the surface.
    E_b              : Workpiece Young's modulus (Pa).
    t_plate          : Plate thickness (m). Default 3 mm.
    L_plate          : Plate length (m). Default 10 mm.

    Returns
    -------
    arc_height : float (m). Positive = convex upward (peened side concave).
    """
    if sR_depth_profile.shape[0] < 2:
        return 0.0

    z = sR_depth_profile[:, 0]
    sR = sR_depth_profile[:, 1]

    # Clip integration to the plate thickness
    mask = z <= t_plate
    if mask.sum() < 2:
        mask = np.ones(len(z), dtype=bool)
    z_t = z[mask]
    sR_t = sR[mask]

    # Bending moment per unit width (N/m)
    M_b = float(np.trapz(sR_t * z_t, z_t))

    # Plate second moment of area per unit width (m³)
    I_per_w = t_plate**3 / 12.0

    # Curvature (1/m)
    kappa = M_b / (E_b * I_per_w) if E_b > 0 else 0.0

    # Midpoint arc-height (m)
    arc_height = kappa * L_plate**2 / 8.0

    return float(arc_height)


# ---------------------------------------------------------------------------
# 8.  Element-to-nodal stress averaging
# ---------------------------------------------------------------------------


def element_to_nodal_stress(
    element_stresses: np.ndarray,
    connectivity: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """Average element stresses to nodes by simple unweighted averaging.

    For each node, average the stresses of all elements that share it.

    Parameters
    ----------
    element_stresses : (N_elems, 4) float — [S11, S22, S33, S12] per element.
    connectivity     : (N_elems, 4) int   — node indices of each quad element
                       (0-based, matching element_stresses row order).
    num_nodes        : Total number of nodes.

    Returns
    -------
    nodal_stresses : (num_nodes, 4) float32
    """
    n_comp = element_stresses.shape[1]
    accum = np.zeros((num_nodes, n_comp), dtype=np.float64)
    count = np.zeros(num_nodes, dtype=np.float64)

    for e_idx, nodes in enumerate(connectivity):
        for n_idx in nodes:
            if 0 <= n_idx < num_nodes:
                accum[n_idx] += element_stresses[e_idx]
                count[n_idx] += 1.0

    # Avoid divide-by-zero for isolated nodes
    safe = count > 0
    result = np.zeros((num_nodes, n_comp), dtype=np.float32)
    result[safe] = (accum[safe] / count[safe, None]).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# 9.  Plotting
# ---------------------------------------------------------------------------


def plot_results(results: Dict, show: bool = True, save_dir: Optional[str] = None) -> None:
    """Four-panel summary plot for a multi-shot simulation.

    Panels:
      [0] Residual stress depth profile (mean ± std)
      [1] Surface displacement magnitude heatmap (dent map)
      [2] Shot impact positions
      [3] Checkerboard shot-density map
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    mesh = results["mesh"]
    coords = mesh["node_coords"]  # (N, 3)
    disp = results["displacements"]  # (N, 3)
    centres = results["centres"]  # (M, 2)
    sR = results["sR_depth_profile"]  # (L, 2)
    cb = results["checkerboard"]  # (G, G)
    Lx = results["params"].Lx
    Ly = results["params"].Ly

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # ---- [0] Residual stress depth profile ----
    ax = axs[0, 0]
    ax.plot(sR[:, 0] * 1e6, sR[:, 1] / 1e6, color="steelblue", linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Depth z (µm)")
    ax.set_ylabel("Residual Stress σR (MPa)")
    ax.set_title("Mean Residual Stress Depth Profile")
    ax.set_xlim(left=0)

    # ---- [1] Surface uz heatmap ----
    ax = axs[0, 1]
    surf_mask = np.abs(coords[:, 2]) < 1e-9
    xs_surf = coords[surf_mask, 0]
    ys_surf = coords[surf_mask, 1]
    uz_surf = disp[surf_mask, 2]

    G_heat = 80
    x_edges = np.linspace(0, Lx, G_heat + 1)
    y_edges = np.linspace(0, Ly, G_heat + 1)
    heat = np.zeros((G_heat, G_heat))
    cnt = np.zeros((G_heat, G_heat), dtype=int)
    for x, y, u in zip(xs_surf, ys_surf, uz_surf):
        ci = min(G_heat - 1, int(x / Lx * G_heat))
        ri = min(G_heat - 1, int(y / Ly * G_heat))
        heat[ri, ci] += u
        cnt[ri, ci] += 1
    mask = cnt > 0
    heat[mask] /= cnt[mask]

    im = ax.imshow(heat * 1e6, origin="lower", extent=[0, Lx * 1e3, 0, Ly * 1e3], cmap="RdBu_r", aspect="equal")
    plt.colorbar(im, ax=ax, label="uz (µm)")
    ax.set_title("Surface Displacement uz (µm)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    # ---- [2] Shot positions ----
    ax = axs[1, 0]
    ax.scatter(centres[:, 0] * 1e3, centres[:, 1] * 1e3, s=4, color="firebrick", alpha=0.6)
    ax.set_xlim(0, Lx * 1e3)
    ax.set_ylim(0, Ly * 1e3)
    ax.set_aspect("equal")
    ax.set_title(
        f"Shot Positions (N={len(centres)})\n"
        f"Coverage: {results['coverage']['coverage_percent']:.1f}%  |  "
        f"Almen: {results['almen_intensity_MPa']:.0f} MPa"
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    # ---- [3] Shot-density checkerboard ----
    ax = axs[1, 1]
    im2 = ax.imshow(
        cb,
        origin="lower",
        cmap="viridis",
        aspect="equal",
        extent=[0, Lx * 1e3, 0, Ly * 1e3],
        vmin=0,
        vmax=cb.max() or 1.0,
    )
    plt.colorbar(im2, ax=ax, label="Normalised shot density")
    ax.set_title("Shot Density Checkerboard (ML input)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    fig.suptitle(
        f"Multi-Shot Simulation  |  N={len(centres)} shots  |  "
        f"V={results['params'].base_params.V:.0f} m/s  |  "
        f"D={results['params'].base_params.D*1e3:.2f} mm",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    if save_dir:
        fig.savefig(os.path.join(save_dir, "multi_shot_summary.png"), dpi=150, bbox_inches="tight")

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-shot peening simulation (analytical superposition).")
    parser.add_argument("--output", default="./multi_shot_output")
    parser.add_argument("--n_shots", type=int, default=50)
    parser.add_argument("--V", type=float, default=35.9)
    parser.add_argument("--D", type=float, default=0.0005)
    parser.add_argument("--Lx", type=float, default=0.010)
    parser.add_argument("--Ly", type=float, default=0.010)
    parser.add_argument("--Nx", type=int, default=50)
    parser.add_argument("--Ny", type=int, default=50)
    parser.add_argument("--dist", default="random", choices=["random", "grid", "poisson", "checkerboard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--V_scatter", type=float, default=2.0)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    bp = ShotPeenParams(V=args.V, D=args.D)
    msp = MultiShotParams(
        base_params=bp,
        n_shots=args.n_shots,
        distribution=args.dist,
        seed=args.seed,
        V_scatter=args.V_scatter,
        Lx=args.Lx,
        Ly=args.Ly,
        Nx=args.Nx,
        Ny=args.Ny,
    )

    results = run_multi_shot_simulation(
        params=msp,
        output_dir=args.output,
        grid_size=args.grid_size,
    )

    print(f"\nCoverage:        {results['coverage_fraction']*100:.1f}%")
    print(f"Almen intensity: {results['almen_intensity_MPa']:.1f} MPa")

    if args.plot:
        plot_results(results, save_dir=args.output)
