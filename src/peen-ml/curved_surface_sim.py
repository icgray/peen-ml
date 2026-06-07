"""
curved_surface_sim.py
=====================
Physics-based shot peening simulation on arbitrary 3D curved surfaces.

Extends the flat-plate multi_shot_sim pipeline to:
  • Accept an STL file (via STLSurface) as the target surface instead of a
    structured flat plate.
  • Accept a NozzleTrajectory so the nozzle scans along a path rather than
    staying at a fixed position.

Algorithm (per trajectory step)
---------------------------------
1. Sample shot impacts from the Gaussian nozzle model centred at the current
   nozzle position (sample_gaussian_nozzle_shots).
2. Project 2D shot positions onto the 3D STL surface (STLSurface.project_shots).
3. Correct impact velocity for local surface angle:
   V_eff = V_n × cos(angle_from_normal).
4. Run the Shen & Atluri (2006) single-shot analytical model for each impact
   (impact_sim: compute_contact_params → compute_stress_field → compute_plastic_zone
   → map_displacements → map_stresses).
5. Accumulate displacements and stresses at STL vertices.

Approximations
--------------
• Displacements are computed in the global XYZ frame: `uz` follows the global
  Z axis, not the local surface normal.  For gently curved surfaces the error
  is small.  Apply STLSurface.vertex_normal_rotation_matrices() after the
  simulation if you need surface-normal-aligned displacements.
• Radial distance from impact uses XY components only (same as flat-plate code).
• Element "depth" (z_elem) for stress is |vertex_z| in global frame.

Flat-plate fallback
-------------------
When stl_path is None the call delegates to run_multi_shot_simulation()
from multi_shot_sim.py with no behaviour change.

Output schema
-------------
Same .npy files as multi_shot_sim:
    node_coords.npy, node_labels.npy,
    element_connectivity.npy, element_labels.npy,
    displacements.npy, disp_node_labels.npy,
    stresses.npy, stress_element_labels.npy,
    sR_depth_profile.npy, shot_positions.npy,
    checkerboard.npy, coverage_report.txt
Plus STL-specific extras:
    stl_vertex_normals.npy, stl_face_normals.npy

Usage
-----
from curved_surface_sim import CurvedSurfaceSimParams, run_curved_surface_sim
from nozzle_trajectory import ScanParams, raster_scan

traj   = raster_scan(ScanParams(Lx=0.04, Ly=0.04, z_standoff=0.15,
                                scan_speed=0.10, line_spacing=0.005))
params = CurvedSurfaceSimParams(stl_path="panel.stl", trajectory=traj)
results = run_curved_surface_sim(params)
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from impact_sim import (
    ShotPeenParams,
    compute_contact_params,
    compute_stress_field,
    compute_plastic_zone,
    compute_energy_balance,
    map_displacements,
    map_stresses,
)
from multi_shot_sim import (
    MultiShotParams,
    compute_coverage,
    run_multi_shot_simulation,
)
from materials import SHOT_MATERIALS  # noqa: F401 (also imported by gaussian_nozzle_dataset_gen)
from gaussian_nozzle_dataset_gen import (
    sample_gaussian_nozzle_shots,
)
from stl_surface import STLSurface
from nozzle_trajectory import NozzleTrajectory

__all__ = ["CurvedSurfaceSimParams", "run_curved_surface_sim"]


@dataclass
class CurvedSurfaceSimParams:
    """Parameters for a curved-surface shot peening simulation.

    Surface
    -------
    stl_path    : Path to an STL file.  None → flat-plate fallback.

    Nozzle movement
    ---------------
    trajectory  : NozzleTrajectory with (T, 3) nozzle positions.
                  None → single static Gaussian footprint at the surface centre.

    Nozzle / shot physics
    ---------------------
    h_nozzle         : Standoff height (m). Used when trajectory is None.
    theta_div        : Jet cone half-angle (rad).
    V_mean           : Mean exit velocity (m/s).
    sigma_V_frac     : Velocity CV = sigma_V / V_mean.
    V_exit_min       : Minimum allowed exit velocity (m/s).
    n_shots_per_step : Shots sampled per trajectory step (or total shots when
                       trajectory is None).
    D                : Shot diameter (m).
    shot_material    : One of 'steel', 'ceramic', 'glass', 'cast_iron'.
    sigma_yield      : Workpiece yield stress (Pa).
    E_b              : Workpiece Young's modulus (Pa).
    nu_b             : Workpiece Poisson's ratio.
    c                : Bilinear hardening modulus (Pa).

    Output
    ------
    G          : Checkerboard resolution (G × G).
    output_dir : Directory to write .npy files.
    save_npy   : Write .npy files to disk.
    verbose    : Print progress.
    seed       : RNG seed.
    """

    # Surface
    stl_path: Optional[str] = None

    # Nozzle movement
    trajectory: Optional[NozzleTrajectory] = None

    # Nozzle / shot physics
    h_nozzle: float = 0.150
    theta_div: float = math.radians(15.0)
    V_mean: float = 50.0
    sigma_V_frac: float = 0.10
    V_exit_min: float = 5.0
    n_shots_per_step: int = 10
    D: float = 0.0005
    shot_material: str = "steel"
    sigma_yield: float = 276e6
    E_b: float = 113.8e9
    nu_b: float = 0.34
    c: float = 3.0e9

    # Output
    G: int = 20
    output_dir: str = "./curved_output"
    save_npy: bool = True
    verbose: bool = True
    seed: int = 42


def run_curved_surface_sim(params: CurvedSurfaceSimParams) -> Dict:
    """Run a curved-surface shot peening simulation.

    When params.stl_path is None, delegates to run_multi_shot_simulation()
    (flat-plate mode, no behaviour change from existing pipeline).

    Parameters
    ----------
    params : CurvedSurfaceSimParams

    Returns
    -------
    dict with keys:
        node_coords, node_labels, element_connectivity, element_labels,
        displacements, stresses, sR_depth_profile,
        shot_positions_all, V_normal_all,
        checkerboard, coverage, coverage_fraction, almen_intensity_MPa,
        stl_surface (STLSurface instance, or None for flat-plate)
    """
    _log = print if params.verbose else (lambda *a, **kw: None)

    # ------------------------------------------------------------------
    # Flat-plate fallback
    # ------------------------------------------------------------------
    if params.stl_path is None:
        _log("[CurvedSurfaceSim] No STL provided — running flat-plate simulation.")
        mat = SHOT_MATERIALS.get(params.shot_material, SHOT_MATERIALS["steel"])
        n_total = params.n_shots_per_step * (params.trajectory.n_steps if params.trajectory else 1)
        ms_params = MultiShotParams(
            base_params=ShotPeenParams(
                V=params.V_mean,
                D=params.D,
                sigma_yield=params.sigma_yield,
                E_b=params.E_b,
                nu_b=params.nu_b,
                c=params.c,
                E_s=mat["E_s"],
                nu_s=mat["nu_s"],
                rho_s=mat["rho_s"],
            ),
            n_shots=n_total,
            distribution="random",
            seed=params.seed,
        )
        result = run_multi_shot_simulation(
            ms_params,
            output_dir=params.output_dir,
            save_npy=params.save_npy,
            verbose=params.verbose,
            grid_size=params.G,
        )
        result["stl_surface"] = None
        return result

    # ------------------------------------------------------------------
    # Load STL surface
    # ------------------------------------------------------------------
    _log(f"[CurvedSurfaceSim] Loading STL: {params.stl_path}")
    surface = STLSurface(params.stl_path)
    _log(f"  {surface.n_vertices} vertices, {surface.n_faces} faces")

    node_coords, node_labels = surface.to_node_arrays()
    elem_conn, elem_labels = surface.to_element_arrays()
    N_nodes = surface.n_vertices
    N_elems = surface.n_faces

    # Surface bounding box (for Gaussian nozzle placement)
    bounds = surface.bounds()
    x_min = float(bounds[0, 0])
    y_min = float(bounds[0, 1])
    Lx = max(float(bounds[1, 0]) - x_min, 1e-9)
    Ly = max(float(bounds[1, 1]) - y_min, 1e-9)

    # ------------------------------------------------------------------
    # Material / shot shared base params
    # ------------------------------------------------------------------
    mat = SHOT_MATERIALS.get(params.shot_material, SHOT_MATERIALS["steel"])
    base_sp = ShotPeenParams(
        E_s=mat["E_s"],
        nu_s=mat["nu_s"],
        rho_s=mat["rho_s"],
        D=params.D,
        V=params.V_mean,
        E_b=params.E_b,
        nu_b=params.nu_b,
        sigma_yield=params.sigma_yield,
        c=params.c,
    )

    rng = np.random.default_rng(params.seed)
    sigma_V = params.V_mean * params.sigma_V_frac

    # Mesh dict expected by map_displacements / map_stresses
    mesh_dict = {
        "node_labels": node_labels,
        "node_coords": node_coords,
        "element_labels": elem_labels,
        "element_connectivity": elem_conn,
        "impact_center": np.array([0.0, 0.0, 0.0]),
    }

    # ------------------------------------------------------------------
    # Trajectory steps
    # ------------------------------------------------------------------
    if params.trajectory is not None:
        nozzle_positions = params.trajectory.positions  # (T, 3)
    else:
        cx = x_min + Lx / 2.0
        cy = y_min + Ly / 2.0
        nozzle_positions = np.array([[cx, cy, params.h_nozzle]], dtype=np.float32)

    T = len(nozzle_positions)
    _log(
        f"[CurvedSurfaceSim] {T} trajectory step(s) × "
        f"{params.n_shots_per_step} shots/step = "
        f"{T * params.n_shots_per_step} total impacts"
    )

    # ------------------------------------------------------------------
    # Accumulation arrays
    # ------------------------------------------------------------------
    disp_total = np.zeros((N_nodes, 3), dtype=np.float64)
    stress_total = np.zeros((N_elems, 4), dtype=np.float64)
    all_shot_xyz: List[np.ndarray] = []
    all_V_normal: List[np.ndarray] = []
    sR_profiles: List[np.ndarray] = []
    processed = 0

    _log("[CurvedSurfaceSim] Simulating impacts ...")

    for t_idx, nozzle_pos in enumerate(nozzle_positions):
        nx, ny, nz = float(nozzle_pos[0]), float(nozzle_pos[1]), float(nozzle_pos[2])
        h_eff = max(abs(nz), 0.001)

        # Sample shots: nozzle_x/y in [0, Lx/Ly] frame
        centres_2d, V_norm, _, _ = sample_gaussian_nozzle_shots(
            h_nozzle=h_eff,
            theta_div=params.theta_div,
            V_mean=params.V_mean,
            sigma_V=sigma_V,
            n_shots=params.n_shots_per_step,
            Lx=Lx,
            Ly=Ly,
            nozzle_x=nx - x_min,
            nozzle_y=ny - y_min,
            V_exit_min=params.V_exit_min,
            rng=rng,
        )

        # Convert back to global coordinate frame
        shot_xy_global = centres_2d.copy()
        shot_xy_global[:, 0] += x_min
        shot_xy_global[:, 1] += y_min

        # Project onto STL surface (orthographic XY snap)
        hit_xyz, surf_normals, impact_angles = surface.project_shots_onto_surface(shot_xy_global, z_nozzle=nz)

        all_shot_xyz.append(hit_xyz)
        all_V_normal.append(V_norm)

        # Process each individual impact
        for i in range(len(hit_xyz)):
            # Normal-component velocity corrected for surface angle
            V_n_i = float(V_norm[i]) * math.cos(float(impact_angles[i]))
            V_n_i = max(V_n_i, params.V_exit_min)

            p_i = ShotPeenParams(
                E_s=mat["E_s"],
                nu_s=mat["nu_s"],
                rho_s=mat["rho_s"],
                D=params.D,
                V=V_n_i,
                phi=math.pi / 2.0,  # normal component already applied above
                E_b=params.E_b,
                nu_b=params.nu_b,
                sigma_yield=params.sigma_yield,
                c=params.c,
                n_depth=1000,  # coarser for speed
            )

            contact_i = compute_contact_params(p_i)
            plastic_i = compute_plastic_zone(p_i)
            sf_i = compute_stress_field(contact_i, p_i)

            ic = hit_xyz[i].astype(np.float64)

            _, disp_i = map_displacements(mesh_dict, contact_i, plastic_i, p_i, ic)
            _, stress_i = map_stresses(mesh_dict, sf_i, plastic_i, p_i, ic)

            disp_total += disp_i.astype(np.float64)
            stress_total += stress_i.astype(np.float64)
            sR_profiles.append(np.stack([sf_i["Z"], sf_i["sR"]], axis=1))

            processed += 1

        if params.verbose and T > 1 and (t_idx + 1) % max(1, T // 10) == 0:
            _log(f"  Step {t_idx + 1}/{T} — {processed} impacts processed ...")

    disp_f32 = disp_total.astype(np.float32)
    stress_f32 = stress_total.astype(np.float32)

    # ------------------------------------------------------------------
    # Aggregate shot data
    # ------------------------------------------------------------------
    all_shot_xyz_np = np.concatenate(all_shot_xyz, axis=0)  # (total, 3)
    all_V_normal_np = np.concatenate(all_V_normal, axis=0)  # (total,)

    # ------------------------------------------------------------------
    # Checkerboard: energy-weighted XY projection
    # ------------------------------------------------------------------
    checkerboard = surface.shots_to_checkerboard(all_shot_xyz_np, all_V_normal_np**2, params.G)

    # ------------------------------------------------------------------
    # Mean residual stress depth profile
    # ------------------------------------------------------------------
    if sR_profiles:
        min_len = min(p.shape[0] for p in sR_profiles)
        sR_stack = np.stack([p[:min_len] for p in sR_profiles])
        sR_mean = sR_stack.mean(axis=0)
    else:
        sR_mean = np.zeros((10, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Coverage and Almen intensity
    # ------------------------------------------------------------------
    contact0 = compute_contact_params(base_sp)
    plastic0 = compute_plastic_zone(base_sp)
    coverage_info = compute_coverage(
        disp_f32,
        plastic0,
        node_coords,
        Lx,
        Ly,
        centres=all_shot_xyz_np[:, :2],
    )
    almen_MPa = float(np.min(stress_f32[:, 0]) / 1e6)

    _log(
        f"[CurvedSurfaceSim] Coverage: {coverage_info['coverage_percent']:.1f}% | " f"Almen proxy: {almen_MPa:.1f} MPa"
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if params.save_npy:
        os.makedirs(params.output_dir, exist_ok=True)
        np.save(os.path.join(params.output_dir, "node_coords.npy"), node_coords)
        np.save(os.path.join(params.output_dir, "node_labels.npy"), node_labels)
        np.save(os.path.join(params.output_dir, "element_connectivity.npy"), elem_conn)
        np.save(os.path.join(params.output_dir, "element_labels.npy"), elem_labels)
        np.save(os.path.join(params.output_dir, "displacements.npy"), disp_f32)
        np.save(os.path.join(params.output_dir, "disp_node_labels.npy"), node_labels)
        np.save(os.path.join(params.output_dir, "stresses.npy"), stress_f32)
        np.save(os.path.join(params.output_dir, "stress_element_labels.npy"), elem_labels)
        np.save(os.path.join(params.output_dir, "sR_depth_profile.npy"), sR_mean)
        np.save(os.path.join(params.output_dir, "shot_positions.npy"), all_shot_xyz_np[:, :2])
        np.save(os.path.join(params.output_dir, "checkerboard.npy"), checkerboard)
        np.save(os.path.join(params.output_dir, "stl_vertex_normals.npy"), surface.vertex_normals)
        np.save(os.path.join(params.output_dir, "stl_face_normals.npy"), surface.face_normals)

        with open(os.path.join(params.output_dir, "coverage_report.txt"), "w") as fh:
            fh.write(f"n_total_impacts: {processed}\n")
            fh.write(f"n_trajectory_steps: {T}\n")
            fh.write(f"shots_per_step: {params.n_shots_per_step}\n")
            fh.write(f"stl_path: {params.stl_path}\n")
            for k, v in coverage_info.items():
                fh.write(f"{k}: {v}\n")
            fh.write(f"almen_intensity_MPa: {almen_MPa:.3f}\n")

        _log(f"[CurvedSurfaceSim] Results saved to: {params.output_dir}")

    return {
        "node_coords": node_coords,
        "node_labels": node_labels,
        "element_connectivity": elem_conn,
        "element_labels": elem_labels,
        "displacements": disp_f32,
        "stresses": stress_f32,
        "sR_depth_profile": sR_mean,
        "shot_positions_all": all_shot_xyz_np,
        "V_normal_all": all_V_normal_np,
        "checkerboard": checkerboard,
        "coverage": coverage_info,
        "coverage_fraction": coverage_info["coverage_fraction"],
        "almen_intensity_MPa": almen_MPa,
        "stl_surface": surface,
    }
