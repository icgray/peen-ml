"""
impact_sim.py
=============
Python-native single-shot impact simulation for shot peening.

Implements the Shen & Atluri (2006) analytical model
"An Analytical Model for Shot-Peening Induced Residual Stresses"
CMC: Computers, Materials & Continua, vol. 4, no. 2, pp. 75-85.

This module:
    1. Generates a structured hex/quad mesh (same .npy schema as Abaqus outputs).
    2. Computes elastic contact parameters via Hertzian contact theory (Eqs 1-8).
    3. Solves elastic-plastic loading with bilinear hardening (Eqs 15-26).
    4. Computes plastic zone geometry: dent radius a_p, plastic zone radius r_p (Eqs 42-45).
    5. Builds the post-unloading residual stress depth profile σR(z) (Eqs 27-36).
    6. Tracks energy balance: KE_initial → W_plastic + KE_rebound + W_wave.
    7. Maps the analytical stress/displacement fields onto mesh nodes and elements.
    8. Saves all results as .npy files compatible with data_viz.py.

Usage
-----
    from impact_sim import ShotPeenParams, run_simulation

    params = ShotPeenParams()                    # titanium / S170 defaults
    results = run_simulation(params, output_dir="./Simulation_0")

    # Or override material/shot parameters:
    params = ShotPeenParams(V=40.0, sigma_yield=350e6, D=0.0008)
    results = run_simulation(params, output_dir="./custom_sim", Nx=20, Ny=20)

Author: PeenML project
Reference: Shen & Atluri (2006), CMC vol. 4 no. 2 pp. 75-85.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    from materials import WORKPIECE_MATERIALS, SHOT_MATERIALS  # noqa: F401
except ImportError:
    WORKPIECE_MATERIALS = {}
    SHOT_MATERIALS = {}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "ShotPeenParams",
    "generate_mesh",
    "compute_contact_params",
    "compute_stress_field",
    "compute_plastic_zone",
    "compute_energy_balance",
    "map_displacements",
    "map_stresses",
    "run_simulation",
    "plot_residual_stress",
]


# ---------------------------------------------------------------------------
# 1.  Material / shot parameter container
# ---------------------------------------------------------------------------


@dataclass
class ShotPeenParams:
    """All physical parameters for a single shot-peening impact.

    Shot (steel S170 defaults)
    --------------------------
    E_s       : Young's modulus of shot (Pa)
    nu_s      : Poisson's ratio of shot
    D         : Shot diameter (m)
    rho_s     : Shot density (kg/m³)

    Target (aerospace-grade titanium alloy defaults)
    ------------------------------------------------
    E_b       : Young's modulus of target (Pa)
    nu_b      : Poisson's ratio of target
    sigma_yield : Yield stress of target (Pa)
    c         : Bilinear hardening modulus — equals (2/3)·E_p (Pa)

    Impact conditions
    -----------------
    V         : Impact speed (m/s)
    phi       : Impact angle from surface (rad); pi/2 = normal impact
    k         : Elasticity factor (dimensionless, typically 0.8)

    Depth-profile resolution
    ------------------------
    n_depth   : Number of depth points for the stress profile
    depth_max_factor : Profile runs to depth_max_factor × ae
    """

    # Shot
    E_s: float = 210e9
    nu_s: float = 0.3
    D: float = 0.0005
    rho_s: float = 2000.0

    # Target
    E_b: float = 113.8e9
    nu_b: float = 0.34
    sigma_yield: float = 276e6
    c: float = 3.0e9  # bilinear hardening slope

    # Impact
    V: float = 35.9
    phi: float = math.pi / 2
    k: float = 0.8

    # Depth resolution
    n_depth: int = 300_000
    depth_max_factor: float = 8.0

    # ------------------------------------------------------------------ #
    # Derived quantities (computed lazily from the fields above)
    # ------------------------------------------------------------------ #

    @property
    def R(self) -> float:
        """Shot radius (m)."""
        return self.D / 2.0

    @property
    def Ms(self) -> float:
        """Shot mass (kg)."""
        return (4.0 / 3.0) * math.pi * self.R**3 * self.rho_s

    @property
    def Vn(self) -> float:
        """Normal component of impact velocity (m/s)."""
        return self.V * math.sin(self.phi)


# ---------------------------------------------------------------------------
# 2.  Structured quad / hex mesh generator
# ---------------------------------------------------------------------------


def generate_mesh(
    Lx: float = 0.005,
    Ly: float = 0.005,
    Lz: float = 0.002,
    Nx: int = 10,
    Ny: int = 10,
    Nz: int = 1,
) -> Dict[str, np.ndarray]:
    """Create a structured quad (2-D) or hex (3-D) mesh.

    Parameters
    ----------
    Lx, Ly, Lz : Physical dimensions (m). Lz is ignored when Nz==1.
    Nx, Ny, Nz : Element counts along each axis. Nz=1 gives a surface mesh.

    Returns
    -------
    dict with keys:
        node_labels          : (N,)   int32
        node_coords          : (N, 3) float32  — [x, y, z]
        element_labels       : (E,)   int32
        element_connectivity : (E, 4) int32    — 4-node quad per element (surface)
                               (E, 8) int32    — 8-node hex (Nz > 1)
        impact_center        : (3,)   float64  — centre of the top surface
    """
    # Node grid: (Nx+1) × (Ny+1) × (Nz+1)
    xs = np.linspace(0.0, Lx, Nx + 1, dtype=np.float32)
    ys = np.linspace(0.0, Ly, Ny + 1, dtype=np.float32)

    if Nz == 1:
        # ---- 2-D surface mesh ----------------------------------------
        xx, yy = np.meshgrid(xs, ys, indexing="ij")  # (Nx+1, Ny+1)
        zz = np.zeros_like(xx)
        coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)  # (N, 3)

        N = coords.shape[0]
        labels = np.arange(1, N + 1, dtype=np.int32)

        # Build quad connectivity (row-major node indexing)
        def node_id(ix, iy):
            return ix * (Ny + 1) + iy  # 0-based

        quads = []
        for ix in range(Nx):
            for iy in range(Ny):
                n0 = node_id(ix, iy)
                n1 = node_id(ix + 1, iy)
                n2 = node_id(ix + 1, iy + 1)
                n3 = node_id(ix, iy + 1)
                quads.append([labels[n0], labels[n1], labels[n2], labels[n3]])

        connectivity = np.array(quads, dtype=np.int32)  # (Nx*Ny, 4)
        elem_labels = np.arange(1, len(quads) + 1, dtype=np.int32)

        impact_center = np.array([Lx / 2.0, Ly / 2.0, 0.0])

    else:
        # ---- 3-D hex mesh --------------------------------------------
        zs = np.linspace(0.0, -Lz, Nz + 1, dtype=np.float32)  # z goes downward (negative)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")  # (Nx+1, Ny+1, Nz+1)
        coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

        N = coords.shape[0]
        labels = np.arange(1, N + 1, dtype=np.int32)

        nNy1 = Ny + 1
        nNz1 = Nz + 1

        def node_id3(ix, iy, iz):
            return ix * nNy1 * nNz1 + iy * nNz1 + iz  # 0-based

        hexes = []
        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    n = [
                        node_id3(ix, iy, iz),
                        node_id3(ix + 1, iy, iz),
                        node_id3(ix + 1, iy + 1, iz),
                        node_id3(ix, iy + 1, iz),
                        node_id3(ix, iy, iz + 1),
                        node_id3(ix + 1, iy, iz + 1),
                        node_id3(ix + 1, iy + 1, iz + 1),
                        node_id3(ix, iy + 1, iz + 1),
                    ]
                    hexes.append([labels[k] for k in n])

        connectivity = np.array(hexes, dtype=np.int32)
        elem_labels = np.arange(1, len(hexes) + 1, dtype=np.int32)
        impact_center = np.array([Lx / 2.0, Ly / 2.0, 0.0])

    return {
        "node_labels": labels,
        "node_coords": coords,
        "element_labels": elem_labels,
        "element_connectivity": connectivity,
        "impact_center": impact_center,
    }


# ---------------------------------------------------------------------------
# 3.  Hertzian contact parameters   (Eqs 1-8)
# ---------------------------------------------------------------------------


def compute_contact_params(params: ShotPeenParams) -> Dict[str, float]:
    """Compute Hertzian contact parameters.

    Equations follow Shen & Atluri (2006):
        Eq 3  — equivalent elastic modulus E_eq
        Eq 2  — elastic contact radius ae
        Impact force F estimated from Hertz impact time (Johnson 1985, p. 353)

    Returns
    -------
    dict with keys: E_eq, ae, p0, F, t, delta (max indentation)
    """
    p = params

    # Eq 3: equivalent modulus
    E_eq = 1.0 / ((1.0 - p.nu_s) / p.E_s + (1.0 - p.nu_b) / p.E_b)

    # Eq 2: elastic contact radius
    ae = p.R * ((5.0 * math.pi * p.k * p.rho_s * p.Vn**2) / (4.0 * E_eq)) ** (1.0 / 5.0)

    # Impact time (Hertz, Johnson 1985)
    t = 2.87 * (1.0 - p.nu_s**2) * 0.4 * (p.Ms**2 / (p.R * p.E_s**2 * p.Vn)) ** 0.2

    # Average contact force
    F = p.Ms * 2.0 * p.Vn / t

    # Peak Hertz pressure
    p0 = (1.0 / math.pi) * (6.0 * F * E_eq**2 / p.R**2) ** (1.0 / 3.0)

    # Maximum indentation depth (Hertz)
    delta = ae**2 / p.R

    return {"E_eq": E_eq, "ae": ae, "p0": p0, "F": F, "t": t, "delta": delta}


# ---------------------------------------------------------------------------
# 4.  Elastic stress field + elastic-plastic loading (Eqs 4-26)
# ---------------------------------------------------------------------------


def compute_stress_field(contact: Dict[str, float], params: ShotPeenParams) -> Dict[str, np.ndarray]:
    """Compute the through-depth stress field during loading and after unloading.

    Follows Shen & Atluri (2006) Equations 4–26, then residual stress Eqs 27–36.

    Parameters
    ----------
    contact : output of compute_contact_params()
    params  : ShotPeenParams instance

    Returns
    -------
    dict with numpy arrays indexed along the depth axis:
        Z           : (n,) depth from surface (m)
        Z_bar       : (n,) = Z / ae  (dimensionless)
        sigma_xe    : (n,) elastic σx  (Pa)
        sigma_ze    : (n,) elastic σz  (Pa)
        sigma_eqe   : (n,) equivalent elastic stress (Pa)
        eps_load_p  : (n,) plastic strain during loading  (Eq 20)
        eps_unload_p: (n,) plastic strain after unloading (Eq 26)
        Sxl         : (n,) deviatoric stress during loading (Eq 21)
        Sxu         : (n,) deviatoric stress after unloading (Eq 23)
        eps_avg     : (n,) average plastic strain (Eq 47a/b)
        sxs         : (n,) stress driving residual (Eq 22)
        sR          : (n,) residual stress σR (Pa)  (Eq 35)
    """
    p = params
    ae = contact["ae"]
    p0 = contact["p0"]
    E_b = p.E_b
    nu_b = p.nu_b
    sy = p.sigma_yield
    c = p.c

    # Depth axis
    Z = np.linspace(ae / 1000.0, params.depth_max_factor * ae, params.n_depth)
    Z_bar = Z / ae

    # ---- Hertz elastic stress field under contact centre ---- #
    # Eq 5a, 5b
    A = 1.0 / (1.0 + Z_bar**2)
    B = 1.0 - Z_bar * np.arctan(1.0 / Z_bar)

    # Eq 4
    sigma_xe = -p0 * (-A / 2.0 + (1.0 + nu_b) * B)  # = sigma_ye by symmetry
    sigma_ze = -p0 * A

    # Eq 6: von-Mises equivalent stress (biaxial, sigma_y = sigma_x)
    sigma_eqe = sigma_xe - sigma_ze

    # ---- Elastic strains (Eq 7a, 7b, 8) ---- #
    epsilon_xe = (1.0 / E_b) * (sigma_xe * (1.0 - nu_b) - nu_b * sigma_xe)  # Eq 7a  # noqa: F841
    epsilon_ze = (1.0 / E_b) * (sigma_ze - 2.0 * sigma_xe * nu_b)  # Eq 7b  # noqa: F841

    # ---- Plastic loading (Eq 20, 21) ---- #
    # Eq 20: plastic strain during loading
    disc_load = sy**2 + (3.0 * c / (2.0 * E_b)) * sigma_eqe**2
    disc_load = np.maximum(disc_load, 0.0)
    eps_load_p = (np.sqrt(disc_load) - sy) / (3.0 * c)
    eps_load_p = np.maximum(eps_load_p, 0.0)

    # Eq 21: deviatoric stress during loading
    Sxl = sy / 3.0 + c * eps_load_p

    # ---- Unloading (Eq 26, 23, 30) ---- #
    # Eq 26: equivalent plastic strain at unloading
    disc_unload = 4.0 * sy**2 + (3.0 * c / (2.0 * E_b)) * sigma_eqe**2
    disc_unload = np.maximum(disc_unload, 0.0)
    exup = (np.sqrt(disc_unload) - 2.0 * sy) / (3.0 * c)
    exup = np.maximum(exup, 0.0)

    # Eq 23: deviatoric stress at unloading
    Sxu = Sxl - (2.0 / 3.0) * sy - c * exup

    # Eq 30: stored plastic strain
    eps_s = eps_load_p - exup

    # ---- Residual stress (Eqs 22, 35, 47a/b) ---- #
    # Scale factor Phi = ratio of average plastic strain to its maximum
    eps_Mp_approx = np.max(eps_load_p) if np.max(eps_load_p) > 0 else 1.0
    Phi = eps_Mp_approx / eps_Mp_approx  # will be updated after plastic zone computed  # noqa: F841

    eps_avg = np.zeros_like(sigma_eqe)
    sxs = np.zeros_like(sigma_eqe)

    mask_above_2sy = sigma_eqe > 2.0 * sy
    mask_between = (sigma_eqe > sy) & (sigma_eqe <= 2.0 * sy)

    # Eq 47b: deeply loaded zone
    eps_avg[mask_above_2sy] = eps_load_p[mask_above_2sy] - exup[mask_above_2sy]
    sxs[mask_above_2sy] = Sxu[mask_above_2sy]  # Eq 22

    # Eq 47a: partially loaded zone
    eps_avg[mask_between] = eps_load_p[mask_between]
    sxs[mask_between] = Sxl[mask_between] - sigma_eqe[mask_between] / 3.0  # Eq 22

    # Eq 35: residual stress in x (= y by biaxial symmetry)
    sR = (sxs * (1.0 + nu_b) / (1.0 - nu_b) - E_b * eps_avg / (1.0 - nu_b)) / 2.0

    return {
        "Z": Z,
        "Z_bar": Z_bar,
        "sigma_xe": sigma_xe,
        "sigma_ze": sigma_ze,
        "sigma_eqe": sigma_eqe,
        "eps_load_p": eps_load_p,
        "eps_unload_p": exup,
        "Sxl": Sxl,
        "Sxu": Sxu,
        "eps_s": eps_s,
        "eps_avg": eps_avg,
        "sxs": sxs,
        "sR": sR,
    }


# ---------------------------------------------------------------------------
# 5.  Plastic zone geometry  (Eqs 41-45)
# ---------------------------------------------------------------------------


def compute_plastic_zone(params: ShotPeenParams) -> Dict[str, float]:
    """Compute plastic zone geometry from Shen & Atluri Eqs 41-45.

    Returns
    -------
    dict with:
        a_p       : dent radius (Eq 44)
        r_p       : plastic zone radius (Eq 43)
        epsilon_Mp: mean plastic strain (Eq 45)
        V_p       : plastic zone volume (Eq 42)
        W_t       : total plastic strain energy (Eq 41)
    """
    p = params

    # Eq 44: dent (contact imprint) radius
    a_p = p.D * (p.rho_s * p.Vn**2 / (18.0 * p.sigma_yield)) ** (1.0 / 4.0)

    # Eq 43: plastic zone radius
    r_p = a_p * (2.0 * p.E_b / (3.0 * p.sigma_yield)) ** (1.0 / 3.0)

    # Eq 45: mean plastic strain in zone
    epsilon_Mp = (9.0 / 4.0) * a_p**4 / (p.D * r_p**3)

    # Eq 42: plastic zone volume (hemisphere)
    V_p = (2.0 * math.pi / 3.0) * r_p**3

    # Eq 41: total plastic strain energy
    W_t = V_p * p.sigma_yield * epsilon_Mp

    return {
        "a_p": a_p,
        "r_p": r_p,
        "epsilon_Mp": epsilon_Mp,
        "V_p": V_p,
        "W_t": W_t,
    }


# ---------------------------------------------------------------------------
# 6.  Energy balance
# ---------------------------------------------------------------------------


def compute_energy_balance(
    params: ShotPeenParams,
    contact: Dict[str, float],
    plastic: Dict[str, float],
) -> Dict[str, float]:
    """Partition the initial kinetic energy into plastic work, rebound KE, and waves.

    Model
    -----
    KE_initial = 0.5 * Ms * V²
    e  = sqrt( max(0, 1 - W_t / KE_initial) )    (coefficient of restitution)
    KE_rebound = 0.5 * Ms * (e*V)²  = e² * KE_initial
    W_wave     = KE_initial - W_t - KE_rebound    (elastic wave / heat)

    Returns
    -------
    dict: KE_initial, W_plastic, KE_rebound, W_wave, e (COR)
    """
    KE_initial = 0.5 * params.Ms * params.V**2
    W_t = plastic["W_t"]

    ratio = min(W_t / KE_initial, 1.0) if KE_initial > 0 else 0.0
    e = math.sqrt(max(0.0, 1.0 - ratio))

    KE_rebound = e**2 * KE_initial
    W_wave = max(0.0, KE_initial - W_t - KE_rebound)

    return {
        "KE_initial": KE_initial,
        "W_plastic": W_t,
        "KE_rebound": KE_rebound,
        "W_wave": W_wave,
        "e": e,
        "COR": e,
    }


# ---------------------------------------------------------------------------
# 7.  Map displacements to mesh nodes
# ---------------------------------------------------------------------------


def map_displacements(
    mesh: Dict[str, np.ndarray],
    contact: Dict[str, float],
    plastic: Dict[str, float],
    params: ShotPeenParams,
    impact_center: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute residual nodal displacements from the analytical impact model.

    Physical model
    --------------
    After the shot rebounds, the permanent surface deformation is a spherical dent:

        Permanent indentation depth:  δ_p = a_p² / (2 R)
        Within plastic zone (r ≤ a_p):
            uz(r) = -δ_p * (1 - (r/a_p)²)          [paraboloid dent]
        Transition region (a_p < r ≤ r_p):
            uz(r) = -δ_p * (a_p/r)² * exp(-((r-a_p)/a_p))   [decaying]
        Outside plastic zone (r > r_p):
            uz(r) ≈ 0

    Radial (lateral) displacement from plastic bulge:
        ur(r) = (2/3) * δ_p * (r / r_p) * exp(-(r/r_p)²)   [outward bulge]
        Decomposed: ux = ur * cos(θ),  uy = ur * sin(θ)

    Parameters
    ----------
    mesh          : output of generate_mesh()
    contact       : output of compute_contact_params()
    plastic       : output of compute_plastic_zone()
    params        : ShotPeenParams
    impact_center : (3,) array — default uses mesh["impact_center"]

    Returns
    -------
    node_labels   : (N,) int32
    displacements : (N, 3) float32  — [ux, uy, uz]
    """
    coords = mesh["node_coords"]  # (N, 3)
    labels = mesh["node_labels"]
    ic = impact_center if impact_center is not None else mesh["impact_center"]

    a_p = plastic["a_p"]
    r_p = plastic["r_p"]
    delta_p = a_p**2 / (2.0 * params.R)  # permanent indentation depth

    dx = coords[:, 0] - ic[0]
    dy = coords[:, 1] - ic[1]
    r = np.sqrt(dx**2 + dy**2)  # radial distance

    # Surface normal displacement (uz, negative = into material)
    uz = np.zeros(len(labels), dtype=np.float64)

    mask_dent = r <= a_p
    mask_transition = (r > a_p) & (r <= r_p)

    uz[mask_dent] = -delta_p * (1.0 - (r[mask_dent] / a_p) ** 2)
    uz[mask_transition] = -delta_p * (a_p / r[mask_transition]) ** 2 * np.exp(-(r[mask_transition] - a_p) / a_p)

    # Radial bulge displacement (ur, outward)
    safe_rp = r_p if r_p > 0 else 1e-12
    ur = (2.0 / 3.0) * delta_p * (r / safe_rp) * np.exp(-((r / safe_rp) ** 2))

    # Decompose radial into x, y (handle r=0 safely)
    safe_r = np.where(r > 0, r, 1.0)
    cos_t = np.where(r > 0, dx / safe_r, 0.0)
    sin_t = np.where(r > 0, dy / safe_r, 0.0)

    ux = ur * cos_t
    uy = ur * sin_t

    displacements = np.stack([ux, uy, uz], axis=1).astype(np.float32)
    return labels, displacements


# ---------------------------------------------------------------------------
# 8.  Map stresses to mesh elements
# ---------------------------------------------------------------------------


def map_stresses(
    mesh: Dict[str, np.ndarray],
    stress_field: Dict[str, np.ndarray],
    plastic: Dict[str, float],
    params: ShotPeenParams,
    impact_center: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign residual stress components to each mesh element.

    Each element centroid is located at (x_c, y_c, z_c).
    The stress at depth |z_c| is interpolated from the residual stress depth
    profile sR(Z), then attenuated radially with a Gaussian kernel (r-falloff).

    Stress tensor output (Abaqus convention):
        S11 = S22 = σR(depth) * G(r)   — biaxial residual compression
        S33 = 0                         — through-thickness (free surface)
        S12 = 0                         — shear (normal impact, no torsion)

    Parameters
    ----------
    mesh         : output of generate_mesh()
    stress_field : output of compute_stress_field()
    plastic      : output of compute_plastic_zone()
    params       : ShotPeenParams
    impact_center: (3,) array

    Returns
    -------
    elem_labels : (E,) int32
    stresses    : (E, 4) float32  — [S11, S22, S33, S12]
    """
    coords_n = mesh["node_coords"]  # (N, 3)
    labels_n = mesh["node_labels"]
    label_to_idx = {int(lbl): i for i, lbl in enumerate(labels_n)}

    connectivity = mesh["element_connectivity"]  # (E, 4) or (E, 8)
    elem_labels = mesh["element_labels"]

    ic = impact_center if impact_center is not None else mesh["impact_center"]

    # Precompute element centroids
    n_elem = len(elem_labels)
    centroids = np.zeros((n_elem, 3), dtype=np.float64)
    for e, row in enumerate(connectivity):
        node_idxs = [label_to_idx[int(nl)] for nl in row if int(nl) in label_to_idx]
        centroids[e] = coords_n[node_idxs].mean(axis=0)

    Z_prof = stress_field["Z"]  # depth array
    sR_prof = stress_field["sR"]  # residual stress profile

    r_p = plastic["r_p"]
    a_p = plastic["a_p"]  # noqa: F841

    # Radial distance of each element centroid from impact point
    dx = centroids[:, 0] - ic[0]
    dy = centroids[:, 1] - ic[1]
    r_elem = np.sqrt(dx**2 + dy**2)

    # Depth of element centroid (positive downward)
    z_elem = np.abs(centroids[:, 2])

    # Interpolate sR at each element's depth
    sR_at_depth = np.interp(z_elem, Z_prof, sR_prof, left=sR_prof[0], right=0.0)

    # Radial Gaussian attenuation centred on plastic zone
    sigma_r = r_p / 2.0 if r_p > 0 else 1e-12
    G_r = np.exp(-(r_elem**2) / (2.0 * sigma_r**2))

    # Only apply within influence zone (r ≤ 3 r_p)
    G_r[r_elem > 3.0 * r_p] = 0.0

    S11 = (sR_at_depth * G_r).astype(np.float32)
    S22 = S11.copy()
    S33 = np.zeros(n_elem, dtype=np.float32)
    S12 = np.zeros(n_elem, dtype=np.float32)

    stresses = np.stack([S11, S22, S33, S12], axis=1)  # (E, 4)
    return elem_labels, stresses


# ---------------------------------------------------------------------------
# 9.  Main simulation runner
# ---------------------------------------------------------------------------


def run_simulation(
    params: Optional[ShotPeenParams] = None,
    output_dir: str = "./impact_sim_output",
    Lx: float = 0.005,
    Ly: float = 0.005,
    Lz: float = 0.002,
    Nx: int = 10,
    Ny: int = 10,
    Nz: int = 1,
    save_npy: bool = True,
    verbose: bool = True,
) -> Dict:
    """Run a full single-shot impact simulation.

    Orchestrates:
      mesh → contact → stress field → plastic zone → energy balance
      → displacements on mesh → stresses on mesh → (optional) save .npy

    Parameters
    ----------
    params     : ShotPeenParams (defaults to titanium / S170)
    output_dir : Directory to save .npy files (created if needed)
    Lx, Ly, Lz: Mesh dimensions (m)
    Nx, Ny, Nz: Element counts (Nz=1 → surface mesh)
    save_npy   : Whether to write .npy output files
    verbose    : Print progress and energy balance

    Returns
    -------
    results dict with sub-dicts:
        mesh, contact, stress_field, plastic, energy,
        displacements (array), stresses (array),
        node_labels, elem_labels, disp_node_labels,
        stress_elem_labels
    """
    if params is None:
        params = ShotPeenParams()

    _log = print if verbose else (lambda *a, **k: None)

    _log("=" * 60)
    _log("Impact Simulation  —  Shen & Atluri (2006)")
    _log("=" * 60)

    # ---- Mesh ----
    _log("[1/6] Generating mesh …")
    mesh = generate_mesh(Lx=Lx, Ly=Ly, Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz)
    N_nodes = len(mesh["node_labels"])
    N_elems = len(mesh["element_labels"])
    _log(f"      {N_nodes} nodes, {N_elems} elements")

    # ---- Contact ----
    _log("[2/6] Computing Hertzian contact parameters …")
    contact = compute_contact_params(params)
    _log(f"      ae = {contact['ae']*1e6:.3f} µm  |  p0 = {contact['p0']/1e9:.3f} GPa  |  F = {contact['F']:.2f} N")

    # ---- Stress field ----
    _log("[3/6] Computing through-depth stress field …")
    stress_field = compute_stress_field(contact, params)

    # ---- Plastic zone ----
    _log("[4/6] Computing plastic zone geometry …")
    plastic = compute_plastic_zone(params)
    _log(
        f"      a_p = {plastic['a_p']*1e6:.3f} µm  |  r_p = {plastic['r_p']*1e6:.3f} µm  |  W_t = {plastic['W_t']*1e6:.4f} µJ"
    )

    # Update Phi now that we know epsilon_Mp
    max_load_p = np.max(stress_field["eps_load_p"])
    if max_load_p > 0:
        Phi = plastic["epsilon_Mp"] / max_load_p
        # Re-scale eps_avg and recompute sR
        stress_field["eps_avg"] *= Phi
        nu_b, E_b = params.nu_b, params.E_b
        stress_field["sR"] = (
            stress_field["sxs"] * (1.0 + nu_b) / (1.0 - nu_b) - E_b * stress_field["eps_avg"] / (1.0 - nu_b)
        ) / 2.0

    # ---- Energy balance ----
    _log("[5/6] Computing energy balance …")
    energy = compute_energy_balance(params, contact, plastic)
    _log(f"      KE_initial = {energy['KE_initial']*1e6:.4f} µJ")
    _log(f"      W_plastic  = {energy['W_plastic']*1e6:.4f} µJ  ({100*energy['W_plastic']/energy['KE_initial']:.1f}%)")
    _log(
        f"      KE_rebound = {energy['KE_rebound']*1e6:.4f} µJ  ({100*energy['KE_rebound']/energy['KE_initial']:.1f}%)"
    )
    _log(f"      W_wave     = {energy['W_wave']*1e6:.4f} µJ      ({100*energy['W_wave']/energy['KE_initial']:.1f}%)")
    _log(f"      COR (e)    = {energy['e']:.4f}")

    # ---- Map to mesh ----
    _log("[6/6] Mapping fields onto mesh …")
    disp_node_labels, displacements = map_displacements(mesh, contact, plastic, params)
    stress_elem_labels, stresses = map_stresses(mesh, stress_field, plastic, params)

    # ---- Save .npy files ----
    if save_npy:
        os.makedirs(output_dir, exist_ok=True)
        _log(f"      Saving .npy files to: {output_dir}")

        np.save(os.path.join(output_dir, "node_labels.npy"), mesh["node_labels"])
        np.save(os.path.join(output_dir, "node_coords.npy"), mesh["node_coords"])
        np.save(os.path.join(output_dir, "element_labels.npy"), mesh["element_labels"])
        np.save(os.path.join(output_dir, "element_connectivity.npy"), mesh["element_connectivity"])
        np.save(os.path.join(output_dir, "disp_node_labels.npy"), disp_node_labels)
        np.save(os.path.join(output_dir, "displacements.npy"), displacements)
        np.save(os.path.join(output_dir, "stress_element_labels.npy"), stress_elem_labels)
        np.save(os.path.join(output_dir, "stresses.npy"), stresses)

        # Also save depth profile for post-processing
        np.save(
            os.path.join(output_dir, "sR_depth_profile.npy"), np.stack([stress_field["Z"], stress_field["sR"]], axis=1)
        )
        np.save(
            os.path.join(output_dir, "sigma_eqe_profile.npy"),
            np.stack([stress_field["Z_bar"], stress_field["sigma_eqe"]], axis=1),
        )

        # Energy balance as a simple CSV-style text file
        energy_path = os.path.join(output_dir, "energy_balance.txt")
        with open(energy_path, "w") as fh:
            for k, v in energy.items():
                fh.write(f"{k}: {v}\n")
            for k, v in plastic.items():
                fh.write(f"{k}: {v}\n")
            for k, v in contact.items():
                fh.write(f"{k}: {v}\n")

        _log("      Done.")

    _log("=" * 60)

    return {
        "params": params,
        "mesh": mesh,
        "contact": contact,
        "stress_field": stress_field,
        "plastic": plastic,
        "energy": energy,
        "node_labels": mesh["node_labels"],
        "elem_labels": mesh["element_labels"],
        "disp_node_labels": disp_node_labels,
        "displacements": displacements,
        "stress_elem_labels": stress_elem_labels,
        "stresses": stresses,
    }


# ---------------------------------------------------------------------------
# 10.  Stand-alone plotting of the residual stress profile
# ---------------------------------------------------------------------------


def plot_residual_stress(
    results: Dict,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Plot the residual stress depth profile and energy bar chart.

    Parameters
    ----------
    results   : dict returned by run_simulation()
    show      : Call plt.show() at the end
    save_path : If provided, save the figure to this path

    Produces a 3-panel figure:
        [0] σR (residual stress) vs Z̄ = z/ae
        [1] Average plastic strain vs Z̄
        [2] Energy balance bar chart
    """
    import matplotlib.pyplot as plt

    sf = results["stress_field"]
    en = results["energy"]

    Z_bar = sf["Z_bar"]
    sR = sf["sR"]
    eps_avg = sf["eps_avg"]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 0: residual stress profile
    axs[0].plot(Z_bar, sR * 1e-6, color="steelblue", linewidth=1.0)
    axs[0].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axs[0].set_xlabel(r"$\bar{z} = z / a_e$")
    axs[0].set_ylabel(r"Residual Stress $\sigma_R$ (MPa)")
    axs[0].set_title("Residual Stress Depth Profile")
    axs[0].set_xlim([0, min(8, Z_bar[-1])])

    # Panel 1: average plastic strain
    axs[1].plot(Z_bar, eps_avg, color="darkorange", linewidth=1.0)
    axs[1].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axs[1].set_xlabel(r"$\bar{z} = z / a_e$")
    axs[1].set_ylabel(r"Average Plastic Strain $\bar{\varepsilon}^p$")
    axs[1].set_title("Plastic Strain Distribution")
    axs[1].set_xlim([0, min(8, Z_bar[-1])])

    # Panel 2: energy balance bar chart
    labels_e = ["KE initial", "W plastic", "KE rebound", "W wave"]
    values_e = [
        en["KE_initial"] * 1e6,
        en["W_plastic"] * 1e6,
        en["KE_rebound"] * 1e6,
        en["W_wave"] * 1e6,
    ]
    colors_e = ["royalblue", "firebrick", "seagreen", "goldenrod"]
    bars = axs[2].bar(labels_e, values_e, color=colors_e, edgecolor="k", linewidth=0.5)
    axs[2].set_ylabel("Energy (µJ)")
    axs[2].set_title(f"Energy Balance  (COR e = {en['e']:.3f})")
    axs[2].bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single shot-peen impact simulation.")
    parser.add_argument("--output", default="./impact_sim_output", help="Output directory")
    parser.add_argument("--V", type=float, default=35.9, help="Impact velocity (m/s)")
    parser.add_argument("--D", type=float, default=0.0005, help="Shot diameter (m)")
    parser.add_argument("--Nx", type=int, default=10, help="Elements in X")
    parser.add_argument("--Ny", type=int, default=10, help="Elements in Y")
    parser.add_argument("--Nz", type=int, default=1, help="Layers in Z (1=surface)")
    parser.add_argument("--plot", action="store_true", help="Show result plots")
    args = parser.parse_args()

    p = ShotPeenParams(V=args.V, D=args.D)
    res = run_simulation(
        params=p,
        output_dir=args.output,
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
    )

    if args.plot:
        plot_residual_stress(res, save_path=os.path.join(args.output, "residual_stress.png"))
