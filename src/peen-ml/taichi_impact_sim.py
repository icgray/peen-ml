"""
taichi_impact_sim.py
====================
Numerical ground-truth simulation of a single shot-peen impact using the
Moving Least Squares Material Point Method (MLS-MPM, Hu et al. SIGGRAPH 2018).

Constitutive model
------------------
Von Mises elastoplasticity with isotropic bilinear hardening, implemented via
SVD-based radial return mapping in Hencky (logarithmic) strain space.
The hardening slope H = (3/2)·c exactly mirrors the parameter in impact_sim.py
(c = 2/3·Ep where Ep is the plastic tangent modulus).

Physics
-------
  - Target plate: titanium alloy (or any elastic-plastic solid).
  - Shot: treated as a rigid sphere (kinematic boundary condition on grid).
    Steel is ~2× stiffer than titanium, so the rigid-sphere approximation
    introduces <5% error in contact radius (validated in literature).
  - Contact: naturally handled by the MPM grid — no explicit detection needed.
  - Gravity: negligible for a ~140 ns impact event (g·t² << deformation).

Output
------
All .npy files match the schema of impact_sim.py / Abaqus data,
so data_viz.py and any downstream ML pipeline can consume them directly.

Comparison
----------
After running both simulations, call compare_results() to plot side-by-side:
  - Residual stress depth profile σR(z)
  - Surface deformation uz(r)
  - Energy partitioning (KE, plastic work, rebound)
  - Equivalent plastic strain ε_p(z)

Usage
-----
    from taichi_impact_sim import MPMShotPeenSolver, ShotPeenParams, compare_results

    solver = MPMShotPeenSolver(ShotPeenParams(), arch="cpu", n_grid=48)
    solver.initialize()
    solver.run(n_steps=600, record_every=20)
    results = solver.extract_results(output_dir="./mpm_sim_output")
    solver.plot_energy_history()

    # Compare with analytical
    from impact_sim import run_simulation
    analytical = run_simulation(ShotPeenParams(), output_dir="./analytical_output",
                                 Nx=30, Ny=30, verbose=False)
    compare_results(results, analytical)

CLI
---
    python taichi_impact_sim.py --arch cpu --n_grid 48 --steps 600 --plot

References
----------
  Hu Y-T et al. (2018) "A moving least squares material point method with
    displacement discontinuity and two-way rigid body coupling."
    ACM Trans. Graph. 37, 4 (SIGGRAPH 2018).
  Shen & Atluri (2006) CMC vol. 4 no. 2 pp. 75-85.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional Taichi import — stub decorators keep the class importable even
# when taichi is not installed.  The constructor raises ImportError before
# any kernel is actually executed.
# ---------------------------------------------------------------------------
try:
    import taichi as ti

    _TAICHI_AVAILABLE = True
except ImportError:
    _TAICHI_AVAILABLE = False

    # Minimal stubs so @ti.kernel / @ti.func don't fail at class-definition time
    class _TaichiStub:  # type: ignore
        @staticmethod
        def kernel(f=None, **_kw):
            if f is not None:
                return f
            return lambda fn: fn

        @staticmethod
        def func(f=None, **_kw):
            if f is not None:
                return f
            return lambda fn: fn

        @staticmethod
        def template():
            return None

        class field:  # noqa: E501
            pass

        WARN = 0

    ti = _TaichiStub()  # type: ignore

# ---------------------------------------------------------------------------
# ShotPeenParams — reuse from impact_sim if available
# ---------------------------------------------------------------------------
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    from impact_sim import (
        ShotPeenParams,
        compute_contact_params,
        compute_plastic_zone,
        compute_stress_field,
        run_simulation as run_analytical_simulation,
    )
except ImportError:

    @dataclass
    class ShotPeenParams:  # type: ignore
        """Minimal fallback — see impact_sim.py for full documentation."""

        E_s: float = 210e9
        nu_s: float = 0.3
        D: float = 0.0005
        rho_s: float = 2000.0
        E_b: float = 113.8e9
        nu_b: float = 0.34
        sigma_yield: float = 276e6
        c: float = 3.0e9
        V: float = 35.9
        phi: float = math.pi / 2
        k: float = 0.8

        @property
        def R(self) -> float:
            return self.D / 2.0

        @property
        def Ms(self) -> float:
            return (4.0 / 3.0) * math.pi * self.R**3 * self.rho_s

        @property
        def Vn(self) -> float:
            return self.V * math.sin(self.phi)

    def run_analytical_simulation(*args, **kwargs):
        raise ImportError("impact_sim.py not found — cannot run analytical comparison.")


# ---------------------------------------------------------------------------
# MPM Solver
# ---------------------------------------------------------------------------


class MPMShotPeenSolver:
    """
    3-D MLS-MPM solver for a single shot-peen impact.

    Grid convention
    ---------------
    All coordinates are in SI (metres).
    z = 0 is the TOP surface of the target plate.
    The plate occupies z ∈ [-Lz, 0].
    The shot starts just above z = 0 and moves downward (−z direction).

    Constitutive model
    ------------------
    Neo-Hookean elastic part (Hencky log-strain formulation) + von Mises
    isotropic hardening via SVD-based radial return mapping.

    Parameters
    ----------
    params         : ShotPeenParams
    Lx, Ly, Lz    : Target plate dimensions (m)
    n_grid         : Number of grid cells along the longest dimension
    ppc            : (initial) particles per grid cell (integer approximation)
    rho_target     : Target material density (kg/m³) — default Ti alloy
    arch           : Taichi backend: "cpu", "cuda", "metal", "vulkan"
    use_f64        : Use float64 instead of float32 (slower, more accurate)
    verbose        : Print progress
    """

    # --------------------------------------------------------------------- #
    # Construction                                                           #
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        params: Optional[ShotPeenParams] = None,
        Lx: float = 0.004,
        Ly: float = 0.004,
        Lz: float = 0.002,
        n_grid: int = 48,
        ppc: int = 2,
        rho_target: float = 4500.0,  # titanium alloy
        arch: str = "cpu",
        use_f64: bool = False,
        verbose: bool = True,
    ):
        if not _TAICHI_AVAILABLE:
            raise ImportError("taichi is required for MPM simulation.\n" "Install with:  pip install taichi")

        self.params = params if params is not None else ShotPeenParams()
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.verbose = verbose
        self.rho_target = rho_target

        p = self.params

        # ---- grid ---- #
        # Uniform cell size, sized to longest dimension
        self.dx = max(Lx, Ly, Lz) / n_grid
        self.inv_dx = 1.0 / self.dx

        # Total domain includes air above surface (for shot trajectory)
        air_gap = p.R * 3.0  # space for shot approach
        Lz_total = Lz + air_gap  # z ∈ [-Lz, air_gap]

        self.nx = max(4, int(math.ceil(Lx / self.dx)) + 2)
        self.ny = max(4, int(math.ceil(Ly / self.dx)) + 2)
        self.nz = max(4, int(math.ceil(Lz_total / self.dx)) + 2)

        # Shift so target surface is at z_grid = nz_surface * dx
        self.nz_surface = int(math.ceil(Lz / self.dx)) + 1  # grid index of z=0
        self.z_offset = self.nz_surface * self.dx  # physical z = z_grid - z_offset

        # ---- material (target) ---- #
        E, nu = p.E_b, p.nu_b
        self.mu = E / (2.0 * (1.0 + nu))
        self.la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.sigma_y0 = p.sigma_yield
        # Isotropic hardening slope: H = (3/2)*c  (c = 2/3*E_p in Shen & Atluri)
        self.H_hard = 1.5 * p.c

        # ---- particles ---- #
        ppc_1d = max(1, int(round(ppc ** (1.0 / 3.0))))
        self.ppc_1d = ppc_1d
        n_plate_x = int(math.floor(Lx / self.dx)) * ppc_1d
        n_plate_y = int(math.floor(Ly / self.dx)) * ppc_1d
        n_plate_z = int(math.floor(Lz / self.dx)) * ppc_1d
        self.n_particles = n_plate_x * n_plate_y * n_plate_z
        self.p_vol = (self.dx / ppc_1d) ** 3
        self.p_mass_val = rho_target * self.p_vol

        # ---- timestep (CFL) ---- #
        c_p = math.sqrt((self.la + 2.0 * self.mu) / rho_target)
        self.dt = 0.3 * self.dx / (c_p + p.V)

        # ---- shot state (tracked in Python, enforced as kinematic BC) ---- #
        # Shot centre: start just above surface
        self.shot_center = np.array([Lx / 2.0, Ly / 2.0, p.R + 1e-6], dtype=np.float64)
        self.shot_vel = np.array([0.0, 0.0, -p.Vn], dtype=np.float64)
        self.shot_mass = p.Ms
        self.shot_radius = p.R

        # ---- result storage ---- #
        self.time_hist: list = []
        self.ke_target_hist: list = []
        self.ke_shot_hist: list = []
        self.plastic_diss_hist: list = []
        self.shot_vel_z_hist: list = []
        self.impulse_z_hist: list = []

        self._initialized = False

        # ---- initialize Taichi ---- #
        dtype = ti.f64 if use_f64 else ti.f32
        try:
            ti.init(arch=getattr(ti, arch), default_fp=dtype, log_level=ti.WARN)
        except Exception:
            ti.init(arch=ti.cpu, default_fp=dtype, log_level=ti.WARN)

        # ---- allocate Taichi fields ---- #
        self._alloc_fields(dtype)

        if verbose:
            self._print_setup()

    # --------------------------------------------------------------------- #
    # Field allocation                                                       #
    # --------------------------------------------------------------------- #

    def _alloc_fields(self, dtype):
        N = self.n_particles
        nx, ny, nz = self.nx, self.ny, self.nz

        # --- particle fields ---
        self.x = ti.Vector.field(3, dtype=dtype, shape=N)  # position
        self.v = ti.Vector.field(3, dtype=dtype, shape=N)  # velocity
        self.C = ti.Matrix.field(3, 3, dtype=dtype, shape=N)  # APIC affine
        self.F = ti.Matrix.field(3, 3, dtype=dtype, shape=N)  # deformation grad
        self.Jp = ti.field(dtype=dtype, shape=N)  # equiv. plastic strain
        self.sigma = ti.Matrix.field(3, 3, dtype=dtype, shape=N)  # Cauchy stress (output)

        # --- grid fields ---
        self.grid_v = ti.Vector.field(3, dtype=dtype, shape=(nx, ny, nz))
        self.grid_m = ti.field(dtype=dtype, shape=(nx, ny, nz))
        # Impulse exchanged with rigid sphere (atomic accumulation)
        self.grid_imp = ti.Vector.field(3, dtype=dtype, shape=(nx, ny, nz))

        # --- scalars (accumulated in kernels) ---
        self.ke_sum = ti.field(dtype=dtype, shape=())
        self.imp_sum = ti.Vector.field(3, dtype=dtype, shape=())

        # --- impulse accumulation fields (allocated here, before any kernel is compiled)
        # In Taichi 1.7.x, ALL fields must be placed before the first kernel compiles.
        # run() previously allocated these lazily, which raised a FieldsBuilder error.
        self.imp_x_field = ti.field(dtype=dtype, shape=())
        self.imp_y_field = ti.field(dtype=dtype, shape=())
        self.imp_z_field = ti.field(dtype=dtype, shape=())

        # --- shot rigid body (as Taichi fields for kernel access) ---
        self.shot_c_ti = ti.Vector.field(3, dtype=dtype, shape=())
        self.shot_v_ti = ti.Vector.field(3, dtype=dtype, shape=())
        self.shot_r_ti = ti.field(dtype=dtype, shape=())

    # --------------------------------------------------------------------- #
    # Initialise particles                                                   #
    # --------------------------------------------------------------------- #

    def initialize(self):
        """Populate the target plate with evenly spaced particles and reset state."""
        self._init_particles_kernel()
        self._sync_shot_to_ti()
        self._initialized = True
        if self.verbose:
            print("[MPM] Particles initialized.")

    @ti.kernel
    def _init_particles_kernel(self):
        """Fill the target plate with a regular particle lattice."""
        Lx = float(self.Lx)
        Ly = float(self.Ly)
        Lz = float(self.Lz)
        zo = float(self.z_offset)  # z_offset: z_physical = z_grid - z_offset

        dx_p = Lx / float(self.n_particles ** (1.0 / 3.0) + 1e-9)
        # Use a grid-aligned lattice based on particle count
        # Total particles = nx_p * ny_p * nz_p
        npx = ti.static(int(self.n_particles ** (1.0 / 3.0) + 0.5))
        for p in self.x:
            # 3D index decomposition
            pz = p % npx
            py = (p // npx) % npx
            px = p // (npx * npx)

            # Particle physical coordinates (inside plate: z ∈ [-Lz, 0])
            frac_x = (float(px) + 0.5) / float(npx)
            frac_y = (float(py) + 0.5) / float(npx)
            frac_z = (float(pz) + 0.5) / float(npx)

            xi = frac_x * Lx + self.dx * 0.5
            yi = frac_y * Ly + self.dx * 0.5
            zi = -frac_z * Lz + zo  # z_grid coords: surface at zo, plate goes down

            self.x[p] = [xi, yi, zi]
            self.v[p] = [0.0, 0.0, 0.0]
            self.C[p] = ti.Matrix.zero(float, 3, 3)
            self.F[p] = ti.Matrix.identity(float, 3)
            self.Jp[p] = 0.0
            self.sigma[p] = ti.Matrix.zero(float, 3, 3)

    # --------------------------------------------------------------------- #
    # Constitutive model (von Mises, Hencky strain, radial return)           #
    # --------------------------------------------------------------------- #

    @ti.func
    def _kirchhoff_stress(self, F_el):
        """
        Compute Kirchhoff stress for a given elastic deformation gradient F_el
        using the Hencky (log-strain) linear elasticity:
            τ = 2μ·ε_H + λ·tr(ε_H)·I   (Kirchhoff)
        where ε_H = log(Σ) is the tensor of log singular values.
        Returns: (tau, U, sig_vec, Vt) where sig_vec are the singular values.
        """
        U, sig_vec, Vt = ti.svd(F_el, float)

        # Log strains (diagonal; sig_vec[i,i] are singular values)
        log_s = ti.Vector(
            [
                ti.log(sig_vec[0, 0]),
                ti.log(sig_vec[1, 1]),
                ti.log(sig_vec[2, 2]),
            ]
        )

        tr_log = log_s[0] + log_s[1] + log_s[2]
        lame_la = float(self.la)
        lame_mu = float(self.mu)

        # Kirchhoff principal stresses
        tau_p = ti.Vector(
            [
                2.0 * lame_mu * log_s[0] + lame_la * tr_log,
                2.0 * lame_mu * log_s[1] + lame_la * tr_log,
                2.0 * lame_mu * log_s[2] + lame_la * tr_log,
            ]
        )

        # Assemble full Kirchhoff stress
        tau = ti.Matrix.zero(float, 3, 3)
        tau[0, 0] = tau_p[0]
        tau[1, 1] = tau_p[1]
        tau[2, 2] = tau_p[2]
        # Rotate back to world frame:  τ_world = U · τ_principal · U^T
        tau_world = U @ tau @ U.transpose()

        return tau_world, U, sig_vec, Vt, tau_p

    @ti.func
    def _von_mises_return(self, tau_trial_p, Jp_n):
        """
        Radial return mapping for von Mises + isotropic bilinear hardening
        in the principal Kirchhoff stress space.

        Parameters
        ----------
        tau_trial_p : ti.Vector(3) — principal Kirchhoff trial stresses
        Jp_n        : equivalent plastic strain at step n

        Returns
        -------
        tau_corrected : ti.Vector(3) — corrected principal stresses
        Jp_new        : updated equivalent plastic strain
        """
        sy0 = float(self.sigma_y0)
        H = float(self.H_hard)
        mu = float(self.mu)

        # Hydrostatic part (unchanged by plastic flow)
        p_hyd = (tau_trial_p[0] + tau_trial_p[1] + tau_trial_p[2]) / 3.0

        # Deviatoric (in principal space, diagonal)
        s_dev = ti.Vector(
            [
                tau_trial_p[0] - p_hyd,
                tau_trial_p[1] - p_hyd,
                tau_trial_p[2] - p_hyd,
            ]
        )

        # Frobenius norm of deviatoric (diagonal 3×3 so = vector norm)
        norm_s = ti.sqrt(s_dev[0] ** 2 + s_dev[1] ** 2 + s_dev[2] ** 2)

        # Von Mises yield function  f = ||s|| - sqrt(2/3)·σ_y_eff
        sy_eff = sy0 + H * Jp_n
        f_trial = norm_s - ti.sqrt(2.0 / 3.0) * sy_eff

        tau_corrected = tau_trial_p
        Jp_new = Jp_n

        if f_trial > 0.0 and norm_s > 1e-12:
            # Plastic multiplier Δγ from consistency condition
            # f_trial - 2μ·Δγ - sqrt(2/3)·H·sqrt(2/3)·Δγ = 0
            # Δγ = f_trial / (2μ + (2/3)·H)
            delta_gamma = f_trial / (2.0 * mu + (2.0 / 3.0) * H)

            # Radial return: scale down the deviatoric part
            scale = 1.0 - 2.0 * mu * delta_gamma / norm_s
            s_corrected = ti.Vector(
                [
                    s_dev[0] * scale,
                    s_dev[1] * scale,
                    s_dev[2] * scale,
                ]
            )

            tau_corrected = ti.Vector(
                [
                    s_corrected[0] + p_hyd,
                    s_corrected[1] + p_hyd,
                    s_corrected[2] + p_hyd,
                ]
            )

            # Update equivalent plastic strain  Δε_p_eq = sqrt(2/3)·Δγ
            Jp_new = Jp_n + ti.sqrt(2.0 / 3.0) * delta_gamma

        return tau_corrected, Jp_new

    # --------------------------------------------------------------------- #
    # MPM kernels: P→G, grid ops, G→P                                       #
    # --------------------------------------------------------------------- #

    @ti.kernel
    def _p2g(self):
        """Particle-to-Grid: transfer mass and momentum (MLS-MPM, quadratic B-spline)."""
        inv_dx = float(self.inv_dx)

        # Clear grid
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_v[I] = [0.0, 0.0, 0.0]
            self.grid_imp[I] = [0.0, 0.0, 0.0]

        for p in self.x:
            # Grid base index for quadratic stencil
            Xp = self.x[p] * inv_dx
            base = (Xp - 0.5).cast(int)  # lower-left cell (vector cast, not Python int())
            fx = Xp - base.cast(float)  # fractional offset (vector cast, not Python float())

            # Quadratic B-spline weights
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]

            # Affine momentum matrix D^{-1} = 4·inv_dx² (for quadratic spline)
            stress_p = self.sigma[p]
            affine = (
                -4.0 * float(self.dt) * float(self.p_vol) * inv_dx * inv_dx * stress_p
                + float(self.p_mass_val) * self.C[p]
            )

            for di, dj, dk in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([di, dj, dk])
                dpos = (offset.cast(float) - fx) / inv_dx  # vector from particle to node
                weight = w[di][0] * w[dj][1] * w[dk][2]
                node = base + offset

                self.grid_m[node] += weight * float(self.p_mass_val)
                self.grid_v[node] += weight * (float(self.p_mass_val) * self.v[p] + affine @ dpos)

    @ti.kernel
    def _grid_ops(
        self,
        shot_cx: float,
        shot_cy: float,
        shot_cz: float,
        shot_vx: float,
        shot_vy: float,
        shot_vz: float,
        shot_r: float,
        imp_x_out: ti.template(),
        imp_y_out: ti.template(),
        imp_z_out: ti.template(),
    ):
        """
        Grid operations:
          1. Normalise grid momentum → velocity.
          2. Apply rigid-sphere kinematic constraint (shot boundary condition).
          3. Apply plate boundary conditions (fixed bottom, free surface).
          4. Accumulate impulse on the rigid shot.
        """
        dx = float(self.dx)
        dt = float(self.dt)
        zo = float(self.z_offset)

        # Reset impulse accumulators at kernel start (Python locals can't be used
        # with ti.atomic_add — must be Taichi field elements)
        imp_x_out[None] = 0.0
        imp_y_out[None] = 0.0
        imp_z_out[None] = 0.0

        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 1e-20:
                self.grid_v[I] = self.grid_v[I] / self.grid_m[I]

            # Physical position of this grid node
            # Use ti.cast() — Python float() is unreliable on Taichi grouped-index scalars
            ix = ti.cast(I[0], float) * dx
            iy = ti.cast(I[1], float) * dx
            iz = ti.cast(I[2], float) * dx  # z_grid coordinate
            iz_phys = iz - zo  # z_physical (< 0 inside plate)

            v_node = self.grid_v[I]

            # ---- Rigid sphere BC ----
            sx, sy, sz_phys = shot_cx, shot_cy, shot_cz
            dix = ix - sx
            diy = iy - sy
            diz = iz_phys - sz_phys
            dist = ti.sqrt(dix**2 + diy**2 + diz**2)

            if dist < shot_r and dist > 1e-12:
                # Normal outward from sphere
                nx = dix / dist
                ny = diy / dist
                nz = diz / dist

                # Relative velocity of material node w.r.t. shot
                rv_n = (v_node[0] - shot_vx) * nx + (v_node[1] - shot_vy) * ny + (v_node[2] - shot_vz) * nz

                # Prevent interpenetration: if node approaching sphere, push out
                if rv_n < 0.0:
                    # Impulse: bring node to sphere surface velocity in normal direction
                    dv_x = -rv_n * nx
                    dv_y = -rv_n * ny
                    dv_z = -rv_n * nz
                    v_node = ti.Vector(
                        [
                            v_node[0] + dv_x,
                            v_node[1] + dv_y,
                            v_node[2] + dv_z,
                        ]
                    )
                    # Reaction impulse on shot (Newton's 3rd law)
                    # Must use ti.atomic_add on Taichi field elements, not Python locals
                    dm = self.grid_m[I]
                    ti.atomic_add(imp_x_out[None], -dm * dv_x)
                    ti.atomic_add(imp_y_out[None], -dm * dv_y)
                    ti.atomic_add(imp_z_out[None], -dm * dv_z)

                self.grid_v[I] = v_node

            # ---- Plate BCs ----
            # Fixed bottom (z_phys ~ -Lz)
            if iz_phys < -float(self.Lz) + dx:
                self.grid_v[I] = [0.0, 0.0, 0.0]

            # Periodic-ish: clamp x/y boundaries to zero velocity
            if I[0] < 2 or I[0] >= self.nx - 2:
                self.grid_v[I][0] = 0.0
            if I[1] < 2 or I[1] >= self.ny - 2:
                self.grid_v[I][1] = 0.0

    @ti.kernel
    def _g2p(self):
        """
        Grid-to-Particle:
          1. Interpolate grid velocity back to particles.
          2. Update particle positions.
          3. Update deformation gradient F.
          4. Compute new stress via elastoplastic constitutive update.
        """
        inv_dx = float(self.inv_dx)
        dt = float(self.dt)

        for p in self.x:
            Xp = self.x[p] * inv_dx
            base = (Xp - 0.5).cast(int)  # vector cast — Python int() doesn't work on ti.Vector
            fx = Xp - base.cast(float)  # vector cast — Python float() doesn't work on ti.Vector

            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]

            new_v = ti.Vector.zero(float, 3)
            new_C = ti.Matrix.zero(float, 3, 3)

            for di, dj, dk in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([di, dj, dk])
                dpos = (offset.cast(float) - fx) / inv_dx  # vector cast — float(ti.Vector) invalid
                weight = w[di][0] * w[dj][1] * w[dk][2]
                node = base + offset
                gv = self.grid_v[node]

                new_v += weight * gv
                new_C += 4.0 * inv_dx * weight * gv.outer_product(dpos)

            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += dt * new_v

            # ---- Update deformation gradient ----
            F_new = (ti.Matrix.identity(float, 3) + dt * new_C) @ self.F[p]
            self.F[p] = F_new

            # ---- Elastoplastic constitutive update ----
            tau_world, U, sig_vec, Vt, tau_principal = self._kirchhoff_stress(F_new)

            # Return mapping in principal stress space
            tau_ret, Jp_new = self._von_mises_return(tau_principal, self.Jp[p])
            self.Jp[p] = Jp_new

            # Reconstruct corrected Kirchhoff stress in world frame
            tau_corrected_diag = ti.Matrix.zero(float, 3, 3)
            tau_corrected_diag[0, 0] = tau_ret[0]
            tau_corrected_diag[1, 1] = tau_ret[1]
            tau_corrected_diag[2, 2] = tau_ret[2]
            tau_corrected_world = U @ tau_corrected_diag @ U.transpose()

            # Cauchy stress = τ / J  (J = det F)
            J = F_new.determinant()
            if J > 1e-12:
                self.sigma[p] = tau_corrected_world / J
            else:
                self.sigma[p] = tau_corrected_world

            # Update F to account for plastic correction (plastic strain removed)
            # Reconstruct corrected singular values from corrected Hencky strains
            log_s_trial = ti.Vector(
                [
                    ti.log(sig_vec[0, 0]),
                    ti.log(sig_vec[1, 1]),
                    ti.log(sig_vec[2, 2]),
                ]
            )
            p_hyd = (tau_principal[0] + tau_principal[1] + tau_principal[2]) / 3.0
            s_trial = ti.Vector(
                [
                    tau_principal[0] - p_hyd,
                    tau_principal[1] - p_hyd,
                    tau_principal[2] - p_hyd,
                ]
            )
            norm_s = ti.sqrt(s_trial[0] ** 2 + s_trial[1] ** 2 + s_trial[2] ** 2)

            if norm_s > 1e-12:
                mu = float(self.mu)
                # Plastic strain increment in log-strain space
                delta_eps_p = (tau_principal - tau_ret) / (2.0 * mu)
                # Corrected log singular values
                log_s_corrected = log_s_trial - delta_eps_p
                # Corrected singular value matrix
                sig_corrected = ti.Matrix.zero(float, 3, 3)
                sig_corrected[0, 0] = ti.exp(log_s_corrected[0])
                sig_corrected[1, 1] = ti.exp(log_s_corrected[1])
                sig_corrected[2, 2] = ti.exp(log_s_corrected[2])
                # Reconstruct F_elastic (plastic strain eliminated)
                self.F[p] = U @ sig_corrected @ Vt

    @ti.kernel
    def _accumulate_ke(self) -> float:
        """Compute total kinetic energy of all target particles."""
        ke = 0.0
        for p in self.v:
            v = self.v[p]
            ke += 0.5 * float(self.p_mass_val) * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        return ke

    # --------------------------------------------------------------------- #
    # Simulation loop helpers                                                #
    # --------------------------------------------------------------------- #

    def _sync_shot_to_ti(self):
        """Copy Python shot state → Taichi fields for kernel access."""
        self.shot_c_ti[None] = self.shot_center.tolist()
        self.shot_v_ti[None] = self.shot_vel.tolist()
        self.shot_r_ti[None] = self.shot_radius

    def _print_setup(self):
        p = self.params
        c_p = math.sqrt((self.la + 2.0 * self.mu) / self.rho_target)
        print("=" * 58)
        print("MPM Shot-Peen Simulation")
        print("=" * 58)
        print(f"  Domain : {self.Lx*1e3:.1f} × {self.Ly*1e3:.1f} × {self.Lz*1e3:.1f} mm")
        print(f"  Grid   : {self.nx} × {self.ny} × {self.nz}  (dx = {self.dx*1e6:.1f} µm)")
        print(f"  Particles: {self.n_particles:,}")
        print(f"  dt     : {self.dt*1e9:.2f} ns  |  c_p = {c_p:.0f} m/s")
        print(f"  Shot   : D={p.D*1e3:.2f} mm  V={p.V:.1f} m/s  Vn={p.Vn:.1f} m/s")
        print(f"  σ_y    : {p.sigma_yield/1e6:.0f} MPa  H = {self.H_hard/1e9:.2f} GPa")
        print("=" * 58)

    # --------------------------------------------------------------------- #
    # Main simulation loop                                                   #
    # --------------------------------------------------------------------- #

    def run(self, n_steps: int = 600, record_every: int = 20):
        """
        Advance the simulation for n_steps timesteps.

        Parameters
        ----------
        n_steps      : Total number of timesteps to run
        record_every : Record energy / shot state every N steps
        """
        if not self._initialized:
            self.initialize()

        # Impulse accumulation fields (pre-allocated in _alloc_fields to avoid
        # Taichi FieldsBuilder error when allocating after kernel compilation)
        imp_x_field = self.imp_x_field
        imp_y_field = self.imp_y_field
        imp_z_field = self.imp_z_field

        impact_occurred = False
        total_impulse_z = 0.0

        if self.verbose:
            print(f"[MPM] Running {n_steps} steps × dt={self.dt*1e9:.2f} ns …")

        for step in range(n_steps):
            t = step * self.dt

            # Update shot position
            self.shot_center += self.shot_vel * self.dt

            # ---- MPM substep ----
            self._p2g()
            self._grid_ops(
                float(self.shot_center[0]),
                float(self.shot_center[1]),
                float(self.shot_center[2]),
                float(self.shot_vel[0]),
                float(self.shot_vel[1]),
                float(self.shot_vel[2]),
                float(self.shot_radius),
                imp_x_field,
                imp_y_field,
                imp_z_field,
            )
            self._g2p()

            # ---- Update shot velocity from impulse ---- #
            imp = np.array(
                [
                    float(imp_x_field[None]),
                    float(imp_y_field[None]),
                    float(imp_z_field[None]),
                ]
            )
            self.shot_vel += imp / self.shot_mass
            if abs(imp[2]) > 1e-20:
                impact_occurred = True
                total_impulse_z += imp[2]

            # ---- Record ----
            if step % record_every == 0:
                ke_t = float(self._accumulate_ke())
                ke_s = 0.5 * self.shot_mass * float(np.dot(self.shot_vel, self.shot_vel))

                self.time_hist.append(t)
                self.ke_target_hist.append(ke_t)
                self.ke_shot_hist.append(ke_s)
                self.shot_vel_z_hist.append(float(self.shot_vel[2]))
                self.impulse_z_hist.append(total_impulse_z)

                # Check if shot has rebounded and exited
                if impact_occurred and self.shot_vel[2] > 0 and self.shot_center[2] > self.shot_radius * 2.0:
                    if self.verbose:
                        print(f"  [t={t*1e9:.0f} ns] Shot rebounded. Stopping early.")
                    break

            if self.verbose and step % 100 == 0:
                ke_t = self.ke_target_hist[-1] if self.ke_target_hist else 0.0
                print(
                    f"  step {step:4d}/{n_steps}  t={t*1e9:.1f} ns  "
                    f"vz_shot={self.shot_vel[2]:.2f} m/s  "
                    f"KE_target={ke_t*1e6:.3f} µJ"
                )

        if self.verbose:
            v_rebound = float(self.shot_vel[2])
            e_cor = abs(v_rebound / self.params.Vn) if self.params.Vn > 0 else 0.0
            print(f"\n[MPM] Done.  v_rebound = {v_rebound:.3f} m/s  COR = {e_cor:.3f}")

    # --------------------------------------------------------------------- #
    # Result extraction                                                      #
    # --------------------------------------------------------------------- #

    def extract_results(
        self,
        output_dir: str = "./mpm_output",
        Nx_out: int = 20,
        Ny_out: int = 20,
        save_npy: bool = True,
    ) -> Dict:
        """
        Interpolate particle data onto a structured surface mesh and save .npy files.

        The output matches the schema of impact_sim.py so data_viz.py can
        consume both analytical and MPM results without modification.

        Parameters
        ----------
        output_dir : where to save .npy files
        Nx_out, Ny_out : output mesh resolution
        save_npy   : whether to write files

        Returns
        -------
        results dict with: node_coords, displacements, stresses, energy, particles
        """
        if self.verbose:
            print("[MPM] Extracting results …")

        # ---- Read particle arrays from Taichi ----
        x_np = self.x.to_numpy()  # (N, 3)  positions in grid coords
        v_np = self.v.to_numpy()  # (N, 3)  velocities
        sigma_np = self.sigma.to_numpy()  # (N, 3, 3) Cauchy stress
        Jp_np = self.Jp.to_numpy()  # (N,)   equiv. plastic strain
        F_np = self.F.to_numpy()  # (N, 3, 3) deformation gradient

        # Convert z to physical coordinates  z_phys = z_grid - z_offset
        x_phys = x_np.copy()
        x_phys[:, 2] -= self.z_offset

        # ---- Surface mesh (z_phys ~ 0) ----
        # Extract only surface particles (within dx of surface)
        tol = self.dx * 1.5
        surface_mask = np.abs(x_phys[:, 2]) < tol
        x_surf = x_phys[surface_mask]  # (Ns, 3)
        v_surf = v_np[surface_mask]
        sigma_surf = sigma_np[surface_mask]  # (Ns, 3, 3)

        # ---- Build structured output mesh ----
        xs = np.linspace(0.0, self.Lx, Nx_out + 1)
        ys = np.linspace(0.0, self.Ly, Ny_out + 1)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        node_coords = np.stack([xx.ravel(), yy.ravel(), np.zeros(len(xx.ravel()))], axis=1)
        node_coords = node_coords.astype(np.float32)
        node_labels = np.arange(1, len(node_coords) + 1, dtype=np.int32)

        # ---- Interpolate particle data to output mesh nodes ----
        # Use inverse-distance weighting from surface particles
        displacements = np.zeros((len(node_coords), 3), dtype=np.float32)
        stresses_nodes = np.zeros((len(node_coords), 6), dtype=np.float32)  # S11,S22,S33,S12,S13,S23

        if len(x_surf) > 0:
            sigma_idw = self.dx * 2.0  # IDW bandwidth
            for ni, nc in enumerate(node_coords):
                dx_vec = x_surf[:, :2] - nc[:2]
                dist_sq = (dx_vec**2).sum(axis=1)
                w = np.exp(-dist_sq / (2.0 * sigma_idw**2))
                w_sum = w.sum()
                if w_sum > 1e-20:
                    w_norm = w / w_sum
                    # Displacement = final position - initial position (x-x0)
                    # Approximate initial z=0, so uz = z_phys_final
                    u_interp = (w_norm[:, None] * v_surf).sum(axis=0)  # velocity proxy
                    # Surface displacement approximated from particle z positions
                    uz_interp = (w_norm * x_surf[:, 2]).sum()
                    ux_interp = 0.0
                    uy_interp = 0.0
                    displacements[ni] = [ux_interp, uy_interp, uz_interp]
                    # Stress interpolation
                    for ci, (ii, jj) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]):
                        stresses_nodes[ni, ci] = (w_norm * sigma_surf[:, ii, jj]).sum()

        # ---- Residual stress depth profile ----
        # Group particles by depth in a column near the impact point
        ic_x, ic_y = self.Lx / 2.0, self.Ly / 2.0
        col_radius = self.dx * 3.0
        col_mask = np.sqrt((x_phys[:, 0] - ic_x) ** 2 + (x_phys[:, 1] - ic_y) ** 2) < col_radius
        x_col = x_phys[col_mask]  # particles in column
        sig_col = sigma_np[col_mask]
        Jp_col = Jp_np[col_mask]

        sR_depth = np.zeros((0, 2))  # fallback
        eps_depth = np.zeros((0, 2))

        if len(x_col) > 5:
            z_col = np.abs(x_col[:, 2])  # depth (positive)
            sR_col = (sig_col[:, 0, 0] + sig_col[:, 1, 1]) / 2.0  # biaxial avg

            sort_idx = np.argsort(z_col)
            sR_depth = np.stack([z_col[sort_idx], sR_col[sort_idx]], axis=1)
            eps_depth = np.stack([z_col[sort_idx], Jp_col[sort_idx]], axis=1)

        # ---- Energy history ----
        KE_initial = 0.5 * self.shot_mass * self.params.V**2
        KE_rebound = 0.5 * self.shot_mass * float(np.dot(self.shot_vel, self.shot_vel))
        W_plastic = max(0.0, KE_initial - KE_rebound)
        v_reb_z = float(self.shot_vel[2])
        COR = abs(v_reb_z / self.params.Vn) if self.params.Vn > 0 else 0.0

        energy = {
            "KE_initial": KE_initial,
            "KE_rebound": KE_rebound,
            "W_plastic": W_plastic,
            "W_wave": 0.0,  # not separately tracked in MPM
            "COR": COR,
            "e": COR,
            "v_rebound_z": v_reb_z,
        }

        # ---- Build element connectivity for output mesh ----
        quads = []
        for ix in range(Nx_out):
            for iy in range(Ny_out):
                n0 = ix * (Ny_out + 1) + iy + 1
                n1 = (ix + 1) * (Ny_out + 1) + iy + 1
                n2 = (ix + 1) * (Ny_out + 1) + iy + 2
                n3 = ix * (Ny_out + 1) + iy + 2
                quads.append([n0, n1, n2, n3])
        connectivity = np.array(quads, dtype=np.int32)
        elem_labels = np.arange(1, len(quads) + 1, dtype=np.int32)

        # Element stresses (average of corner nodes)
        lbl_idx = {int(l): i for i, l in enumerate(node_labels)}
        stresses_elem = np.zeros((len(quads), 4), dtype=np.float32)
        for ei, quad in enumerate(quads):
            node_idxs = [lbl_idx[n] for n in quad]
            stresses_elem[ei] = stresses_nodes[node_idxs, :4].mean(axis=0)

        # ---- Package results ----
        results = {
            "params": self.params,
            "node_labels": node_labels,
            "node_coords": node_coords,
            "element_labels": elem_labels,
            "element_connectivity": connectivity,
            "disp_node_labels": node_labels,
            "displacements": displacements,
            "stress_elem_labels": elem_labels,
            "stresses": stresses_elem,
            "energy": energy,
            "sR_depth_profile": sR_depth,
            "eps_depth_profile": eps_depth,
            "time_hist": np.array(self.time_hist),
            "ke_target_hist": np.array(self.ke_target_hist),
            "ke_shot_hist": np.array(self.ke_shot_hist),
            "shot_vel_z_hist": np.array(self.shot_vel_z_hist),
            "all_particle_pos": x_phys,
            "all_particle_stress": sigma_np,
            "all_particle_Jp": Jp_np,
        }

        # ---- Save .npy ----
        if save_npy:
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, "node_labels.npy"), node_labels)
            np.save(os.path.join(output_dir, "node_coords.npy"), node_coords)
            np.save(os.path.join(output_dir, "element_labels.npy"), elem_labels)
            np.save(os.path.join(output_dir, "element_connectivity.npy"), connectivity)
            np.save(os.path.join(output_dir, "disp_node_labels.npy"), node_labels)
            np.save(os.path.join(output_dir, "displacements.npy"), displacements)
            np.save(os.path.join(output_dir, "stress_element_labels.npy"), elem_labels)
            np.save(os.path.join(output_dir, "stresses.npy"), stresses_elem)
            np.save(os.path.join(output_dir, "sR_depth_profile.npy"), sR_depth)
            np.save(os.path.join(output_dir, "eps_depth_profile.npy"), eps_depth)
            np.save(
                os.path.join(output_dir, "energy_history.npy"),
                (
                    np.stack(
                        [np.array(self.time_hist), np.array(self.ke_target_hist), np.array(self.ke_shot_hist)], axis=1
                    )
                    if self.time_hist
                    else np.zeros((0, 3))
                ),
            )
            with open(os.path.join(output_dir, "energy_balance.txt"), "w") as fh:
                for k, v in energy.items():
                    fh.write(f"{k}: {v}\n")
            if self.verbose:
                print(f"[MPM] Saved to: {output_dir}")

        return results

    # --------------------------------------------------------------------- #
    # Plotting                                                               #
    # --------------------------------------------------------------------- #

    def plot_energy_history(self, show: bool = True, save_path: Optional[str] = None) -> None:
        """Plot KE of target and shot vs time, and shot z-velocity."""
        import matplotlib.pyplot as plt

        t = np.array(self.time_hist) * 1e9  # ns
        ke_t = np.array(self.ke_target_hist) * 1e6  # µJ
        ke_s = np.array(self.ke_shot_hist) * 1e6
        vz = np.array(self.shot_vel_z_hist)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        axs[0].plot(t, ke_t, label="Target KE", color="steelblue")
        axs[0].plot(t, ke_s, label="Shot KE", color="firebrick", linestyle="--")
        axs[0].plot(t, ke_t + ke_s, label="Total KE", color="k", linewidth=0.5)
        axs[0].set_xlabel("Time (ns)")
        axs[0].set_ylabel("Kinetic Energy (µJ)")
        axs[0].set_title("Energy Transfer During Impact")
        axs[0].legend()

        axs[1].plot(t, vz, color="darkorange")
        axs[1].axhline(0, color="k", linewidth=0.5, linestyle="--")
        axs[1].set_xlabel("Time (ns)")
        axs[1].set_ylabel("Shot v_z (m/s)")
        axs[1].set_title("Shot Velocity (−z = approach, +z = rebound)")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()


# ---------------------------------------------------------------------------
# Comparison: MPM  vs  Shen & Atluri analytical
# ---------------------------------------------------------------------------


def compare_results(
    mpm: Dict,
    analytical: Dict,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side comparison of MPM numerical simulation vs
    Shen & Atluri (2006) analytical results.

    Produces a 2×2 figure:
      [0,0] Residual stress depth profile σR(z)
      [0,1] Equivalent plastic strain depth profile
      [1,0] Energy partitioning bar chart (both methods)
      [1,1] Shot velocity history (MPM only)
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # ---- [0,0] Residual stress depth profile ----
    ax = axs[0, 0]
    if "sR_depth_profile" in mpm and len(mpm["sR_depth_profile"]) > 0:
        sR_m = mpm["sR_depth_profile"]
        ax.plot(sR_m[:, 0] * 1e6, sR_m[:, 1] / 1e6, label="MPM", color="steelblue", linewidth=1.5)

    if "stress_field" in analytical and "Z" in analytical["stress_field"]:
        sf = analytical["stress_field"]
        ax.plot(
            sf["Z"] * 1e6,
            sf["sR"] / 1e6,
            label="Shen & Atluri (analytical)",
            color="firebrick",
            linewidth=1.5,
            linestyle="--",
        )

    ax.axhline(0, color="k", linewidth=0.4, linestyle=":")
    ax.set_xlabel("Depth z (µm)")
    ax.set_ylabel("Residual Stress σR (MPa)")
    ax.set_title("Residual Stress Depth Profile")
    ax.legend()
    ax.set_xlim(left=0)

    # ---- [0,1] Plastic strain depth profile ----
    ax = axs[0, 1]
    if "eps_depth_profile" in mpm and len(mpm["eps_depth_profile"]) > 0:
        ep_m = mpm["eps_depth_profile"]
        ax.plot(ep_m[:, 0] * 1e6, ep_m[:, 1], label="MPM ε_p", color="steelblue", linewidth=1.5)

    if "stress_field" in analytical:
        sf = analytical["stress_field"]
        ax.plot(sf["Z"] * 1e6, sf["eps_avg"], label="Analytical ε̄p", color="firebrick", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Depth z (µm)")
    ax.set_ylabel("Equivalent Plastic Strain")
    ax.set_title("Plastic Strain Distribution")
    ax.legend()
    ax.set_xlim(left=0)

    # ---- [1,0] Energy partitioning ----
    ax = axs[1, 0]
    labels_e = ["KE initial", "W plastic", "KE rebound"]

    mpm_e = mpm.get("energy", {})
    ana_e = analytical.get("energy", {})

    mpm_vals = [
        mpm_e.get("KE_initial", 0) * 1e6,
        mpm_e.get("W_plastic", 0) * 1e6,
        mpm_e.get("KE_rebound", 0) * 1e6,
    ]
    ana_vals = [
        ana_e.get("KE_initial", 0) * 1e6,
        ana_e.get("W_plastic", 0) * 1e6,
        ana_e.get("KE_rebound", 0) * 1e6,
    ]

    x_pos = np.arange(len(labels_e))
    width = 0.35
    ax.bar(x_pos - width / 2, mpm_vals, width, label="MPM", color="steelblue")
    ax.bar(x_pos + width / 2, ana_vals, width, label="Analytical", color="firebrick", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_e)
    ax.set_ylabel("Energy (µJ)")
    ax.set_title("Energy Balance Comparison")
    mpm_cor = mpm_e.get("COR", float("nan"))
    ana_cor = ana_e.get("COR", float("nan"))
    ax.set_xlabel(f"MPM COR = {mpm_cor:.3f}   |   Analytical COR = {ana_cor:.3f}")
    ax.legend()

    # ---- [1,1] Shot velocity history (MPM) ----
    ax = axs[1, 1]
    if len(mpm.get("time_hist", [])) > 0:
        t = mpm["time_hist"] * 1e9
        vz = mpm["shot_vel_z_hist"]
        ax.plot(t, vz, color="darkorange", linewidth=1.5)
        ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
        ax.fill_between(t, vz, 0, where=(np.array(vz) < 0), alpha=0.15, color="firebrick", label="Approach phase")
        ax.fill_between(t, vz, 0, where=(np.array(vz) > 0), alpha=0.15, color="steelblue", label="Rebound phase")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Shot velocity v_z (m/s)")
        ax.set_title("Shot Velocity History (MPM)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No time history\n(run simulation first)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Shot Velocity History")

    fig.suptitle("MPM Numerical  vs  Shen & Atluri (2006) Analytical", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Convenience function: run everything
# ---------------------------------------------------------------------------


def run_mpm_simulation(
    params: Optional[ShotPeenParams] = None,
    output_dir: str = "./mpm_output",
    n_grid: int = 48,
    n_steps: int = 600,
    arch: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """
    End-to-end MPM simulation runner.

    Equivalent to run_simulation() in impact_sim.py but uses the full
    MPM physics solver instead of the analytical model.

    Parameters
    ----------
    params     : ShotPeenParams (defaults to Ti alloy / S170)
    output_dir : Output directory for .npy files
    n_grid     : Grid resolution (cells along longest dimension)
    n_steps    : Number of timesteps
    arch       : Taichi backend ("cpu", "cuda", "metal")
    verbose    : Print progress

    Returns
    -------
    results dict (same schema as impact_sim.run_simulation)
    """
    if params is None:
        params = ShotPeenParams()

    solver = MPMShotPeenSolver(
        params=params,
        n_grid=n_grid,
        arch=arch,
        verbose=verbose,
    )
    solver.initialize()
    solver.run(n_steps=n_steps, record_every=max(1, n_steps // 30))
    return solver.extract_results(output_dir=output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MPM shot-peen impact simulation (Taichi backend).")
    parser.add_argument("--output", default="./mpm_output", help="Output directory")
    parser.add_argument("--arch", default="cpu", help="Taichi backend: cpu / cuda / metal")
    parser.add_argument("--n_grid", type=int, default=48, help="Grid resolution")
    parser.add_argument("--steps", type=int, default=600, help="Number of timesteps")
    parser.add_argument("--V", type=float, default=35.9, help="Impact velocity (m/s)")
    parser.add_argument("--D", type=float, default=0.0005, help="Shot diameter (m)")
    parser.add_argument("--plot", action="store_true", help="Show plots after simulation")
    parser.add_argument("--compare", action="store_true", help="Also run Shen & Atluri analytical and plot comparison")
    args = parser.parse_args()

    p = ShotPeenParams(V=args.V, D=args.D)

    solver = MPMShotPeenSolver(params=p, n_grid=args.n_grid, arch=args.arch, verbose=True)
    solver.initialize()
    solver.run(n_steps=args.steps)
    mpm_results = solver.extract_results(output_dir=args.output)

    if args.plot:
        solver.plot_energy_history(save_path=os.path.join(args.output, "energy_history.png"))

    if args.compare:
        try:
            analytical = run_analytical_simulation(
                params=p,
                output_dir=os.path.join(args.output, "analytical"),
                Nx=20,
                Ny=20,
                verbose=False,
            )
            compare_results(
                mpm_results,
                analytical,
                save_path=os.path.join(args.output, "comparison.png"),
                show=args.plot,
            )
        except Exception as exc:
            print(f"[warn] Could not run analytical comparison: {exc}")
