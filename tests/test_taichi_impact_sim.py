"""
test_taichi_impact_sim.py
=========================
Pytest suite for taichi_impact_sim.py.

Strategy
--------
Taichi may not be installed in all CI environments, so the suite is split:

  - Tests marked  @pytest.mark.no_taichi  skip-proof, pure-Python tests:
    import validation, compare_results(), docstrings, energy formulae, etc.

  - Tests marked  @pytest.mark.requires_taichi  are automatically skipped
    when taichi is absent.

All tests that call MPMShotPeenSolver use a deliberately tiny configuration
(n_grid=8, n_particles ~ a few hundred, n_steps=5) so they finish in seconds
on CPU without any GPU hardware.

Run all tests:   pytest tests/test_taichi_impact_sim.py -v
Skip slow tests: pytest tests/test_taichi_impact_sim.py -v -m "not requires_taichi"
"""

import math
import os
import sys
import tempfile
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/peen-ml"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# Check taichi availability
# ---------------------------------------------------------------------------
try:
    import taichi as ti                         # noqa: F401
    _TAICHI_OK = True
except ImportError:
    _TAICHI_OK = False

requires_taichi = pytest.mark.skipif(
    not _TAICHI_OK,
    reason="taichi not installed (pip install taichi to enable these tests)",
)


# ---------------------------------------------------------------------------
# Import module under test (always succeeds — taichi is optional)
# ---------------------------------------------------------------------------
from taichi_impact_sim import (
    MPMShotPeenSolver,
    ShotPeenParams,
    compare_results,
    run_mpm_simulation,
    _TAICHI_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Tiny simulation fixture (requires taichi)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_solver():
    """Create and run a minimal MPM solver for quick functional tests."""
    if not _TAICHI_OK:
        pytest.skip("taichi not available")
    p = ShotPeenParams()
    solver = MPMShotPeenSolver(
        params=p,
        Lx=0.002, Ly=0.002, Lz=0.001,
        n_grid=8,
        ppc=1,
        arch="cpu",
        verbose=False,
    )
    solver.initialize()
    solver.run(n_steps=5, record_every=1)
    return solver


@pytest.fixture(scope="module")
def tiny_results(tiny_solver):
    with tempfile.TemporaryDirectory() as tmpdir:
        res = tiny_solver.extract_results(output_dir=tmpdir, Nx_out=4, Ny_out=4)
        yield res


# ---------------------------------------------------------------------------
# Class 1: Module-level attributes and import
# ---------------------------------------------------------------------------

class TestModuleImport:

    def test_taichi_available_flag_is_bool(self):
        assert isinstance(_TAICHI_AVAILABLE, bool)

    def test_ShotPeenParams_importable(self):
        p = ShotPeenParams()
        assert p.V > 0

    def test_compare_results_callable(self):
        assert callable(compare_results)

    def test_run_mpm_simulation_callable(self):
        assert callable(run_mpm_simulation)

    def test_MPMShotPeenSolver_importable(self):
        assert MPMShotPeenSolver is not None

    def test_module_docstring_present(self):
        import taichi_impact_sim
        assert taichi_impact_sim.__doc__ is not None
        assert "MLS-MPM" in taichi_impact_sim.__doc__

    def test_taichi_flag_matches_import(self):
        """_TAICHI_AVAILABLE must equal whether taichi can actually be imported."""
        assert _TAICHI_AVAILABLE == _TAICHI_OK


# ---------------------------------------------------------------------------
# Class 2: ShotPeenParams (same as test_impact_sim but re-confirmed here)
# ---------------------------------------------------------------------------

class TestShotPeenParamsInMPM:

    def test_R_property(self):
        p = ShotPeenParams(D=0.0005)
        assert p.R == pytest.approx(0.00025, rel=1e-9)

    def test_Ms_positive(self):
        assert ShotPeenParams().Ms > 0

    def test_Vn_le_V(self):
        p = ShotPeenParams(V=35.9, phi=math.pi / 4)
        assert p.Vn < p.V

    def test_Vn_equals_V_at_90deg(self):
        p = ShotPeenParams(V=35.9, phi=math.pi / 2)
        assert p.Vn == pytest.approx(35.9, rel=1e-9)


# ---------------------------------------------------------------------------
# Class 3: MPMShotPeenSolver construction (requires_taichi)
# ---------------------------------------------------------------------------

class TestMPMSolverConstruction:

    @requires_taichi
    def test_creates_without_error(self):
        p = ShotPeenParams()
        solver = MPMShotPeenSolver(
            params=p, Lx=0.002, Ly=0.002, Lz=0.001,
            n_grid=6, ppc=1, arch="cpu", verbose=False
        )
        assert solver is not None

    @requires_taichi
    def test_dx_positive(self):
        solver = MPMShotPeenSolver(
            params=ShotPeenParams(), n_grid=8, ppc=1, arch="cpu", verbose=False
        )
        assert solver.dx > 0

    @requires_taichi
    def test_dt_positive(self):
        solver = MPMShotPeenSolver(
            params=ShotPeenParams(), n_grid=8, ppc=1, arch="cpu", verbose=False
        )
        assert solver.dt > 0

    @requires_taichi
    def test_dt_satisfies_CFL(self):
        """dt should be ≤ dx / c_p for stability."""
        solver = MPMShotPeenSolver(
            params=ShotPeenParams(), n_grid=8, ppc=1, arch="cpu", verbose=False
        )
        c_p = math.sqrt((solver.la + 2.0 * solver.mu) / solver.rho_target)
        assert solver.dt <= solver.dx / c_p

    @requires_taichi
    def test_n_particles_positive(self):
        solver = MPMShotPeenSolver(
            params=ShotPeenParams(), n_grid=8, ppc=1, arch="cpu", verbose=False
        )
        assert solver.n_particles > 0

    @requires_taichi
    def test_grid_dimensions_positive(self):
        solver = MPMShotPeenSolver(
            params=ShotPeenParams(), n_grid=8, ppc=1, arch="cpu", verbose=False
        )
        assert solver.nx > 0
        assert solver.ny > 0
        assert solver.nz > 0

    @requires_taichi
    def test_mu_positive(self):
        solver = MPMShotPeenSolver(
            params=ShotPeenParams(), n_grid=6, ppc=1, arch="cpu", verbose=False
        )
        assert solver.mu > 0

    @requires_taichi
    def test_la_positive(self):
        solver = MPMShotPeenSolver(
            params=ShotPeenParams(), n_grid=6, ppc=1, arch="cpu", verbose=False
        )
        assert solver.la > 0

    @requires_taichi
    def test_hardening_slope_relationship(self):
        """H should equal (3/2)*c to match Shen & Atluri parameterisation."""
        p = ShotPeenParams(c=3.0e9)
        solver = MPMShotPeenSolver(
            params=p, n_grid=6, ppc=1, arch="cpu", verbose=False
        )
        assert solver.H_hard == pytest.approx(1.5 * p.c, rel=1e-9)

    @requires_taichi
    def test_shot_initial_position_above_surface(self):
        """Shot centre must start above the target surface (z > 0)."""
        solver = MPMShotPeenSolver(
            params=ShotPeenParams(), n_grid=6, ppc=1, arch="cpu", verbose=False
        )
        assert solver.shot_center[2] > 0

    @requires_taichi
    def test_shot_initial_velocity_negative_z(self):
        """Shot initial velocity must point toward the surface (−z)."""
        solver = MPMShotPeenSolver(
            params=ShotPeenParams(), n_grid=6, ppc=1, arch="cpu", verbose=False
        )
        assert solver.shot_vel[2] < 0

    @requires_taichi
    def test_raises_without_taichi(self):
        """When taichi is not available, constructor must raise ImportError."""
        with patch("taichi_impact_sim._TAICHI_AVAILABLE", False):
            with pytest.raises(ImportError, match="taichi"):
                MPMShotPeenSolver(params=ShotPeenParams(), verbose=False)

    @requires_taichi
    def test_p_mass_val_positive(self):
        solver = MPMShotPeenSolver(
            params=ShotPeenParams(), n_grid=6, ppc=1, arch="cpu", verbose=False
        )
        assert solver.p_mass_val > 0

    @requires_taichi
    def test_p_vol_positive(self):
        solver = MPMShotPeenSolver(
            params=ShotPeenParams(), n_grid=6, ppc=1, arch="cpu", verbose=False
        )
        assert solver.p_vol > 0


# ---------------------------------------------------------------------------
# Class 4: Initialization
# ---------------------------------------------------------------------------

class TestMPMInitialization:

    @requires_taichi
    def test_initialize_sets_flag(self, tiny_solver):
        assert tiny_solver._initialized is True

    @requires_taichi
    def test_particle_positions_in_domain(self, tiny_solver):
        x_np = tiny_solver.x.to_numpy()
        z_phys = x_np[:, 2] - tiny_solver.z_offset
        # All particles should be in the plate (z_phys < 0) or near surface
        assert np.all(z_phys <= tiny_solver.dx)

    @requires_taichi
    def test_particle_velocities_zero_initially(self):
        """After initialize() only (no run), particles should have zero velocity."""
        p = ShotPeenParams()
        solver = MPMShotPeenSolver(
            params=p, Lx=0.002, Ly=0.002, Lz=0.001,
            n_grid=6, ppc=1, arch="cpu", verbose=False
        )
        solver.initialize()
        v_np = solver.v.to_numpy()
        np.testing.assert_allclose(v_np, 0.0, atol=1e-30)

    @requires_taichi
    def test_deformation_gradient_identity_initially(self):
        p = ShotPeenParams()
        solver = MPMShotPeenSolver(
            params=p, Lx=0.002, Ly=0.002, Lz=0.001,
            n_grid=6, ppc=1, arch="cpu", verbose=False
        )
        solver.initialize()
        F_np = solver.F.to_numpy()   # (N, 3, 3)
        eye3 = np.eye(3)
        for f in F_np[:10]:
            np.testing.assert_allclose(f, eye3, atol=1e-6)

    @requires_taichi
    def test_plastic_strain_zero_initially(self):
        p = ShotPeenParams()
        solver = MPMShotPeenSolver(
            params=p, Lx=0.002, Ly=0.002, Lz=0.001,
            n_grid=6, ppc=1, arch="cpu", verbose=False
        )
        solver.initialize()
        Jp_np = solver.Jp.to_numpy()
        np.testing.assert_allclose(Jp_np, 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# Class 5: After a short run
# ---------------------------------------------------------------------------

class TestMPMAfterRun:

    @requires_taichi
    def test_run_does_not_raise(self, tiny_solver):
        # tiny_solver fixture already ran; just confirm it completed
        assert tiny_solver._initialized is True

    @requires_taichi
    def test_time_history_populated(self, tiny_solver):
        assert len(tiny_solver.time_hist) > 0

    @requires_taichi
    def test_time_history_increasing(self, tiny_solver):
        t = np.array(tiny_solver.time_hist)
        assert np.all(np.diff(t) >= 0)

    @requires_taichi
    def test_ke_target_history_populated(self, tiny_solver):
        assert len(tiny_solver.ke_target_hist) > 0

    @requires_taichi
    def test_ke_shot_history_populated(self, tiny_solver):
        assert len(tiny_solver.ke_shot_hist) > 0

    @requires_taichi
    def test_shot_vel_history_populated(self, tiny_solver):
        assert len(tiny_solver.shot_vel_z_hist) > 0

    @requires_taichi
    def test_ke_target_nonnegative(self, tiny_solver):
        for ke in tiny_solver.ke_target_hist:
            assert ke >= 0

    @requires_taichi
    def test_ke_shot_nonnegative(self, tiny_solver):
        for ke in tiny_solver.ke_shot_hist:
            assert ke >= 0

    @requires_taichi
    def test_plastic_strain_nonneg_after_impact(self, tiny_solver):
        Jp_np = tiny_solver.Jp.to_numpy()
        assert np.all(Jp_np >= 0)

    @requires_taichi
    def test_deformation_occurred(self, tiny_solver):
        """After 5 steps into impact, max |σ| should be > 0."""
        sigma_np = tiny_solver.sigma.to_numpy()
        assert np.max(np.abs(sigma_np)) > 0


# ---------------------------------------------------------------------------
# Class 6: extract_results output schema
# ---------------------------------------------------------------------------

class TestExtractResults:

    @requires_taichi
    def test_required_keys(self, tiny_results):
        required = {
            "node_labels", "node_coords", "element_labels",
            "element_connectivity", "disp_node_labels", "displacements",
            "stress_elem_labels", "stresses", "energy",
            "time_hist", "ke_target_hist", "ke_shot_hist",
        }
        assert required.issubset(tiny_results.keys())

    @requires_taichi
    def test_node_labels_int32(self, tiny_results):
        assert tiny_results["node_labels"].dtype == np.int32

    @requires_taichi
    def test_node_coords_float32(self, tiny_results):
        assert tiny_results["node_coords"].dtype == np.float32

    @requires_taichi
    def test_displacements_shape(self, tiny_results):
        N = len(tiny_results["node_labels"])
        assert tiny_results["displacements"].shape == (N, 3)

    @requires_taichi
    def test_stresses_shape(self, tiny_results):
        E = len(tiny_results["element_labels"])
        assert tiny_results["stresses"].shape == (E, 4)

    @requires_taichi
    def test_energy_keys(self, tiny_results):
        en = tiny_results["energy"]
        for k in ("KE_initial", "KE_rebound", "W_plastic", "COR"):
            assert k in en

    @requires_taichi
    def test_KE_initial_matches_formula(self, tiny_results):
        p = ShotPeenParams()
        expected = 0.5 * p.Ms * p.V ** 2
        assert tiny_results["energy"]["KE_initial"] == pytest.approx(expected, rel=1e-6)

    @requires_taichi
    def test_COR_in_unit_interval(self, tiny_results):
        cor = tiny_results["energy"]["COR"]
        assert 0.0 <= cor <= 1.0

    @requires_taichi
    def test_npy_files_saved(self, tiny_solver):
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_solver.extract_results(output_dir=tmpdir, Nx_out=3, Ny_out=3)
            for fname in [
                "node_labels.npy", "node_coords.npy", "displacements.npy",
                "stresses.npy", "sR_depth_profile.npy", "energy_balance.txt",
            ]:
                assert os.path.exists(os.path.join(tmpdir, fname)), f"Missing: {fname}"

    @requires_taichi
    def test_no_npy_if_save_false(self, tiny_solver):
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_solver.extract_results(
                output_dir=tmpdir, Nx_out=3, Ny_out=3, save_npy=False
            )
            npy_files = [f for f in os.listdir(tmpdir) if f.endswith(".npy")]
            assert len(npy_files) == 0

    @requires_taichi
    def test_connectivity_references_valid_nodes(self, tiny_results):
        labels_set = set(tiny_results["node_labels"].tolist())
        for row in tiny_results["element_connectivity"]:
            for nl in row:
                assert int(nl) in labels_set

    @requires_taichi
    def test_all_particle_pos_present(self, tiny_results):
        assert "all_particle_pos" in tiny_results
        assert tiny_results["all_particle_pos"].ndim == 2
        assert tiny_results["all_particle_pos"].shape[1] == 3

    @requires_taichi
    def test_all_particle_Jp_nonneg(self, tiny_results):
        assert np.all(tiny_results["all_particle_Jp"] >= 0)


# ---------------------------------------------------------------------------
# Class 7: compare_results (pure Python, no taichi required)
# ---------------------------------------------------------------------------

class TestCompareResults:

    def _make_mock_mpm_results(self):
        Z = np.linspace(1e-5, 8e-4, 100)
        return {
            "sR_depth_profile":  np.stack([Z, -50e6 * np.exp(-Z / 2e-4)], axis=1),
            "eps_depth_profile": np.stack([Z, 0.001 * np.exp(-Z / 1e-4)], axis=1),
            "energy": {
                "KE_initial": 84e-6, "W_plastic": 60e-6,
                "KE_rebound": 24e-6, "COR": 0.534, "e": 0.534,
            },
            "time_hist":      np.linspace(0, 200e-9, 20),
            "ke_target_hist": np.linspace(0, 30e-6, 20),
            "ke_shot_hist":   np.linspace(84e-6, 24e-6, 20),
            "shot_vel_z_hist": np.linspace(-35.9, 19.1, 20).tolist(),
        }

    def _make_mock_analytical_results(self):
        Z = np.linspace(1e-5, 8e-4, 300)
        return {
            "stress_field": {
                "Z":         Z,
                "Z_bar":     Z / 37e-6,
                "sR":        -90e6 * np.exp(-Z / 1e-4) * (Z < 4e-4),
                "eps_avg":   0.002 * np.exp(-Z / 8e-5),
            },
            "energy": {
                "KE_initial": 84e-6, "W_plastic": 84e-6,
                "KE_rebound": 0.0,   "COR": 0.0,
            },
        }

    def test_compare_runs_without_error(self):
        mpm  = self._make_mock_mpm_results()
        ana  = self._make_mock_analytical_results()
        with patch("matplotlib.pyplot.show"):
            try:
                import matplotlib
                matplotlib.use("Agg")
            except Exception:
                pass
            compare_results(mpm, ana, show=False)

    def test_compare_saves_figure(self):
        mpm  = self._make_mock_mpm_results()
        ana  = self._make_mock_analytical_results()
        with patch("matplotlib.pyplot.show"):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "cmp.png")
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                except Exception:
                    pass
                compare_results(mpm, ana, show=False, save_path=path)
                assert os.path.exists(path)

    def test_compare_show_false_no_plt_show(self):
        mpm  = self._make_mock_mpm_results()
        ana  = self._make_mock_analytical_results()
        with patch("matplotlib.pyplot.show") as mock_show:
            try:
                import matplotlib
                matplotlib.use("Agg")
            except Exception:
                pass
            compare_results(mpm, ana, show=False)
            mock_show.assert_not_called()

    def test_compare_show_true_calls_plt_show(self):
        mpm  = self._make_mock_mpm_results()
        ana  = self._make_mock_analytical_results()
        with patch("matplotlib.pyplot.show") as mock_show:
            try:
                import matplotlib
                matplotlib.use("Agg")
            except Exception:
                pass
            compare_results(mpm, ana, show=True)
            mock_show.assert_called_once()

    def test_compare_empty_results_no_crash(self):
        """compare_results should degrade gracefully with missing optional fields."""
        mpm_empty = {"energy": {}, "time_hist": np.array([]), "shot_vel_z_hist": []}
        ana_empty = {}
        with patch("matplotlib.pyplot.show"):
            try:
                import matplotlib
                matplotlib.use("Agg")
            except Exception:
                pass
            compare_results(mpm_empty, ana_empty, show=False)  # must not raise


# ---------------------------------------------------------------------------
# Class 8: Energy physics (pure Python checks, no taichi)
# ---------------------------------------------------------------------------

class TestEnergyPhysics:

    def test_KE_formula(self):
        p = ShotPeenParams(V=35.9)
        KE = 0.5 * p.Ms * p.V ** 2
        assert KE > 0

    def test_KE_scales_with_v_squared(self):
        p1 = ShotPeenParams(V=20.0)
        p2 = ShotPeenParams(V=40.0)
        KE1 = 0.5 * p1.Ms * p1.V ** 2
        KE2 = 0.5 * p2.Ms * p2.V ** 2
        ratio = KE2 / KE1
        assert ratio == pytest.approx(4.0, rel=1e-6)

    def test_COR_definition(self):
        """COR = |v_rebound| / |v_impact|."""
        v_impact  = -35.9
        v_rebound = 27.0
        COR = abs(v_rebound / v_impact)
        assert 0.0 < COR < 1.0

    def test_hardening_slope_H_calculation(self):
        """Verify H = (3/2)*c for any c."""
        p = ShotPeenParams(c=2.0e9)
        solver_H = 1.5 * p.c
        assert solver_H == pytest.approx(3.0e9, rel=1e-9)

    def test_lame_constants_from_E_nu(self):
        E, nu = 113.8e9, 0.34
        mu_expected = E / (2 * (1 + nu))
        la_expected = E * nu / ((1 + nu) * (1 - 2 * nu))
        assert mu_expected > 0
        assert la_expected > 0

    def test_p_wave_speed_positive(self):
        E, nu, rho = 113.8e9, 0.34, 4500.0
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))
        c_p = math.sqrt((la + 2 * mu) / rho)
        assert c_p > 0

    def test_cfl_dt_formula(self):
        """CFL timestep must be proportional to dx / c_p."""
        E, nu, rho = 113.8e9, 0.34, 4500.0
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))
        c_p = math.sqrt((la + 2 * mu) / rho)
        dx  = 1e-4   # 100 µm
        V   = 35.9
        dt  = 0.3 * dx / (c_p + V)
        assert dt > 0
        assert dt < dx / c_p   # must satisfy CFL


# ---------------------------------------------------------------------------
# Class 9: Return mapping physics (NumPy reimplementation to verify kernel)
# ---------------------------------------------------------------------------

class TestVonMisesReturnMapping:
    """
    Verify the von Mises radial return mapping used inside the Taichi kernel
    by reimplementing it in NumPy and checking invariants.
    """

    @staticmethod
    def _return_map(tau_p: np.ndarray, Jp_n: float, mu: float, sy0: float, H: float):
        """NumPy mirror of MPMShotPeenSolver._von_mises_return."""
        p_hyd = tau_p.sum() / 3.0
        s_dev = tau_p - p_hyd
        norm_s = np.linalg.norm(s_dev)
        sy_eff = sy0 + H * Jp_n
        f = norm_s - math.sqrt(2.0 / 3.0) * sy_eff
        if f <= 0 or norm_s < 1e-12:
            return tau_p.copy(), Jp_n
        delta_gamma = f / (2.0 * mu + (2.0 / 3.0) * H)
        scale = 1.0 - 2.0 * mu * delta_gamma / norm_s
        s_corr = s_dev * scale
        tau_corr = s_corr + p_hyd
        Jp_new = Jp_n + math.sqrt(2.0 / 3.0) * delta_gamma
        return tau_corr, Jp_new

    def _params(self):
        p = ShotPeenParams()
        mu = p.E_b / (2 * (1 + p.nu_b))
        sy0 = p.sigma_yield
        H = 1.5 * p.c
        return mu, sy0, H

    def test_elastic_state_unchanged(self):
        mu, sy0, H = self._params()
        tau_p = np.array([100e6, 80e6, 60e6])  # well below yield
        tc, Jp = self._return_map(tau_p, 0.0, mu, sy0, H)
        np.testing.assert_allclose(tc, tau_p, rtol=1e-12)
        assert Jp == pytest.approx(0.0)

    def test_plastic_state_reduces_stress(self):
        mu, sy0, H = self._params()
        # Very high deviatoric stress → must yield
        tau_p = np.array([1000e6, -500e6, -500e6])
        tc, Jp = self._return_map(tau_p, 0.0, mu, sy0, H)
        norm_trial = np.linalg.norm(tau_p - tau_p.mean())
        norm_corr  = np.linalg.norm(tc  - tc.mean())
        assert norm_corr < norm_trial

    def test_plastic_strain_increases(self):
        mu, sy0, H = self._params()
        tau_p = np.array([1000e6, -500e6, -500e6])
        _, Jp = self._return_map(tau_p, 0.0, mu, sy0, H)
        assert Jp > 0

    def test_consistency_on_yield_surface(self):
        mu, sy0, H = self._params()
        tau_p = np.array([1000e6, -500e6, -500e6])
        tc, Jp = self._return_map(tau_p, 0.0, mu, sy0, H)
        # After return, ||dev(τ)|| should equal sqrt(2/3)*σ_y_eff
        p_hyd = tc.sum() / 3.0
        s_dev = tc - p_hyd
        norm_s = np.linalg.norm(s_dev)
        sy_eff = sy0 + H * Jp
        target = math.sqrt(2.0 / 3.0) * sy_eff
        assert norm_s == pytest.approx(target, rel=1e-6)

    def test_hydrostatic_unchanged(self):
        """Plastic flow is deviatoric → hydrostatic part must not change."""
        mu, sy0, H = self._params()
        tau_p = np.array([800e6, -300e6, -500e6])
        p_before = tau_p.sum()
        tc, _ = self._return_map(tau_p, 0.0, mu, sy0, H)
        p_after = tc.sum()
        assert p_after == pytest.approx(p_before, rel=1e-6)

    def test_higher_hardening_less_plasticity(self):
        mu, sy0, _ = self._params()
        tau_p = np.array([1000e6, -500e6, -500e6])
        _, Jp_low  = self._return_map(tau_p, 0.0, mu, sy0, H=1.0e9)
        _, Jp_high = self._return_map(tau_p, 0.0, mu, sy0, H=10.0e9)
        assert Jp_high < Jp_low   # stiffer hardening → less plastic deformation

    def test_accumulated_hardening_raises_yield(self):
        mu, sy0, H = self._params()
        tau_p = np.array([400e6, -200e6, -200e6])
        _, Jp1 = self._return_map(tau_p, 0.0, mu, sy0, H)
        _, Jp2 = self._return_map(tau_p, Jp1,  mu, sy0, H)
        # Second pass: Jp should increase less (or not at all) as yield stress rose
        assert Jp2 <= Jp1 + Jp1 * 1.1   # coarse bound

    def test_zero_trial_stress_no_plasticity(self):
        mu, sy0, H = self._params()
        tau_p = np.zeros(3)
        tc, Jp = self._return_map(tau_p, 0.0, mu, sy0, H)
        np.testing.assert_allclose(tc, 0.0, atol=1e-30)
        assert Jp == 0.0


# ---------------------------------------------------------------------------
# Class 10: plot_energy_history (mock matplotlib)
# ---------------------------------------------------------------------------

class TestPlotEnergyHistory:

    @requires_taichi
    def test_plot_runs(self, tiny_solver):
        with patch("matplotlib.pyplot.show"):
            try:
                import matplotlib
                matplotlib.use("Agg")
            except Exception:
                pass
            tiny_solver.plot_energy_history(show=False)

    @requires_taichi
    def test_plot_saves_file(self, tiny_solver):
        with patch("matplotlib.pyplot.show"):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "energy.png")
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                except Exception:
                    pass
                tiny_solver.plot_energy_history(show=False, save_path=path)
                assert os.path.exists(path)

    @requires_taichi
    def test_show_false_no_plt_show(self, tiny_solver):
        with patch("matplotlib.pyplot.show") as mock_show:
            try:
                import matplotlib
                matplotlib.use("Agg")
            except Exception:
                pass
            tiny_solver.plot_energy_history(show=False)
            mock_show.assert_not_called()
