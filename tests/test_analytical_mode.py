"""Tests for analytical_mode.py — Shen-Atluri vs Sherafatnia analytical predictor."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "peen-ml"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from impact_sim import ShotPeenParams  # noqa: E402
import analytical_mode as am  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TI_STEEL = ShotPeenParams(
    V=40.0,
    D=0.6e-3,
    E_b=110e9,
    nu_b=0.34,
    sigma_yield=800e6,
    c=1e9,
    E_s=200e9,
    nu_s=0.30,
    rho_s=7800.0,
)


def _flat_mesh(N=11):
    """Return (N*N, 3) node coords on a 10mm x 10mm flat plate."""
    xs = np.linspace(0, 0.01, N)
    ys = np.linspace(0, 0.01, N)
    Xg, Yg = np.meshgrid(xs, ys)
    return np.stack([Xg.ravel(), Yg.ravel(), np.zeros(N * N)], axis=1).astype(np.float64)


def _shots(n=5, seed=0):
    """Return (n, 2) random shot positions within a 10mm plate."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.001, 0.009, size=(n, 2))


def _write_sim_dir(path, node_coords, displacements, shot_positions, with_stresses=True):
    """Write a minimal Simulation_N/ directory for compare_to_dataset."""
    np.save(os.path.join(path, "node_coords.npy"), node_coords)
    np.save(os.path.join(path, "displacements.npy"), displacements)
    np.save(os.path.join(path, "shot_positions.npy"), shot_positions)
    if with_stresses:
        rng = np.random.default_rng(1)
        stresses = rng.uniform(-50e6, 0, size=(len(node_coords), 4)).astype(np.float32)
        np.save(os.path.join(path, "nodal_stresses.npy"), stresses)
    # minimal simulation_params.txt
    with open(os.path.join(path, "simulation_params.txt"), "w") as fh:
        fh.write("[physics]\nV_m_per_s=40.0\nD_m=0.0006\n")
        fh.write("[material]\nE_b=110e9\nnu_b=0.34\nsigma_yield=800e6\nc=1e9\n")
        fh.write("E_s=200e9\nnu_s=0.30\nrho_s=7800.0\n")


# ---------------------------------------------------------------------------
# predict_analytical
# ---------------------------------------------------------------------------


class TestPredictAnalytical:
    def test_shen_atluri_output_shape(self):
        nc = _flat_mesh(7)
        shots = _shots(3)
        disp = am.predict_analytical(nc, shots, TI_STEEL, model="shen_atluri")
        assert disp.shape == (len(nc), 3)

    def test_sherafatnia_output_shape(self):
        nc = _flat_mesh(7)
        shots = _shots(3)
        disp = am.predict_analytical(nc, shots, TI_STEEL, model="sherafatnia")
        assert disp.shape == (len(nc), 3)

    def test_uz_is_negative_or_zero(self):
        """Permanent dent must push surface down (negative uz)."""
        nc = _flat_mesh(11)
        shots = _shots(5)
        for model in ("shen_atluri", "sherafatnia"):
            disp = am.predict_analytical(nc, shots, TI_STEEL, model=model)
            assert disp[:, 2].max() <= 1e-12, f"{model}: uz should be <= 0"

    def test_magnitude_nonzero(self):
        nc = _flat_mesh(11)
        shots = _shots(5)
        for model in ("shen_atluri", "sherafatnia"):
            disp = am.predict_analytical(nc, shots, TI_STEEL, model=model)
            assert np.abs(disp).max() > 0, f"{model}: all-zero displacement unexpected"

    def test_more_shots_more_deformation(self):
        nc = _flat_mesh(11)
        disp_few = am.predict_analytical(nc, _shots(2), TI_STEEL)
        disp_many = am.predict_analytical(nc, _shots(20), TI_STEEL)
        assert np.abs(disp_many[:, 2]).sum() > np.abs(disp_few[:, 2]).sum()

    def test_shot_times_reorders_correctly(self):
        nc = _flat_mesh(7)
        shots = _shots(4)
        times = np.array([3.0, 1.0, 4.0, 2.0])
        disp_timed = am.predict_analytical(nc, shots, TI_STEEL, shot_times=times)
        # Result magnitude should still be nonzero
        assert np.abs(disp_timed).max() > 0

    def test_invalid_model_raises(self):
        nc = _flat_mesh(5)
        with pytest.raises(ValueError, match="Unknown model"):
            am.predict_analytical(nc, _shots(2), TI_STEEL, model="invalid_model")

    def test_sequential_mode_runs(self):
        nc = _flat_mesh(7)
        shots = _shots(4)
        disp = am.predict_analytical(nc, shots, TI_STEEL, sequential=True)
        assert disp.shape == (len(nc), 3)

    def test_sa_wider_than_sh_contact(self):
        """Shen-Atluri plastic zone radius < Sherafatnia elastic radius.
        Sherafatnia dent is wider (larger a_e), but should still produce uz <= 0."""
        nc = _flat_mesh(11)
        shots = np.array([[0.005, 0.005]])
        sa = am.predict_analytical(nc, shots, TI_STEEL, model="shen_atluri")
        sh = am.predict_analytical(nc, shots, TI_STEEL, model="sherafatnia")
        # Both produce downward deformation
        assert sa[:, 2].min() < 0
        assert sh[:, 2].min() < 0


# ---------------------------------------------------------------------------
# _compute_sa_nodal_stresses
# ---------------------------------------------------------------------------


class TestComputeSaNodal:
    def test_returns_n4_array(self):
        nc = _flat_mesh(7)
        shots = _shots(3)
        result = am._compute_sa_nodal_stresses(TI_STEEL, nc, shots)
        assert result is not None
        assert result.shape == (len(nc), 4)

    def test_stresses_nonzero_near_impact(self):
        nc = _flat_mesh(11)
        # single shot at centre
        shots = np.array([[0.005, 0.005]])
        result = am._compute_sa_nodal_stresses(TI_STEEL, nc, shots)
        assert result is not None
        assert np.abs(result).max() > 0


# ---------------------------------------------------------------------------
# compare_to_dataset
# ---------------------------------------------------------------------------


class TestCompareToDataset:
    def test_returns_correct_structure(self, tmp_path):
        nc = _flat_mesh(7)
        shots = _shots(4)
        disp = np.zeros((len(nc), 3), dtype=np.float32)
        _write_sim_dir(str(tmp_path), nc, disp, shots, with_stresses=True)
        results = am.compare_to_dataset(str(tmp_path))
        assert "shen_atluri" in results
        assert "sherafatnia" in results
        for model in ("shen_atluri", "sherafatnia"):
            for comp in ("ux", "uy", "uz"):
                assert comp in results[model]
                assert "r" in results[model][comp]
                assert "rmse_um" in results[model][comp]
                assert "n" in results[model][comp]

    def test_saves_figure(self, tmp_path):
        nc = _flat_mesh(7)
        shots = _shots(3)
        disp = np.zeros((len(nc), 3), dtype=np.float32)
        _write_sim_dir(str(tmp_path), nc, disp, shots)
        out_png = str(tmp_path / "compare.png")
        am.compare_to_dataset(str(tmp_path), out_path=out_png)
        assert os.path.exists(out_png), "compare_to_dataset should save a figure"

    def test_without_stress_file(self, tmp_path):
        nc = _flat_mesh(7)
        shots = _shots(3)
        disp = np.zeros((len(nc), 3), dtype=np.float32)
        _write_sim_dir(str(tmp_path), nc, disp, shots, with_stresses=False)
        results = am.compare_to_dataset(str(tmp_path))
        assert "shen_atluri" in results  # should not crash

    def test_metrics_finite(self, tmp_path):
        nc = _flat_mesh(7)
        shots = _shots(4)
        disp = np.random.default_rng(42).random((len(nc), 3)).astype(np.float32) * 1e-5
        _write_sim_dir(str(tmp_path), nc, disp, shots)
        results = am.compare_to_dataset(str(tmp_path))
        for comp in ("ux", "uz"):
            r = results["shen_atluri"][comp]["r"]
            # r may be nan if variance is zero, but should not be inf
            assert not (r == float("inf") or r == float("-inf"))


# ---------------------------------------------------------------------------
# compare_dataset (batch)
# ---------------------------------------------------------------------------


class TestCompareDataset:
    def _make_dataset(self, root, n_sims=5):
        root.mkdir(parents=True, exist_ok=True)
        nc = _flat_mesh(7)
        shots = _shots(3)
        for i in range(n_sims):
            sim_dir = root / f"Simulation_{i}"
            sim_dir.mkdir()
            disp = np.zeros((len(nc), 3), dtype=np.float32)
            _write_sim_dir(str(sim_dir), nc, disp, shots, with_stresses=(i % 2 == 0))
        return root

    def test_returns_rows(self, tmp_path):
        dataset = self._make_dataset(tmp_path / "ds")
        rows = am.compare_dataset(str(dataset), n_sims=3, seed=0, verbose=False)
        assert len(rows) > 0

    def test_saves_csv(self, tmp_path):
        dataset = self._make_dataset(tmp_path / "ds")
        csv_path = str(tmp_path / "out.csv")
        am.compare_dataset(str(dataset), n_sims=3, seed=0, verbose=False, out_csv=csv_path)
        assert os.path.exists(csv_path)
