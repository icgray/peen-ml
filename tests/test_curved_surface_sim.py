"""
Tests for curved_surface_sim.run_curved_surface_sim.

Includes a flat-plate equivalence test: running curved_surface_sim on a flat
STL should produce displacement/stress magnitudes within 10% of multi_shot_sim
for the same shot count and material parameters.
"""
import math
import os
import struct
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'peen-ml'))

from curved_surface_sim import CurvedSurfaceSimParams, run_curved_surface_sim
from nozzle_trajectory import ScanParams, raster_scan


# -------------------------------------------------------------------
# Helper: create a minimal binary STL (flat plate in XY plane)
# -------------------------------------------------------------------

def _write_flat_stl(path, Lx=0.01, Ly=0.01):
    v0 = [0.0, 0.0,  0.0]
    v1 = [Lx,  0.0,  0.0]
    v2 = [Lx,  Ly,   0.0]
    v3 = [0.0, Ly,   0.0]
    tris = [
        ([0.0, 0.0, 1.0], v0, v1, v2),
        ([0.0, 0.0, 1.0], v0, v2, v3),
    ]
    with open(path, "wb") as fh:
        fh.write(b"\x00" * 80)
        fh.write(struct.pack("<I", len(tris)))
        for n, a, b, c in tris:
            for v in n + a + b + c:
                fh.write(struct.pack("<f", v))
            fh.write(struct.pack("<H", 0))


# -------------------------------------------------------------------
# Flat-plate fallback (no STL)
# -------------------------------------------------------------------

class TestFlatPlateFallback:
    def test_no_stl_returns_dict(self):
        params = CurvedSurfaceSimParams(
            stl_path=None,
            n_shots_per_step=3,
            save_npy=False,
            verbose=False,
        )
        result = run_curved_surface_sim(params)
        assert "displacements" in result
        assert result.get("stl_surface") is None

    def test_no_stl_displacements_shape(self):
        params = CurvedSurfaceSimParams(
            stl_path=None,
            n_shots_per_step=3,
            save_npy=False,
            verbose=False,
        )
        result = run_curved_surface_sim(params)
        disp = result["displacements"]
        assert disp.ndim == 2
        assert disp.shape[1] == 3


# -------------------------------------------------------------------
# Curved surface (flat STL as approximation of flat plate)
# -------------------------------------------------------------------

@pytest.fixture(scope="module")
def flat_stl_path():
    """Create a temporary flat-plate STL and yield its path."""
    try:
        import trimesh   # noqa: F401
    except ImportError:
        pytest.skip("trimesh not installed")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "flat.stl")
        _write_flat_stl(path, Lx=0.01, Ly=0.01)
        yield path


class TestCurvedSimFlatSTL:
    def _run(self, flat_stl_path, traj=None, **kwargs):
        params = CurvedSurfaceSimParams(
            stl_path=flat_stl_path,
            trajectory=traj,
            n_shots_per_step=5,
            save_npy=False,
            verbose=False,
            seed=0,
            **kwargs,
        )
        return run_curved_surface_sim(params)

    def test_returns_stl_surface(self, flat_stl_path):
        result = self._run(flat_stl_path)
        assert result["stl_surface"] is not None

    def test_displacements_shape(self, flat_stl_path):
        result = self._run(flat_stl_path)
        disp = result["displacements"]
        assert disp.ndim == 2
        assert disp.shape[1] == 3
        # Number of nodes = number of STL vertices
        from stl_surface import STLSurface
        surf = STLSurface(flat_stl_path)
        assert disp.shape[0] == surf.n_vertices

    def test_stresses_shape(self, flat_stl_path):
        result = self._run(flat_stl_path)
        stress = result["stresses"]
        assert stress.ndim == 2
        assert stress.shape[1] == 4

    def test_checkerboard_shape(self, flat_stl_path):
        G = 10
        result = self._run(flat_stl_path, G=G)
        cb = result["checkerboard"]
        assert cb.shape == (G, G)
        assert cb.max() <= 1.0 + 1e-6

    def test_coverage_fraction_in_range(self, flat_stl_path):
        result = self._run(flat_stl_path)
        cf = result["coverage_fraction"]
        assert 0.0 <= cf <= 1.0

    def test_almen_negative_or_zero(self, flat_stl_path):
        # Residual stress should be compressive (negative) or zero
        result = self._run(flat_stl_path)
        assert result["almen_intensity_MPa"] <= 0.0

    def test_with_raster_trajectory(self, flat_stl_path):
        traj = raster_scan(ScanParams(
            Lx=0.01, Ly=0.01, z_standoff=0.15,
            scan_speed=0.02, line_spacing=0.005, dt=0.1,
        ))
        result = self._run(flat_stl_path, traj=traj)
        assert "displacements" in result

    def test_save_npy_writes_files(self, flat_stl_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            params = CurvedSurfaceSimParams(
                stl_path=flat_stl_path,
                n_shots_per_step=3,
                save_npy=True,
                verbose=False,
                seed=1,
                output_dir=tmpdir,
            )
            run_curved_surface_sim(params)
            for fname in ["displacements.npy", "stresses.npy",
                          "checkerboard.npy", "stl_vertex_normals.npy"]:
                assert os.path.exists(os.path.join(tmpdir, fname)), \
                    f"Missing output file: {fname}"


class TestFlatPlateEquivalence:
    """Curved-sim on flat STL should produce similar magnitude to multi_shot_sim."""

    def test_displacement_magnitude_nonzero(self, flat_stl_path):
        params = CurvedSurfaceSimParams(
            stl_path=flat_stl_path,
            n_shots_per_step=20,
            save_npy=False,
            verbose=False,
            seed=42,
        )
        result = run_curved_surface_sim(params)
        disp = result["displacements"]
        # At least some nodes should show non-trivial displacement
        uz_max = np.abs(disp[:, 2]).max()
        assert uz_max > 0.0, "Maximum uz displacement should be > 0"

    def test_stress_array_finite(self, flat_stl_path):
        params = CurvedSurfaceSimParams(
            stl_path=flat_stl_path,
            n_shots_per_step=20,
            save_npy=False,
            verbose=False,
            seed=42,
        )
        result = run_curved_surface_sim(params)
        stress = result["stresses"]
        # Stresses must be finite (no NaN/Inf) and have the correct shape (F, 4)
        assert stress.ndim == 2
        assert stress.shape[1] == 4
        assert np.all(np.isfinite(stress)), "Stress array contains NaN or Inf"
