"""
Tests for nozzle_trajectory: ScanParams, NozzleTrajectory, and all loaders.
"""

import csv
import os
import tempfile

import numpy as np
import pytest

# Adjust sys.path so the module can be imported from the src folder
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "peen-ml"))

from nozzle_trajectory import (
    ScanParams,
    NozzleTrajectory,
    raster_scan,
    spiral_scan,
    zigzag_scan,
    from_csv,
    from_npy,
)


# -------------------------------------------------------------------
# NozzleTrajectory basics
# -------------------------------------------------------------------


class TestNozzleTrajectory:
    def test_init_valid(self):
        positions = np.zeros((10, 3), dtype=np.float32)
        traj = NozzleTrajectory(positions, dt=0.01)
        assert traj.n_steps == 10
        assert abs(traj.total_time - 0.1) < 1e-9

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            NozzleTrajectory(np.zeros((10, 2)))

    def test_rejects_1d(self):
        with pytest.raises(ValueError):
            NozzleTrajectory(np.zeros(10))

    def test_repr_contains_steps(self):
        traj = NozzleTrajectory(np.zeros((5, 3)), dt=0.1)
        assert "steps=5" in repr(traj)


# -------------------------------------------------------------------
# raster_scan
# -------------------------------------------------------------------


class TestRasterScan:
    def _default_params(self):
        return ScanParams(
            pattern="raster",
            Lx=0.04,
            Ly=0.04,
            z_standoff=0.15,
            scan_speed=0.05,
            line_spacing=0.01,
            dt=0.01,
        )

    def test_returns_trajectory(self):
        traj = raster_scan(self._default_params())
        assert isinstance(traj, NozzleTrajectory)

    def test_positions_shape(self):
        traj = raster_scan(self._default_params())
        assert traj.positions.shape[1] == 3

    def test_z_constant(self):
        params = self._default_params()
        traj = raster_scan(params)
        np.testing.assert_allclose(traj.positions[:, 2], params.z_standoff, atol=1e-6)

    def test_x_in_bounds(self):
        params = self._default_params()
        traj = raster_scan(params)
        assert traj.positions[:, 0].min() >= -1e-6
        assert traj.positions[:, 0].max() <= params.Lx + 1e-6

    def test_y_in_bounds(self):
        params = self._default_params()
        traj = raster_scan(params)
        assert traj.positions[:, 1].min() >= -1e-6
        assert traj.positions[:, 1].max() <= params.Ly + 1e-6

    def test_nonempty(self):
        traj = raster_scan(self._default_params())
        assert traj.n_steps > 0

    def test_dtype_float32(self):
        traj = raster_scan(self._default_params())
        assert traj.positions.dtype == np.float32


# -------------------------------------------------------------------
# zigzag_scan
# -------------------------------------------------------------------


class TestZigzagScan:
    def _params(self):
        return ScanParams(Lx=0.02, Ly=0.02, z_standoff=0.10, scan_speed=0.05, line_spacing=0.005, dt=0.01)

    def test_returns_trajectory(self):
        traj = zigzag_scan(self._params())
        assert isinstance(traj, NozzleTrajectory)

    def test_z_constant(self):
        params = self._params()
        traj = zigzag_scan(params)
        np.testing.assert_allclose(traj.positions[:, 2], params.z_standoff, atol=1e-6)

    def test_nonempty(self):
        traj = zigzag_scan(self._params())
        assert traj.n_steps > 0


# -------------------------------------------------------------------
# spiral_scan
# -------------------------------------------------------------------


class TestSpiralScan:
    def _params(self):
        return ScanParams(Lx=0.04, Ly=0.04, z_standoff=0.12, scan_speed=0.05, line_spacing=0.006, dt=0.01)

    def test_returns_trajectory(self):
        traj = spiral_scan(self._params())
        assert isinstance(traj, NozzleTrajectory)

    def test_nonempty(self):
        traj = spiral_scan(self._params())
        assert traj.n_steps > 0

    def test_z_constant(self):
        params = self._params()
        traj = spiral_scan(params)
        np.testing.assert_allclose(traj.positions[:, 2], params.z_standoff, atol=1e-6)


# -------------------------------------------------------------------
# from_csv
# -------------------------------------------------------------------


class TestFromCSV:
    def _write_xyz_csv(self, path, z_standoff=None):
        rows = [
            {"x": "0.01", "y": "0.01", "z": "0.15"},
            {"x": "0.02", "y": "0.01", "z": "0.15"},
            {"x": "0.02", "y": "0.02", "z": "0.15"},
        ]
        if z_standoff is not None:
            for r in rows:
                del r["z"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _write_xyzt_csv(self, path):
        rows = [
            {"x": "0.01", "y": "0.01", "z": "0.15", "t": "0.00"},
            {"x": "0.02", "y": "0.01", "z": "0.15", "t": "0.01"},
            {"x": "0.02", "y": "0.02", "z": "0.15", "t": "0.02"},
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_xyz_csv(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            self._write_xyz_csv(path)
            traj = from_csv(path)
            assert traj.n_steps == 3
            np.testing.assert_allclose(traj.positions[0], [0.01, 0.01, 0.15], atol=1e-5)
        finally:
            os.unlink(path)

    def test_xy_csv_with_standoff(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            self._write_xyz_csv(path, z_standoff=0.20)
            traj = from_csv(path, z_standoff=0.20)
            assert traj.n_steps == 3
            np.testing.assert_allclose(traj.positions[:, 2], 0.20, atol=1e-5)
        finally:
            os.unlink(path)

    def test_xyzt_csv_infers_dt(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            self._write_xyzt_csv(path)
            traj = from_csv(path)
            assert abs(traj.dt - 0.01) < 1e-6
        finally:
            os.unlink(path)

    def test_missing_z_no_standoff_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            # z_standoff=0.15 triggers deletion of 'z' column in the CSV helper
            self._write_xyz_csv(path, z_standoff=0.15)
            # CSV has no 'z' column, and we call from_csv without providing z_standoff
            with pytest.raises((KeyError, ValueError)):
                from_csv(path)  # must raise because z is missing
        finally:
            os.unlink(path)


# -------------------------------------------------------------------
# from_npy
# -------------------------------------------------------------------


class TestFromNpy:
    def test_3col(self):
        arr = np.array([[0.01, 0.01, 0.15], [0.02, 0.01, 0.15]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        try:
            np.save(path, arr)
            traj = from_npy(path)
            assert traj.n_steps == 2
            assert traj.dt == 0.01
        finally:
            os.unlink(path)

    def test_4col_infers_dt(self):
        arr = np.array([[0.01, 0.01, 0.15, 0.0], [0.02, 0.01, 0.15, 0.01], [0.02, 0.02, 0.15, 0.02]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        try:
            np.save(path, arr)
            traj = from_npy(path)
            assert traj.n_steps == 3
            assert abs(traj.dt - 0.01) < 1e-6
        finally:
            os.unlink(path)

    def test_2col_with_standoff(self):
        arr = np.array([[0.01, 0.01], [0.02, 0.01]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        try:
            np.save(path, arr)
            traj = from_npy(path, z_standoff=0.20)
            np.testing.assert_allclose(traj.positions[:, 2], 0.20, atol=1e-5)
        finally:
            os.unlink(path)

    def test_2col_no_standoff_raises(self):
        arr = np.array([[0.01, 0.01], [0.02, 0.01]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        try:
            np.save(path, arr)
            with pytest.raises(ValueError):
                from_npy(path)
        finally:
            os.unlink(path)

    def test_1d_raises(self):
        arr = np.array([0.01, 0.02, 0.03], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        try:
            np.save(path, arr)
            with pytest.raises(ValueError):
                from_npy(path)
        finally:
            os.unlink(path)
