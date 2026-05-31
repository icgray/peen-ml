"""
Tests for stl_surface.STLSurface.

Uses a synthetic flat-plate STL (known normals = [0, 0, 1]) so tests are
self-contained — no external .stl files required.
"""
import math
import os
import struct
import tempfile

import numpy as np
import pytest

# -------------------------------------------------------------------
# Helper: create a minimal binary STL for a flat plate in the XY plane
# -------------------------------------------------------------------

def _write_flat_stl(path, Lx=0.04, Ly=0.04):
    """Write a binary STL with two right-angle triangles forming a flat XY plate."""
    v0 = [0.0, 0.0,  0.0]
    v1 = [Lx,  0.0,  0.0]
    v2 = [Lx,  Ly,   0.0]
    v3 = [0.0, Ly,   0.0]
    triangles = [
        ([0.0, 0.0, 1.0], v0, v1, v2),
        ([0.0, 0.0, 1.0], v0, v2, v3),
    ]
    with open(path, "wb") as fh:
        fh.write(b"\x00" * 80)              # header
        fh.write(struct.pack("<I", len(triangles)))
        for n, a, b, c in triangles:
            for val in n + a + b + c:
                fh.write(struct.pack("<f", val))
            fh.write(struct.pack("<H", 0))  # attribute byte count


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture(scope="module")
def flat_stl_path():
    try:
        import trimesh   # noqa: F401
    except ImportError:
        pytest.skip("trimesh not installed")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "flat_plate.stl")
        _write_flat_stl(path)
        yield path


@pytest.fixture(scope="module")
def flat_surface(flat_stl_path):
    from stl_surface import STLSurface
    return STLSurface(flat_stl_path)


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

class TestSTLSurfaceLoad:
    def test_vertex_count(self, flat_surface):
        assert flat_surface.n_vertices >= 3

    def test_face_count(self, flat_surface):
        assert flat_surface.n_faces == 2

    def test_face_normals_upward(self, flat_surface):
        # All face normals should point in +Z direction
        nz = flat_surface.face_normals[:, 2]
        assert np.all(nz > 0.9), f"face normals not upward: {flat_surface.face_normals}"

    def test_vertex_normals_shape(self, flat_surface):
        V = flat_surface.n_vertices
        assert flat_surface.vertex_normals.shape == (V, 3)

    def test_vertex_normals_unit(self, flat_surface):
        norms = np.linalg.norm(flat_surface.vertex_normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_vertex_normals_upward(self, flat_surface):
        nz = flat_surface.vertex_normals[:, 2]
        assert np.all(nz > 0.9), f"vertex normals not upward: {flat_surface.vertex_normals}"


class TestShotProjection:
    def test_returns_correct_shapes(self, flat_surface):
        shots = np.array([[0.01, 0.01], [0.02, 0.03]], dtype=np.float32)
        hit_xyz, normals, angles = flat_surface.project_shots_onto_surface(shots, z_nozzle=0.15)
        assert hit_xyz.shape == (2, 3)
        assert normals.shape == (2, 3)
        assert angles.shape == (2,)

    def test_hit_points_on_surface(self, flat_surface):
        shots = np.array([[0.01, 0.01], [0.02, 0.03]], dtype=np.float32)
        hit_xyz, _, _ = flat_surface.project_shots_onto_surface(shots, z_nozzle=0.15)
        # For a flat plate in z=0 plane, hit z should be ~0
        np.testing.assert_allclose(hit_xyz[:, 2], 0.0, atol=1e-3)

    def test_angles_in_range(self, flat_surface):
        shots = np.array([[0.02, 0.02]], dtype=np.float32)
        _, _, angles = flat_surface.project_shots_onto_surface(shots, z_nozzle=0.15)
        assert 0.0 <= float(angles[0]) <= math.pi / 2.0 + 1e-6


class TestCheckerboard:
    def test_shape(self, flat_surface):
        shots   = flat_surface.vertices[:5, :3]
        weights = np.ones(5, dtype=np.float32)
        cb = flat_surface.shots_to_checkerboard(shots, weights, G=5)
        assert cb.shape == (5, 5)

    def test_range(self, flat_surface):
        shots   = flat_surface.vertices[:5, :3]
        weights = np.ones(5, dtype=np.float32)
        cb = flat_surface.shots_to_checkerboard(shots, weights, G=5)
        assert cb.min() >= 0.0
        assert cb.max() <= 1.0 + 1e-6

    def test_dtype(self, flat_surface):
        shots   = flat_surface.vertices[:3, :3]
        weights = np.ones(3, dtype=np.float32)
        cb = flat_surface.shots_to_checkerboard(shots, weights, G=4)
        assert cb.dtype == np.float32


class TestNodeElementArrays:
    def test_node_labels_1indexed(self, flat_surface):
        coords, labels = flat_surface.to_node_arrays()
        assert labels.min() == 1
        assert labels.max() == flat_surface.n_vertices

    def test_element_labels_1indexed(self, flat_surface):
        conn, elems = flat_surface.to_element_arrays()
        assert elems.min() == 1
        assert elems.max() == flat_surface.n_faces

    def test_connectivity_references_valid_nodes(self, flat_surface):
        conn, _ = flat_surface.to_element_arrays()
        # All node references in connectivity should be in [1, n_vertices]
        assert conn.min() >= 1
        assert conn.max() <= flat_surface.n_vertices


class TestRotationMatrices:
    def test_shape(self, flat_surface):
        R = flat_surface.vertex_normal_rotation_matrices()
        assert R.shape == (flat_surface.n_vertices, 3, 3)

    def test_flat_plate_identity(self, flat_surface):
        R = flat_surface.vertex_normal_rotation_matrices()
        # For a flat plate (normals = [0,0,1]), R should be the identity
        for Ri in R:
            np.testing.assert_allclose(Ri, np.eye(3), atol=1e-5,
                                       err_msg="Flat-plate rotation should be identity")

    def test_rotates_z_to_normal(self, flat_surface):
        R       = flat_surface.vertex_normal_rotation_matrices()
        z_hat   = np.array([0.0, 0.0, 1.0])
        normals = flat_surface.vertex_normals.astype(np.float64)
        for i, (Ri, ni) in enumerate(zip(R, normals)):
            rotated = Ri.astype(np.float64) @ z_hat
            np.testing.assert_allclose(rotated, ni, atol=1e-4,
                                       err_msg=f"Vertex {i}: R@z != normal")


class TestSaveArrays:
    def test_saves_all_files(self, flat_surface):
        with tempfile.TemporaryDirectory() as tmpdir:
            flat_surface.save_arrays(tmpdir)
            for fname in ["node_coords.npy", "node_labels.npy",
                          "element_connectivity.npy", "element_labels.npy",
                          "stl_vertex_normals.npy", "stl_face_normals.npy"]:
                assert os.path.exists(os.path.join(tmpdir, fname)), \
                    f"Missing file: {fname}"
