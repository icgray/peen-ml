"""
Tests for data_viz.py — the mesh and field visualization module.

Coverage targets:
    - load_data                   (happy path, invalid file, empty file)
    - visualize_checkerboard      (renders without error, handles bad folder)
    - compute_deformed_mesh       (correct values, scale factor, label alignment,
                                   missing files, returns None on partial data)
    - visualize_mesh              (runs without opening a window)
    - visualize_stress_field      (runs without opening a window)
    - visualize_deformation       (runs without opening a window)
    - visualize_all               (full pipeline smoke test)

All plt.show() calls are patched to prevent GUI windows from appearing during CI.

Test categories:
    Smoke    – verifies basic execution without crashing.
    One-shot – tests a single specific behaviour with known inputs/outputs.
    Edge     – boundary conditions, missing files, empty inputs, etc.
    Property – invariants that must always hold.
"""

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/peen-ml"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from data_viz import (  # noqa: E402
    compute_deformed_mesh,
    load_data,
    visualize_all,
    visualize_checkerboard,
    visualize_deformation,
    visualize_mesh,
    visualize_stress_field,
)

# ---------------------------------------------------------------------------
# Constants (must stay consistent with conftest.py)
# ---------------------------------------------------------------------------
NUM_NODES = 10
NUM_ELEMENTS = 4
GRID_SIZE = 5

# Patch target for all matplotlib show calls inside data_viz
_PLT_SHOW = "data_viz.plt.show"
_PLT_ION = "data_viz.plt.ion"     # not used in data_viz, but safe to include
_PLT_IOFF = "data_viz.plt.ioff"   # not used in data_viz, but safe to include


# ============================================================================
# load_data
# ============================================================================

class TestLoadData:
    """Tests for the safe numpy file loader."""

    # -- Smoke ----------------------------------------------------------------

    def test_smoke_load_real_file(self, sim_folder):
        """Smoke: loads a valid .npy file from the shared sim fixture."""
        path = sim_folder / "checkerboard.npy"
        data = load_data(str(path))
        assert data is not None

    # -- One-shot -------------------------------------------------------------

    def test_returns_correct_array(self, sim_folder):
        """One-shot: loaded array matches what was saved."""
        original = np.random.default_rng(99).uniform(0, 1, (5, 5))
        path = sim_folder / "_test_array.npy"
        np.save(str(path), original)
        loaded = load_data(str(path))
        assert np.allclose(loaded, original)

    def test_returns_none_for_nonexistent_file(self):
        """One-shot: missing file path returns None (no exception raised)."""
        result = load_data("/absolutely/not/a/real/path.npy", "test nonexistent")
        assert result is None

    def test_description_param_does_not_affect_return(self, sim_folder):
        """One-shot: description is only for logging — output is unchanged."""
        path = sim_folder / "node_coords.npy"
        r1 = load_data(str(path), "with description")
        r2 = load_data(str(path), "")
        assert np.array_equal(r1, r2)

    # -- Edge -----------------------------------------------------------------

    def test_returns_none_for_empty_file(self, tmp_path):
        """Edge: an empty (zero-byte) file returns None."""
        empty = tmp_path / "empty.npy"
        empty.write_bytes(b"")
        result = load_data(str(empty), "empty file")
        assert result is None

    def test_returns_none_for_corrupt_file(self, tmp_path):
        """Edge: a file with garbage bytes returns None."""
        corrupt = tmp_path / "corrupt.npy"
        corrupt.write_bytes(b"\x00\xFF\xAB\x12garbage data here")
        result = load_data(str(corrupt), "corrupt")
        assert result is None

    def test_returns_correct_shape(self, sim_folder):
        """Property: shape of returned array matches what was saved."""
        path = sim_folder / "checkerboard.npy"
        data = load_data(str(path))
        assert data.shape == (GRID_SIZE, GRID_SIZE)


# ============================================================================
# visualize_checkerboard
# ============================================================================

class TestVisualizeCheckerboard:
    """Tests for checkerboard pattern rendering."""

    # -- Smoke ----------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_smoke_runs_without_error(self, mock_show, sim_folder):
        """Smoke: function completes without raising."""
        visualize_checkerboard(str(sim_folder))
        mock_show.assert_called_once()

    # -- One-shot -------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_uses_real_simulation_folder(self, mock_show):
        """One-shot: runs on the real tests/simulation_0 fixture data."""
        folder = os.path.join(os.getcwd(), "tests", "simulation_0")
        visualize_checkerboard(folder)
        mock_show.assert_called()

    # -- Edge -----------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_missing_checkerboard_file_does_not_crash(self, mock_show, tmp_path):
        """Edge: folder without checkerboard.npy returns silently (no crash)."""
        # Should print an error but not raise
        visualize_checkerboard(str(tmp_path))
        # plt.show should NOT have been called if file is missing
        mock_show.assert_not_called()

    @patch(_PLT_SHOW)
    def test_nonexistent_folder_does_not_crash(self, mock_show):
        """Edge: non-existent folder path is handled gracefully."""
        visualize_checkerboard("/this/folder/does/not/exist")
        mock_show.assert_not_called()


# ============================================================================
# compute_deformed_mesh
# ============================================================================

class TestComputeDeformedMesh:
    """Tests for mesh deformation computation."""

    # -- Smoke ----------------------------------------------------------------

    def test_smoke_returns_three_values(self, sim_folder):
        """Smoke: function returns a 3-tuple."""
        result = compute_deformed_mesh(str(sim_folder))
        assert len(result) == 3

    # -- One-shot -------------------------------------------------------------

    def test_returns_correct_node_coords(self, sim_folder):
        """One-shot: returned node_coords matches the saved node_coords.npy."""
        expected_coords = np.load(str(sim_folder / "node_coords.npy"))
        node_coords, _, _ = compute_deformed_mesh(str(sim_folder))
        assert np.allclose(node_coords, expected_coords)

    def test_deformed_coords_equals_coords_plus_displacements(self, sim_folder):
        """One-shot: deformed_coords == node_coords + displacements (scale=1, aligned labels)."""
        node_coords = np.load(str(sim_folder / "node_coords.npy"))
        displacements = np.load(str(sim_folder / "displacements.npy"))
        # In sim_folder, disp_node_labels == node_labels (same order)
        expected_deformed = node_coords + displacements

        _, deformed_coords, _ = compute_deformed_mesh(str(sim_folder), scale_factor=1)
        assert np.allclose(deformed_coords, expected_deformed, atol=1e-5)

    def test_scale_factor_doubles_deformation(self, sim_folder):
        """One-shot: scale_factor=2 produces twice the displacement as scale_factor=1."""
        node_coords_s1, deformed_s1, _ = compute_deformed_mesh(str(sim_folder), scale_factor=1)
        node_coords_s2, deformed_s2, _ = compute_deformed_mesh(str(sim_folder), scale_factor=2)

        disp_s1 = deformed_s1 - node_coords_s1
        disp_s2 = deformed_s2 - node_coords_s2
        assert np.allclose(disp_s2, 2 * disp_s1, atol=1e-5)

    def test_scale_factor_zero_returns_original_coords(self, sim_folder):
        """One-shot: scale_factor=0 → deformed_coords == node_coords."""
        node_coords, deformed_coords, _ = compute_deformed_mesh(str(sim_folder), scale_factor=0)
        assert np.allclose(node_coords, deformed_coords, atol=1e-5)

    def test_element_nodes_is_list(self, sim_folder):
        """One-shot: element_nodes is a list (of lists of node indices)."""
        _, _, element_nodes = compute_deformed_mesh(str(sim_folder))
        assert isinstance(element_nodes, list)

    def test_element_nodes_count_matches_connectivity(self, sim_folder):
        """Property: number of entries in element_nodes == number of rows in element_connectivity.npy."""
        connectivity = np.load(str(sim_folder / "element_connectivity.npy"))
        _, _, element_nodes = compute_deformed_mesh(str(sim_folder))
        assert len(element_nodes) == len(connectivity)

    def test_shuffled_label_alignment(self, shuffled_labels_sim_folder):
        """Property: displacements are correctly re-indexed when disp labels differ from node labels."""
        folder, node_coords, displacements, disp_node_labels, node_labels = (
            shuffled_labels_sim_folder
        )

        # Build expected deformed coords manually with the correct alignment
        node_label_to_index = {lbl: idx for idx, lbl in enumerate(node_labels)}
        aligned_disp = np.zeros_like(node_coords)
        for idx, lbl in enumerate(disp_node_labels):
            aligned_disp[node_label_to_index[lbl]] = displacements[idx]
        expected_deformed = node_coords + aligned_disp

        _, deformed_coords, _ = compute_deformed_mesh(str(folder), scale_factor=1)
        assert np.allclose(deformed_coords, expected_deformed, atol=1e-5)

    # -- One-shot (real data) -------------------------------------------------

    def test_correct_data_with_real_simulation_0(self):
        """One-shot: uses the real tests/simulation_0 data (regression guard)."""
        folder = os.path.join(os.getcwd(), "tests", "simulation_0")
        expected_coords = np.load(os.path.join(folder, "node_coords.npy"))
        node_coords, deformed_coords, element_nodes = compute_deformed_mesh(folder, 1)

        assert node_coords is not None
        assert np.allclose(node_coords, expected_coords)
        assert deformed_coords is not None
        assert element_nodes is not None

    # -- Edge -----------------------------------------------------------------

    def test_missing_node_coords_returns_none_triple(self, tmp_path):
        """Edge: absent node_coords.npy → (None, None, None)."""
        # Only provide some files
        np.save(tmp_path / "node_labels.npy", np.arange(1, 4, dtype=np.int32))
        np.save(tmp_path / "displacements.npy", np.zeros((3, 3), dtype=np.float32))
        np.save(tmp_path / "disp_node_labels.npy", np.arange(1, 4, dtype=np.int32))
        np.save(tmp_path / "element_connectivity.npy", np.array([[1, 2, 3]], dtype=np.int32))
        # node_coords.npy is intentionally absent
        result = compute_deformed_mesh(str(tmp_path))
        assert result == (None, None, None)

    def test_missing_element_connectivity_returns_none_triple(self, sim_folder, tmp_path):
        """Edge: absent element_connectivity.npy → (None, None, None)."""
        import shutil  # noqa: PLC0415
        # Copy all files except element_connectivity
        for fname in os.listdir(str(sim_folder)):
            if fname.endswith(".npy") and fname != "element_connectivity.npy":
                shutil.copy(str(sim_folder / fname), str(tmp_path / fname))
        result = compute_deformed_mesh(str(tmp_path))
        assert result == (None, None, None)

    def test_empty_folder_returns_none_triple(self, tmp_path):
        """Edge: completely empty folder → (None, None, None)."""
        result = compute_deformed_mesh(str(tmp_path))
        assert result == (None, None, None)


# ============================================================================
# visualize_mesh
# ============================================================================

class TestVisualizeMesh:
    """Tests for undeformed + deformed mesh line-collection rendering."""

    # -- Smoke ----------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_smoke_runs_without_error(self, mock_show, sim_folder):
        """Smoke: runs without raising on valid mesh data."""
        node_coords, deformed_coords, element_nodes = compute_deformed_mesh(str(sim_folder))
        assert node_coords is not None
        visualize_mesh(node_coords, deformed_coords, element_nodes)
        mock_show.assert_called_once()

    # -- One-shot -------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_calls_plt_show(self, mock_show, sim_folder):
        """One-shot: plt.show is called exactly once."""
        node_coords, deformed_coords, element_nodes = compute_deformed_mesh(str(sim_folder))
        visualize_mesh(node_coords, deformed_coords, element_nodes)
        assert mock_show.call_count == 1

    # -- Edge -----------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_identical_coords_does_not_crash(self, mock_show, sim_folder):
        """Edge: undeformed == deformed (zero displacement) renders without error."""
        node_coords, _, element_nodes = compute_deformed_mesh(str(sim_folder))
        # Pass same coords for both (zero deformation)
        visualize_mesh(node_coords, node_coords, element_nodes)
        mock_show.assert_called_once()


# ============================================================================
# visualize_stress_field
# ============================================================================

class TestVisualizeStressField:
    """Tests for von Mises stress rendering on the deformed mesh."""

    # -- Smoke ----------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_smoke_runs_without_error(self, mock_show, sim_folder):
        """Smoke: runs without error on the synthetic fixture data."""
        _, deformed_coords, element_nodes = compute_deformed_mesh(str(sim_folder))
        visualize_stress_field(str(sim_folder), deformed_coords, element_nodes)
        mock_show.assert_called_once()

    @patch(_PLT_SHOW)
    def test_smoke_with_real_simulation_0(self, mock_show):
        """Smoke: runs on the real tests/simulation_0 fixture data."""
        folder = os.path.join(os.getcwd(), "tests", "simulation_0")
        _, deformed_coords, element_nodes = compute_deformed_mesh(folder)
        if deformed_coords is not None:
            visualize_stress_field(folder, deformed_coords, element_nodes)

    # -- Edge -----------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_missing_stress_file_does_not_crash(self, mock_show, sim_folder, tmp_path):
        """Edge: absent stresses.npy returns silently without raising."""
        import shutil  # noqa: PLC0415
        # Copy all files except stresses
        for fname in os.listdir(str(sim_folder)):
            if fname.endswith(".npy") and "stress" not in fname:
                shutil.copy(str(sim_folder / fname), str(tmp_path / fname))

        _, deformed_coords, element_nodes = compute_deformed_mesh(str(sim_folder))
        # Should not raise even with missing stress files in tmp_path
        visualize_stress_field(str(tmp_path), deformed_coords, element_nodes)


# ============================================================================
# visualize_deformation
# ============================================================================

class TestVisualizeDeformation:
    """Tests for deformation-magnitude rendering on the deformed mesh."""

    # -- Smoke ----------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_smoke_runs_without_error(self, mock_show, sim_folder):
        """Smoke: runs without error."""
        node_coords, deformed_coords, element_nodes = compute_deformed_mesh(str(sim_folder))
        aligned_displacements = deformed_coords - node_coords
        visualize_deformation(str(sim_folder), deformed_coords, element_nodes, aligned_displacements)
        mock_show.assert_called_once()

    # -- One-shot -------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_zero_displacement_does_not_crash(self, mock_show, sim_folder):
        """One-shot: all-zero displacements render a uniform colour map without error."""
        node_coords, deformed_coords, element_nodes = compute_deformed_mesh(str(sim_folder))
        zero_disp = np.zeros_like(node_coords)
        visualize_deformation(str(sim_folder), deformed_coords, element_nodes, zero_disp)
        mock_show.assert_called_once()

    # -- Edge -----------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_large_displacement_values_do_not_crash(self, mock_show, sim_folder):
        """Edge: very large displacement values (potential overflow) handled gracefully."""
        node_coords, deformed_coords, element_nodes = compute_deformed_mesh(str(sim_folder))
        large_disp = np.full_like(node_coords, 1e30)
        visualize_deformation(str(sim_folder), deformed_coords, element_nodes, large_disp)
        mock_show.assert_called_once()


# ============================================================================
# visualize_all
# ============================================================================

class TestVisualizeAll:
    """Tests for the full visualization pipeline wrapper."""

    # -- Smoke ----------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_smoke_runs_without_error(self, mock_show, sim_folder):
        """Smoke: visualize_all completes without raising."""
        visualize_all(str(sim_folder), scale_factor=1)

    @patch(_PLT_SHOW)
    def test_smoke_with_real_simulation_0(self, mock_show):
        """Smoke: runs on the canonical tests/simulation_0 folder."""
        folder = os.path.join(os.getcwd(), "tests", "simulation_0")
        visualize_all(folder, scale_factor=1)

    # -- One-shot -------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_calls_show_multiple_times(self, mock_show, sim_folder):
        """One-shot: plt.show is called for each visualization step (≥ 3 times)."""
        visualize_all(str(sim_folder), scale_factor=1)
        # checkerboard + mesh + stress + deformation = 4 calls
        assert mock_show.call_count >= 3

    @patch(_PLT_SHOW)
    def test_scale_factor_two_does_not_crash(self, mock_show, sim_folder):
        """One-shot: non-unit scale_factor is accepted without error."""
        visualize_all(str(sim_folder), scale_factor=2)

    # -- Edge -----------------------------------------------------------------

    @patch(_PLT_SHOW)
    def test_missing_required_file_exits_early(self, mock_show, tmp_path):
        """Edge: folder missing required mesh files exits before crashing."""
        # Only write checkerboard — mesh computation will fail, returning None triple
        np.save(tmp_path / "checkerboard.npy", np.zeros((GRID_SIZE, GRID_SIZE)))
        # Should print an error and return, but NOT raise
        visualize_all(str(tmp_path), scale_factor=1)
        # plt.show may be called once (for checkerboard) then aborted
        # The test only asserts no exception was raised (implicit via reaching here)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
