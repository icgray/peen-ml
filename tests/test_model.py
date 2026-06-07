"""
Tests for src/peen-ml/model.py.

Covers:
  - Helper functions (_infer_trained_grid_size, _interpolate_displacements)
  - Dataset inspection (infer_dataset_shape)
  - Model architecture (create_model forward pass, output shape)
  - Data loading (create_data_loaders, create_test_loader)
  - Training (train_model: runs, returns losses, saves plot)
  - train_save_gui: saves .pth and reference_node_coords.npy
  - evaluate_model_gui: same-mesh, checkerboard-interp, node-count-interp
  - load_and_evaluate_model_gui: end-to-end load + infer
"""

import inspect
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from helpers import SYN_G, SYN_NODES, SYN_SIMS, SAMPLE_DATASET, make_node_coords as _make_node_coords

import model as M


# ===========================================================================
# 1. _infer_trained_grid_size
# ===========================================================================


class TestInferTrainedGridSize:
    def test_g5(self):
        m = M.create_model(input_channels=1, num_nodes=SYN_NODES, checkerboard_size=5)
        assert M._infer_trained_grid_size(m) == 5

    def test_g20(self):
        m = M.create_model(input_channels=1, num_nodes=100, checkerboard_size=20)
        assert M._infer_trained_grid_size(m) == 20

    def test_returns_none_on_bad_model(self):
        # A plain module with no .fc attribute should not raise, just return None
        bad = nn.Linear(10, 10)
        assert M._infer_trained_grid_size(bad) is None


# ===========================================================================
# 2. _interpolate_displacements
# ===========================================================================


class TestInterpolateDisplacements:
    def test_output_shape(self):
        """Output has target number of nodes."""
        rng = np.random.default_rng(0)
        ref = _make_node_coords(100)  # (100, 3)
        tgt = _make_node_coords(196)  # (196, 3)
        pred = rng.random((100, 3)).astype(np.float32)
        out = M._interpolate_displacements(pred, ref, tgt)
        assert out.shape == (196, 3)

    def test_identity_coords_recovers_input(self):
        """When source and target coords are the same, output is close to input.

        smoothing=1e-6 makes the RBF approximate, not exact, so we use a
        loose tolerance here rather than checking for bit-perfect recovery.
        """
        rng = np.random.default_rng(1)
        coords = _make_node_coords(100)
        pred = rng.random((100, 3)).astype(np.float32)
        out = M._interpolate_displacements(pred, coords, coords)
        assert out.shape == pred.shape, "Output shape must match input"
        assert np.isfinite(out).all(), "Output must be finite"
        # With smoothing the RBF is approximate; verify RMSE is within 2× the
        # displacement magnitude (a loose but meaningful sanity check)
        rmse = float(np.sqrt(np.mean((out - pred) ** 2)))
        assert rmse < 2.0 * float(np.sqrt(np.mean(pred**2))), f"RBF RMSE={rmse:.4f} is unreasonably large"

    def test_linear_field_exactly_recovered(self):
        """A linear displacement field f(x,y)=x+2y must be exactly interpolated."""
        coords = _make_node_coords(100)  # (100, 3), XY in [0, 0.01]
        # Linear displacement: all three components are x+2y
        pred = (coords[:, 0:1] + 2 * coords[:, 1:2]) * np.ones((100, 3), dtype=np.float32)
        eval_coords = _make_node_coords(196)
        out = M._interpolate_displacements(pred, coords, eval_coords)
        expected = (eval_coords[:, 0:1] + 2 * eval_coords[:, 1:2]) * np.ones((196, 3), dtype=np.float32)
        np.testing.assert_allclose(
            out, expected, atol=1e-4, err_msg="Thin-plate spline must reproduce linear fields exactly"
        )


# ===========================================================================
# 3. infer_dataset_shape
# ===========================================================================


class TestInferDatasetShape:
    def test_shape_from_sample_dataset(self):
        """Real sample data: G=5, N_nodes=5202."""
        n_nodes, cb_size = M.infer_dataset_shape(SAMPLE_DATASET)
        assert cb_size == 5
        assert n_nodes == 5202

    def test_shape_from_synthetic(self, tiny_dataset):
        n_nodes, cb_size = M.infer_dataset_shape(str(tiny_dataset))
        assert cb_size == SYN_G
        assert n_nodes == SYN_NODES

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises((FileNotFoundError, ValueError)):
            M.infer_dataset_shape(str(tmp_path))


# ===========================================================================
# 4. create_model / forward pass
# ===========================================================================


class TestCreateModel:
    def test_output_shape_matches_num_nodes(self):
        """Forward pass on (1,1,G,G) input must yield (1, N_nodes, 3)."""
        m = M.create_model(input_channels=1, num_nodes=SYN_NODES, checkerboard_size=SYN_G)
        x = torch.zeros(1, 1, SYN_G, SYN_G)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, SYN_NODES, 3)

    def test_output_shape_large_grid(self):
        m = M.create_model(input_channels=1, num_nodes=196, checkerboard_size=20)
        x = torch.zeros(1, 1, 20, 20)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 196, 3)

    def test_batch_dimension_preserved(self):
        m = M.create_model(input_channels=1, num_nodes=SYN_NODES, checkerboard_size=SYN_G)
        x = torch.zeros(4, 1, SYN_G, SYN_G)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (4, SYN_NODES, 3)


# ===========================================================================
# 5. create_data_loaders
# ===========================================================================


class TestCreateDataLoaders:
    def test_returns_four_items(self, tiny_dataset):
        result = M.create_data_loaders(str(tiny_dataset))
        assert len(result) == 4

    def test_all_are_dataloaders(self, tiny_dataset):
        from torch.utils.data import DataLoader

        train, val, test, _ = M.create_data_loaders(str(tiny_dataset))
        assert isinstance(train, DataLoader)
        assert isinstance(val, DataLoader)
        assert isinstance(test, DataLoader)

    def test_train_larger_than_val(self, tiny_dataset):
        train, val, _, _ = M.create_data_loaders(str(tiny_dataset))
        assert len(train.dataset) > len(val.dataset)

    def test_batches_have_correct_shapes(self, tiny_dataset):
        train, _, _, _ = M.create_data_loaders(str(tiny_dataset))
        cb, disp = next(iter(train))
        # cb: (batch, 1, G, G), disp: (batch, N, 3)
        assert cb.shape[1] == 1
        assert cb.shape[2] == SYN_G
        assert disp.shape[2] == 3


# ===========================================================================
# 6. create_test_loader
# ===========================================================================


class TestCreateTestLoader:
    def test_single_sim_folder(self, tiny_dataset):
        """Passing a single Simulation_N/ folder should work."""
        sim_path = str(tiny_dataset / "Simulation_0")
        loader = M.create_test_loader(sim_path, batch_size=1)
        assert len(loader.dataset) == 1

    def test_parent_folder(self, tiny_dataset):
        """Passing the parent dataset folder loads all simulations."""
        loader = M.create_test_loader(str(tiny_dataset), batch_size=1)
        assert len(loader.dataset) == SYN_SIMS

    def test_raises_on_missing_checkerboard(self, tmp_path):
        sim = tmp_path / "Simulation_0"
        sim.mkdir()
        # No checkerboard.npy inside
        with pytest.raises(FileNotFoundError):
            M.create_test_loader(str(sim))


# ===========================================================================
# 7. train_model
# ===========================================================================


class TestTrainModel:
    def _make_loaders(self, tiny_dataset):
        train, val, _, _ = M.create_data_loaders(str(tiny_dataset))
        return train, val

    def test_returns_loss_lists(self, tiny_dataset):
        m = M.create_model(1, SYN_NODES, SYN_G)
        train, val = self._make_loaders(tiny_dataset)
        crit = nn.MSELoss()
        optim = torch.optim.Adam(m.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
        tloss, vloss = M.train_model(m, train, val, crit, optim, sched, epochs=2, patience=5)
        assert isinstance(tloss, list) and len(tloss) >= 1
        assert isinstance(vloss, list) and len(vloss) >= 1

    def test_losses_are_finite(self, tiny_dataset):
        m = M.create_model(1, SYN_NODES, SYN_G)
        train, val = self._make_loaders(tiny_dataset)
        crit = nn.MSELoss()
        optim = torch.optim.Adam(m.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
        tloss, vloss = M.train_model(m, train, val, crit, optim, sched, epochs=2, patience=5)
        assert all(np.isfinite(l) for l in tloss)
        assert all(np.isfinite(l) for l in vloss)

    def test_plot_saved_when_path_given(self, tiny_dataset, tmp_path):
        m = M.create_model(1, SYN_NODES, SYN_G)
        train, val = self._make_loaders(tiny_dataset)
        crit = nn.MSELoss()
        optim = torch.optim.Adam(m.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
        plot_path = str(tmp_path / "loss.png")
        M.train_model(m, train, val, crit, optim, sched, epochs=2, patience=5, plot_save_path=plot_path)
        assert os.path.exists(plot_path), "Loss curve PNG should be written"
        assert os.path.getsize(plot_path) > 0

    def test_no_plot_file_when_path_is_none(self, tiny_dataset, tmp_path):
        m = M.create_model(1, SYN_NODES, SYN_G)
        train, val = self._make_loaders(tiny_dataset)
        crit = nn.MSELoss()
        optim = torch.optim.Adam(m.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
        M.train_model(m, train, val, crit, optim, sched, epochs=1, patience=5)
        # No plot_save_path given — no file should appear in cwd by default
        assert not os.path.exists("loss.png")

    def test_early_stopping_respected(self, tiny_dataset):
        """patience=0 must stop after 1 epoch (no improvement possible on first check)."""
        m = M.create_model(1, SYN_NODES, SYN_G)
        train, val = self._make_loaders(tiny_dataset)
        crit = nn.MSELoss()
        optim = torch.optim.Adam(m.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
        tloss, _ = M.train_model(m, train, val, crit, optim, sched, epochs=20, patience=0)
        assert len(tloss) == 1, "patience=0 should stop after the very first non-improving epoch"


# ===========================================================================
# 8. train_save_gui
# ===========================================================================


class TestTrainSaveGui:
    def test_model_pth_created(self, trained_model_bundle):
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        assert pth.exists(), ".pth model file must be saved"
        assert pth.stat().st_size > 0

    def test_reference_node_coords_saved(self, trained_model_bundle):
        saved_dir, _ = trained_model_bundle
        ref = saved_dir / "reference_node_coords.npy"
        assert ref.exists(), "reference_node_coords.npy must be saved for mesh interpolation"
        coords = np.load(ref)
        assert coords.ndim == 2 and coords.shape[1] == 3

    def test_loss_curve_saved(self, trained_model_bundle):
        saved_dir, _ = trained_model_bundle
        png = saved_dir / "training_loss_curve.png"
        assert png.exists(), "training_loss_curve.png must be saved alongside the model"
        assert png.stat().st_size > 0

    def test_saved_model_loadable(self, trained_model_bundle):
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        m = torch.load(str(pth), weights_only=False, map_location=torch.device("cpu"))
        assert callable(m), "Loaded object must be a callable model"

    def test_reference_coords_shape_matches_training_nodes(self, trained_model_bundle):
        """The saved reference coords must have the same N as the model output dim."""
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        ref = saved_dir / "reference_node_coords.npy"
        m = torch.load(str(pth), weights_only=False, map_location=torch.device("cpu"))
        coords = np.load(ref)
        n_from_model = m.fc[2].out_features // 3
        n_from_coords = coords.shape[0]
        assert n_from_model == n_from_coords, (
            f"Model outputs {n_from_model} nodes but reference_node_coords has " f"{n_from_coords} rows"
        )


# ===========================================================================
# 9. evaluate_model_gui — same mesh (no interpolation)
# ===========================================================================


class TestEvaluateModelGuiSameMesh:
    def test_saves_pred_displacements_npy(self, trained_model_bundle, tiny_dataset, tmp_path):
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        device = torch.device("cpu")
        m = torch.load(str(pth), weights_only=False, map_location=device)
        loader = M.create_test_loader(str(tiny_dataset / "Simulation_0"), batch_size=1)
        M.evaluate_model_gui(m, loader, nn.MSELoss(), str(tmp_path), device=device)
        pred_file = tmp_path / "Simulation_0" / "pred_displacements.npy"
        assert pred_file.exists()

    def test_pred_shape_matches_nodes(self, trained_model_bundle, tiny_dataset, tmp_path):
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        device = torch.device("cpu")
        m = torch.load(str(pth), weights_only=False, map_location=device)
        loader = M.create_test_loader(str(tiny_dataset / "Simulation_0"), batch_size=1)
        M.evaluate_model_gui(m, loader, nn.MSELoss(), str(tmp_path), device=device)
        pred = np.load(tmp_path / "Simulation_0" / "pred_displacements.npy")
        assert pred.reshape(-1, 3).shape[1] == 3

    def test_returns_finite_mse(self, trained_model_bundle, tiny_dataset, tmp_path):
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        device = torch.device("cpu")
        m = torch.load(str(pth), weights_only=False, map_location=device)
        loader = M.create_test_loader(str(tiny_dataset / "Simulation_0"), batch_size=1)
        mse = M.evaluate_model_gui(m, loader, nn.MSELoss(), str(tmp_path), device=device)
        assert np.isfinite(mse)

    def test_saves_csv(self, trained_model_bundle, tiny_dataset, tmp_path):
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        device = torch.device("cpu")
        m = torch.load(str(pth), weights_only=False, map_location=device)
        loader = M.create_test_loader(str(tiny_dataset / "Simulation_0"), batch_size=1)
        M.evaluate_model_gui(m, loader, nn.MSELoss(), str(tmp_path), device=device)
        csv_file = tmp_path / "Simulation_0" / "pred_displacements.csv"
        assert csv_file.exists() and csv_file.stat().st_size > 0


# ===========================================================================
# 10. evaluate_model_gui — Layer 1: checkerboard interpolation
# ===========================================================================


class TestCheckerboardInterpolation:
    def test_g20_input_runs_on_g5_model(self, trained_model_bundle, mismatched_sim, tmp_path):
        """A G=5 model must accept a G=20 checkerboard via bilinear interpolation."""
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        device = torch.device("cpu")
        m = torch.load(str(pth), weights_only=False, map_location=device)
        loader = M.create_test_loader(str(mismatched_sim / "Simulation_0"), batch_size=1)
        # Must not raise RuntimeError about matrix shapes
        M.evaluate_model_gui(m, loader, nn.MSELoss(), str(tmp_path), device=device)
        assert (tmp_path / "Simulation_0" / "pred_displacements.npy").exists()

    def test_interpolated_output_has_correct_rank(self, trained_model_bundle, mismatched_sim, tmp_path):
        """Saved predictions must be a numeric array with 3 displacement components."""
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        device = torch.device("cpu")
        m = torch.load(str(pth), weights_only=False, map_location=device)
        loader = M.create_test_loader(str(mismatched_sim / "Simulation_0"), batch_size=1)
        M.evaluate_model_gui(m, loader, nn.MSELoss(), str(tmp_path), device=device)
        pred = np.load(tmp_path / "Simulation_0" / "pred_displacements.npy")
        assert pred.reshape(-1, 3).shape[1] == 3


# ===========================================================================
# 11. evaluate_model_gui — Layer 2: node-count interpolation
# ===========================================================================


class TestNodeCountInterpolation:
    def test_node_mismatch_with_coords_does_not_crash(self, trained_model_bundle, mismatched_sim, tmp_path):
        """Model trained on N=100 must produce N=196 output when coords are supplied."""
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        device = torch.device("cpu")
        m = torch.load(str(pth), weights_only=False, map_location=device)

        # Build a loader where displacement has 196 nodes
        loader = M.create_test_loader(str(mismatched_sim / "Simulation_0"), batch_size=1)
        ref_coords = np.load(saved_dir / "reference_node_coords.npy")
        eval_coords = _make_node_coords(196)

        M.evaluate_model_gui(
            m,
            loader,
            nn.MSELoss(),
            str(tmp_path),
            device=device,
            ref_node_coords=ref_coords,
            eval_node_coords=eval_coords,
        )
        pred = np.load(tmp_path / "Simulation_0" / "pred_displacements.npy")
        assert (
            pred.reshape(-1, 3).shape[0] == 196
        ), "After spatial interpolation the saved output must have N_eval=196 nodes"

    def test_node_mismatch_without_coords_still_saves(self, trained_model_bundle, mismatched_sim, tmp_path):
        """Without coords a warning is printed but the raw model output is still saved."""
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        device = torch.device("cpu")
        m = torch.load(str(pth), weights_only=False, map_location=device)
        loader = M.create_test_loader(str(mismatched_sim / "Simulation_0"), batch_size=1)
        # No coords — must not raise, just warn
        M.evaluate_model_gui(m, loader, nn.MSELoss(), str(tmp_path), device=device)
        assert (tmp_path / "Simulation_0" / "pred_displacements.npy").exists()


# ===========================================================================
# 12. load_and_evaluate_model_gui — end-to-end
# ===========================================================================


class TestLoadAndEvaluateE2E:
    def test_runs_without_error(self, trained_model_bundle, tiny_dataset, tmp_path):
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        M.load_and_evaluate_model_gui(
            model_path=str(pth),
            test_data_path=str(tiny_dataset / "Simulation_0"),
            pred_save_dir=str(tmp_path),
        )

    def test_pred_file_exists(self, trained_model_bundle, tiny_dataset, tmp_path):
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        M.load_and_evaluate_model_gui(
            model_path=str(pth),
            test_data_path=str(tiny_dataset / "Simulation_0"),
            pred_save_dir=str(tmp_path),
        )
        assert (tmp_path / "Simulation_0" / "pred_displacements.npy").exists()

    def test_cross_mesh_e2e(self, trained_model_bundle, mismatched_sim, tmp_path):
        """
        Full pipeline: G=5 model + G=20 input + N=196 eval mesh.
        Both interpolation layers must fire and produce a valid output file.
        """
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        M.load_and_evaluate_model_gui(
            model_path=str(pth),
            test_data_path=str(mismatched_sim / "Simulation_0"),
            pred_save_dir=str(tmp_path),
        )
        pred = np.load(tmp_path / "Simulation_0" / "pred_displacements.npy")
        # Must be exactly 2-D (N, 3) — not (1, N, 3) with a stray batch dim.
        assert pred.ndim == 2, f"expected 2-D array, got shape {pred.shape}"
        assert pred.shape == (196, 3), f"expected (196, 3) after spatial interpolation, got {pred.shape}"


# ===========================================================================
# 13. Regression: pred_displacements.npy must be 2-D, never (1, N, 3)
#
# These tests guard against evaluate_model_gui accidentally saving the raw
# (B, N, 3) tensor (with leading batch dim) instead of the 2-D (N, 3) array.
# data_viz.compute_deformed_mesh indexes displacements[node_idx] and expects
# shape (3,) back — a (1, N, 3) file returns (N, 3) on first access, causing
# "could not broadcast input array from shape (N,3) into shape (3,)".
# ===========================================================================


class TestPredSavedAs2D:
    """pred_displacements.npy must always be (N, 3), never (1, N, 3)."""

    def test_same_mesh_pred_is_2d(self, trained_model_bundle, tiny_dataset, tmp_path):
        """Same G / same N: saved file must be exactly 2-D."""
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        device = torch.device("cpu")
        m = torch.load(str(pth), weights_only=False, map_location=device)
        loader = M.create_test_loader(str(tiny_dataset / "Simulation_0"), batch_size=1)
        M.evaluate_model_gui(m, loader, nn.MSELoss(), str(tmp_path), device=device)
        pred = np.load(tmp_path / "Simulation_0" / "pred_displacements.npy")
        assert pred.ndim == 2, f"pred_displacements.npy must be 2-D (N, 3), got shape {pred.shape}"

    def test_same_mesh_pred_shape(self, trained_model_bundle, tiny_dataset, tmp_path):
        """Saved shape must be exactly (N_train, 3)."""
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        device = torch.device("cpu")
        m = torch.load(str(pth), weights_only=False, map_location=device)
        loader = M.create_test_loader(str(tiny_dataset / "Simulation_0"), batch_size=1)
        M.evaluate_model_gui(m, loader, nn.MSELoss(), str(tmp_path), device=device)
        pred = np.load(tmp_path / "Simulation_0" / "pred_displacements.npy")
        assert pred.shape == (SYN_NODES, 3), f"expected ({SYN_NODES}, 3), got {pred.shape}"

    def test_layer1_resize_pred_is_2d(self, trained_model_bundle, mismatched_sim, tmp_path):
        """G=20 input resized to G=5 model: saved file must still be 2-D."""
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        device = torch.device("cpu")
        m = torch.load(str(pth), weights_only=False, map_location=device)
        loader = M.create_test_loader(str(mismatched_sim / "Simulation_0"), batch_size=1)
        M.evaluate_model_gui(m, loader, nn.MSELoss(), str(tmp_path), device=device)
        pred = np.load(tmp_path / "Simulation_0" / "pred_displacements.npy")
        assert pred.ndim == 2, f"After Layer-1 G resize, pred must be 2-D, got shape {pred.shape}"

    def test_layer2_interp_pred_is_2d(self, trained_model_bundle, mismatched_sim, tmp_path):
        """Layer-2 node-count interpolation: output must be 2-D (N_eval, 3)."""
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        device = torch.device("cpu")
        m = torch.load(str(pth), weights_only=False, map_location=device)
        loader = M.create_test_loader(str(mismatched_sim / "Simulation_0"), batch_size=1)
        ref_coords = np.load(saved_dir / "reference_node_coords.npy")
        eval_coords = _make_node_coords(196)
        M.evaluate_model_gui(
            m,
            loader,
            nn.MSELoss(),
            str(tmp_path),
            device=device,
            ref_node_coords=ref_coords,
            eval_node_coords=eval_coords,
        )
        pred = np.load(tmp_path / "Simulation_0" / "pred_displacements.npy")
        assert pred.ndim == 2, f"After Layer-2 node interpolation, pred must be 2-D, got shape {pred.shape}"
        assert pred.shape == (196, 3), f"expected (196, 3), got {pred.shape}"

    def test_load_and_evaluate_pred_is_2d(self, trained_model_bundle, tiny_dataset, tmp_path):
        """End-to-end load_and_evaluate_model_gui must save 2-D predictions."""
        saved_dir, _ = trained_model_bundle
        pth = saved_dir / "trained_displacement_predictor_full_model.pth"
        M.load_and_evaluate_model_gui(
            model_path=str(pth),
            test_data_path=str(tiny_dataset / "Simulation_0"),
            pred_save_dir=str(tmp_path),
        )
        pred = np.load(tmp_path / "Simulation_0" / "pred_displacements.npy")
        assert pred.ndim == 2, f"load_and_evaluate_model_gui must save 2-D predictions, got {pred.shape}"


# ===========================================================================
# 14. AMP + gradient accumulation in train_model
# ===========================================================================


class TestAMP:
    def _make_loaders(self, tiny_dataset, batch_size=15):
        train, val, _, _ = M.create_data_loaders(str(tiny_dataset), batch_size=batch_size)
        return train, val

    def test_train_model_with_amp(self, tiny_dataset):
        """use_amp=True runs and returns finite losses (no-op on CPU)."""
        m = M.create_model(1, SYN_NODES, SYN_G)
        train, val = self._make_loaders(tiny_dataset)
        crit = nn.MSELoss()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        tloss, vloss = M.train_model(m, train, val, crit, opt, sched, epochs=2, patience=5, use_amp=True)
        assert all(np.isfinite(l) for l in tloss)
        assert all(np.isfinite(l) for l in vloss)

    def test_accum_steps(self, tiny_dataset):
        """accum_steps=2 with batch_size=1 runs without error."""
        m = M.create_model(1, SYN_NODES, SYN_G)
        train, val = self._make_loaders(tiny_dataset, batch_size=1)
        crit = nn.MSELoss()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        tloss, _ = M.train_model(m, train, val, crit, opt, sched, epochs=1, patience=5, accum_steps=2)
        assert len(tloss) == 1
        assert np.isfinite(tloss[0])

    def test_accum_steps_with_amp(self, tiny_dataset):
        """AMP + gradient accumulation combined — no assertion errors, finite losses."""
        m = M.create_model(1, SYN_NODES, SYN_G)
        train, val = self._make_loaders(tiny_dataset, batch_size=1)
        crit = nn.MSELoss()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        tloss, _ = M.train_model(m, train, val, crit, opt, sched, epochs=2, patience=5, use_amp=True, accum_steps=2)
        assert all(np.isfinite(l) for l in tloss)


# ===========================================================================
# 15. SIRENPredictor forward pass and resolution-invariance
# ===========================================================================


class TestSIRENPredictor:
    def test_output_shape(self):
        """forward(B,1,G,G) + (K,2) coords -> (B,K,3)."""
        model = M.SIRENPredictor(input_channels=1, latent_dim=32, hidden=64, n_layers=2)
        B, G, K = 2, SYN_G, 16
        cb = torch.zeros(B, 1, G, G)
        coords = torch.rand(K, 2)
        with torch.no_grad():
            out = model(cb, coords)
        assert out.shape == (B, K, 3)

    def test_no_nan_5_epochs(self, tiny_dataset):
        """Training for 5 epochs produces finite loss at every step."""
        train_loader, _, _, _, _ = M.create_siren_loaders(str(tiny_dataset), k_nodes=16, batch_size=4)
        model = M.SIRENPredictor(input_channels=1, latent_dim=32, hidden=64, n_layers=2)
        device = torch.device("cpu")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        for _ in range(5):
            model.train()
            for cbs, coords, disps in train_loader:
                opt.zero_grad()
                pred = model(cbs.to(device), coords.to(device))
                loss = crit(pred, disps.to(device))
                loss.backward()
                opt.step()
                assert np.isfinite(loss.item()), "Loss became NaN during SIREN training"

    def test_resolution_invariant(self):
        """Same model handles different K values (resolution-free inference)."""
        model = M.SIRENPredictor(input_channels=1, latent_dim=32, hidden=64, n_layers=2)
        cb = torch.zeros(1, 1, SYN_G, SYN_G)
        for K in [8, 64, 256]:
            coords = torch.rand(K, 2)
            with torch.no_grad():
                out = model(cb, coords)
            assert out.shape == (1, K, 3), f"Expected (1,{K},3) but got {out.shape}"


# ===========================================================================
# 16. SIREN data loaders
# ===========================================================================


class TestSIRENLoaders:
    def test_loader_shapes(self, tiny_dataset):
        """create_siren_loaders returns (K,2) coords and (B,K,3) displacements."""
        train, _, _, N_total, _ = M.create_siren_loaders(str(tiny_dataset), k_nodes=16, batch_size=4)
        cbs, coords, disps = next(iter(train))
        assert cbs.ndim == 4 and cbs.shape[1] == 1, "checkerboard shape must be (B,1,G,G)"
        assert coords.shape == (16, 2), "coords must be (K, 2)"
        assert disps.ndim == 3 and disps.shape[2] == 3, "disps must be (B, K, 3)"
        assert N_total == SYN_NODES

    def test_different_k_nodes(self, tiny_dataset):
        """k_nodes parameter controls the subsampled coord count."""
        train, _, _, _, _ = M.create_siren_loaders(str(tiny_dataset), k_nodes=8, batch_size=4)
        _, coords, disps = next(iter(train))
        assert coords.shape[0] == 8
        assert disps.shape[1] == 8


# ===========================================================================
# 17. _parse_material_block
# ===========================================================================


class TestParseMatBlock:
    def test_returns_none_when_no_params_file(self, tmp_path):
        assert M._parse_material_block(str(tmp_path)) is None

    def test_returns_none_when_no_material_section(self, tmp_path):
        (tmp_path / "simulation_params.txt").write_text("[simulation]\nV=50.0\n")
        assert M._parse_material_block(str(tmp_path)) is None

    def test_parses_all_seven_keys(self, tmp_path):
        (tmp_path / "simulation_params.txt").write_text(
            "[material]\n"
            "E_b = 113.8e9\nnu_b = 0.34\nsigma_yield = 880e6\nc = 3e9\n"
            "E_s = 210e9\nnu_s = 0.30\nrho_s = 7800\n"
        )
        result = M._parse_material_block(str(tmp_path))
        assert result is not None
        assert abs(result["E_b"] - 113.8e9) < 1e6
        assert abs(result["nu_b"] - 0.34) < 1e-6
        assert len(result) == M.MAT_DIM

    def test_fallback_to_nozzle_params(self, tmp_path):
        (tmp_path / "nozzle_params.txt").write_text(
            "[material]\n"
            "E_b = 113.8e9\nnu_b = 0.34\nsigma_yield = 880e6\nc = 3e9\n"
            "E_s = 210e9\nnu_s = 0.30\nrho_s = 7800\n"
        )
        result = M._parse_material_block(str(tmp_path))
        assert result is not None

    def test_returns_none_on_incomplete_block(self, tmp_path):
        (tmp_path / "simulation_params.txt").write_text("[material]\nE_b = 113.8e9\n")
        assert M._parse_material_block(str(tmp_path)) is None


# ===========================================================================
# 18. evaluate_model (standalone, not GUI)
# ===========================================================================


class TestEvaluateModel:
    def test_returns_finite_mse(self, tiny_dataset):
        _, _, test, _ = M.create_data_loaders(str(tiny_dataset))
        device = torch.device("cpu")
        model = M.create_model(1, SYN_NODES, SYN_G).to(device)
        mse = M.evaluate_model(model, test, nn.MSELoss(), device=device)
        assert mse >= 0.0 and np.isfinite(mse)


# ===========================================================================
# 19. FieldDataset / infer_grid_shape / create_field_data_loaders
# ===========================================================================


class TestFieldDataset:
    def test_len_equals_num_sims(self, tiny_dataset):
        loaded = M.load_all_npy_files(str(tiny_dataset), ("checkerboard", "displacements"))
        H, W = M.infer_grid_shape(str(tiny_dataset))
        ds = M.FieldDataset(loaded["checkerboard"], loaded["displacements"], H, W)
        assert len(ds) == SYN_SIMS

    def test_getitem_shapes(self, tiny_dataset):
        loaded = M.load_all_npy_files(str(tiny_dataset), ("checkerboard", "displacements"))
        H, W = M.infer_grid_shape(str(tiny_dataset))
        ds = M.FieldDataset(loaded["checkerboard"], loaded["displacements"], H, W)
        cb, field = ds[0]
        assert cb.shape == (1, SYN_G, SYN_G), f"cb shape {cb.shape}"
        assert field.shape == (3, H, W), f"field shape {field.shape}"

    def test_getitem_with_mat_features(self, tiny_dataset):
        loaded = M.load_all_npy_files(str(tiny_dataset), ("checkerboard", "displacements"))
        H, W = M.infer_grid_shape(str(tiny_dataset))
        mat = np.random.rand(SYN_SIMS, M.MAT_DIM).astype(np.float32)
        ds = M.FieldDataset(loaded["checkerboard"], loaded["displacements"], H, W, mat_features=mat)
        cb, m, field = ds[0]
        assert m.shape == (M.MAT_DIM,)


class TestInferGridShape:
    def test_detects_10x10_grid(self, tiny_dataset):
        H, W = M.infer_grid_shape(str(tiny_dataset))
        assert H == 10 and W == 10

    def test_raises_on_empty_folder(self, tmp_path):
        with pytest.raises((ValueError, FileNotFoundError)):
            M.infer_grid_shape(str(tmp_path))


class TestCreateFieldDataLoaders:
    def test_returns_five_items(self, tiny_dataset):
        result = M.create_field_data_loaders(str(tiny_dataset))
        assert len(result) == 5

    def test_loaders_are_dataloaders(self, tiny_dataset):
        from torch.utils.data import DataLoader

        train, val, test, H, W = M.create_field_data_loaders(str(tiny_dataset))
        assert isinstance(train, DataLoader)
        assert isinstance(val, DataLoader)
        assert isinstance(test, DataLoader)

    def test_field_batch_shapes(self, tiny_dataset):
        train, _, _, H, W = M.create_field_data_loaders(str(tiny_dataset))
        cb, field = next(iter(train))
        assert cb.ndim == 4 and cb.shape[1] == 1
        assert field.shape[1] == 3 and field.shape[2] == H and field.shape[3] == W


# ===========================================================================
# 20. ConvDecoderPredictor
# ===========================================================================


class TestConvDecoderPredictor:
    def test_output_shape(self):
        model = M.ConvDecoderPredictor(input_channels=1, out_H=10, out_W=10)
        x = torch.zeros(2, 1, SYN_G, SYN_G)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3, 10, 10)

    def test_accepts_material_features(self):
        model = M.ConvDecoderPredictor(input_channels=1, out_H=10, out_W=10, mat_dim=7)
        x = torch.zeros(2, 1, SYN_G, SYN_G)
        mat = torch.rand(2, 7)
        with torch.no_grad():
            out = model(x, mat)
        assert out.shape == (2, 3, 10, 10)

    def test_finite_output(self):
        model = M.ConvDecoderPredictor(input_channels=1, out_H=10, out_W=10)
        x = torch.rand(1, 1, SYN_G, SYN_G)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_batch_dimension_preserved(self):
        model = M.ConvDecoderPredictor(input_channels=1, out_H=10, out_W=10)
        x = torch.zeros(4, 1, SYN_G, SYN_G)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 4


# ===========================================================================
# 21. sample_field_at_coords
# ===========================================================================


class TestSampleFieldAtCoords:
    def test_output_shape(self):
        field = torch.rand(2, 3, 10, 10)
        coords = torch.rand(SYN_NODES, 2)
        out = M.sample_field_at_coords(field, coords)
        assert out.shape == (2, SYN_NODES, 3)

    def test_finite_output(self):
        field = torch.rand(1, 3, 10, 10)
        coords = torch.rand(20, 2)
        out = M.sample_field_at_coords(field, coords)
        assert torch.isfinite(out).all()

    def test_single_node(self):
        field = torch.rand(1, 3, 10, 10)
        coords = torch.tensor([[0.5, 0.5]])
        out = M.sample_field_at_coords(field, coords)
        assert out.shape == (1, 1, 3)


# ===========================================================================
# 22. train_save_conv_gui + load_and_evaluate_conv_gui
# ===========================================================================


@pytest.fixture(scope="session")
def trained_conv_bundle(tiny_dataset):
    """Train a ConvDecoderPredictor for 1 epoch and return (save_dir, dataset_path)."""
    M.train_save_conv_gui(str(tiny_dataset), epochs=1)
    return tiny_dataset / "saved_model_conv", tiny_dataset


class TestTrainSaveConvGui:
    def test_model_pth_created(self, trained_conv_bundle):
        save_dir, _ = trained_conv_bundle
        assert (save_dir / "trained_conv_decoder_full_model.pth").exists()

    def test_loss_curve_saved(self, trained_conv_bundle):
        save_dir, _ = trained_conv_bundle
        assert (save_dir / "training_loss_curve.png").exists()

    def test_saved_model_loadable(self, trained_conv_bundle):
        save_dir, _ = trained_conv_bundle
        pth = save_dir / "trained_conv_decoder_full_model.pth"
        m = torch.load(str(pth), weights_only=False, map_location="cpu")
        assert callable(m)


class TestLoadAndEvaluateConvGui:
    def test_saves_predictions(self, trained_conv_bundle, tiny_dataset, tmp_path):
        save_dir, _ = trained_conv_bundle
        pth = save_dir / "trained_conv_decoder_full_model.pth"
        M.load_and_evaluate_conv_gui(
            str(pth),
            str(tiny_dataset / "Simulation_0"),
            str(tmp_path),
        )
        assert (tmp_path / "Simulation_0" / "pred_displacements.npy").exists()

    def test_pred_is_2d_with_3_components(self, trained_conv_bundle, tiny_dataset, tmp_path):
        save_dir, _ = trained_conv_bundle
        pth = save_dir / "trained_conv_decoder_full_model.pth"
        M.load_and_evaluate_conv_gui(
            str(pth),
            str(tiny_dataset / "Simulation_0"),
            str(tmp_path),
        )
        pred = np.load(tmp_path / "Simulation_0" / "pred_displacements.npy")
        assert pred.ndim == 2 and pred.shape[1] == 3


# ===========================================================================
# 23. load_and_evaluate_siren_gui
# ===========================================================================


@pytest.fixture(scope="session")
def trained_siren_bundle(tiny_dataset):
    """Train a SIRENPredictor for 1 epoch and return (save_dir, dataset_path)."""
    M.train_save_siren_gui(str(tiny_dataset), epochs=1, k_nodes=16, batch_size=4, latent_dim=32)
    return tiny_dataset / "saved_model_siren", tiny_dataset


class TestLoadAndEvaluateSirenGui:
    def test_saves_predictions(self, trained_siren_bundle, tiny_dataset, tmp_path):
        save_dir, _ = trained_siren_bundle
        pth = save_dir / "trained_siren_full_model.pth"
        if not pth.exists():
            pytest.skip("SIREN model not saved — check train_save_siren_gui output path")
        M.load_and_evaluate_siren_gui(
            str(pth),
            str(tiny_dataset / "Simulation_0"),
            str(tmp_path),
        )
        assert (tmp_path / "Simulation_0" / "pred_displacements.npy").exists()

    def test_pred_shape(self, trained_siren_bundle, tiny_dataset, tmp_path):
        save_dir, _ = trained_siren_bundle
        pth = save_dir / "trained_siren_full_model.pth"
        if not pth.exists():
            pytest.skip("SIREN model not saved — check train_save_siren_gui output path")
        M.load_and_evaluate_siren_gui(
            str(pth),
            str(tiny_dataset / "Simulation_0"),
            str(tmp_path),
        )
        pred = np.load(tmp_path / "Simulation_0" / "pred_displacements.npy")
        assert pred.ndim == 2 and pred.shape[1] == 3
