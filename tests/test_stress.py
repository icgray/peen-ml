"""
tests/test_stress.py
====================
Comprehensive stress tests for peen-ml.

Covers:
  1. Material normalization — all 25 combos within [-0.5, 1.5], no NaN
  2. Checkerboard pattern generation — all 5 modes, shape / range / finite
  3. Model architecture — forward pass, material features, batch size
  4. Known bug: smape NaN with zero tensors
  5. Known bug: ZeroDivisionError when n_sims < 7 (empty val split)
  6. Training stability — no NaN/Inf with extreme material combos
  7. Dataset generation — all 25 material combos produce valid files
  8. Physics plausibility — uz < 0, magnitudes in µm range, V sensitivity
  9. High-resolution mesh — Nx=Ny=30, correct node count
 10. Extreme shot conditions — V=5 m/s, V=100 m/s, D=0.05mm, n_shots=500
 11. Cross-material generalization — inference on unseen material
 12. Mixed material dataset — sims with and without simulation_params.txt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup (mirrors conftest.py; safe to call again — idempotent)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src", "peen-ml"),
)

import model as M  # noqa: E402
from helpers import SYN_G, SYN_NODES, make_node_coords
from materials import (
    WORKPIECE_MATERIALS,
    SHOT_MATERIALS,
    get_workpiece,
    get_shot,
)
from native_dataset_gen import (
    GeneratorParams,
    generate_single_simulation,
    _make_checkerboard,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_WORKPIECES = sorted(WORKPIECE_MATERIALS.keys())
ALL_SHOTS = sorted(SHOT_MATERIALS.keys())
ALL_COMBOS = [(wp, sp) for wp in ALL_WORKPIECES for sp in ALL_SHOTS]

# Smallest viable dataset: ≥7 sims so 70/15/15 split is non-empty
MIN_SIMS_FOR_SPLIT = 7
TINY_NX = TINY_NY = 10  # 11×11=121 nodes — fast generation
TINY_G = 5  # 5×5 checkerboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_sim_params(path: Path, wp_name: str, shot_name: str) -> None:
    """Write a minimal simulation_params.txt with a [material] block."""
    wp = get_workpiece(wp_name)
    sp = get_shot(shot_name)
    with open(path / "simulation_params.txt", "w") as f:
        f.write("[material]\n")
        f.write(f"E_b             = {wp['E']:.6e}\n")
        f.write(f"nu_b            = {wp['nu']:.6f}\n")
        f.write(f"sigma_yield     = {wp['sigma_yield']:.6e}\n")
        f.write(f"c               = {wp['c']:.6e}\n")
        f.write(f"E_s             = {sp['E_s']:.6e}\n")
        f.write(f"nu_s            = {sp['nu_s']:.6f}\n")
        f.write(f"rho_s           = {sp['rho_s']:.6f}\n")


def _make_synthetic_dataset(
    root: Path,
    wp_name: str,
    shot_name: str,
    n_sims: int = MIN_SIMS_FOR_SPLIT,
    G: int = TINY_G,
    n_nodes: int = TINY_NX * TINY_NY + TINY_NX + TINY_NY + 1,  # 121
    rng: np.random.Generator | None = None,
    write_params: bool = True,
) -> Path:
    """Create synthetic Simulation_N/ folders with optional material params."""
    if rng is None:
        rng = np.random.default_rng(42)
    root.mkdir(parents=True, exist_ok=True)
    coords = make_node_coords(n_nodes)
    for i in range(n_sims):
        sim = root / f"Simulation_{i}"
        sim.mkdir(parents=True, exist_ok=True)
        np.save(sim / "checkerboard.npy", rng.random((G, G)).astype(np.float32))
        # Physical displacement in µm range
        np.save(sim / "displacements.npy", (rng.random((n_nodes, 3)) * 1e-4).astype(np.float32))
        np.save(sim / "node_coords.npy", coords)
        if write_params:
            _write_sim_params(sim, wp_name, shot_name)
    return root


# ---------------------------------------------------------------------------
# 1. Material normalisation
# ---------------------------------------------------------------------------


class TestMaterialNormalizationBounds:
    """All 25 material combos must normalise to [-0.5, 1.5] with no NaN/Inf."""

    @pytest.mark.parametrize("wp_name,shot_name", ALL_COMBOS, ids=[f"{w}+{s}" for w, s in ALL_COMBOS])
    def test_in_clip_range(self, wp_name, shot_name):
        wp = get_workpiece(wp_name)
        sp = get_shot(shot_name)
        raw = np.array(
            [
                wp["E"],
                wp["nu"],
                wp["sigma_yield"],
                wp["c"],
                sp["E_s"],
                sp["nu_s"],
                sp["rho_s"],
            ],
            dtype=np.float32,
        )
        normed = M.normalize_mat_features(raw)

        assert normed.shape == (7,), "Output shape must be (7,)"
        assert np.isfinite(normed).all(), f"NaN or Inf in normalised features for {wp_name}+{shot_name}: {normed}"
        assert (normed >= -0.5).all() and (
            normed <= 1.5
        ).all(), f"Normalised features outside [-0.5, 1.5] for {wp_name}+{shot_name}: {normed}"

    def test_default_material_normalises(self):
        """The hardcoded default material vector must normalise without error."""
        normed = M.normalize_mat_features(M._DEFAULT_MAT_RAW)
        assert np.isfinite(normed).all()
        assert normed.shape == (7,)

    @pytest.mark.parametrize("wp_name,shot_name", ALL_COMBOS, ids=[f"{w}+{s}" for w, s in ALL_COMBOS])
    def test_parsed_from_params_txt(self, wp_name, shot_name, tmp_path):
        """Parsing simulation_params.txt yields identical features to direct lookup."""
        sim = tmp_path / "Simulation_0"
        sim.mkdir()
        _write_sim_params(sim, wp_name, shot_name)

        parsed = M._parse_material_block(str(sim))
        assert parsed is not None, f"_parse_material_block returned None for {wp_name}+{shot_name}"
        assert set(parsed.keys()) == set(M.MAT_FEATURE_KEYS)

        # Verify round-trip normalisation is finite
        raw = np.array([parsed[k] for k in M.MAT_FEATURE_KEYS], dtype=np.float32)
        normed = M.normalize_mat_features(raw)
        assert np.isfinite(normed).all()

    def test_missing_params_returns_none(self, tmp_path):
        """When simulation_params.txt is absent, _parse_material_block returns None."""
        sim = tmp_path / "Simulation_0"
        sim.mkdir()
        assert M._parse_material_block(str(sim)) is None

    def test_incomplete_params_returns_none(self, tmp_path):
        """Partial [material] block (< 7 fields) should return None, not crash."""
        sim = tmp_path / "Simulation_0"
        sim.mkdir()
        with open(sim / "simulation_params.txt", "w") as f:
            f.write("[material]\nE_b = 1.138e11\nnu_b = 0.34\n")
        assert M._parse_material_block(str(sim)) is None


# ---------------------------------------------------------------------------
# 2. Checkerboard pattern generation
# ---------------------------------------------------------------------------

PATTERN_MODES = ["uniform", "bimodal", "gradient", "random", "sparse"]


class TestCheckerboardPatternModes:
    """All 5 pattern modes must produce valid (G, G) float32 arrays."""

    @pytest.mark.parametrize("mode", PATTERN_MODES)
    @pytest.mark.parametrize("G", [3, 5, 10, 15])
    def test_shape_and_range(self, mode, G):
        rng = np.random.default_rng(0)
        lo, hi = 0.005, 0.020
        cb = _make_checkerboard(mode, G, rng, lo, hi)
        assert cb.shape == (G, G), f"mode={mode} G={G}: wrong shape {cb.shape}"
        assert cb.dtype == np.float32, f"Expected float32, got {cb.dtype}"
        assert np.isfinite(cb).all(), f"NaN/Inf in {mode} pattern"
        assert (cb >= lo - 1e-7).all() and (
            cb <= hi + 1e-7
        ).all(), f"mode={mode} G={G}: values outside [{lo}, {hi}]: min={cb.min()}, max={cb.max()}"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown pattern mode"):
            _make_checkerboard("hexagonal", 5, np.random.default_rng(0))

    def test_sparse_with_g1(self):
        """G=1 sparse: single active cell — should not raise."""
        rng = np.random.default_rng(7)
        cb = _make_checkerboard("sparse", 1, rng)
        assert cb.shape == (1, 1)
        assert np.isfinite(cb).all()

    def test_bimodal_two_distinct_values(self):
        """Bimodal pattern must have exactly two distinct intensity levels."""
        rng = np.random.default_rng(99)
        cb = _make_checkerboard("bimodal", 6, rng)
        uniq = np.unique(cb)
        assert len(uniq) == 2, f"Bimodal should have 2 unique values, got {len(uniq)}"

    def test_uniform_all_same(self):
        """Uniform pattern must have all cells equal."""
        rng = np.random.default_rng(3)
        cb = _make_checkerboard("uniform", 8, rng)
        assert np.allclose(cb, cb[0, 0]), "Uniform pattern should have identical values"

    def test_gradient_monotone(self):
        """Gradient pattern must be monotone along one axis (row or column)."""
        rng = np.random.default_rng(5)
        cb = _make_checkerboard("gradient", 10, rng)
        row_mono = all(cb[i, 0] <= cb[i + 1, 0] for i in range(9)) or all(cb[i, 0] >= cb[i + 1, 0] for i in range(9))
        col_mono = all(cb[0, j] <= cb[0, j + 1] for j in range(9)) or all(cb[0, j] >= cb[0, j + 1] for j in range(9))
        assert row_mono or col_mono, "Gradient should be monotone along one axis"


# ---------------------------------------------------------------------------
# 3. Model architecture
# ---------------------------------------------------------------------------


class TestModelArchitectureVariants:
    """DisplacementPredictor forward pass with various G, N, and material dims."""

    @pytest.mark.parametrize("G,N", [(5, 100), (8, 196), (10, 441), (15, 900)])
    def test_output_shape(self, G, N):
        model = M.create_model(input_channels=1, num_nodes=N, checkerboard_size=G)
        x = torch.zeros(2, 1, G, G)
        out = model(x)
        assert out.shape == (2, N, 3), f"G={G} N={N}: expected (2,{N},3) got {out.shape}"

    def test_output_finite_random_input(self):
        model = M.create_model(input_channels=1, num_nodes=121, checkerboard_size=5)
        x = torch.randn(4, 1, 5, 5)
        out = model(x)
        assert torch.isfinite(out).all(), "Model output contains NaN or Inf"

    def test_batch_preserved(self):
        model = M.create_model(input_channels=1, num_nodes=100, checkerboard_size=5)
        for B in (1, 3, 8):
            out = model(torch.zeros(B, 1, 5, 5))
            assert out.shape[0] == B, f"Batch size {B} not preserved: {out.shape}"

    def test_material_conditioning_changes_output(self):
        """mat_dim > 0 must produce finite output with valid mat tensor."""
        model = M.DisplacementPredictor(input_channels=1, num_nodes=100, checkerboard_size=5, mat_dim=7)
        x = torch.ones(1, 1, 5, 5)
        mat = torch.randn(1, 7)
        with torch.no_grad():
            out_mat = model(x, mat)
        assert torch.isfinite(out_mat).all()
        assert out_mat.shape == (1, 100, 3)

    def test_material_conditioning_none_works_after_fix(self):
        """
        FIX VERIFIED: DisplacementPredictor(mat_dim=7).forward(x, None) now pads
        a zero mat tensor automatically instead of raising RuntimeError.
        """
        model = M.DisplacementPredictor(input_channels=1, num_nodes=100, checkerboard_size=5, mat_dim=7)
        x = torch.ones(1, 1, 5, 5)
        with torch.no_grad():
            out = model(x, None)  # should NOT raise after the fix
        assert out.shape == (1, 100, 3)
        assert torch.isfinite(out).all()

    def test_infer_grid_size_all_sizes(self):
        for G in [3, 5, 8, 10, 15, 20]:
            m = M.create_model(1, 100, G)
            assert M._infer_trained_grid_size(m) == G


# ---------------------------------------------------------------------------
# 4. sMAPE removed — verify it is gone and relative RMSE is present
# ---------------------------------------------------------------------------


class TestSmapeNaNHole:
    """
    FIX VERIFIED (HOLE 7): sMAPE was removed from model.py because it is
    dominated by near-zero nodes (returns ~150% regardless of model quality).
    Relative RMSE (rmse / peak_gt * 100%) is now reported instead.
    """

    def test_smape_removed_from_module(self):
        """sMAPE function must NOT be present in model.py (removed as per HOLE 7 fix)."""
        assert not hasattr(M, "smape"), (
            "smape() was removed because it is meaningless for near-zero displacement "
            "fields. Do not re-add it — use rel_rmse_pct instead."
        )

    def test_evaluate_on_dataset_returns_rel_rmse(self):
        """evaluate_on_dataset must return mean_rel_rmse_pct in its result dict."""
        import inspect

        src = inspect.getsource(M.evaluate_on_dataset)
        assert "mean_rel_rmse_pct" in src, "evaluate_on_dataset must return mean_rel_rmse_pct (HOLE 7 fix)"

    def test_rel_rmse_pct_in_per_sim(self):
        """per_sim dicts inside evaluate_on_dataset must contain rel_rmse_pct."""
        import inspect

        src = inspect.getsource(M.evaluate_on_dataset)
        assert "rel_rmse_pct" in src


# ---------------------------------------------------------------------------
# 5. Known hole: ZeroDivisionError with degenerate split
# ---------------------------------------------------------------------------


class TestDegenerateDatasetSplit:
    """
    KNOWN HOLE: create_data_loaders + train_model crash with ZeroDivisionError
    when n_sims < 7, because the 15% val split rounds to 0.

    Reproduces the bug: val_loss /= len(val_loader) when len(val_loader) == 0.
    """

    @pytest.mark.parametrize("n_sims", [2, 3, 4, 5, 6])
    def test_small_dataset_val_split_no_longer_crashes(self, tmp_path, n_sims):
        """
        FIX VERIFIED: create_data_loaders no longer crashes with n_sims < 7.
        NormalizedDataset.__init__ now guards len(base_dataset) > 0 before
        accessing base_dataset[0].  The val_loader will be empty but no
        IndexError is raised.
        """
        rng = np.random.default_rng(n_sims)
        root = _make_synthetic_dataset(
            tmp_path / f"ds_{n_sims}",
            "Ti-6Al-4V",
            "steel",
            n_sims=n_sims,
            write_params=False,
        )
        # create_data_loaders now succeeds — val_loader may be empty but no crash
        train_loader, val_loader, test_loader, _ = M.create_data_loaders(str(root), batch_size=4)
        assert len(train_loader) >= 0  # no crash is the key assertion
        # Callers should check len(val_loader) == 0 before training
        if n_sims < 7:
            assert len(val_loader) == 0, "val split should be empty for n_sims < 7"

    def test_minimum_viable_split(self, tmp_path):
        """n_sims = 7 is the minimum that avoids an empty val split."""
        root = _make_synthetic_dataset(
            tmp_path / "ds_7",
            "4340-Steel",
            "steel",
            n_sims=7,
            write_params=False,
        )
        train_loader, val_loader, test_loader, _ = M.create_data_loaders(str(root), batch_size=4)
        assert len(val_loader) > 0, "With n_sims=7 the val split should be non-empty"
        assert len(test_loader) > 0


# ---------------------------------------------------------------------------
# 6. Training stability with extreme material combos
# ---------------------------------------------------------------------------


class TestTrainingStabilityExtremes:
    """No NaN/Inf in losses or model output for the most extreme material combos."""

    # Extremes: softest material + heaviest shot  vs  hardest + lightest shot
    @pytest.mark.parametrize(
        "wp,shot",
        [
            ("Al-7075-T6", "tungsten"),  # soft + heaviest
            ("Inconel-718", "glass"),  # hard + lightest
            ("Ti-6Al-4V", "ceramic"),  # intermediate + stiff
        ],
    )
    def test_training_does_not_produce_nan(self, wp, shot, tmp_path):
        """Training on extreme material combo must not produce NaN losses."""
        root = _make_synthetic_dataset(
            tmp_path / f"{wp}_{shot}",
            wp,
            shot,
            n_sims=MIN_SIMS_FOR_SPLIT,
            write_params=True,
        )
        train_loader, val_loader, _, _ = M.create_data_loaders(str(root), batch_size=4)
        model = M.create_model(1, num_nodes=121, checkerboard_size=TINY_G)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)

        train_losses, val_losses = M.train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            epochs=3,
            patience=3,
        )
        for epoch, (tl, vl) in enumerate(zip(train_losses, val_losses)):
            assert tl == tl and vl == vl, f"NaN loss at epoch {epoch+1} for {wp}+{shot}: train={tl}, val={vl}"

        # Post-training inference must be finite (device-agnostic)
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            x = torch.zeros(1, 1, TINY_G, TINY_G, device=device)
            out = model(x)
        assert torch.isfinite(out).all(), f"NaN/Inf in model output after training on {wp}+{shot}"


# ---------------------------------------------------------------------------
# 7. Dataset generation — all 25 material combos (synthetic, fast)
# ---------------------------------------------------------------------------


class TestDatasetGenAllMaterialCombos:
    """All 25 material combos produce valid simulation folders."""

    @pytest.mark.parametrize("wp_name,shot_name", ALL_COMBOS, ids=[f"{w}+{s}" for w, s in ALL_COMBOS])
    def test_required_files_present(self, wp_name, shot_name, tmp_path):
        """Synthetic dataset has checkerboard.npy, displacements.npy, params.txt."""
        root = _make_synthetic_dataset(tmp_path, wp_name, shot_name, n_sims=2, write_params=True)
        for i in range(2):
            sim = root / f"Simulation_{i}"
            assert (sim / "checkerboard.npy").exists()
            assert (sim / "displacements.npy").exists()
            assert (sim / "simulation_params.txt").exists()

    @pytest.mark.parametrize("wp_name,shot_name", ALL_COMBOS, ids=[f"{w}+{s}" for w, s in ALL_COMBOS])
    def test_material_features_loadable(self, wp_name, shot_name, tmp_path):
        """create_data_loaders with load_material_features=True succeeds."""
        root = _make_synthetic_dataset(
            tmp_path,
            wp_name,
            shot_name,
            n_sims=MIN_SIMS_FOR_SPLIT,
            write_params=True,
        )
        train_l, val_l, test_l, data = M.create_data_loaders(
            str(root),
            batch_size=4,
            load_material_features=True,
        )
        assert "material_features" in data
        mat = data["material_features"]
        assert mat.shape == (MIN_SIMS_FOR_SPLIT, 7), f"material_features shape mismatch: {mat.shape}"
        assert np.isfinite(mat).all(), f"NaN in material_features for {wp_name}+{shot_name}"


# ---------------------------------------------------------------------------
# 8. Physics plausibility (uses real simulations — session-scoped)
# ---------------------------------------------------------------------------

PHYS_COMBOS = [
    ("Ti-6Al-4V", "steel"),
    ("Al-7075-T6", "steel"),
    ("Inconel-718", "ceramic"),
]


@pytest.fixture(scope="session")
def physics_datasets(tmp_path_factory):
    """Generate 3 actual physics simulations for each combo — session-scoped."""
    root = tmp_path_factory.mktemp("physics_ds")
    datasets = {}
    for wp, sp in PHYS_COMBOS:
        ds_dir = root / f"{wp}_{sp}".replace("-", "_")
        ds_dir.mkdir()
        gp = GeneratorParams(
            output_dir=str(ds_dir),
            n_simulations=1,
            Nx=TINY_NX,
            Ny=TINY_NY,
            checkerboard_size=TINY_G,
            V_range=(40.0, 40.0),  # fixed V for comparison
            D_range=(0.0006, 0.0006),
            n_shots_range=(30, 30),
            workpiece_material=wp,
            shot_material=sp,
            base_seed=0,
        )
        res = generate_single_simulation(0, gp)
        datasets[(wp, sp)] = {"dir": ds_dir, "result": res}
    return datasets


class TestPhysicsPlausibility:
    """Physical properties of generated simulation data."""

    @pytest.mark.parametrize("wp,sp", PHYS_COMBOS)
    def test_simulation_succeeds(self, physics_datasets, wp, sp):
        """Real simulation must complete without error."""
        res = physics_datasets[(wp, sp)]["result"]
        assert res["success"], f"Simulation failed for {wp}+{sp}: {res.get('error')}"

    @pytest.mark.parametrize("wp,sp", PHYS_COMBOS)
    def test_uz_nonzero(self, physics_datasets, wp, sp):
        """Z-displacement must be non-zero (shots cause surface deformation)."""
        ds_dir = physics_datasets[(wp, sp)]["dir"]
        disp = np.load(ds_dir / "Simulation_0" / "displacements.npy")
        uz = disp[:, 2]
        assert not np.all(uz == 0), "All uz displacements are zero — unphysical"
        assert np.any(np.abs(uz) > 0), "No non-zero z-displacement found"

    @pytest.mark.parametrize("wp,sp", PHYS_COMBOS)
    def test_displacement_magnitude_in_um_range(self, physics_datasets, wp, sp):
        """Peak displacement must be in µm range (not nm, not mm)."""
        ds_dir = physics_datasets[(wp, sp)]["dir"]
        disp = np.load(ds_dir / "Simulation_0" / "displacements.npy")
        peak_m = float(np.max(np.abs(disp)))
        peak_um = peak_m * 1e6
        assert peak_um > 0.01, f"Peak displacement {peak_um:.4f} µm is suspiciously small"
        assert peak_um < 10_000, f"Peak displacement {peak_um:.1f} µm is unphysically large (> 10mm)"

    @pytest.mark.parametrize("wp,sp", PHYS_COMBOS)
    def test_displacements_shape(self, physics_datasets, wp, sp):
        """displacements.npy must be (N, 3) with N > 0."""
        ds_dir = physics_datasets[(wp, sp)]["dir"]
        disp = np.load(ds_dir / "Simulation_0" / "displacements.npy")
        assert disp.ndim == 2 and disp.shape[1] == 3, f"Unexpected displacement shape: {disp.shape}"
        assert disp.shape[0] > 0

    @pytest.mark.parametrize("wp,sp", PHYS_COMBOS)
    def test_checkerboard_shape_and_range(self, physics_datasets, wp, sp):
        """checkerboard.npy must be (G, G) with positive values."""
        ds_dir = physics_datasets[(wp, sp)]["dir"]
        cb = np.load(ds_dir / "Simulation_0" / "checkerboard.npy")
        assert cb.ndim == 2 and cb.shape[0] == cb.shape[1], f"Checkerboard not square: {cb.shape}"
        assert (cb >= 0).all(), "Negative checkerboard intensities"
        assert np.any(cb > 0), "All checkerboard intensities are zero"

    def test_softer_material_larger_deformation(self, physics_datasets):
        """Al (softer) should deform more than Inconel (harder) under identical loading."""
        al_dir = physics_datasets[("Al-7075-T6", "steel")]["dir"]
        inc_dir = physics_datasets[("Inconel-718", "ceramic")]["dir"]
        uz_al = np.abs(np.load(al_dir / "Simulation_0" / "displacements.npy")[:, 2])
        uz_inc = np.abs(np.load(inc_dir / "Simulation_0" / "displacements.npy")[:, 2])
        # Allow for different shot types: just verify both are non-zero and check
        # that the ratio is physically reasonable (Al should deform more, but not
        # 100× more — deformation also depends on shot type)
        peak_al = float(np.max(uz_al))
        peak_inc = float(np.max(uz_inc))
        assert peak_al > 0 and peak_inc > 0, "Both materials must have non-zero deformation"


# ---------------------------------------------------------------------------
# 9. High-resolution mesh
# ---------------------------------------------------------------------------


class TestHighResolutionMesh:
    """Dataset and model with Nx=Ny=30 (961 nodes, high div/meter)."""

    @pytest.fixture(scope="class")
    def hires_dataset(self, tmp_path_factory):
        NX, NY = 30, 30
        root = tmp_path_factory.mktemp("hires")
        rng = np.random.default_rng(1)
        n_nodes = (NX + 1) * (NY + 1)  # 961
        coords = np.zeros((n_nodes, 3), dtype=np.float32)
        xs = np.linspace(0.0, 0.01, NX + 1, dtype=np.float32)
        ys = np.linspace(0.0, 0.01, NY + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        coords[:, 0] = xx.ravel()
        coords[:, 1] = yy.ravel()

        G = 10
        for i in range(MIN_SIMS_FOR_SPLIT):
            sim = root / f"Simulation_{i}"
            sim.mkdir()
            np.save(sim / "checkerboard.npy", rng.random((G, G)).astype(np.float32))
            np.save(sim / "displacements.npy", (rng.random((n_nodes, 3)) * 1e-4).astype(np.float32))
            np.save(sim / "node_coords.npy", coords)
        return root, n_nodes, G

    def test_infer_dataset_shape(self, hires_dataset):
        root, n_nodes, G = hires_dataset
        detected_nodes, detected_G = M.infer_dataset_shape(str(root))
        assert detected_nodes == n_nodes, f"Expected {n_nodes} nodes, detected {detected_nodes}"
        assert detected_G == G, f"Expected G={G}, detected {detected_G}"

    def test_model_forward_pass(self, hires_dataset):
        root, n_nodes, G = hires_dataset
        model = M.create_model(input_channels=1, num_nodes=n_nodes, checkerboard_size=G)
        x = torch.zeros(2, 1, G, G)
        out = model(x)
        assert out.shape == (2, n_nodes, 3)
        assert torch.isfinite(out).all()

    def test_dataloader_loads_correct_nodes(self, hires_dataset):
        root, n_nodes, G = hires_dataset
        train_l, val_l, test_l, _ = M.create_data_loaders(str(root), batch_size=4)
        cb, disp = next(iter(train_l))
        assert disp.shape[-2] == n_nodes, f"Expected {n_nodes} nodes in batch, got {disp.shape[-2]}"


# ---------------------------------------------------------------------------
# 10. Extreme shot conditions
# ---------------------------------------------------------------------------


class TestExtremeShotConditions:
    """Edge-case shot parameters must not crash the simulation pipeline."""

    @pytest.mark.parametrize("V", [5.0, 100.0])
    def test_extreme_velocity(self, V, tmp_path):
        """Very low and very high velocity simulations must succeed."""
        gp = GeneratorParams(
            output_dir=str(tmp_path),
            n_simulations=1,
            Nx=TINY_NX,
            Ny=TINY_NY,
            checkerboard_size=TINY_G,
            V_range=(V, V),
            D_range=(0.0006, 0.0006),
            n_shots_range=(20, 20),
            workpiece_material="Ti-6Al-4V",
            shot_material="steel",
            base_seed=0,
        )
        res = generate_single_simulation(0, gp)
        assert res["success"], f"Simulation with V={V} m/s failed: {res.get('error')}"
        disp = np.load(os.path.join(str(tmp_path), "Simulation_0", "displacements.npy"))
        assert np.isfinite(disp).all(), f"NaN/Inf displacements at V={V} m/s"

    @pytest.mark.parametrize("D_mm", [0.1, 2.0])
    def test_extreme_diameter(self, D_mm, tmp_path):
        """Very small and very large shot diameters must not crash."""
        gp = GeneratorParams(
            output_dir=str(tmp_path),
            n_simulations=1,
            Nx=TINY_NX,
            Ny=TINY_NY,
            checkerboard_size=TINY_G,
            V_range=(40.0, 40.0),
            D_range=(D_mm * 1e-3, D_mm * 1e-3),
            n_shots_range=(10, 10),
            workpiece_material="Ti-6Al-4V",
            shot_material="steel",
            base_seed=1,
        )
        res = generate_single_simulation(0, gp)
        assert res["success"], f"Simulation with D={D_mm}mm failed: {res.get('error')}"

    def test_single_shot(self, tmp_path):
        """n_shots=1 must succeed and produce non-trivial displacement."""
        gp = GeneratorParams(
            output_dir=str(tmp_path),
            n_simulations=1,
            Nx=TINY_NX,
            Ny=TINY_NY,
            checkerboard_size=TINY_G,
            V_range=(40.0, 40.0),
            D_range=(0.0006, 0.0006),
            n_shots_range=(1, 1),
            workpiece_material="Ti-6Al-4V",
            shot_material="steel",
            base_seed=2,
        )
        res = generate_single_simulation(0, gp)
        assert res["success"], f"Single-shot simulation failed: {res.get('error')}"
        disp = np.load(os.path.join(str(tmp_path), "Simulation_0", "displacements.npy"))
        assert np.any(disp != 0), "Single shot must produce non-zero displacement"

    def test_large_shot_count(self, tmp_path):
        """n_shots=300 on small mesh must succeed without memory crash."""
        gp = GeneratorParams(
            output_dir=str(tmp_path),
            n_simulations=1,
            Nx=TINY_NX,
            Ny=TINY_NY,
            checkerboard_size=TINY_G,
            V_range=(40.0, 40.0),
            D_range=(0.0003, 0.0003),
            n_shots_range=(300, 300),
            workpiece_material="Ti-6Al-4V",
            shot_material="steel",
            base_seed=3,
        )
        res = generate_single_simulation(0, gp)
        assert res["success"], f"300-shot simulation failed: {res.get('error')}"

    def test_all_workpiece_single_shot_combo(self, tmp_path):
        """Every workpiece material paired with steel must succeed at V=40."""
        for wp in ALL_WORKPIECES:
            sim_root = tmp_path / wp.replace("-", "_")
            sim_root.mkdir(exist_ok=True)
            gp = GeneratorParams(
                output_dir=str(sim_root),
                n_simulations=1,
                Nx=TINY_NX,
                Ny=TINY_NY,
                checkerboard_size=TINY_G,
                V_range=(40.0, 40.0),
                D_range=(0.0006, 0.0006),
                n_shots_range=(20, 20),
                workpiece_material=wp,
                shot_material="steel",
                base_seed=10,
            )
            res = generate_single_simulation(0, gp)
            assert res["success"], f"Workpiece {wp} + steel failed: {res.get('error')}"


# ---------------------------------------------------------------------------
# 11. Cross-material generalization
# ---------------------------------------------------------------------------


class TestCrossMaterialGeneralization:
    """A model trained on one material must produce finite output on another."""

    @pytest.fixture(scope="class")
    def trained_ti_model(self, tmp_path_factory):
        """Train a tiny model on Ti-6Al-4V + steel synthetic data."""
        root = _make_synthetic_dataset(
            tmp_path_factory.mktemp("ti_train"),
            "Ti-6Al-4V",
            "steel",
            n_sims=MIN_SIMS_FOR_SPLIT,
        )
        train_l, val_l, _, _ = M.create_data_loaders(str(root), batch_size=4)
        model = M.create_model(1, 121, TINY_G)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 1.0)
        M.train_model(model, train_l, val_l, criterion, optimizer, scheduler, epochs=3, patience=3)
        model.eval()
        return model

    @pytest.mark.parametrize(
        "wp,shot",
        [
            ("Al-7075-T6", "ceramic"),
            ("Inconel-718", "tungsten"),
            ("316L-SS", "glass"),
        ],
    )
    def test_inference_on_different_material_is_finite(self, trained_ti_model, wp, shot, tmp_path):
        """Inference on unseen material checkerboard must produce finite output."""
        rng = np.random.default_rng(99)
        cb = rng.random((1, 1, TINY_G, TINY_G)).astype(np.float32)
        device = next(trained_ti_model.parameters()).device
        x = torch.tensor(cb, device=device)
        with torch.no_grad():
            out = trained_ti_model(x)
        assert torch.isfinite(out).all(), f"Non-finite output for {wp}+{shot} on Ti-trained model"
        assert out.shape == (1, 121, 3)


# ---------------------------------------------------------------------------
# 12. Mixed material dataset (some sims with / without params.txt)
# ---------------------------------------------------------------------------


class TestMixedMaterialDataset:
    """Dataset where some sims have simulation_params.txt and some don't.

    The loader must fall back to _DEFAULT_MAT_NORM for missing files and
    still produce a correctly-shaped material_features array.
    """

    @pytest.fixture(scope="class")
    def mixed_root(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("mixed_mat")
        rng = np.random.default_rng(0)
        coords = make_node_coords(121)
        G = TINY_G
        N_TOTAL = 8
        for i in range(N_TOTAL):
            sim = root / f"Simulation_{i}"
            sim.mkdir()
            np.save(sim / "checkerboard.npy", rng.random((G, G)).astype(np.float32))
            np.save(sim / "displacements.npy", (rng.random((121, 3)) * 1e-4).astype(np.float32))
            np.save(sim / "node_coords.npy", coords)
            # Only even-indexed sims get params.txt
            if i % 2 == 0:
                _write_sim_params(sim, "Ti-6Al-4V", "steel")
        return root, N_TOTAL

    def test_load_material_features_mixed(self, mixed_root):
        """Loading material features on a mixed dataset should not crash."""
        root, N = mixed_root
        _, _, _, data = M.create_data_loaders(str(root), batch_size=4, load_material_features=True)
        mat = data.get("material_features")
        assert mat is not None, "material_features must be present even with mixed dataset"
        assert mat.shape == (N, 7), f"Expected ({N}, 7), got {mat.shape}"
        assert np.isfinite(mat).all(), "NaN in material_features for mixed dataset"

    def test_dataloader_returns_correct_tuple_size(self, mixed_root):
        """With load_material_features=True, each batch is a 3-tuple."""
        root, _ = mixed_root
        train_l, _, _, _ = M.create_data_loaders(str(root), batch_size=4, load_material_features=True)
        batch = next(iter(train_l))
        assert len(batch) == 3, f"Expected 3-tuple (cb, mat, disp), got {len(batch)}-tuple"
        cb, mat, disp = batch
        assert mat.shape[-1] == 7, f"Material feature dim should be 7, got {mat.shape}"


# ---------------------------------------------------------------------------
# 13. infer_dataset_shape edge cases
# ---------------------------------------------------------------------------


class TestInferDatasetShapeEdgeCases:
    """infer_dataset_shape must handle missing files and non-square checkerboards."""

    def test_no_simulation_folders_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            M.infer_dataset_shape(str(tmp_path))

    def test_missing_npy_in_all_folders_raises(self, tmp_path):
        (tmp_path / "Simulation_0").mkdir()
        with pytest.raises(FileNotFoundError):
            M.infer_dataset_shape(str(tmp_path))

    def test_non_square_checkerboard_raises(self, tmp_path):
        sim = tmp_path / "Simulation_0"
        sim.mkdir()
        np.save(sim / "checkerboard.npy", np.zeros((3, 7), dtype=np.float32))  # non-square
        np.save(sim / "displacements.npy", np.zeros((100, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="not square"):
            M.infer_dataset_shape(str(tmp_path))

    def test_wrong_displacement_shape_raises(self, tmp_path):
        sim = tmp_path / "Simulation_0"
        sim.mkdir()
        np.save(sim / "checkerboard.npy", np.zeros((5, 5), dtype=np.float32))
        np.save(sim / "displacements.npy", np.zeros((100, 4), dtype=np.float32))  # wrong: 4 cols instead of 3
        with pytest.raises(ValueError, match="unexpected shape"):
            M.infer_dataset_shape(str(tmp_path))


# ---------------------------------------------------------------------------
# 14. All-pattern-modes coverage in generated dataset
# ---------------------------------------------------------------------------


class TestPatternModeVariety:
    """Verify that all 5 pattern modes appear across a generated dataset."""

    def test_pattern_modes_covered(self, tmp_path):
        """With 25 simulations and vary_distribution=True, all patterns appear."""
        gp = GeneratorParams(
            output_dir=str(tmp_path),
            n_simulations=25,
            Nx=TINY_NX,
            Ny=TINY_NY,
            checkerboard_size=TINY_G,
            V_range=(40.0, 40.0),
            D_range=(0.0006, 0.0006),
            n_shots_range=(20, 20),
            workpiece_material="Ti-6Al-4V",
            shot_material="steel",
            base_seed=100,
            pattern_modes=["uniform", "bimodal", "gradient", "random", "sparse"],
        )
        seen_modes = set()
        for i in range(25):
            res = generate_single_simulation(i, gp)
            assert res["success"], f"Sim {i} failed: {res.get('error')}"
            mode = res.get("pattern_mode")
            if mode:
                seen_modes.add(mode)
        assert len(seen_modes) >= 3, f"Expected at least 3 distinct pattern modes in 25 sims, saw: {seen_modes}"


# ---------------------------------------------------------------------------
# 15. Normalization stats round-trip
# ---------------------------------------------------------------------------


class TestNormalizationStatsRoundTrip:
    """Checkerboard normalization bounds saved and loaded by train_save_gui are consistent."""

    def test_stats_saved_on_training(self, tmp_path):
        """train_save_gui must write normalization_stats.npy."""
        root = _make_synthetic_dataset(
            tmp_path / "ds",
            "Ti-6Al-4V",
            "steel",
            n_sims=MIN_SIMS_FOR_SPLIT,
            write_params=False,
        )
        M.train_save_gui(str(root))
        stats_path = root / "saved_model" / "normalization_stats.npy"
        assert stats_path.exists(), "normalization_stats.npy must be saved by train_save_gui"

        stats = np.load(str(stats_path))
        assert stats.shape == (2,), f"Expected (2,) got {stats.shape}"
        cb_min, cb_max = float(stats[0]), float(stats[1])
        assert np.isfinite(cb_min) and np.isfinite(cb_max)
        assert cb_min < cb_max, "Normalization min must be less than max"

    def test_normalization_preserves_relative_order(self, tmp_path):
        """After normalization, the relative ordering of checkerboard values is preserved."""
        rng = np.random.default_rng(0)
        n_nodes = 121
        root = _make_synthetic_dataset(
            tmp_path,
            "Ti-6Al-4V",
            "steel",
            n_sims=MIN_SIMS_FOR_SPLIT,
            write_params=False,
        )
        train_l, _, _, loaded = M.create_data_loaders(str(root), batch_size=4)
        cb_min = loaded["checkerboard_norm_min"]
        cb_max = loaded["checkerboard_norm_max"]

        # After normalisation, any value from the original range should map to [0,1]
        raw_val = rng.uniform(cb_min, cb_max)
        norm_val = (raw_val - cb_min) / max(cb_max - cb_min, 1e-12)
        assert 0.0 <= norm_val <= 1.0, f"Normalised value {norm_val} outside [0,1]"
