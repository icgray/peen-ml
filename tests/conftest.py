"""
Shared pytest fixtures for peen-ml tests.

Provides:
  tiny_dataset          — 10 synthetic Simulation_N/ dirs (G=5, N_nodes=100)
  mismatched_sim        — single Simulation_0/ with G=20, N_nodes=196 (14x14)
  trained_model_bundle  — (saved_model_dir, dataset_path) after one training run
  sim_folder            — synthetic data_viz simulation folder (10 nodes, 4 elements)
  shuffled_labels_sim_folder — same but disp labels shuffled for alignment tests
  tk_root               — hidden Tk window, destroyed after each test
"""
import os
import sys
import numpy as np
import pytest

# tests/ is already on sys.path when pytest runs, but add it explicitly so
# conftest can import helpers even when run directly.
sys.path.insert(0, os.path.dirname(__file__))

# Make src/peen-ml importable without installation
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src", "peen-ml"),
)

from helpers import SYN_G, SYN_NODES, SYN_SIMS, make_node_coords


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """
    10 synthetic Simulation_N/ folders: G=5 checkerboard, N=100 nodes.
    Session-scoped so the folder is created once and reused across all tests.
    """
    root = tmp_path_factory.mktemp("tiny_ds")
    rng  = np.random.default_rng(42)
    coords = make_node_coords(SYN_NODES)

    for i in range(SYN_SIMS):
        sim = root / f"Simulation_{i}"
        sim.mkdir()
        np.save(sim / "checkerboard.npy",
                rng.random((SYN_G, SYN_G)).astype(np.float32))
        np.save(sim / "displacements.npy",
                rng.random((SYN_NODES, 3)).astype(np.float32))
        np.save(sim / "node_coords.npy", coords)

    return root  # pathlib.Path


@pytest.fixture(scope="session")
def mismatched_sim(tmp_path_factory):
    """
    Single Simulation_0/ with G=20, N=196 — intentionally mismatched
    with tiny_dataset models (G=5, N=100) to exercise interpolation paths.
    196 = 14x14 (perfect square, avoids fractional-grid issues).
    """
    root   = tmp_path_factory.mktemp("mismatch_ds")
    rng    = np.random.default_rng(7)
    coords = make_node_coords(196)
    sim    = root / "Simulation_0"
    sim.mkdir()
    np.save(sim / "checkerboard.npy",
            rng.random((20, 20)).astype(np.float32))
    np.save(sim / "displacements.npy",
            rng.random((196, 3)).astype(np.float32))
    np.save(sim / "node_coords.npy", coords)
    return root


@pytest.fixture(scope="session")
def trained_model_bundle(tiny_dataset):
    """
    Run train_save_gui on tiny_dataset once; return (saved_model_dir, dataset_path).
    Session-scoped so training only happens once regardless of how many tests use it.
    """
    from model import train_save_gui
    train_save_gui(str(tiny_dataset))
    return tiny_dataset / "saved_model", tiny_dataset


def _write_sim_folder(path, rng, n=10, grid=5, shuffle_disp_labels=False):
    """Write synthetic simulation files into *path* and return metadata."""
    node_labels = np.arange(1, n + 1, dtype=np.int32)
    node_coords = np.column_stack([
        np.linspace(0, 0.01, n, dtype=np.float32),
        np.linspace(0, 0.01, n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    ])
    displacements = rng.random((n, 3)).astype(np.float32) * 1e-4
    disp_node_labels = node_labels.copy()
    if shuffle_disp_labels:
        rng.shuffle(disp_node_labels)

    element_connectivity = np.array([
        [1, 2, 3, 4],
        [3, 4, 5, 6],
        [5, 6, 7, 8],
        [7, 8, 9, 10],
    ], dtype=np.int32)
    element_labels = np.array([101, 102, 103, 104], dtype=np.int32)
    stresses = rng.random((4, 6)).astype(np.float32)

    np.save(path / "checkerboard.npy", rng.random((grid, grid)).astype(np.float32))
    np.save(path / "node_coords.npy", node_coords)
    np.save(path / "node_labels.npy", node_labels)
    np.save(path / "displacements.npy", displacements)
    np.save(path / "disp_node_labels.npy", disp_node_labels)
    np.save(path / "element_connectivity.npy", element_connectivity)
    np.save(path / "stresses.npy", stresses)
    np.save(path / "stress_element_labels.npy", element_labels)

    return node_coords, displacements, disp_node_labels, node_labels


@pytest.fixture
def sim_folder(tmp_path_factory):
    """
    Synthetic simulation folder for data_viz tests.
    Matches the NUM_NODES=10, NUM_ELEMENTS=4, GRID_SIZE=5 constants in test_data_viz.py.
    Uses tmp_path_factory so each test gets its own isolated directory even when
    the test also requests tmp_path directly.
    """
    folder = tmp_path_factory.mktemp("sim_folder")
    _write_sim_folder(folder, np.random.default_rng(0))
    return folder  # pathlib.Path


@pytest.fixture
def shuffled_labels_sim_folder(tmp_path_factory):
    """
    Like sim_folder but disp_node_labels is shuffled so alignment logic is exercised.
    Returns (folder_path, node_coords, displacements, disp_node_labels, node_labels).
    """
    folder = tmp_path_factory.mktemp("shuffled_sim")
    node_coords, displacements, disp_node_labels, node_labels = _write_sim_folder(
        folder, np.random.default_rng(42), shuffle_disp_labels=True
    )
    return folder, node_coords, displacements, disp_node_labels, node_labels


@pytest.fixture
def tk_root():
    """Hidden Tk root window; destroyed after each test automatically."""
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass
