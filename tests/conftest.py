"""
Shared pytest fixtures for peen-ml tests.

Provides:
  tiny_dataset        — 10 synthetic Simulation_N/ dirs (G=5, N_nodes=100)
  mismatched_sim      — single Simulation_0/ with G=20, N_nodes=196 (14x14)
  trained_model_bundle— (saved_model_dir, dataset_path) after one training run
  tk_root             — hidden Tk window, destroyed after each test
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
