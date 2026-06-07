"""Shared constants and helpers imported by both conftest.py and test modules."""

import os
import numpy as np

SYN_G = 5  # checkerboard grid size used in tiny_dataset
SYN_NODES = 100  # mesh nodes (10 x 10)
SYN_SIMS = 10  # simulations in tiny_dataset

SAMPLE_DATASET = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "dataset_sample",
        "dataset1_sample",
        "TestBatch",
    )
)


def make_node_coords(n_nodes: int) -> np.ndarray:
    """Return (n_nodes, 3) XY grid on a 10 mm square plate, Z=0."""
    side = int(n_nodes**0.5)
    xs = np.linspace(0.0, 0.01, side, dtype=np.float32)
    ys = np.linspace(0.0, 0.01, side, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel(), np.zeros(side * side, dtype=np.float32)])
