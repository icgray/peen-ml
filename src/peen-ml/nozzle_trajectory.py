"""
nozzle_trajectory.py
====================
Nozzle movement patterns for curved-surface shot peening simulation.

Provides built-in parametric scan patterns (raster, spiral, zigzag) and
loaders for arbitrary waypoint files (CSV or .npy), all returning a
NozzleTrajectory containing (T, 3) nozzle positions sampled at a fixed dt.

Usage
-----
from nozzle_trajectory import ScanParams, raster_scan, NozzleTrajectory, from_csv

params = ScanParams(
    pattern="raster", Lx=0.04, Ly=0.04,
    z_standoff=0.15, scan_speed=0.10, line_spacing=0.005,
)
traj = raster_scan(params)
print(traj)   # NozzleTrajectory(steps=..., dt=0.0100s, total_time=...s)
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

__all__ = [
    "ScanParams",
    "NozzleTrajectory",
    "raster_scan",
    "spiral_scan",
    "zigzag_scan",
    "from_csv",
    "from_npy",
]


@dataclass
class ScanParams:
    """Parameters for built-in parametric nozzle scan patterns.

    Attributes
    ----------
    pattern      : 'raster' | 'spiral' | 'zigzag'
    Lx, Ly       : Coverage area (m). Scan covers [0, Lx] × [0, Ly].
    z_standoff   : Nozzle height above the reference XY plane (m).
    scan_speed   : Nozzle travel speed along the scan path (m/s).
    line_spacing : Perpendicular distance between successive scan passes (m).
    dt           : Time step between output positions (s). Finer dt → more
                   nozzle positions → more shot-sampling calls per simulation.
    """

    pattern: str = "raster"
    Lx: float = 0.040
    Ly: float = 0.040
    z_standoff: float = 0.100
    scan_speed: float = 0.050
    line_spacing: float = 0.005
    dt: float = 0.010


class NozzleTrajectory:
    """Container for a sampled nozzle trajectory.

    Attributes
    ----------
    positions : (T, 3) float32 — nozzle (x, y, z) at each time step
    dt        : float          — time step between successive positions (s)
    """

    def __init__(self, positions: np.ndarray, dt: float = 0.01) -> None:
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (T, 3), got shape {positions.shape}")
        self.positions = positions
        self.dt = float(dt)

    @property
    def n_steps(self) -> int:
        return len(self.positions)

    @property
    def total_time(self) -> float:
        return self.n_steps * self.dt

    def __repr__(self) -> str:
        return f"NozzleTrajectory(steps={self.n_steps}, dt={self.dt:.4f}s, " f"total_time={self.total_time:.2f}s)"


# ---------------------------------------------------------------------------
# Built-in parametric patterns
# ---------------------------------------------------------------------------


def raster_scan(params: ScanParams) -> NozzleTrajectory:
    """Boustrophedon (serpentine) raster scan over [0, Lx] × [0, Ly].

    Sweeps along X at constant Y; reverses direction at each line end;
    steps by line_spacing in Y.  Starts at (0, 0).

    Parameters
    ----------
    params : ScanParams

    Returns
    -------
    NozzleTrajectory
    """
    positions: List[np.ndarray] = []
    n_lines = max(1, math.ceil(params.Ly / params.line_spacing) + 1)
    z = params.z_standoff
    step_m = params.scan_speed * params.dt

    for i in range(n_lines):
        y = min(i * params.line_spacing, params.Ly)
        if i % 2 == 0:
            x_vals = np.arange(0.0, params.Lx + step_m, step_m)
        else:
            x_vals = np.arange(params.Lx, -step_m, -step_m)
        x_vals = np.clip(x_vals, 0.0, params.Lx)
        for x in x_vals:
            positions.append(np.array([x, y, z], dtype=np.float32))

    return NozzleTrajectory(np.stack(positions), dt=params.dt)


def zigzag_scan(params: ScanParams) -> NozzleTrajectory:
    """Zigzag scan — nozzle advances along X at 45° while stepping along Y.

    Produces a sawtooth path across the surface: forward diagonal on even
    passes, backward diagonal on odd passes.

    Parameters
    ----------
    params : ScanParams

    Returns
    -------
    NozzleTrajectory
    """
    positions: List[np.ndarray] = []
    z = params.z_standoff
    step_m = params.scan_speed * params.dt * math.cos(math.radians(45.0))
    n_lines = max(1, math.ceil(params.Ly / params.line_spacing) + 1)

    for i in range(n_lines):
        y_base = min(i * params.line_spacing, params.Ly)
        n_steps = max(1, math.ceil(params.Lx / max(step_m, 1e-9)))
        xs = np.linspace(0.0, params.Lx, n_steps + 1)
        if i % 2 != 0:
            xs = xs[::-1]
        for x in xs:
            positions.append(np.array([float(x), float(np.clip(y_base, 0.0, params.Ly)), z], dtype=np.float32))

    return NozzleTrajectory(np.stack(positions), dt=params.dt)


def spiral_scan(params: ScanParams) -> NozzleTrajectory:
    """Inward rectangular spiral scan (Archimedean approximation).

    Traces concentric rectangles shrinking toward the plate centre by
    line_spacing each lap.  Covers the full [0, Lx] × [0, Ly] area.

    Parameters
    ----------
    params : ScanParams

    Returns
    -------
    NozzleTrajectory
    """
    positions: List[np.ndarray] = []
    z = params.z_standoff
    step_m = max(params.scan_speed * params.dt, 1e-9)
    cx = params.Lx / 2.0
    cy = params.Ly / 2.0
    sp = params.line_spacing
    hx, hy = cx, cy  # current half-widths of the rectangle

    while hx > sp / 2.0 and hy > sp / 2.0:
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy

        for x in np.arange(x0, x1, step_m):  # bottom left → right
            positions.append(np.array([float(np.clip(x, x0, x1)), y0, z], dtype=np.float32))
        for y in np.arange(y0, y1, step_m):  # right bottom → top
            positions.append(np.array([x1, float(np.clip(y, y0, y1)), z], dtype=np.float32))
        for x in np.arange(x1, x0, -step_m):  # top right → left
            positions.append(np.array([float(np.clip(x, x0, x1)), y1, z], dtype=np.float32))
        for y in np.arange(y1, y0, -step_m):  # left top → bottom
            positions.append(np.array([x0, float(np.clip(y, y0, y1)), z], dtype=np.float32))

        hx = max(0.0, hx - sp)
        hy = max(0.0, hy - sp)

    if not positions:
        positions.append(np.array([cx, cy, z], dtype=np.float32))

    return NozzleTrajectory(np.stack(positions), dt=params.dt)


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------


def from_csv(path: str, z_standoff: Optional[float] = None) -> NozzleTrajectory:
    """Load a nozzle trajectory from a CSV file.

    The CSV must have a header row with at least columns ``x`` and ``y``
    (case-insensitive).  If a ``z`` column is absent, ``z_standoff`` is used
    as a constant height.  If a ``t`` (time) column is present, ``dt`` is
    estimated as the median time step.

    Parameters
    ----------
    path       : str
    z_standoff : float — constant height (m) when no z column in CSV

    Returns
    -------
    NozzleTrajectory
    """
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    def _get(row: dict, *names: str) -> float:
        for name in names:
            if name in row:
                return float(row[name])
        raise KeyError(f"None of {names!r} found in CSV columns {list(row)}")

    xs, ys, zs, ts = [], [], [], []
    for raw in rows:
        row = {k.strip().lower(): v.strip() for k, v in raw.items()}
        xs.append(_get(row, "x"))
        ys.append(_get(row, "y"))
        if "z" in row:
            zs.append(_get(row, "z"))
        elif z_standoff is not None:
            zs.append(float(z_standoff))
        else:
            raise ValueError("CSV has no 'z' column and z_standoff was not provided.")
        if "t" in row:
            ts.append(_get(row, "t"))

    positions = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.01
    return NozzleTrajectory(positions, dt=dt)


def from_npy(path: str, z_standoff: Optional[float] = None) -> NozzleTrajectory:
    """Load a nozzle trajectory from a .npy file.

    Accepted array shapes
    ---------------------
    (T, 2) : [x, y] columns — z_standoff required
    (T, 3) : [x, y, z] columns — dt assumed 0.01 s
    (T, 4) : [x, y, z, t] columns — dt estimated from t column

    Parameters
    ----------
    path       : str
    z_standoff : float — used only when the array has 2 columns

    Returns
    -------
    NozzleTrajectory
    """
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"npy trajectory must be 2D, got shape {arr.shape}: {path}")

    if arr.shape[1] == 4:
        ts = arr[:, 3]
        positions = arr[:, :3].astype(np.float32)
        dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.01
    elif arr.shape[1] == 3:
        positions = arr.astype(np.float32)
        dt = 0.01
    elif arr.shape[1] == 2:
        if z_standoff is None:
            raise ValueError("npy file has only 2 columns (x, y) — z_standoff is required.")
        z_col = np.full((len(arr), 1), float(z_standoff), dtype=np.float32)
        positions = np.hstack([arr.astype(np.float32), z_col])
        dt = 0.01
    else:
        raise ValueError(f"npy trajectory must have 2, 3, or 4 columns; got {arr.shape[1]}: {path}")

    return NozzleTrajectory(positions, dt=dt)
