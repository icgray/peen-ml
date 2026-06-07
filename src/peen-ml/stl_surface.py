"""
stl_surface.py
==============
Utilities for loading and querying arbitrary 3D shell surfaces from STL files
for use with the curved-surface shot peening simulation pipeline.

Key class: STLSurface
  - Load binary or ASCII STL → vertices, faces, face normals, vertex normals
  - KD-tree nearest-vertex lookup (scipy.spatial.KDTree)
  - Project 2D shot positions (from nozzle XY plane) onto the 3D surface
    via orthographic nearest-vertex snapping
  - Map 3D shot positions to a (G, G) checkerboard via orthographic XY projection
  - Compute per-vertex rotation matrices from flat-plate [0, 0, 1] frame to
    local surface normals (used to rotate predicted displacements)
  - Emit node_coords / element arrays in the same .npy schema as
    impact_sim.generate_mesh() / multi_shot_sim output

Dependency: trimesh (pip install trimesh)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

try:
    import trimesh

    _TRIMESH_OK = True
except ImportError:  # pragma: no cover
    _TRIMESH_OK = False

__all__ = ["STLSurface"]


class STLSurface:
    """Wrap an STL surface mesh and expose geometry queries.

    Attributes
    ----------
    vertices       : (V, 3) float32 — vertex coordinates (m)
    faces          : (F, 3) int32   — triangle vertex indices (0-based)
    face_normals   : (F, 3) float32 — outward unit normal per face
    vertex_normals : (V, 3) float32 — area-weighted average normal per vertex
    path           : str            — source STL file path
    """

    def __init__(self, stl_path: str) -> None:
        if not _TRIMESH_OK:
            raise ImportError("trimesh is required for STL surface support. " "Install it with:  pip install trimesh")
        mesh = trimesh.load_mesh(stl_path, process=True, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(
                f"Could not load a triangular surface mesh from: {stl_path}\n"
                "Only triangle-mesh STL files are supported."
            )

        self.vertices: np.ndarray = np.asarray(mesh.vertices, dtype=np.float32)
        self.faces: np.ndarray = np.asarray(mesh.faces, dtype=np.int32)
        self.face_normals: np.ndarray = np.asarray(mesh.face_normals, dtype=np.float32)
        self.vertex_normals: np.ndarray = np.asarray(mesh.vertex_normals, dtype=np.float32)
        self.path = stl_path

        # KD-tree for fast nearest-vertex queries in full 3D
        self._kdtree = KDTree(self.vertices)
        # Separate XY-only tree for orthographic projection queries
        self._kdtree_xy = KDTree(self.vertices[:, :2])

    # ------------------------------------------------------------------
    # Geometry properties
    # ------------------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def bounds(self) -> np.ndarray:
        """Return (2, 3) array [[x_min, y_min, z_min], [x_max, y_max, z_max]]."""
        return np.stack([self.vertices.min(0), self.vertices.max(0)])

    # ------------------------------------------------------------------
    # Nearest-vertex query (full 3D)
    # ------------------------------------------------------------------

    def nearest_vertex(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (distances, vertex_indices) for each row of points (M, 3)."""
        dist, idx = self._kdtree.query(points.astype(np.float32))
        return dist.astype(np.float32), idx.astype(np.int32)

    # ------------------------------------------------------------------
    # Shot projection: nozzle-plane XY → 3D surface
    # ------------------------------------------------------------------

    def project_shots_onto_surface(
        self,
        shot_xy: np.ndarray,
        z_nozzle: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 2D nozzle-plane shot positions onto the 3D surface.

        Each shot at (x, y) is snapped to the nearest surface vertex via
        orthographic (XY) projection.  Valid for gently curved surfaces;
        for highly curved surfaces the approximation degrades.

        Parameters
        ----------
        shot_xy   : (N, 2) float — [x, y] impact positions in the nozzle plane
        z_nozzle  : float        — nozzle standoff height above XY plane (m)

        Returns
        -------
        hit_xyz         : (N, 3) float32 — 3D surface position for each shot
        surface_normals : (N, 3) float32 — vertex normal at each hit
        impact_angles   : (N,)   float32 — angle from surface normal (rad)
        """
        shot_xy = np.asarray(shot_xy, dtype=np.float32)

        # Find nearest vertex in XY projection (orthographic downward cast)
        _, vert_idx = self._kdtree_xy.query(shot_xy)

        hit_xyz = self.vertices[vert_idx]  # (N, 3)
        surface_normals = self.vertex_normals[vert_idx]  # (N, 3)

        # Shot direction: from (x, y, z_nozzle) toward hit_xyz
        shot_3d = np.zeros((len(shot_xy), 3), dtype=np.float32)
        shot_3d[:, :2] = shot_xy
        shot_3d[:, 2] = float(z_nozzle)

        shot_dir = hit_xyz - shot_3d  # (N, 3)
        norms = np.linalg.norm(shot_dir, axis=1, keepdims=True).clip(1e-12)
        shot_dir_unit = shot_dir / norms

        # Angle between incoming shot direction (pointing down) and outward normal
        cos_a = np.einsum("ij,ij->i", -shot_dir_unit, surface_normals)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        impact_angles = np.arccos(cos_a).astype(np.float32)

        return hit_xyz, surface_normals, impact_angles

    # ------------------------------------------------------------------
    # Checkerboard from 3D shot positions (orthographic XY projection)
    # ------------------------------------------------------------------

    def shots_to_checkerboard(
        self,
        shot_xyz: np.ndarray,
        energy_weights: np.ndarray,
        G: int,
    ) -> np.ndarray:
        """Map 3D shot positions → (G, G) checkerboard via orthographic XY projection.

        The surface XY bounding box is divided into a G×G grid; each shot's
        energy weight (e.g. V_n²) is accumulated in the corresponding cell.
        Output is normalised to [0, 1].

        Parameters
        ----------
        shot_xyz       : (N, 3) float — 3D impact positions on the surface
        energy_weights : (N,)   float — per-shot weight (V_n² recommended)
        G              : int          — checkerboard resolution

        Returns
        -------
        cb : (G, G) float32 in [0, 1]
        """
        bounds = self.bounds()
        x_min, y_min = float(bounds[0, 0]), float(bounds[0, 1])
        Lx = max(float(bounds[1, 0]) - x_min, 1e-9)
        Ly = max(float(bounds[1, 1]) - y_min, 1e-9)

        cb = np.zeros((G, G), dtype=np.float64)
        col_idx = np.clip(((shot_xyz[:, 0] - x_min) / Lx * G).astype(int), 0, G - 1)
        row_idx = np.clip(((shot_xyz[:, 1] - y_min) / Ly * G).astype(int), 0, G - 1)

        for col, row, w in zip(col_idx, row_idx, energy_weights):
            cb[row, col] += float(w)

        mx = cb.max()
        if mx > 0:
            cb /= mx
        return cb.astype(np.float32)

    # ------------------------------------------------------------------
    # Node / element arrays compatible with impact_sim .npy schema
    # ------------------------------------------------------------------

    def to_node_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (node_coords, node_labels) in the impact_sim .npy schema.

        Returns
        -------
        node_coords : (V, 3) float32
        node_labels : (V,)   int32, 1-indexed
        """
        node_labels = np.arange(1, self.n_vertices + 1, dtype=np.int32)
        return self.vertices.copy(), node_labels

    def to_element_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (element_connectivity, element_labels) using triangular faces.

        Returns
        -------
        element_connectivity : (F, 3) int32, 1-indexed node IDs
        element_labels       : (F,)   int32, 1-indexed
        """
        connectivity = (self.faces + 1).astype(np.int32)  # 0-based → 1-based
        elem_labels = np.arange(1, self.n_faces + 1, dtype=np.int32)
        return connectivity, elem_labels

    # ------------------------------------------------------------------
    # Per-vertex rotation matrices: global [0,0,1] → local surface normal
    # ------------------------------------------------------------------

    def vertex_normal_rotation_matrices(self) -> np.ndarray:
        """Compute (V, 3, 3) rotation matrices that rotate [0, 0, 1] to each vertex normal.

        Used to re-project ML-predicted displacements (flat-plate frame with
        surface normal = [0, 0, 1]) into the local surface frame of a 3D shell.

        Algorithm: Rodrigues' rotation formula for each vertex normal.

        Returns
        -------
        R : (V, 3, 3) float32
        """
        V = self.n_vertices
        norms = self.vertex_normals.astype(np.float64)
        z_hat = np.array([0.0, 0.0, 1.0])
        R = np.zeros((V, 3, 3), dtype=np.float32)

        for i, n in enumerate(norms):
            mag = np.linalg.norm(n)
            if mag < 1e-15:
                R[i] = np.eye(3, dtype=np.float32)
                continue
            n /= mag

            axis = np.cross(z_hat, n)
            sin_a = np.linalg.norm(axis)
            cos_a = float(np.dot(z_hat, n))

            if sin_a < 1e-12:
                # n ≈ ±z_hat
                if cos_a > 0:
                    R[i] = np.eye(3, dtype=np.float32)
                else:
                    R[i] = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
            else:
                axis /= sin_a
                K = np.array(
                    [
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ]
                )
                rot = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)
                R[i] = rot.astype(np.float32)

        return R

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_arrays(self, output_dir: str) -> None:
        """Save vertex/face arrays in the standard .npy schema."""
        os.makedirs(output_dir, exist_ok=True)
        coords, labels = self.to_node_arrays()
        conn, elems = self.to_element_arrays()
        np.save(os.path.join(output_dir, "node_coords.npy"), coords)
        np.save(os.path.join(output_dir, "node_labels.npy"), labels)
        # disp_node_labels == node_labels for STL: every vertex has a displacement.
        np.save(os.path.join(output_dir, "disp_node_labels.npy"), labels)
        np.save(os.path.join(output_dir, "element_connectivity.npy"), conn)
        np.save(os.path.join(output_dir, "element_labels.npy"), elems)
        np.save(os.path.join(output_dir, "stl_vertex_normals.npy"), self.vertex_normals)
        np.save(os.path.join(output_dir, "stl_face_normals.npy"), self.face_normals)

    def __repr__(self) -> str:
        return f"STLSurface(path={os.path.basename(self.path)!r}, " f"vertices={self.n_vertices}, faces={self.n_faces})"
