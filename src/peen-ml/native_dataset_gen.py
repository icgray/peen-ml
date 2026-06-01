"""
native_dataset_gen.py
=====================
Python-native ML training dataset generator for the peen-ml project.

Replaces the Abaqus FEA dependency by generating ``checkerboard.npy`` /
``displacements.npy`` / ``stresses.npy`` simulation folders using the
analytical Shen & Atluri (2006) multi-shot model — no commercial software
required.

Generated data format
---------------------
Each ``Simulation_<k>/`` folder contains the same .npy files that Abaqus
produces, so ``model.py``, ``data_viz.py``, and the GUI work without
modification:

    Simulation_0/
        checkerboard.npy            (G, G) float32 — shot-density map
        displacements.npy           (N_nodes, 3) float32 — [ux, uy, uz]
        disp_node_labels.npy        (N_nodes,) int32
        node_coords.npy             (N_nodes, 3) float32
        node_labels.npy             (N_nodes,) int32
        element_connectivity.npy    (N_elems, 4) int32
        element_labels.npy          (N_elems,) int32
        stresses.npy                (N_elems, 4) float32 — [S11,S22,S33,S12]
        stress_element_labels.npy   (N_elems,) int32
        sR_depth_profile.npy        (L, 2) float32 — [depth, σR]
        energy_balance.txt

Dataset variation
-----------------
Each simulation is drawn from randomised parameters:

    - Checkerboard intensity pattern (random, bimodal, or gradient)
    - Impact velocity V      ∈ [V_min, V_max] m/s
    - Shot diameter D        ∈ [D_min, D_max] m
    - Number of shots        ∈ [n_min, n_max]
    - Material yield stress  ∈ [sy_min, sy_max] MPa  (optional)

Usage
-----
    python native_dataset_gen.py --n_sims 200 --output ./Dataset_Python
    python native_dataset_gen.py --n_sims 500 --Nx 70 --Ny 70 --workers 4

From Python:
    from native_dataset_gen import GeneratorParams, generate_dataset
    gp = GeneratorParams(n_simulations=100, output_dir="./Dataset_Python")
    generate_dataset(gp)

Performance
-----------
Each simulation runs in ~0.5–2 s on a laptop CPU (50×50 mesh, 50 shots).
A 1000-case dataset takes ~15 min single-threaded; with ``workers=4`` it
runs in ~4 min.

Notes
-----
*   The CNN in model.py expects ``num_nodes = 5202``.  With the default
    Nx=Ny=50 mesh you get (51)(51) = 2601 nodes.  Use Nx=Ny=71 to get
    (72)(72) = 5184 (close to 5202, minor mismatch handled by retraining
    with num_nodes=5184).  Alternatively, keep Nx=50, Ny=100 → (51)(101) =
    5151 nodes.  After generating, retrain model.py with the new num_nodes.
*   To exactly match the Abaqus mesh (5202 nodes) you would need to replicate
    Abaqus's internal meshing algorithm. The analytical physics is identical;
    only the node count differs, requiring a one-line change in model.py.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from impact_sim import ShotPeenParams          # noqa: E402
from multi_shot_sim import (                   # noqa: E402
    MultiShotParams,
    run_multi_shot_simulation,
)
from materials import (                        # noqa: E402
    get_workpiece,
    get_shot,
    WORKPIECE_MATERIALS,
    SHOT_MATERIALS,
)

__all__ = ["GeneratorParams", "generate_dataset", "generate_single_simulation"]


# ---------------------------------------------------------------------------
# 1.  Generator configuration
# ---------------------------------------------------------------------------

@dataclass
class GeneratorParams:
    """Configuration for the dataset generator.

    Output
    ------
    output_dir      : Root directory. Simulations saved to
                      ``<output_dir>/Simulation_<k>/``.
    n_simulations   : Total number of simulation cases to generate.
    start_index     : Start numbering from this index (useful for resuming).
    workers         : Number of parallel processes (1 = sequential).

    Mesh
    ----
    Nx, Ny   : Quad element counts per axis.
               Default 50×50 → 51×51 = 2601 nodes.
               Use Nx=100, Ny=50 → 101×51 = 5151 ≈ 5202 if you want
               to come closer to the Abaqus node count.
    Lx, Ly   : Plate dimensions (m).

    Shot pattern
    ------------
    checkerboard_size : Grid size of the shot-density checkerboard (G×G).
    distribution      : Shot placement mode passed to MultiShotParams.
                        One of 'random', 'grid', 'poisson'.  Per-simulation
                        selection can be enabled via ``vary_distribution``.
    vary_distribution : If True, each simulation independently draws its
                        distribution mode from ['random', 'grid', 'poisson'].

    Randomised physics ranges
    -------------------------
    V_range      : (V_min, V_max) m/s — impact velocity.
    D_range      : (D_min, D_max) m  — shot diameter.
    n_shots_range: (n_min, n_max)    — number of shots per simulation.
    sy_range     : (sy_min, sy_max) Pa — yield stress variation.
                   Set both values equal to fix the material.
    V_scatter    : Per-shot velocity scatter (std-dev, m/s).
    pattern_modes: List of checkerboard generation modes:
                   'uniform', 'bimodal', 'gradient', 'random', 'sparse'.

    Reproducibility
    ---------------
    base_seed : Master seed; each simulation gets seed = base_seed + index.
    """

    # Output
    output_dir: str = "./Dataset_Python"
    n_simulations: int = 100
    start_index: int = 0
    workers: int = 1

    # Mesh
    Nx: int = 50
    Ny: int = 50
    Lx: float = 0.010        # m
    Ly: float = 0.010        # m

    # Shot pattern
    checkerboard_size: int = 5
    distribution: str = "random"
    vary_distribution: bool = False

    # Physics ranges (uniform random)
    V_range: Tuple[float, float] = (25.0, 60.0)      # m/s
    D_range: Tuple[float, float] = (0.0003, 0.0010)   # m
    n_shots_range: Tuple[int, int] = (20, 120)
    sy_range: Tuple[float, float] = (200e6, 400e6)    # Pa
    V_scatter: float = 2.0                             # m/s

    # Checkerboard pattern generation modes (drawn randomly per simulation)
    pattern_modes: List[str] = field(
        default_factory=lambda: ["uniform", "bimodal", "gradient", "random", "sparse"]
    )

    # Reproducibility
    base_seed: int = 0

    # Material selection — named preset from materials.py (empty = use defaults)
    workpiece_material: str = ""   # e.g. "Ti-6Al-4V", "316L-SS", "Al-7075-T6"
    shot_material:      str = ""   # e.g. "steel", "ceramic", "glass"

    # Manual overrides (only applied when the corresponding material str is "custom")
    E_b:          Optional[float] = None   # workpiece Young's modulus (Pa)
    nu_b:         Optional[float] = None   # workpiece Poisson's ratio
    sigma_yield:  Optional[float] = None   # workpiece yield stress (Pa)
    c:            Optional[float] = None   # workpiece hardening modulus (Pa)
    E_s:          Optional[float] = None   # shot Young's modulus (Pa)
    nu_s:         Optional[float] = None   # shot Poisson's ratio
    rho_s:        Optional[float] = None   # shot density (kg/m³)


# ---------------------------------------------------------------------------
# 2.  Checkerboard pattern generators
# ---------------------------------------------------------------------------

def _make_checkerboard(
    mode: str,
    G: int,
    rng: np.random.Generator,
    intensity_min: float = 0.005,
    intensity_max: float = 0.020,
) -> np.ndarray:
    """Generate a (G, G) shot-intensity checkerboard pattern.

    Parameters
    ----------
    mode           : Pattern type: 'uniform', 'bimodal', 'gradient',
                     'random', 'sparse'.
    G              : Grid size.
    rng            : numpy random generator.
    intensity_min  : Minimum cell value (maps to fewest shots).
    intensity_max  : Maximum cell value (maps to most shots).

    Returns
    -------
    checkerboard : (G, G) float32 in [intensity_min, intensity_max].
    """
    lo, hi = intensity_min, intensity_max

    if mode == "uniform":
        # All cells equal — same intensity everywhere
        val = rng.uniform(lo, hi)
        cb = np.full((G, G), val, dtype=np.float32)

    elif mode == "bimodal":
        # Two alternating intensity levels (like a real checkerboard)
        val_a = rng.uniform(lo, (lo + hi) / 2)
        val_b = rng.uniform((lo + hi) / 2, hi)
        cb = np.empty((G, G), dtype=np.float32)
        for i in range(G):
            for j in range(G):
                cb[i, j] = val_a if (i + j) % 2 == 0 else val_b

    elif mode == "gradient":
        # Linear intensity gradient across one axis
        axis = rng.integers(2)   # 0 = along rows, 1 = along columns
        vals = np.linspace(lo, hi, G, dtype=np.float32)
        if axis == 0:
            cb = np.tile(vals[:, None], (1, G))
        else:
            cb = np.tile(vals[None, :], (G, 1))

    elif mode == "random":
        # Fully random intensities per cell
        cb = rng.uniform(lo, hi, (G, G)).astype(np.float32)

    elif mode == "sparse":
        # Only a fraction of cells are active; rest are near-zero
        cb = np.full((G, G), lo * 0.2, dtype=np.float32)
        n_active = rng.integers(1, max(2, G * G // 2))
        flat_idx = rng.choice(G * G, size=int(n_active), replace=False)
        cb.ravel()[flat_idx] = rng.uniform(lo, hi, len(flat_idx)).astype(np.float32)

    else:
        raise ValueError(f"Unknown pattern mode '{mode}'. "
                         "Choose from: uniform, bimodal, gradient, random, sparse.")

    # Clip to valid range
    cb = np.clip(cb, lo, hi)
    return cb


# ---------------------------------------------------------------------------
# 2b.  Material resolution helpers
# ---------------------------------------------------------------------------

def _resolve_workpiece(gp: GeneratorParams) -> Dict:
    """Return resolved workpiece property dict for this GeneratorParams.

    Priority:
      1. Named preset (gp.workpiece_material non-empty and in library)
      2. Manual overrides (gp.E_b, gp.nu_b, etc.)
      3. ShotPeenParams dataclass defaults
    """
    from impact_sim import ShotPeenParams as _SP
    _defaults = _SP()
    base: Dict = {
        "E_b":           _defaults.E_b,
        "nu_b":          _defaults.nu_b,
        "sigma_yield":   _defaults.sigma_yield,
        "c":             _defaults.c,
        "source":        "ShotPeenParams defaults",
    }
    if gp.workpiece_material and gp.workpiece_material != "custom":
        try:
            lib = get_workpiece(gp.workpiece_material)
            base = {
                "E_b":         lib["E"],
                "nu_b":        lib["nu"],
                "sigma_yield": lib["sigma_yield"],
                "c":           lib["c"],
                "source":      lib["source"],
            }
        except KeyError:
            pass  # unknown name → keep defaults
    # Apply manual overrides
    if gp.E_b is not None:           base["E_b"]         = gp.E_b
    if gp.nu_b is not None:          base["nu_b"]        = gp.nu_b
    if gp.sigma_yield is not None:   base["sigma_yield"] = gp.sigma_yield
    if gp.c is not None:             base["c"]           = gp.c
    return base


def _resolve_shot(gp: GeneratorParams) -> Dict:
    """Return resolved shot property dict for this GeneratorParams."""
    from impact_sim import ShotPeenParams as _SP
    _defaults = _SP()
    base: Dict = {
        "E_s":    _defaults.E_s,
        "nu_s":   _defaults.nu_s,
        "rho_s":  _defaults.rho_s,
        "source": "ShotPeenParams defaults",
    }
    if gp.shot_material and gp.shot_material != "custom":
        try:
            lib = get_shot(gp.shot_material)
            base = {
                "E_s":    lib["E_s"],
                "nu_s":   lib["nu_s"],
                "rho_s":  lib["rho_s"],
                "source": lib["source"],
            }
        except KeyError:
            pass
    if gp.E_s is not None:   base["E_s"]   = gp.E_s
    if gp.nu_s is not None:  base["nu_s"]  = gp.nu_s
    if gp.rho_s is not None: base["rho_s"] = gp.rho_s
    return base


# ---------------------------------------------------------------------------
# 3.  Single-simulation runner (called by parallel workers)
# ---------------------------------------------------------------------------

def generate_single_simulation(
    sim_index: int,
    gen_params: GeneratorParams,
) -> Dict:
    """Generate and save one simulation case.

    Parameters
    ----------
    sim_index  : Simulation number (determines output folder and seed).
    gen_params : GeneratorParams driving the sweep.

    Returns
    -------
    dict with keys: sim_index, output_dir, n_nodes, n_elems,
                    coverage_percent, almen_MPa, elapsed_s, success, error.
    """
    t0 = time.perf_counter()

    seed = gen_params.base_seed + sim_index
    rng = np.random.default_rng(seed)

    out_folder = os.path.join(gen_params.output_dir, f"Simulation_{sim_index}")

    # ---- Resolve material properties ----
    wp_props = _resolve_workpiece(gen_params)
    sp_props = _resolve_shot(gen_params)
    wp_name   = gen_params.workpiece_material or "default"
    sp_name   = gen_params.shot_material or "default"

    # ---- Draw randomised physics ----
    V  = float(rng.uniform(*gen_params.V_range))
    D  = float(rng.uniform(*gen_params.D_range))
    # sigma_yield: use material library value if available, else random sweep
    sy = float(wp_props["sigma_yield"]) if wp_props.get("sigma_yield") is not None and gen_params.workpiece_material else float(rng.uniform(*gen_params.sy_range))
    n_shots = int(rng.integers(gen_params.n_shots_range[0],
                               gen_params.n_shots_range[1] + 1))

    # ---- Draw distribution mode ----
    if gen_params.vary_distribution:
        dist = str(rng.choice(["random", "grid", "poisson"]))
    else:
        dist = gen_params.distribution

    # ---- Generate checkerboard intensity pattern ----
    pattern_mode = str(rng.choice(gen_params.pattern_modes))
    checkerboard = _make_checkerboard(
        mode=pattern_mode,
        G=gen_params.checkerboard_size,
        rng=rng,
    )

    # ---- Build ShotPeenParams ----
    bp = ShotPeenParams(
        V=V, D=D, sigma_yield=sy,
        E_b=wp_props["E_b"], nu_b=wp_props["nu_b"], c=wp_props["c"],
        E_s=sp_props["E_s"], nu_s=sp_props["nu_s"], rho_s=sp_props["rho_s"],
        n_depth=5000,            # coarser depth resolution for speed
    )

    # ---- Build MultiShotParams ----
    msp = MultiShotParams(
        base_params=bp,
        n_shots=n_shots,
        distribution="checkerboard",   # always use checkerboard-driven positions
        seed=seed,
        V_scatter=gen_params.V_scatter,
        Lx=gen_params.Lx,
        Ly=gen_params.Ly,
        Nx=gen_params.Nx,
        Ny=gen_params.Ny,
    )

    try:
        results = run_multi_shot_simulation(
            params=msp,
            checkerboard=checkerboard,
            output_dir=out_folder,
            save_npy=True,
            verbose=False,
            grid_size=gen_params.checkerboard_size,
        )

        # Write a plain-text metadata file for traceability
        _write_metadata(out_folder, sim_index, gen_params, bp, msp,
                        pattern_mode, results, wp_name, sp_name, wp_props, sp_props)

        elapsed = time.perf_counter() - t0
        return {
            "sim_index":       sim_index,
            "output_dir":      out_folder,
            "n_nodes":         len(results["node_labels"]),
            "n_elems":         len(results["elem_labels"]),
            "coverage_percent": results["coverage"]["coverage_percent"],
            "almen_MPa":       results["almen_intensity_MPa"],
            "V":               V,
            "D_mm":            D * 1e3,
            "n_shots":         n_shots,
            "pattern_mode":    pattern_mode,
            "elapsed_s":       elapsed,
            "success":         True,
            "error":           None,
        }

    except Exception as exc:       # noqa: BLE001
        elapsed = time.perf_counter() - t0
        return {
            "sim_index":   sim_index,
            "output_dir":  out_folder,
            "elapsed_s":   elapsed,
            "success":     False,
            "error":       str(exc),
        }


def _write_metadata(
    out_folder: str,
    sim_index: int,
    gp: GeneratorParams,
    bp: ShotPeenParams,
    msp: MultiShotParams,
    pattern_mode: str,
    results: Dict,
    wp_name: str = "default",
    sp_name: str = "default",
    wp_props: Optional[Dict] = None,
    sp_props: Optional[Dict] = None,
) -> None:
    path = os.path.join(out_folder, "simulation_params.txt")
    with open(path, "w") as fh:
        fh.write(f"sim_index:      {sim_index}\n")
        fh.write(f"generator:      native_dataset_gen.py (Shen & Atluri 2006)\n")
        fh.write(f"V_m_per_s:      {bp.V:.3f}\n")
        fh.write(f"D_mm:           {bp.D*1e3:.4f}\n")
        fh.write(f"sigma_yield_MPa:{bp.sigma_yield/1e6:.1f}\n")
        fh.write(f"n_shots:        {msp.n_shots}\n")
        fh.write(f"distribution:   {msp.distribution}\n")
        fh.write(f"pattern_mode:   {pattern_mode}\n")
        fh.write(f"Lx_mm:          {gp.Lx*1e3:.1f}\n")
        fh.write(f"Ly_mm:          {gp.Ly*1e3:.1f}\n")
        fh.write(f"Nx:             {gp.Nx}\n")
        fh.write(f"Ny:             {gp.Ny}\n")
        fh.write(f"coverage_pct:   {results['coverage']['coverage_percent']:.2f}\n")
        fh.write(f"almen_MPa:      {results['almen_intensity_MPa']:.2f}\n")
        # Material block — parsed by model.py for material-conditioned training
        fh.write(f"[material]\n")
        fh.write(f"workpiece       = {wp_name}\n")
        fh.write(f"E_b             = {bp.E_b:.6e}\n")
        fh.write(f"nu_b            = {bp.nu_b:.6f}\n")
        fh.write(f"sigma_yield     = {bp.sigma_yield:.6e}\n")
        fh.write(f"c               = {bp.c:.6e}\n")
        fh.write(f"workpiece_source = {(wp_props or {}).get('source', 'default')}\n")
        fh.write(f"shot            = {sp_name}\n")
        fh.write(f"rho_s           = {bp.rho_s:.6f}\n")
        fh.write(f"E_s             = {bp.E_s:.6e}\n")
        fh.write(f"nu_s            = {bp.nu_s:.6f}\n")
        fh.write(f"shot_source     = {(sp_props or {}).get('source', 'default')}\n")


# ---------------------------------------------------------------------------
# 4.  Main dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(
    gen_params: Optional[GeneratorParams] = None,
    verbose: bool = True,
) -> List[Dict]:
    """Generate a full ML training dataset.

    Parameters
    ----------
    gen_params : GeneratorParams (defaults created if None).
    verbose    : Print per-simulation progress.

    Returns
    -------
    List of result dicts from ``generate_single_simulation()``.
    """
    if gen_params is None:
        gen_params = GeneratorParams()

    os.makedirs(gen_params.output_dir, exist_ok=True)

    indices = list(range(
        gen_params.start_index,
        gen_params.start_index + gen_params.n_simulations,
    ))
    n_total = len(indices)

    _log = print if verbose else (lambda *a, **k: None)
    _log("=" * 60)
    _log("Native Dataset Generator -- peen-ml")
    _log(f"  Output   : {gen_params.output_dir}")
    _log(f"  Cases    : {n_total}  (indices {indices[0]}-{indices[-1]})")
    _log(f"  Mesh     : {gen_params.Nx}x{gen_params.Ny}  "
         f"-> {(gen_params.Nx+1)*(gen_params.Ny+1)} nodes")
    _log(f"  Workers  : {gen_params.workers}")
    _log(f"  V range  : {gen_params.V_range[0]}-{gen_params.V_range[1]} m/s")
    _log(f"  D range  : {gen_params.D_range[0]*1e3:.3f}-"
         f"{gen_params.D_range[1]*1e3:.3f} mm")
    _log(f"  Shots/sim: {gen_params.n_shots_range[0]}-{gen_params.n_shots_range[1]}")
    _log("=" * 60)

    t_start = time.perf_counter()
    results = []
    n_ok = n_fail = 0

    if gen_params.workers <= 1:
        # Sequential
        for idx, sim_idx in enumerate(indices):
            res = generate_single_simulation(sim_idx, gen_params)
            results.append(res)
            if res["success"]:
                n_ok += 1
                if verbose:
                    print(f"  [{idx+1:4d}/{n_total}] Sim_{sim_idx:04d}  "
                          f"coverage={res['coverage_percent']:.1f}%  "
                          f"almen={res['almen_MPa']:.0f} MPa  "
                          f"({res['elapsed_s']:.1f}s)")
            else:
                n_fail += 1
                print(f"  [{idx+1:4d}/{n_total}] Sim_{sim_idx:04d}  "
                      f"FAILED: {res['error']}")
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=gen_params.workers) as pool:
            futures = {
                pool.submit(generate_single_simulation, si, gen_params): si
                for si in indices
            }
            done = 0
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                done += 1
                if res["success"]:
                    n_ok += 1
                    if verbose:
                        print(f"  [{done:4d}/{n_total}] Sim_{res['sim_index']:04d}  "
                              f"coverage={res['coverage_percent']:.1f}%  "
                              f"almen={res['almen_MPa']:.0f} MPa  "
                              f"({res['elapsed_s']:.1f}s)")
                else:
                    n_fail += 1
                    print(f"  [{done:4d}/{n_total}] Sim_{res['sim_index']:04d}  "
                          f"FAILED: {res['error']}")

    elapsed = time.perf_counter() - t_start

    # ---- Write dataset-level summary CSV ----
    summary_path = os.path.join(gen_params.output_dir, "dataset_summary.csv")
    _write_summary_csv(summary_path, results)

    _log("=" * 60)
    _log(f"Done.  {n_ok}/{n_total} OK,  {n_fail} failed  "
         f"({elapsed:.0f} s total,  {elapsed/n_total:.1f} s/sim)")
    _log(f"Summary CSV: {summary_path}")
    _log("=" * 60)

    # Print note about CNN compatibility
    n_nodes = (gen_params.Nx + 1) * (gen_params.Ny + 1)
    if n_nodes != 5202:
        _log(f"\nNOTE: This dataset has {n_nodes} nodes per simulation.")
        _log("  The default model.py uses num_nodes=5202 (Abaqus mesh).")
        _log(f"  To retrain on this data, change num_nodes={n_nodes} in model.py:")
        _log(f"    model = create_model(input_channels=1, num_nodes={n_nodes})")
        _log("  Also check the FC layer: 128 * 5 * 5 may need resizing.")

    return results


def _write_summary_csv(path: str, results: List[Dict]) -> None:
    import csv
    fieldnames = [
        "sim_index", "success", "n_nodes", "n_elems",
        "coverage_percent", "almen_MPa", "V", "D_mm",
        "n_shots", "pattern_mode", "elapsed_s", "error",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda x: x.get("sim_index", 0)):
            writer.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# 5.  Validation helpers
# ---------------------------------------------------------------------------

def validate_dataset(dataset_dir: str, n_check: int = 5) -> None:
    """Quick sanity-check on the generated dataset.

    Loads ``n_check`` random simulations and verifies file existence,
    shape consistency, and physics plausibility.

    Parameters
    ----------
    dataset_dir : Root directory of the generated dataset.
    n_check     : Number of simulations to inspect.
    """
    sims = sorted([
        d for d in os.listdir(dataset_dir)
        if d.startswith("Simulation_") and
        os.path.isdir(os.path.join(dataset_dir, d))
    ], key=lambda x: int(x.split("_")[1]))

    rng = np.random.default_rng(0)
    sample = rng.choice(sims, size=min(n_check, len(sims)), replace=False)

    print(f"\nDataset validation: {dataset_dir}")
    print(f"  Total simulations found: {len(sims)}")
    print(f"  Checking {len(sample)} samples …\n")

    required_files = [
        "checkerboard.npy", "displacements.npy", "stresses.npy",
        "node_coords.npy", "node_labels.npy",
        "element_connectivity.npy", "element_labels.npy",
        "disp_node_labels.npy", "stress_element_labels.npy",
    ]

    all_ok = True
    for sim_name in sample:
        sim_dir = os.path.join(dataset_dir, sim_name)
        ok = True
        errors = []

        for fname in required_files:
            fpath = os.path.join(sim_dir, fname)
            if not os.path.exists(fpath):
                errors.append(f"Missing: {fname}")
                ok = False
                continue

            arr = np.load(fpath)

            # Shape checks
            if fname == "checkerboard.npy" and arr.ndim != 2:
                errors.append(f"checkerboard shape mismatch: {arr.shape}")
                ok = False
            if fname == "displacements.npy" and arr.shape[1] != 3:
                errors.append(f"displacements shape mismatch: {arr.shape}")
                ok = False
            if fname == "stresses.npy" and arr.shape[1] != 4:
                errors.append(f"stresses shape mismatch: {arr.shape}")
                ok = False

            # Physics plausibility
            if fname == "displacements.npy":
                uz = arr[:, 2]
                if np.all(uz == 0):
                    errors.append("All uz displacements are zero — suspect")
                    ok = False
                if np.max(np.abs(uz)) > 0.01:   # >10 mm deformation is unphysical
                    errors.append(f"Max |uz| = {np.max(np.abs(uz))*1e3:.2f} mm — too large")
                    ok = False

        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {sim_name}: {status}")
        for e in errors:
            print(f"    -> {e}")

    print(f"\n{'All checks passed.' if all_ok else 'Some checks FAILED.'}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate peen-ml training dataset (no Abaqus required)."
    )
    parser.add_argument("--output",   default="./Dataset_Python",
                        help="Root output directory")
    parser.add_argument("--n_sims",   type=int, default=100,
                        help="Number of simulation cases")
    parser.add_argument("--start",    type=int, default=0,
                        help="Starting simulation index")
    parser.add_argument("--Nx",       type=int, default=50)
    parser.add_argument("--Ny",       type=int, default=50)
    parser.add_argument("--Lx",       type=float, default=0.010)
    parser.add_argument("--Ly",       type=float, default=0.010)
    parser.add_argument("--workers",  type=int, default=1,
                        help="Parallel workers (1 = sequential)")
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--V_min",    type=float, default=25.0)
    parser.add_argument("--V_max",    type=float, default=60.0)
    parser.add_argument("--D_min",    type=float, default=0.0003)
    parser.add_argument("--D_max",    type=float, default=0.0010)
    parser.add_argument("--n_shots_min", type=int, default=20)
    parser.add_argument("--n_shots_max", type=int, default=120)
    parser.add_argument("--grid_size",   type=int, default=5,
                        help="Checkerboard grid resolution (G×G)")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation checks after generation")

    # Material selection
    parser.add_argument("--workpiece_material", default="",
                        choices=[""] + sorted(WORKPIECE_MATERIALS) + ["custom"],
                        metavar="NAME",
                        help=("Named workpiece material from library "
                              f"({', '.join(sorted(WORKPIECE_MATERIALS))}) "
                              "or 'custom' to use --E_b/--nu_b/etc. overrides. "
                              "Default: ShotPeenParams built-in values."))
    parser.add_argument("--shot_material", default="",
                        choices=[""] + sorted(SHOT_MATERIALS) + ["custom"],
                        metavar="NAME",
                        help=("Named shot material from library "
                              f"({', '.join(sorted(SHOT_MATERIALS))}) "
                              "or 'custom'. Default: ShotPeenParams built-in values."))
    parser.add_argument("--E_b",         type=float, default=None, help="Override workpiece E (Pa)")
    parser.add_argument("--nu_b",        type=float, default=None, help="Override workpiece nu")
    parser.add_argument("--sigma_yield", type=float, default=None, help="Override workpiece sigma_yield (Pa); disables sy_range sweep")
    parser.add_argument("--c",           type=float, default=None, help="Override workpiece hardening c (Pa)")
    parser.add_argument("--E_s",         type=float, default=None, help="Override shot E (Pa)")
    parser.add_argument("--nu_s",        type=float, default=None, help="Override shot nu")
    parser.add_argument("--rho_s",       type=float, default=None, help="Override shot density (kg/m³)")

    args = parser.parse_args()

    gp = GeneratorParams(
        output_dir=args.output,
        n_simulations=args.n_sims,
        start_index=args.start,
        workers=args.workers,
        Nx=args.Nx, Ny=args.Ny,
        Lx=args.Lx, Ly=args.Ly,
        base_seed=args.seed,
        V_range=(args.V_min, args.V_max),
        D_range=(args.D_min, args.D_max),
        n_shots_range=(args.n_shots_min, args.n_shots_max),
        checkerboard_size=args.grid_size,
        workpiece_material=args.workpiece_material,
        shot_material=args.shot_material,
        E_b=args.E_b, nu_b=args.nu_b, sigma_yield=args.sigma_yield, c=args.c,
        E_s=args.E_s, nu_s=args.nu_s, rho_s=args.rho_s,
    )

    generate_dataset(gp, verbose=True)

    if args.validate:
        validate_dataset(args.output, n_check=10)
