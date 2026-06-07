"""
materials.py
============
Centralised material property library for the peen-ml shot peening pipeline.

All values include a ``source`` string for traceability (DOI or datasheet).
Use :func:`get_workpiece` and :func:`get_shot` to look up named presets;
both raise ``KeyError`` with a helpful message if the name is unknown.

Usage
-----
    from materials import get_workpiece, get_shot, WORKPIECE_MATERIALS, SHOT_MATERIALS

    wp = get_workpiece("Ti-6Al-4V")   # {'E': 113.8e9, 'nu': 0.342, ...}
    sp = get_shot("steel")             # {'rho_s': 7800.0, 'E_s': 210e9, ...}
"""

from __future__ import annotations

from typing import Dict

__all__ = [
    "WORKPIECE_MATERIALS",
    "SHOT_MATERIALS",
    "get_workpiece",
    "get_shot",
]

# ---------------------------------------------------------------------------
# Workpiece material library
# Keys: E (Pa), nu (–), sigma_yield (Pa), c (Pa), source
# ---------------------------------------------------------------------------
WORKPIECE_MATERIALS: Dict[str, Dict] = {
    "Ti-6Al-4V": {
        "E": 113.8e9,
        "nu": 0.342,
        "sigma_yield": 880e6,
        "c": 3.0e9,
        "source": "ASM Aerospace Specification Metals, doi:10.31399/asm.hb.v02.a0001054",
    },
    "316L-SS": {
        "E": 193e9,
        "nu": 0.265,
        "sigma_yield": 290e6,
        "c": 2.0e9,
        "source": "ASME Boiler & Pressure Vessel Code Section II-D Table 1A (2023)",
    },
    "4340-Steel": {
        "E": 200e9,
        "nu": 0.290,
        "sigma_yield": 470e6,
        "c": 3.5e9,
        "source": "MatWeb AISI 4340 annealed, www.matweb.com",
    },
    "Al-7075-T6": {
        "E": 71.7e9,
        "nu": 0.330,
        "sigma_yield": 503e6,
        "c": 1.2e9,
        "source": "MIL-HDBK-5J Table 3.7.6.0(b), US DoD (2003)",
    },
    "Inconel-718": {
        "E": 200e9,
        "nu": 0.290,
        "sigma_yield": 1100e6,
        "c": 4.0e9,
        "source": "Special Metals datasheet SMC-045 (2007)",
    },
}

# ---------------------------------------------------------------------------
# Shot material library
# Keys: rho_s (kg/m³), E_s (Pa), nu_s (–), source
# ---------------------------------------------------------------------------
SHOT_MATERIALS: Dict[str, Dict] = {
    "steel": {
        "rho_s": 7800.0,
        "E_s": 210e9,
        "nu_s": 0.30,
        "source": "ASM Handbook vol. 4 shot peening practice",
    },
    "ceramic": {
        "rho_s": 6000.0,
        "E_s": 380e9,
        "nu_s": 0.22,
        "source": "Zircoa Inc. ZrO2 engineering datasheet (2021)",
    },
    "glass": {
        "rho_s": 2500.0,
        "E_s": 70e9,
        "nu_s": 0.22,
        "source": "MIL-S-851D glass bead specification",
    },
    "cast_iron": {
        "rho_s": 7300.0,
        "E_s": 170e9,
        "nu_s": 0.26,
        "source": "ASM Handbook vol. 1 cast irons, doi:10.31399/asm.hb.v01.a0001015",
    },
    "tungsten": {
        "rho_s": 19300.0,
        "E_s": 411e9,
        "nu_s": 0.28,
        "source": "Plansee AG tungsten engineering datasheet (2022)",
    },
}


# ---------------------------------------------------------------------------
# Accessors with friendly error messages
# ---------------------------------------------------------------------------


def get_workpiece(name: str) -> Dict:
    """Return workpiece material dict for *name*.

    Returns a copy — callers may mutate it without affecting the library.

    Raises
    ------
    KeyError
        If *name* is not in WORKPIECE_MATERIALS.  The error message lists all
        valid names so the user knows what to type.
    """
    if name not in WORKPIECE_MATERIALS:
        valid = ", ".join(sorted(WORKPIECE_MATERIALS))
        raise KeyError(f"Unknown workpiece material {name!r}. " f"Valid names: {valid}")
    return dict(WORKPIECE_MATERIALS[name])


def get_shot(name: str) -> Dict:
    """Return shot material dict for *name*.

    Returns a copy.

    Raises
    ------
    KeyError
        If *name* is not in SHOT_MATERIALS.
    """
    if name not in SHOT_MATERIALS:
        valid = ", ".join(sorted(SHOT_MATERIALS))
        raise KeyError(f"Unknown shot material {name!r}. " f"Valid names: {valid}")
    return dict(SHOT_MATERIALS[name])
