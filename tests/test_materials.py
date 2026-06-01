"""
Tests for materials.py — validate that the material library is complete,
physically plausible, and that the helper functions work correctly.
"""
import sys
import os
import pytest

_SRC = os.path.join(os.path.dirname(__file__), "..", "src", "peen-ml")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from materials import (
    WORKPIECE_MATERIALS,
    SHOT_MATERIALS,
    get_workpiece,
    get_shot,
)


# ---------------------------------------------------------------------------
# Library completeness
# ---------------------------------------------------------------------------

def test_workpiece_count():
    """At least 5 workpiece materials are defined."""
    assert len(WORKPIECE_MATERIALS) >= 5, (
        f"Expected >= 5 workpiece materials, found {len(WORKPIECE_MATERIALS)}"
    )


def test_shot_count():
    """At least 5 shot materials are defined."""
    assert len(SHOT_MATERIALS) >= 5, (
        f"Expected >= 5 shot materials, found {len(SHOT_MATERIALS)}"
    )


def test_workpiece_fields():
    """Every workpiece entry has the required keys."""
    required = {"E", "nu", "sigma_yield", "c", "source"}
    for name, props in WORKPIECE_MATERIALS.items():
        missing = required - props.keys()
        assert not missing, (
            f"Workpiece '{name}' is missing fields: {missing}"
        )


def test_shot_fields():
    """Every shot material entry has the required keys."""
    required = {"rho_s", "E_s", "nu_s", "source"}
    for name, props in SHOT_MATERIALS.items():
        missing = required - props.keys()
        assert not missing, (
            f"Shot material '{name}' is missing fields: {missing}"
        )


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------

def test_workpiece_values_physical():
    """Workpiece property values are within physically plausible ranges."""
    for name, p in WORKPIECE_MATERIALS.items():
        assert p["E"] > 1e9, f"{name}: E = {p['E']} Pa is suspiciously low"
        assert 0 < p["nu"] < 0.5, f"{name}: nu = {p['nu']} is outside (0, 0.5)"
        assert p["sigma_yield"] > 0, f"{name}: sigma_yield = {p['sigma_yield']} Pa <= 0"
        assert p["c"] > 0, f"{name}: c = {p['c']} Pa <= 0"
        assert isinstance(p["source"], str) and len(p["source"]) > 0, (
            f"{name}: 'source' must be a non-empty string"
        )


def test_shot_values_physical():
    """Shot property values are within physically plausible ranges."""
    for name, p in SHOT_MATERIALS.items():
        assert p["rho_s"] > 100, f"{name}: rho_s = {p['rho_s']} kg/m³ is suspiciously low"
        assert p["E_s"] > 1e9, f"{name}: E_s = {p['E_s']} Pa is suspiciously low"
        assert 0 < p["nu_s"] < 0.5, f"{name}: nu_s = {p['nu_s']} is outside (0, 0.5)"
        assert isinstance(p["source"], str) and len(p["source"]) > 0, (
            f"{name}: 'source' must be a non-empty string"
        )


# ---------------------------------------------------------------------------
# Accessor helpers
# ---------------------------------------------------------------------------

def test_get_workpiece_valid():
    """get_workpiece returns a copy with all fields for a valid name."""
    for name in WORKPIECE_MATERIALS:
        props = get_workpiece(name)
        assert props == WORKPIECE_MATERIALS[name]


def test_get_shot_valid():
    """get_shot returns a copy with all fields for a valid name."""
    for name in SHOT_MATERIALS:
        props = get_shot(name)
        assert props == SHOT_MATERIALS[name]


def test_get_workpiece_returns_copy():
    """Mutating the returned dict does not affect the library."""
    props = get_workpiece(next(iter(WORKPIECE_MATERIALS)))
    original_E = props["E"]
    props["E"] = 0.0
    props2 = get_workpiece(next(iter(WORKPIECE_MATERIALS)))
    assert props2["E"] == original_E, "get_workpiece should return a copy"


def test_get_workpiece_unknown():
    """get_workpiece raises KeyError for an unknown name."""
    with pytest.raises(KeyError, match="Unknown workpiece material"):
        get_workpiece("unobtainium")


def test_get_shot_unknown():
    """get_shot raises KeyError for an unknown name."""
    with pytest.raises(KeyError, match="Unknown shot material"):
        get_shot("dark_matter")


def test_get_workpiece_error_lists_valid_names():
    """The KeyError message lists all valid workpiece material names."""
    try:
        get_workpiece("bad_name")
    except KeyError as exc:
        msg = str(exc)
        for name in WORKPIECE_MATERIALS:
            assert name in msg, (
                f"Valid name '{name}' not listed in error message: {msg}"
            )


def test_get_shot_error_lists_valid_names():
    """The KeyError message lists all valid shot material names."""
    try:
        get_shot("bad_name")
    except KeyError as exc:
        msg = str(exc)
        for name in SHOT_MATERIALS:
            assert name in msg, (
                f"Valid name '{name}' not listed in error message: {msg}"
            )
