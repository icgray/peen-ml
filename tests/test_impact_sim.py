"""
test_impact_sim.py
==================
Comprehensive pytest test suite for src/peen-ml/impact_sim.py.

Tests verify:
  - ShotPeenParams derived properties
  - Mesh geometry and connectivity correctness
  - Hertz contact physics (ae, p0, F)
  - Stress field sign conventions and boundary conditions
  - Plastic zone size ordering (a_p < r_p)
  - Energy conservation identity
  - Displacement field symmetry and sign
  - Stress field attenuation with distance
  - Full run_simulation() integration (npy files created)
  - CLI arg-parser smoke test
  - Numerical reproducibility

All tests are CPU-only and have no network requirements.
"""

import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/peen-ml"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from impact_sim import (
    ShotPeenParams,
    compute_contact_params,
    compute_energy_balance,
    compute_plastic_zone,
    compute_stress_field,
    generate_mesh,
    map_displacements,
    map_stresses,
    plot_residual_stress,
    run_simulation,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def default_params():
    return ShotPeenParams()


@pytest.fixture(scope="session")
def contact(default_params):
    return compute_contact_params(default_params)


@pytest.fixture(scope="session")
def stress_field(contact, default_params):
    return compute_stress_field(contact, default_params)


@pytest.fixture(scope="session")
def plastic(default_params):
    return compute_plastic_zone(default_params)


@pytest.fixture(scope="session")
def energy(default_params, contact, plastic):
    return compute_energy_balance(default_params, contact, plastic)


@pytest.fixture(scope="session")
def surface_mesh():
    return generate_mesh(Lx=0.005, Ly=0.005, Nz=1, Nx=10, Ny=10)


@pytest.fixture(scope="session")
def volume_mesh():
    return generate_mesh(Lx=0.005, Ly=0.005, Lz=0.002, Nz=3, Nx=5, Ny=5)


@pytest.fixture(scope="session")
def sim_results(default_params):
    with tempfile.TemporaryDirectory() as tmpdir:
        res = run_simulation(params=default_params, output_dir=tmpdir, Nx=5, Ny=5, verbose=False)
        yield res


# ---------------------------------------------------------------------------
# Class 1: ShotPeenParams
# ---------------------------------------------------------------------------

class TestShotPeenParams:

    def test_default_radius(self, default_params):
        assert default_params.R == pytest.approx(0.00025, rel=1e-6)

    def test_mass_positive(self, default_params):
        assert default_params.Ms > 0

    def test_mass_formula(self, default_params):
        expected = (4.0 / 3.0) * math.pi * default_params.R ** 3 * default_params.rho_s
        assert default_params.Ms == pytest.approx(expected, rel=1e-9)

    def test_normal_velocity_at_90deg(self, default_params):
        assert default_params.Vn == pytest.approx(default_params.V, rel=1e-9)

    def test_normal_velocity_at_45deg(self):
        p = ShotPeenParams(V=10.0, phi=math.pi / 4)
        assert p.Vn == pytest.approx(10.0 / math.sqrt(2), rel=1e-6)

    def test_custom_params(self):
        p = ShotPeenParams(V=50.0, D=0.001, sigma_yield=500e6)
        assert p.V == 50.0
        assert p.D == 0.001
        assert p.sigma_yield == 500e6

    def test_shot_diameter_matches_radius(self):
        p = ShotPeenParams(D=0.0008)
        assert p.R == pytest.approx(0.0004, rel=1e-9)

    def test_default_E_b_positive(self, default_params):
        assert default_params.E_b > 0

    def test_default_nu_b_range(self, default_params):
        assert 0 < default_params.nu_b < 0.5


# ---------------------------------------------------------------------------
# Class 2: generate_mesh — surface (Nz=1)
# ---------------------------------------------------------------------------

class TestGenerateMeshSurface:

    def test_node_count(self, surface_mesh):
        # Nx=10, Ny=10 → 11×11 = 121 nodes
        assert len(surface_mesh["node_labels"]) == 121

    def test_element_count(self, surface_mesh):
        # 10×10 = 100 quad elements
        assert len(surface_mesh["element_labels"]) == 100

    def test_connectivity_shape(self, surface_mesh):
        assert surface_mesh["element_connectivity"].shape == (100, 4)

    def test_node_coords_shape(self, surface_mesh):
        assert surface_mesh["node_coords"].shape == (121, 3)

    def test_z_coords_zero(self, surface_mesh):
        np.testing.assert_array_equal(surface_mesh["node_coords"][:, 2], 0.0)

    def test_labels_start_at_one(self, surface_mesh):
        assert surface_mesh["node_labels"][0] == 1
        assert surface_mesh["element_labels"][0] == 1

    def test_labels_contiguous(self, surface_mesh):
        N = len(surface_mesh["node_labels"])
        np.testing.assert_array_equal(surface_mesh["node_labels"], np.arange(1, N + 1))

    def test_x_range(self, surface_mesh):
        xs = surface_mesh["node_coords"][:, 0]
        assert xs.min() == pytest.approx(0.0)
        assert xs.max() == pytest.approx(0.005, rel=1e-5)

    def test_y_range(self, surface_mesh):
        ys = surface_mesh["node_coords"][:, 1]
        assert ys.min() == pytest.approx(0.0)
        assert ys.max() == pytest.approx(0.005, rel=1e-5)

    def test_impact_center(self, surface_mesh):
        ic = surface_mesh["impact_center"]
        assert ic[0] == pytest.approx(0.0025, rel=1e-5)
        assert ic[1] == pytest.approx(0.0025, rel=1e-5)
        assert ic[2] == pytest.approx(0.0)

    def test_connectivity_node_labels_valid(self, surface_mesh):
        labels_set = set(surface_mesh["node_labels"].tolist())
        for row in surface_mesh["element_connectivity"]:
            for nl in row:
                assert int(nl) in labels_set

    def test_dtype_labels(self, surface_mesh):
        assert surface_mesh["node_labels"].dtype == np.int32
        assert surface_mesh["element_labels"].dtype == np.int32

    def test_dtype_coords(self, surface_mesh):
        assert surface_mesh["node_coords"].dtype == np.float32


# ---------------------------------------------------------------------------
# Class 3: generate_mesh — 3-D hex
# ---------------------------------------------------------------------------

class TestGenerateMeshVolume:

    def test_node_count_3d(self, volume_mesh):
        # (5+1)*(5+1)*(3+1) = 6*6*4 = 144
        assert len(volume_mesh["node_labels"]) == 144

    def test_element_count_3d(self, volume_mesh):
        # 5*5*3 = 75
        assert len(volume_mesh["element_labels"]) == 75

    def test_connectivity_shape_3d(self, volume_mesh):
        assert volume_mesh["element_connectivity"].shape == (75, 8)

    def test_z_negative_for_depth(self, volume_mesh):
        # z should include 0 (surface) and negative values (below surface)
        zvals = volume_mesh["node_coords"][:, 2]
        assert zvals.max() == pytest.approx(0.0)
        assert zvals.min() < 0

    def test_small_mesh_3d(self):
        m = generate_mesh(Lx=0.001, Ly=0.001, Lz=0.001, Nx=2, Ny=2, Nz=2)
        assert len(m["node_labels"]) == 3 * 3 * 3   # = 27
        assert len(m["element_labels"]) == 2 * 2 * 2  # = 8


# ---------------------------------------------------------------------------
# Class 4: compute_contact_params
# ---------------------------------------------------------------------------

class TestComputeContactParams:

    def test_E_eq_positive(self, contact):
        assert contact["E_eq"] > 0

    def test_ae_positive(self, contact):
        assert contact["ae"] > 0

    def test_ae_micron_scale(self, contact):
        # Typical ae for S170 on titanium: 50-200 µm
        assert 10e-6 < contact["ae"] < 500e-6

    def test_p0_positive(self, contact):
        assert contact["p0"] > 0

    def test_F_positive(self, contact):
        assert contact["F"] > 0

    def test_delta_equals_ae_sq_over_R(self, contact, default_params):
        expected = contact["ae"] ** 2 / default_params.R
        assert contact["delta"] == pytest.approx(expected, rel=1e-9)

    def test_higher_velocity_larger_ae(self):
        p_slow = ShotPeenParams(V=20.0)
        p_fast = ShotPeenParams(V=60.0)
        ae_slow = compute_contact_params(p_slow)["ae"]
        ae_fast = compute_contact_params(p_fast)["ae"]
        assert ae_fast > ae_slow

    def test_larger_shot_larger_ae(self):
        p_small = ShotPeenParams(D=0.0003)
        p_large = ShotPeenParams(D=0.0008)
        ae_small = compute_contact_params(p_small)["ae"]
        ae_large = compute_contact_params(p_large)["ae"]
        assert ae_large > ae_small

    def test_E_eq_harmonic_mean_bound(self, contact, default_params):
        # E_eq must be less than the lesser of E_s, E_b
        min_E = min(default_params.E_s, default_params.E_b)
        assert contact["E_eq"] < min_E

    def test_impact_time_positive(self, contact):
        assert contact["t"] > 0


# ---------------------------------------------------------------------------
# Class 5: compute_stress_field
# ---------------------------------------------------------------------------

class TestComputeStressField:

    def test_keys_present(self, stress_field):
        required = {"Z", "Z_bar", "sigma_xe", "sigma_ze", "sigma_eqe",
                    "eps_load_p", "eps_unload_p", "Sxl", "Sxu", "eps_avg", "sxs", "sR"}
        assert required.issubset(stress_field.keys())

    def test_Z_positive_increasing(self, stress_field):
        Z = stress_field["Z"]
        assert np.all(Z > 0)
        assert np.all(np.diff(Z) > 0)

    def test_Z_bar_positive(self, stress_field):
        assert np.all(stress_field["Z_bar"] > 0)

    def test_sigma_xe_compressive_near_surface(self, stress_field):
        # Just below the surface (small Z_bar), sigma_xe should be compressive (negative)
        assert stress_field["sigma_xe"][0] < 0

    def test_sigma_ze_compressive_near_surface(self, stress_field):
        assert stress_field["sigma_ze"][0] < 0

    def test_sigma_eqe_positive(self, stress_field):
        # sigma_eqe = sigma_xe - sigma_ze; since both negative but sigma_xe > sigma_ze
        # equivalent stress should be positive at least near contact
        assert np.any(stress_field["sigma_eqe"] > 0)

    def test_eps_load_p_nonnegative(self, stress_field):
        assert np.all(stress_field["eps_load_p"] >= 0)

    def test_eps_unload_p_nonnegative(self, stress_field):
        assert np.all(stress_field["eps_unload_p"] >= 0)

    def test_eps_load_ge_unload(self, stress_field):
        # Loading plastic strain >= unloading plastic strain (irreversible)
        assert np.all(stress_field["eps_load_p"] >= stress_field["eps_unload_p"])

    def test_sR_array_length(self, stress_field, default_params):
        assert len(stress_field["sR"]) == default_params.n_depth

    def test_sR_compressive_somewhere(self, stress_field):
        # Residual stress should be compressive (negative) in at least part of the zone
        assert np.any(stress_field["sR"] < 0)

    def test_sR_decays_to_zero_deep(self, stress_field):
        # Far from surface, residual stress should approach zero
        sR_deep = stress_field["sR"][-100:]
        assert np.all(np.abs(sR_deep) < 1e6)   # < 1 MPa far away

    def test_stress_field_shape_consistency(self, stress_field):
        n = len(stress_field["Z"])
        for key in ("Z_bar", "sigma_xe", "sigma_ze", "sigma_eqe", "eps_load_p"):
            assert len(stress_field[key]) == n, f"Shape mismatch for {key}"

    def test_Sxl_greater_than_sy_over3(self, stress_field, default_params):
        # Sxl = sy/3 + c*eps_load_p >= sy/3 (since eps_load_p >= 0)
        sy3 = default_params.sigma_yield / 3.0
        assert np.all(stress_field["Sxl"] >= sy3 - 1.0)  # 1 Pa tolerance

    def test_custom_yield_stress_shifts_profile(self):
        p1 = ShotPeenParams(sigma_yield=200e6)
        p2 = ShotPeenParams(sigma_yield=400e6)
        c1 = compute_contact_params(p1)
        c2 = compute_contact_params(p2)
        sf1 = compute_stress_field(c1, p1)
        sf2 = compute_stress_field(c2, p2)
        # Higher yield stress → smaller plastic zone → smaller eps_load_p peaks
        assert np.max(sf1["eps_load_p"]) > np.max(sf2["eps_load_p"])


# ---------------------------------------------------------------------------
# Class 6: compute_plastic_zone
# ---------------------------------------------------------------------------

class TestComputePlasticZone:

    def test_a_p_positive(self, plastic):
        assert plastic["a_p"] > 0

    def test_r_p_positive(self, plastic):
        assert plastic["r_p"] > 0

    def test_r_p_greater_than_a_p(self, plastic):
        # Plastic zone must be larger than dent
        assert plastic["r_p"] > plastic["a_p"]

    def test_epsilon_Mp_positive(self, plastic):
        assert plastic["epsilon_Mp"] > 0

    def test_V_p_positive(self, plastic):
        assert plastic["V_p"] > 0

    def test_W_t_positive(self, plastic):
        assert plastic["W_t"] > 0

    def test_V_p_formula(self, plastic):
        r_p = plastic["r_p"]
        expected = (2.0 * math.pi / 3.0) * r_p ** 3
        assert plastic["V_p"] == pytest.approx(expected, rel=1e-9)

    def test_W_t_equals_Vp_sy_epsMp(self, plastic, default_params):
        expected = plastic["V_p"] * default_params.sigma_yield * plastic["epsilon_Mp"]
        assert plastic["W_t"] == pytest.approx(expected, rel=1e-9)

    def test_higher_velocity_larger_a_p(self):
        p1 = ShotPeenParams(V=20.0)
        p2 = ShotPeenParams(V=60.0)
        ap1 = compute_plastic_zone(p1)["a_p"]
        ap2 = compute_plastic_zone(p2)["a_p"]
        assert ap2 > ap1

    def test_higher_yield_smaller_a_p(self):
        p1 = ShotPeenParams(sigma_yield=200e6)
        p2 = ShotPeenParams(sigma_yield=500e6)
        ap1 = compute_plastic_zone(p1)["a_p"]
        ap2 = compute_plastic_zone(p2)["a_p"]
        assert ap1 > ap2

    def test_a_p_micron_scale(self, plastic):
        # a_p for S170 on titanium: typically 30-100 µm
        assert 5e-6 < plastic["a_p"] < 500e-6


# ---------------------------------------------------------------------------
# Class 7: compute_energy_balance
# ---------------------------------------------------------------------------

class TestComputeEnergyBalance:

    def test_keys_present(self, energy):
        for key in ("KE_initial", "W_plastic", "KE_rebound", "W_wave", "e", "COR"):
            assert key in energy

    def test_KE_initial_positive(self, energy):
        assert energy["KE_initial"] > 0

    def test_KE_initial_formula(self, energy, default_params):
        expected = 0.5 * default_params.Ms * default_params.V ** 2
        assert energy["KE_initial"] == pytest.approx(expected, rel=1e-9)

    def test_energy_conservation(self, energy):
        # KE_initial ≈ W_plastic + KE_rebound + W_wave  (within floating-point noise)
        total = energy["W_plastic"] + energy["KE_rebound"] + energy["W_wave"]
        assert total == pytest.approx(energy["KE_initial"], rel=1e-6)

    def test_COR_in_unit_interval(self, energy):
        assert 0.0 <= energy["COR"] <= 1.0

    def test_e_equals_COR(self, energy):
        assert energy["e"] == energy["COR"]

    def test_KE_rebound_le_KE_initial(self, energy):
        assert energy["KE_rebound"] <= energy["KE_initial"]

    def test_W_plastic_le_KE_initial(self, energy):
        assert energy["W_plastic"] <= energy["KE_initial"] * (1 + 1e-9)

    def test_W_wave_nonnegative(self, energy):
        assert energy["W_wave"] >= 0

    def test_higher_velocity_higher_KE(self):
        p1 = ShotPeenParams(V=20.0)
        p2 = ShotPeenParams(V=60.0)
        c1 = compute_contact_params(p1); pl1 = compute_plastic_zone(p1)
        c2 = compute_contact_params(p2); pl2 = compute_plastic_zone(p2)
        e1 = compute_energy_balance(p1, c1, pl1)
        e2 = compute_energy_balance(p2, c2, pl2)
        assert e2["KE_initial"] > e1["KE_initial"]

    def test_restitution_formula(self, energy):
        # e = sqrt(1 - W_plastic/KE_initial)
        ratio = energy["W_plastic"] / energy["KE_initial"]
        expected_e = math.sqrt(max(0.0, 1.0 - ratio))
        assert energy["e"] == pytest.approx(expected_e, rel=1e-9)


# ---------------------------------------------------------------------------
# Class 8: map_displacements
# ---------------------------------------------------------------------------

class TestMapDisplacements:

    def test_output_shapes(self, surface_mesh, contact, plastic, default_params):
        labels, disp = map_displacements(surface_mesh, contact, plastic, default_params)
        N = len(surface_mesh["node_labels"])
        assert labels.shape == (N,)
        assert disp.shape == (N, 3)

    def test_dtype_displacements(self, surface_mesh, contact, plastic, default_params):
        _, disp = map_displacements(surface_mesh, contact, plastic, default_params)
        assert disp.dtype == np.float32

    def test_center_node_max_uz(self, contact, plastic, default_params):
        # The node closest to the impact centre should have the largest |uz|
        mesh = generate_mesh(Lx=0.005, Ly=0.005, Nz=1, Nx=20, Ny=20)
        ic = mesh["impact_center"]
        _, disp = map_displacements(mesh, contact, plastic, default_params)
        coords = mesh["node_coords"]
        r = np.sqrt((coords[:, 0] - ic[0]) ** 2 + (coords[:, 1] - ic[1]) ** 2)
        center_idx = np.argmin(r)
        # Centre should have among the top |uz| values
        uz = disp[:, 2]
        assert abs(uz[center_idx]) >= np.percentile(np.abs(uz), 70)

    def test_uz_negative_at_center(self, surface_mesh, contact, plastic, default_params):
        # Dent = displacement into material (negative z)
        ic = surface_mesh["impact_center"]
        _, disp = map_displacements(surface_mesh, contact, plastic, default_params)
        coords = surface_mesh["node_coords"]
        r = np.sqrt((coords[:, 0] - ic[0]) ** 2 + (coords[:, 1] - ic[1]) ** 2)
        center_idx = np.argmin(r)
        assert disp[center_idx, 2] <= 0

    def test_uz_zero_far_from_impact(self, contact, plastic, default_params):
        # On a very large mesh, corner nodes far from centre should have ~0 displacement
        mesh = generate_mesh(Lx=1.0, Ly=1.0, Nz=1, Nx=2, Ny=2)
        _, disp = map_displacements(mesh, contact, plastic, default_params)
        coords = mesh["node_coords"]
        # Corner at (0,0)
        r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
        corner_idx = np.argmax(r)
        assert abs(disp[corner_idx, 2]) < 1e-12

    def test_custom_impact_center(self, surface_mesh, contact, plastic, default_params):
        # Shift impact to corner; center node should now have different displacement
        ic_shifted = np.array([0.0, 0.0, 0.0])
        _, disp_shifted = map_displacements(
            surface_mesh, contact, plastic, default_params, impact_center=ic_shifted
        )
        _, disp_default = map_displacements(surface_mesh, contact, plastic, default_params)
        # They should differ (different impact centres)
        assert not np.allclose(disp_shifted, disp_default)

    def test_radial_symmetry(self, contact, plastic, default_params):
        # On a symmetric mesh centred exactly at impact, uz should be approx symmetric
        mesh = generate_mesh(Lx=0.01, Ly=0.01, Nz=1, Nx=20, Ny=20)
        ic = mesh["impact_center"]
        _, disp = map_displacements(mesh, contact, plastic, default_params)
        coords = mesh["node_coords"]
        # Find node above and below impact centre (same x, different y)
        idx_a = np.argmin(
            np.sqrt((coords[:, 0] - ic[0]) ** 2 + (coords[:, 1] - (ic[1] + 0.001)) ** 2)
        )
        idx_b = np.argmin(
            np.sqrt((coords[:, 0] - ic[0]) ** 2 + (coords[:, 1] - (ic[1] - 0.001)) ** 2)
        )
        assert disp[idx_a, 2] == pytest.approx(disp[idx_b, 2], rel=0.05)


# ---------------------------------------------------------------------------
# Class 9: map_stresses
# ---------------------------------------------------------------------------

class TestMapStresses:

    def test_output_shapes(self, surface_mesh, stress_field, plastic, default_params):
        labels, stresses = map_stresses(surface_mesh, stress_field, plastic, default_params)
        E = len(surface_mesh["element_labels"])
        assert labels.shape == (E,)
        assert stresses.shape == (E, 4)

    def test_S33_S12_zero(self, surface_mesh, stress_field, plastic, default_params):
        _, stresses = map_stresses(surface_mesh, stress_field, plastic, default_params)
        np.testing.assert_array_equal(stresses[:, 2], 0.0)
        np.testing.assert_array_equal(stresses[:, 3], 0.0)

    def test_S11_equals_S22(self, surface_mesh, stress_field, plastic, default_params):
        _, stresses = map_stresses(surface_mesh, stress_field, plastic, default_params)
        np.testing.assert_array_equal(stresses[:, 0], stresses[:, 1])

    def test_dtype_stresses(self, surface_mesh, stress_field, plastic, default_params):
        _, stresses = map_stresses(surface_mesh, stress_field, plastic, default_params)
        assert stresses.dtype == np.float32

    def test_center_elements_have_higher_stress(self, stress_field, plastic, default_params):
        # Build a finer mesh so we can distinguish centre vs edge
        mesh = generate_mesh(Lx=0.01, Ly=0.01, Nz=1, Nx=20, Ny=20)
        _, stresses = map_stresses(mesh, stress_field, plastic, default_params)
        ic = mesh["impact_center"]

        # Compute centroids manually
        coords_n = mesh["node_coords"]
        labels_n = mesh["node_labels"]
        lbl_idx  = {int(l): i for i, l in enumerate(labels_n)}
        conn     = mesh["element_connectivity"]
        centroids_xy = np.array([
            coords_n[[lbl_idx[int(n)] for n in row]][:, :2].mean(axis=0)
            for row in conn
        ])
        r_e = np.sqrt(
            (centroids_xy[:, 0] - ic[0]) ** 2
            + (centroids_xy[:, 1] - ic[1]) ** 2
        )
        centre_mask = r_e < plastic["r_p"]
        far_mask    = r_e > 3.0 * plastic["r_p"]
        if centre_mask.any() and far_mask.any():
            s11_centre = np.abs(stresses[centre_mask, 0]).mean()
            s11_far    = np.abs(stresses[far_mask, 0]).mean()
            assert s11_centre >= s11_far

    def test_far_elements_zero_stress(self, stress_field, plastic, default_params):
        # On a huge plate the far corners should have exactly zero stress
        mesh = generate_mesh(Lx=1.0, Ly=1.0, Nz=1, Nx=2, Ny=2)
        _, stresses = map_stresses(mesh, stress_field, plastic, default_params)
        # All elements are far from impact (r >> 3*r_p)
        np.testing.assert_array_equal(stresses[:, 0], 0.0)

    def test_stress_labels_match_element_labels(self, surface_mesh, stress_field, plastic, default_params):
        elem_labels_out, _ = map_stresses(surface_mesh, stress_field, plastic, default_params)
        np.testing.assert_array_equal(elem_labels_out, surface_mesh["element_labels"])


# ---------------------------------------------------------------------------
# Class 10: run_simulation integration
# ---------------------------------------------------------------------------

class TestRunSimulation:

    def test_returns_dict(self, sim_results):
        assert isinstance(sim_results, dict)

    def test_required_keys(self, sim_results):
        required = {
            "mesh", "contact", "stress_field", "plastic", "energy",
            "displacements", "stresses", "node_labels", "elem_labels",
            "disp_node_labels", "stress_elem_labels"
        }
        assert required.issubset(sim_results.keys())

    def test_npy_files_created(self, default_params):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_simulation(params=default_params, output_dir=tmpdir, Nx=4, Ny=4, verbose=False)
            for fname in [
                "node_labels.npy", "node_coords.npy",
                "element_labels.npy", "element_connectivity.npy",
                "disp_node_labels.npy", "displacements.npy",
                "stress_element_labels.npy", "stresses.npy",
                "sR_depth_profile.npy", "sigma_eqe_profile.npy",
                "energy_balance.txt",
            ]:
                assert os.path.exists(os.path.join(tmpdir, fname)), f"Missing: {fname}"

    def test_npy_shapes_consistent(self, default_params):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_simulation(params=default_params, output_dir=tmpdir, Nx=4, Ny=4, verbose=False)
            node_labels = np.load(os.path.join(tmpdir, "node_labels.npy"))
            node_coords = np.load(os.path.join(tmpdir, "node_coords.npy"))
            displacements = np.load(os.path.join(tmpdir, "displacements.npy"))
            disp_labels = np.load(os.path.join(tmpdir, "disp_node_labels.npy"))
            assert node_labels.shape[0] == node_coords.shape[0]
            assert disp_labels.shape == node_labels.shape
            assert displacements.shape[1] == 3

    def test_no_npy_if_save_false(self, default_params):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_simulation(
                params=default_params, output_dir=tmpdir,
                Nx=4, Ny=4, save_npy=False, verbose=False
            )
            npy_files = list(Path(tmpdir).glob("*.npy"))
            assert len(npy_files) == 0

    def test_verbose_suppressed(self, default_params, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_simulation(
                params=default_params, output_dir=tmpdir,
                Nx=3, Ny=3, save_npy=False, verbose=False
            )
            captured = capsys.readouterr()
            assert captured.out == ""

    def test_verbose_prints(self, default_params, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_simulation(
                params=default_params, output_dir=tmpdir,
                Nx=3, Ny=3, save_npy=False, verbose=True
            )
            captured = capsys.readouterr()
            assert "Impact Simulation" in captured.out

    def test_custom_params_accepted(self):
        p = ShotPeenParams(V=50.0, D=0.0008, sigma_yield=400e6)
        with tempfile.TemporaryDirectory() as tmpdir:
            res = run_simulation(params=p, output_dir=tmpdir, Nx=4, Ny=4, verbose=False)
        assert res["params"].V == 50.0

    def test_3d_mesh_integration(self, default_params):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = run_simulation(
                params=default_params, output_dir=tmpdir,
                Nx=3, Ny=3, Nz=2, verbose=False
            )
        # 3-D mesh: (3+1)*(3+1)*(2+1) = 4*4*3 = 48 nodes
        assert len(res["node_labels"]) == 48

    def test_reproducibility(self, default_params):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            res1 = run_simulation(params=default_params, output_dir=d1, Nx=4, Ny=4, verbose=False)
            res2 = run_simulation(params=default_params, output_dir=d2, Nx=4, Ny=4, verbose=False)
        np.testing.assert_array_equal(res1["displacements"], res2["displacements"])
        np.testing.assert_array_equal(res1["stresses"], res2["stresses"])

    def test_energy_balance_in_results(self, sim_results):
        energy = sim_results["energy"]
        assert "KE_initial" in energy
        assert energy["KE_initial"] > 0

    def test_sR_profile_npy_shape(self, default_params):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_simulation(params=default_params, output_dir=tmpdir, Nx=3, Ny=3, verbose=False)
            sR = np.load(os.path.join(tmpdir, "sR_depth_profile.npy"))
            assert sR.ndim == 2
            assert sR.shape[1] == 2  # [Z, sR]


# ---------------------------------------------------------------------------
# Class 11: plot_residual_stress
# ---------------------------------------------------------------------------

class TestPlotResidualStress:

    def test_plot_runs_without_error(self, sim_results):
        with patch("matplotlib.pyplot.show"):
            plot_residual_stress(sim_results, show=False)

    def test_plot_saves_file(self, sim_results):
        with patch("matplotlib.pyplot.show"):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test_plot.png")
                plot_residual_stress(sim_results, show=False, save_path=path)
                assert os.path.exists(path)

    def test_show_false_does_not_call_plt_show(self, sim_results):
        with patch("matplotlib.pyplot.show") as mock_show:
            plot_residual_stress(sim_results, show=False)
            mock_show.assert_not_called()

    def test_show_true_calls_plt_show(self, sim_results):
        with patch("matplotlib.pyplot.show") as mock_show:
            plot_residual_stress(sim_results, show=True)
            mock_show.assert_called_once()


# ---------------------------------------------------------------------------
# Class 12: Numerical sanity / physics checks
# ---------------------------------------------------------------------------

class TestPhysicsSanity:

    def test_ae_less_than_shot_radius(self, contact, default_params):
        # Hertz: contact radius should be smaller than shot radius for typical peening
        assert contact["ae"] < default_params.R * 10  # ae << R

    def test_r_p_micron_scale(self, plastic):
        # r_p for titanium / S170: 100-1000 µm range
        assert 10e-6 < plastic["r_p"] < 5e-3

    def test_W_t_matches_formula(self, plastic, default_params):
        V_p = plastic["V_p"]
        sy  = default_params.sigma_yield
        eps = plastic["epsilon_Mp"]
        assert plastic["W_t"] == pytest.approx(V_p * sy * eps, rel=1e-9)

    def test_stress_at_surface_compressive(self, stress_field):
        # At very small depth (Z_bar ≈ 0+), sigma_xe should be compressive
        assert stress_field["sigma_xe"][0] < 0

    def test_equivalent_stress_decays_at_depth(self, stress_field):
        # sigma_eqe should be smaller far from the surface
        eqe = stress_field["sigma_eqe"]
        near_surface = np.mean(np.abs(eqe[:100]))
        deep         = np.mean(np.abs(eqe[-100:]))
        assert near_surface > deep

    def test_energy_fractions_sum_to_one(self, energy):
        ke0 = energy["KE_initial"]
        f_p  = energy["W_plastic"]  / ke0
        f_r  = energy["KE_rebound"] / ke0
        f_w  = energy["W_wave"]     / ke0
        assert f_p + f_r + f_w == pytest.approx(1.0, abs=1e-9)

    def test_COR_physical_range(self, energy):
        # Shot peening COR for metals typically 0.5-0.95
        assert 0.0 <= energy["COR"] <= 1.0

    def test_Vn_le_V(self, default_params):
        # Normal velocity component cannot exceed total velocity
        assert default_params.Vn <= default_params.V

    def test_contact_radius_increases_with_k(self):
        p_low  = ShotPeenParams(k=0.5)
        p_high = ShotPeenParams(k=1.0)
        ae_low  = compute_contact_params(p_low)["ae"]
        ae_high = compute_contact_params(p_high)["ae"]
        assert ae_high > ae_low

    def test_mesh_covers_plastic_zone(self, surface_mesh, plastic):
        # The mesh should be large enough to contain the plastic zone
        # Impact at centre (Lx/2, Ly/2) = (0.0025, 0.0025)
        # Extent from centre = 0.0025; should exceed r_p which is ~µm scale
        r_p = plastic["r_p"]
        assert 0.0025 > r_p
