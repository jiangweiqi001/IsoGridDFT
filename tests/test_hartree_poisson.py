"""Sanity checks for the first Hartree and open-boundary Poisson slice."""

from __future__ import annotations

from importlib import import_module

import numpy as np

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.audit.local_hamiltonian_h2_trial_audit import build_symmetric_h2_trial_orbital
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ks import build_singlet_like_spin_densities
from isogrid.ks import build_total_density
from isogrid.ks import evaluate_static_ks_terms
from isogrid.poisson import build_hartree_action
from isogrid.poisson import evaluate_hartree_energy
from isogrid.poisson import solve_hartree_potential
from isogrid.poisson.open_boundary import _compute_multipole_boundary_condition


def test_poisson_and_hartree_modules_import() -> None:
    poisson_module = import_module("isogrid.poisson")
    open_boundary_module = import_module("isogrid.poisson.open_boundary")
    audit_module = import_module("isogrid.audit.static_ks_h2_hartree_audit")

    assert hasattr(poisson_module, "solve_hartree_potential")
    assert hasattr(open_boundary_module, "solve_open_boundary_poisson")
    assert hasattr(audit_module, "main")


def test_hartree_potential_runs_for_simple_positive_density() -> None:
    grid_geometry = build_default_h2_grid_geometry()
    rho = np.exp(
        -0.6 * (grid_geometry.x_points**2 + grid_geometry.y_points**2 + grid_geometry.z_points**2)
    )
    result = solve_hartree_potential(grid_geometry=grid_geometry, rho=rho)

    assert result.potential.shape == grid_geometry.spec.shape
    assert np.all(np.isfinite(result.potential))
    assert result.potential[grid_geometry.spec.nx // 2, grid_geometry.spec.ny // 2, grid_geometry.spec.nz // 2] > result.potential[0, 0, 0]


def test_default_h2_trial_density_hartree_potential_shape_and_symmetry() -> None:
    psi, grid_geometry = build_symmetric_h2_trial_orbital()
    rho_up, rho_down = build_singlet_like_spin_densities(psi, grid_geometry=grid_geometry)
    rho_total = build_total_density(rho_up=rho_up, rho_down=rho_down, grid_geometry=grid_geometry)
    result = solve_hartree_potential(grid_geometry=grid_geometry, rho=rho_total)

    assert result.potential.shape == psi.shape
    assert np.all(np.isfinite(result.potential))
    assert np.allclose(result.potential, result.potential[:, :, ::-1])


def test_hartree_action_and_energy_are_finite() -> None:
    psi, grid_geometry = build_symmetric_h2_trial_orbital()
    rho_up, rho_down = build_singlet_like_spin_densities(psi, grid_geometry=grid_geometry)
    rho_total = build_total_density(rho_up=rho_up, rho_down=rho_down, grid_geometry=grid_geometry)
    result = solve_hartree_potential(grid_geometry=grid_geometry, rho=rho_total)
    action = build_hartree_action(
        psi=psi,
        grid_geometry=grid_geometry,
        hartree_potential=result,
    )
    energy = evaluate_hartree_energy(
        rho=rho_total,
        grid_geometry=grid_geometry,
        hartree_potential=result,
    )

    assert action.shape == psi.shape
    assert np.all(np.isfinite(action))
    assert np.isfinite(energy)
    assert energy > 0.0


def test_static_ks_with_hartree_runs_and_is_finite() -> None:
    psi, grid_geometry = build_symmetric_h2_trial_orbital()
    rho_up, rho_down = build_singlet_like_spin_densities(psi, grid_geometry=grid_geometry)
    terms = evaluate_static_ks_terms(
        psi=psi,
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
    )

    assert terms.total_action.shape == psi.shape
    assert np.all(np.isfinite(terms.total_action))
    assert np.all(np.isfinite(terms.hartree_potential))
    assert np.allclose(terms.hartree_action, terms.hartree_action[:, :, ::-1])


def test_monitor_grid_multipole_boundary_reduces_centered_gaussian_fake_quadrupole() -> None:
    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(19, 19, 23),
        box_half_extents=(8.0, 8.0, 10.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    rho = np.exp(
        -0.5
        * (
            grid_geometry.x_points**2
            + grid_geometry.y_points**2
            + grid_geometry.z_points**2
        )
    )
    rho = 2.0 * rho / np.sum(rho * grid_geometry.cell_volumes, dtype=np.float64)
    dx = grid_geometry.x_points
    dy = grid_geometry.y_points
    dz = grid_geometry.z_points
    radius_squared = dx * dx + dy * dy + dz * dz
    nodal_quadrupole = np.array(
        [
            [
                np.sum(rho * (3.0 * dx * dx - radius_squared) * grid_geometry.cell_volumes, dtype=np.float64),
                np.sum(rho * (3.0 * dx * dy) * grid_geometry.cell_volumes, dtype=np.float64),
                np.sum(rho * (3.0 * dx * dz) * grid_geometry.cell_volumes, dtype=np.float64),
            ],
            [
                np.sum(rho * (3.0 * dy * dx) * grid_geometry.cell_volumes, dtype=np.float64),
                np.sum(rho * (3.0 * dy * dy - radius_squared) * grid_geometry.cell_volumes, dtype=np.float64),
                np.sum(rho * (3.0 * dy * dz) * grid_geometry.cell_volumes, dtype=np.float64),
            ],
            [
                np.sum(rho * (3.0 * dz * dx) * grid_geometry.cell_volumes, dtype=np.float64),
                np.sum(rho * (3.0 * dz * dy) * grid_geometry.cell_volumes, dtype=np.float64),
                np.sum(rho * (3.0 * dz * dz - radius_squared) * grid_geometry.cell_volumes, dtype=np.float64),
            ],
        ],
        dtype=np.float64,
    )
    boundary = _compute_multipole_boundary_condition(
        grid_geometry=grid_geometry,
        rho=rho,
        multipole_order=2,
    )

    assert np.linalg.norm(boundary.quadrupole_tensor) < 0.95 * np.linalg.norm(nodal_quadrupole)
