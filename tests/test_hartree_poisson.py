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
from isogrid.ops import integrate_field
from isogrid.poisson import build_hartree_action
from isogrid.poisson import evaluate_hartree_energy
from isogrid.poisson import solve_hartree_potential
from isogrid.poisson.open_boundary import _compute_multipole_boundary_condition
from isogrid.poisson.open_boundary import _cell_and_neighbor_slices
from isogrid.poisson.open_boundary import _cell_average_from_nodal_field
from isogrid.poisson.open_boundary import _evaluate_quadratic_fit
from isogrid.poisson.open_boundary import _local_quadratic_fit_coefficients
from isogrid.poisson.open_boundary import _MONITOR_MOMENT_RECONSTRUCTION_JACOBIAN_QUANTILE
from isogrid.poisson.open_boundary import _MONITOR_MOMENT_RECONSTRUCTION_SUBCELL_DIVISIONS
from isogrid.poisson.open_boundary import _monitor_grid_nodal_region_moments
from isogrid.poisson.open_boundary import _trilinear_cell_value


def _boundary_mask(shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :, :] = True
    mask[-1, :, :] = True
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, 0] = True
    mask[:, :, -1] = True
    return mask


def _boundary_values_from_moments(
    *,
    grid_geometry,
    total_charge: float,
    dipole_moment: np.ndarray,
    quadrupole_tensor: np.ndarray,
    reference_center: tuple[float, float, float],
) -> np.ndarray:
    dx = grid_geometry.x_points - reference_center[0]
    dy = grid_geometry.y_points - reference_center[1]
    dz = grid_geometry.z_points - reference_center[2]
    radius = np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float64)
    mask = _boundary_mask(grid_geometry.spec.shape)
    boundary_dx = dx[mask]
    boundary_dy = dy[mask]
    boundary_dz = dz[mask]
    boundary_radius = radius[mask]
    boundary_potential = total_charge / boundary_radius
    boundary_potential = boundary_potential + (
        dipole_moment[0] * boundary_dx
        + dipole_moment[1] * boundary_dy
        + dipole_moment[2] * boundary_dz
    ) / (boundary_radius**3)
    quadrupole_contraction = (
        quadrupole_tensor[0, 0] * boundary_dx * boundary_dx
        + 2.0 * quadrupole_tensor[0, 1] * boundary_dx * boundary_dy
        + 2.0 * quadrupole_tensor[0, 2] * boundary_dx * boundary_dz
        + quadrupole_tensor[1, 1] * boundary_dy * boundary_dy
        + 2.0 * quadrupole_tensor[1, 2] * boundary_dy * boundary_dz
        + quadrupole_tensor[2, 2] * boundary_dz * boundary_dz
    )
    boundary_potential = boundary_potential + 0.5 * quadrupole_contraction / (boundary_radius**5)
    boundary_values = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    boundary_values[mask] = boundary_potential
    return boundary_values


def _direct_selected_region_boundary_delta(
    grid_geometry,
    rho: np.ndarray,
) -> np.ndarray:
    cell_jacobian = _cell_average_from_nodal_field(np.asarray(grid_geometry.jacobian, dtype=np.float64))
    cell_shape = cell_jacobian.shape
    boundary_mask = _boundary_mask(cell_shape)
    interior_mask = ~boundary_mask
    high_threshold = float(
        np.quantile(
            cell_jacobian[interior_mask],
            _MONITOR_MOMENT_RECONSTRUCTION_JACOBIAN_QUANTILE,
        )
    )
    selected_cells = np.argwhere(interior_mask & (cell_jacobian >= high_threshold))
    logical_x = np.asarray(grid_geometry.logical_x, dtype=np.float64)
    logical_y = np.asarray(grid_geometry.logical_y, dtype=np.float64)
    logical_z = np.asarray(grid_geometry.logical_z, dtype=np.float64)
    logical_cell_volume = (
        float(np.diff(logical_x)[0])
        * float(np.diff(logical_y)[0])
        * float(np.diff(logical_z)[0])
    )
    sx, sy, sz = _MONITOR_MOMENT_RECONSTRUCTION_SUBCELL_DIVISIONS
    subcell_volume = logical_cell_volume / float(sx * sy * sz)
    x_points = np.asarray(grid_geometry.x_points, dtype=np.float64)
    y_points = np.asarray(grid_geometry.y_points, dtype=np.float64)
    z_points = np.asarray(grid_geometry.z_points, dtype=np.float64)
    jacobian = np.asarray(grid_geometry.jacobian, dtype=np.float64)
    density = np.asarray(rho, dtype=np.float64)
    mask = _boundary_mask(grid_geometry.spec.shape)
    boundary_x = x_points[mask]
    boundary_y = y_points[mask]
    boundary_z = z_points[mask]
    delta = np.zeros_like(boundary_x)

    for cell_index_array in selected_cells:
        i, j, k = (int(value) for value in cell_index_array)
        x_cell = x_points[i : i + 2, j : j + 2, k : k + 2]
        y_cell = y_points[i : i + 2, j : j + 2, k : k + 2]
        z_cell = z_points[i : i + 2, j : j + 2, k : k + 2]
        jacobian_cell = jacobian[i : i + 2, j : j + 2, k : k + 2]
        rho_cell = density[i : i + 2, j : j + 2, k : k + 2]
        sx_slice, sy_slice, sz_slice = _cell_and_neighbor_slices(
            (i, j, k),
            shape=grid_geometry.spec.shape,
        )
        coefficients = _local_quadratic_fit_coefficients(
            x_stencil=x_points[sx_slice, sy_slice, sz_slice],
            y_stencil=y_points[sx_slice, sy_slice, sz_slice],
            z_stencil=z_points[sx_slice, sy_slice, sz_slice],
            values=density[sx_slice, sy_slice, sz_slice],
        )
        for ix in range(sx):
            u = (ix + 0.5) / sx
            for iy in range(sy):
                v = (iy + 0.5) / sy
                for iz in range(sz):
                    w = (iz + 0.5) / sz
                    x_value = _trilinear_cell_value(x_cell, u=u, v=v, w=w)
                    y_value = _trilinear_cell_value(y_cell, u=u, v=v, w=w)
                    z_value = _trilinear_cell_value(z_cell, u=u, v=v, w=w)
                    jacobian_value = _trilinear_cell_value(jacobian_cell, u=u, v=v, w=w)
                    rho_trilinear = _trilinear_cell_value(rho_cell, u=u, v=v, w=w)
                    rho_quadratic = _evaluate_quadratic_fit(
                        coefficients,
                        x_value=x_value,
                        y_value=y_value,
                        z_value=z_value,
                    )
                    weighted_delta = (rho_quadratic - rho_trilinear) * jacobian_value * subcell_volume
                    distance = np.sqrt(
                        (boundary_x - x_value) ** 2
                        + (boundary_y - y_value) ** 2
                        + (boundary_z - z_value) ** 2
                    )
                    delta += weighted_delta / distance
    boundary_delta = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    boundary_delta[mask] = delta
    return boundary_delta


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


def test_monitor_grid_multipole_boundary_keeps_centered_gaussian_charge_close_to_two() -> None:
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

    boundary = _compute_multipole_boundary_condition(
        grid_geometry=grid_geometry,
        rho=rho,
        multipole_order=2,
    )

    assert abs(boundary.total_charge - 2.0) < 5.0e-2


def test_monitor_grid_boundary_values_track_corrected_source_for_shifted_gaussian() -> None:
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
            + (grid_geometry.z_points - 1.5) ** 2
        )
    )
    rho = 2.0 * rho / np.sum(rho * grid_geometry.cell_volumes, dtype=np.float64)
    reference_center = (0.0, 0.0, 0.0)

    baseline_total_charge = float(integrate_field(rho, grid_geometry=grid_geometry))
    dx = grid_geometry.x_points - reference_center[0]
    dy = grid_geometry.y_points - reference_center[1]
    dz = grid_geometry.z_points - reference_center[2]
    radius_squared = dx * dx + dy * dy + dz * dz
    baseline_dipole = np.array(
        [
            integrate_field(rho * dx, grid_geometry=grid_geometry),
            integrate_field(rho * dy, grid_geometry=grid_geometry),
            integrate_field(rho * dz, grid_geometry=grid_geometry),
        ],
        dtype=np.float64,
    )
    baseline_quadrupole = np.array(
        [
            [
                integrate_field(rho * (3.0 * dx * dx - radius_squared), grid_geometry=grid_geometry),
                integrate_field(rho * (3.0 * dx * dy), grid_geometry=grid_geometry),
                integrate_field(rho * (3.0 * dx * dz), grid_geometry=grid_geometry),
            ],
            [
                integrate_field(rho * (3.0 * dy * dx), grid_geometry=grid_geometry),
                integrate_field(rho * (3.0 * dy * dy - radius_squared), grid_geometry=grid_geometry),
                integrate_field(rho * (3.0 * dy * dz), grid_geometry=grid_geometry),
            ],
            [
                integrate_field(rho * (3.0 * dz * dx), grid_geometry=grid_geometry),
                integrate_field(rho * (3.0 * dz * dy), grid_geometry=grid_geometry),
                integrate_field(rho * (3.0 * dz * dz - radius_squared), grid_geometry=grid_geometry),
            ],
        ],
        dtype=np.float64,
    )
    corrected_boundary_reference = _boundary_values_from_moments(
        grid_geometry=grid_geometry,
        total_charge=baseline_total_charge,
        dipole_moment=baseline_dipole,
        quadrupole_tensor=baseline_quadrupole,
        reference_center=reference_center,
    ) + _direct_selected_region_boundary_delta(grid_geometry, rho)

    boundary = _compute_multipole_boundary_condition(
        grid_geometry=grid_geometry,
        rho=rho,
        multipole_order=2,
        reference_center=reference_center,
    )

    mask = _boundary_mask(grid_geometry.spec.shape)
    rms_error = float(
        np.sqrt(
            np.mean(
                (
                    boundary.boundary_values[mask]
                    - corrected_boundary_reference[mask]
                )
                ** 2
            )
        )
    )
    assert rms_error < 1.0e-5


def test_monitor_grid_nodal_region_moments_recover_full_nodal_moments_for_all_cells() -> None:
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
    selected_cell_mask = np.ones(
        tuple(dimension - 1 for dimension in grid_geometry.spec.shape),
        dtype=bool,
    )

    total_charge, dipole_moment, quadrupole_tensor = _monitor_grid_nodal_region_moments(
        grid_geometry,
        rho,
        reference_center=(0.0, 0.0, 0.0),
        selected_cell_mask=selected_cell_mask,
    )

    dx = grid_geometry.x_points
    dy = grid_geometry.y_points
    dz = grid_geometry.z_points
    radius_squared = dx * dx + dy * dy + dz * dz
    assert np.isclose(total_charge, np.sum(rho * grid_geometry.cell_volumes, dtype=np.float64))
    assert np.allclose(
        dipole_moment,
        np.array(
            [
                np.sum(rho * dx * grid_geometry.cell_volumes, dtype=np.float64),
                np.sum(rho * dy * grid_geometry.cell_volumes, dtype=np.float64),
                np.sum(rho * dz * grid_geometry.cell_volumes, dtype=np.float64),
            ],
            dtype=np.float64,
        ),
    )
    assert np.allclose(
        quadrupole_tensor,
        np.array(
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
        ),
    )
