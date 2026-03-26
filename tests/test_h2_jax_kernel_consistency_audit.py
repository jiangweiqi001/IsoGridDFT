"""Minimal smoke tests for the first-batch JAX kernel migration."""

from __future__ import annotations

import numpy as np

from isogrid.audit.h2_jax_kernel_consistency_audit import (
    run_h2_jax_kernel_consistency_audit,
)
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.ks.hamiltonian_local_jax import apply_fixed_potential_static_local_operator_jax
from isogrid.ks import prepare_fixed_potential_static_local_operator
from isogrid.ops.reductions_jax import weighted_inner_product_jax
from isogrid.poisson.poisson_jax import apply_monitor_open_boundary_poisson_operator_jax


def _build_trial_orbital(grid_geometry) -> np.ndarray:
    atom_fields = []
    for atom in H2_BENCHMARK_CASE.geometry.atoms:
        dx = grid_geometry.x_points - atom.position[0]
        dy = grid_geometry.y_points - atom.position[1]
        dz = grid_geometry.z_points - atom.position[2]
        atom_fields.append(np.exp(-0.8 * (dx * dx + dy * dy + dz * dz)))
    orbital = np.asarray(atom_fields[0] + atom_fields[1], dtype=np.float64)
    weights = grid_geometry.cell_volumes
    return orbital / np.sqrt(np.sum(orbital * orbital * weights))


def test_weighted_inner_product_jax_runs() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    weights = np.array([0.25, 0.5, 1.0], dtype=np.float64)
    result = float(np.real_if_close(weighted_inner_product_jax(values, values, weights)))
    assert result > 0.0


def test_monitor_poisson_apply_jax_runs() -> None:
    grid_geometry = build_h2_local_patch_development_monitor_grid()
    values = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    action = np.asarray(
        apply_monitor_open_boundary_poisson_operator_jax(
            values,
            grid_geometry=grid_geometry,
        ),
        dtype=np.float64,
    )
    assert action.shape == grid_geometry.spec.shape


def test_h2_jax_kernel_consistency_result_fields() -> None:
    grid_geometry = build_h2_local_patch_development_monitor_grid()
    trial_orbital = _build_trial_orbital(grid_geometry)
    operator_context = prepare_fixed_potential_static_local_operator(
        grid_geometry=grid_geometry,
        rho_up=np.abs(trial_orbital) ** 2,
        rho_down=np.abs(trial_orbital) ** 2,
        spin_channel="up",
        use_monitor_patch=True,
        kinetic_version="trial_fix",
    )
    action = np.asarray(
        apply_fixed_potential_static_local_operator_jax(
            trial_orbital,
            operator_context=operator_context,
        ),
        dtype=np.float64,
    )
    assert action.shape == grid_geometry.spec.shape

    result = run_h2_jax_kernel_consistency_audit()
    assert result.reductions.overlap_matrix_max_abs_diff >= 0.0
    assert result.poisson.iteration_count >= 0
    assert result.local_hamiltonian.action_max_abs_diff >= 0.0
