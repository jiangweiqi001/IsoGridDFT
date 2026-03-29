"""Minimal smoke tests for the A-grid fixed-potential eigensolver audit."""

from importlib import import_module

import numpy as np

from isogrid.audit.h2_monitor_grid_fixed_potential_eigensolver_audit import (
    H2FixedPotentialCenterlineSample,
)
from isogrid.audit.h2_monitor_grid_fixed_potential_eigensolver_audit import (
    H2FixedPotentialRouteResult,
)
from isogrid.audit.h2_monitor_grid_patch_local_audit import H2MonitorPatchParameterSummary


def test_h2_monitor_grid_fixed_potential_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_fixed_potential_eigensolver_audit")

    assert hasattr(module, "run_h2_monitor_grid_fixed_potential_eigensolver_audit")
    assert hasattr(module, "print_h2_monitor_grid_fixed_potential_eigensolver_summary")


def test_import_jax_native_fixed_potential_entrypoint() -> None:
    module = import_module("isogrid.ks")

    assert hasattr(module, "solve_fixed_potential_static_local_eigenproblem_jax")


def test_construct_h2_fixed_potential_route_result() -> None:
    route = H2FixedPotentialRouteResult(
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
        grid_parameter_summary="shape=(67,67,81), box=(8,8,10)",
        patch_parameter_summary=H2MonitorPatchParameterSummary(
            patch_radius_scale=0.75,
            patch_grid_shape=(25, 25, 25),
            correction_strength=1.30,
            interpolation_neighbors=8,
        ),
        target_orbitals=2,
        solver_backend="jax",
        use_scipy_fallback=False,
        iteration_count=397,
        eigenvalues=np.asarray([-0.2, -0.19]),
        orbital_weighted_norms=np.asarray([1.0, 1.0]),
        max_orthogonality_error=1.0e-12,
        residual_norms=np.asarray([9.0e-4, 1.0e-3]),
        converged=True,
        solver_method="jax_block_subspace_iteration",
        solver_note="audit smoke",
        frozen_density_integral=2.0,
        rho_up_integral=1.0,
        rho_down_integral=1.0,
        patch_embedding_energy_mismatch=0.0,
        patch_embedded_correction_mha=77.815,
        centerline_samples=(
            H2FixedPotentialCenterlineSample(
                orbital_index=0,
                sample_index=40,
                z_coordinate_bohr=0.0,
                orbital_value=0.1,
            ),
        ),
    )

    assert route.path_type == "monitor_a_grid_plus_patch"
    assert route.solver_backend == "jax"
    assert route.use_scipy_fallback is False
    assert route.iteration_count == 397
    assert float(route.eigenvalues[0]) == -0.2
    assert float(route.residual_norms[1]) == 1.0e-3
    assert route.converged is True
    assert route.use_jax_block_kernels is False
    assert route.use_jax_cached_kernels is False
    assert route.wall_time_seconds is None
