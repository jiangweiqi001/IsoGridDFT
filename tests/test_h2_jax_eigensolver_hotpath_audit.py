"""Minimal smoke tests for the JAX eigensolver hot-path audit."""

from importlib import import_module

import numpy as np

from isogrid.audit.h2_jax_eigensolver_hotpath_audit import (
    H2JaxEigensolverHotpathAuditResult,
)
from isogrid.audit.h2_jax_eigensolver_hotpath_audit import (
    H2JaxEigensolverHotpathComparison,
)
from isogrid.audit.h2_monitor_grid_fixed_potential_eigensolver_audit import (
    H2FixedPotentialRouteResult,
)


def _route(*, use_jax_block_kernels: bool) -> H2FixedPotentialRouteResult:
    return H2FixedPotentialRouteResult(
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
        grid_parameter_summary="shape=(67,67,81), box=(8,8,10)",
        patch_parameter_summary=None,
        target_orbitals=1,
        eigenvalues=np.asarray([-0.1866]),
        orbital_weighted_norms=np.asarray([1.0]),
        max_orthogonality_error=1.0e-12,
        residual_norms=np.asarray([1.0e-4]),
        converged=True,
        solver_method="scipy_eigsh_lanczos",
        solver_note="audit smoke",
        frozen_density_integral=2.0,
        rho_up_integral=1.0,
        rho_down_integral=1.0,
        patch_embedding_energy_mismatch=0.0,
        patch_embedded_correction_mha=77.815,
        centerline_samples=(),
        use_jax_block_kernels=use_jax_block_kernels,
        wall_time_seconds=0.1 if use_jax_block_kernels else 0.2,
    )


def test_h2_jax_eigensolver_hotpath_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_jax_eigensolver_hotpath_audit")

    assert hasattr(module, "run_h2_jax_eigensolver_hotpath_audit")
    assert hasattr(module, "print_h2_jax_eigensolver_hotpath_summary")


def test_construct_h2_jax_eigensolver_hotpath_result() -> None:
    comparison = H2JaxEigensolverHotpathComparison(
        target_orbitals=1,
        old_route=_route(use_jax_block_kernels=False),
        jax_route=_route(use_jax_block_kernels=True),
        eigenvalue_max_abs_diff=1.0e-12,
        residual_max_abs_diff=1.0e-12,
        orthogonality_abs_diff=1.0e-13,
        wall_time_ratio_jax_over_old=0.5,
    )
    result = H2JaxEigensolverHotpathAuditResult(
        k1_comparison=comparison,
        k2_comparison=None,
        note="smoke",
    )

    assert result.k1_comparison.jax_route.use_jax_block_kernels is True
    assert float(result.k1_comparison.jax_route.eigenvalues[0]) == -0.1866
    assert float(result.k1_comparison.residual_max_abs_diff) == 1.0e-12
    assert float(result.k1_comparison.orthogonality_abs_diff) == 1.0e-13
