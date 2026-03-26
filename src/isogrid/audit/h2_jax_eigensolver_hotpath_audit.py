"""Very small old-vs-JAX hot-path audit for the H2 fixed-potential eigensolver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE

from .h2_monitor_grid_fixed_potential_eigensolver_audit import (
    H2FixedPotentialRouteResult,
)
from .h2_monitor_grid_fixed_potential_eigensolver_audit import (
    _evaluate_route,
)


@dataclass(frozen=True)
class H2JaxEigensolverHotpathComparison:
    """Old-vs-JAX comparison for one fixed-potential target-orbital count."""

    target_orbitals: int
    old_route: H2FixedPotentialRouteResult
    jax_route: H2FixedPotentialRouteResult
    eigenvalue_max_abs_diff: float
    residual_max_abs_diff: float
    orthogonality_abs_diff: float
    wall_time_ratio_jax_over_old: float | None


@dataclass(frozen=True)
class H2JaxEigensolverHotpathAuditResult:
    """Very small correctness/timing audit for the JAX eigensolver hot path."""

    k1_comparison: H2JaxEigensolverHotpathComparison
    k2_comparison: H2JaxEigensolverHotpathComparison | None
    note: str


def _build_comparison(
    *,
    old_route: H2FixedPotentialRouteResult,
    jax_route: H2FixedPotentialRouteResult,
) -> H2JaxEigensolverHotpathComparison:
    old_time = old_route.wall_time_seconds
    jax_time = jax_route.wall_time_seconds
    if old_time is None or old_time <= 0.0 or jax_time is None:
        time_ratio = None
    else:
        time_ratio = float(jax_time / old_time)
    return H2JaxEigensolverHotpathComparison(
        target_orbitals=old_route.target_orbitals,
        old_route=old_route,
        jax_route=jax_route,
        eigenvalue_max_abs_diff=float(
            np.max(np.abs(old_route.eigenvalues - jax_route.eigenvalues))
        ),
        residual_max_abs_diff=float(
            np.max(np.abs(old_route.residual_norms - jax_route.residual_norms))
        ),
        orthogonality_abs_diff=float(
            abs(old_route.max_orthogonality_error - jax_route.max_orthogonality_error)
        ),
        wall_time_ratio_jax_over_old=time_ratio,
    )


def run_h2_jax_eigensolver_hotpath_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2JaxEigensolverHotpathAuditResult:
    """Compare the old and JAX block hot paths on the H2 fixed-potential route."""

    _evaluate_route(
        case=case,
        path_type="monitor_a_grid_plus_patch",
        k=1,
        kinetic_version="trial_fix",
        use_jax_block_kernels=True,
    )

    k1 = _build_comparison(
        old_route=_evaluate_route(
            case=case,
            path_type="monitor_a_grid_plus_patch",
            k=1,
            kinetic_version="trial_fix",
            use_jax_block_kernels=False,
        ),
        jax_route=_evaluate_route(
            case=case,
            path_type="monitor_a_grid_plus_patch",
            k=1,
            kinetic_version="trial_fix",
            use_jax_block_kernels=True,
        ),
    )
    k2 = _build_comparison(
        old_route=_evaluate_route(
            case=case,
            path_type="monitor_a_grid_plus_patch",
            k=2,
            kinetic_version="trial_fix",
            use_jax_block_kernels=False,
        ),
        jax_route=_evaluate_route(
            case=case,
            path_type="monitor_a_grid_plus_patch",
            k=2,
            kinetic_version="trial_fix",
            use_jax_block_kernels=True,
        ),
    )
    return H2JaxEigensolverHotpathAuditResult(
        k1_comparison=k1,
        k2_comparison=k2,
        note=(
            "Very small H2 fixed-potential audit for the first JAX eigensolver block-hot-path "
            "handoff. The outer eigensolver iteration, Ritz solve, and convergence control remain "
            "in Python; only the block Hamiltonian apply and weighted block linear algebra are "
            "switched between the old and JAX hot paths. Timing is only a rough post-warmup "
            "reference and should not be interpreted as a benchmark."
        ),
    )


def print_h2_jax_eigensolver_hotpath_summary(
    result: H2JaxEigensolverHotpathAuditResult,
) -> None:
    """Print the compact old-vs-JAX hot-path summary."""

    print("IsoGridDFT H2 JAX eigensolver hot-path audit")
    print(f"note: {result.note}")
    for comparison in (result.k1_comparison, result.k2_comparison):
        if comparison is None:
            continue
        print()
        print(f"k = {comparison.target_orbitals}")
        print(
            "  old route: "
            f"converged={comparison.old_route.converged}, "
            f"eigenvalues={comparison.old_route.eigenvalues.tolist()}, "
            f"residuals={comparison.old_route.residual_norms.tolist()}, "
            f"orth_err={comparison.old_route.max_orthogonality_error:.6e}, "
            f"wall_time={comparison.old_route.wall_time_seconds:.6f}s"
        )
        print(
            "  jax route: "
            f"converged={comparison.jax_route.converged}, "
            f"eigenvalues={comparison.jax_route.eigenvalues.tolist()}, "
            f"residuals={comparison.jax_route.residual_norms.tolist()}, "
            f"orth_err={comparison.jax_route.max_orthogonality_error:.6e}, "
            f"wall_time={comparison.jax_route.wall_time_seconds:.6f}s"
        )
        print(
            "  diffs: "
            f"eig_max={comparison.eigenvalue_max_abs_diff:.6e}, "
            f"residual_max={comparison.residual_max_abs_diff:.6e}, "
            f"orth={comparison.orthogonality_abs_diff:.6e}"
        )
        if comparison.wall_time_ratio_jax_over_old is not None:
            print(
                "  timing ratio (jax/old): "
                f"{comparison.wall_time_ratio_jax_over_old:.6f}"
            )


def main() -> int:
    result = run_h2_jax_eigensolver_hotpath_audit()
    print_h2_jax_eigensolver_hotpath_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
