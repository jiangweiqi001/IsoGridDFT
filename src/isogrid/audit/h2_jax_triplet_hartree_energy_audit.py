"""Very rough triplet-only SCF audit for JAX Hartree repeated-solve reuse.

This audit keeps the A-grid H2 dry-run restricted to the current local-only
Hamiltonian

    T + V_loc,ion + V_H + V_xc

with the repaired monitor-grid Hartree path, patch-assisted local ionic slice,
and the kinetic trial-fix branch. It compares two otherwise identical JAX-backed
SCF routes on the already-converged H2 triplet case:

- jax-hartree-baseline: JAX Poisson backend without explicit repeated-solve reuse
- jax-hartree-optimized: the same JAX backend with cached operator reuse enabled

The goal is not a formal benchmark. It is a small, auditable profile that
answers whether repeated monitor-grid Poisson solves in the SCF loop are paying
avoidable build/compile overhead and whether a thin cache actually lowers the
dominant Hartree bucket.
"""

from __future__ import annotations

from dataclasses import dataclass

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf import SinglePointEnergyComponents
from isogrid.scf import run_h2_monitor_grid_scf_dry_run

_TRIPLET_MAX_ITERATIONS = 20
_A_GRID_DRY_RUN_MIXING = 0.20
_A_GRID_DRY_RUN_DENSITY_TOLERANCE = 5.0e-3
_A_GRID_DRY_RUN_ENERGY_TOLERANCE = 5.0e-5
_A_GRID_DRY_RUN_EIGENSOLVER_TOLERANCE = 1.0e-3
_A_GRID_DRY_RUN_EIGENSOLVER_NCV = 20


@dataclass(frozen=True)
class H2TripletHartreeEnergyTimingBreakdown:
    """Very rough triplet SCF timing buckets for one route."""

    eigensolver_wall_time_seconds: float
    static_local_prepare_wall_time_seconds: float
    hartree_solve_wall_time_seconds: float
    local_ionic_resolve_wall_time_seconds: float
    xc_resolve_wall_time_seconds: float
    energy_evaluation_wall_time_seconds: float
    kinetic_energy_wall_time_seconds: float
    local_ionic_energy_wall_time_seconds: float
    hartree_energy_wall_time_seconds: float
    xc_energy_wall_time_seconds: float
    ion_ion_energy_wall_time_seconds: float
    density_update_wall_time_seconds: float
    bookkeeping_wall_time_seconds: float


@dataclass(frozen=True)
class H2TripletHartreeEnergyRouteResult:
    """Compact triplet SCF profiling summary for one JAX Hartree route."""

    path_label: str
    spin_state_label: str
    kinetic_version: str
    hartree_backend: str
    use_jax_block_kernels: bool
    use_step_local_static_local_reuse: bool
    use_jax_hartree_cached_operator: bool
    converged: bool
    iteration_count: int
    final_total_energy_ha: float
    lowest_eigenvalue_ha: float | None
    final_density_residual: float | None
    total_wall_time_seconds: float
    average_iteration_wall_time_seconds: float | None
    hartree_solve_call_count: int
    average_hartree_solve_wall_time_seconds: float | None
    first_hartree_solve_wall_time_seconds: float | None
    repeated_hartree_solve_average_wall_time_seconds: float | None
    average_hartree_cg_iterations: float | None
    first_hartree_cg_iterations: int | None
    repeated_hartree_cg_iteration_average: float | None
    hartree_cached_operator_usage_count: int
    hartree_cached_operator_first_solve_count: int
    timing_breakdown: H2TripletHartreeEnergyTimingBreakdown
    parameter_summary: str
    final_energy_components: SinglePointEnergyComponents


@dataclass(frozen=True)
class H2TripletHartreeEnergyAuditResult:
    """Top-level triplet-only SCF audit for JAX Hartree repeated solves."""

    jax_hartree_baseline_route: H2TripletHartreeEnergyRouteResult
    jax_hartree_optimized_route: H2TripletHartreeEnergyRouteResult
    note: str


def _monitor_parameter_summary(result: H2StaticLocalScfDryRunResult) -> str:
    parameters = result.parameter_summary
    return (
        "A-grid+patch+trial-fix triplet SCF JAX Hartree repeated-solve audit: "
        f"shape={parameters.grid_shape}, "
        f"box={parameters.box_half_extents_bohr}, "
        f"weight_scale={parameters.weight_scale:.2f}, "
        f"radius_scale={parameters.radius_scale:.2f}, "
        f"patch_radius_scale={parameters.patch_radius_scale:.2f}, "
        f"patch_grid_shape={parameters.patch_grid_shape}, "
        f"strength={parameters.correction_strength:.2f}, "
        f"neighbors={parameters.interpolation_neighbors}, "
        f"kinetic={parameters.kinetic_version}, "
        f"hartree_backend={parameters.hartree_backend}, "
        f"use_jax_hartree_cached_operator={parameters.use_jax_hartree_cached_operator}, "
        f"use_jax_block_kernels={parameters.use_jax_block_kernels}, "
        f"use_step_local_static_local_reuse={parameters.use_step_local_static_local_reuse}, "
        f"mixing={_A_GRID_DRY_RUN_MIXING:.2f}, "
        f"density_tol={_A_GRID_DRY_RUN_DENSITY_TOLERANCE:.1e}, "
        f"energy_tol={_A_GRID_DRY_RUN_ENERGY_TOLERANCE:.1e}, "
        f"eig_tol={_A_GRID_DRY_RUN_EIGENSOLVER_TOLERANCE:.1e}, "
        f"ncv={_A_GRID_DRY_RUN_EIGENSOLVER_NCV}"
    )


def _build_route_result(
    result: H2StaticLocalScfDryRunResult,
    *,
    path_label: str,
) -> H2TripletHartreeEnergyRouteResult:
    return H2TripletHartreeEnergyRouteResult(
        path_label=path_label,
        spin_state_label=result.spin_state_label,
        kinetic_version=result.kinetic_version,
        hartree_backend=result.hartree_backend,
        use_jax_block_kernels=bool(result.use_jax_block_kernels),
        use_step_local_static_local_reuse=bool(result.use_step_local_static_local_reuse),
        use_jax_hartree_cached_operator=bool(result.use_jax_hartree_cached_operator),
        converged=bool(result.converged),
        iteration_count=int(result.iteration_count),
        final_total_energy_ha=float(result.energy.total),
        lowest_eigenvalue_ha=result.lowest_eigenvalue,
        final_density_residual=(
            None if not result.history else float(result.history[-1].density_residual)
        ),
        total_wall_time_seconds=float(result.total_wall_time_seconds),
        average_iteration_wall_time_seconds=result.average_iteration_wall_time_seconds,
        hartree_solve_call_count=int(result.hartree_solve_call_count),
        average_hartree_solve_wall_time_seconds=result.average_hartree_solve_wall_time_seconds,
        first_hartree_solve_wall_time_seconds=result.first_hartree_solve_wall_time_seconds,
        repeated_hartree_solve_average_wall_time_seconds=(
            result.repeated_hartree_solve_average_wall_time_seconds
        ),
        average_hartree_cg_iterations=result.average_hartree_cg_iterations,
        first_hartree_cg_iterations=result.first_hartree_cg_iterations,
        repeated_hartree_cg_iteration_average=result.repeated_hartree_cg_iteration_average,
        hartree_cached_operator_usage_count=int(result.hartree_cached_operator_usage_count),
        hartree_cached_operator_first_solve_count=int(result.hartree_cached_operator_first_solve_count),
        timing_breakdown=H2TripletHartreeEnergyTimingBreakdown(
            eigensolver_wall_time_seconds=float(result.eigensolver_wall_time_seconds),
            static_local_prepare_wall_time_seconds=float(
                result.static_local_prepare_wall_time_seconds
            ),
            hartree_solve_wall_time_seconds=float(result.hartree_solve_wall_time_seconds),
            local_ionic_resolve_wall_time_seconds=float(
                result.local_ionic_resolve_wall_time_seconds
            ),
            xc_resolve_wall_time_seconds=float(result.xc_resolve_wall_time_seconds),
            energy_evaluation_wall_time_seconds=float(result.energy_evaluation_wall_time_seconds),
            kinetic_energy_wall_time_seconds=float(result.kinetic_energy_wall_time_seconds),
            local_ionic_energy_wall_time_seconds=float(
                result.local_ionic_energy_wall_time_seconds
            ),
            hartree_energy_wall_time_seconds=float(result.hartree_energy_wall_time_seconds),
            xc_energy_wall_time_seconds=float(result.xc_energy_wall_time_seconds),
            ion_ion_energy_wall_time_seconds=float(result.ion_ion_energy_wall_time_seconds),
            density_update_wall_time_seconds=float(result.density_update_wall_time_seconds),
            bookkeeping_wall_time_seconds=float(result.bookkeeping_wall_time_seconds),
        ),
        parameter_summary=_monitor_parameter_summary(result),
        final_energy_components=result.energy,
    )


def _run_route(
    *,
    case: BenchmarkCase,
    use_jax_hartree_cached_operator: bool,
) -> H2TripletHartreeEnergyRouteResult:
    path_label = (
        "jax-hartree-optimized"
        if use_jax_hartree_cached_operator
        else "jax-hartree-baseline"
    )
    return _build_route_result(
        run_h2_monitor_grid_scf_dry_run(
            "triplet",
            case=case,
            max_iterations=_TRIPLET_MAX_ITERATIONS,
            mixing=_A_GRID_DRY_RUN_MIXING,
            density_tolerance=_A_GRID_DRY_RUN_DENSITY_TOLERANCE,
            energy_tolerance=_A_GRID_DRY_RUN_ENERGY_TOLERANCE,
            eigensolver_tolerance=_A_GRID_DRY_RUN_EIGENSOLVER_TOLERANCE,
            eigensolver_ncv=_A_GRID_DRY_RUN_EIGENSOLVER_NCV,
            kinetic_version="trial_fix",
            hartree_backend="jax",
            use_jax_hartree_cached_operator=use_jax_hartree_cached_operator,
            use_jax_block_kernels=True,
            use_step_local_static_local_reuse=True,
        ),
        path_label=path_label,
    )


def run_h2_jax_triplet_hartree_energy_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2TripletHartreeEnergyAuditResult:
    """Run the triplet-only SCF profiling audit for JAX Hartree reuse."""

    jax_hartree_baseline_route = _run_route(
        case=case,
        use_jax_hartree_cached_operator=False,
    )
    jax_hartree_optimized_route = _run_route(
        case=case,
        use_jax_hartree_cached_operator=True,
    )
    return H2TripletHartreeEnergyAuditResult(
        jax_hartree_baseline_route=jax_hartree_baseline_route,
        jax_hartree_optimized_route=jax_hartree_optimized_route,
        note=(
            "Triplet-only A-grid SCF audit for JAX Hartree repeated-solve reuse. "
            "Both routes keep the JAX eigensolver hot path and step-local static-local reuse "
            "enabled; the only intended difference is whether the JAX Poisson operator callable "
            "is cached and reused across repeated solves inside the same SCF route. "
            "The `static_local_prepare` and `hartree_solve` buckets are diagnostic and overlap "
            "with `eigensolver` / `energy_eval`; they should not be summed with the total wall time."
        ),
    )


def _print_route(route: H2TripletHartreeEnergyRouteResult) -> None:
    print(f"  route: {route.path_label}")
    print(f"    hartree_backend: {route.hartree_backend}")
    print(f"    use_jax_hartree_cached_operator: {route.use_jax_hartree_cached_operator}")
    print(f"    converged: {route.converged}")
    print(f"    iterations: {route.iteration_count}")
    print(f"    final total energy [Ha]: {route.final_total_energy_ha:.12f}")
    if route.lowest_eigenvalue_ha is None:
        print("    final lowest eigenvalue [Ha]: n/a")
    else:
        print(f"    final lowest eigenvalue [Ha]: {route.lowest_eigenvalue_ha:.12f}")
    print(
        "    timing [s]: "
        f"total={route.total_wall_time_seconds:.6f}, "
        f"avg/iter={(route.average_iteration_wall_time_seconds or 0.0):.6f}, "
        f"eigensolver={route.timing_breakdown.eigensolver_wall_time_seconds:.6f}, "
        f"prepare={route.timing_breakdown.static_local_prepare_wall_time_seconds:.6f}, "
        f"hartree_solve={route.timing_breakdown.hartree_solve_wall_time_seconds:.6f}, "
        f"energy_eval={route.timing_breakdown.energy_evaluation_wall_time_seconds:.6f}"
    )
    print(
        "    repeated-solve [s]: "
        f"first={0.0 if route.first_hartree_solve_wall_time_seconds is None else route.first_hartree_solve_wall_time_seconds:.6f}, "
        f"avg={0.0 if route.average_hartree_solve_wall_time_seconds is None else route.average_hartree_solve_wall_time_seconds:.6f}, "
        f"repeated_avg={0.0 if route.repeated_hartree_solve_average_wall_time_seconds is None else route.repeated_hartree_solve_average_wall_time_seconds:.6f}"
    )
    print(
        "    repeated-solve iters: "
        f"first={route.first_hartree_cg_iterations}, "
        f"avg={route.average_hartree_cg_iterations}, "
        f"repeated_avg={route.repeated_hartree_cg_iteration_average}"
    )
    print(
        "    cache stats: "
        f"solve_calls={route.hartree_solve_call_count}, "
        f"cached_use_count={route.hartree_cached_operator_usage_count}, "
        f"first_cached_solve_count={route.hartree_cached_operator_first_solve_count}"
    )


def print_h2_jax_triplet_hartree_energy_summary(
    result: H2TripletHartreeEnergyAuditResult,
) -> None:
    """Print the compact triplet profiling summary."""

    print("IsoGridDFT H2 triplet JAX Hartree repeated-solve audit")
    print(f"note: {result.note}")
    print()
    _print_route(result.jax_hartree_baseline_route)
    _print_route(result.jax_hartree_optimized_route)
    print()
    print(
        "  total timing delta [s]: "
        f"{result.jax_hartree_optimized_route.total_wall_time_seconds - result.jax_hartree_baseline_route.total_wall_time_seconds:+.6f}"
    )
    if result.jax_hartree_baseline_route.total_wall_time_seconds > 0.0:
        print(
            "  timing ratio (optimized/baseline): "
            f"{result.jax_hartree_optimized_route.total_wall_time_seconds / result.jax_hartree_baseline_route.total_wall_time_seconds:.6f}"
        )


def main() -> int:
    result = run_h2_jax_triplet_hartree_energy_audit()
    print_h2_jax_triplet_hartree_energy_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
