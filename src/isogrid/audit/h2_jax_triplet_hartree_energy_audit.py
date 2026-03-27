"""Very rough triplet-only SCF audit for Hartree and energy-evaluation reuse.

This audit keeps the A-grid H2 dry-run restricted to the current local-only
Hamiltonian

    T + V_loc,ion + V_H + V_xc

with the repaired monitor-grid Hartree path, patch-assisted local ionic slice,
and the kinetic trial-fix branch. It compares two JAX-backed SCF routes on the
already-converged H2 triplet case:

- the current JAX SCF hot path without per-step local reuse
- the same JAX SCF hot path with very small step-local reuse of static-local
  preparation and single-point energy inputs

The goal is not a formal benchmark. It is a small, auditable profile that
answers whether repeated Poisson/Hartree and energy-evaluation work inside one
SCF step can be reduced without changing the outer SCF algorithm.
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
    """Compact triplet SCF profiling summary for one JAX route."""

    path_label: str
    spin_state_label: str
    kinetic_version: str
    use_jax_block_kernels: bool
    use_step_local_static_local_reuse: bool
    converged: bool
    iteration_count: int
    final_total_energy_ha: float
    lowest_eigenvalue_ha: float | None
    final_density_residual: float | None
    total_wall_time_seconds: float
    average_iteration_wall_time_seconds: float | None
    hartree_solve_call_count: int
    timing_breakdown: H2TripletHartreeEnergyTimingBreakdown
    parameter_summary: str
    final_energy_components: SinglePointEnergyComponents


@dataclass(frozen=True)
class H2TripletHartreeEnergyAuditResult:
    """Top-level triplet-only SCF audit for repeated Hartree/energy work."""

    jax_baseline: H2TripletHartreeEnergyRouteResult
    jax_optimized: H2TripletHartreeEnergyRouteResult
    note: str


def _monitor_parameter_summary(result: H2StaticLocalScfDryRunResult) -> str:
    parameters = result.parameter_summary
    return (
        "A-grid+patch+trial-fix triplet SCF Hartree/energy audit: "
        f"shape={parameters.grid_shape}, "
        f"box={parameters.box_half_extents_bohr}, "
        f"weight_scale={parameters.weight_scale:.2f}, "
        f"radius_scale={parameters.radius_scale:.2f}, "
        f"patch_radius_scale={parameters.patch_radius_scale:.2f}, "
        f"patch_grid_shape={parameters.patch_grid_shape}, "
        f"strength={parameters.correction_strength:.2f}, "
        f"neighbors={parameters.interpolation_neighbors}, "
        f"kinetic={parameters.kinetic_version}, "
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
        use_jax_block_kernels=bool(result.use_jax_block_kernels),
        use_step_local_static_local_reuse=bool(result.use_step_local_static_local_reuse),
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
    use_step_local_static_local_reuse: bool,
) -> H2TripletHartreeEnergyRouteResult:
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
            use_jax_block_kernels=True,
            use_step_local_static_local_reuse=use_step_local_static_local_reuse,
        ),
        path_label="jax-optimized" if use_step_local_static_local_reuse else "jax-baseline",
    )


def run_h2_jax_triplet_hartree_energy_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2TripletHartreeEnergyAuditResult:
    """Run the triplet-only SCF profiling audit for step-local reuse."""

    jax_baseline = _run_route(case=case, use_step_local_static_local_reuse=False)
    jax_optimized = _run_route(case=case, use_step_local_static_local_reuse=True)
    return H2TripletHartreeEnergyAuditResult(
        jax_baseline=jax_baseline,
        jax_optimized=jax_optimized,
        note=(
            "Triplet-only A-grid SCF audit for the current repaired JAX hot path. The baseline "
            "route uses the JAX eigensolver hot path without step-local reuse, while the optimized "
            "route reuses the density-independent base local ionic evaluation and evaluates the "
            "single-point energy from one freshly prepared output-density context inside each SCF "
            "step. Nonlocal remains absent."
        ),
    )


def _print_route(route: H2TripletHartreeEnergyRouteResult) -> None:
    print(f"  route: {route.path_label}")
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
        "    sub-breakdown [s]: "
        f"local_resolve={route.timing_breakdown.local_ionic_resolve_wall_time_seconds:.6f}, "
        f"xc_resolve={route.timing_breakdown.xc_resolve_wall_time_seconds:.6f}, "
        f"kinetic_E={route.timing_breakdown.kinetic_energy_wall_time_seconds:.6f}, "
        f"local_E={route.timing_breakdown.local_ionic_energy_wall_time_seconds:.6f}, "
        f"hartree_E={route.timing_breakdown.hartree_energy_wall_time_seconds:.6f}, "
        f"xc_E={route.timing_breakdown.xc_energy_wall_time_seconds:.6f}, "
        f"ion_ion_E={route.timing_breakdown.ion_ion_energy_wall_time_seconds:.6f}, "
        f"density_update={route.timing_breakdown.density_update_wall_time_seconds:.6f}, "
        f"bookkeeping={route.timing_breakdown.bookkeeping_wall_time_seconds:.6f}"
    )
    print(f"    hartree_solve_call_count: {route.hartree_solve_call_count}")


def print_h2_jax_triplet_hartree_energy_summary(
    result: H2TripletHartreeEnergyAuditResult,
) -> None:
    """Print the compact triplet profiling summary."""

    print("IsoGridDFT H2 triplet JAX Hartree/energy audit")
    print(f"note: {result.note}")
    print()
    _print_route(result.jax_baseline)
    _print_route(result.jax_optimized)
    print()
    print(
        "  total timing delta [s]: "
        f"{result.jax_optimized.total_wall_time_seconds - result.jax_baseline.total_wall_time_seconds:+.6f}"
    )
    if result.jax_baseline.total_wall_time_seconds > 0.0:
        print(
            "  timing ratio (optimized/baseline): "
            f"{result.jax_optimized.total_wall_time_seconds / result.jax_baseline.total_wall_time_seconds:.6f}"
        )


def main() -> int:
    result = run_h2_jax_triplet_hartree_energy_audit()
    print_h2_jax_triplet_hartree_energy_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
