"""Very rough triplet-only SCF audit for the JAX Hartree stronger-PCG path.

This audit keeps the A-grid H2 dry-run restricted to the current local-only
Hamiltonian

    T + V_loc,ion + V_H + V_xc

with the repaired monitor-grid Hartree path, patch-assisted local ionic slice,
and the kinetic trial-fix branch. It compares two otherwise identical JAX-backed
SCF routes on the already-converged H2 triplet case:

- jax-hartree-cgloop: cached JAX Hartree operator with `cg_impl="jax_loop"`
  and no preconditioner
- jax-hartree-pcg-stronger: the same JAX-native CG inner loop plus one stronger
  metric-aware line preconditioner that exactly solves the 1D tridiagonal
  block along the stiffest logical axis while ignoring the weaker transverse
  couplings

The goal is not a formal benchmark. It is a small, auditable profile that
answers whether a stronger-but-still-small preconditioner can push the
still-large Hartree iteration count down enough to matter end-to-end for the
H2 triplet SCF dry-run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.poisson import solve_hartree_potential
from isogrid.poisson.poisson_jax import clear_monitor_poisson_jax_kernel_cache
from isogrid.poisson.poisson_jax import get_last_monitor_poisson_jax_solve_diagnostics
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf import SinglePointEnergyComponents
from isogrid.scf import build_h2_initial_density_guess
from isogrid.scf import resolve_h2_spin_occupations
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
    cg_impl: str
    cg_preconditioner: str
    use_jax_block_kernels: bool
    use_step_local_static_local_reuse: bool
    use_jax_hartree_cached_operator: bool
    matvec_timing_is_estimated: bool
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
    repeated_hartree_solve_min_wall_time_seconds: float | None
    repeated_hartree_solve_max_wall_time_seconds: float | None
    average_hartree_cg_iterations: float | None
    first_hartree_cg_iterations: int | None
    repeated_hartree_cg_iteration_average: float | None
    average_hartree_boundary_condition_wall_time_seconds: float | None
    average_hartree_build_wall_time_seconds: float | None
    average_hartree_rhs_assembly_wall_time_seconds: float | None
    average_hartree_cg_wall_time_seconds: float | None
    average_hartree_cg_other_overhead_wall_time_seconds: float | None
    average_hartree_matvec_call_count: float | None
    average_hartree_matvec_wall_time_seconds: float | None
    average_hartree_matvec_wall_time_per_call_seconds: float | None
    average_hartree_preconditioner_apply_count: float | None
    average_hartree_preconditioner_apply_wall_time_seconds: float | None
    average_hartree_preconditioner_apply_wall_time_per_call_seconds: float | None
    average_hartree_preconditioner_setup_wall_time_seconds: float | None
    average_hartree_preconditioner_axis_reorder_wall_time_seconds: float | None
    average_hartree_preconditioner_tridiagonal_solve_wall_time_seconds: float | None
    average_hartree_preconditioner_other_overhead_wall_time_seconds: float | None
    average_hartree_cg_iteration_wall_time_seconds: float | None
    average_hartree_matvec_wall_time_per_iteration_seconds: float | None
    average_hartree_other_cg_overhead_wall_time_per_iteration_seconds: float | None
    first_hartree_matvec_call_count: int | None
    repeated_hartree_matvec_call_count_average: float | None
    first_hartree_matvec_wall_time_seconds: float | None
    repeated_hartree_matvec_average_wall_time_seconds: float | None
    first_hartree_matvec_wall_time_per_call_seconds: float | None
    repeated_hartree_matvec_wall_time_per_call_seconds: float | None
    hartree_cached_operator_usage_count: int
    hartree_cached_operator_first_solve_count: int
    timing_breakdown: H2TripletHartreeEnergyTimingBreakdown
    parameter_summary: str
    final_energy_components: SinglePointEnergyComponents


@dataclass(frozen=True)
class H2TripletHartreeSingleSolveResult:
    """Same-density single-solve comparison for one JAX CG implementation."""

    path_label: str
    cg_impl: str
    cg_preconditioner: str
    converged: bool
    residual_max: float
    iteration_count: int
    total_solve_time_seconds: float
    cg_wall_time_seconds: float
    matvec_wall_time_seconds: float
    cg_other_overhead_wall_time_seconds: float
    preconditioner_apply_count: int
    preconditioner_apply_wall_time_seconds: float
    preconditioner_setup_wall_time_seconds: float
    preconditioner_axis_reorder_wall_time_seconds: float
    preconditioner_tridiagonal_solve_wall_time_seconds: float
    preconditioner_other_overhead_wall_time_seconds: float
    matvec_call_count: int
    average_iteration_wall_time_seconds: float | None
    average_matvec_wall_time_seconds: float | None
    average_matvec_wall_time_per_call_seconds: float | None
    average_preconditioner_apply_wall_time_seconds: float | None
    average_preconditioner_apply_wall_time_per_call_seconds: float | None
    matvec_timing_is_estimated: bool


@dataclass(frozen=True)
class H2TripletHartreeEnergyAuditResult:
    """Top-level triplet-only SCF audit for the JAX Hartree PCG prototype."""

    jax_hartree_cgloop_route: H2TripletHartreeEnergyRouteResult
    jax_hartree_pcg_stronger_route: H2TripletHartreeEnergyRouteResult
    single_solve_cgloop: H2TripletHartreeSingleSolveResult
    single_solve_diag: H2TripletHartreeSingleSolveResult | None
    single_solve_pcg_stronger: H2TripletHartreeSingleSolveResult
    note: str


def _average(values: tuple[float, ...] | tuple[int, ...]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _monitor_parameter_summary(result: H2StaticLocalScfDryRunResult) -> str:
    parameters = result.parameter_summary
    return (
        "A-grid+patch+trial-fix triplet SCF JAX Hartree PCG feasibility audit: "
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
        f"jax_hartree_cg_impl={parameters.jax_hartree_cg_impl}, "
        f"jax_hartree_cg_preconditioner={parameters.jax_hartree_cg_preconditioner}, "
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
    cg_iterations = result.hartree_cg_iterations_history
    cg_times = result.hartree_cg_wall_time_seconds_history
    matvec_counts = result.hartree_matvec_call_count_history
    matvec_times = result.hartree_matvec_wall_time_seconds_history
    repeated_matvec_counts = matvec_counts[1:]
    repeated_matvec_times = matvec_times[1:]
    average_other_cg_overhead = (
        None
        if not cg_times or not matvec_times
        else float(
            np.mean(
                [max(0.0, cg - mv) for cg, mv in zip(cg_times, matvec_times, strict=False)]
            )
        )
    )
    average_iteration_wall = (
        None
        if not cg_times or not cg_iterations or sum(cg_iterations) == 0
        else float(sum(cg_times) / sum(cg_iterations))
    )
    average_matvec_per_iteration = (
        None
        if not matvec_times or not cg_iterations or sum(cg_iterations) == 0
        else float(sum(matvec_times) / sum(cg_iterations))
    )
    average_other_overhead_per_iteration = (
        None
        if average_iteration_wall is None or average_matvec_per_iteration is None
        else max(0.0, average_iteration_wall - average_matvec_per_iteration)
    )
    first_matvec_per_call = (
        None
        if not matvec_times or not matvec_counts or matvec_counts[0] == 0
        else float(matvec_times[0] / matvec_counts[0])
    )
    repeated_matvec_average = _average(repeated_matvec_times)
    repeated_matvec_count_average = _average(repeated_matvec_counts)
    repeated_matvec_per_call = (
        None
        if repeated_matvec_average is None
        or repeated_matvec_count_average is None
        or repeated_matvec_count_average == 0.0
        else float(repeated_matvec_average / repeated_matvec_count_average)
    )
    return H2TripletHartreeEnergyRouteResult(
        path_label=path_label,
        spin_state_label=result.spin_state_label,
        kinetic_version=result.kinetic_version,
        hartree_backend=result.hartree_backend,
        cg_impl=result.jax_hartree_cg_impl,
        cg_preconditioner=result.jax_hartree_cg_preconditioner,
        use_jax_block_kernels=bool(result.use_jax_block_kernels),
        use_step_local_static_local_reuse=bool(result.use_step_local_static_local_reuse),
        use_jax_hartree_cached_operator=bool(result.use_jax_hartree_cached_operator),
        matvec_timing_is_estimated=(result.jax_hartree_cg_impl == "jax_loop"),
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
        repeated_hartree_solve_min_wall_time_seconds=(
            result.repeated_hartree_solve_min_wall_time_seconds
        ),
        repeated_hartree_solve_max_wall_time_seconds=(
            result.repeated_hartree_solve_max_wall_time_seconds
        ),
        average_hartree_cg_iterations=result.average_hartree_cg_iterations,
        first_hartree_cg_iterations=result.first_hartree_cg_iterations,
        repeated_hartree_cg_iteration_average=result.repeated_hartree_cg_iteration_average,
        average_hartree_boundary_condition_wall_time_seconds=(
            result.average_hartree_boundary_condition_wall_time_seconds
        ),
        average_hartree_build_wall_time_seconds=result.average_hartree_build_wall_time_seconds,
        average_hartree_rhs_assembly_wall_time_seconds=(
            result.average_hartree_rhs_assembly_wall_time_seconds
        ),
        average_hartree_cg_wall_time_seconds=result.average_hartree_cg_wall_time_seconds,
        average_hartree_cg_other_overhead_wall_time_seconds=average_other_cg_overhead,
        average_hartree_matvec_call_count=result.average_hartree_matvec_call_count,
        average_hartree_matvec_wall_time_seconds=result.average_hartree_matvec_wall_time_seconds,
        average_hartree_matvec_wall_time_per_call_seconds=(
            result.average_hartree_matvec_wall_time_per_call_seconds
        ),
        average_hartree_preconditioner_apply_count=(
            result.average_hartree_preconditioner_apply_count
        ),
        average_hartree_preconditioner_apply_wall_time_seconds=(
            result.average_hartree_preconditioner_apply_wall_time_seconds
        ),
        average_hartree_preconditioner_apply_wall_time_per_call_seconds=(
            result.average_hartree_preconditioner_apply_wall_time_per_call_seconds
        ),
        average_hartree_preconditioner_setup_wall_time_seconds=(
            result.average_hartree_preconditioner_setup_wall_time_seconds
        ),
        average_hartree_preconditioner_axis_reorder_wall_time_seconds=(
            result.average_hartree_preconditioner_axis_reorder_wall_time_seconds
        ),
        average_hartree_preconditioner_tridiagonal_solve_wall_time_seconds=(
            result.average_hartree_preconditioner_tridiagonal_solve_wall_time_seconds
        ),
        average_hartree_preconditioner_other_overhead_wall_time_seconds=(
            result.average_hartree_preconditioner_other_overhead_wall_time_seconds
        ),
        average_hartree_cg_iteration_wall_time_seconds=average_iteration_wall,
        average_hartree_matvec_wall_time_per_iteration_seconds=average_matvec_per_iteration,
        average_hartree_other_cg_overhead_wall_time_per_iteration_seconds=(
            average_other_overhead_per_iteration
        ),
        first_hartree_matvec_call_count=(None if not matvec_counts else int(matvec_counts[0])),
        repeated_hartree_matvec_call_count_average=repeated_matvec_count_average,
        first_hartree_matvec_wall_time_seconds=(None if not matvec_times else float(matvec_times[0])),
        repeated_hartree_matvec_average_wall_time_seconds=repeated_matvec_average,
        first_hartree_matvec_wall_time_per_call_seconds=first_matvec_per_call,
        repeated_hartree_matvec_wall_time_per_call_seconds=repeated_matvec_per_call,
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
    cg_preconditioner: str,
) -> tuple[H2TripletHartreeEnergyRouteResult, H2StaticLocalScfDryRunResult]:
    if cg_preconditioner == "none":
        path_label = "jax-hartree-cgloop"
    elif cg_preconditioner == "line":
        path_label = "jax-hartree-pcg-stronger"
    else:
        path_label = f"jax-hartree-{cg_preconditioner}"
    raw_result = run_h2_monitor_grid_scf_dry_run(
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
        use_jax_hartree_cached_operator=True,
        jax_hartree_cg_impl="jax_loop",
        jax_hartree_cg_preconditioner=cg_preconditioner,
        use_jax_block_kernels=True,
        use_step_local_static_local_reuse=True,
    )
    return _build_route_result(raw_result, path_label=path_label), raw_result


def _run_single_solve(
    *,
    rho_total: np.ndarray,
    cg_preconditioner: str,
) -> H2TripletHartreeSingleSolveResult:
    grid_geometry = build_h2_local_patch_development_monitor_grid()
    clear_monitor_poisson_jax_kernel_cache()
    solve_hartree_potential(
        grid_geometry=grid_geometry,
        rho=rho_total,
        backend="jax",
        use_jax_cached_operator=True,
        cg_impl="jax_loop",
        cg_preconditioner=cg_preconditioner,
    )
    diagnostics = get_last_monitor_poisson_jax_solve_diagnostics()
    if diagnostics is None:
        raise RuntimeError("Expected JAX Hartree solve diagnostics for single-solve audit.")
    average_iteration_wall = (
        None
        if diagnostics.iteration_count <= 0
        else float(diagnostics.cg_wall_time_seconds / diagnostics.iteration_count)
    )
    average_preconditioner_apply_wall = (
        None
        if diagnostics.preconditioner_apply_count <= 0
        else float(
            diagnostics.preconditioner_apply_wall_time_seconds
            / diagnostics.preconditioner_apply_count
        )
    )
    average_matvec_wall = (
        None
        if diagnostics.matvec_call_count <= 0
        else float(diagnostics.matvec_wall_time_seconds / diagnostics.matvec_call_count)
    )
    return H2TripletHartreeSingleSolveResult(
        path_label=(
            "single-solve-cgloop"
            if cg_preconditioner == "none"
            else "single-solve-diag"
            if cg_preconditioner == "diag"
            else "single-solve-pcg-stronger"
        ),
        cg_impl="jax_loop",
        cg_preconditioner=cg_preconditioner,
        converged=bool(diagnostics.converged),
        residual_max=float(diagnostics.residual_max),
        iteration_count=int(diagnostics.iteration_count),
        total_solve_time_seconds=float(diagnostics.total_wall_time_seconds),
        cg_wall_time_seconds=float(diagnostics.cg_wall_time_seconds),
        matvec_wall_time_seconds=float(diagnostics.matvec_wall_time_seconds),
        cg_other_overhead_wall_time_seconds=float(
            diagnostics.cg_other_overhead_wall_time_seconds
        ),
        preconditioner_apply_count=int(diagnostics.preconditioner_apply_count),
        preconditioner_apply_wall_time_seconds=float(
            diagnostics.preconditioner_apply_wall_time_seconds
        ),
        preconditioner_setup_wall_time_seconds=float(
            diagnostics.preconditioner_setup_wall_time_seconds
        ),
        preconditioner_axis_reorder_wall_time_seconds=float(
            diagnostics.preconditioner_axis_reorder_wall_time_seconds
        ),
        preconditioner_tridiagonal_solve_wall_time_seconds=float(
            diagnostics.preconditioner_tridiagonal_solve_wall_time_seconds
        ),
        preconditioner_other_overhead_wall_time_seconds=float(
            diagnostics.preconditioner_other_overhead_wall_time_seconds
        ),
        matvec_call_count=int(diagnostics.matvec_call_count),
        average_iteration_wall_time_seconds=average_iteration_wall,
        average_matvec_wall_time_seconds=average_matvec_wall,
        average_matvec_wall_time_per_call_seconds=average_matvec_wall,
        average_preconditioner_apply_wall_time_seconds=average_preconditioner_apply_wall,
        average_preconditioner_apply_wall_time_per_call_seconds=average_preconditioner_apply_wall,
        matvec_timing_is_estimated=bool(diagnostics.matvec_timing_is_estimated),
    )


def run_h2_jax_triplet_hartree_energy_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2TripletHartreeEnergyAuditResult:
    """Run the triplet-only SCF profiling audit for the stronger JAX Hartree PCG path."""

    jax_hartree_cgloop_route, cgloop_raw = _run_route(case=case, cg_preconditioner="none")
    jax_hartree_pcg_stronger_route, _ = _run_route(case=case, cg_preconditioner="line")
    occupations = resolve_h2_spin_occupations("triplet", case=case)
    initial_rho_up, initial_rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=case,
        grid_geometry=build_h2_local_patch_development_monitor_grid(),
    )
    rho_total = np.asarray(initial_rho_up + initial_rho_down, dtype=np.float64)
    single_solve_cgloop = _run_single_solve(rho_total=rho_total, cg_preconditioner="none")
    single_solve_diag = _run_single_solve(rho_total=rho_total, cg_preconditioner="diag")
    single_solve_pcg_stronger = _run_single_solve(
        rho_total=rho_total,
        cg_preconditioner="line",
    )
    return H2TripletHartreeEnergyAuditResult(
        jax_hartree_cgloop_route=jax_hartree_cgloop_route,
        jax_hartree_pcg_stronger_route=jax_hartree_pcg_stronger_route,
        single_solve_cgloop=single_solve_cgloop,
        single_solve_diag=single_solve_diag,
        single_solve_pcg_stronger=single_solve_pcg_stronger,
        note=(
            "Triplet-only A-grid SCF audit for the JAX Hartree stronger-PCG feasibility path. "
            "Both main routes keep the JAX eigensolver hot path, step-local static-local reuse, "
            "cached Hartree operator reuse, and `cg_impl='jax_loop'` enabled; the only intended "
            "difference is `cg_preconditioner='none'` versus the stronger "
            "`cg_preconditioner='line'` route. The current tiny `diag` route is retained only "
            "as a same-density single-solve reference. The `static_local_prepare` and `hartree_solve` "
            "buckets are diagnostic and overlap with `eigensolver` / `energy_eval`; they should "
            "not be summed with the total wall time. For `cg_impl='jax_loop'`, matvec timing "
            "remains probe-based and should be interpreted as an approximate attribution."
        ),
    )


def _print_route(route: H2TripletHartreeEnergyRouteResult) -> None:
    print(f"  route: {route.path_label}")
    print(f"    hartree_backend: {route.hartree_backend}")
    print(f"    cg_impl: {route.cg_impl}")
    print(f"    cg_preconditioner: {route.cg_preconditioner}")
    print(f"    use_jax_hartree_cached_operator: {route.use_jax_hartree_cached_operator}")
    print(f"    matvec_timing_is_estimated: {route.matvec_timing_is_estimated}")
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
        f"repeated_avg={0.0 if route.repeated_hartree_solve_average_wall_time_seconds is None else route.repeated_hartree_solve_average_wall_time_seconds:.6f}, "
        f"repeated_min={0.0 if route.repeated_hartree_solve_min_wall_time_seconds is None else route.repeated_hartree_solve_min_wall_time_seconds:.6f}, "
        f"repeated_max={0.0 if route.repeated_hartree_solve_max_wall_time_seconds is None else route.repeated_hartree_solve_max_wall_time_seconds:.6f}"
    )
    print(
        "    repeated-solve iters: "
        f"first={route.first_hartree_cg_iterations}, "
        f"avg={route.average_hartree_cg_iterations}, "
        f"repeated_avg={route.repeated_hartree_cg_iteration_average}"
    )
    print(
        "    solve breakdown avg [s]: "
        f"boundary={0.0 if route.average_hartree_boundary_condition_wall_time_seconds is None else route.average_hartree_boundary_condition_wall_time_seconds:.6f}, "
        f"build={0.0 if route.average_hartree_build_wall_time_seconds is None else route.average_hartree_build_wall_time_seconds:.6f}, "
        f"rhs={0.0 if route.average_hartree_rhs_assembly_wall_time_seconds is None else route.average_hartree_rhs_assembly_wall_time_seconds:.6f}, "
        f"cg={0.0 if route.average_hartree_cg_wall_time_seconds is None else route.average_hartree_cg_wall_time_seconds:.6f}, "
        f"cg_other={0.0 if route.average_hartree_cg_other_overhead_wall_time_seconds is None else route.average_hartree_cg_other_overhead_wall_time_seconds:.6f}"
    )
    print(
        "    preconditioner avg [s]: "
        f"count={route.average_hartree_preconditioner_apply_count}, "
        f"apply={route.average_hartree_preconditioner_apply_wall_time_seconds}, "
        f"setup={route.average_hartree_preconditioner_setup_wall_time_seconds}, "
        f"reorder={route.average_hartree_preconditioner_axis_reorder_wall_time_seconds}, "
        f"tridiag={route.average_hartree_preconditioner_tridiagonal_solve_wall_time_seconds}, "
        f"other={route.average_hartree_preconditioner_other_overhead_wall_time_seconds}"
    )
    print(
        "    matvec avg: "
        f"calls={route.average_hartree_matvec_call_count}, "
        f"wall={route.average_hartree_matvec_wall_time_seconds}, "
        f"per_call={route.average_hartree_matvec_wall_time_per_call_seconds}"
    )
    print(
        "    first/repeated matvec: "
        f"first_calls={route.first_hartree_matvec_call_count}, "
        f"first_wall={route.first_hartree_matvec_wall_time_seconds}, "
        f"first_per_call={route.first_hartree_matvec_wall_time_per_call_seconds}, "
        f"repeated_calls_avg={route.repeated_hartree_matvec_call_count_average}, "
        f"repeated_wall_avg={route.repeated_hartree_matvec_average_wall_time_seconds}, "
        f"repeated_per_call={route.repeated_hartree_matvec_wall_time_per_call_seconds}"
    )
    print(
        "    per-iteration avg [s]: "
        f"cg={route.average_hartree_cg_iteration_wall_time_seconds}, "
        f"matvec={route.average_hartree_matvec_wall_time_per_iteration_seconds}, "
        f"other_cg={route.average_hartree_other_cg_overhead_wall_time_per_iteration_seconds}"
    )
    print(
        "    cache stats: "
        f"solve_calls={route.hartree_solve_call_count}, "
        f"cached_use_count={route.hartree_cached_operator_usage_count}, "
        f"first_cached_solve_count={route.hartree_cached_operator_first_solve_count}"
    )


def _print_single_solve(result: H2TripletHartreeSingleSolveResult) -> None:
    print(f"  single solve: {result.path_label}")
    print(
        f"    cg_impl={result.cg_impl}, cg_preconditioner={result.cg_preconditioner}, "
        f"converged={result.converged}, iterations={result.iteration_count}, "
        f"residual={result.residual_max:.6e}"
    )
    print(
        "    timing [s]: "
        f"total={result.total_solve_time_seconds:.6f}, "
        f"cg={result.cg_wall_time_seconds:.6f}, "
        f"matvec={result.matvec_wall_time_seconds:.6f}, "
        f"cg_other={result.cg_other_overhead_wall_time_seconds:.6f}"
    )
    print(
        "    preconditioner [s]: "
        f"count={result.preconditioner_apply_count}, "
        f"apply={result.preconditioner_apply_wall_time_seconds:.6f}, "
        f"setup={result.preconditioner_setup_wall_time_seconds:.6f}, "
        f"reorder={result.preconditioner_axis_reorder_wall_time_seconds:.6f}, "
        f"tridiag={result.preconditioner_tridiagonal_solve_wall_time_seconds:.6f}, "
        f"other={result.preconditioner_other_overhead_wall_time_seconds:.6f}"
    )
    print(
        "    averages [s]: "
        f"per_iter={0.0 if result.average_iteration_wall_time_seconds is None else result.average_iteration_wall_time_seconds:.6f}, "
        f"per_matvec={0.0 if result.average_matvec_wall_time_per_call_seconds is None else result.average_matvec_wall_time_per_call_seconds:.6f}, "
        f"per_prec={0.0 if result.average_preconditioner_apply_wall_time_per_call_seconds is None else result.average_preconditioner_apply_wall_time_per_call_seconds:.6f}, "
        f"matvec_calls={result.matvec_call_count}"
    )
    print(f"    matvec_timing_is_estimated: {result.matvec_timing_is_estimated}")


def print_h2_jax_triplet_hartree_energy_summary(
    result: H2TripletHartreeEnergyAuditResult,
) -> None:
    """Print the compact triplet profiling summary."""

    print("IsoGridDFT H2 triplet JAX Hartree PCG feasibility audit")
    print(f"note: {result.note}")
    print()
    _print_single_solve(result.single_solve_cgloop)
    if result.single_solve_diag is not None:
        _print_single_solve(result.single_solve_diag)
    _print_single_solve(result.single_solve_pcg_stronger)
    print()
    _print_route(result.jax_hartree_cgloop_route)
    _print_route(result.jax_hartree_pcg_stronger_route)
    print()
    print(
        "  total timing delta [s]: "
        f"{result.jax_hartree_pcg_stronger_route.total_wall_time_seconds - result.jax_hartree_cgloop_route.total_wall_time_seconds:+.6f}"
    )
    if result.jax_hartree_cgloop_route.total_wall_time_seconds > 0.0:
        print(
            "  timing ratio (pcg_stronger/cgloop): "
            f"{result.jax_hartree_pcg_stronger_route.total_wall_time_seconds / result.jax_hartree_cgloop_route.total_wall_time_seconds:.6f}"
        )


def main() -> int:
    result = run_h2_jax_triplet_hartree_energy_audit()
    print_h2_jax_triplet_hartree_energy_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
