"""Very rough old-vs-JAX SCF hot-path audit for the H2 monitor-grid dry-run.

This audit keeps the A-grid dry-run Hamiltonian restricted to

    T + V_loc,ion + V_H + V_xc

with the repaired monitor-grid Hartree/Poisson path, patch-assisted local ionic
slice, and the kinetic trial-fix branch. Nonlocal remains excluded here, so the
timing/profiling results only measure the current local-only monitor-grid SCF
dry-run line.
"""

from __future__ import annotations

from dataclasses import dataclass

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.ops import integrate_field
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf import SinglePointEnergyComponents
from isogrid.scf import run_h2_monitor_grid_scf_dry_run

_TRIPLET_PROFILING_MAX_ITERATIONS = 20
_SINGLET_REFERENCE_MAX_ITERATIONS = 10
_A_GRID_DRY_RUN_MIXING = 0.20
_A_GRID_DRY_RUN_DENSITY_TOLERANCE = 5.0e-3
_A_GRID_DRY_RUN_ENERGY_TOLERANCE = 5.0e-5
_A_GRID_DRY_RUN_EIGENSOLVER_TOLERANCE = 1.0e-3
_A_GRID_DRY_RUN_EIGENSOLVER_NCV = 20


@dataclass(frozen=True)
class H2JaxScfTimingBreakdown:
    """Very rough aggregated timing breakdown for one SCF dry-run route."""

    eigensolver_wall_time_seconds: float
    energy_evaluation_wall_time_seconds: float
    density_update_wall_time_seconds: float
    bookkeeping_wall_time_seconds: float


@dataclass(frozen=True)
class H2JaxScfHotpathRouteResult:
    """Compact SCF hot-path profiling summary for one route and one spin state."""

    path_type: str
    spin_state_label: str
    kinetic_version: str
    use_jax_block_kernels: bool
    converged: bool
    iteration_count: int
    final_total_energy_ha: float
    lowest_eigenvalue_ha: float | None
    energy_history_ha: tuple[float, ...]
    density_residual_history: tuple[float, ...]
    energy_change_history_ha: tuple[float | None, ...]
    final_density_residual: float | None
    final_energy_change_ha: float | None
    final_rho_up_electrons: float
    final_rho_down_electrons: float
    total_wall_time_seconds: float
    average_iteration_wall_time_seconds: float | None
    timing_breakdown: H2JaxScfTimingBreakdown
    parameter_summary: str
    final_energy_components: SinglePointEnergyComponents


@dataclass(frozen=True)
class H2JaxScfHotpathAuditResult:
    """Top-level H2 SCF hot-path audit for old and JAX monitor-grid routes."""

    triplet_old: H2JaxScfHotpathRouteResult
    triplet_jax: H2JaxScfHotpathRouteResult
    singlet_old: H2JaxScfHotpathRouteResult | None
    singlet_jax: H2JaxScfHotpathRouteResult | None
    note: str


def _monitor_parameter_summary(result: H2StaticLocalScfDryRunResult) -> str:
    parameters = result.parameter_summary
    return (
        "A-grid+patch+trial-fix SCF hot-path audit: "
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
        f"mixing={_A_GRID_DRY_RUN_MIXING:.2f}, "
        f"density_tol={_A_GRID_DRY_RUN_DENSITY_TOLERANCE:.1e}, "
        f"energy_tol={_A_GRID_DRY_RUN_ENERGY_TOLERANCE:.1e}, "
        f"eig_tol={_A_GRID_DRY_RUN_EIGENSOLVER_TOLERANCE:.1e}, "
        f"ncv={_A_GRID_DRY_RUN_EIGENSOLVER_NCV}"
    )


def _build_route_result(
    result: H2StaticLocalScfDryRunResult,
) -> H2JaxScfHotpathRouteResult:
    grid_geometry = (
        result.solve_up.operator_context.grid_geometry
        if result.solve_up is not None
        else result.solve_down.operator_context.grid_geometry
    )
    return H2JaxScfHotpathRouteResult(
        path_type=result.path_type,
        spin_state_label=result.spin_state_label,
        kinetic_version=result.kinetic_version,
        use_jax_block_kernels=bool(result.use_jax_block_kernels),
        converged=bool(result.converged),
        iteration_count=int(result.iteration_count),
        final_total_energy_ha=float(result.energy.total),
        lowest_eigenvalue_ha=result.lowest_eigenvalue,
        energy_history_ha=tuple(float(value) for value in result.energy_history),
        density_residual_history=tuple(float(value) for value in result.density_residual_history),
        energy_change_history_ha=tuple(
            None if record.energy_change is None else float(record.energy_change)
            for record in result.history
        ),
        final_density_residual=(
            None if not result.history else float(result.history[-1].density_residual)
        ),
        final_energy_change_ha=(
            None
            if not result.history or result.history[-1].energy_change is None
            else float(result.history[-1].energy_change)
        ),
        final_rho_up_electrons=float(integrate_field(result.rho_up, grid_geometry=grid_geometry)),
        final_rho_down_electrons=float(integrate_field(result.rho_down, grid_geometry=grid_geometry)),
        total_wall_time_seconds=float(result.total_wall_time_seconds),
        average_iteration_wall_time_seconds=result.average_iteration_wall_time_seconds,
        timing_breakdown=H2JaxScfTimingBreakdown(
            eigensolver_wall_time_seconds=float(result.eigensolver_wall_time_seconds),
            energy_evaluation_wall_time_seconds=float(result.energy_evaluation_wall_time_seconds),
            density_update_wall_time_seconds=float(result.density_update_wall_time_seconds),
            bookkeeping_wall_time_seconds=float(result.bookkeeping_wall_time_seconds),
        ),
        parameter_summary=_monitor_parameter_summary(result),
        final_energy_components=result.energy,
    )


def _run_route(
    spin_label: str,
    *,
    case: BenchmarkCase,
    use_jax_block_kernels: bool,
    max_iterations: int,
) -> H2JaxScfHotpathRouteResult:
    return _build_route_result(
        run_h2_monitor_grid_scf_dry_run(
            spin_label,
            case=case,
            max_iterations=max_iterations,
            mixing=_A_GRID_DRY_RUN_MIXING,
            density_tolerance=_A_GRID_DRY_RUN_DENSITY_TOLERANCE,
            energy_tolerance=_A_GRID_DRY_RUN_ENERGY_TOLERANCE,
            eigensolver_tolerance=_A_GRID_DRY_RUN_EIGENSOLVER_TOLERANCE,
            eigensolver_ncv=_A_GRID_DRY_RUN_EIGENSOLVER_NCV,
            kinetic_version="trial_fix",
            use_jax_block_kernels=use_jax_block_kernels,
        )
    )


def run_h2_jax_scf_hotpath_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    *,
    include_singlet_reference: bool = True,
) -> H2JaxScfHotpathAuditResult:
    """Run the very rough H2 SCF hot-path audit on the monitor-grid route."""

    triplet_old = _run_route(
        "triplet",
        case=case,
        use_jax_block_kernels=False,
        max_iterations=_TRIPLET_PROFILING_MAX_ITERATIONS,
    )
    triplet_jax = _run_route(
        "triplet",
        case=case,
        use_jax_block_kernels=True,
        max_iterations=_TRIPLET_PROFILING_MAX_ITERATIONS,
    )

    singlet_old = None
    singlet_jax = None
    if include_singlet_reference:
        singlet_old = _run_route(
            "singlet",
            case=case,
            use_jax_block_kernels=False,
            max_iterations=_SINGLET_REFERENCE_MAX_ITERATIONS,
        )
        singlet_jax = _run_route(
            "singlet",
            case=case,
            use_jax_block_kernels=True,
            max_iterations=_SINGLET_REFERENCE_MAX_ITERATIONS,
        )

    return H2JaxScfHotpathAuditResult(
        triplet_old=triplet_old,
        triplet_jax=triplet_jax,
        singlet_old=singlet_old,
        singlet_jax=singlet_jax,
        note=(
            "Very rough H2 monitor-grid SCF hot-path audit after wiring the already-correct JAX "
            "block kernels into the A-grid dry-run loop. The triplet route is the main profiling "
            "case because it already converges on the local static chain; singlet is only an "
            "auxiliary 10-step reference. The A-grid Hamiltonian still contains only "
            "T + V_loc,ion + V_H + V_xc, without nonlocal."
        ),
    )


def _print_route(route: H2JaxScfHotpathRouteResult) -> None:
    print(
        f"  route: {'jax-hotpath' if route.use_jax_block_kernels else 'old-hotpath'} "
        f"({route.spin_state_label})"
    )
    print(f"    converged: {route.converged}")
    print(f"    iterations: {route.iteration_count}")
    print(f"    final total energy [Ha]: {route.final_total_energy_ha:.12f}")
    if route.lowest_eigenvalue_ha is not None:
        print(f"    final lowest eigenvalue [Ha]: {route.lowest_eigenvalue_ha:.12f}")
    else:
        print("    final lowest eigenvalue [Ha]: n/a")
    print(
        "    final electrons: "
        f"rho_up={route.final_rho_up_electrons:.12f}, "
        f"rho_down={route.final_rho_down_electrons:.12f}"
    )
    print(
        "    timing [s]: "
        f"total={route.total_wall_time_seconds:.6f}, "
        f"avg/iter={(route.average_iteration_wall_time_seconds or 0.0):.6f}, "
        f"eigensolver={route.timing_breakdown.eigensolver_wall_time_seconds:.6f}, "
        f"energy_eval={route.timing_breakdown.energy_evaluation_wall_time_seconds:.6f}, "
        f"density_update={route.timing_breakdown.density_update_wall_time_seconds:.6f}, "
        f"bookkeeping={route.timing_breakdown.bookkeeping_wall_time_seconds:.6f}"
    )


def print_h2_jax_scf_hotpath_summary(
    result: H2JaxScfHotpathAuditResult,
) -> None:
    """Print the compact H2 SCF hot-path profiling summary."""

    print("IsoGridDFT H2 JAX SCF hot-path audit")
    print(f"note: {result.note}")
    print()
    print("triplet main profiling case:")
    _print_route(result.triplet_old)
    _print_route(result.triplet_jax)
    print(
        "  triplet timing delta [s]: "
        f"{result.triplet_jax.total_wall_time_seconds - result.triplet_old.total_wall_time_seconds:+.6f}"
    )
    if result.triplet_old.total_wall_time_seconds > 0.0:
        print(
            "  triplet timing ratio (jax/old): "
            f"{result.triplet_jax.total_wall_time_seconds / result.triplet_old.total_wall_time_seconds:.6f}"
        )
    if result.singlet_old is not None and result.singlet_jax is not None:
        print()
        print("singlet auxiliary reference:")
        _print_route(result.singlet_old)
        _print_route(result.singlet_jax)


def main() -> int:
    result = run_h2_jax_scf_hotpath_audit()
    print_h2_jax_scf_hotpath_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
