"""Very small H2 triplet smoke audit for the JAX-native eigensolver reintegration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf import SinglePointEnergyComponents
from isogrid.scf import run_h2_monitor_grid_scf_dry_run

_TRIPLET_REINTEGRATION_MAX_ITERATIONS = 20
_TRIPLET_REINTEGRATION_MIXING = 0.20
_TRIPLET_REINTEGRATION_DENSITY_TOLERANCE = 5.0e-3
_TRIPLET_REINTEGRATION_ENERGY_TOLERANCE = 5.0e-5
_TRIPLET_REINTEGRATION_EIGENSOLVER_TOLERANCE = 1.0e-3
_TRIPLET_REINTEGRATION_EIGENSOLVER_NCV = 20


@dataclass(frozen=True)
class H2JaxTripletReintegrationSmokeParameterSummary:
    """Frozen parameter summary for the triplet reintegration smoke route."""

    grid_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    weight_scale: float
    radius_scale: float
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    kinetic_version: str
    hartree_backend: str
    use_jax_hartree_cached_operator: bool
    jax_hartree_cg_impl: str
    jax_hartree_cg_preconditioner: str
    use_jax_block_kernels: bool
    use_step_local_static_local_reuse: bool
    eigensolver_ncv: int
    max_iterations: int
    mixing: float


@dataclass(frozen=True)
class H2JaxTripletReintegrationSmokeRouteResult:
    """Compact triplet reintegration smoke summary for one mainline route."""

    path_label: str
    spin_state_label: str
    path_type: str
    solver_backend: str
    timed_out: bool
    smoke_timeout_seconds: float | None
    converged: bool
    iteration_count: int
    final_total_energy_ha: float | None
    final_lowest_eigenvalue_ha: float | None
    final_density_residual: float | None
    final_energy_change_ha: float | None
    total_wall_time_seconds: float
    average_iteration_wall_time_seconds: float | None
    behavior_verdict: str
    earliest_issue_sign: str | None
    parameter_summary: H2JaxTripletReintegrationSmokeParameterSummary
    final_energy_components: SinglePointEnergyComponents


def _resolve_solver_backend(result: H2StaticLocalScfDryRunResult) -> str:
    labels: list[str] = []
    for solve_result in (result.solve_up, result.solve_down):
        if solve_result is None:
            continue
        if solve_result.target_orbitals <= 0:
            continue
        labels.append(str(solve_result.solver_backend))
    if not labels:
        return "none"
    unique_labels = tuple(dict.fromkeys(labels))
    if len(unique_labels) == 1:
        return unique_labels[0]
    return "+".join(unique_labels)


def _tail_energy_changes(result: H2StaticLocalScfDryRunResult) -> tuple[float, ...]:
    tail_changes: list[float] = []
    for record in result.history[-4:]:
        if record.energy_change is not None:
            tail_changes.append(float(record.energy_change))
    return tuple(tail_changes)


def _build_behavior_summary(result: H2StaticLocalScfDryRunResult) -> tuple[str, str | None]:
    if result.converged:
        return "converged", None
    residuals = np.asarray(result.density_residual_history, dtype=np.float64)
    if residuals.size < 2:
        return "insufficient_history", "history shorter than two iterations"
    ratios = residuals[1:] / np.clip(residuals[:-1], 1.0e-16, None)
    growth_indices = np.nonzero(ratios > 1.05)[0]
    if growth_indices.size:
        first_bad = int(growth_indices[0]) + 2
        return "diverging", f"density residual first grew by >5% at iteration {first_bad}"
    tail = ratios[-4:] if ratios.size >= 4 else ratios
    if tail.size and float(np.mean(tail)) >= 0.995 and float(np.std(tail)) <= 0.02:
        return "plateau_or_stall", "tail residual ratios settled near 1"
    return "stable_not_converged", "residual decreased but did not cross tolerance"


def _lowest_eigenvalue(result: H2StaticLocalScfDryRunResult) -> float | None:
    return None if result.lowest_eigenvalue is None else float(result.lowest_eigenvalue)


def _build_parameter_summary(
    result: H2StaticLocalScfDryRunResult,
) -> H2JaxTripletReintegrationSmokeParameterSummary:
    parameters = result.parameter_summary
    return H2JaxTripletReintegrationSmokeParameterSummary(
        grid_shape=parameters.grid_shape,
        box_half_extents_bohr=parameters.box_half_extents_bohr,
        weight_scale=float(parameters.weight_scale),
        radius_scale=float(parameters.radius_scale),
        patch_radius_scale=float(parameters.patch_radius_scale),
        patch_grid_shape=parameters.patch_grid_shape,
        correction_strength=float(parameters.correction_strength),
        interpolation_neighbors=int(parameters.interpolation_neighbors),
        kinetic_version=parameters.kinetic_version,
        hartree_backend=parameters.hartree_backend,
        use_jax_hartree_cached_operator=bool(parameters.use_jax_hartree_cached_operator),
        jax_hartree_cg_impl=parameters.jax_hartree_cg_impl,
        jax_hartree_cg_preconditioner=parameters.jax_hartree_cg_preconditioner,
        use_jax_block_kernels=bool(parameters.use_jax_block_kernels),
        use_step_local_static_local_reuse=bool(parameters.use_step_local_static_local_reuse),
        eigensolver_ncv=_TRIPLET_REINTEGRATION_EIGENSOLVER_NCV,
        max_iterations=_TRIPLET_REINTEGRATION_MAX_ITERATIONS,
        mixing=_TRIPLET_REINTEGRATION_MIXING,
    )


def _build_route_result(
    result: H2StaticLocalScfDryRunResult,
) -> H2JaxTripletReintegrationSmokeRouteResult:
    behavior_verdict, earliest_issue_sign = _build_behavior_summary(result)
    final_energy_change = None
    if result.history and result.history[-1].energy_change is not None:
        final_energy_change = float(result.history[-1].energy_change)
    final_density_residual = None
    if result.history:
        final_density_residual = float(result.history[-1].density_residual)
    return H2JaxTripletReintegrationSmokeRouteResult(
        path_label="jax-native-eigensolver-triplet-mainline",
        spin_state_label=result.spin_state_label,
        path_type=result.path_type,
        solver_backend=_resolve_solver_backend(result),
        timed_out=False,
        smoke_timeout_seconds=None,
        converged=bool(result.converged),
        iteration_count=int(result.iteration_count),
        final_total_energy_ha=float(result.energy.total),
        final_lowest_eigenvalue_ha=_lowest_eigenvalue(result),
        final_density_residual=final_density_residual,
        final_energy_change_ha=final_energy_change,
        total_wall_time_seconds=float(result.total_wall_time_seconds),
        average_iteration_wall_time_seconds=result.average_iteration_wall_time_seconds,
        behavior_verdict=behavior_verdict,
        earliest_issue_sign=earliest_issue_sign,
        parameter_summary=_build_parameter_summary(result),
        final_energy_components=result.energy,
    )


def run_h2_jax_triplet_reintegration_smoke_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2JaxTripletReintegrationSmokeRouteResult:
    """Run one very small triplet dry-run to verify JAX eigensolver reintegration."""

    result = run_h2_monitor_grid_scf_dry_run(
        "triplet",
        case=case,
        max_iterations=_TRIPLET_REINTEGRATION_MAX_ITERATIONS,
        mixing=_TRIPLET_REINTEGRATION_MIXING,
        density_tolerance=_TRIPLET_REINTEGRATION_DENSITY_TOLERANCE,
        energy_tolerance=_TRIPLET_REINTEGRATION_ENERGY_TOLERANCE,
        eigensolver_tolerance=_TRIPLET_REINTEGRATION_EIGENSOLVER_TOLERANCE,
        eigensolver_ncv=_TRIPLET_REINTEGRATION_EIGENSOLVER_NCV,
        kinetic_version="trial_fix",
        hartree_backend="jax",
        use_jax_hartree_cached_operator=True,
        jax_hartree_cg_impl="jax_loop",
        jax_hartree_cg_preconditioner="none",
        use_jax_block_kernels=True,
        use_step_local_static_local_reuse=True,
    )
    return _build_route_result(result)


def print_h2_jax_triplet_reintegration_smoke_summary(
    result: H2JaxTripletReintegrationSmokeRouteResult,
) -> None:
    """Print a compact summary for the triplet reintegration smoke result."""

    print(f"path: {result.path_label}")
    print(f"  spin: {result.spin_state_label}")
    print(f"  solver backend: {result.solver_backend}")
    print(f"  timed out: {result.timed_out}")
    print(f"  converged: {result.converged}")
    print(f"  iterations: {result.iteration_count}")
    if result.final_total_energy_ha is None:
        print("  final total energy [Ha]: n/a")
    else:
        print(f"  final total energy [Ha]: {result.final_total_energy_ha:.12f}")
    if result.final_lowest_eigenvalue_ha is None:
        print("  final lowest eigenvalue [Ha]: n/a")
    else:
        print(f"  final lowest eigenvalue [Ha]: {result.final_lowest_eigenvalue_ha:.12f}")
    if result.final_density_residual is None:
        print("  final density residual: n/a")
    else:
        print(f"  final density residual: {result.final_density_residual:.12e}")
    if result.final_energy_change_ha is None:
        print("  final energy change [Ha]: n/a")
    else:
        print(f"  final energy change [Ha]: {result.final_energy_change_ha:.12e}")
    print(f"  total wall time [s]: {result.total_wall_time_seconds:.3f}")
    print(
        "  average iteration wall time [s]: "
        f"{result.average_iteration_wall_time_seconds:.3f}"
        if result.average_iteration_wall_time_seconds is not None
        else "  average iteration wall time [s]: n/a"
    )
    print(f"  behavior verdict: {result.behavior_verdict}")
    print(
        "  earliest issue sign: "
        f"{result.earliest_issue_sign}" if result.earliest_issue_sign is not None else "  earliest issue sign: none"
    )


if __name__ == "__main__":
    print_h2_jax_triplet_reintegration_smoke_summary(
        run_h2_jax_triplet_reintegration_smoke_audit()
    )
