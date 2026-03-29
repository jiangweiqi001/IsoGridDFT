"""Very small H2 triplet end-to-end micro-profile audit on the JAX mainline."""

from __future__ import annotations

from dataclasses import dataclass

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf import SinglePointEnergyComponents
from isogrid.scf import run_h2_monitor_grid_scf_dry_run

_TRIPLET_MICRO_PROFILE_MAX_ITERATIONS = 20
_TRIPLET_MICRO_PROFILE_MIXING = 0.20
_TRIPLET_MICRO_PROFILE_DENSITY_TOLERANCE = 5.0e-3
_TRIPLET_MICRO_PROFILE_ENERGY_TOLERANCE = 5.0e-5
_TRIPLET_MICRO_PROFILE_EIGENSOLVER_TOLERANCE = 1.0e-3
_TRIPLET_MICRO_PROFILE_EIGENSOLVER_NCV = 20


@dataclass(frozen=True)
class H2JaxTripletMicroProfileParameterSummary:
    """Frozen parameter summary for the triplet micro-profile route."""

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
class H2JaxTripletMicroProfileStep:
    """Per-step timing/profile record for one triplet mainline step."""

    step_index: int
    solver_backend: str
    total_step_wall_time_seconds: float
    eigensolver_wall_time_seconds: float
    static_local_prepare_wall_time_seconds: float
    hartree_solve_wall_time_seconds: float
    energy_eval_wall_time_seconds: float
    density_residual: float
    energy_change_ha: float | None


@dataclass(frozen=True)
class H2JaxTripletEndToEndMicroProfileResult:
    """Compact triplet end-to-end micro-profile result."""

    path_label: str
    spin_state_label: str
    path_type: str
    solver_backend: str
    converged: bool
    completed_full_20_steps: bool
    actual_iteration_count: int
    final_total_energy_ha: float
    final_lowest_eigenvalue_ha: float | None
    final_density_residual: float | None
    final_energy_change_ha: float | None
    total_wall_time_seconds: float
    average_iteration_wall_time_seconds: float | None
    behavior_verdict: str
    dominant_timing_bucket: str
    dominant_timing_bucket_fraction_of_total: float | None
    eigensolver_fraction_of_total: float | None
    step_profiles: tuple[H2JaxTripletMicroProfileStep, ...]
    parameter_summary: H2JaxTripletMicroProfileParameterSummary
    final_energy_components: SinglePointEnergyComponents


def _resolve_solver_backend(result: H2StaticLocalScfDryRunResult) -> str:
    labels = tuple(dict.fromkeys(result.solver_backend_iteration_history))
    filtered = tuple(label for label in labels if label and label != "none")
    if not filtered:
        return "none"
    if len(filtered) == 1:
        return filtered[0]
    return "+".join(filtered)


def _build_behavior_verdict(result: H2StaticLocalScfDryRunResult) -> str:
    if result.converged:
        return "converged"
    residuals = result.density_residual_history
    if len(residuals) < 2:
        return "insufficient_history"
    last = residuals[-1]
    prev = residuals[-2]
    if prev > 0.0 and last / prev > 1.05:
        return "diverging"
    if prev > 0.0 and abs(last / prev - 1.0) <= 0.02:
        return "plateau_or_stall"
    return "stable_not_converged"


def _build_parameter_summary(
    result: H2StaticLocalScfDryRunResult,
) -> H2JaxTripletMicroProfileParameterSummary:
    parameters = result.parameter_summary
    return H2JaxTripletMicroProfileParameterSummary(
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
        eigensolver_ncv=_TRIPLET_MICRO_PROFILE_EIGENSOLVER_NCV,
        max_iterations=_TRIPLET_MICRO_PROFILE_MAX_ITERATIONS,
        mixing=_TRIPLET_MICRO_PROFILE_MIXING,
    )


def _build_step_profiles(
    result: H2StaticLocalScfDryRunResult,
) -> tuple[H2JaxTripletMicroProfileStep, ...]:
    profiles: list[H2JaxTripletMicroProfileStep] = []
    for index, record in enumerate(result.history):
        profiles.append(
            H2JaxTripletMicroProfileStep(
                step_index=index + 1,
                solver_backend=result.solver_backend_iteration_history[index],
                total_step_wall_time_seconds=result.total_step_wall_time_seconds_history[index],
                eigensolver_wall_time_seconds=result.eigensolver_iteration_wall_time_seconds[index],
                static_local_prepare_wall_time_seconds=(
                    result.static_local_prepare_iteration_wall_time_seconds[index]
                ),
                hartree_solve_wall_time_seconds=(
                    result.hartree_solve_iteration_wall_time_seconds[index]
                ),
                energy_eval_wall_time_seconds=(
                    result.energy_evaluation_iteration_wall_time_seconds[index]
                ),
                density_residual=float(record.density_residual),
                energy_change_ha=(
                    None if record.energy_change is None else float(record.energy_change)
                ),
            )
        )
    return tuple(profiles)


def _dominant_timing_bucket(
    result: H2StaticLocalScfDryRunResult,
) -> tuple[str, float | None, float | None]:
    total = float(result.total_wall_time_seconds)
    if total <= 0.0:
        return "none", None, None
    top_level_buckets = {
        "eigensolver": float(result.eigensolver_wall_time_seconds),
        "energy_eval": float(result.energy_evaluation_wall_time_seconds),
        "density_update": float(result.density_update_wall_time_seconds),
        "bookkeeping": float(result.bookkeeping_wall_time_seconds),
    }
    label = max(top_level_buckets, key=top_level_buckets.__getitem__)
    value = top_level_buckets[label]
    return label, float(value / total), float(result.eigensolver_wall_time_seconds / total)


def _lowest_eigenvalue(result: H2StaticLocalScfDryRunResult) -> float | None:
    return None if result.lowest_eigenvalue is None else float(result.lowest_eigenvalue)


def _build_result(
    result: H2StaticLocalScfDryRunResult,
) -> H2JaxTripletEndToEndMicroProfileResult:
    dominant_bucket, dominant_fraction, eigensolver_fraction = _dominant_timing_bucket(result)
    final_energy_change = None
    if result.history and result.history[-1].energy_change is not None:
        final_energy_change = float(result.history[-1].energy_change)
    final_density_residual = None
    if result.history:
        final_density_residual = float(result.history[-1].density_residual)
    return H2JaxTripletEndToEndMicroProfileResult(
        path_label="jax-native-eigensolver-triplet-mainline-micro-profile",
        spin_state_label=result.spin_state_label,
        path_type=result.path_type,
        solver_backend=_resolve_solver_backend(result),
        converged=bool(result.converged),
        completed_full_20_steps=(int(result.iteration_count) >= _TRIPLET_MICRO_PROFILE_MAX_ITERATIONS),
        actual_iteration_count=int(result.iteration_count),
        final_total_energy_ha=float(result.energy.total),
        final_lowest_eigenvalue_ha=_lowest_eigenvalue(result),
        final_density_residual=final_density_residual,
        final_energy_change_ha=final_energy_change,
        total_wall_time_seconds=float(result.total_wall_time_seconds),
        average_iteration_wall_time_seconds=result.average_iteration_wall_time_seconds,
        behavior_verdict=_build_behavior_verdict(result),
        dominant_timing_bucket=dominant_bucket,
        dominant_timing_bucket_fraction_of_total=dominant_fraction,
        eigensolver_fraction_of_total=eigensolver_fraction,
        step_profiles=_build_step_profiles(result),
        parameter_summary=_build_parameter_summary(result),
        final_energy_components=result.energy,
    )


def run_h2_jax_triplet_end_to_end_micro_profile_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2JaxTripletEndToEndMicroProfileResult:
    """Run one triplet end-to-end micro-profile on the frozen JAX mainline."""

    result = run_h2_monitor_grid_scf_dry_run(
        "triplet",
        case=case,
        max_iterations=_TRIPLET_MICRO_PROFILE_MAX_ITERATIONS,
        mixing=_TRIPLET_MICRO_PROFILE_MIXING,
        density_tolerance=_TRIPLET_MICRO_PROFILE_DENSITY_TOLERANCE,
        energy_tolerance=_TRIPLET_MICRO_PROFILE_ENERGY_TOLERANCE,
        eigensolver_tolerance=_TRIPLET_MICRO_PROFILE_EIGENSOLVER_TOLERANCE,
        eigensolver_ncv=_TRIPLET_MICRO_PROFILE_EIGENSOLVER_NCV,
        kinetic_version="trial_fix",
        hartree_backend="jax",
        use_jax_hartree_cached_operator=True,
        jax_hartree_cg_impl="jax_loop",
        jax_hartree_cg_preconditioner="none",
        use_jax_block_kernels=True,
        use_step_local_static_local_reuse=True,
    )
    return _build_result(result)


def print_h2_jax_triplet_end_to_end_micro_profile_summary(
    result: H2JaxTripletEndToEndMicroProfileResult,
) -> None:
    """Print a compact per-step summary for the triplet micro-profile audit."""

    print(f"path: {result.path_label}")
    print(f"  spin: {result.spin_state_label}")
    print(f"  solver backend: {result.solver_backend}")
    print(f"  converged: {result.converged}")
    print(f"  iterations: {result.actual_iteration_count}")
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
    if result.average_iteration_wall_time_seconds is None:
        print("  average iteration wall time [s]: n/a")
    else:
        print(
            "  average iteration wall time [s]: "
            f"{result.average_iteration_wall_time_seconds:.3f}"
        )
    print(f"  behavior verdict: {result.behavior_verdict}")
    print(
        "  dominant top-level timing bucket: "
        f"{result.dominant_timing_bucket}"
    )
    if result.dominant_timing_bucket_fraction_of_total is not None:
        print(
            "  dominant bucket fraction of total: "
            f"{result.dominant_timing_bucket_fraction_of_total:.3f}"
        )
    if result.eigensolver_fraction_of_total is not None:
        print(
            "  eigensolver fraction of total: "
            f"{result.eigensolver_fraction_of_total:.3f}"
        )
    for step in result.step_profiles:
        print(
            f"step={step.step_index} backend={step.solver_backend} "
            f"total={step.total_step_wall_time_seconds:.3f}s "
            f"eig={step.eigensolver_wall_time_seconds:.3f}s "
            f"prepare={step.static_local_prepare_wall_time_seconds:.3f}s "
            f"hartree={step.hartree_solve_wall_time_seconds:.3f}s "
            f"energy={step.energy_eval_wall_time_seconds:.3f}s "
            f"residual={step.density_residual:.12e} "
            f"dE={step.energy_change_ha if step.energy_change_ha is not None else 'n/a'}"
        )


if __name__ == "__main__":
    print_h2_jax_triplet_end_to_end_micro_profile_summary(
        run_h2_jax_triplet_end_to_end_micro_profile_audit()
    )
