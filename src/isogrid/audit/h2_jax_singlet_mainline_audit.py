"""Productionish singlet-only Anderson audit on the frozen JAX A-grid mainline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.audit.baselines import H2_JAX_SINGLET_MAINLINE_BASELINE
from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf import SinglePointEnergyComponents
from isogrid.scf import run_h2_monitor_grid_scf_dry_run

_SINGLET_MAINLINE_MAX_ITERATIONS = 20
_SINGLET_MAINLINE_SUPPLEMENTAL_MAX_ITERATIONS = 40
_SINGLET_MAINLINE_BASELINE_MIXING = 0.10
_SINGLET_MAINLINE_DENSITY_TOLERANCE = 5.0e-3
_SINGLET_MAINLINE_ENERGY_TOLERANCE = 5.0e-5
_SINGLET_MAINLINE_EIGENSOLVER_TOLERANCE = 1.0e-3
_SINGLET_MAINLINE_EIGENSOLVER_NCV = 20
_SINGLET_MAINLINE_DIIS_WARMUP = 3
_SINGLET_MAINLINE_DIIS_HISTORY = 4
_SINGLET_MAINLINE_ANDERSON_WARMUP = 3
_SINGLET_MAINLINE_ANDERSON_HISTORY = 4
_SINGLET_MAINLINE_ANDERSON_REGULARIZATION = 1.0e-8
_SINGLET_MAINLINE_ANDERSON_DAMPING = 0.5
_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_HISTORY = 6
_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_REGULARIZATION = 1.0e-8
_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_DAMPING = 0.55
_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_STEP_CLIP = 1.0
_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_RESET_ON_GROWTH = True
_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_RESET_GROWTH_FACTOR = 1.05
_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_ADAPTIVE_DAMPING = True
_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_MIN_DAMPING = 0.35
_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_MAX_DAMPING = 0.75
_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_ACCEPTANCE_RATIO = 1.02
_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_COLLINEARITY = 0.995
_TAIL_SUMMARY_LENGTH = 5


@dataclass(frozen=True)
class H2JaxSingletMainlineTimingBreakdown:
    """Very rough timing buckets for one frozen singlet mainline route."""

    eigensolver_wall_time_seconds: float
    static_local_prepare_wall_time_seconds: float
    hartree_solve_wall_time_seconds: float
    energy_evaluation_wall_time_seconds: float
    density_update_wall_time_seconds: float
    bookkeeping_wall_time_seconds: float


@dataclass(frozen=True)
class H2JaxSingletMainlineBehavior:
    """Very small convergence-behavior summary for one singlet route."""

    detected_two_cycle: bool
    tail_length: int
    even_odd_energy_gap_ha: float | None
    even_odd_residual_gap: float | None
    verdict: str
    tail_energy_history_ha: tuple[float, ...]
    tail_density_residual_history: tuple[float, ...]
    tail_energy_change_history_ha: tuple[float | None, ...]
    tail_residual_ratios: tuple[float, ...]
    average_tail_residual_ratio: float | None
    tail_residual_ratio_std: float | None
    entered_plateau: bool


@dataclass(frozen=True)
class H2JaxSingletMainlineParameterSummary:
    """Frozen parameter summary for one singlet mainline route."""

    grid_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    weight_scale: float
    radius_scale: float
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    kinetic_version: str
    mixing: float
    mixer: str
    solver_variant: str
    max_iterations: int
    density_tolerance: float
    energy_tolerance: float
    eigensolver_tolerance: float
    eigensolver_ncv: int
    hartree_backend: str
    use_jax_hartree_cached_operator: bool
    jax_hartree_cg_impl: str
    jax_hartree_cg_preconditioner: str
    jax_hartree_line_preconditioner_impl: str
    use_jax_block_kernels: bool
    use_step_local_static_local_reuse: bool
    diis_enabled: bool
    diis_warmup_iterations: int
    diis_history_length: int
    diis_residual_definition: str
    anderson_enabled: bool
    anderson_warmup_iterations: int
    anderson_history_length: int
    anderson_regularization: float
    anderson_damping: float
    anderson_step_clip_factor: float | None
    anderson_reset_on_growth: bool
    anderson_reset_growth_factor: float
    anderson_adaptive_damping_enabled: bool
    anderson_min_damping: float
    anderson_max_damping: float
    anderson_acceptance_residual_ratio_threshold: float
    anderson_collinearity_cosine_threshold: float
    anderson_residual_definition: str


@dataclass(frozen=True)
class H2JaxSingletMainlineRouteResult:
    """Compact audit result for one frozen singlet mainline route."""

    path_label: str
    spin_state_label: str
    path_type: str
    kinetic_version: str
    includes_nonlocal: bool
    max_iterations: int
    mixing: float
    mixer: str
    solver_variant: str
    formal_mixer_history_length: int | None
    formal_mixer_regularization: float | None
    formal_mixer_damping: float | None
    formal_mixer_step_clip_factor: float | None
    formal_mixer_reset_on_growth: bool
    formal_mixer_reset_growth_factor: float | None
    formal_mixer_adaptive_damping_enabled: bool
    formal_mixer_min_damping: float | None
    formal_mixer_max_damping: float | None
    formal_mixer_acceptance_residual_ratio_threshold: float | None
    formal_mixer_collinearity_cosine_threshold: float | None
    converged: bool
    iteration_count: int
    final_total_energy_ha: float
    final_lowest_eigenvalue_ha: float | None
    final_density_residual: float | None
    final_energy_change_ha: float | None
    total_wall_time_seconds: float
    average_iteration_wall_time_seconds: float | None
    parameter_summary: H2JaxSingletMainlineParameterSummary
    timing_breakdown: H2JaxSingletMainlineTimingBreakdown
    behavior: H2JaxSingletMainlineBehavior
    diis_used_iterations: tuple[int, ...]
    diis_fallback_iterations: tuple[int, ...]
    anderson_used_iterations: tuple[int, ...]
    anderson_fallback_iterations: tuple[int, ...]
    anderson_rejected_iterations: tuple[int, ...]
    anderson_reset_iterations: tuple[int, ...]
    anderson_filtered_history_sizes: tuple[int, ...]
    anderson_effective_damping_history: tuple[float, ...]
    anderson_projected_residual_ratio_history: tuple[float | None, ...]
    final_energy_components: SinglePointEnergyComponents
    note: str


@dataclass(frozen=True)
class H2JaxSingletMainlineAuditResult:
    """Productionish singlet Anderson audit for the frozen JAX A-grid mainline."""

    path_label: str
    spin_state_label: str
    path_type: str
    baseline_linear_route: H2JaxSingletMainlineRouteResult
    diis_route: H2JaxSingletMainlineRouteResult
    anderson_baseline_route: H2JaxSingletMainlineRouteResult
    anderson_productionish_route: H2JaxSingletMainlineRouteResult
    supplemental_anderson_route: H2JaxSingletMainlineRouteResult
    diagnosis: str
    note: str


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.std(values))


def _tail_energy_change_history(
    result: H2StaticLocalScfDryRunResult,
    *,
    length: int = _TAIL_SUMMARY_LENGTH,
) -> tuple[float | None, ...]:
    tail = result.history[-length:]
    return tuple(
        None if item.energy_change is None else float(item.energy_change)
        for item in tail
    )


def _tail_residual_ratios(
    tail_density_residual_history: tuple[float, ...],
) -> tuple[float, ...]:
    ratios: list[float] = []
    for previous, current in zip(
        tail_density_residual_history[:-1],
        tail_density_residual_history[1:],
        strict=True,
    ):
        if abs(previous) <= 1.0e-16:
            continue
        ratios.append(float(current / previous))
    return tuple(ratios)


def _build_behavior(
    result: H2StaticLocalScfDryRunResult,
    *,
    converged: bool,
) -> H2JaxSingletMainlineBehavior:
    energy_history_ha = tuple(float(value) for value in result.energy_history)
    density_residual_history = tuple(float(value) for value in result.density_residual_history)
    tail_energy_history_ha = tuple(energy_history_ha[-_TAIL_SUMMARY_LENGTH:])
    tail_density_residual_history = tuple(density_residual_history[-_TAIL_SUMMARY_LENGTH:])
    tail_energy_change_history_ha = _tail_energy_change_history(result)
    residual_ratios = _tail_residual_ratios(tail_density_residual_history)
    ratio_array = np.asarray(residual_ratios, dtype=np.float64)
    average_tail_residual_ratio = _safe_mean(ratio_array)
    tail_residual_ratio_std = _safe_std(ratio_array)
    entered_plateau = bool(
        average_tail_residual_ratio is not None
        and tail_residual_ratio_std is not None
        and 0.985 <= average_tail_residual_ratio <= 1.015
        and tail_residual_ratio_std < 0.05
    )

    if converged or len(energy_history_ha) < 6 or len(density_residual_history) < 6:
        return H2JaxSingletMainlineBehavior(
            detected_two_cycle=False,
            tail_length=0,
            even_odd_energy_gap_ha=None,
            even_odd_residual_gap=None,
            verdict="converged" if converged else "insufficient_tail",
            tail_energy_history_ha=tail_energy_history_ha,
            tail_density_residual_history=tail_density_residual_history,
            tail_energy_change_history_ha=tail_energy_change_history_ha,
            tail_residual_ratios=residual_ratios,
            average_tail_residual_ratio=average_tail_residual_ratio,
            tail_residual_ratio_std=tail_residual_ratio_std,
            entered_plateau=entered_plateau,
        )

    tail_start = len(energy_history_ha) // 2
    tail_energies = np.asarray(energy_history_ha[tail_start:], dtype=np.float64)
    tail_residuals = np.asarray(density_residual_history[tail_start:], dtype=np.float64)
    if tail_energies.size % 2 != 0:
        tail_energies = tail_energies[1:]
        tail_residuals = tail_residuals[1:]

    even_energies = tail_energies[::2]
    odd_energies = tail_energies[1::2]
    even_residuals = tail_residuals[::2]
    odd_residuals = tail_residuals[1::2]

    even_energy_mean = _safe_mean(even_energies)
    odd_energy_mean = _safe_mean(odd_energies)
    even_residual_mean = _safe_mean(even_residuals)
    odd_residual_mean = _safe_mean(odd_residuals)
    energy_gap = (
        None
        if even_energy_mean is None or odd_energy_mean is None
        else float(abs(even_energy_mean - odd_energy_mean))
    )
    residual_gap = (
        None
        if even_residual_mean is None or odd_residual_mean is None
        else float(abs(even_residual_mean - odd_residual_mean))
    )
    energy_noise_floor = max(_safe_std(even_energies) or 0.0, _safe_std(odd_energies) or 0.0, 1.0e-16)
    residual_noise_floor = max(
        _safe_std(even_residuals) or 0.0,
        _safe_std(odd_residuals) or 0.0,
        1.0e-16,
    )
    detected_two_cycle = bool(
        energy_gap is not None
        and residual_gap is not None
        and energy_gap > 5.0 * energy_noise_floor
        and residual_gap > 5.0 * residual_noise_floor
    )
    if detected_two_cycle:
        verdict = "weak_two_cycle"
    elif entered_plateau:
        verdict = "plateau_or_stall"
    elif abs(tail_residuals[-1] - tail_residuals[0]) < 5.0e-3:
        verdict = "stable_not_converged"
    elif tail_residuals[-1] > tail_residuals[0]:
        verdict = "diverging"
    else:
        verdict = "slow_monotone_or_damped"
    return H2JaxSingletMainlineBehavior(
        detected_two_cycle=detected_two_cycle,
        tail_length=int(tail_energies.size),
        even_odd_energy_gap_ha=energy_gap,
        even_odd_residual_gap=residual_gap,
        verdict=verdict,
        tail_energy_history_ha=tail_energy_history_ha,
        tail_density_residual_history=tail_density_residual_history,
        tail_energy_change_history_ha=tail_energy_change_history_ha,
        tail_residual_ratios=residual_ratios,
        average_tail_residual_ratio=average_tail_residual_ratio,
        tail_residual_ratio_std=tail_residual_ratio_std,
        entered_plateau=entered_plateau,
    )


def _build_parameter_summary(
    result: H2StaticLocalScfDryRunResult,
    *,
    mixing: float,
    mixer: str,
    solver_variant: str,
    max_iterations: int,
) -> H2JaxSingletMainlineParameterSummary:
    parameters = result.parameter_summary
    return H2JaxSingletMainlineParameterSummary(
        grid_shape=parameters.grid_shape,
        box_half_extents_bohr=parameters.box_half_extents_bohr,
        weight_scale=parameters.weight_scale,
        radius_scale=parameters.radius_scale,
        patch_radius_scale=parameters.patch_radius_scale,
        patch_grid_shape=parameters.patch_grid_shape,
        correction_strength=parameters.correction_strength,
        interpolation_neighbors=parameters.interpolation_neighbors,
        kinetic_version=parameters.kinetic_version,
        mixing=mixing,
        mixer=mixer,
        solver_variant=solver_variant,
        max_iterations=max_iterations,
        density_tolerance=_SINGLET_MAINLINE_DENSITY_TOLERANCE,
        energy_tolerance=_SINGLET_MAINLINE_ENERGY_TOLERANCE,
        eigensolver_tolerance=_SINGLET_MAINLINE_EIGENSOLVER_TOLERANCE,
        eigensolver_ncv=_SINGLET_MAINLINE_EIGENSOLVER_NCV,
        hartree_backend=parameters.hartree_backend,
        use_jax_hartree_cached_operator=parameters.use_jax_hartree_cached_operator,
        jax_hartree_cg_impl=parameters.jax_hartree_cg_impl,
        jax_hartree_cg_preconditioner=parameters.jax_hartree_cg_preconditioner,
        jax_hartree_line_preconditioner_impl=parameters.jax_hartree_line_preconditioner_impl,
        use_jax_block_kernels=parameters.use_jax_block_kernels,
        use_step_local_static_local_reuse=parameters.use_step_local_static_local_reuse,
        diis_enabled=parameters.diis_enabled,
        diis_warmup_iterations=parameters.diis_warmup_iterations,
        diis_history_length=parameters.diis_history_length,
        diis_residual_definition=parameters.diis_residual_definition,
        anderson_enabled=parameters.anderson_enabled,
        anderson_warmup_iterations=parameters.anderson_warmup_iterations,
        anderson_history_length=parameters.anderson_history_length,
        anderson_regularization=parameters.anderson_regularization,
        anderson_damping=parameters.anderson_damping,
        anderson_step_clip_factor=parameters.anderson_step_clip_factor,
        anderson_reset_on_growth=parameters.anderson_reset_on_growth,
        anderson_reset_growth_factor=parameters.anderson_reset_growth_factor,
        anderson_adaptive_damping_enabled=parameters.anderson_adaptive_damping_enabled,
        anderson_min_damping=parameters.anderson_min_damping,
        anderson_max_damping=parameters.anderson_max_damping,
        anderson_acceptance_residual_ratio_threshold=(
            parameters.anderson_acceptance_residual_ratio_threshold
        ),
        anderson_collinearity_cosine_threshold=parameters.anderson_collinearity_cosine_threshold,
        anderson_residual_definition=parameters.anderson_residual_definition,
    )


def _build_route_result(
    result: H2StaticLocalScfDryRunResult,
    *,
    mixing: float,
    mixer: str,
    solver_variant: str,
    max_iterations: int,
    formal_mixer_history_length: int | None,
    formal_mixer_regularization: float | None,
    formal_mixer_damping: float | None,
    formal_mixer_step_clip_factor: float | None,
    formal_mixer_reset_on_growth: bool,
    formal_mixer_reset_growth_factor: float | None,
    formal_mixer_adaptive_damping_enabled: bool,
    formal_mixer_min_damping: float | None,
    formal_mixer_max_damping: float | None,
    formal_mixer_acceptance_residual_ratio_threshold: float | None,
    formal_mixer_collinearity_cosine_threshold: float | None,
) -> H2JaxSingletMainlineRouteResult:
    return H2JaxSingletMainlineRouteResult(
        path_label=f"jax-singlet-mainline-{solver_variant}",
        spin_state_label=result.spin_state_label,
        path_type=result.path_type,
        kinetic_version=result.kinetic_version,
        includes_nonlocal=result.includes_nonlocal,
        max_iterations=max_iterations,
        mixing=mixing,
        mixer=mixer,
        solver_variant=solver_variant,
        formal_mixer_history_length=formal_mixer_history_length,
        formal_mixer_regularization=formal_mixer_regularization,
        formal_mixer_damping=formal_mixer_damping,
        formal_mixer_step_clip_factor=formal_mixer_step_clip_factor,
        formal_mixer_reset_on_growth=formal_mixer_reset_on_growth,
        formal_mixer_reset_growth_factor=formal_mixer_reset_growth_factor,
        formal_mixer_adaptive_damping_enabled=formal_mixer_adaptive_damping_enabled,
        formal_mixer_min_damping=formal_mixer_min_damping,
        formal_mixer_max_damping=formal_mixer_max_damping,
        formal_mixer_acceptance_residual_ratio_threshold=formal_mixer_acceptance_residual_ratio_threshold,
        formal_mixer_collinearity_cosine_threshold=formal_mixer_collinearity_cosine_threshold,
        converged=result.converged,
        iteration_count=result.iteration_count,
        final_total_energy_ha=float(result.energy.total),
        final_lowest_eigenvalue_ha=result.lowest_eigenvalue,
        final_density_residual=(
            None if not result.history else float(result.history[-1].density_residual)
        ),
        final_energy_change_ha=(
            None
            if not result.history or result.history[-1].energy_change is None
            else float(result.history[-1].energy_change)
        ),
        total_wall_time_seconds=float(result.total_wall_time_seconds),
        average_iteration_wall_time_seconds=result.average_iteration_wall_time_seconds,
        parameter_summary=_build_parameter_summary(
            result,
            mixing=mixing,
            mixer=mixer,
            solver_variant=solver_variant,
            max_iterations=max_iterations,
        ),
        timing_breakdown=H2JaxSingletMainlineTimingBreakdown(
            eigensolver_wall_time_seconds=float(result.eigensolver_wall_time_seconds),
            static_local_prepare_wall_time_seconds=float(result.static_local_prepare_wall_time_seconds),
            hartree_solve_wall_time_seconds=float(result.hartree_solve_wall_time_seconds),
            energy_evaluation_wall_time_seconds=float(result.energy_evaluation_wall_time_seconds),
            density_update_wall_time_seconds=float(result.density_update_wall_time_seconds),
            bookkeeping_wall_time_seconds=float(result.bookkeeping_wall_time_seconds),
        ),
        behavior=_build_behavior(result, converged=bool(result.converged)),
        diis_used_iterations=tuple(int(value) for value in result.diis_used_iterations),
        diis_fallback_iterations=tuple(int(value) for value in result.diis_fallback_iterations),
        anderson_used_iterations=tuple(int(value) for value in result.anderson_used_iterations),
        anderson_fallback_iterations=tuple(int(value) for value in result.anderson_fallback_iterations),
        anderson_rejected_iterations=tuple(int(value) for value in result.anderson_rejected_iterations),
        anderson_reset_iterations=tuple(int(value) for value in result.anderson_reset_iterations),
        anderson_filtered_history_sizes=tuple(
            int(value) for value in result.anderson_filtered_history_sizes
        ),
        anderson_effective_damping_history=tuple(
            float(value) for value in result.anderson_effective_damping_history
        ),
        anderson_projected_residual_ratio_history=tuple(
            None if value is None else float(value)
            for value in result.anderson_projected_residual_ratio_history
        ),
        final_energy_components=result.energy,
        note=(
            "Frozen A-grid local-only mainline: use_jax_block_kernels=True, "
            "use_step_local_static_local_reuse=True, hartree_backend='jax', "
            "use_jax_hartree_cached_operator=True, cg_impl='jax_loop', "
            "cg_preconditioner='none'. Nonlocal remains absent."
        ),
    )


def _run_route(
    *,
    case: BenchmarkCase,
    max_iterations: int,
    mixing: float,
    mixer: str,
    solver_variant: str,
    enable_diis: bool = False,
    diis_warmup_iterations: int = _SINGLET_MAINLINE_DIIS_WARMUP,
    diis_history_length: int = _SINGLET_MAINLINE_DIIS_HISTORY,
    enable_anderson: bool = False,
    anderson_warmup_iterations: int = _SINGLET_MAINLINE_ANDERSON_WARMUP,
    anderson_history_length: int = _SINGLET_MAINLINE_ANDERSON_HISTORY,
    anderson_regularization: float = _SINGLET_MAINLINE_ANDERSON_REGULARIZATION,
    anderson_damping: float = _SINGLET_MAINLINE_ANDERSON_DAMPING,
    anderson_step_clip_factor: float | None = None,
    anderson_reset_on_growth: bool = False,
    anderson_reset_growth_factor: float = 1.10,
    anderson_adaptive_damping_enabled: bool = False,
    anderson_min_damping: float = 0.35,
    anderson_max_damping: float = 0.75,
    anderson_acceptance_residual_ratio_threshold: float = 1.02,
    anderson_collinearity_cosine_threshold: float = 0.995,
) -> H2JaxSingletMainlineRouteResult:
    result = run_h2_monitor_grid_scf_dry_run(
        "singlet",
        case=case,
        max_iterations=max_iterations,
        mixing=mixing,
        density_tolerance=_SINGLET_MAINLINE_DENSITY_TOLERANCE,
        energy_tolerance=_SINGLET_MAINLINE_ENERGY_TOLERANCE,
        eigensolver_tolerance=_SINGLET_MAINLINE_EIGENSOLVER_TOLERANCE,
        eigensolver_ncv=_SINGLET_MAINLINE_EIGENSOLVER_NCV,
        kinetic_version="trial_fix",
        hartree_backend="jax",
        use_jax_hartree_cached_operator=True,
        jax_hartree_cg_impl="jax_loop",
        jax_hartree_cg_preconditioner="none",
        jax_hartree_line_preconditioner_impl="baseline",
        use_jax_block_kernels=True,
        use_step_local_static_local_reuse=True,
        enable_diis=enable_diis,
        diis_warmup_iterations=diis_warmup_iterations,
        diis_history_length=diis_history_length,
        enable_anderson=enable_anderson,
        anderson_warmup_iterations=anderson_warmup_iterations,
        anderson_history_length=anderson_history_length,
        anderson_regularization=anderson_regularization,
        anderson_damping=anderson_damping,
        anderson_step_clip_factor=anderson_step_clip_factor,
        anderson_reset_on_growth=anderson_reset_on_growth,
        anderson_reset_growth_factor=anderson_reset_growth_factor,
        anderson_adaptive_damping_enabled=anderson_adaptive_damping_enabled,
        anderson_min_damping=anderson_min_damping,
        anderson_max_damping=anderson_max_damping,
        anderson_acceptance_residual_ratio_threshold=anderson_acceptance_residual_ratio_threshold,
        anderson_collinearity_cosine_threshold=anderson_collinearity_cosine_threshold,
    )
    formal_mixer_history_length = None
    formal_mixer_regularization = None
    formal_mixer_damping = None
    formal_mixer_step_clip_factor = None
    formal_mixer_reset_on_growth = False
    formal_mixer_reset_growth_factor = None
    formal_mixer_adaptive_damping_enabled = False
    formal_mixer_min_damping = None
    formal_mixer_max_damping = None
    formal_mixer_acceptance_residual_ratio_threshold = None
    formal_mixer_collinearity_cosine_threshold = None
    if mixer == "anderson":
        formal_mixer_history_length = int(anderson_history_length)
        formal_mixer_regularization = float(anderson_regularization)
        formal_mixer_damping = float(anderson_damping)
        formal_mixer_step_clip_factor = (
            None if anderson_step_clip_factor is None else float(anderson_step_clip_factor)
        )
        formal_mixer_reset_on_growth = bool(anderson_reset_on_growth)
        formal_mixer_reset_growth_factor = float(anderson_reset_growth_factor)
        formal_mixer_adaptive_damping_enabled = bool(anderson_adaptive_damping_enabled)
        formal_mixer_min_damping = float(anderson_min_damping)
        formal_mixer_max_damping = float(anderson_max_damping)
        formal_mixer_acceptance_residual_ratio_threshold = float(
            anderson_acceptance_residual_ratio_threshold
        )
        formal_mixer_collinearity_cosine_threshold = float(
            anderson_collinearity_cosine_threshold
        )
    return _build_route_result(
        result,
        mixing=mixing,
        mixer=mixer,
        solver_variant=solver_variant,
        max_iterations=max_iterations,
        formal_mixer_history_length=formal_mixer_history_length,
        formal_mixer_regularization=formal_mixer_regularization,
        formal_mixer_damping=formal_mixer_damping,
        formal_mixer_step_clip_factor=formal_mixer_step_clip_factor,
        formal_mixer_reset_on_growth=formal_mixer_reset_on_growth,
        formal_mixer_reset_growth_factor=formal_mixer_reset_growth_factor,
        formal_mixer_adaptive_damping_enabled=formal_mixer_adaptive_damping_enabled,
        formal_mixer_min_damping=formal_mixer_min_damping,
        formal_mixer_max_damping=formal_mixer_max_damping,
        formal_mixer_acceptance_residual_ratio_threshold=formal_mixer_acceptance_residual_ratio_threshold,
        formal_mixer_collinearity_cosine_threshold=formal_mixer_collinearity_cosine_threshold,
    )


def _select_best_anderson_route(
    *routes: H2JaxSingletMainlineRouteResult,
) -> H2JaxSingletMainlineRouteResult:
    def _score(route: H2JaxSingletMainlineRouteResult) -> tuple[int, float, float]:
        residual = route.final_density_residual
        return (
            0 if route.converged else 1,
            float("inf") if residual is None else residual,
            route.total_wall_time_seconds,
        )

    return min(routes, key=_score)


def _build_diagnosis(
    *,
    linear: H2JaxSingletMainlineRouteResult,
    diis: H2JaxSingletMainlineRouteResult,
    anderson_baseline: H2JaxSingletMainlineRouteResult,
    anderson_productionish: H2JaxSingletMainlineRouteResult,
    supplemental_anderson: H2JaxSingletMainlineRouteResult,
) -> str:
    if anderson_productionish.converged:
        return (
            "The productionish Anderson route converges within the 20-step main acceptance window. "
            "That means the previous Anderson failure was not primarily a statement that the singlet "
            "fixed-point map is hopeless; the smaller baseline implementation simply was not mature enough."
        )
    productionish_beats_baseline = (
        anderson_productionish.final_density_residual is not None
        and anderson_baseline.final_density_residual is not None
        and anderson_productionish.final_density_residual
        < anderson_baseline.final_density_residual - 1.0e-2
    )
    simple_residuals = tuple(
        value
        for value in (linear.final_density_residual, diis.final_density_residual)
        if value is not None
    )
    best_simple_residual = min(simple_residuals) if simple_residuals else None
    productionish_beats_simple_baselines = (
        best_simple_residual is not None
        and anderson_productionish.final_density_residual is not None
        and anderson_productionish.final_density_residual < best_simple_residual - 1.0e-2
    )
    steady_supplement = (
        supplemental_anderson.behavior.average_tail_residual_ratio is not None
        and supplemental_anderson.behavior.average_tail_residual_ratio < 0.98
        and not supplemental_anderson.behavior.entered_plateau
        and supplemental_anderson.behavior.verdict not in {"diverging", "weak_two_cycle"}
    )
    stalled_supplement = (
        supplemental_anderson.behavior.entered_plateau
        or supplemental_anderson.behavior.verdict in {"weak_two_cycle", "plateau_or_stall"}
        or (
            supplemental_anderson.behavior.average_tail_residual_ratio is not None
            and supplemental_anderson.behavior.average_tail_residual_ratio >= 0.99
        )
    )
    if productionish_beats_baseline and productionish_beats_simple_baselines and steady_supplement:
        return (
            "The productionish Anderson route does not quite converge in 20 steps, but it is materially better than the "
            "current Anderson baseline and the simpler linear/DIIS references, while the longer 40-step supplement "
            "still shows genuine residual contraction. That supports the interpretation that the earlier Anderson "
            "prototype was still too immature, rather than proving the singlet map itself is fundamentally unsalvageable."
        )
    if stalled_supplement:
        return (
            "The more mature Anderson route still fails to converge in 20 steps, and the 40-step supplement either "
            "enters a plateau or falls into tail oscillation instead of maintaining strong contraction. That now points "
            "more toward a hard singlet fixed-point map than toward a small remaining Anderson implementation gap."
        )
    return (
        "The productionish Anderson route changes the tail behavior only modestly and does not clearly beat the current "
        "Anderson baseline by enough to change the pass/fail conclusion. The longer supplement also does not deliver "
        "clean, decisive contraction to the acceptance threshold. That suggests the remaining obstacle is already at "
        "least partly the singlet fixed-point map itself, even if a fuller Anderson implementation could still help at the margins."
    )


def run_h2_jax_singlet_mainline_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2JaxSingletMainlineAuditResult:
    """Run the frozen H2 singlet productionish Anderson audit."""

    baseline_linear_route = _run_route(
        case=case,
        max_iterations=_SINGLET_MAINLINE_MAX_ITERATIONS,
        mixing=_SINGLET_MAINLINE_BASELINE_MIXING,
        mixer="linear",
        solver_variant="linear-0p10",
    )
    diis_route = _run_route(
        case=case,
        max_iterations=_SINGLET_MAINLINE_MAX_ITERATIONS,
        mixing=_SINGLET_MAINLINE_BASELINE_MIXING,
        mixer="diis",
        solver_variant="diis-prototype",
        enable_diis=True,
    )
    anderson_baseline_route = _run_route(
        case=case,
        max_iterations=_SINGLET_MAINLINE_MAX_ITERATIONS,
        mixing=_SINGLET_MAINLINE_BASELINE_MIXING,
        mixer="anderson",
        solver_variant="anderson-baseline",
        enable_anderson=True,
    )
    anderson_productionish_route = _run_route(
        case=case,
        max_iterations=_SINGLET_MAINLINE_MAX_ITERATIONS,
        mixing=_SINGLET_MAINLINE_BASELINE_MIXING,
        mixer="anderson",
        solver_variant="anderson-productionish",
        enable_anderson=True,
        anderson_history_length=_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_HISTORY,
        anderson_regularization=_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_REGULARIZATION,
        anderson_damping=_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_DAMPING,
        anderson_step_clip_factor=_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_STEP_CLIP,
        anderson_reset_on_growth=_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_RESET_ON_GROWTH,
        anderson_reset_growth_factor=_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_RESET_GROWTH_FACTOR,
        anderson_adaptive_damping_enabled=_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_ADAPTIVE_DAMPING,
        anderson_min_damping=_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_MIN_DAMPING,
        anderson_max_damping=_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_MAX_DAMPING,
        anderson_acceptance_residual_ratio_threshold=_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_ACCEPTANCE_RATIO,
        anderson_collinearity_cosine_threshold=_SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_COLLINEARITY,
    )
    best_anderson_20 = _select_best_anderson_route(
        anderson_baseline_route,
        anderson_productionish_route,
    )
    supplemental_anderson_route = _run_route(
        case=case,
        max_iterations=_SINGLET_MAINLINE_SUPPLEMENTAL_MAX_ITERATIONS,
        mixing=best_anderson_20.mixing,
        mixer="anderson",
        solver_variant=f"{best_anderson_20.solver_variant}-long40",
        enable_anderson=True,
        anderson_history_length=(
            _SINGLET_MAINLINE_ANDERSON_HISTORY
            if best_anderson_20.formal_mixer_history_length is None
            else best_anderson_20.formal_mixer_history_length
        ),
        anderson_regularization=(
            _SINGLET_MAINLINE_ANDERSON_REGULARIZATION
            if best_anderson_20.formal_mixer_regularization is None
            else best_anderson_20.formal_mixer_regularization
        ),
        anderson_damping=(
            _SINGLET_MAINLINE_ANDERSON_DAMPING
            if best_anderson_20.formal_mixer_damping is None
            else best_anderson_20.formal_mixer_damping
        ),
        anderson_step_clip_factor=best_anderson_20.formal_mixer_step_clip_factor,
        anderson_reset_on_growth=best_anderson_20.formal_mixer_reset_on_growth,
        anderson_reset_growth_factor=(
            _SINGLET_MAINLINE_ANDERSON_EXTENDED_RESET_GROWTH_FACTOR
            if best_anderson_20.formal_mixer_reset_growth_factor is None
            else best_anderson_20.formal_mixer_reset_growth_factor
        ),
    )
    return H2JaxSingletMainlineAuditResult(
        path_label="jax-singlet-mainline",
        spin_state_label="singlet",
        path_type=baseline_linear_route.path_type,
        baseline_linear_route=baseline_linear_route,
        diis_route=diis_route,
        anderson_baseline_route=anderson_baseline_route,
        anderson_productionish_route=anderson_productionish_route,
        supplemental_anderson_route=supplemental_anderson_route,
        diagnosis=_build_diagnosis(
            linear=baseline_linear_route,
            diis=diis_route,
            anderson_baseline=anderson_baseline_route,
            anderson_productionish=anderson_productionish_route,
            supplemental_anderson=supplemental_anderson_route,
        ),
        note=(
            "Formal singlet-only productionish Anderson audit on the frozen A-grid local-only mainline. The physical "
            "chain is held fixed and only the singlet mixer behavior changes between linear-0p10, a minimal "
            "DIIS/Pulay-style density mixer, the frozen Anderson baseline, and a more productionish Anderson route. "
            "The productionish Anderson keeps the same residual definition rho_out-rho_in and the same density-mixing "
            "target, but adds same-family engineering safeguards: stronger history management, adaptive damping, "
            "residual-based accept/reject, and residual-growth-triggered restart. A separate 40-step supplemental "
            "view is recorded only to distinguish implementation adequacy from a genuinely hard fixed-point map."
        ),
    )


def _print_route(route: H2JaxSingletMainlineRouteResult) -> None:
    print(f"route: {route.solver_variant}")
    print(
        f"  mixer={route.mixer}, mixing={route.mixing:.2f}, max_iterations={route.max_iterations}"
    )
    print(f"  converged: {route.converged}")
    print(f"  iterations: {route.iteration_count}")
    print(f"  final total energy [Ha]: {route.final_total_energy_ha:.12f}")
    if route.final_lowest_eigenvalue_ha is None:
        print("  final lowest eigenvalue [Ha]: n/a")
    else:
        print(f"  final lowest eigenvalue [Ha]: {route.final_lowest_eigenvalue_ha:.12f}")
    print(f"  final density residual: {route.final_density_residual}")
    print(f"  final energy change [Ha]: {route.final_energy_change_ha}")
    print(
        "  timing [s]: "
        f"total={route.total_wall_time_seconds:.6f}, "
        f"avg/iter={0.0 if route.average_iteration_wall_time_seconds is None else route.average_iteration_wall_time_seconds:.6f}, "
        f"eigensolver={route.timing_breakdown.eigensolver_wall_time_seconds:.6f}, "
        f"prepare={route.timing_breakdown.static_local_prepare_wall_time_seconds:.6f}, "
        f"hartree={route.timing_breakdown.hartree_solve_wall_time_seconds:.6f}, "
        f"energy_eval={route.timing_breakdown.energy_evaluation_wall_time_seconds:.6f}"
    )
    print(
        "  behavior: "
        f"verdict={route.behavior.verdict}, "
        f"detected_two_cycle={route.behavior.detected_two_cycle}, "
        f"energy_gap={route.behavior.even_odd_energy_gap_ha}, "
        f"residual_gap={route.behavior.even_odd_residual_gap}, "
        f"avg_tail_ratio={route.behavior.average_tail_residual_ratio}, "
        f"ratio_std={route.behavior.tail_residual_ratio_std}, "
        f"entered_plateau={route.behavior.entered_plateau}"
    )
    if route.mixer == "diis":
        print(f"  diis used iterations: {route.diis_used_iterations}")
        print(f"  diis fallback iterations: {route.diis_fallback_iterations}")
    if route.mixer == "anderson":
        print(
            "  anderson params: "
            f"history={route.formal_mixer_history_length}, "
            f"regularization={route.formal_mixer_regularization}, "
            f"damping={route.formal_mixer_damping}, "
            f"step_clip={route.formal_mixer_step_clip_factor}, "
            f"reset_on_growth={route.formal_mixer_reset_on_growth}, "
            f"reset_growth_factor={route.formal_mixer_reset_growth_factor}, "
            f"adaptive_damping={route.formal_mixer_adaptive_damping_enabled}, "
            f"min_damping={route.formal_mixer_min_damping}, "
            f"max_damping={route.formal_mixer_max_damping}, "
            f"accept_ratio={route.formal_mixer_acceptance_residual_ratio_threshold}, "
            f"collinearity={route.formal_mixer_collinearity_cosine_threshold}"
        )
        print(f"  anderson used iterations: {route.anderson_used_iterations}")
        print(f"  anderson fallback iterations: {route.anderson_fallback_iterations}")
        print(f"  anderson rejected iterations: {route.anderson_rejected_iterations}")
        print(f"  anderson reset iterations: {route.anderson_reset_iterations}")
        print(f"  anderson filtered history sizes: {route.anderson_filtered_history_sizes}")
        print(
            "  anderson effective damping history: "
            f"{route.anderson_effective_damping_history}"
        )
        print(
            "  anderson projected residual ratios: "
            f"{route.anderson_projected_residual_ratio_history}"
        )
    print(f"  tail energies [Ha]: {route.behavior.tail_energy_history_ha}")
    print(f"  tail residuals: {route.behavior.tail_density_residual_history}")
    print(f"  tail residual ratios: {route.behavior.tail_residual_ratios}")
    print(f"  tail dE [Ha]: {route.behavior.tail_energy_change_history_ha}")


def print_h2_jax_singlet_mainline_summary(result: H2JaxSingletMainlineAuditResult) -> None:
    """Print the compact H2 singlet Anderson adequacy audit summary."""

    print("IsoGridDFT H2 singlet productionish Anderson audit")
    print(f"note: {result.note}")
    print(
        "previous frozen-mainline baseline: "
        f"linear residual={H2_JAX_SINGLET_MAINLINE_BASELINE.baseline_linear_route.final_density_residual}, "
        f"anderson residual={H2_JAX_SINGLET_MAINLINE_BASELINE.anderson_baseline_route.final_density_residual}"
    )
    print(f"diagnosis: {result.diagnosis}")
    _print_route(result.baseline_linear_route)
    _print_route(result.diis_route)
    _print_route(result.anderson_baseline_route)
    _print_route(result.anderson_productionish_route)
    _print_route(result.supplemental_anderson_route)


def main() -> int:
    result = run_h2_jax_singlet_mainline_audit()
    print_h2_jax_singlet_mainline_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
