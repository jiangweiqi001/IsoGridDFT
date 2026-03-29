"""Singlet fixed-point local-difficulty audit on the frozen JAX A-grid mainline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from isogrid.audit.baselines import H2_JAX_SINGLET_MAINLINE_BASELINE
from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.ks import prepare_fixed_potential_static_local_operator_profiled
from isogrid.pseudo import LocalPotentialPatchParameters
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
_LOCAL_DIFFICULTY_PLATEAU_RATIO_LOWER = 0.985
_LOCAL_DIFFICULTY_PLATEAU_RATIO_UPPER = 1.015
_LOCAL_DIFFICULTY_PLATEAU_RATIO_STD_THRESHOLD = 0.02
_LOCAL_DIFFICULTY_PLATEAU_AMPLITUDE_THRESHOLD = 1.0e-2
_LOCAL_DIFFICULTY_NONCONTRACTIVE_RATIO_THRESHOLD = 1.002
_LOCAL_DIFFICULTY_POORLY_CONTRACTIVE_RATIO_THRESHOLD = 0.995
_LOCAL_DIFFICULTY_SECANT_COLLINEARITY_THRESHOLD = 0.995
_LOCAL_DIFFICULTY_SECANT_MIN_NORM = 1.0e-14


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
class H2JaxSingletFixedPointLocalDifficulty:
    """Very small local fixed-point difficulty summary near the singlet tail."""

    tail_window_length: int
    average_tail_residual_ratio: float | None
    tail_residual_ratio_std: float | None
    maximum_tail_residual_ratio: float | None
    entered_plateau: bool
    plateau_window_length: int
    tail_residual_amplitude: float | None
    weak_cycle_indicator: bool
    local_contraction_verdict: str
    secant_subspace_condition_proxy: float | None
    secant_collinearity_max_abs_cosine: float | None
    diagnosis: str


@dataclass(frozen=True)
class H2JaxSingletResponseChannelDifficulty:
    """Very small channel-wise tail difficulty summary near the singlet fixed-point tail."""

    tail_pair_iterations: tuple[int, int] | None
    density_secant_norm: float | None
    total_output_response_proxy: float | None
    total_effective_potential_amplification_proxy: float | None
    hartree_potential_amplification_proxy: float | None
    xc_potential_amplification_proxy: float | None
    local_orbital_potential_amplification_proxy: float | None
    hartree_potential_contribution_share: float | None
    xc_potential_contribution_share: float | None
    local_orbital_potential_contribution_share: float | None
    hartree_output_sensitivity_proxy: float | None
    xc_output_sensitivity_proxy: float | None
    local_orbital_output_sensitivity_proxy: float | None
    coupling_excess_output_sensitivity_proxy: float | None
    primary_difficulty_channel: str | None
    dominant_coupling_label: str | None
    diagnosis: str


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
    fixed_point_local_difficulty: H2JaxSingletFixedPointLocalDifficulty
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
    response_channel_difficulty: H2JaxSingletResponseChannelDifficulty | None = None


@dataclass(frozen=True)
class H2JaxSingletMainlineAuditResult:
    """Singlet fixed-point local-difficulty audit on the frozen JAX A-grid mainline."""

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


def _spin_density_residual_vector(record: Any) -> np.ndarray:
    residual_up = np.asarray(record.output_rho_up - record.input_rho_up, dtype=np.float64)
    residual_down = np.asarray(record.output_rho_down - record.input_rho_down, dtype=np.float64)
    return np.concatenate((residual_up.ravel(), residual_down.ravel()))


def _tail_plateau_window_length(ratios: tuple[float, ...]) -> int:
    if not ratios:
        return 0
    window = 0
    for ratio in reversed(ratios):
        if _LOCAL_DIFFICULTY_PLATEAU_RATIO_LOWER <= ratio <= _LOCAL_DIFFICULTY_PLATEAU_RATIO_UPPER:
            window += 1
        else:
            break
    return window


def _estimate_secant_subspace_condition(
    residual_vectors: tuple[np.ndarray, ...],
) -> float | None:
    if len(residual_vectors) < 3:
        return None
    secants = []
    for previous, current in zip(residual_vectors[:-1], residual_vectors[1:], strict=True):
        delta = np.asarray(current - previous, dtype=np.float64)
        if float(np.linalg.norm(delta)) > _LOCAL_DIFFICULTY_SECANT_MIN_NORM:
            secants.append(delta)
    if len(secants) < 2:
        return None
    secant_matrix = np.stack(secants, axis=1)
    gram = np.asarray(secant_matrix.T @ secant_matrix, dtype=np.float64)
    try:
        return float(np.linalg.cond(gram))
    except np.linalg.LinAlgError:
        return None


def _estimate_secant_collinearity(
    residual_vectors: tuple[np.ndarray, ...],
) -> float | None:
    if len(residual_vectors) < 3:
        return None
    secants = []
    for previous, current in zip(residual_vectors[:-1], residual_vectors[1:], strict=True):
        delta = np.asarray(current - previous, dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        if norm > _LOCAL_DIFFICULTY_SECANT_MIN_NORM:
            secants.append(delta / norm)
    if len(secants) < 2:
        return None
    max_abs_cosine = 0.0
    for idx, left in enumerate(secants[:-1]):
        for right in secants[idx + 1 :]:
            max_abs_cosine = max(max_abs_cosine, float(abs(np.dot(left, right))))
    return max_abs_cosine


def _build_fixed_point_local_difficulty(
    result: H2StaticLocalScfDryRunResult,
    *,
    behavior: H2JaxSingletMainlineBehavior,
) -> H2JaxSingletFixedPointLocalDifficulty:
    tail_residuals = np.asarray(behavior.tail_density_residual_history, dtype=np.float64)
    tail_ratios = np.asarray(behavior.tail_residual_ratios, dtype=np.float64)
    average_tail_residual_ratio = _safe_mean(tail_ratios)
    tail_residual_ratio_std = _safe_std(tail_ratios)
    maximum_tail_residual_ratio = (
        None if tail_ratios.size == 0 else float(np.max(tail_ratios))
    )
    tail_residual_amplitude = (
        None if tail_residuals.size == 0 else float(np.max(tail_residuals) - np.min(tail_residuals))
    )
    plateau_window_length = _tail_plateau_window_length(behavior.tail_residual_ratios)
    residual_vectors = tuple(
        _spin_density_residual_vector(record) for record in result.history[-_TAIL_SUMMARY_LENGTH:]
    )
    secant_subspace_condition_proxy = _estimate_secant_subspace_condition(residual_vectors)
    secant_collinearity_max_abs_cosine = _estimate_secant_collinearity(residual_vectors)
    weak_cycle_indicator = bool(
        behavior.detected_two_cycle
        or (
            behavior.even_odd_residual_gap is not None
            and tail_residual_amplitude is not None
            and tail_residual_amplitude > 0.0
            and behavior.even_odd_residual_gap > 0.35 * tail_residual_amplitude
        )
    )

    if average_tail_residual_ratio is None:
        local_contraction_verdict = "insufficient_tail"
    elif weak_cycle_indicator and average_tail_residual_ratio >= _LOCAL_DIFFICULTY_POORLY_CONTRACTIVE_RATIO_THRESHOLD:
        local_contraction_verdict = "oscillatory_near_neutral"
    elif (
        average_tail_residual_ratio >= _LOCAL_DIFFICULTY_NONCONTRACTIVE_RATIO_THRESHOLD
        or (
            maximum_tail_residual_ratio is not None
            and maximum_tail_residual_ratio >= _LOCAL_DIFFICULTY_NONCONTRACTIVE_RATIO_THRESHOLD
        )
    ):
        local_contraction_verdict = "locally_noncontractive_or_expansive"
    elif average_tail_residual_ratio >= _LOCAL_DIFFICULTY_POORLY_CONTRACTIVE_RATIO_THRESHOLD:
        local_contraction_verdict = "poorly_contractive_near_unity"
    else:
        local_contraction_verdict = "contractive_but_slow"

    plateau_like = bool(
        behavior.entered_plateau
        or (
            average_tail_residual_ratio is not None
            and tail_residual_ratio_std is not None
            and tail_residual_amplitude is not None
            and _LOCAL_DIFFICULTY_PLATEAU_RATIO_LOWER
            <= average_tail_residual_ratio
            <= _LOCAL_DIFFICULTY_PLATEAU_RATIO_UPPER
            and tail_residual_ratio_std <= _LOCAL_DIFFICULTY_PLATEAU_RATIO_STD_THRESHOLD
            and tail_residual_amplitude <= _LOCAL_DIFFICULTY_PLATEAU_AMPLITUDE_THRESHOLD
        )
    )
    entered_plateau = bool(plateau_like)

    if entered_plateau and local_contraction_verdict in {
        "locally_noncontractive_or_expansive",
        "poorly_contractive_near_unity",
    }:
        diagnosis = (
            "tail residual ratios sit very close to unity and the residual norm enters a narrow plateau, "
            "which is consistent with a locally near-noncontractive singlet fixed-point map"
        )
    elif weak_cycle_indicator:
        diagnosis = (
            "tail history shows weak periodic structure, so the local map looks oscillatory and only marginally damped"
        )
    elif local_contraction_verdict == "contractive_but_slow":
        diagnosis = (
            "tail ratios still contract on average, but the contraction is weak enough that the current 20-step window is insufficient"
        )
    else:
        diagnosis = (
            "tail history does not show decisive contraction, which points more to a hard local map than to a small mixer tweak"
        )

    if (
        secant_collinearity_max_abs_cosine is not None
        and secant_collinearity_max_abs_cosine >= _LOCAL_DIFFICULTY_SECANT_COLLINEARITY_THRESHOLD
    ):
        diagnosis += "; recent secant directions are nearly collinear, so the mixer subspace is also becoming low-rank"
    if (
        secant_subspace_condition_proxy is not None
        and np.isfinite(secant_subspace_condition_proxy)
        and secant_subspace_condition_proxy >= 1.0e8
    ):
        diagnosis += "; the secant Gram proxy is very ill-conditioned"

    return H2JaxSingletFixedPointLocalDifficulty(
        tail_window_length=len(behavior.tail_density_residual_history),
        average_tail_residual_ratio=average_tail_residual_ratio,
        tail_residual_ratio_std=tail_residual_ratio_std,
        maximum_tail_residual_ratio=maximum_tail_residual_ratio,
        entered_plateau=entered_plateau,
        plateau_window_length=plateau_window_length,
        tail_residual_amplitude=tail_residual_amplitude,
        weak_cycle_indicator=weak_cycle_indicator,
        local_contraction_verdict=local_contraction_verdict,
        secant_subspace_condition_proxy=secant_subspace_condition_proxy,
        secant_collinearity_max_abs_cosine=secant_collinearity_max_abs_cosine,
        diagnosis=diagnosis,
    )


def _default_patch_parameters() -> LocalPotentialPatchParameters:
    return LocalPotentialPatchParameters(
        patch_radius_scale=0.75,
        patch_grid_shape=(25, 25, 25),
        correction_strength=1.30,
        interpolation_neighbors=8,
    )


def _stack_spin_fields(field_up: np.ndarray, field_down: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(field_up, dtype=np.float64).ravel(),
            np.asarray(field_down, dtype=np.float64).ravel(),
        )
    )


def _safe_norm_ratio(numerator: np.ndarray, denominator_norm: float) -> float | None:
    if denominator_norm <= 1.0e-16:
        return None
    return float(np.linalg.norm(np.asarray(numerator, dtype=np.float64)) / denominator_norm)


def _channel_contribution_share(
    channel_vector: np.ndarray,
    total_vector: np.ndarray,
) -> float | None:
    total_field = np.asarray(total_vector, dtype=np.float64)
    denominator = float(np.dot(total_field, total_field))
    if denominator <= 1.0e-16:
        return None
    channel_field = np.asarray(channel_vector, dtype=np.float64)
    return float(np.dot(channel_field, total_field) / denominator)


def _build_spin_contexts(
    *,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    case: BenchmarkCase,
    patch_parameters: LocalPotentialPatchParameters,
):
    grid_geometry = build_h2_local_patch_development_monitor_grid()
    up_context, _ = prepare_fixed_potential_static_local_operator_profiled(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
        case=case,
        use_monitor_patch=True,
        patch_parameters=patch_parameters,
        kinetic_version="trial_fix",
        hartree_backend="jax",
        use_jax_hartree_cached_operator=True,
        jax_hartree_cg_impl="jax_loop",
        jax_hartree_cg_preconditioner="none",
        jax_hartree_line_preconditioner_impl="baseline",
    )
    down_context, _ = prepare_fixed_potential_static_local_operator_profiled(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="down",
        case=case,
        use_monitor_patch=True,
        patch_parameters=patch_parameters,
        kinetic_version="trial_fix",
        hartree_backend="jax",
        use_jax_hartree_cached_operator=True,
        jax_hartree_cg_impl="jax_loop",
        jax_hartree_cg_preconditioner="none",
        jax_hartree_line_preconditioner_impl="baseline",
    )
    return grid_geometry, up_context, down_context


def _build_response_channel_difficulty(
    result: H2StaticLocalScfDryRunResult,
    *,
    case: BenchmarkCase,
) -> H2JaxSingletResponseChannelDifficulty | None:
    if len(result.history) < 2:
        return None
    previous_record = result.history[-2]
    current_record = result.history[-1]
    patch_parameters = _default_patch_parameters()
    _, previous_up_context, previous_down_context = _build_spin_contexts(
        rho_up=previous_record.input_rho_up,
        rho_down=previous_record.input_rho_down,
        case=case,
        patch_parameters=patch_parameters,
    )
    _, current_up_context, current_down_context = _build_spin_contexts(
        rho_up=current_record.input_rho_up,
        rho_down=current_record.input_rho_down,
        case=case,
        patch_parameters=patch_parameters,
    )
    delta_rho_in = _stack_spin_fields(
        current_record.input_rho_up - previous_record.input_rho_up,
        current_record.input_rho_down - previous_record.input_rho_down,
    )
    density_secant_norm = float(np.linalg.norm(delta_rho_in))
    if density_secant_norm <= 1.0e-16:
        return None
    delta_rho_out = _stack_spin_fields(
        current_record.output_rho_up - previous_record.output_rho_up,
        current_record.output_rho_down - previous_record.output_rho_down,
    )
    total_output_response_proxy = float(np.linalg.norm(delta_rho_out) / density_secant_norm)

    delta_v_hartree = _stack_spin_fields(
        current_up_context.hartree_potential - previous_up_context.hartree_potential,
        current_down_context.hartree_potential - previous_down_context.hartree_potential,
    )
    delta_v_xc = _stack_spin_fields(
        current_up_context.xc_potential - previous_up_context.xc_potential,
        current_down_context.xc_potential - previous_down_context.xc_potential,
    )
    delta_v_total = _stack_spin_fields(
        current_up_context.effective_local_potential - previous_up_context.effective_local_potential,
        current_down_context.effective_local_potential - previous_down_context.effective_local_potential,
    )
    delta_v_local_orbital = np.asarray(
        delta_v_total - delta_v_hartree - delta_v_xc,
        dtype=np.float64,
    )
    hartree_potential_contribution_share = _channel_contribution_share(
        delta_v_hartree,
        delta_v_total,
    )
    xc_potential_contribution_share = _channel_contribution_share(delta_v_xc, delta_v_total)
    local_orbital_potential_contribution_share = _channel_contribution_share(
        delta_v_local_orbital,
        delta_v_total,
    )
    hartree_output_sensitivity_proxy = (
        None
        if hartree_potential_contribution_share is None
        else float(hartree_potential_contribution_share * total_output_response_proxy)
    )
    xc_output_sensitivity_proxy = (
        None
        if xc_potential_contribution_share is None
        else float(xc_potential_contribution_share * total_output_response_proxy)
    )
    local_orbital_output_sensitivity_proxy = (
        None
        if local_orbital_potential_contribution_share is None
        else float(local_orbital_potential_contribution_share * total_output_response_proxy)
    )
    sensitivity_pairs = {
        "hartree": abs(hartree_output_sensitivity_proxy or 0.0),
        "xc": abs(xc_output_sensitivity_proxy or 0.0),
        "local_orbital": abs(local_orbital_output_sensitivity_proxy or 0.0),
    }
    ordered_channels = sorted(sensitivity_pairs.items(), key=lambda item: item[1], reverse=True)
    primary_channel = ordered_channels[0][0] if ordered_channels and ordered_channels[0][1] > 0.0 else None
    dominant_coupling_label = None
    if len(ordered_channels) >= 2:
        top_name, top_value = ordered_channels[0]
        second_name, second_value = ordered_channels[1]
        if top_value > 0.0 and second_value >= 0.75 * top_value:
            dominant_coupling_label = f"{top_name}+{second_name}"
    coupling_excess_output_sensitivity_proxy = None
    if (
        total_output_response_proxy is not None
        and hartree_output_sensitivity_proxy is not None
        and xc_output_sensitivity_proxy is not None
        and local_orbital_output_sensitivity_proxy is not None
    ):
        coupling_excess_output_sensitivity_proxy = float(
            abs(hartree_output_sensitivity_proxy)
            + abs(xc_output_sensitivity_proxy)
            + abs(local_orbital_output_sensitivity_proxy)
            - abs(total_output_response_proxy)
        )
    diagnosis = (
        "Channel-wise tail difficulty proxy built from the last singlet secant pair. "
        "Hartree/XC/local_orbital potential proxies use stacked spin potential differences divided by "
        "the input-density secant norm. The channel output proxies are secant-based decompositions of the "
        "observed total output-response amplification, weighted by each channel's directional contribution "
        "to the full effective-potential change. The local_orbital label is only the closest audit proxy: "
        "it captures density-dependent local ionic/patch changes plus the residual orbital-response-aligned "
        "part of the map that is not cleanly attributable to Hartree or XC, and it is not a strict isolated "
        "kinetic linear-response channel."
    )
    return H2JaxSingletResponseChannelDifficulty(
        tail_pair_iterations=(int(previous_record.iteration), int(current_record.iteration)),
        density_secant_norm=density_secant_norm,
        total_output_response_proxy=total_output_response_proxy,
        total_effective_potential_amplification_proxy=_safe_norm_ratio(
            delta_v_total,
            density_secant_norm,
        ),
        hartree_potential_amplification_proxy=_safe_norm_ratio(delta_v_hartree, density_secant_norm),
        xc_potential_amplification_proxy=_safe_norm_ratio(delta_v_xc, density_secant_norm),
        local_orbital_potential_amplification_proxy=_safe_norm_ratio(
            delta_v_local_orbital,
            density_secant_norm,
        ),
        hartree_potential_contribution_share=hartree_potential_contribution_share,
        xc_potential_contribution_share=xc_potential_contribution_share,
        local_orbital_potential_contribution_share=local_orbital_potential_contribution_share,
        hartree_output_sensitivity_proxy=hartree_output_sensitivity_proxy,
        xc_output_sensitivity_proxy=xc_output_sensitivity_proxy,
        local_orbital_output_sensitivity_proxy=local_orbital_output_sensitivity_proxy,
        coupling_excess_output_sensitivity_proxy=coupling_excess_output_sensitivity_proxy,
        primary_difficulty_channel=primary_channel,
        dominant_coupling_label=dominant_coupling_label,
        diagnosis=diagnosis,
    )


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
    case: BenchmarkCase,
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
    behavior = _build_behavior(result, converged=bool(result.converged))
    fixed_point_local_difficulty = _build_fixed_point_local_difficulty(
        result,
        behavior=behavior,
    )
    response_channel_difficulty = None
    if mixer == "anderson" and "productionish" in solver_variant:
        response_channel_difficulty = _build_response_channel_difficulty(
            result,
            case=case,
        )
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
        behavior=behavior,
        fixed_point_local_difficulty=fixed_point_local_difficulty,
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
        response_channel_difficulty=response_channel_difficulty,
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
        case=case,
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
    productionish_difficulty = anderson_productionish.fixed_point_local_difficulty
    supplemental_difficulty = supplemental_anderson.fixed_point_local_difficulty
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
        supplemental_difficulty.average_tail_residual_ratio is not None
        and supplemental_difficulty.average_tail_residual_ratio < 0.98
        and not supplemental_difficulty.entered_plateau
        and supplemental_anderson.behavior.verdict not in {"diverging", "weak_two_cycle"}
    )
    stalled_supplement = (
        supplemental_difficulty.entered_plateau
        or supplemental_anderson.behavior.verdict in {"weak_two_cycle", "plateau_or_stall"}
        or (
            supplemental_difficulty.average_tail_residual_ratio is not None
            and supplemental_difficulty.average_tail_residual_ratio >= 0.99
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
            "The more mature Anderson route still fails to converge in 20 steps. Its tail residual ratios do not "
            f"recover clean contraction (about {productionish_difficulty.average_tail_residual_ratio}), and the 40-step "
            f"supplement also stays at or above neutral contraction on average (tail ratio about "
            f"{supplemental_difficulty.average_tail_residual_ratio}). That combination points more toward a locally "
            "near-noncontractive or badly conditioned singlet fixed-point map than toward a small remaining Anderson "
            "implementation gap."
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
    """Run the frozen H2 singlet tail response-channel audit."""

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
    supplemental_anderson_route = _run_route(
        case=case,
        max_iterations=_SINGLET_MAINLINE_SUPPLEMENTAL_MAX_ITERATIONS,
        mixing=anderson_productionish_route.mixing,
        mixer="anderson",
        solver_variant=f"{anderson_productionish_route.solver_variant}-long40",
        enable_anderson=True,
        anderson_history_length=(
            _SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_HISTORY
            if anderson_productionish_route.formal_mixer_history_length is None
            else anderson_productionish_route.formal_mixer_history_length
        ),
        anderson_regularization=(
            _SINGLET_MAINLINE_ANDERSON_REGULARIZATION
            if anderson_productionish_route.formal_mixer_regularization is None
            else anderson_productionish_route.formal_mixer_regularization
        ),
        anderson_damping=(
            _SINGLET_MAINLINE_ANDERSON_DAMPING
            if anderson_productionish_route.formal_mixer_damping is None
            else anderson_productionish_route.formal_mixer_damping
        ),
        anderson_step_clip_factor=anderson_productionish_route.formal_mixer_step_clip_factor,
        anderson_reset_on_growth=anderson_productionish_route.formal_mixer_reset_on_growth,
        anderson_reset_growth_factor=(
            _SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_RESET_GROWTH_FACTOR
            if anderson_productionish_route.formal_mixer_reset_growth_factor is None
            else anderson_productionish_route.formal_mixer_reset_growth_factor
        ),
        anderson_adaptive_damping_enabled=anderson_productionish_route.formal_mixer_adaptive_damping_enabled,
        anderson_min_damping=(
            _SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_MIN_DAMPING
            if anderson_productionish_route.formal_mixer_min_damping is None
            else anderson_productionish_route.formal_mixer_min_damping
        ),
        anderson_max_damping=(
            _SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_MAX_DAMPING
            if anderson_productionish_route.formal_mixer_max_damping is None
            else anderson_productionish_route.formal_mixer_max_damping
        ),
        anderson_acceptance_residual_ratio_threshold=(
            _SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_ACCEPTANCE_RATIO
            if anderson_productionish_route.formal_mixer_acceptance_residual_ratio_threshold is None
            else anderson_productionish_route.formal_mixer_acceptance_residual_ratio_threshold
        ),
        anderson_collinearity_cosine_threshold=(
            _SINGLET_MAINLINE_ANDERSON_PRODUCTIONISH_COLLINEARITY
            if anderson_productionish_route.formal_mixer_collinearity_cosine_threshold is None
            else anderson_productionish_route.formal_mixer_collinearity_cosine_threshold
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
            "Singlet-only fixed-point local-difficulty and tail response-channel audit on the frozen A-grid local-only "
            "mainline. The physical chain is held fixed and only the singlet mixer behavior changes between linear-0p10, "
            "a minimal DIIS/Pulay-style density mixer, the frozen Anderson baseline, and a more productionish Anderson "
            "route. The productionish Anderson keeps the same residual definition rho_out-rho_in and the same density-"
            "mixing target, but adds same-family engineering safeguards: stronger history management, adaptive damping, "
            "residual-based accept/reject, and residual-growth-triggered restart. The productionish tail is also "
            "diagnosed with a secant-based response-channel decomposition that compares Hartree, XC, and the closest "
            "available local-orbital residual proxy without changing the physical map. A separate 40-step supplemental "
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
    print(
        "  fixed-point proxy: "
        f"contraction={route.fixed_point_local_difficulty.local_contraction_verdict}, "
        f"avg_tail_ratio={route.fixed_point_local_difficulty.average_tail_residual_ratio}, "
        f"ratio_std={route.fixed_point_local_difficulty.tail_residual_ratio_std}, "
        f"max_tail_ratio={route.fixed_point_local_difficulty.maximum_tail_residual_ratio}, "
        f"plateau={route.fixed_point_local_difficulty.entered_plateau}, "
        f"plateau_window={route.fixed_point_local_difficulty.plateau_window_length}, "
        f"tail_amplitude={route.fixed_point_local_difficulty.tail_residual_amplitude}, "
        f"secant_cond={route.fixed_point_local_difficulty.secant_subspace_condition_proxy}, "
        f"secant_collinearity={route.fixed_point_local_difficulty.secant_collinearity_max_abs_cosine}"
    )
    print(f"  fixed-point diagnosis: {route.fixed_point_local_difficulty.diagnosis}")
    if route.response_channel_difficulty is not None:
        print(
            "  response-channel proxy: "
            f"tail_pair={route.response_channel_difficulty.tail_pair_iterations}, "
            f"primary={route.response_channel_difficulty.primary_difficulty_channel}, "
            f"coupling={route.response_channel_difficulty.dominant_coupling_label}, "
            f"total_output={route.response_channel_difficulty.total_output_response_proxy}, "
            f"hartree_output={route.response_channel_difficulty.hartree_output_sensitivity_proxy}, "
            f"xc_output={route.response_channel_difficulty.xc_output_sensitivity_proxy}, "
            f"local_orbital_output={route.response_channel_difficulty.local_orbital_output_sensitivity_proxy}, "
            f"total_potential={route.response_channel_difficulty.total_effective_potential_amplification_proxy}, "
            f"hartree_share={route.response_channel_difficulty.hartree_potential_contribution_share}, "
            f"xc_share={route.response_channel_difficulty.xc_potential_contribution_share}, "
            f"local_share={route.response_channel_difficulty.local_orbital_potential_contribution_share}"
        )
        print(f"  response-channel diagnosis: {route.response_channel_difficulty.diagnosis}")
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
    """Print the compact H2 singlet fixed-point local-difficulty audit summary."""

    print("IsoGridDFT H2 singlet fixed-point and tail response-channel audit")
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
