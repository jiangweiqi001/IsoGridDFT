"""Reusable charge/spin-channel SCF controller utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.ks import build_total_density
from isogrid.ops import integrate_field
from isogrid.ops import validate_orbital_field

GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry


@dataclass(frozen=True)
class ScfChannelResiduals:
    """Charge/spin residual decomposition on the current grid."""

    charge_residual: np.ndarray
    spin_residual: np.ndarray
    charge_residual_norm: float
    spin_residual_norm: float


@dataclass(frozen=True)
class ScfControllerSignals:
    """Controller-facing nonlinear-response indicators for one SCF step."""

    density_residual_ratio: float | None
    hartree_share: float | None
    occupied_orbital_overlap_abs: float | None
    lowest_subspace_rotation_max_angle_deg: float | None
    lowest_gap_ha: float | None


@dataclass(frozen=True)
class ScfControllerState:
    """Persistent controller state carried across SCF iterations.

    `iteration_index` is used by the opening policy to keep the first few
    closed-shell singlet steps on larger grids more conservative before the
    regular Hartree-aware recovery logic takes over.
    """

    charge_mixing: float
    spin_mixing: float
    charge_cautious_steps_remaining: int
    stable_steps: int
    iteration_index: int
    last_flags: tuple[str, ...]
    previous_hartree_share: float | None = None
    recent_charge_residual_history: tuple[np.ndarray, ...] = ()

    @classmethod
    def initial(
        cls,
        *,
        charge_mixing: float = 0.2,
        spin_mixing: float = 0.2,
    ) -> ScfControllerState:
        return cls(
            charge_mixing=float(charge_mixing),
            spin_mixing=float(spin_mixing),
            charge_cautious_steps_remaining=0,
            stable_steps=0,
            iteration_index=0,
            last_flags=(),
            previous_hartree_share=None,
            recent_charge_residual_history=(),
        )


@dataclass(frozen=True)
class ScfControllerConfig:
    """Static controller configuration for one SCF route.

    The opening-policy fields are deliberately lightweight. They are not a full
    size-aware dielectric model; they only cap early-step charge mixing on
    larger closed-shell singlet monitor grids where step-1 precursor signals
    can still look deceptively mild.
    """

    name: str
    baseline_charge_mixing: float
    baseline_spin_mixing: float
    min_charge_mixing: float
    min_spin_mixing: float
    severe_charge_mixing: float
    charge_release_rate: float
    spin_release_rate: float
    cautious_hold_steps: int
    stable_steps_to_release: int
    opening_steps: int
    opening_charge_mixing: float
    opening_charge_mixing_large_risk: float
    opening_grid_point_threshold: int
    preconditioned_grid_point_boost_threshold: int
    recovery_pause_hartree_share: float
    recovery_pause_hartree_rise: float
    preconditioned_high_frequency_mixing: float
    preconditioned_smoothing_passes: int
    modal_history_length: int
    modal_min_explained_fraction: float
    modal_boost_mixing: float
    modal_persistent_extra_mixing: float
    modal_persistent_min_history: int
    modal_persistent_grid_point_threshold: int
    low_rank_modal_ratio_floor: float
    low_rank_modal_ratio_ceiling: float
    hartree_share_trigger: float
    residual_ratio_trigger: float
    residual_ratio_severe_trigger: float
    occupied_overlap_trigger: float
    subspace_rotation_trigger_deg: float
    lowest_gap_trigger_ha: float

    @classmethod
    def generic_charge_spin(
        cls,
        *,
        baseline_mixing: float = 0.2,
    ) -> ScfControllerConfig:
        return cls(
            name="generic_charge_spin",
            baseline_charge_mixing=float(baseline_mixing),
            baseline_spin_mixing=float(baseline_mixing),
            min_charge_mixing=max(0.02, 0.10 * float(baseline_mixing)),
            min_spin_mixing=max(0.10, 0.75 * float(baseline_mixing)),
            severe_charge_mixing=max(0.005, 0.025 * float(baseline_mixing)),
            charge_release_rate=max(0.005, 0.05 * float(baseline_mixing)),
            spin_release_rate=max(0.008, 0.05 * float(baseline_mixing)),
            cautious_hold_steps=4,
            stable_steps_to_release=4,
            opening_steps=2,
            opening_charge_mixing=max(0.02, 0.10 * float(baseline_mixing)),
            opening_charge_mixing_large_risk=max(0.005, 0.025 * float(baseline_mixing)),
            opening_grid_point_threshold=2000,
            preconditioned_grid_point_boost_threshold=0,
            recovery_pause_hartree_share=0.20,
            recovery_pause_hartree_rise=0.10,
            preconditioned_high_frequency_mixing=0.0,
            preconditioned_smoothing_passes=0,
            modal_history_length=0,
            modal_min_explained_fraction=1.1,
            modal_boost_mixing=0.0,
            modal_persistent_extra_mixing=0.0,
            modal_persistent_min_history=0,
            modal_persistent_grid_point_threshold=0,
            low_rank_modal_ratio_floor=1.0,
            low_rank_modal_ratio_ceiling=1.0,
            hartree_share_trigger=0.60,
            residual_ratio_trigger=0.98,
            residual_ratio_severe_trigger=1.05,
            occupied_overlap_trigger=0.25,
            subspace_rotation_trigger_deg=70.0,
            lowest_gap_trigger_ha=0.08,
        )

    @classmethod
    def generic_charge_spin_preconditioned(
        cls,
        *,
        baseline_mixing: float = 0.2,
    ) -> ScfControllerConfig:
        return cls(
            name="generic_charge_spin_preconditioned",
            baseline_charge_mixing=float(baseline_mixing),
            baseline_spin_mixing=float(baseline_mixing),
            min_charge_mixing=max(0.02, 0.10 * float(baseline_mixing)),
            min_spin_mixing=max(0.10, 0.75 * float(baseline_mixing)),
            severe_charge_mixing=max(0.005, 0.025 * float(baseline_mixing)),
            charge_release_rate=max(0.005, 0.05 * float(baseline_mixing)),
            spin_release_rate=max(0.008, 0.05 * float(baseline_mixing)),
            cautious_hold_steps=4,
            stable_steps_to_release=4,
            opening_steps=2,
            opening_charge_mixing=max(0.02, 0.10 * float(baseline_mixing)),
            opening_charge_mixing_large_risk=max(0.005, 0.025 * float(baseline_mixing)),
            opening_grid_point_threshold=2000,
            preconditioned_grid_point_boost_threshold=3000,
            recovery_pause_hartree_share=0.20,
            recovery_pause_hartree_rise=0.10,
            preconditioned_high_frequency_mixing=max(0.03, 0.15 * float(baseline_mixing)),
            preconditioned_smoothing_passes=3,
            modal_history_length=4,
            modal_min_explained_fraction=0.90,
            modal_boost_mixing=max(0.03, 0.20 * float(baseline_mixing)),
            modal_persistent_extra_mixing=max(0.02, 0.10 * float(baseline_mixing)),
            modal_persistent_min_history=4,
            modal_persistent_grid_point_threshold=3500,
            low_rank_modal_ratio_floor=0.85,
            low_rank_modal_ratio_ceiling=0.995,
            hartree_share_trigger=0.60,
            residual_ratio_trigger=0.98,
            residual_ratio_severe_trigger=1.05,
            occupied_overlap_trigger=0.25,
            subspace_rotation_trigger_deg=70.0,
            lowest_gap_trigger_ha=0.08,
        )


@dataclass(frozen=True)
class ControllerStepResult:
    """Next-density proposal from the generic SCF controller."""

    rho_up_next: np.ndarray
    rho_down_next: np.ndarray
    channel_residuals: ScfChannelResiduals
    signals: ScfControllerSignals
    charge_mixing: float
    spin_mixing: float
    state: ScfControllerState
    flags: tuple[str, ...]
    rho_charge_unbounded_trial: np.ndarray
    rho_charge_trial: np.ndarray
    rho_charge_preclip_next: np.ndarray
    rho_charge_postclip_next: np.ndarray


def _renormalize_density(
    rho: np.ndarray,
    *,
    target_electrons: float,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    if target_electrons == 0.0:
        return np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    density = validate_orbital_field(
        rho,
        grid_geometry=grid_geometry,
        name="rho",
    ).astype(np.float64)
    integral = float(integrate_field(density, grid_geometry=grid_geometry))
    if integral <= 0.0:
        raise ValueError("A positive density integral is required for renormalization.")
    return np.asarray(density * (target_electrons / integral), dtype=np.float64)


def _weighted_field_norm(
    field: np.ndarray,
    *,
    grid_geometry: GridGeometryLike,
) -> float:
    return float(
        np.sqrt(
            max(
                float(
                    integrate_field(
                        np.asarray(field, dtype=np.float64) ** 2,
                        grid_geometry=grid_geometry,
                    )
                ),
                0.0,
            )
        )
    )


def _smooth_axis_edge(field: np.ndarray, axis: int) -> np.ndarray:
    pad_width = [(0, 0)] * field.ndim
    pad_width[axis] = (1, 1)
    padded = np.pad(np.asarray(field, dtype=np.float64), pad_width, mode="edge")
    center_slices = [slice(None)] * field.ndim
    center_slices[axis] = slice(1, -1)
    minus_slices = list(center_slices)
    plus_slices = list(center_slices)
    minus_slices[axis] = slice(0, -2)
    plus_slices[axis] = slice(2, None)
    return np.asarray(
        0.25 * padded[tuple(minus_slices)]
        + 0.50 * padded[tuple(center_slices)]
        + 0.25 * padded[tuple(plus_slices)],
        dtype=np.float64,
    )


def _smooth_charge_residual(
    field: np.ndarray,
    *,
    smoothing_passes: int,
) -> np.ndarray:
    smoothed = np.asarray(field, dtype=np.float64)
    for _ in range(max(int(smoothing_passes), 0)):
        for axis in range(smoothed.ndim):
            smoothed = _smooth_axis_edge(smoothed, axis)
    return smoothed


def _density_support_weight(
    rho_charge_current: np.ndarray,
) -> np.ndarray:
    """Return a smooth 0..1 molecular-support weight for charge preconditioning.

    The weight is intentionally simple: dense molecular regions get values near
    one, while the surrounding vacuum remains near zero. This gives the charge
    controller a cheap real-space proxy for "where the molecule can respond",
    without introducing any H2-specific logic.
    """

    density = np.maximum(np.asarray(rho_charge_current, dtype=np.float64), 0.0)
    peak_density = float(np.max(density))
    if not np.isfinite(peak_density) or peak_density <= 0.0:
        return np.zeros_like(density)
    normalized_density = np.clip(density / peak_density, 0.0, 1.0)
    return np.asarray(np.sqrt(normalized_density), dtype=np.float64)


def _grid_preconditioner_scale(
    *,
    grid_point_count: int,
    reference_grid_point_count: int,
) -> float:
    if reference_grid_point_count <= 0:
        return 1.0
    return float(
        min(
            1.0,
            float(reference_grid_point_count) / max(float(grid_point_count), 1.0),
        )
    )


def _append_recent_charge_residual(
    history: tuple[np.ndarray, ...],
    residual: np.ndarray,
    *,
    max_length: int,
) -> tuple[np.ndarray, ...]:
    if max_length <= 0:
        return ()
    updated = tuple(np.asarray(field, dtype=np.float64) for field in history) + (
        np.asarray(residual, dtype=np.float64),
    )
    if len(updated) > max_length:
        updated = updated[-max_length:]
    return updated


def _weighted_inner(
    field_a: np.ndarray,
    field_b: np.ndarray,
    *,
    grid_geometry: GridGeometryLike,
) -> float:
    return float(
        integrate_field(
            np.asarray(field_a, dtype=np.float64) * np.asarray(field_b, dtype=np.float64),
            grid_geometry=grid_geometry,
        )
    )


def _principal_residual_mode(
    *,
    history: tuple[np.ndarray, ...],
    grid_geometry: GridGeometryLike,
) -> tuple[np.ndarray, float] | None:
    if len(history) < 2:
        return None
    sqrt_weights = np.sqrt(np.asarray(grid_geometry.cell_volumes, dtype=np.float64)).reshape(-1)
    matrix = np.stack(
        [np.asarray(field, dtype=np.float64).reshape(-1) * sqrt_weights for field in history],
        axis=0,
    )
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    total_energy = float(np.sum(singular_values**2, dtype=np.float64))
    if total_energy <= 1.0e-16:
        return None
    mode_weighted = np.asarray(right_vectors[0], dtype=np.float64).reshape(grid_geometry.spec.shape)
    mode = np.asarray(
        mode_weighted / np.maximum(np.sqrt(np.asarray(grid_geometry.cell_volumes, dtype=np.float64)), 1.0e-30),
        dtype=np.float64,
    )
    norm = _weighted_field_norm(mode, grid_geometry=grid_geometry)
    if norm <= 1.0e-16:
        return None
    mode = mode / norm
    explained_fraction = float((singular_values[0] ** 2) / total_energy)
    return mode, explained_fraction


def _modal_history_is_persistent(
    *,
    history: tuple[np.ndarray, ...],
    mode: np.ndarray,
    grid_geometry: GridGeometryLike,
    min_history: int,
) -> bool:
    if len(history) < max(int(min_history), 2):
        return False
    coefficients = [
        _weighted_inner(field, mode, grid_geometry=grid_geometry) for field in history
    ]
    nonzero = [value for value in coefficients if abs(value) > 1.0e-16]
    if len(nonzero) < max(int(min_history), 2):
        return False
    sign = 1.0 if nonzero[0] > 0.0 else -1.0
    return all(value * sign > 0.0 for value in nonzero)


def _modal_contraction_ratio(
    *,
    history: tuple[np.ndarray, ...],
    mode: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> float | None:
    if len(history) < 2:
        return None
    coefficients = [
        _weighted_inner(field, mode, grid_geometry=grid_geometry) for field in history
    ]
    ratios: list[float] = []
    for previous, current in zip(coefficients[:-1], coefficients[1:]):
        if abs(previous) <= 1.0e-16:
            continue
        ratio = abs(float(current / previous))
        if np.isfinite(ratio):
            ratios.append(ratio)
    if not ratios:
        return None
    return float(np.median(np.asarray(ratios, dtype=np.float64)))


def _limit_charge_trial_to_nonnegative(
    *,
    rho_charge_current: np.ndarray,
    rho_charge_trial: np.ndarray,
) -> tuple[np.ndarray, bool]:
    current = np.asarray(rho_charge_current, dtype=np.float64)
    trial = np.asarray(rho_charge_trial, dtype=np.float64)
    delta = np.asarray(trial - current, dtype=np.float64)
    negative_mask = delta < 0.0
    if not np.any(negative_mask):
        return trial, False
    denominator = -delta[negative_mask]
    numerator = current[negative_mask]
    if denominator.size == 0:
        return trial, False
    alpha_max = float(np.min(numerator / np.maximum(denominator, 1.0e-30)))
    if not np.isfinite(alpha_max) or alpha_max >= 1.0:
        return trial, False
    alpha_limited = max(0.0, 0.999 * alpha_max)
    limited = np.asarray(current + alpha_limited * delta, dtype=np.float64)
    return limited, True


def _is_closed_shell_singlet(occupations) -> bool:
    return bool(
        occupations.spin == 0
        and occupations.n_alpha == occupations.n_beta
        and occupations.n_alpha > 0
    )


def _channel_residuals(
    *,
    rho_up_current: np.ndarray,
    rho_down_current: np.ndarray,
    rho_up_output: np.ndarray,
    rho_down_output: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> ScfChannelResiduals:
    current_charge = build_total_density(
        rho_up=np.asarray(rho_up_current, dtype=np.float64),
        rho_down=np.asarray(rho_down_current, dtype=np.float64),
        grid_geometry=grid_geometry,
    )
    output_charge = build_total_density(
        rho_up=np.asarray(rho_up_output, dtype=np.float64),
        rho_down=np.asarray(rho_down_output, dtype=np.float64),
        grid_geometry=grid_geometry,
    )
    current_spin = np.asarray(rho_up_current - rho_down_current, dtype=np.float64)
    output_spin = np.asarray(rho_up_output - rho_down_output, dtype=np.float64)
    charge_residual = np.asarray(output_charge - current_charge, dtype=np.float64)
    spin_residual = np.asarray(output_spin - current_spin, dtype=np.float64)
    return ScfChannelResiduals(
        charge_residual=charge_residual,
        spin_residual=spin_residual,
        charge_residual_norm=_weighted_field_norm(
            charge_residual,
            grid_geometry=grid_geometry,
        ),
        spin_residual_norm=_weighted_field_norm(
            spin_residual,
            grid_geometry=grid_geometry,
        ),
    )


def _trigger_flags(
    *,
    signals: ScfControllerSignals,
    config: ScfControllerConfig,
    closed_shell_singlet: bool,
) -> tuple[bool, bool, tuple[str, ...]]:
    flags: list[str] = []
    hartree_dominated = (
        signals.hartree_share is not None
        and np.isfinite(signals.hartree_share)
        and signals.hartree_share >= config.hartree_share_trigger
    )
    residual_growing = (
        signals.density_residual_ratio is not None
        and np.isfinite(signals.density_residual_ratio)
        and signals.density_residual_ratio >= config.residual_ratio_trigger
    )
    residual_severe = (
        signals.density_residual_ratio is not None
        and np.isfinite(signals.density_residual_ratio)
        and signals.density_residual_ratio >= config.residual_ratio_severe_trigger
    )
    overlap_bad = (
        signals.occupied_orbital_overlap_abs is not None
        and np.isfinite(signals.occupied_orbital_overlap_abs)
        and signals.occupied_orbital_overlap_abs <= config.occupied_overlap_trigger
    )
    rotation_bad = (
        signals.lowest_subspace_rotation_max_angle_deg is not None
        and np.isfinite(signals.lowest_subspace_rotation_max_angle_deg)
        and signals.lowest_subspace_rotation_max_angle_deg >= config.subspace_rotation_trigger_deg
    )
    gap_small = (
        signals.lowest_gap_ha is not None
        and np.isfinite(signals.lowest_gap_ha)
        and signals.lowest_gap_ha <= config.lowest_gap_trigger_ha
    )
    if hartree_dominated:
        flags.append("hartree_dominated")
    if residual_growing:
        flags.append("residual_growing")
    if overlap_bad:
        flags.append("overlap_drop")
    if rotation_bad:
        flags.append("subspace_rotation")
    if gap_small:
        flags.append("small_gap")
    caution = bool(
        residual_severe
        or (hartree_dominated and residual_growing)
        or (residual_growing and (overlap_bad or rotation_bad))
        or (hartree_dominated and gap_small)
        or (closed_shell_singlet and rotation_bad and gap_small)
        or (overlap_bad and rotation_bad)
    )
    severe = bool(
        (residual_severe and (hartree_dominated or overlap_bad or rotation_bad))
        or (closed_shell_singlet and rotation_bad and gap_small)
        or (closed_shell_singlet and overlap_bad and rotation_bad)
    )
    return caution, severe, tuple(flags)


def propose_next_density(
    *,
    occupations,
    rho_up_current: np.ndarray,
    rho_down_current: np.ndarray,
    rho_up_output: np.ndarray,
    rho_down_output: np.ndarray,
    grid_geometry: GridGeometryLike,
    config: ScfControllerConfig,
    state: ScfControllerState,
    signals: ScfControllerSignals,
) -> ControllerStepResult:
    """Return the next density proposal using a reusable charge/spin controller.

    The controller runs in two stages:

    1. an optional opening phase for the first few closed-shell singlet steps
       on larger grids, where charge mixing is capped even before the usual
       Hartree/share precursor signals become reliable
    2. the regular Hartree-aware control branch, which reacts to residual-ratio,
       Hartree-share, overlap, rotation, and gap diagnostics
    """

    channel_residuals = _channel_residuals(
        rho_up_current=rho_up_current,
        rho_down_current=rho_down_current,
        rho_up_output=rho_up_output,
        rho_down_output=rho_down_output,
        grid_geometry=grid_geometry,
    )
    closed_shell_singlet = _is_closed_shell_singlet(occupations)
    caution, severe, flags = _trigger_flags(
        signals=signals,
        config=config,
        closed_shell_singlet=closed_shell_singlet,
    )
    current_charge_mixing = float(state.charge_mixing)
    current_spin_mixing = float(state.spin_mixing)
    charge_min_mixing = float(config.min_charge_mixing)
    charge_severe_mixing = float(config.severe_charge_mixing)
    if not closed_shell_singlet:
        charge_min_mixing = max(charge_min_mixing, 0.06)
        charge_severe_mixing = max(charge_severe_mixing, 0.04)
    grid_point_count = int(np.prod(grid_geometry.spec.shape))
    opening_large_grid_risk = bool(
        closed_shell_singlet and grid_point_count >= int(config.opening_grid_point_threshold)
    )
    opening_phase_active = bool(
        closed_shell_singlet and int(state.iteration_index) < int(config.opening_steps)
    )
    current_hartree_share = (
        float(signals.hartree_share)
        if signals.hartree_share is not None and np.isfinite(signals.hartree_share)
        else None
    )
    previous_hartree_share = (
        float(state.previous_hartree_share)
        if state.previous_hartree_share is not None and np.isfinite(state.previous_hartree_share)
        else None
    )
    recovery_pause_due_to_hartree_trend = bool(
        closed_shell_singlet
        and current_hartree_share is not None
        and previous_hartree_share is not None
        and current_hartree_share >= float(config.recovery_pause_hartree_share)
        and current_hartree_share
        >= previous_hartree_share + float(config.recovery_pause_hartree_rise)
    )

    if caution:
        next_charge_mixing = charge_severe_mixing if severe else charge_min_mixing
        next_spin_mixing = max(config.min_spin_mixing, 0.85 * config.baseline_spin_mixing)
        cautious_steps_remaining = int(config.cautious_hold_steps)
        stable_steps = 0
        step_flags = tuple(flags + ("charge_caution",))
    else:
        stable_steps = int(state.stable_steps + 1)
        cautious_steps_remaining = max(int(state.charge_cautious_steps_remaining) - 1, 0)
        next_charge_mixing = current_charge_mixing
        next_spin_mixing = current_spin_mixing
        step_flags_list = list(flags)
        if cautious_steps_remaining > 0:
            step_flags_list.append("charge_hold")
        elif recovery_pause_due_to_hartree_trend:
            step_flags_list.append("charge_recovery_paused")
        elif stable_steps >= int(config.stable_steps_to_release):
            if next_charge_mixing < config.baseline_charge_mixing:
                next_charge_mixing = min(
                    config.baseline_charge_mixing,
                    next_charge_mixing + config.charge_release_rate,
                )
                step_flags_list.append("charge_recovery")
            if next_spin_mixing < config.baseline_spin_mixing:
                next_spin_mixing = min(
                    config.baseline_spin_mixing,
                    next_spin_mixing + config.spin_release_rate,
                )
        if not step_flags_list:
            step_flags_list.append("stable")
        step_flags = tuple(step_flags_list)

    if opening_phase_active:
        # Larger monitor-grid singlet cases can blow up on 1->2 before the usual
        # precursor signals have fully developed, so the opening policy caps
        # charge mixing first and lets the regular Hartree-aware branch react
        # afterwards.
        opening_charge_cap = (
            float(config.opening_charge_mixing_large_risk)
            if opening_large_grid_risk
            else float(config.opening_charge_mixing)
        )
        next_charge_mixing = min(float(next_charge_mixing), opening_charge_cap)
        opening_flags = list(step_flags)
        opening_flags.append("opening_phase")
        if opening_large_grid_risk:
            opening_flags.append("opening_large_grid_risk")
        step_flags = tuple(opening_flags)

    rho_charge_current = build_total_density(
        rho_up=np.asarray(rho_up_current, dtype=np.float64),
        rho_down=np.asarray(rho_down_current, dtype=np.float64),
        grid_geometry=grid_geometry,
    )
    rho_charge_output = build_total_density(
        rho_up=np.asarray(rho_up_output, dtype=np.float64),
        rho_down=np.asarray(rho_down_output, dtype=np.float64),
        grid_geometry=grid_geometry,
    )
    rho_spin_current = np.asarray(rho_up_current - rho_down_current, dtype=np.float64)
    rho_spin_output = np.asarray(rho_up_output - rho_down_output, dtype=np.float64)
    updated_recent_charge_history = _append_recent_charge_residual(
        state.recent_charge_residual_history,
        channel_residuals.charge_residual,
        max_length=int(config.modal_history_length),
    )

    rho_charge_unbounded_trial = None
    if config.name == "generic_charge_spin_preconditioned":
        smoothed_charge_residual = _smooth_charge_residual(
            channel_residuals.charge_residual,
            smoothing_passes=config.preconditioned_smoothing_passes,
        )
        allow_high_frequency_boost = bool(
            not opening_phase_active
            and not caution
            and int(state.iteration_index)
            >= int(config.opening_steps + config.stable_steps_to_release)
            and int(stable_steps) >= int(config.stable_steps_to_release)
            and current_hartree_share is not None
            and current_hartree_share < float(config.recovery_pause_hartree_share)
            and not recovery_pause_due_to_hartree_trend
            and "hartree_dominated" not in step_flags
            and "overlap_drop" not in step_flags
            and "subspace_rotation" not in step_flags
            and "small_gap" not in step_flags
            and "charge_recovery_paused" not in step_flags
        )
        if allow_high_frequency_boost:
            grid_preconditioner_scale = _grid_preconditioner_scale(
                grid_point_count=grid_point_count,
                reference_grid_point_count=int(
                    config.preconditioned_grid_point_boost_threshold
                ),
            )
            peak_charge_mixing = float(
                min(
                    config.baseline_charge_mixing,
                    max(
                        next_charge_mixing,
                        config.preconditioned_high_frequency_mixing,
                    ),
                )
            )
            support_weight = _density_support_weight(rho_charge_current)
            effective_charge_residual = np.asarray(
                (1.0 - grid_preconditioner_scale * support_weight)
                * smoothed_charge_residual
                + (grid_preconditioner_scale * support_weight)
                * channel_residuals.charge_residual,
                dtype=np.float64,
            )
            effective_charge_mixing = np.asarray(
                next_charge_mixing
                + (grid_preconditioner_scale * support_weight)
                * (peak_charge_mixing - next_charge_mixing),
                dtype=np.float64,
            )
            rho_charge_trial = np.asarray(
                rho_charge_current + effective_charge_mixing * effective_charge_residual,
                dtype=np.float64,
            )
            modal_mode_result = _principal_residual_mode(
                history=updated_recent_charge_history,
                grid_geometry=grid_geometry,
            )
            if (
                modal_mode_result is not None
                and modal_mode_result[1] >= float(config.modal_min_explained_fraction)
            ):
                modal_mode, _ = modal_mode_result
                modal_boost_mixing = float(config.modal_boost_mixing)
                if _modal_history_is_persistent(
                    history=updated_recent_charge_history,
                    mode=modal_mode,
                    grid_geometry=grid_geometry,
                    min_history=int(config.modal_persistent_min_history),
                ) and grid_point_count >= int(config.modal_persistent_grid_point_threshold):
                    modal_ratio = _modal_contraction_ratio(
                        history=updated_recent_charge_history,
                        mode=modal_mode,
                        grid_geometry=grid_geometry,
                    )
                    if modal_ratio is not None:
                        ratio_floor = float(config.low_rank_modal_ratio_floor)
                        ratio_ceiling = float(config.low_rank_modal_ratio_ceiling)
                        if ratio_ceiling > ratio_floor:
                            low_rank_scale = float(
                                np.clip(
                                    (modal_ratio - ratio_floor)
                                    / (ratio_ceiling - ratio_floor),
                                    0.0,
                                    1.0,
                                )
                            )
                        else:
                            low_rank_scale = 0.0
                        modal_boost_mixing += (
                            float(config.modal_persistent_extra_mixing) * low_rank_scale
                        )
                        step_flags = tuple(
                            step_flags + ("modal_persistent", "low_rank_modal_preconditioner")
                        )
                modal_coefficient = _weighted_inner(
                    channel_residuals.charge_residual,
                    modal_mode,
                    grid_geometry=grid_geometry,
                )
                modal_boost = np.asarray(
                    modal_boost_mixing * modal_coefficient * modal_mode,
                    dtype=np.float64,
                )
                rho_charge_trial = np.asarray(rho_charge_trial + modal_boost, dtype=np.float64)
                step_flags = tuple(step_flags + ("modal_boost",))
        else:
            rho_charge_trial = np.asarray(
                rho_charge_current + next_charge_mixing * channel_residuals.charge_residual,
                dtype=np.float64,
            )
    else:
        rho_charge_trial = (
            (1.0 - next_charge_mixing) * rho_charge_current
            + next_charge_mixing * rho_charge_output
        )
    rho_charge_unbounded_trial = np.asarray(rho_charge_trial, dtype=np.float64)
    rho_charge_trial, charge_trial_limited = _limit_charge_trial_to_nonnegative(
        rho_charge_current=rho_charge_current,
        rho_charge_trial=rho_charge_unbounded_trial,
    )
    if charge_trial_limited:
        step_flags = tuple(step_flags + ("charge_trial_limited",))
    rho_charge_next = _renormalize_density(
        rho_charge_trial,
        target_electrons=float(occupations.total_electrons),
        grid_geometry=grid_geometry,
    )
    if closed_shell_singlet:
        rho_spin_next = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    else:
        rho_spin_next = np.asarray(
            (1.0 - next_spin_mixing) * rho_spin_current
            + next_spin_mixing * rho_spin_output,
            dtype=np.float64,
        )

    rho_up_next = np.maximum(0.5 * (rho_charge_next + rho_spin_next), 0.0)
    rho_down_next = np.maximum(0.5 * (rho_charge_next - rho_spin_next), 0.0)
    rho_up_next = _renormalize_density(
        rho_up_next,
        target_electrons=float(occupations.n_alpha),
        grid_geometry=grid_geometry,
    )
    rho_down_next = _renormalize_density(
        rho_down_next,
        target_electrons=float(occupations.n_beta),
        grid_geometry=grid_geometry,
    )
    rho_charge_postclip_next = build_total_density(
        rho_up=np.asarray(rho_up_next, dtype=np.float64),
        rho_down=np.asarray(rho_down_next, dtype=np.float64),
        grid_geometry=grid_geometry,
    )

    next_state = ScfControllerState(
        charge_mixing=float(next_charge_mixing),
        spin_mixing=float(next_spin_mixing),
        charge_cautious_steps_remaining=int(cautious_steps_remaining),
        stable_steps=int(stable_steps),
        iteration_index=int(state.iteration_index + 1),
        last_flags=step_flags,
        previous_hartree_share=current_hartree_share,
        recent_charge_residual_history=updated_recent_charge_history,
    )
    return ControllerStepResult(
        rho_up_next=np.asarray(rho_up_next, dtype=np.float64),
        rho_down_next=np.asarray(rho_down_next, dtype=np.float64),
        channel_residuals=channel_residuals,
        signals=signals,
        charge_mixing=float(next_charge_mixing),
        spin_mixing=float(next_spin_mixing),
        state=next_state,
        flags=step_flags,
        rho_charge_unbounded_trial=np.asarray(rho_charge_unbounded_trial, dtype=np.float64),
        rho_charge_trial=np.asarray(rho_charge_trial, dtype=np.float64),
        rho_charge_preclip_next=np.asarray(rho_charge_next, dtype=np.float64),
        rho_charge_postclip_next=np.asarray(rho_charge_postclip_next, dtype=np.float64),
    )
