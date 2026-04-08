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
    """Persistent controller state carried across SCF iterations."""

    charge_mixing: float
    spin_mixing: float
    charge_cautious_steps_remaining: int
    stable_steps: int
    last_flags: tuple[str, ...]

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
            last_flags=(),
        )


@dataclass(frozen=True)
class ScfControllerConfig:
    """Static controller configuration for one SCF route."""

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
        or (overlap_bad and rotation_bad)
    )
    severe = bool(
        (residual_severe and (hartree_dominated or overlap_bad or rotation_bad))
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
    """Return the next density proposal using a reusable charge/spin controller."""

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

    rho_charge_next = _renormalize_density(
        (1.0 - next_charge_mixing) * rho_charge_current
        + next_charge_mixing * rho_charge_output,
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

    next_state = ScfControllerState(
        charge_mixing=float(next_charge_mixing),
        spin_mixing=float(next_spin_mixing),
        charge_cautious_steps_remaining=int(cautious_steps_remaining),
        stable_steps=int(stable_steps),
        last_flags=step_flags,
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
    )
