"""Unit tests for the generic charge/spin SCF controller."""

from dataclasses import replace

import numpy as np

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ops import integrate_field
from isogrid.scf import build_h2_initial_density_guess
from isogrid.scf import resolve_h2_spin_occupations
from isogrid.scf.controller import ScfControllerConfig
from isogrid.scf.controller import ScfControllerSignals
from isogrid.scf.controller import ScfControllerState
from isogrid.scf.controller import _append_recent_charge_residual
from isogrid.scf.controller import _build_preconditioned_base_charge_trial
from isogrid.scf.controller import _channel_residuals
from isogrid.scf.controller import _dominant_residual_modes
from isogrid.scf.controller import _prepare_low_rank_charge_variable
from isogrid.scf.controller import _smooth_charge_residual
from isogrid.scf.controller import _weighted_inner
from isogrid.scf.controller import propose_next_density


def _small_grid_geometry():
    return build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(9, 9, 11),
        box_half_extents=(6.0, 6.0, 8.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )


def _xlarge_grid_geometry():
    return build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(13, 13, 15),
        box_half_extents=(8.0, 8.0, 10.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )


def _xxlarge_grid_geometry():
    return build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(15, 15, 17),
        box_half_extents=(9.0, 9.0, 11.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )


def test_generic_charge_spin_controller_preserves_closed_shell_symmetry() -> None:
    grid_geometry = _small_grid_geometry()
    occupations = resolve_h2_spin_occupations("singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    rho_up_out = np.asarray(1.15 * rho_up, dtype=np.float64)
    rho_down_out = np.asarray(0.85 * rho_down, dtype=np.float64)

    result = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=rho_up_out,
        rho_down_output=rho_down_out,
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin(),
        state=ScfControllerState.initial(),
        signals=ScfControllerSignals(
            density_residual_ratio=1.08,
            hartree_share=0.81,
            occupied_orbital_overlap_abs=0.12,
            lowest_subspace_rotation_max_angle_deg=72.0,
            lowest_gap_ha=0.08,
        ),
    )

    assert np.allclose(result.rho_up_next, result.rho_down_next)
    assert np.isclose(
        integrate_field(result.rho_up_next, grid_geometry=grid_geometry),
        occupations.n_alpha,
    )
    assert np.isclose(
        integrate_field(result.rho_down_next, grid_geometry=grid_geometry),
        occupations.n_beta,
    )
    assert result.charge_mixing < result.spin_mixing
    assert result.state.charge_cautious_steps_remaining > 0


def test_generic_charge_spin_controller_recovers_charge_mixing_after_stable_steps() -> None:
    grid_geometry = _small_grid_geometry()
    occupations = resolve_h2_spin_occupations("triplet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    state = ScfControllerState(
        charge_mixing=0.06,
        spin_mixing=0.14,
        charge_cautious_steps_remaining=0,
        stable_steps=3,
        iteration_index=3,
        last_flags=("recovering",),
        previous_hartree_share=0.18,
    )

    result = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(0.9 * rho_up, dtype=np.float64),
        rho_down_output=np.asarray(1.05 * rho_down, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin(),
        state=state,
        signals=ScfControllerSignals(
            density_residual_ratio=0.85,
            hartree_share=0.22,
            occupied_orbital_overlap_abs=0.995,
            lowest_subspace_rotation_max_angle_deg=2.0,
            lowest_gap_ha=0.16,
        ),
    )

    assert result.charge_mixing > state.charge_mixing
    assert result.state.stable_steps > state.stable_steps
    assert "charge_recovery" in result.flags


def test_generic_charge_spin_controller_uses_grid_risk_opening_for_xlarge_singlet() -> None:
    grid_geometry = _xlarge_grid_geometry()
    occupations = resolve_h2_spin_occupations("singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )

    result = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(1.02 * rho_up, dtype=np.float64),
        rho_down_output=np.asarray(0.98 * rho_down, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin(),
        state=ScfControllerState.initial(),
        signals=ScfControllerSignals(
            density_residual_ratio=None,
            hartree_share=0.01,
            occupied_orbital_overlap_abs=None,
            lowest_subspace_rotation_max_angle_deg=None,
            lowest_gap_ha=None,
        ),
    )

    assert result.charge_mixing <= 0.01
    assert "opening_phase" in result.flags


def test_generic_charge_spin_controller_pauses_recovery_when_hartree_share_rises() -> None:
    grid_geometry = _xlarge_grid_geometry()
    occupations = resolve_h2_spin_occupations("singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    state = ScfControllerState(
        charge_mixing=0.015,
        spin_mixing=0.18,
        charge_cautious_steps_remaining=0,
        stable_steps=4,
        iteration_index=9,
        last_flags=("charge_recovery",),
        previous_hartree_share=0.09,
    )

    result = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(1.001 * rho_up, dtype=np.float64),
        rho_down_output=np.asarray(0.999 * rho_down, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin(),
        state=state,
        signals=ScfControllerSignals(
            density_residual_ratio=0.976,
            hartree_share=0.31,
            occupied_orbital_overlap_abs=0.9994,
            lowest_subspace_rotation_max_angle_deg=1.95,
            lowest_gap_ha=0.18,
        ),
    )

    assert result.charge_mixing <= state.charge_mixing
    assert "charge_recovery" not in result.flags


def test_preconditioned_controller_base_charge_trial_uses_smoothed_residual_when_low_rank_is_disabled() -> None:
    grid_geometry = _xxlarge_grid_geometry()
    occupations = resolve_h2_spin_occupations("singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    bump = np.zeros_like(rho_up)
    center = tuple(length // 2 for length in bump.shape)
    bump[center] = 0.08 * float(np.max(rho_up))
    config = replace(
        ScfControllerConfig.generic_charge_spin_preconditioned(),
        modal_history_length=0,
    )
    state = ScfControllerState(
        charge_mixing=0.02,
        spin_mixing=0.18,
        charge_cautious_steps_remaining=0,
        stable_steps=6,
        iteration_index=10,
        last_flags=("charge_recovery",),
        previous_hartree_share=0.02,
    )
    signals = ScfControllerSignals(
        density_residual_ratio=0.98,
        hartree_share=0.03,
        occupied_orbital_overlap_abs=0.999,
        lowest_subspace_rotation_max_angle_deg=0.5,
        lowest_gap_ha=0.2,
    )

    result = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(rho_up + bump, dtype=np.float64),
        rho_down_output=np.asarray(rho_down + bump, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=config,
        state=state,
        signals=signals,
    )

    charge_residual = np.asarray(
        (result.channel_residuals.charge_residual),
        dtype=np.float64,
    )
    smoothed_charge_residual = _smooth_charge_residual(
        charge_residual,
        smoothing_passes=config.preconditioned_smoothing_passes,
    )
    rho_charge_current = np.asarray(rho_up + rho_down, dtype=np.float64)
    expected_trial = _build_preconditioned_base_charge_trial(
        rho_charge_current=rho_charge_current,
        charge_residual=charge_residual,
        next_charge_mixing=float(result.charge_mixing),
        config=config,
        grid_geometry=grid_geometry,
    )

    assert np.allclose(expected_trial.smoothed_charge_residual, smoothed_charge_residual)
    assert np.allclose(result.rho_charge_unbounded_trial, expected_trial.rho_charge_trial)
    assert "modal_boost" not in result.flags


def test_preconditioned_controller_boosts_local_charge_update_without_breaking_symmetry() -> None:
    grid_geometry = _xlarge_grid_geometry()
    occupations = resolve_h2_spin_occupations("singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    bump = np.zeros_like(rho_up)
    center = tuple(length // 2 for length in bump.shape)
    bump[center] = 0.05 * float(np.max(rho_up))
    state = ScfControllerState(
        charge_mixing=0.005,
        spin_mixing=0.18,
        charge_cautious_steps_remaining=0,
        stable_steps=4,
        iteration_index=8,
        last_flags=("stable",),
        previous_hartree_share=0.02,
    )
    signals = ScfControllerSignals(
        density_residual_ratio=0.99,
        hartree_share=0.03,
        occupied_orbital_overlap_abs=0.999,
        lowest_subspace_rotation_max_angle_deg=0.5,
        lowest_gap_ha=0.2,
    )

    generic = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(rho_up + bump, dtype=np.float64),
        rho_down_output=np.asarray(rho_down + bump, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin(),
        state=state,
        signals=signals,
    )
    preconditioned = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(rho_up + bump, dtype=np.float64),
        rho_down_output=np.asarray(rho_down + bump, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin_preconditioned(),
        state=state,
        signals=signals,
    )

    generic_step = np.linalg.norm(generic.rho_up_next - rho_up)
    preconditioned_step = np.linalg.norm(preconditioned.rho_up_next - rho_up)

    assert np.allclose(preconditioned.rho_up_next, preconditioned.rho_down_next)
    assert preconditioned_step > generic_step


def test_preconditioned_controller_disables_local_boost_when_hartree_dominated() -> None:
    grid_geometry = _xlarge_grid_geometry()
    occupations = resolve_h2_spin_occupations("singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    bump = np.zeros_like(rho_up)
    center = tuple(length // 2 for length in bump.shape)
    bump[center] = 0.05 * float(np.max(rho_up))
    state = ScfControllerState(
        charge_mixing=0.015,
        spin_mixing=0.18,
        charge_cautious_steps_remaining=0,
        stable_steps=4,
        iteration_index=5,
        last_flags=("charge_recovery_paused",),
        previous_hartree_share=0.75,
    )
    signals = ScfControllerSignals(
        density_residual_ratio=0.92,
        hartree_share=0.84,
        occupied_orbital_overlap_abs=0.99,
        lowest_subspace_rotation_max_angle_deg=5.0,
        lowest_gap_ha=0.20,
    )

    generic = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(rho_up + bump, dtype=np.float64),
        rho_down_output=np.asarray(rho_down + bump, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin(),
        state=state,
        signals=signals,
    )
    preconditioned = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(rho_up + bump, dtype=np.float64),
        rho_down_output=np.asarray(rho_down + bump, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin_preconditioned(),
        state=state,
        signals=signals,
    )

    assert "modal_boost" not in preconditioned.flags
    assert "low_rank_modal_preconditioner" not in preconditioned.flags
    assert np.max(np.abs(preconditioned.rho_up_next - generic.rho_up_next)) < 3.0e-4
    assert np.max(np.abs(preconditioned.rho_down_next - generic.rho_down_next)) < 3.0e-4


def test_preconditioned_controller_adds_modal_boost_on_persistent_charge_mode() -> None:
    grid_geometry = _xlarge_grid_geometry()
    occupations = resolve_h2_spin_occupations("singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    bump = np.zeros_like(rho_up)
    center = tuple(length // 2 for length in bump.shape)
    bump[center] = 0.04 * float(np.max(rho_up))
    modal_history = tuple(np.asarray(scale * bump, dtype=np.float64) for scale in (1.0, 0.98, 0.96))
    state_without_history = ScfControllerState(
        charge_mixing=0.015,
        spin_mixing=0.18,
        charge_cautious_steps_remaining=0,
        stable_steps=5,
        iteration_index=9,
        last_flags=("charge_recovery",),
        previous_hartree_share=0.03,
    )
    state_with_history = ScfControllerState(
        charge_mixing=0.015,
        spin_mixing=0.18,
        charge_cautious_steps_remaining=0,
        stable_steps=5,
        iteration_index=9,
        last_flags=("charge_recovery",),
        previous_hartree_share=0.03,
        recent_charge_residual_history=modal_history,
    )
    signals = ScfControllerSignals(
        density_residual_ratio=0.985,
        hartree_share=0.04,
        occupied_orbital_overlap_abs=0.999,
        lowest_subspace_rotation_max_angle_deg=0.4,
        lowest_gap_ha=0.22,
    )

    without_history = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(rho_up + bump, dtype=np.float64),
        rho_down_output=np.asarray(rho_down + bump, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin_preconditioned(),
        state=state_without_history,
        signals=signals,
    )
    with_history = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(rho_up + bump, dtype=np.float64),
        rho_down_output=np.asarray(rho_down + bump, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin_preconditioned(),
        state=state_with_history,
        signals=signals,
    )

    without_history_step = np.linalg.norm(without_history.rho_up_next - rho_up)
    with_history_step = np.linalg.norm(with_history.rho_up_next - rho_up)

    assert with_history_step > without_history_step
    assert "modal_boost" in with_history.flags


def test_preconditioned_controller_disables_modal_boost_during_charge_caution() -> None:
    grid_geometry = _xlarge_grid_geometry()
    occupations = resolve_h2_spin_occupations("singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    bump = np.zeros_like(rho_up)
    center = tuple(length // 2 for length in bump.shape)
    bump[center] = 0.04 * float(np.max(rho_up))
    modal_history = tuple(np.asarray(scale * bump, dtype=np.float64) for scale in (1.0, 0.98, 0.96))
    state = ScfControllerState(
        charge_mixing=0.015,
        spin_mixing=0.18,
        charge_cautious_steps_remaining=0,
        stable_steps=5,
        iteration_index=9,
        last_flags=("charge_recovery",),
        previous_hartree_share=0.70,
        recent_charge_residual_history=modal_history,
    )
    signals = ScfControllerSignals(
        density_residual_ratio=1.08,
        hartree_share=0.82,
        occupied_orbital_overlap_abs=0.15,
        lowest_subspace_rotation_max_angle_deg=75.0,
        lowest_gap_ha=0.05,
    )

    with_history = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(rho_up + bump, dtype=np.float64),
        rho_down_output=np.asarray(rho_down + bump, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin_preconditioned(),
        state=state,
        signals=signals,
    )
    without_history = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(rho_up + bump, dtype=np.float64),
        rho_down_output=np.asarray(rho_down + bump, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin_preconditioned(),
        state=ScfControllerState(
            charge_mixing=0.015,
            spin_mixing=0.18,
            charge_cautious_steps_remaining=0,
            stable_steps=5,
            iteration_index=9,
            last_flags=("charge_recovery",),
            previous_hartree_share=0.70,
        ),
        signals=signals,
    )

    assert "modal_boost" not in with_history.flags
    assert np.allclose(with_history.rho_up_next, without_history.rho_up_next)
    assert np.allclose(with_history.rho_down_next, without_history.rho_down_next)


def test_preconditioned_controller_strengthens_modal_boost_for_persistent_mode() -> None:
    grid_geometry = _xxlarge_grid_geometry()
    occupations = resolve_h2_spin_occupations("singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    bump = np.zeros_like(rho_up)
    center = tuple(length // 2 for length in bump.shape)
    bump[center] = 0.04 * float(np.max(rho_up))
    coherent_history = tuple(np.asarray(scale * bump, dtype=np.float64) for scale in (1.00, 0.99, 0.98, 0.97))
    incoherent_history = (
        np.asarray(1.00 * bump, dtype=np.float64),
        np.asarray(-0.95 * bump, dtype=np.float64),
        np.asarray(0.90 * bump, dtype=np.float64),
        np.asarray(-0.85 * bump, dtype=np.float64),
    )
    common_kwargs = dict(
        charge_mixing=0.015,
        spin_mixing=0.18,
        charge_cautious_steps_remaining=0,
        stable_steps=6,
        iteration_index=10,
        last_flags=("charge_recovery",),
        previous_hartree_share=0.03,
    )
    signals = ScfControllerSignals(
        density_residual_ratio=0.985,
        hartree_share=0.04,
        occupied_orbital_overlap_abs=0.999,
        lowest_subspace_rotation_max_angle_deg=0.4,
        lowest_gap_ha=0.22,
    )

    coherent = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(rho_up + bump, dtype=np.float64),
        rho_down_output=np.asarray(rho_down + bump, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin_preconditioned(),
        state=ScfControllerState(
            recent_charge_residual_history=coherent_history,
            **common_kwargs,
        ),
        signals=signals,
    )
    incoherent = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(rho_up + bump, dtype=np.float64),
        rho_down_output=np.asarray(rho_down + bump, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin_preconditioned(),
        state=ScfControllerState(
            recent_charge_residual_history=incoherent_history,
            **common_kwargs,
        ),
        signals=signals,
    )

    assert "modal_boost" in coherent.flags
    assert "modal_persistent" in coherent.flags
    assert "low_rank_modal_preconditioner" in coherent.flags
    assert "low_rank_modal_preconditioner" not in incoherent.flags


def test_preconditioned_controller_strengthens_low_rank_update_for_weaker_modal_contraction() -> None:
    grid_geometry = _xxlarge_grid_geometry()
    occupations = resolve_h2_spin_occupations("singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    bump = np.zeros_like(rho_up)
    center = tuple(length // 2 for length in bump.shape)
    bump[center] = 0.04 * float(np.max(rho_up))
    weakly_contracting_history = tuple(
        np.asarray(scale * bump, dtype=np.float64) for scale in (1.00, 0.995, 0.990, 0.985)
    )
    strongly_contracting_history = tuple(
        np.asarray(scale * bump, dtype=np.float64) for scale in (1.00, 0.80, 0.64, 0.51)
    )
    common_kwargs = dict(
        charge_mixing=0.015,
        spin_mixing=0.18,
        charge_cautious_steps_remaining=0,
        stable_steps=6,
        iteration_index=10,
        last_flags=("charge_recovery",),
        previous_hartree_share=0.03,
    )
    signals = ScfControllerSignals(
        density_residual_ratio=0.985,
        hartree_share=0.04,
        occupied_orbital_overlap_abs=0.999,
        lowest_subspace_rotation_max_angle_deg=0.4,
        lowest_gap_ha=0.22,
    )

    weak = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(rho_up + bump, dtype=np.float64),
        rho_down_output=np.asarray(rho_down + bump, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin_preconditioned(),
        state=ScfControllerState(
            recent_charge_residual_history=weakly_contracting_history,
            **common_kwargs,
        ),
        signals=signals,
    )
    strong = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=np.asarray(rho_up + bump, dtype=np.float64),
        rho_down_output=np.asarray(rho_down + bump, dtype=np.float64),
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin_preconditioned(),
        state=ScfControllerState(
            recent_charge_residual_history=strongly_contracting_history,
            **common_kwargs,
        ),
        signals=signals,
    )

    weak_step = np.linalg.norm(weak.rho_up_next - rho_up)
    strong_step = np.linalg.norm(strong.rho_up_next - rho_up)

    assert weak_step > strong_step
    assert "low_rank_modal_preconditioner" in weak.flags


def test_preconditioned_controller_rank2_captures_secondary_modal_direction() -> None:
    grid_geometry = _xxlarge_grid_geometry()
    occupations = resolve_h2_spin_occupations("singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    mode1 = np.zeros_like(rho_up)
    mode2 = np.zeros_like(rho_up)
    center = tuple(length // 2 for length in mode1.shape)
    mode1[center] = 0.03 * float(np.max(rho_up))
    mode2[(center[0] - 2, center[1], center[2])] = 0.03 * float(np.max(rho_up))
    residual_history = (
        np.asarray(mode1 + mode2, dtype=np.float64),
        np.asarray(mode1 - mode2, dtype=np.float64),
        np.asarray(-mode1 + mode2, dtype=np.float64),
        np.asarray(-mode1 - mode2, dtype=np.float64),
    )
    signals = ScfControllerSignals(
        density_residual_ratio=0.99,
        hartree_share=0.04,
        occupied_orbital_overlap_abs=0.999,
        lowest_subspace_rotation_max_angle_deg=0.4,
        lowest_gap_ha=0.22,
    )
    state = ScfControllerState(
        charge_mixing=0.015,
        spin_mixing=0.18,
        charge_cautious_steps_remaining=0,
        stable_steps=6,
        iteration_index=10,
        last_flags=("charge_recovery",),
        previous_hartree_share=0.03,
        recent_charge_residual_history=residual_history,
    )
    rho_up_out = np.asarray(rho_up + mode2, dtype=np.float64)
    rho_down_out = np.asarray(rho_down + mode2, dtype=np.float64)
    rank2_config = ScfControllerConfig.generic_charge_spin_preconditioned()
    rank2_config = replace(
        rank2_config,
        modal_min_explained_fraction=0.85,
        low_rank_secondary_min_explained_fraction=0.10,
        preconditioned_smoothing_passes=0,
    )
    rank1_config = replace(
        rank2_config,
        low_rank_mode_count=1,
    )

    rank1 = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=rho_up_out,
        rho_down_output=rho_down_out,
        grid_geometry=grid_geometry,
        config=rank1_config,
        state=state,
        signals=signals,
    )
    rank2 = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=rho_up_out,
        rho_down_output=rho_down_out,
        grid_geometry=grid_geometry,
        config=rank2_config,
        state=state,
        signals=signals,
    )

    channel_residuals = _channel_residuals(
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=rho_up_out,
        rho_down_output=rho_down_out,
        grid_geometry=grid_geometry,
    )
    low_rank_history, _ = _prepare_low_rank_charge_variable(
        history=_append_recent_charge_residual(
            residual_history,
            channel_residuals.charge_residual,
            max_length=rank2_config.modal_history_length,
        ),
        current_residual=channel_residuals.charge_residual,
        smoothing_passes=rank2_config.preconditioned_smoothing_passes,
    )
    dominant_modes = _dominant_residual_modes(
        history=low_rank_history,
        grid_geometry=grid_geometry,
        max_modes=2,
    )
    assert len(dominant_modes) == 2
    secondary_mode, _ = dominant_modes[1]
    current_charge = np.asarray(rho_up + rho_down, dtype=np.float64)
    rank1_update = rank1.rho_charge_unbounded_trial - current_charge
    rank2_update = rank2.rho_charge_unbounded_trial - current_charge
    rank1_secondary_projection = abs(
        _weighted_inner(rank1_update, secondary_mode, grid_geometry=grid_geometry)
    )
    rank2_secondary_projection = abs(
        _weighted_inner(rank2_update, secondary_mode, grid_geometry=grid_geometry)
    )

    assert rank2_secondary_projection > rank1_secondary_projection
    assert "rank2_modal_boost" in rank2.flags


def test_preconditioned_controller_rank3_captures_tertiary_modal_direction() -> None:
    grid_geometry = _xxlarge_grid_geometry()
    occupations = resolve_h2_spin_occupations("singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    mode1 = np.zeros_like(rho_up)
    mode2 = np.zeros_like(rho_up)
    mode3 = np.zeros_like(rho_up)
    center = tuple(length // 2 for length in mode1.shape)
    peak = 0.02 * float(np.max(rho_up))
    mode1[center] = peak
    mode2[(center[0] - 2, center[1], center[2])] = peak
    mode3[(center[0], center[1] + 2, center[2])] = peak
    residual_history = (
        np.asarray(mode1 + mode2 + mode3, dtype=np.float64),
        np.asarray(mode1 - mode2 + mode3, dtype=np.float64),
        np.asarray(-mode1 + mode2 + mode3, dtype=np.float64),
        np.asarray(-mode1 - mode2 + mode3, dtype=np.float64),
    )
    signals = ScfControllerSignals(
        density_residual_ratio=0.99,
        hartree_share=0.04,
        occupied_orbital_overlap_abs=0.999,
        lowest_subspace_rotation_max_angle_deg=0.4,
        lowest_gap_ha=0.22,
    )
    state = ScfControllerState(
        charge_mixing=0.015,
        spin_mixing=0.18,
        charge_cautious_steps_remaining=0,
        stable_steps=6,
        iteration_index=10,
        last_flags=("charge_recovery",),
        previous_hartree_share=0.03,
        recent_charge_residual_history=residual_history,
    )
    rho_up_out = np.asarray(rho_up + mode3, dtype=np.float64)
    rho_down_out = np.asarray(rho_down + mode3, dtype=np.float64)
    rank3_config = replace(
        ScfControllerConfig.generic_charge_spin_preconditioned(),
        modal_min_explained_fraction=0.50,
        low_rank_secondary_min_explained_fraction=0.05,
        low_rank_mode_count=3,
        preconditioned_smoothing_passes=0,
    )
    rank1_config = replace(rank3_config, low_rank_mode_count=1)

    rank1 = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=rho_up_out,
        rho_down_output=rho_down_out,
        grid_geometry=grid_geometry,
        config=rank1_config,
        state=state,
        signals=signals,
    )
    rank3 = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=rho_up_out,
        rho_down_output=rho_down_out,
        grid_geometry=grid_geometry,
        config=rank3_config,
        state=state,
        signals=signals,
    )

    channel_residuals = _channel_residuals(
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=rho_up_out,
        rho_down_output=rho_down_out,
        grid_geometry=grid_geometry,
    )
    low_rank_history, _ = _prepare_low_rank_charge_variable(
        history=_append_recent_charge_residual(
            residual_history,
            channel_residuals.charge_residual,
            max_length=rank3_config.modal_history_length,
        ),
        current_residual=channel_residuals.charge_residual,
        smoothing_passes=rank3_config.preconditioned_smoothing_passes,
    )
    dominant_modes = _dominant_residual_modes(
        history=low_rank_history,
        grid_geometry=grid_geometry,
        max_modes=3,
    )
    assert len(dominant_modes) == 3
    tertiary_mode, _ = dominant_modes[2]
    current_charge = np.asarray(rho_up + rho_down, dtype=np.float64)
    rank1_update = rank1.rho_charge_unbounded_trial - current_charge
    rank3_update = rank3.rho_charge_unbounded_trial - current_charge
    rank1_tertiary_projection = abs(
        _weighted_inner(rank1_update, tertiary_mode, grid_geometry=grid_geometry)
    )
    rank3_tertiary_projection = abs(
        _weighted_inner(rank3_update, tertiary_mode, grid_geometry=grid_geometry)
    )

    assert rank3_tertiary_projection > rank1_tertiary_projection
    assert "rank3_subspace_boost" in rank3.flags


def test_preconditioned_controller_low_rank_variable_uses_smoothed_charge_residual() -> None:
    grid_geometry = _xxlarge_grid_geometry()
    residual = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    residual[7, 7, 8] = 1.0
    residual[8, 7, 8] = -1.0
    history = (
        np.asarray(residual, dtype=np.float64),
        np.asarray(-residual, dtype=np.float64),
    )

    prepared_history, prepared_current = _prepare_low_rank_charge_variable(
        history=history,
        current_residual=residual,
        smoothing_passes=3,
    )

    assert len(prepared_history) == 2
    assert np.allclose(
        prepared_current,
        _smooth_charge_residual(residual, smoothing_passes=3),
    )
    assert np.allclose(
        prepared_history[0],
        _smooth_charge_residual(history[0], smoothing_passes=3),
    )


def test_preconditioned_controller_limits_negative_charge_trial_before_clipping() -> None:
    grid_geometry = _xxlarge_grid_geometry()
    occupations = resolve_h2_spin_occupations("singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down, _, _ = build_h2_initial_density_guess(
        occupations=occupations,
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    center = tuple(length // 2 for length in rho_up.shape)
    depletion = np.zeros_like(rho_up)
    depletion[center] = 5.0 * float(rho_up[center])
    rho_up_out = np.asarray(np.maximum(rho_up - depletion, 0.0), dtype=np.float64)
    rho_down_out = np.asarray(np.maximum(rho_down - depletion, 0.0), dtype=np.float64)
    state = ScfControllerState(
        charge_mixing=0.045,
        spin_mixing=0.18,
        charge_cautious_steps_remaining=0,
        stable_steps=6,
        iteration_index=10,
        last_flags=("charge_recovery",),
        previous_hartree_share=0.03,
        recent_charge_residual_history=tuple(
            np.asarray(scale * (rho_up_out + rho_down_out - rho_up - rho_down), dtype=np.float64)
            for scale in (1.0, 0.99, 0.98, 0.97)
        ),
    )
    signals = ScfControllerSignals(
        density_residual_ratio=0.985,
        hartree_share=0.04,
        occupied_orbital_overlap_abs=0.999,
        lowest_subspace_rotation_max_angle_deg=0.4,
        lowest_gap_ha=0.22,
    )

    result = propose_next_density(
        occupations=occupations,
        rho_up_current=rho_up,
        rho_down_current=rho_down,
        rho_up_output=rho_up_out,
        rho_down_output=rho_down_out,
        grid_geometry=grid_geometry,
        config=ScfControllerConfig.generic_charge_spin_preconditioned(),
        state=state,
        signals=signals,
    )

    assert np.min(result.rho_charge_unbounded_trial) < 0.0
    assert np.min(result.rho_charge_trial) >= -1.0e-14
    assert "charge_trial_limited" in result.flags
