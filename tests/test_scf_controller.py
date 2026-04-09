"""Unit tests for the generic charge/spin SCF controller."""

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

    assert np.allclose(preconditioned.rho_up_next, generic.rho_up_next)
    assert np.allclose(preconditioned.rho_down_next, generic.rho_down_next)


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
