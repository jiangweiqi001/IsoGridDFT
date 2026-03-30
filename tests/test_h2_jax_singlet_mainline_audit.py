"""Minimal smoke tests for the H2 singlet Hartree-tail mitigation audit."""

from importlib import import_module

from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletAcceptanceAuditResult
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletFixedPointLocalDifficulty
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletHartreeTailGuardAuditResult
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineAuditResult
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineBehavior
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineParameterSummary
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineRouteResult
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletStructuralStabilizerAuditResult
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineTimingBreakdown
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletResponseChannelDifficulty
from isogrid.scf import SinglePointEnergyComponents


def _build_route(
    *,
    solver_variant: str,
    mitigation_enabled: bool,
) -> H2JaxSingletMainlineRouteResult:
    return H2JaxSingletMainlineRouteResult(
        path_label=f"jax-singlet-mainline-{solver_variant}",
        spin_state_label="singlet",
        solver_backend="jax",
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
        includes_nonlocal=False,
        max_iterations=20,
        mixing=0.10,
        mixer="anderson",
        solver_variant=solver_variant,
        formal_mixer_history_length=6,
        formal_mixer_regularization=1.0e-8,
        formal_mixer_damping=0.55,
        formal_mixer_step_clip_factor=1.0,
        formal_mixer_reset_on_growth=True,
        formal_mixer_reset_growth_factor=1.05,
        formal_mixer_adaptive_damping_enabled=True,
        formal_mixer_min_damping=0.35,
        formal_mixer_max_damping=0.75,
        formal_mixer_acceptance_residual_ratio_threshold=1.02,
        formal_mixer_collinearity_cosine_threshold=0.995,
        converged=False,
        iteration_count=20,
        final_total_energy_ha=-0.17,
        final_lowest_eigenvalue_ha=-0.38,
        final_density_residual=0.30,
        final_energy_change_ha=0.01,
        total_wall_time_seconds=100.0,
        average_iteration_wall_time_seconds=5.0,
        parameter_summary=H2JaxSingletMainlineParameterSummary(
            grid_shape=(67, 67, 81),
            box_half_extents_bohr=(8.0, 8.0, 10.0),
            weight_scale=4.0,
            radius_scale=0.70,
            patch_radius_scale=0.75,
            patch_grid_shape=(25, 25, 25),
            correction_strength=1.30,
            interpolation_neighbors=8,
            kinetic_version="trial_fix",
            mixing=0.10,
            mixer="anderson",
            solver_variant=solver_variant,
            max_iterations=20,
            density_tolerance=5.0e-3,
            energy_tolerance=5.0e-5,
            eigensolver_tolerance=1.0e-3,
            eigensolver_ncv=20,
            hartree_backend="jax",
            use_jax_hartree_cached_operator=True,
            jax_hartree_cg_impl="jax_loop",
            jax_hartree_cg_preconditioner="none",
            jax_hartree_line_preconditioner_impl="baseline",
            use_jax_block_kernels=True,
            use_step_local_static_local_reuse=True,
            diis_enabled=False,
            diis_warmup_iterations=3,
            diis_history_length=4,
            diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
            anderson_enabled=True,
            anderson_warmup_iterations=3,
            anderson_history_length=6,
            anderson_regularization=1.0e-8,
            anderson_damping=0.55,
            anderson_step_clip_factor=1.0,
            anderson_reset_on_growth=True,
            anderson_reset_growth_factor=1.05,
            anderson_adaptive_damping_enabled=True,
            anderson_min_damping=0.35,
            anderson_max_damping=0.75,
            anderson_acceptance_residual_ratio_threshold=1.02,
            anderson_collinearity_cosine_threshold=0.995,
            anderson_residual_definition="density_fixed_point_residual=rho_out-rho_in",
            singlet_hartree_tail_mitigation_enabled=mitigation_enabled,
            singlet_hartree_tail_mitigation_weight=0.7 if mitigation_enabled else None,
            singlet_hartree_tail_residual_ratio_trigger=1.0 if mitigation_enabled else None,
            singlet_hartree_tail_projected_ratio_trigger=0.6 if mitigation_enabled else None,
            singlet_hartree_tail_hartree_share_trigger=0.8 if mitigation_enabled else None,
        ),
        timing_breakdown=H2JaxSingletMainlineTimingBreakdown(
            eigensolver_wall_time_seconds=60.0,
            static_local_prepare_wall_time_seconds=20.0,
            hartree_solve_wall_time_seconds=10.0,
            energy_evaluation_wall_time_seconds=9.0,
            density_update_wall_time_seconds=0.2,
            bookkeeping_wall_time_seconds=0.1,
        ),
        behavior=H2JaxSingletMainlineBehavior(
            detected_two_cycle=False,
            tail_length=5,
            even_odd_energy_gap_ha=None,
            even_odd_residual_gap=None,
            verdict="plateau_or_stall",
            tail_energy_history_ha=(-0.17, -0.19, -0.16, -0.18, -0.17),
            tail_density_residual_history=(0.31, 0.30, 0.31, 0.30, 0.30),
            tail_energy_change_history_ha=(0.01, -0.02, 0.03, -0.01, 0.01),
            tail_residual_ratios=(0.99, 1.01, 1.00, 1.00),
            average_tail_residual_ratio=1.0,
            tail_residual_ratio_std=0.01,
            entered_plateau=True,
        ),
        fixed_point_local_difficulty=H2JaxSingletFixedPointLocalDifficulty(
            tail_window_length=5,
            average_tail_residual_ratio=1.0,
            tail_residual_ratio_std=0.01,
            maximum_tail_residual_ratio=1.01,
            entered_plateau=True,
            plateau_window_length=4,
            tail_residual_amplitude=0.01,
            weak_cycle_indicator=False,
            local_contraction_verdict="poorly_contractive_near_unity",
            secant_subspace_condition_proxy=1.0e6,
            secant_collinearity_max_abs_cosine=0.99,
            diagnosis="tail proxy smoke",
        ),
        diis_used_iterations=(),
        diis_fallback_iterations=(),
        anderson_used_iterations=(4, 5, 6),
        anderson_fallback_iterations=(),
        anderson_rejected_iterations=(),
        anderson_reset_iterations=(2, 3),
        anderson_filtered_history_sizes=(1, 2, 2),
        anderson_effective_damping_history=(0.55, 0.45, 0.45),
        anderson_projected_residual_ratio_history=(0.58, 0.55, 0.54),
        singlet_hartree_tail_mitigation_enabled=mitigation_enabled,
        singlet_hartree_tail_mitigation_weight=0.7 if mitigation_enabled else None,
        singlet_hartree_tail_residual_ratio_trigger=1.0 if mitigation_enabled else None,
        singlet_hartree_tail_projected_ratio_trigger=0.6 if mitigation_enabled else None,
        singlet_hartree_tail_hartree_share_trigger=0.8 if mitigation_enabled else None,
        singlet_hartree_tail_mitigation_triggered_iterations=((5, 7) if mitigation_enabled else ()),
        singlet_hartree_tail_hartree_share_history=((0.81, 0.82) if mitigation_enabled else (None,)),
        singlet_hartree_tail_residual_ratio_history=((1.03, 1.04) if mitigation_enabled else (None,)),
        singlet_hartree_tail_projected_ratio_history=((0.62, 0.61) if mitigation_enabled else (None,)),
        final_energy_components=SinglePointEnergyComponents(
            kinetic=0.45,
            local_ionic=-1.65,
            nonlocal_ionic=0.0,
            hartree=1.42,
            xc=-0.39,
            ion_ion_repulsion=0.714285714286,
            total=-0.17,
        ),
        note="singlet hartree-tail mitigation smoke",
        response_channel_difficulty=H2JaxSingletResponseChannelDifficulty(
            tail_pair_iterations=(19, 20),
            density_secant_norm=0.15,
            total_output_response_proxy=1.01,
            total_effective_potential_amplification_proxy=1.8,
            hartree_potential_amplification_proxy=0.7,
            xc_potential_amplification_proxy=0.4,
            local_orbital_potential_amplification_proxy=0.7,
            hartree_potential_contribution_share=0.45,
            xc_potential_contribution_share=0.20,
            local_orbital_potential_contribution_share=0.35,
            hartree_output_sensitivity_proxy=0.44,
            xc_output_sensitivity_proxy=0.19,
            local_orbital_output_sensitivity_proxy=0.38,
            coupling_excess_output_sensitivity_proxy=0.0,
            primary_difficulty_channel="hartree",
            dominant_coupling_label="hartree+local_orbital",
            diagnosis="channel proxy smoke",
        ),
    )


def test_h2_jax_singlet_mainline_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_jax_singlet_mainline_audit")

    assert hasattr(module, "run_h2_jax_singlet_hartree_tail_guard_audit")
    assert hasattr(module, "run_h2_jax_singlet_acceptance_audit")
    assert hasattr(module, "run_h2_jax_singlet_mainline_audit")
    assert hasattr(module, "run_h2_jax_singlet_structural_stabilizer_audit")
    assert hasattr(module, "print_h2_jax_singlet_mainline_summary")


def test_construct_h2_jax_singlet_mainline_result() -> None:
    result = H2JaxSingletMainlineAuditResult(
        path_label="jax-singlet-mainline",
        spin_state_label="singlet",
        path_type="monitor_a_grid_plus_patch",
        anderson_productionish_route=_build_route(
            solver_variant="anderson-productionish",
            mitigation_enabled=False,
        ),
        hartree_tail_mitigation_route=_build_route(
            solver_variant="hartree-tail-mitigation",
            mitigation_enabled=True,
        ),
        supplemental_hartree_tail_mitigation_route=None,
        diagnosis="singlet hartree-tail mitigation smoke",
        note="hartree-tail mitigation smoke",
    )

    assert result.anderson_productionish_route.mixer == "anderson"
    assert result.hartree_tail_mitigation_route.mixer == "anderson"
    assert result.hartree_tail_mitigation_route.singlet_hartree_tail_mitigation_enabled is True
    assert result.hartree_tail_mitigation_route.singlet_hartree_tail_mitigation_weight == 0.7
    assert result.hartree_tail_mitigation_route.final_density_residual == 0.30
    assert result.hartree_tail_mitigation_route.behavior.verdict == "plateau_or_stall"
    assert (
        result.hartree_tail_mitigation_route.response_channel_difficulty.primary_difficulty_channel
        == "hartree"
    )


def test_construct_h2_jax_singlet_acceptance_result() -> None:
    route = _build_route(
        solver_variant="anderson-productionish",
        mitigation_enabled=False,
    )
    result = H2JaxSingletAcceptanceAuditResult(
        path_label="jax-singlet-acceptance-mainline",
        spin_state_label="singlet",
        path_type="monitor_a_grid_plus_patch",
        acceptance_route=route,
        supplemental_route=None,
        diagnosis="acceptance smoke",
        note="single-route acceptance smoke",
    )

    assert result.acceptance_route.solver_backend == "jax"
    assert result.acceptance_route.mixer == "anderson"
    assert result.acceptance_route.converged is False
    assert result.acceptance_route.final_density_residual == 0.30


def test_construct_h2_jax_singlet_hartree_tail_guard_result() -> None:
    baseline_route = _build_route(
        solver_variant="anderson-productionish",
        mitigation_enabled=False,
    )
    guard_route = _build_route(
        solver_variant="anderson-plus-hartree-tail-guard",
        mitigation_enabled=False,
    )
    guard_route = H2JaxSingletMainlineRouteResult(
        **{
            **guard_route.__dict__,
            "guard_name": "hartree_tail_guard",
            "guard_enabled": True,
            "guard_triggered": True,
            "guard_trigger_count": 1,
            "guard_triggered_iterations": (4,),
            "guard_alpha": 0.45,
            "guard_residual_ratio_trigger": 0.995,
            "guard_projected_ratio_trigger": 0.60,
            "guard_hartree_share_trigger": 0.80,
            "guard_hartree_share_history": (0.82, 0.83),
            "guard_residual_ratio_history": (1.01, 1.00),
            "guard_projected_ratio_history": (0.62, 0.55),
        }
    )
    result = H2JaxSingletHartreeTailGuardAuditResult(
        path_label="jax-singlet-hartree-tail-guard",
        spin_state_label="singlet",
        path_type="monitor_a_grid_plus_patch",
        baseline_route=baseline_route,
        guard_route=guard_route,
        supplemental_guard_route=None,
        diagnosis="guard smoke",
        note="guard smoke",
    )

    assert result.guard_route.solver_backend == "jax"
    assert result.guard_route.mixer == "anderson"
    assert result.guard_route.guard_enabled is True
    assert result.guard_route.guard_triggered is True
    assert result.guard_route.guard_trigger_count == 1
    assert result.guard_route.final_density_residual == 0.30


def test_construct_h2_jax_singlet_hartree_tail_guard_v2_result() -> None:
    baseline_route = _build_route(
        solver_variant="anderson-productionish",
        mitigation_enabled=False,
    )
    guard_route = _build_route(
        solver_variant="anderson-plus-hartree-tail-guard-v2",
        mitigation_enabled=False,
    )
    guard_route = H2JaxSingletMainlineRouteResult(
        **{
            **guard_route.__dict__,
            "guard_name": "hartree_tail_guard_v2",
            "guard_enabled": True,
            "guard_triggered": True,
            "guard_trigger_count": 1,
            "guard_triggered_iterations": (4,),
            "guard_hold_steps": 3,
            "guard_exit_residual_ratio": 0.995,
            "guard_exit_stable_steps": 2,
            "guard_entry_iterations": (4,),
            "guard_exit_iterations": (7,),
            "guard_hold_lengths": (3,),
            "guard_active_iteration_history": (
                False,
                False,
                False,
                False,
                True,
                True,
                True,
            ),
            "guard_alpha": 0.45,
            "guard_residual_ratio_trigger": 0.995,
            "guard_projected_ratio_trigger": 0.60,
            "guard_hartree_share_trigger": 0.80,
            "guard_hartree_share_history": (0.82, 0.83),
            "guard_residual_ratio_history": (1.01, 0.99),
            "guard_projected_ratio_history": (0.62, 0.55),
        }
    )

    assert guard_route.solver_backend == "jax"
    assert guard_route.guard_name == "hartree_tail_guard_v2"
    assert guard_route.guard_enabled is True
    assert guard_route.guard_hold_steps == 3
    assert guard_route.guard_hold_lengths == (3,)
    assert guard_route.final_density_residual == 0.30


def test_construct_h2_jax_singlet_structural_stabilizer_result() -> None:
    baseline_route = _build_route(
        solver_variant="anderson-productionish",
        mitigation_enabled=False,
    )
    stabilizer_route = _build_route(
        solver_variant="anderson-plus-hartree-tail-freeze-guard",
        mitigation_enabled=False,
    )
    stabilizer_route = H2JaxSingletMainlineRouteResult(
        **{
            **stabilizer_route.__dict__,
            "guard_name": "hartree_tail_freeze_guard",
            "guard_strategy": "frozen_potential",
            "guard_enabled": True,
            "guard_triggered": True,
            "guard_trigger_count": 1,
            "guard_triggered_iterations": (4,),
            "guard_hold_steps": 2,
            "guard_exit_residual_ratio": 0.995,
            "guard_exit_stable_steps": 2,
            "guard_entry_iterations": (4,),
            "guard_exit_iterations": (6,),
            "guard_hold_lengths": (2,),
            "guard_active_iteration_history": (
                False,
                False,
                False,
                False,
                True,
                True,
            ),
        }
    )
    result = H2JaxSingletStructuralStabilizerAuditResult(
        path_label="jax-singlet-structural-stabilizer",
        spin_state_label="singlet",
        path_type="monitor_a_grid_plus_patch",
        baseline_route=baseline_route,
        stabilizer_route=stabilizer_route,
        supplemental_stabilizer_route=None,
        diagnosis="structural stabilizer smoke",
        note="structural stabilizer smoke",
    )

    assert result.stabilizer_route.solver_backend == "jax"
    assert result.stabilizer_route.mixer == "anderson"
    assert result.stabilizer_route.guard_name == "hartree_tail_freeze_guard"
    assert result.stabilizer_route.guard_strategy == "frozen_potential"
    assert result.stabilizer_route.guard_hold_steps == 2
    assert result.stabilizer_route.final_density_residual == 0.30
