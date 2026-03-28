"""Minimal smoke tests for the H2 singlet Anderson adequacy audit."""

from importlib import import_module

from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineAuditResult
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineBehavior
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineParameterSummary
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineRouteResult
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineTimingBreakdown
from isogrid.scf import SinglePointEnergyComponents


def _build_route(
    *,
    mixer: str,
    solver_variant: str,
    max_iterations: int = 20,
    formal_mixer_history_length: int | None = None,
    formal_mixer_regularization: float | None = None,
    formal_mixer_damping: float | None = None,
    formal_mixer_step_clip_factor: float | None = None,
    formal_mixer_reset_on_growth: bool = False,
    formal_mixer_reset_growth_factor: float | None = None,
) -> H2JaxSingletMainlineRouteResult:
    return H2JaxSingletMainlineRouteResult(
        path_label=f"jax-singlet-mainline-{solver_variant}",
        spin_state_label="singlet",
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
        includes_nonlocal=False,
        max_iterations=max_iterations,
        mixing=0.10,
        mixer=mixer,
        solver_variant=solver_variant,
        formal_mixer_history_length=formal_mixer_history_length,
        formal_mixer_regularization=formal_mixer_regularization,
        formal_mixer_damping=formal_mixer_damping,
        formal_mixer_step_clip_factor=formal_mixer_step_clip_factor,
        formal_mixer_reset_on_growth=formal_mixer_reset_on_growth,
        formal_mixer_reset_growth_factor=formal_mixer_reset_growth_factor,
        converged=False,
        iteration_count=max_iterations,
        final_total_energy_ha=-0.17,
        final_lowest_eigenvalue_ha=-0.40,
        final_density_residual=0.30,
        final_energy_change_ha=0.01,
        total_wall_time_seconds=95.0,
        average_iteration_wall_time_seconds=4.75,
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
            mixer=mixer,
            solver_variant=solver_variant,
            max_iterations=max_iterations,
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
            diis_enabled=(mixer == "diis"),
            diis_warmup_iterations=3,
            diis_history_length=4,
            diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
            anderson_enabled=(mixer == "anderson"),
            anderson_warmup_iterations=3,
            anderson_history_length=4,
            anderson_regularization=1.0e-8,
            anderson_damping=0.5,
            anderson_step_clip_factor=formal_mixer_step_clip_factor,
            anderson_reset_on_growth=formal_mixer_reset_on_growth,
            anderson_reset_growth_factor=1.05,
            anderson_residual_definition="density_fixed_point_residual=rho_out-rho_in",
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
            tail_residual_ratios=(0.97, 1.02, 1.00, 1.00),
            average_tail_residual_ratio=0.9975,
            tail_residual_ratio_std=0.018,
            entered_plateau=True,
        ),
        diis_used_iterations=(3, 4, 5) if mixer == "diis" else (),
        diis_fallback_iterations=(6,) if mixer == "diis" else (),
        anderson_used_iterations=(4, 5, 7) if mixer == "anderson" else (),
        anderson_fallback_iterations=(),
        anderson_reset_iterations=(2, 3) if mixer == "anderson" else (),
        final_energy_components=SinglePointEnergyComponents(
            kinetic=0.45,
            local_ionic=-1.65,
            nonlocal_ionic=0.0,
            hartree=1.42,
            xc=-0.39,
            ion_ion_repulsion=0.714285714286,
            total=-0.17,
        ),
        note="singlet adequacy smoke",
    )


def test_h2_jax_singlet_mainline_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_jax_singlet_mainline_audit")

    assert hasattr(module, "run_h2_jax_singlet_mainline_audit")
    assert hasattr(module, "print_h2_jax_singlet_mainline_summary")


def test_construct_h2_jax_singlet_mainline_result() -> None:
    result = H2JaxSingletMainlineAuditResult(
        path_label="jax-singlet-mainline",
        spin_state_label="singlet",
        path_type="monitor_a_grid_plus_patch",
        baseline_linear_route=_build_route(mixer="linear", solver_variant="linear-0p10"),
        diis_route=_build_route(mixer="diis", solver_variant="diis-prototype"),
        anderson_baseline_route=_build_route(
            mixer="anderson",
            solver_variant="anderson-baseline",
            formal_mixer_history_length=4,
            formal_mixer_regularization=1.0e-8,
            formal_mixer_damping=0.5,
        ),
        anderson_extended_route=_build_route(
            mixer="anderson",
            solver_variant="anderson-extended",
            formal_mixer_history_length=4,
            formal_mixer_regularization=1.0e-8,
            formal_mixer_damping=0.5,
            formal_mixer_step_clip_factor=1.0,
            formal_mixer_reset_on_growth=True,
            formal_mixer_reset_growth_factor=1.05,
        ),
        supplemental_anderson_route=_build_route(
            mixer="anderson",
            solver_variant="anderson-extended-long40",
            max_iterations=40,
            formal_mixer_history_length=4,
            formal_mixer_regularization=1.0e-8,
            formal_mixer_damping=0.5,
            formal_mixer_step_clip_factor=1.0,
            formal_mixer_reset_on_growth=True,
            formal_mixer_reset_growth_factor=1.05,
        ),
        diagnosis="singlet adequacy smoke",
        note="formal Anderson adequacy smoke",
    )

    assert result.baseline_linear_route.mixer == "linear"
    assert result.diis_route.mixer == "diis"
    assert result.anderson_baseline_route.mixer == "anderson"
    assert result.anderson_extended_route.formal_mixer_step_clip_factor == 1.0
    assert result.anderson_extended_route.formal_mixer_reset_on_growth is True
    assert result.supplemental_anderson_route.max_iterations == 40
    assert result.anderson_extended_route.final_density_residual == 0.30
    assert result.anderson_extended_route.behavior.verdict == "plateau_or_stall"
