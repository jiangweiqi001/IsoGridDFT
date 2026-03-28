"""Minimal smoke tests for the H2 singlet different-formal-mixer audit."""

from importlib import import_module

from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineAuditResult
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineBehavior
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineParameterSummary
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineRouteResult
from isogrid.audit.h2_jax_singlet_mainline_audit import H2JaxSingletMainlineTimingBreakdown
from isogrid.scf import SinglePointEnergyComponents


def _build_route(mixer: str, solver_variant: str) -> H2JaxSingletMainlineRouteResult:
    return H2JaxSingletMainlineRouteResult(
        path_label=f"jax-singlet-mainline-{solver_variant}",
        spin_state_label="singlet",
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
        includes_nonlocal=False,
        mixing=0.10,
        mixer=mixer,
        solver_variant=solver_variant,
        formal_mixer_history_length=None if mixer not in {"anderson", "broyden_like"} else 4,
        formal_mixer_regularization=None if mixer not in {"anderson", "broyden_like"} else 1.0e-8,
        formal_mixer_damping=None if mixer not in {"anderson", "broyden_like"} else 0.5,
        converged=False,
        iteration_count=20,
        final_total_energy_ha=-0.17 if mixer == "linear" else -0.16,
        final_lowest_eigenvalue_ha=-0.39 if mixer == "linear" else -0.41,
        final_density_residual=0.31 if mixer == "linear" else 0.30,
        final_energy_change_ha=-0.008 if mixer == "linear" else 0.005,
        total_wall_time_seconds=86.0 if mixer == "linear" else 98.0,
        average_iteration_wall_time_seconds=4.3 if mixer == "linear" else 4.9,
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
            diis_enabled=(mixer == "diis"),
            diis_warmup_iterations=3,
            diis_history_length=4,
            diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
            anderson_enabled=(mixer == "anderson"),
            anderson_warmup_iterations=3,
            anderson_history_length=4,
            anderson_regularization=1.0e-8,
            anderson_damping=0.5,
            anderson_residual_definition="density_fixed_point_residual=rho_out-rho_in",
            broyden_enabled=(mixer == "broyden_like"),
            broyden_warmup_iterations=3,
            broyden_history_length=4,
            broyden_regularization=1.0e-8,
            broyden_damping=0.5,
            broyden_residual_definition="density_fixed_point_residual=rho_out-rho_in",
        ),
        timing_breakdown=H2JaxSingletMainlineTimingBreakdown(
            eigensolver_wall_time_seconds=64.0,
            static_local_prepare_wall_time_seconds=24.0,
            hartree_solve_wall_time_seconds=14.0,
            energy_evaluation_wall_time_seconds=13.0,
            density_update_wall_time_seconds=0.2,
            bookkeeping_wall_time_seconds=14.0,
        ),
        behavior=H2JaxSingletMainlineBehavior(
            detected_two_cycle=False,
            tail_length=10,
            even_odd_energy_gap_ha=8.5e-3,
            even_odd_residual_gap=2.5e-3,
            verdict="stable_not_converged",
            tail_energy_history_ha=(-0.17, -0.18, -0.16, -0.17, -0.16),
            tail_density_residual_history=(0.31, 0.31, 0.30, 0.30, 0.29),
            tail_energy_change_history_ha=(0.01, -0.01, 0.01, -0.01, 0.01),
        ),
        diis_used_iterations=() if mixer != "diis" else (3, 4, 5),
        diis_fallback_iterations=() if mixer != "diis" else (6,),
        anderson_used_iterations=() if mixer != "anderson" else (3, 4, 5),
        anderson_fallback_iterations=() if mixer != "anderson" else (6,),
        broyden_used_iterations=() if mixer != "broyden_like" else (3, 4, 5),
        broyden_fallback_iterations=() if mixer != "broyden_like" else (6,),
        final_energy_components=SinglePointEnergyComponents(
            kinetic=0.44,
            local_ionic=-1.64,
            nonlocal_ionic=0.0,
            hartree=1.40,
            xc=-0.39,
            ion_ion_repulsion=0.714285714286,
            total=-0.17 if mixer == "linear" else -0.16,
        ),
        note="singlet mainline smoke",
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
        baseline_linear_route=_build_route("linear", "linear-0p10"),
        diis_route=_build_route("diis", "diis-prototype"),
        anderson_baseline_route=_build_route("anderson", "anderson-baseline"),
        different_formal_mixer_route=_build_route("broyden_like", "broyden-like-prototype"),
        diagnosis="singlet fixed-point smoke",
        note="formal mixer smoke",
    )

    assert result.baseline_linear_route.mixer == "linear"
    assert result.diis_route.mixer == "diis"
    assert result.anderson_baseline_route.mixer == "anderson"
    assert result.different_formal_mixer_route.mixer == "broyden_like"
    assert result.anderson_baseline_route.formal_mixer_history_length == 4
    assert result.different_formal_mixer_route.formal_mixer_history_length == 4
    assert result.different_formal_mixer_route.parameter_summary.broyden_enabled is True
    assert result.different_formal_mixer_route.final_density_residual == 0.30
    assert result.different_formal_mixer_route.behavior.verdict == "stable_not_converged"
