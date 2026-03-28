"""Minimal smoke tests for the H2 singlet JAX formal-mixer audit."""

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
        converged=False,
        iteration_count=20,
        final_total_energy_ha=-0.1649 if mixer == "linear" else -0.1738,
        final_lowest_eigenvalue_ha=-0.3937 if mixer == "linear" else -0.3763,
        final_density_residual=0.3109 if mixer == "linear" else 0.2930,
        final_energy_change_ha=0.0120 if mixer == "linear" else 0.0082,
        total_wall_time_seconds=92.7 if mixer == "linear" else 91.8,
        average_iteration_wall_time_seconds=4.63 if mixer == "linear" else 4.59,
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
        diis_used_iterations=() if mixer == "linear" else (3, 4, 5),
        diis_fallback_iterations=() if mixer == "linear" else (6,),
        final_energy_components=SinglePointEnergyComponents(
            kinetic=0.44,
            local_ionic=-1.64,
            nonlocal_ionic=0.0,
            hartree=1.40,
            xc=-0.39,
            ion_ion_repulsion=0.714285714286,
            total=-0.1649 if mixer == "linear" else -0.1738,
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
        formal_mixer_route=_build_route("diis", "diis-prototype"),
        diagnosis="singlet fixed-point smoke",
        note="formal mixer smoke",
    )

    assert result.baseline_linear_route.mixer == "linear"
    assert result.formal_mixer_route.mixer == "diis"
    assert result.formal_mixer_route.parameter_summary.diis_enabled is True
    assert result.formal_mixer_route.final_density_residual == 0.2930
    assert result.formal_mixer_route.behavior.verdict == "stable_not_converged"
