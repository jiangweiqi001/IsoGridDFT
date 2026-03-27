"""Minimal smoke tests for the H2 JAX SCF hot-path audit."""

from importlib import import_module

from isogrid.audit.h2_jax_scf_hotpath_audit import H2JaxScfHotpathAuditResult
from isogrid.audit.h2_jax_scf_hotpath_audit import H2JaxScfHotpathRouteResult
from isogrid.audit.h2_jax_scf_hotpath_audit import H2JaxScfTimingBreakdown
from isogrid.scf import SinglePointEnergyComponents


def test_h2_jax_scf_hotpath_module_imports() -> None:
    module = import_module("isogrid.audit.h2_jax_scf_hotpath_audit")

    assert hasattr(module, "run_h2_jax_scf_hotpath_audit")
    assert hasattr(module, "print_h2_jax_scf_hotpath_summary")


def test_construct_h2_jax_scf_hotpath_result() -> None:
    route = H2JaxScfHotpathRouteResult(
        path_type="monitor_a_grid_plus_patch",
        spin_state_label="triplet",
        kinetic_version="trial_fix",
        use_jax_block_kernels=True,
        converged=True,
        iteration_count=8,
        final_total_energy_ha=-1.22,
        lowest_eigenvalue_ha=-0.41,
        energy_history_ha=(-1.0, -1.1, -1.22),
        density_residual_history=(0.1, 0.02, 0.004),
        energy_change_history_ha=(None, -0.1, -0.12),
        final_density_residual=0.004,
        final_energy_change_ha=-0.12,
        final_rho_up_electrons=2.0,
        final_rho_down_electrons=0.0,
        total_wall_time_seconds=12.0,
        average_iteration_wall_time_seconds=1.5,
        timing_breakdown=H2JaxScfTimingBreakdown(
            eigensolver_wall_time_seconds=6.0,
            energy_evaluation_wall_time_seconds=5.0,
            density_update_wall_time_seconds=0.5,
            bookkeeping_wall_time_seconds=0.5,
        ),
        parameter_summary="A-grid hotpath smoke",
        final_energy_components=SinglePointEnergyComponents(
            kinetic=1.0,
            local_ionic=-3.0,
            nonlocal_ionic=0.0,
            hartree=1.5,
            xc=-0.7,
            ion_ion_repulsion=0.714285714286,
            total=-1.22,
        ),
    )
    audit_result = H2JaxScfHotpathAuditResult(
        triplet_old=route,
        triplet_jax=route,
        singlet_old=None,
        singlet_jax=None,
        note="hotpath smoke",
    )

    assert route.use_jax_block_kernels is True
    assert route.total_wall_time_seconds == 12.0
    assert route.timing_breakdown.eigensolver_wall_time_seconds == 6.0
    assert audit_result.triplet_jax.final_total_energy_ha == -1.22
