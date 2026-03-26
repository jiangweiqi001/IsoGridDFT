"""Minimal smoke tests for the A-grid H2 SCF dry-run audit."""

from importlib import import_module

from isogrid.audit.h2_monitor_grid_scf_dry_run_audit import (
    H2MonitorGridScfDryRunAuditResult,
)
from isogrid.audit.h2_monitor_grid_scf_dry_run_audit import H2ScfDryRunRouteResult
from isogrid.scf import SinglePointEnergyComponents


def test_h2_monitor_grid_scf_dry_run_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_scf_dry_run_audit")

    assert hasattr(module, "run_h2_monitor_grid_scf_dry_run_audit")
    assert hasattr(module, "print_h2_monitor_grid_scf_dry_run_summary")


def test_construct_h2_monitor_grid_scf_dry_run_result() -> None:
    route = H2ScfDryRunRouteResult(
        path_type="monitor_a_grid_plus_patch",
        spin_state_label="singlet",
        kinetic_version="trial_fix",
        includes_nonlocal=False,
        parameter_summary="shape=(67,67,81), dry-run",
        converged=False,
        iteration_count=10,
        final_total_energy_ha=-0.13,
        lowest_eigenvalue_ha=-0.45,
        energy_history_ha=(-1.0, -0.2, -0.13),
        density_residual_history=(0.2, 0.3, 0.33),
        energy_change_history_ha=(None, 0.8, -0.01),
        final_density_residual=0.33,
        final_energy_change_ha=-0.01,
        final_rho_up_electrons=1.0,
        final_rho_down_electrons=1.0,
        final_energy_components=SinglePointEnergyComponents(
            kinetic=1.2,
            local_ionic=-3.7,
            nonlocal_ionic=0.0,
            hartree=2.5,
            xc=-0.8,
            ion_ion_repulsion=0.714285714286,
            total=-0.13,
        ),
    )
    audit_result = H2MonitorGridScfDryRunAuditResult(
        legacy_singlet=route,
        monitor_singlet=route,
        legacy_triplet=None,
        monitor_triplet=None,
        note="audit smoke",
    )

    assert route.kinetic_version == "trial_fix"
    assert route.converged is False
    assert route.iteration_count == 10
    assert audit_result.monitor_singlet.energy_history_ha[-1] == -0.13
