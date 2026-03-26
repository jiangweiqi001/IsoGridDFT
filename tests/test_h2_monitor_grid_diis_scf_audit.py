"""Tiny regression scaffolding for the H2 A-grid DIIS SCF audit."""

from isogrid.audit.h2_monitor_grid_diis_scf_audit import H2DiisScfParameterSummary
from isogrid.audit.h2_monitor_grid_diis_scf_audit import H2DiisScfRouteResult
from isogrid.audit.h2_monitor_grid_diis_scf_audit import H2DiisScfSpinAuditResult
from isogrid.audit.h2_monitor_grid_diis_scf_audit import H2MonitorGridDiisScfAuditResult
from isogrid.scf import SinglePointEnergyComponents


def test_construct_h2_monitor_grid_diis_scf_audit_result() -> None:
    parameters = H2DiisScfParameterSummary(
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
        max_iterations=20,
        density_tolerance=5.0e-3,
        energy_tolerance=5.0e-5,
        eigensolver_tolerance=1.0e-3,
        eigensolver_ncv=20,
        diis_enabled=True,
        diis_warmup_iterations=3,
        diis_history_length=4,
        diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
    )
    energy = SinglePointEnergyComponents(
        kinetic=1.0,
        local_ionic=-2.0,
        nonlocal_ionic=0.0,
        hartree=0.5,
        xc=-0.1,
        ion_ion_repulsion=0.7,
        total=0.1,
    )
    route = H2DiisScfRouteResult(
        spin_state_label="singlet",
        scheme_label="diis-prototype",
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
        includes_nonlocal=False,
        parameter_summary=parameters,
        converged=False,
        iteration_count=10,
        final_total_energy_ha=0.1,
        final_lowest_eigenvalue_ha=-0.45,
        energy_history_ha=(0.2, 0.1),
        density_residual_history=(0.4, 0.3),
        energy_change_history_ha=(None, -0.1),
        final_density_residual=0.3,
        final_energy_change_ha=-0.1,
        final_rho_up_electrons=1.0,
        final_rho_down_electrons=1.0,
        final_energy_components=energy,
        trajectory_verdict="stable_not_converged",
        diis_enabled=True,
        diis_warmup_iterations=3,
        diis_history_length=4,
        diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
        diis_used_iterations=(4, 5, 6),
        diis_history_sizes=(1, 2, 3, 4),
        diis_fallback_iterations=(7,),
    )
    spin = H2DiisScfSpinAuditResult(
        spin_state_label="singlet",
        baseline_route=route,
        smaller_mixing_route=route,
        diis_prototype_route=route,
    )
    result = H2MonitorGridDiisScfAuditResult(
        singlet=spin,
        triplet=spin,
        note="audit",
    )

    assert result.singlet.diis_prototype_route.diis_enabled is True
    assert result.singlet.diis_prototype_route.diis_fallback_iterations == (7,)
    assert result.triplet.baseline_route.energy_history_ha[-1] == 0.1
