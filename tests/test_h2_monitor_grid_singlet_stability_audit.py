"""Tiny regression scaffolding for the singlet stability audit."""

from isogrid.audit.h2_monitor_grid_singlet_stability_audit import H2SingletStabilityAuditResult
from isogrid.audit.h2_monitor_grid_singlet_stability_audit import H2SingletStabilityParameterSummary
from isogrid.audit.h2_monitor_grid_singlet_stability_audit import H2SingletStabilityRouteResult
from isogrid.audit.h2_monitor_grid_singlet_stability_audit import H2TwoCycleDiagnostics
from isogrid.scf import SinglePointEnergyComponents


def test_construct_h2_singlet_stability_result() -> None:
    parameters = H2SingletStabilityParameterSummary(
        grid_shape=(67, 67, 81),
        box_half_extents_bohr=(8.0, 8.0, 10.0),
        weight_scale=4.0,
        radius_scale=0.70,
        patch_radius_scale=0.75,
        patch_grid_shape=(25, 25, 25),
        correction_strength=1.30,
        interpolation_neighbors=8,
        kinetic_version="trial_fix",
        mixing=0.20,
        max_iterations=20,
        density_tolerance=5.0e-3,
        energy_tolerance=5.0e-5,
        eigensolver_tolerance=1.0e-3,
        eigensolver_ncv=20,
    )
    two_cycle = H2TwoCycleDiagnostics(
        detected_two_cycle=True,
        tail_length=10,
        even_energy_mean_ha=-0.13,
        odd_energy_mean_ha=-0.14,
        even_odd_energy_gap_ha=0.01,
        even_energy_std_ha=1.0e-4,
        odd_energy_std_ha=1.0e-4,
        even_residual_mean=0.337,
        odd_residual_mean=0.3372,
        even_odd_residual_gap=2.0e-4,
        even_residual_std=1.0e-5,
        odd_residual_std=1.0e-5,
        verdict="two_cycle",
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
    route = H2SingletStabilityRouteResult(
        scheme_label="baseline",
        path_type="monitor_a_grid_plus_patch",
        spin_state_label="singlet",
        kinetic_version="trial_fix",
        includes_nonlocal=False,
        parameter_summary=parameters,
        converged=False,
        iteration_count=20,
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
        two_cycle=two_cycle,
    )
    result = H2SingletStabilityAuditResult(
        baseline_route=route,
        smaller_mixing_route=route,
        note="audit",
    )

    assert result.baseline_route.two_cycle.detected_two_cycle is True
    assert result.baseline_route.final_lowest_eigenvalue_ha == -0.45
    assert result.smaller_mixing_route.energy_history_ha[-1] == 0.1
