"""Minimal smoke tests for the H2 grid/domain convergence audit layer."""

from isogrid.audit.baselines import H2_DEFAULT_PYSCF_REGRESSION_BASELINE
from isogrid.audit.h2_grid_convergence_audit import EnergyComponentDrift
from isogrid.audit.h2_grid_convergence_audit import H2GridConvergenceScanPoint
from isogrid.audit.h2_grid_convergence_audit import H2GridScanParameters
from isogrid.scf import SinglePointEnergyComponents


def test_construct_grid_convergence_scan_point() -> None:
    parameters = H2GridScanParameters(
        label="grid_51",
        family="grid_shape",
        description="Current default H2 audit point.",
        grid_shape=(51, 51, 51),
        box_half_extents_bohr=(8.0, 8.0, 10.0),
        min_cell_widths_bohr=(0.132, 0.132, 0.137),
    )
    components = SinglePointEnergyComponents(
        kinetic=0.8,
        local_ionic=-2.0,
        nonlocal_ionic=0.0,
        hartree=0.6,
        xc=-0.3,
        ion_ion_repulsion=0.7142857142857143,
        total=-1.1857142857142857,
    )
    drift = EnergyComponentDrift(
        kinetic=0.0,
        local_ionic=0.0,
        nonlocal_ionic=0.0,
        hartree=0.0,
        xc=0.0,
        ion_ion_repulsion=0.0,
        total=0.0,
    )
    point = H2GridConvergenceScanPoint(
        parameters=parameters,
        isogrid_total_energy=-1.18,
        pyscf_total_energy=-1.17,
        total_error_ha=-0.01,
        total_error_mha=-10.0,
        isogrid_converged=True,
        pyscf_converged=True,
        iteration_count=5,
        energy_components=components,
        baseline_drift=drift,
        dominant_component_drifts_mha=(("T_s", 0.0),),
    )

    assert point.parameters.family == "grid_shape"
    assert point.total_error_mha == -10.0
    assert point.energy_components.total == -1.1857142857142857
    assert point.baseline_drift.total == 0.0


def test_import_regression_baseline_fields() -> None:
    baseline = H2_DEFAULT_PYSCF_REGRESSION_BASELINE

    assert baseline.singlet_total_error_mha == -15.231
    assert baseline.triplet_total_error_mha == -37.934
    assert baseline.singlet_triplet_gap_error_mha == -22.703
    assert baseline.grid_shape == (51, 51, 51)
