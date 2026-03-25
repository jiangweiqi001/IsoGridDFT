"""Minimal smoke tests for the H2 A-grid fairness calibration audit."""

from isogrid.audit.h2_monitor_grid_fair_calibration_audit import H2MonitorFairCalibrationParameters
from isogrid.audit.h2_monitor_grid_fair_calibration_audit import H2MonitorFairCalibrationPoint
from isogrid.audit.h2_monitor_grid_fair_calibration_audit import run_h2_monitor_grid_fair_calibration_audit
from isogrid.audit.h2_monitor_grid_ts_eloc_audit import H2GridEnergyGeometrySummary


def test_run_h2_monitor_grid_fair_calibration_audit_single_point() -> None:
    result = run_h2_monitor_grid_fair_calibration_audit(
        scan_parameters=(
            H2MonitorFairCalibrationParameters(
                label="smoke",
                shape=(21, 21, 25),
                box_half_extents_bohr=(8.0, 8.0, 10.0),
                weight_scale=1.4,
                radius_scale=1.0,
            ),
        )
    )

    assert len(result.fair_scan_points) == 1
    point = result.fair_scan_points[0]
    assert point.parameters.label == "smoke"
    assert isinstance(point.is_fair_point, bool)
    assert point.geometry_summary.near_core_min_spacing_bohr > 0.0


def test_construct_fair_calibration_point_object() -> None:
    summary = H2GridEnergyGeometrySummary(
        grid_type="monitor_a_grid",
        grid_shape=(29, 29, 29),
        box_half_extents_bohr=(8.0, 8.0, 10.0),
        min_spacing_estimate_bohr=0.12,
        near_core_min_spacing_bohr=0.10,
        near_atom_spacing_bohr=0.22,
        far_field_spacing_bohr=0.44,
        min_jacobian=1.0,
        max_jacobian=2.0,
        center_line_local_potential_center=-1.0,
        center_line_local_potential_near_atom=-2.0,
    )
    point = H2MonitorFairCalibrationPoint(
        parameters=H2MonitorFairCalibrationParameters(
            label="synthetic",
            shape=(29, 29, 29),
            box_half_extents_bohr=(8.0, 8.0, 10.0),
            weight_scale=1.5,
            radius_scale=0.9,
        ),
        geometry_summary=summary,
        box_not_smaller_than_legacy=True,
        near_core_not_coarser_than_legacy=True,
        positive_jacobian=True,
        is_fair_point=True,
        kinetic_energy=1.0,
        local_ionic_energy=-2.0,
        ts_plus_eloc_energy=-1.0,
        delta_ts_vs_legacy_ha=-0.1,
        delta_eloc_vs_legacy_ha=0.05,
        delta_ts_plus_eloc_vs_legacy_ha=-0.05,
        delta_ts_vs_legacy_mha=-100.0,
        delta_eloc_vs_legacy_mha=50.0,
        delta_ts_plus_eloc_vs_legacy_mha=-50.0,
        reference_offset_ha=-0.2,
        reference_offset_mha=-200.0,
        improvement_vs_legacy_ha=0.01,
        improvement_vs_legacy_mha=10.0,
    )

    assert point.is_fair_point is True
    assert point.geometry_summary.near_core_min_spacing_bohr == 0.10
    assert point.improvement_vs_legacy_mha == 10.0
