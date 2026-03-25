"""Minimal smoke tests for the H2 Hartree tail-recheck audit."""

from importlib import import_module

from isogrid.audit.h2_hartree_tail_recheck_audit import H2HartreeTailRecheckPoint
from isogrid.audit.h2_monitor_grid_poisson_operator_audit import ScalarFieldSummary


def test_h2_hartree_tail_recheck_module_imports() -> None:
    module = import_module("isogrid.audit.h2_hartree_tail_recheck_audit")

    assert hasattr(module, "run_h2_hartree_tail_recheck_audit")
    assert hasattr(module, "print_h2_hartree_tail_recheck_summary")


def test_construct_h2_hartree_tail_recheck_point() -> None:
    point = H2HartreeTailRecheckPoint(
        geometry_point_label="baseline",
        grid_parameter_summary="shape=(67,67,81), box=(8,8,10)",
        shape=(67, 67, 81),
        box_half_extents_bohr=(8.0, 8.0, 10.0),
        hartree_energy=1.7657,
        potential_summary=ScalarFieldSummary(minimum=0.1, maximum=2.4, mean=0.3, rms=0.5),
        residual_summary=ScalarFieldSummary(minimum=-1.0e-8, maximum=1.0e-8, mean=0.0, rms=5.0e-9),
        negative_interior_fraction=0.0,
        far_field_potential_mean=0.20,
        far_field_residual_rms=1.4e-9,
        far_field_negative_potential_fraction=0.0,
        centerline_far_field_potential_mean=0.28,
        hartree_delta_vs_legacy_mha=17.086,
        hartree_delta_vs_baseline_mha=0.0,
        solver_method="scipy_bicgstab_monitor",
        solver_reported_residual_max=4.8e-8,
    )

    assert point.geometry_point_label == "baseline"
    assert point.hartree_energy == 1.7657
    assert point.residual_summary.rms == 5.0e-9
    assert point.far_field_negative_potential_fraction == 0.0
