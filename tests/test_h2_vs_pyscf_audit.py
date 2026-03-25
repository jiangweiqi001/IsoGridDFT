"""Minimal smoke tests for the H2 vs PySCF audit layer."""

from isogrid.audit.h2_vs_pyscf_audit import AuditParameterSummary
from isogrid.audit.h2_vs_pyscf_audit import H2GapComparison
from isogrid.audit.h2_vs_pyscf_audit import H2ParameterScanResult
from isogrid.audit.h2_vs_pyscf_audit import H2SpinComparisonResult


def test_construct_spin_comparison_result() -> None:
    parameter_summary = AuditParameterSummary(
        grid_shape=(51, 51, 51),
        min_cell_widths_bohr=(0.132, 0.132, 0.137),
        density_tolerance=2.5e-3,
        eigensolver_tolerance=5.0e-3,
        mixing=0.6,
        max_iterations=8,
        eigensolver_ncv=20,
    )
    result = H2SpinComparisonResult(
        spin_state_label="singlet",
        isogrid_total_energy=-1.1,
        pyscf_total_energy=-1.0,
        absolute_error_ha=-0.1,
        absolute_error_mha=-100.0,
        isogrid_converged=True,
        pyscf_converged=True,
        isogrid_iteration_count=5,
        lowest_isogrid_eigenvalue=-0.2,
        parameter_summary=parameter_summary,
    )

    assert result.spin_state_label == "singlet"
    assert result.absolute_error_mha == -100.0
    assert result.parameter_summary.grid_shape == (51, 51, 51)


def test_construct_gap_and_scan_result() -> None:
    parameter_summary = AuditParameterSummary(
        grid_shape=(41, 41, 41),
        min_cell_widths_bohr=(0.165, 0.165, 0.171),
        density_tolerance=1.0e-3,
        eigensolver_tolerance=1.0e-3,
        mixing=0.6,
        max_iterations=10,
        eigensolver_ncv=20,
    )
    comparison = H2SpinComparisonResult(
        spin_state_label="triplet",
        isogrid_total_energy=-0.7,
        pyscf_total_energy=-0.6,
        absolute_error_ha=-0.1,
        absolute_error_mha=-100.0,
        isogrid_converged=False,
        pyscf_converged=True,
        isogrid_iteration_count=8,
        lowest_isogrid_eigenvalue=-0.05,
        parameter_summary=parameter_summary,
    )
    gap = H2GapComparison(
        isogrid_gap_ha=0.3,
        pyscf_gap_ha=0.4,
        gap_error_ha=-0.1,
        gap_error_mha=-100.0,
        ordering_consistent=True,
        lower_spin_state_isogrid="singlet",
        lower_spin_state_pyscf="singlet",
    )
    scan_result = H2ParameterScanResult(
        label="baseline",
        description="Current default audit point.",
        spin_state_label="triplet",
        comparison=comparison,
    )

    assert gap.ordering_consistent is True
    assert scan_result.comparison.parameter_summary.eigensolver_tolerance == 1.0e-3
    assert scan_result.label == "baseline"
