"""Quantitative H2 error audit against the current PySCF reference baseline.

This module compares the first minimal IsoGridDFT H2 SCF single-point loop to
the current PySCF audit reference under the same nominal physical model:

- H2 at R = 1.4 Bohr
- gth-pade pseudopotential
- lda,vwn LSDA
- UKS reference path for both singlet and triplet

The goal here is not to claim final acceptance. It is to quantify the present
gap to the PySCF baseline and to provide one small parameter scan that helps
separate solver-side error from grid/discretization error.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_default_h2_grid_spec
from isogrid.grid import build_grid_geometry
from isogrid.scf import run_h2_minimal_scf

from .pyscf_h2_reference import ReferenceResult
from .pyscf_h2_reference import run_reference_case


_AUDIT_NOTE = (
    "IsoGridDFT is still using the first minimal H2 SCF closed loop. "
    "This audit is the first formal error baseline against PySCF, not the "
    "final 1 mHa acceptance report."
)
_DEFAULT_SCAN_GRID_SHAPE = (41, 41, 41)
_DEFAULT_SCAN_DENSITY_TOLERANCE = 1.0e-3
_DEFAULT_SCAN_EIGENSOLVER_TOLERANCE = 1.0e-3
_ONE_MHA = 1.0


@dataclass(frozen=True)
class AuditParameterSummary:
    """Compact summary of the key isogrid-side audit parameters."""

    grid_shape: tuple[int, int, int]
    min_cell_widths_bohr: tuple[float, float, float]
    density_tolerance: float
    eigensolver_tolerance: float
    mixing: float
    max_iterations: int
    eigensolver_ncv: int


@dataclass(frozen=True)
class H2SpinComparisonResult:
    """Quantitative comparison for one H2 candidate spin state."""

    spin_state_label: str
    isogrid_total_energy: float
    pyscf_total_energy: float
    absolute_error_ha: float
    absolute_error_mha: float
    isogrid_converged: bool
    pyscf_converged: bool
    isogrid_iteration_count: int
    lowest_isogrid_eigenvalue: float | None
    parameter_summary: AuditParameterSummary


@dataclass(frozen=True)
class H2GapComparison:
    """Singlet-triplet gap comparison against the PySCF baseline."""

    isogrid_gap_ha: float
    pyscf_gap_ha: float
    gap_error_ha: float
    gap_error_mha: float
    ordering_consistent: bool
    lower_spin_state_isogrid: str
    lower_spin_state_pyscf: str


@dataclass(frozen=True)
class H2ParameterScanResult:
    """One small audit point used to localize present error sources."""

    label: str
    description: str
    spin_state_label: str
    comparison: H2SpinComparisonResult


@dataclass(frozen=True)
class H2VsPySCFAuditResult:
    """Top-level H2 audit result for both spin candidates and the small scan."""

    singlet: H2SpinComparisonResult
    triplet: H2SpinComparisonResult
    gap: H2GapComparison
    scan_results: tuple[H2ParameterScanResult, ...]
    reference_model_summary: str
    note: str


def _find_reference_result(
    reference_results: tuple[ReferenceResult, ...],
    spin_state_label: str,
) -> ReferenceResult:
    normalized = spin_state_label.strip().lower()
    for result in reference_results:
        if result.spin_state.label.lower() == normalized:
            return result
    available = ", ".join(result.spin_state.label for result in reference_results)
    raise ValueError(
        f"Spin state `{spin_state_label}` is not available in the reference results. "
        f"Available states: {available}."
    )


def _build_parameter_summary(
    grid_geometry: StructuredGridGeometry,
    *,
    density_tolerance: float,
    eigensolver_tolerance: float,
    mixing: float,
    max_iterations: int,
    eigensolver_ncv: int,
) -> AuditParameterSummary:
    return AuditParameterSummary(
        grid_shape=grid_geometry.spec.shape,
        min_cell_widths_bohr=(
            float(grid_geometry.cell_widths_x.min()),
            float(grid_geometry.cell_widths_y.min()),
            float(grid_geometry.cell_widths_z.min()),
        ),
        density_tolerance=float(density_tolerance),
        eigensolver_tolerance=float(eigensolver_tolerance),
        mixing=float(mixing),
        max_iterations=int(max_iterations),
        eigensolver_ncv=int(eigensolver_ncv),
    )


def _build_scan_grid_geometry(
    shape: tuple[int, int, int],
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> StructuredGridGeometry:
    base_spec = build_default_h2_grid_spec(case=case)
    grid_spec = replace(
        base_spec,
        name=f"{base_spec.name}_{shape[0]}x{shape[1]}x{shape[2]}",
        description=(
            f"{base_spec.description} "
            f"Audit scan variant with shape {shape[0]} x {shape[1]} x {shape[2]}."
        ),
        nx=shape[0],
        ny=shape[1],
        nz=shape[2],
    )
    return build_grid_geometry(grid_spec)


def run_spin_state_comparison(
    spin_state_label: str,
    *,
    reference_results: tuple[ReferenceResult, ...],
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: StructuredGridGeometry | None = None,
    max_iterations: int = 8,
    mixing: float = 0.6,
    density_tolerance: float = 2.5e-3,
    eigensolver_tolerance: float = 5.0e-3,
    eigensolver_ncv: int = 20,
) -> H2SpinComparisonResult:
    """Run one H2 isogrid vs PySCF comparison for a fixed spin state."""

    if grid_geometry is None:
        grid_geometry = build_default_h2_grid_geometry(case=case)

    scf_result = run_h2_minimal_scf(
        spin_label=spin_state_label,
        case=case,
        grid_geometry=grid_geometry,
        max_iterations=max_iterations,
        mixing=mixing,
        density_tolerance=density_tolerance,
        eigensolver_tolerance=eigensolver_tolerance,
        eigensolver_ncv=eigensolver_ncv,
    )
    reference_result = _find_reference_result(reference_results, spin_state_label)
    absolute_error_ha = scf_result.energy.total - reference_result.total_energy
    lowest_eigenvalue = (
        float(scf_result.eigenvalues_up[0])
        if scf_result.eigenvalues_up.size > 0
        else None
    )
    return H2SpinComparisonResult(
        spin_state_label=spin_state_label,
        isogrid_total_energy=float(scf_result.energy.total),
        pyscf_total_energy=float(reference_result.total_energy),
        absolute_error_ha=float(absolute_error_ha),
        absolute_error_mha=float(absolute_error_ha * 1000.0),
        isogrid_converged=bool(scf_result.converged),
        pyscf_converged=bool(reference_result.converged),
        isogrid_iteration_count=int(scf_result.iteration_count),
        lowest_isogrid_eigenvalue=lowest_eigenvalue,
        parameter_summary=_build_parameter_summary(
            grid_geometry,
            density_tolerance=density_tolerance,
            eigensolver_tolerance=eigensolver_tolerance,
            mixing=mixing,
            max_iterations=max_iterations,
            eigensolver_ncv=eigensolver_ncv,
        ),
    )


def _build_gap_comparison(
    singlet: H2SpinComparisonResult,
    triplet: H2SpinComparisonResult,
) -> H2GapComparison:
    isogrid_gap = triplet.isogrid_total_energy - singlet.isogrid_total_energy
    pyscf_gap = triplet.pyscf_total_energy - singlet.pyscf_total_energy
    gap_error = isogrid_gap - pyscf_gap
    lower_isogrid = "singlet" if singlet.isogrid_total_energy <= triplet.isogrid_total_energy else "triplet"
    lower_pyscf = "singlet" if singlet.pyscf_total_energy <= triplet.pyscf_total_energy else "triplet"
    return H2GapComparison(
        isogrid_gap_ha=float(isogrid_gap),
        pyscf_gap_ha=float(pyscf_gap),
        gap_error_ha=float(gap_error),
        gap_error_mha=float(gap_error * 1000.0),
        ordering_consistent=bool(lower_isogrid == lower_pyscf),
        lower_spin_state_isogrid=lower_isogrid,
        lower_spin_state_pyscf=lower_pyscf,
    )


def run_minimal_parameter_scan(
    *,
    reference_results: tuple[ReferenceResult, ...],
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> tuple[H2ParameterScanResult, ...]:
    """Run one small singlet-focused scan to localize dominant error sources."""

    baseline_geometry = build_default_h2_grid_geometry(case=case)
    coarse_geometry = _build_scan_grid_geometry(_DEFAULT_SCAN_GRID_SHAPE, case=case)
    scan_points = (
        H2ParameterScanResult(
            label="baseline",
            description=(
                "Current H2 default grid and current minimal SCF development tolerances."
            ),
            spin_state_label="singlet",
            comparison=run_spin_state_comparison(
                "singlet",
                reference_results=reference_results,
                case=case,
                grid_geometry=baseline_geometry,
            ),
        ),
        H2ParameterScanResult(
            label="tight_solver",
            description=(
                "Same grid, tighter density and eigensolver tolerances; used to gauge "
                "solver-side sensitivity without changing the discretization."
            ),
            spin_state_label="singlet",
            comparison=run_spin_state_comparison(
                "singlet",
                reference_results=reference_results,
                case=case,
                grid_geometry=baseline_geometry,
                max_iterations=10,
                density_tolerance=_DEFAULT_SCAN_DENSITY_TOLERANCE,
                eigensolver_tolerance=_DEFAULT_SCAN_EIGENSOLVER_TOLERANCE,
            ),
        ),
        H2ParameterScanResult(
            label="coarser_grid",
            description=(
                "Reduced grid shape with the same physical box and stretch; used to gauge "
                "how much the present error responds to a visibly coarser center spacing."
            ),
            spin_state_label="singlet",
            comparison=run_spin_state_comparison(
                "singlet",
                reference_results=reference_results,
                case=case,
                grid_geometry=coarse_geometry,
            ),
        ),
    )
    return scan_points


def run_h2_vs_pyscf_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2VsPySCFAuditResult:
    """Run the quantitative H2 audit against the current PySCF baseline."""

    reference_results = run_reference_case(case=case)
    singlet = run_spin_state_comparison("singlet", reference_results=reference_results, case=case)
    triplet = run_spin_state_comparison("triplet", reference_results=reference_results, case=case)
    gap = _build_gap_comparison(singlet, triplet)
    reference_model = case.reference_model
    return H2VsPySCFAuditResult(
        singlet=singlet,
        triplet=triplet,
        gap=gap,
        scan_results=run_minimal_parameter_scan(reference_results=reference_results, case=case),
        reference_model_summary=(
            f"{reference_model.mean_field.upper()} / {reference_model.pseudo} / "
            f"{reference_model.basis} / {reference_model.xc}"
        ),
        note=_AUDIT_NOTE,
    )


def _format_target_gap(error_mha: float) -> str:
    distance = abs(error_mha) - _ONE_MHA
    if distance <= 0.0:
        return "meets the 1 mHa target at this audit point"
    return f"misses the 1 mHa target by {distance:.3f} mHa"


def _print_spin_comparison(result: H2SpinComparisonResult) -> None:
    summary = result.parameter_summary
    print(f"spin state: {result.spin_state_label}")
    print(f"  isogrid converged: {result.isogrid_converged}")
    print(f"  pyscf converged: {result.pyscf_converged}")
    print(f"  isogrid total energy [Ha]: {result.isogrid_total_energy:.12f}")
    print(f"  pyscf total energy [Ha]:   {result.pyscf_total_energy:.12f}")
    print(f"  absolute error [Ha]: {result.absolute_error_ha:+.12f}")
    print(f"  absolute error [mHa]: {result.absolute_error_mha:+.3f}")
    print(f"  1 mHa status: {_format_target_gap(result.absolute_error_mha)}")
    print(f"  isogrid iterations: {result.isogrid_iteration_count}")
    if result.lowest_isogrid_eigenvalue is not None:
        print(f"  lowest isogrid eigenvalue [Ha]: {result.lowest_isogrid_eigenvalue:.12f}")
    print(
        "  parameter summary: "
        f"grid={summary.grid_shape}, "
        f"min_d=({summary.min_cell_widths_bohr[0]:.3f}, "
        f"{summary.min_cell_widths_bohr[1]:.3f}, "
        f"{summary.min_cell_widths_bohr[2]:.3f}) Bohr, "
        f"density_tol={summary.density_tolerance:.1e}, "
        f"eigensolver_tol={summary.eigensolver_tolerance:.1e}"
    )


def print_h2_vs_pyscf_summary(result: H2VsPySCFAuditResult) -> None:
    """Print the compact quantitative H2 vs PySCF audit summary."""

    print("IsoGridDFT H2 vs PySCF quantitative audit")
    print(f"benchmark: {H2_BENCHMARK_CASE.name}")
    print(f"geometry: H2, R = 1.4 Bohr")
    print(f"reference model: {result.reference_model_summary}")
    print(f"note: {result.note}")
    print()
    _print_spin_comparison(result.singlet)
    print()
    _print_spin_comparison(result.triplet)
    print()
    print("singlet-triplet relative comparison")
    print(f"  isogrid gap [Ha]: {result.gap.isogrid_gap_ha:.12f}")
    print(f"  pyscf gap [Ha]:   {result.gap.pyscf_gap_ha:.12f}")
    print(f"  gap error [Ha]: {result.gap.gap_error_ha:+.12f}")
    print(f"  gap error [mHa]: {result.gap.gap_error_mha:+.3f}")
    print(f"  gap 1 mHa status: {_format_target_gap(result.gap.gap_error_mha)}")
    print(
        "  ordering consistent: "
        f"{result.gap.ordering_consistent} "
        f"(isogrid={result.gap.lower_spin_state_isogrid}, "
        f"pyscf={result.gap.lower_spin_state_pyscf})"
    )
    print()
    print("minimal singlet parameter scan")
    for scan_result in result.scan_results:
        comparison = scan_result.comparison
        summary = comparison.parameter_summary
        print(f"  scan point: {scan_result.label}")
        print(f"    description: {scan_result.description}")
        print(f"    total energy [Ha]: {comparison.isogrid_total_energy:.12f}")
        print(f"    reference [Ha]:    {comparison.pyscf_total_energy:.12f}")
        print(f"    error [mHa]: {comparison.absolute_error_mha:+.3f}")
        print(
            "    parameters: "
            f"grid={summary.grid_shape}, "
            f"min_d=({summary.min_cell_widths_bohr[0]:.3f}, "
            f"{summary.min_cell_widths_bohr[1]:.3f}, "
            f"{summary.min_cell_widths_bohr[2]:.3f}) Bohr, "
            f"density_tol={summary.density_tolerance:.1e}, "
            f"eigensolver_tol={summary.eigensolver_tolerance:.1e}"
        )


def main() -> int:
    """Run the H2 isogrid-vs-PySCF quantitative audit."""

    result = run_h2_vs_pyscf_audit()
    print_h2_vs_pyscf_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
