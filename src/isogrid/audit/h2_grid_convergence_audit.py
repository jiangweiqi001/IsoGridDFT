"""Grid/box convergence audit for the current H2 singlet SCF path.

This module is intentionally narrow. It does not change the physical model or
the SCF workflow. It only scans a small set of geometry-discretization choices
around the current H2 singlet default point so we can separate:

- grid-resolution sensitivity
- finite-domain / open-boundary sensitivity

The two scan families are organized as follows:

1. grid-shape scan:
   keep the physical box fixed and vary the logical grid shape
2. box-half-extent scan:
   vary the physical box and adjust the companion grid shape so the center
   spacing stays close to the current baseline; this keeps the box scan focused
   on domain/open-boundary effects instead of folding in a large resolution jump

This is a first geometric-discretization audit, not a final convergence claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import AxisStretchSpec
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_default_h2_grid_spec
from isogrid.grid import build_grid_geometry
from isogrid.scf import H2ScfResult
from isogrid.scf import SinglePointEnergyComponents
from isogrid.scf import run_h2_minimal_scf

from .pyscf_h2_reference import run_reference_spin_state


_AUDIT_NOTE = (
    "This is the first formal H2 singlet geometry-discretization audit. "
    "It localizes present grid/domain error trends, but it is not yet a final "
    "convergence proof or a 1 mHa acceptance statement."
)
_DEFAULT_SCF_PARAMETERS = {
    "max_iterations": 8,
    "mixing": 0.6,
    "density_tolerance": 2.5e-3,
    "eigensolver_tolerance": 5.0e-3,
    "eigensolver_ncv": 20,
}
_GRID_SCAN_POINTS = (
    ("grid_41", "Coarser logical grid at the default physical box.", (41, 41, 41), (8.0, 8.0, 10.0)),
    ("grid_51", "Current default H2 audit point.", (51, 51, 51), (8.0, 8.0, 10.0)),
    ("grid_61", "Finer logical grid at the default physical box.", (61, 61, 61), (8.0, 8.0, 10.0)),
)
_BOX_SCAN_POINTS = (
    (
        "box_7_9",
        "Smaller physical box with a companion grid chosen to keep the center spacing near baseline.",
        (45, 45, 45),
        (7.0, 7.0, 9.0),
    ),
    (
        "box_8_10",
        "Current default H2 audit point.",
        (51, 51, 51),
        (8.0, 8.0, 10.0),
    ),
    (
        "box_9_11",
        "Larger physical box with a companion grid chosen to keep the center spacing near baseline.",
        (57, 57, 57),
        (9.0, 9.0, 11.0),
    ),
)


@dataclass(frozen=True)
class EnergyComponentDrift:
    """Per-component drift relative to the current baseline point."""

    kinetic: float
    local_ionic: float
    nonlocal_ionic: float
    hartree: float
    xc: float
    ion_ion_repulsion: float
    total: float


@dataclass(frozen=True)
class H2GridScanParameters:
    """Geometry-discretization parameters for one audit point."""

    label: str
    family: str
    description: str
    grid_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    min_cell_widths_bohr: tuple[float, float, float]


@dataclass(frozen=True)
class H2GridConvergenceScanPoint:
    """One H2 singlet geometry-discretization audit point."""

    parameters: H2GridScanParameters
    isogrid_total_energy: float
    pyscf_total_energy: float
    total_error_ha: float
    total_error_mha: float
    isogrid_converged: bool
    pyscf_converged: bool
    iteration_count: int
    energy_components: SinglePointEnergyComponents
    baseline_drift: EnergyComponentDrift
    dominant_component_drifts_mha: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class H2GridConvergenceAuditResult:
    """Top-level H2 singlet grid/domain audit result."""

    baseline: H2GridConvergenceScanPoint
    grid_shape_scan: tuple[H2GridConvergenceScanPoint, ...]
    box_half_extent_scan: tuple[H2GridConvergenceScanPoint, ...]
    reference_model_summary: str
    note: str


def _find_singlet_spin_state(case: BenchmarkCase):
    for spin_state in case.spin_states:
        if spin_state.label.lower() == "singlet":
            return spin_state
    raise ValueError(f"`{case.name}` does not define a singlet spin state.")


def _build_grid_geometry_variant(
    *,
    label: str,
    description: str,
    shape: tuple[int, int, int],
    half_extents_bohr: tuple[float, float, float],
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> StructuredGridGeometry:
    base_spec = build_default_h2_grid_spec(case=case)
    half_extent_x, half_extent_y, half_extent_z = half_extents_bohr
    grid_spec = replace(
        base_spec,
        name=f"{base_spec.name}_{label}",
        description=f"{base_spec.description} {description}",
        nx=shape[0],
        ny=shape[1],
        nz=shape[2],
        x_axis=AxisStretchSpec(
            label="x",
            lower_offset=-half_extent_x,
            upper_offset=half_extent_x,
            stretch=base_spec.x_axis.stretch,
        ),
        y_axis=AxisStretchSpec(
            label="y",
            lower_offset=-half_extent_y,
            upper_offset=half_extent_y,
            stretch=base_spec.y_axis.stretch,
        ),
        z_axis=AxisStretchSpec(
            label="z",
            lower_offset=-half_extent_z,
            upper_offset=half_extent_z,
            stretch=base_spec.z_axis.stretch,
        ),
    )
    return build_grid_geometry(grid_spec)


def _build_parameters(
    *,
    label: str,
    family: str,
    description: str,
    grid_geometry: StructuredGridGeometry,
) -> H2GridScanParameters:
    x_axis = grid_geometry.spec.x_axis
    y_axis = grid_geometry.spec.y_axis
    z_axis = grid_geometry.spec.z_axis
    return H2GridScanParameters(
        label=label,
        family=family,
        description=description,
        grid_shape=grid_geometry.spec.shape,
        box_half_extents_bohr=(
            float(max(abs(x_axis.lower_offset), abs(x_axis.upper_offset))),
            float(max(abs(y_axis.lower_offset), abs(y_axis.upper_offset))),
            float(max(abs(z_axis.lower_offset), abs(z_axis.upper_offset))),
        ),
        min_cell_widths_bohr=(
            float(grid_geometry.cell_widths_x.min()),
            float(grid_geometry.cell_widths_y.min()),
            float(grid_geometry.cell_widths_z.min()),
        ),
    )


def _energy_drift(
    current: SinglePointEnergyComponents,
    baseline: SinglePointEnergyComponents,
) -> EnergyComponentDrift:
    return EnergyComponentDrift(
        kinetic=float(current.kinetic - baseline.kinetic),
        local_ionic=float(current.local_ionic - baseline.local_ionic),
        nonlocal_ionic=float(current.nonlocal_ionic - baseline.nonlocal_ionic),
        hartree=float(current.hartree - baseline.hartree),
        xc=float(current.xc - baseline.xc),
        ion_ion_repulsion=float(current.ion_ion_repulsion - baseline.ion_ion_repulsion),
        total=float(current.total - baseline.total),
    )


def _dominant_component_drifts(
    drift: EnergyComponentDrift,
    *,
    top_n: int = 3,
) -> tuple[tuple[str, float], ...]:
    component_map = {
        "T_s": drift.kinetic,
        "E_loc,ion": drift.local_ionic,
        "E_nl,ion": drift.nonlocal_ionic,
        "E_H": drift.hartree,
        "E_xc": drift.xc,
        "E_II": drift.ion_ion_repulsion,
    }
    ordered = sorted(component_map.items(), key=lambda item: abs(item[1]), reverse=True)
    return tuple((name, value * 1000.0) for name, value in ordered[:top_n])


def _run_singlet_point(
    *,
    label: str,
    family: str,
    description: str,
    grid_geometry: StructuredGridGeometry,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> tuple[H2GridScanParameters, H2ScfResult, float, bool]:
    singlet = _find_singlet_spin_state(case)
    scf_result = run_h2_minimal_scf(
        "singlet",
        case=case,
        grid_geometry=grid_geometry,
        max_iterations=_DEFAULT_SCF_PARAMETERS["max_iterations"],
        mixing=_DEFAULT_SCF_PARAMETERS["mixing"],
        density_tolerance=_DEFAULT_SCF_PARAMETERS["density_tolerance"],
        eigensolver_tolerance=_DEFAULT_SCF_PARAMETERS["eigensolver_tolerance"],
        eigensolver_ncv=_DEFAULT_SCF_PARAMETERS["eigensolver_ncv"],
    )
    reference_result = run_reference_spin_state(case=case, spin_state=singlet)
    return (
        _build_parameters(
            label=label,
            family=family,
            description=description,
            grid_geometry=grid_geometry,
        ),
        scf_result,
        float(reference_result.total_energy),
        bool(reference_result.converged),
    )


def _assemble_scan_point(
    parameters: H2GridScanParameters,
    scf_result: H2ScfResult,
    pyscf_total_energy: float,
    pyscf_converged: bool,
    baseline_energy: SinglePointEnergyComponents,
) -> H2GridConvergenceScanPoint:
    total_error_ha = scf_result.energy.total - pyscf_total_energy
    drift = _energy_drift(scf_result.energy, baseline_energy)
    return H2GridConvergenceScanPoint(
        parameters=parameters,
        isogrid_total_energy=float(scf_result.energy.total),
        pyscf_total_energy=float(pyscf_total_energy),
        total_error_ha=float(total_error_ha),
        total_error_mha=float(total_error_ha * 1000.0),
        isogrid_converged=bool(scf_result.converged),
        pyscf_converged=bool(pyscf_converged),
        iteration_count=int(scf_result.iteration_count),
        energy_components=scf_result.energy,
        baseline_drift=drift,
        dominant_component_drifts_mha=_dominant_component_drifts(drift),
    )


def run_h2_grid_convergence_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2GridConvergenceAuditResult:
    """Run the first H2 singlet grid/domain convergence audit."""

    baseline_geometry = build_default_h2_grid_geometry(case=case)
    baseline_parameters, baseline_scf, baseline_reference, baseline_reference_converged = _run_singlet_point(
        label="baseline",
        family="baseline",
        description="Current default H2 audit point.",
        grid_geometry=baseline_geometry,
        case=case,
    )
    baseline_point = _assemble_scan_point(
        baseline_parameters,
        baseline_scf,
        baseline_reference,
        baseline_reference_converged,
        baseline_scf.energy,
    )

    grid_shape_scan = []
    for label, description, shape, half_extents in _GRID_SCAN_POINTS:
        geometry = _build_grid_geometry_variant(
            label=label,
            description=description,
            shape=shape,
            half_extents_bohr=half_extents,
            case=case,
        )
        parameters, scf_result, reference_total, reference_converged = _run_singlet_point(
            label=label,
            family="grid_shape",
            description=description,
            grid_geometry=geometry,
            case=case,
        )
        grid_shape_scan.append(
            _assemble_scan_point(
                parameters,
                scf_result,
                reference_total,
                reference_converged,
                baseline_scf.energy,
            )
        )

    box_half_extent_scan = []
    for label, description, shape, half_extents in _BOX_SCAN_POINTS:
        geometry = _build_grid_geometry_variant(
            label=label,
            description=description,
            shape=shape,
            half_extents_bohr=half_extents,
            case=case,
        )
        parameters, scf_result, reference_total, reference_converged = _run_singlet_point(
            label=label,
            family="box_half_extent",
            description=description,
            grid_geometry=geometry,
            case=case,
        )
        box_half_extent_scan.append(
            _assemble_scan_point(
                parameters,
                scf_result,
                reference_total,
                reference_converged,
                baseline_scf.energy,
            )
        )

    reference_model = case.reference_model
    return H2GridConvergenceAuditResult(
        baseline=baseline_point,
        grid_shape_scan=tuple(grid_shape_scan),
        box_half_extent_scan=tuple(box_half_extent_scan),
        reference_model_summary=(
            f"{reference_model.mean_field.upper()} / {reference_model.pseudo} / "
            f"{reference_model.basis} / {reference_model.xc}"
        ),
        note=_AUDIT_NOTE,
    )


def _format_one_mha_status(error_mha: float) -> str:
    distance = abs(error_mha) - 1.0
    if distance <= 0.0:
        return "meets 1 mHa at this point"
    return f"misses 1 mHa by {distance:.3f} mHa"


def _print_component_drift(drift: EnergyComponentDrift) -> None:
    print(
        "    component drift vs baseline [mHa]: "
        f"T_s={drift.kinetic * 1000.0:+.3f}, "
        f"E_loc,ion={drift.local_ionic * 1000.0:+.3f}, "
        f"E_nl,ion={drift.nonlocal_ionic * 1000.0:+.3f}, "
        f"E_H={drift.hartree * 1000.0:+.3f}, "
        f"E_xc={drift.xc * 1000.0:+.3f}, "
        f"E_II={drift.ion_ion_repulsion * 1000.0:+.3f}, "
        f"E_total={drift.total * 1000.0:+.3f}"
    )


def _print_scan_point(point: H2GridConvergenceScanPoint) -> None:
    parameters = point.parameters
    print(f"  scan point: {parameters.label}")
    print(f"    description: {parameters.description}")
    print(f"    grid shape: {parameters.grid_shape}")
    print(
        "    box half-extents [Bohr]: "
        f"({parameters.box_half_extents_bohr[0]:.1f}, "
        f"{parameters.box_half_extents_bohr[1]:.1f}, "
        f"{parameters.box_half_extents_bohr[2]:.1f})"
    )
    print(
        "    finest center spacing estimate [Bohr]: "
        f"({parameters.min_cell_widths_bohr[0]:.3f}, "
        f"{parameters.min_cell_widths_bohr[1]:.3f}, "
        f"{parameters.min_cell_widths_bohr[2]:.3f})"
    )
    print(f"    isogrid total energy [Ha]: {point.isogrid_total_energy:.12f}")
    print(f"    pyscf total energy [Ha]:   {point.pyscf_total_energy:.12f}")
    print(f"    total error [Ha]: {point.total_error_ha:+.12f}")
    print(f"    total error [mHa]: {point.total_error_mha:+.3f}")
    print(f"    1 mHa status: {_format_one_mha_status(point.total_error_mha)}")
    print(f"    isogrid converged: {point.isogrid_converged}")
    print(f"    pyscf converged: {point.pyscf_converged}")
    print(f"    scf iterations: {point.iteration_count}")
    _print_component_drift(point.baseline_drift)
    dominant = ", ".join(
        f"{name}={drift_mha:+.3f} mHa"
        for name, drift_mha in point.dominant_component_drifts_mha
    )
    print(f"    dominant drift terms: {dominant}")


def print_h2_grid_convergence_summary(result: H2GridConvergenceAuditResult) -> None:
    """Print the compact H2 singlet grid/domain convergence summary."""

    print("IsoGridDFT H2 singlet grid/domain convergence audit")
    print(f"benchmark: {H2_BENCHMARK_CASE.name}")
    print("geometry: H2, R = 1.4 Bohr, singlet only")
    print(f"reference model: {result.reference_model_summary}")
    print(f"note: {result.note}")
    print()
    print("baseline point")
    _print_scan_point(result.baseline)
    print()
    print("grid-shape scan (fixed physical box)")
    for point in result.grid_shape_scan:
        _print_scan_point(point)
    print()
    print("box-half-extent scan (companion grid keeps center spacing near baseline)")
    for point in result.box_half_extent_scan:
        _print_scan_point(point)


def main() -> int:
    """Run the H2 singlet grid/domain convergence audit."""

    result = run_h2_grid_convergence_audit()
    print_h2_grid_convergence_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
