"""H2 singlet audit for the first A-grid `T_s + E_loc,ion` reconnect."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_default_h2_monitor_grid
from isogrid.ops import apply_legacy_kinetic_operator
from isogrid.ops import apply_monitor_grid_kinetic_operator
from isogrid.ops import integrate_field
from isogrid.pseudo import evaluate_legacy_local_ionic_potential
from isogrid.pseudo import evaluate_monitor_grid_local_ionic_potential

from .pyscf_h2_reference import run_reference_spin_state

GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry
_SINGLET_OCCUPATION = 2.0


@dataclass(frozen=True)
class H2GridEnergyGeometrySummary:
    """Compact geometry summary for one H2 `T_s + E_loc,ion` audit point."""

    grid_type: str
    grid_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    min_spacing_estimate_bohr: float
    near_atom_spacing_bohr: float
    far_field_spacing_bohr: float
    min_jacobian: float
    max_jacobian: float
    center_line_local_potential_center: float
    center_line_local_potential_near_atom: float


@dataclass(frozen=True)
class H2TsElocGridResult:
    """Resolved `T_s + E_loc,ion` audit result on one grid family."""

    grid_type: str
    geometry_summary: H2GridEnergyGeometrySummary
    kinetic_energy: float
    local_ionic_energy: float
    ts_plus_eloc_energy: float
    reference_pyscf_total_energy: float
    reference_offset_ha: float
    reference_offset_mha: float
    improvement_vs_legacy_ha: float
    improvement_vs_legacy_mha: float


@dataclass(frozen=True)
class H2MonitorGridTsElocAuditResult:
    """Top-level H2 singlet comparison between legacy and A-grid."""

    legacy: H2TsElocGridResult
    monitor_grid: H2TsElocGridResult
    reference_model_summary: str
    note: str


def _find_h2_singlet(case: BenchmarkCase):
    for spin_state in case.spin_states:
        if spin_state.label.lower() == "singlet":
            return spin_state
    raise ValueError(f"`{case.name}` does not define a singlet state.")


def _geometry_center(case: BenchmarkCase) -> tuple[float, float, float]:
    atoms = case.geometry.atoms
    count = float(len(atoms))
    return (
        sum(atom.position[0] for atom in atoms) / count,
        sum(atom.position[1] for atom in atoms) / count,
        sum(atom.position[2] for atom in atoms) / count,
    )


def _build_h2_bonding_trial_orbital(
    case: BenchmarkCase,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    atom_fields = []
    for atom in case.geometry.atoms:
        dx = grid_geometry.x_points - atom.position[0]
        dy = grid_geometry.y_points - atom.position[1]
        dz = grid_geometry.z_points - atom.position[2]
        atom_fields.append(np.exp(-0.8 * (dx * dx + dy * dy + dz * dz)))
    orbital = np.asarray(atom_fields[0] + atom_fields[1], dtype=np.float64)
    norm = float(np.sqrt(integrate_field(orbital * orbital, grid_geometry=grid_geometry)))
    if norm <= 0.0:
        raise ValueError("The H2 bonding trial orbital must have positive norm.")
    return orbital / norm


def _box_half_extents_bohr(grid_geometry: GridGeometryLike) -> tuple[float, float, float]:
    if isinstance(grid_geometry, MonitorGridGeometry):
        bounds = grid_geometry.spec.box_bounds
        return (
            0.5 * (bounds[0][1] - bounds[0][0]),
            0.5 * (bounds[1][1] - bounds[1][0]),
            0.5 * (bounds[2][1] - bounds[2][0]),
        )
    return (
        max(abs(grid_geometry.spec.x_axis.lower_offset), abs(grid_geometry.spec.x_axis.upper_offset)),
        max(abs(grid_geometry.spec.y_axis.lower_offset), abs(grid_geometry.spec.y_axis.upper_offset)),
        max(abs(grid_geometry.spec.z_axis.lower_offset), abs(grid_geometry.spec.z_axis.upper_offset)),
    )


def _spacing_measure_on_legacy_grid(grid_geometry: StructuredGridGeometry) -> np.ndarray:
    return (
        grid_geometry.cell_widths_x[:, None, None]
        + grid_geometry.cell_widths_y[None, :, None]
        + grid_geometry.cell_widths_z[None, None, :]
    ) / 3.0


def _near_far_spacing_summary(
    case: BenchmarkCase,
    grid_geometry: GridGeometryLike,
) -> tuple[float, float]:
    center = _geometry_center(case)
    nearest_atom_distance = np.full(grid_geometry.spec.shape, np.inf, dtype=np.float64)
    near_mask = np.zeros(grid_geometry.spec.shape, dtype=bool)

    for atom in case.geometry.atoms:
        dx = grid_geometry.x_points - atom.position[0]
        dy = grid_geometry.y_points - atom.position[1]
        dz = grid_geometry.z_points - atom.position[2]
        radius = np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float64)
        nearest_atom_distance = np.minimum(nearest_atom_distance, radius)
        near_mask |= radius <= 1.25

    if isinstance(grid_geometry, MonitorGridGeometry):
        spacing_measure = grid_geometry.spacing_measure
    else:
        spacing_measure = _spacing_measure_on_legacy_grid(grid_geometry)

    dx_center = grid_geometry.x_points - center[0]
    dy_center = grid_geometry.y_points - center[1]
    dz_center = grid_geometry.z_points - center[2]
    radius_from_center = np.sqrt(dx_center * dx_center + dy_center * dy_center + dz_center * dz_center, dtype=np.float64)
    far_mask = radius_from_center >= max(_box_half_extents_bohr(grid_geometry)) * 0.75
    if not np.any(far_mask):
        far_mask = nearest_atom_distance >= np.quantile(nearest_atom_distance, 0.75)

    return (
        float(np.mean(spacing_measure[near_mask])),
        float(np.mean(spacing_measure[far_mask])),
    )


def _jacobian_range(grid_geometry: GridGeometryLike) -> tuple[float, float]:
    if isinstance(grid_geometry, MonitorGridGeometry):
        jacobian = grid_geometry.jacobian
    else:
        jacobian = grid_geometry.point_jacobian
    return float(np.min(jacobian)), float(np.max(jacobian))


def _min_spacing_estimate(grid_geometry: GridGeometryLike) -> float:
    if isinstance(grid_geometry, MonitorGridGeometry):
        return float(np.min(grid_geometry.spacing_measure))
    return float(
        min(
            np.min(grid_geometry.cell_widths_x),
            np.min(grid_geometry.cell_widths_y),
            np.min(grid_geometry.cell_widths_z),
        )
    )


def _center_line_local_samples(
    case: BenchmarkCase,
    grid_geometry: GridGeometryLike,
    local_potential: np.ndarray,
) -> tuple[float, float]:
    center = _geometry_center(case)
    radius_to_center = (
        (grid_geometry.x_points - center[0]) ** 2
        + (grid_geometry.y_points - center[1]) ** 2
        + (grid_geometry.z_points - center[2]) ** 2
    )
    center_index = np.unravel_index(np.argmin(radius_to_center), grid_geometry.spec.shape)
    center_sample = float(local_potential[center_index])

    first_atom = case.geometry.atoms[0]
    radius_to_atom = (
        (grid_geometry.x_points - first_atom.position[0]) ** 2
        + (grid_geometry.y_points - first_atom.position[1]) ** 2
        + (grid_geometry.z_points - first_atom.position[2]) ** 2
    )
    atom_index = np.unravel_index(np.argmin(radius_to_atom), grid_geometry.spec.shape)
    atom_sample = float(local_potential[atom_index])
    return center_sample, atom_sample


def _geometry_summary(
    case: BenchmarkCase,
    grid_type: str,
    grid_geometry: GridGeometryLike,
    local_potential: np.ndarray,
) -> H2GridEnergyGeometrySummary:
    near_spacing, far_spacing = _near_far_spacing_summary(case, grid_geometry)
    min_jacobian, max_jacobian = _jacobian_range(grid_geometry)
    center_sample, atom_sample = _center_line_local_samples(case, grid_geometry, local_potential)
    return H2GridEnergyGeometrySummary(
        grid_type=grid_type,
        grid_shape=grid_geometry.spec.shape,
        box_half_extents_bohr=_box_half_extents_bohr(grid_geometry),
        min_spacing_estimate_bohr=_min_spacing_estimate(grid_geometry),
        near_atom_spacing_bohr=near_spacing,
        far_field_spacing_bohr=far_spacing,
        min_jacobian=min_jacobian,
        max_jacobian=max_jacobian,
        center_line_local_potential_center=center_sample,
        center_line_local_potential_near_atom=atom_sample,
    )


def evaluate_h2_singlet_ts_eloc_on_legacy_grid(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: StructuredGridGeometry | None = None,
) -> tuple[float, float, float, H2GridEnergyGeometrySummary]:
    """Evaluate `T_s + E_loc,ion` on the legacy H2 grid."""

    if grid_geometry is None:
        grid_geometry = build_default_h2_grid_geometry(case=case)
    orbital = _build_h2_bonding_trial_orbital(case=case, grid_geometry=grid_geometry)
    kinetic_action = apply_legacy_kinetic_operator(orbital, grid_geometry=grid_geometry)
    local_evaluation = evaluate_legacy_local_ionic_potential(case=case, grid_geometry=grid_geometry)
    rho_total = _SINGLET_OCCUPATION * np.abs(orbital) ** 2
    kinetic_energy = _SINGLET_OCCUPATION * float(
        np.real_if_close(integrate_field(orbital * kinetic_action, grid_geometry=grid_geometry))
    )
    local_ionic_energy = float(
        integrate_field(rho_total * local_evaluation.total_local_potential, grid_geometry=grid_geometry)
    )
    summary = _geometry_summary(
        case=case,
        grid_type="legacy",
        grid_geometry=grid_geometry,
        local_potential=local_evaluation.total_local_potential,
    )
    return kinetic_energy, local_ionic_energy, kinetic_energy + local_ionic_energy, summary


def evaluate_h2_singlet_ts_eloc_on_monitor_grid(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
) -> tuple[float, float, float, H2GridEnergyGeometrySummary]:
    """Evaluate `T_s + E_loc,ion` on the new A-grid."""

    if grid_geometry is None:
        grid_geometry = build_default_h2_monitor_grid()
    orbital = _build_h2_bonding_trial_orbital(case=case, grid_geometry=grid_geometry)
    kinetic_action = apply_monitor_grid_kinetic_operator(orbital, grid_geometry=grid_geometry)
    local_evaluation = evaluate_monitor_grid_local_ionic_potential(case=case, grid_geometry=grid_geometry)
    rho_total = _SINGLET_OCCUPATION * np.abs(orbital) ** 2
    kinetic_energy = _SINGLET_OCCUPATION * float(
        np.real_if_close(integrate_field(orbital * kinetic_action, grid_geometry=grid_geometry))
    )
    local_ionic_energy = float(
        integrate_field(rho_total * local_evaluation.total_local_potential, grid_geometry=grid_geometry)
    )
    summary = _geometry_summary(
        case=case,
        grid_type="monitor_a_grid",
        grid_geometry=grid_geometry,
        local_potential=local_evaluation.total_local_potential,
    )
    return kinetic_energy, local_ionic_energy, kinetic_energy + local_ionic_energy, summary


def _build_result(
    grid_type: str,
    geometry_summary: H2GridEnergyGeometrySummary,
    kinetic_energy: float,
    local_ionic_energy: float,
    ts_plus_eloc_energy: float,
    reference_pyscf_total_energy: float,
    legacy_reference_offset_ha: float,
) -> H2TsElocGridResult:
    reference_offset_ha = ts_plus_eloc_energy - reference_pyscf_total_energy
    improvement = abs(legacy_reference_offset_ha) - abs(reference_offset_ha)
    return H2TsElocGridResult(
        grid_type=grid_type,
        geometry_summary=geometry_summary,
        kinetic_energy=float(kinetic_energy),
        local_ionic_energy=float(local_ionic_energy),
        ts_plus_eloc_energy=float(ts_plus_eloc_energy),
        reference_pyscf_total_energy=float(reference_pyscf_total_energy),
        reference_offset_ha=float(reference_offset_ha),
        reference_offset_mha=float(reference_offset_ha * 1000.0),
        improvement_vs_legacy_ha=float(improvement),
        improvement_vs_legacy_mha=float(improvement * 1000.0),
    )


def run_h2_monitor_grid_ts_eloc_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2MonitorGridTsElocAuditResult:
    """Compare legacy vs A-grid `T_s + E_loc,ion` for H2 singlet."""

    singlet = _find_h2_singlet(case)
    reference_result = run_reference_spin_state(case=case, spin_state=singlet)
    legacy_t, legacy_eloc, legacy_sum, legacy_summary = evaluate_h2_singlet_ts_eloc_on_legacy_grid(case=case)
    legacy_offset = legacy_sum - reference_result.total_energy
    monitor_t, monitor_eloc, monitor_sum, monitor_summary = evaluate_h2_singlet_ts_eloc_on_monitor_grid(case=case)
    reference_model = case.reference_model
    return H2MonitorGridTsElocAuditResult(
        legacy=_build_result(
            "legacy",
            legacy_summary,
            legacy_t,
            legacy_eloc,
            legacy_sum,
            reference_result.total_energy,
            legacy_offset,
        ),
        monitor_grid=_build_result(
            "monitor_a_grid",
            monitor_summary,
            monitor_t,
            monitor_eloc,
            monitor_sum,
            reference_result.total_energy,
            legacy_offset,
        ),
        reference_model_summary=(
            f"{reference_model.mean_field.upper()} / {reference_model.pseudo} / "
            f"{reference_model.basis} / {reference_model.xc}"
        ),
        note=(
            "This audit reconnects only T_s and E_loc,ion to the new A-grid. "
            "Nonlocal, Hartree, XC, eigensolver, and SCF are still on the legacy path."
        ),
    )


def _print_grid_result(result: H2TsElocGridResult) -> None:
    summary = result.geometry_summary
    print(f"grid type: {result.grid_type}")
    print(f"  grid shape: {summary.grid_shape}")
    print(
        "  box half-extents [Bohr]: "
        f"({summary.box_half_extents_bohr[0]:.3f}, "
        f"{summary.box_half_extents_bohr[1]:.3f}, "
        f"{summary.box_half_extents_bohr[2]:.3f})"
    )
    print(f"  min spacing estimate [Bohr]: {summary.min_spacing_estimate_bohr:.6f}")
    print(
        "  near/far spacing [Bohr]: "
        f"{summary.near_atom_spacing_bohr:.6f} / {summary.far_field_spacing_bohr:.6f}"
    )
    print(
        "  jacobian range: "
        f"[{summary.min_jacobian:.6e}, {summary.max_jacobian:.6e}]"
    )
    print(f"  T_s [Ha]: {result.kinetic_energy:.12f}")
    print(f"  E_loc,ion [Ha]: {result.local_ionic_energy:.12f}")
    print(f"  T_s + E_loc,ion [Ha]: {result.ts_plus_eloc_energy:.12f}")
    print(f"  offset vs PySCF singlet total [Ha]: {result.reference_offset_ha:+.12f}")
    print(f"  offset vs PySCF singlet total [mHa]: {result.reference_offset_mha:+.3f}")
    print(f"  improvement vs legacy [mHa]: {result.improvement_vs_legacy_mha:+.3f}")
    print(
        "  center-line local potential samples [Ha]: "
        f"center={summary.center_line_local_potential_center:.12f}, "
        f"near_atom={summary.center_line_local_potential_near_atom:.12f}"
    )


def print_h2_monitor_grid_ts_eloc_summary(result: H2MonitorGridTsElocAuditResult) -> None:
    """Print the compact H2 singlet legacy-vs-A-grid `T_s + E_loc,ion` summary."""

    print("IsoGridDFT H2 singlet legacy-vs-A-grid T_s + E_loc,ion audit")
    print(f"reference model: {result.reference_model_summary}")
    print(f"note: {result.note}")
    print()
    _print_grid_result(result.legacy)
    print()
    _print_grid_result(result.monitor_grid)


def main() -> int:
    result = run_h2_monitor_grid_ts_eloc_audit()
    print_h2_monitor_grid_ts_eloc_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
