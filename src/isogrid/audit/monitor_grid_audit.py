"""Audit entrypoint for the new 3D monitor-driven main grid core."""

from __future__ import annotations

from dataclasses import dataclass

from isogrid.config import CO_AUDIT_CASE
from isogrid.config import H2O_AUDIT_CASE
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.config import N2_AUDIT_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_default_co_monitor_grid
from isogrid.grid import build_default_h2_monitor_grid
from isogrid.grid import build_default_h2o_monitor_grid
from isogrid.grid import build_default_n2_monitor_grid


@dataclass(frozen=True)
class MonitorGridAuditResult:
    """Compact audit summary for one generated monitor grid."""

    molecule_name: str
    monitor_min: float
    monitor_max: float
    grid_shape: tuple[int, int, int]
    min_jacobian: float
    max_jacobian: float
    min_cell_volume: float
    max_cell_volume: float
    near_atom_spacing: float
    far_field_spacing: float
    near_to_far_ratio: float
    has_nonpositive_jacobian: bool


def _summarize_geometry(molecule_name: str, geometry: MonitorGridGeometry) -> MonitorGridAuditResult:
    quality = geometry.quality_report
    return MonitorGridAuditResult(
        molecule_name=molecule_name,
        monitor_min=geometry.monitor_field.minimum_value,
        monitor_max=geometry.monitor_field.maximum_value,
        grid_shape=geometry.spec.shape,
        min_jacobian=quality.min_jacobian,
        max_jacobian=quality.max_jacobian,
        min_cell_volume=quality.min_cell_volume,
        max_cell_volume=quality.max_cell_volume,
        near_atom_spacing=quality.mean_near_atom_spacing,
        far_field_spacing=quality.mean_far_field_spacing,
        near_to_far_ratio=quality.near_to_far_spacing_ratio,
        has_nonpositive_jacobian=quality.has_nonpositive_jacobian,
    )


def run_monitor_grid_audit() -> tuple[MonitorGridAuditResult, ...]:
    """Generate the new monitor grid for the four audit molecules."""

    cases = (
        (H2_BENCHMARK_CASE.geometry.name, build_default_h2_monitor_grid()),
        (N2_AUDIT_CASE.geometry.name, build_default_n2_monitor_grid()),
        (CO_AUDIT_CASE.geometry.name, build_default_co_monitor_grid()),
        (H2O_AUDIT_CASE.geometry.name, build_default_h2o_monitor_grid()),
    )
    return tuple(_summarize_geometry(name, geometry) for name, geometry in cases)


def print_monitor_grid_audit(results: tuple[MonitorGridAuditResult, ...]) -> None:
    """Print the compact monitor-grid audit report."""

    print("IsoGridDFT monitor-grid core audit")
    print("note: this audits only the new 3D monitor-driven grid core.")
    print("downstream physical operators are not reconnected yet.")
    for result in results:
        print(f"molecule: {result.molecule_name}")
        print(
            "  monitor range: "
            f"[{result.monitor_min:.6f}, {result.monitor_max:.6f}]"
        )
        print(f"  grid shape: {result.grid_shape}")
        print(
            "  jacobian range: "
            f"[{result.min_jacobian:.6e}, {result.max_jacobian:.6e}]"
        )
        print(
            "  cell volume range: "
            f"[{result.min_cell_volume:.6e}, {result.max_cell_volume:.6e}]"
        )
        print(
            "  spacing near atom vs far field [Bohr]: "
            f"{result.near_atom_spacing:.6f} vs {result.far_field_spacing:.6f}"
        )
        print(f"  near/far spacing ratio: {result.near_to_far_ratio:.6f}")
        print(f"  has nonpositive jacobian: {result.has_nonpositive_jacobian}")


def main() -> int:
    results = run_monitor_grid_audit()
    print_monitor_grid_audit(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
