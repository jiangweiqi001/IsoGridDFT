"""Structured adaptive grid geometry and mapping helpers."""

from __future__ import annotations

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.config import BenchmarkCase

from .geometry import StructuredGridGeometry
from .geometry import build_grid_geometry
from .mapping import AxisMapping
from .mapping import build_axis_mapping
from .mapping import build_axis_mappings
from .mapping import build_grid_point_coordinates
from .mapping import logical_axis_coordinates
from .monitor_builder import build_default_co_monitor_grid
from .monitor_builder import build_default_h2_monitor_grid
from .monitor_builder import build_default_h2o_monitor_grid
from .monitor_builder import build_default_n2_monitor_grid
from .monitor_builder import build_default_near_core_element_parameters
from .monitor_builder import build_monitor_grid_for_case
from .monitor_builder import build_monitor_grid_spec_for_case
from .monitor_geometry import build_reference_box_coordinates
from .monitor_geometry import evaluate_global_monitor_field
from .monitor_geometry import generate_monitor_grid_geometry
from .monitor_model import AtomicMonitorContribution
from .monitor_model import GlobalMonitorField
from .monitor_model import MonitorGridGeometry
from .monitor_model import MonitorGridQualityReport
from .monitor_model import MonitorGridSpec
from .monitor_model import MonitorPatchInterface
from .monitor_model import NearCoreElementParameters
from .model import AxisStretchSpec
from .model import StructuredGridSpec

H2_DEFAULT_GRID_SHAPE = (51, 51, 51)
H2_DEFAULT_GRID_HALF_EXTENTS_BOHR = (8.0, 8.0, 10.0)
H2_DEFAULT_GRID_STRETCH = (2.5, 2.5, 2.8)


def compute_geometry_reference_center(case: BenchmarkCase) -> tuple[float, float, float]:
    """Compute the simple geometric center of the benchmark nuclei."""

    atoms = case.geometry.atoms
    if not atoms:
        raise ValueError("Benchmark geometry must contain at least one atom.")

    count = float(len(atoms))
    center_x = sum(atom.position[0] for atom in atoms) / count
    center_y = sum(atom.position[1] for atom in atoms) / count
    center_z = sum(atom.position[2] for atom in atoms) / count
    return (center_x, center_y, center_z)


def build_default_h2_grid_spec(case: BenchmarkCase = H2_BENCHMARK_CASE) -> StructuredGridSpec:
    """Build the first structured grid baseline for the default H2 benchmark."""

    half_extent_x, half_extent_y, half_extent_z = H2_DEFAULT_GRID_HALF_EXTENTS_BOHR
    stretch_x, stretch_y, stretch_z = H2_DEFAULT_GRID_STRETCH
    reference_center = compute_geometry_reference_center(case)
    nx, ny, nz = H2_DEFAULT_GRID_SHAPE

    return StructuredGridSpec(
        name="h2_r1p4_structured_grid",
        description=(
            "First geometry-driven structured grid baseline for H2. "
            "The mapping is separable and uses per-axis sinh stretching; "
            "this is a formal default, not a final adaptive strategy."
        ),
        nx=nx,
        ny=ny,
        nz=nz,
        reference_center=reference_center,
        unit=case.geometry.unit,
        x_axis=AxisStretchSpec(
            label="x",
            lower_offset=-half_extent_x,
            upper_offset=half_extent_x,
            stretch=stretch_x,
        ),
        y_axis=AxisStretchSpec(
            label="y",
            lower_offset=-half_extent_y,
            upper_offset=half_extent_y,
            stretch=stretch_y,
        ),
        z_axis=AxisStretchSpec(
            label="z",
            lower_offset=-half_extent_z,
            upper_offset=half_extent_z,
            stretch=stretch_z,
        ),
    )


def build_default_h2_grid_geometry(case: BenchmarkCase = H2_BENCHMARK_CASE) -> StructuredGridGeometry:
    """Build the geometry objects for the default H2 structured grid."""

    return build_grid_geometry(build_default_h2_grid_spec(case=case))


__all__ = [
    "AtomicMonitorContribution",
    "AxisMapping",
    "AxisStretchSpec",
    "GlobalMonitorField",
    "StructuredGridGeometry",
    "StructuredGridSpec",
    "H2_DEFAULT_GRID_SHAPE",
    "H2_DEFAULT_GRID_HALF_EXTENTS_BOHR",
    "H2_DEFAULT_GRID_STRETCH",
    "MonitorGridGeometry",
    "MonitorGridQualityReport",
    "MonitorGridSpec",
    "MonitorPatchInterface",
    "NearCoreElementParameters",
    "build_axis_mapping",
    "build_axis_mappings",
    "build_default_co_monitor_grid",
    "build_default_h2_grid_geometry",
    "build_default_h2_grid_spec",
    "build_default_h2_monitor_grid",
    "build_default_h2o_monitor_grid",
    "build_default_n2_monitor_grid",
    "build_default_near_core_element_parameters",
    "build_grid_geometry",
    "build_grid_point_coordinates",
    "build_monitor_grid_for_case",
    "build_monitor_grid_spec_for_case",
    "build_reference_box_coordinates",
    "compute_geometry_reference_center",
    "evaluate_global_monitor_field",
    "generate_monitor_grid_geometry",
    "logical_axis_coordinates",
]
