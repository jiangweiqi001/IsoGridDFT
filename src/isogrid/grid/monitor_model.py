"""Data models for the next-generation 3D atom-centered monitor grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Coordinate3D = tuple[float, float, float]
BoxBounds3D = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class NearCoreElementParameters:
    """Element-level near-core parameters for the 3D monitor field."""

    element: str
    near_core_radius: float
    local_radius: float
    projector_radius: float
    patch_radius: float
    kinetic_weight: float
    local_weight: float
    projector_weight: float
    kinetic_exponent: float
    local_exponent: float
    projector_exponent: float
    monitor_cap: float

    def __post_init__(self) -> None:
        if self.near_core_radius <= 0.0:
            raise ValueError("near_core_radius must be positive.")
        if self.local_radius <= 0.0:
            raise ValueError("local_radius must be positive.")
        if self.projector_radius <= 0.0:
            raise ValueError("projector_radius must be positive.")
        if self.patch_radius <= 0.0:
            raise ValueError("patch_radius must be positive.")
        if self.kinetic_exponent <= 0.0 or self.local_exponent <= 0.0 or self.projector_exponent <= 0.0:
            raise ValueError("All monitor exponents must be positive.")
        if self.monitor_cap <= 1.0:
            raise ValueError("monitor_cap must be larger than 1.")


@dataclass(frozen=True)
class MonitorPatchInterface:
    """Minimal interface placeholder for an atom-centered auxiliary fine patch."""

    atom_index: int
    element: str
    center: Coordinate3D
    patch_radius: float
    purposes: tuple[str, ...]
    relation_to_main_grid: str
    implemented_purposes: tuple[str, ...] = ()
    implemented: bool = False


@dataclass(frozen=True)
class AtomicMonitorContribution:
    """One atom's resolved 3D monitor contribution on the current grid."""

    atom_index: int
    element: str
    position: Coordinate3D
    parameters: NearCoreElementParameters
    kinetic_component: np.ndarray
    local_component: np.ndarray
    projector_component: np.ndarray
    raw_total: np.ndarray
    capped_total: np.ndarray


@dataclass(frozen=True)
class GlobalMonitorField:
    """Global 3D monitor field assembled from all atomic contributions."""

    values: np.ndarray
    raw_values: np.ndarray
    atomic_contributions: tuple[AtomicMonitorContribution, ...]
    baseline_value: float
    minimum_value: float
    maximum_value: float


@dataclass(frozen=True)
class MonitorCellLocalQuadrature:
    """Auditably explicit logical-cell quadrature on the mapped monitor grid."""

    subcell_divisions: tuple[int, int, int]
    logical_cell_volume: float
    subcell_logical_volume: float
    sample_points: tuple[tuple[float, float, float], ...]
    sample_weights: np.ndarray


@dataclass(frozen=True)
class MonitorGridSpec:
    """Specification for a full 3D monitor-driven structured grid."""

    name: str
    description: str
    nx: int
    ny: int
    nz: int
    unit: str
    box_bounds: BoxBounds3D
    logical_bounds: BoxBounds3D
    element_parameters: dict[str, NearCoreElementParameters]
    harmonic_outer_iterations: int
    harmonic_inner_iterations: int
    harmonic_tolerance: float
    harmonic_relaxation: float
    inner_relaxation: float
    monitor_smoothing: float

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 3 or self.nz < 3:
            raise ValueError("Monitor grids require at least three points per direction.")
        if self.harmonic_outer_iterations <= 0 or self.harmonic_inner_iterations <= 0:
            raise ValueError("Harmonic iteration counts must be positive.")
        if self.harmonic_tolerance <= 0.0:
            raise ValueError("harmonic_tolerance must be positive.")
        if not (0.0 < self.harmonic_relaxation <= 1.0):
            raise ValueError("harmonic_relaxation must satisfy 0 < value <= 1.")
        if not (0.0 < self.inner_relaxation <= 1.0):
            raise ValueError("inner_relaxation must satisfy 0 < value <= 1.")
        if not (0.0 <= self.monitor_smoothing < 1.0):
            raise ValueError("monitor_smoothing must satisfy 0 <= value < 1.")
        if not self.element_parameters:
            raise ValueError("element_parameters must not be empty.")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)


@dataclass(frozen=True)
class MonitorGridQualityReport:
    """Basic quality indicators for the generated monitor grid."""

    min_jacobian: float
    max_jacobian: float
    has_nonpositive_jacobian: bool
    min_cell_volume: float
    max_cell_volume: float
    mean_near_atom_spacing: float
    mean_far_field_spacing: float
    near_to_far_spacing_ratio: float


@dataclass(frozen=True)
class MonitorGridGeometry:
    """Generated 3D monitor-driven structured grid geometry."""

    spec: MonitorGridSpec
    logical_x: np.ndarray
    logical_y: np.ndarray
    logical_z: np.ndarray
    x_points: np.ndarray
    y_points: np.ndarray
    z_points: np.ndarray
    covariant_basis: np.ndarray
    jacobian: np.ndarray
    metric_tensor: np.ndarray
    inverse_metric_tensor: np.ndarray
    cell_volumes: np.ndarray
    spacing_x: np.ndarray
    spacing_y: np.ndarray
    spacing_z: np.ndarray
    spacing_measure: np.ndarray
    monitor_field: GlobalMonitorField
    patch_interfaces: tuple[MonitorPatchInterface, ...]
    quality_report: MonitorGridQualityReport
