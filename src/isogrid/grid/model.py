"""Minimal data models for a structured adaptive grid."""

from __future__ import annotations

from dataclasses import dataclass

Coordinate3D = tuple[float, float, float]


@dataclass(frozen=True)
class AxisStretchSpec:
    """One-dimensional structured stretch about a reference center."""

    label: str
    lower_offset: float
    upper_offset: float
    stretch: float

    def __post_init__(self) -> None:
        if self.lower_offset >= self.upper_offset:
            raise ValueError("Axis bounds must satisfy lower_offset < upper_offset.")
        if not (self.lower_offset < 0.0 < self.upper_offset):
            raise ValueError("The reference center must lie inside the axis bounds.")
        if self.stretch < 0.0:
            raise ValueError("Axis stretch must be non-negative.")

    def physical_bounds(self, center_coordinate: float) -> tuple[float, float]:
        """Return the physical bounds for this axis."""

        return (
            center_coordinate + self.lower_offset,
            center_coordinate + self.upper_offset,
        )


@dataclass(frozen=True)
class StructuredGridSpec:
    """Minimal specification for a structured adaptive 3D grid."""

    name: str
    description: str
    nx: int
    ny: int
    nz: int
    reference_center: Coordinate3D
    unit: str
    x_axis: AxisStretchSpec
    y_axis: AxisStretchSpec
    z_axis: AxisStretchSpec

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2 or self.nz < 2:
            raise ValueError("Each grid direction must have at least two points.")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the logical grid shape."""

        return (self.nx, self.ny, self.nz)
