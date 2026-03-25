"""First-stage kinetic operators for both legacy and monitor-driven grids.

Legacy structured grid:

For the separable orthogonal mapping x = x(u), y = y(v), z = z(w), with
point scale factors h_x = dx/du, h_y = dy/dv, h_z = dz/dw, the legacy path uses

    T psi = -1/2 [
        1/h_x d/du (1/h_x dpsi/du)
      + 1/h_y d/dv (1/h_y dpsi/dv)
      + 1/h_z d/dw (1/h_z dpsi/dw)
    ]

The derivatives are discretized on the uniform logical grid with a second-order
centered flux form and face-averaged scale factors. At the outer boundary, zero
ghost cells are used.

Monitor-driven A-grid:

For the full 3D curvilinear mapping x = x(xi_1, xi_2, xi_3), the new path uses

    T psi = -1/2 * (1/J) * d/dxi_a [ J g^{ab} dpsi/dxi_b ]

where J is the Jacobian and g^{ab} is the inverse metric tensor of the current
monitor grid. The derivatives are evaluated on the uniform logical cube with
second-order finite differences; at the outer boundary, NumPy's second-order
one-sided derivatives are used. This is a first finite-domain A-grid kinetic
slice, not the final production operator.
"""

from __future__ import annotations

import numpy as np

from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry

GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry


def validate_orbital_field(
    psi: np.ndarray,
    grid_geometry: GridGeometryLike,
    name: str = "psi",
) -> np.ndarray:
    """Validate that a field matches the 3D grid shape."""

    field = np.asarray(psi)
    if field.ndim != 3:
        raise ValueError(f"{name} must be a 3D field; received ndim={field.ndim}.")
    if field.shape != grid_geometry.spec.shape:
        raise ValueError(
            f"{name} must match the grid shape {grid_geometry.spec.shape}; "
            f"received {field.shape}."
        )
    if not np.issubdtype(field.dtype, np.inexact):
        field = field.astype(np.float64)
    return field


def integrate_field(field: np.ndarray, grid_geometry: GridGeometryLike):
    """Integrate a 3D field with the point-centered cell volumes."""

    values = validate_orbital_field(field, grid_geometry=grid_geometry, name="field")
    sum_dtype = np.result_type(values.dtype, np.float64)
    return np.sum(values * grid_geometry.cell_volumes, dtype=sum_dtype)


def weighted_l2_norm(field: np.ndarray, grid_geometry: GridGeometryLike) -> float:
    """Return the weighted L2 norm of a 3D field on the structured grid."""

    values = validate_orbital_field(field, grid_geometry=grid_geometry, name="field")
    norm_squared = np.sum(np.abs(values) ** 2 * grid_geometry.cell_volumes, dtype=np.float64)
    return float(np.sqrt(norm_squared))


def _logical_spacing(logical_coordinates: np.ndarray, axis_label: str) -> float:
    coordinates = np.asarray(logical_coordinates, dtype=np.float64)
    spacings = np.diff(coordinates)
    if not np.allclose(spacings, spacings[0]):
        raise ValueError(
            f"The current kinetic operator expects a uniform logical {axis_label}-axis."
        )
    return float(spacings[0])


def _apply_axis_laplacian(
    field: np.ndarray,
    scale_factors: np.ndarray,
    logical_spacing: float,
    axis: int,
) -> np.ndarray:
    """Apply one separable axis contribution to the transformed Laplacian."""

    moved_field = np.moveaxis(field, axis, 0)
    scale = np.asarray(scale_factors, dtype=np.float64)
    dtype = np.result_type(moved_field.dtype, np.float64)

    moved_field = moved_field.astype(dtype, copy=False)
    face_scale = 0.5 * (scale[:-1] + scale[1:])
    face_view = face_scale[(slice(None),) + (None,) * (moved_field.ndim - 1)]
    scale_view = scale[(slice(None),) + (None,) * (moved_field.ndim - 1)]

    flux_minus = np.empty_like(moved_field, dtype=dtype)
    flux_plus = np.empty_like(moved_field, dtype=dtype)

    flux_minus[0] = moved_field[0] / (logical_spacing * scale[0])
    flux_minus[1:] = (moved_field[1:] - moved_field[:-1]) / (logical_spacing * face_view)

    flux_plus[:-1] = (moved_field[1:] - moved_field[:-1]) / (logical_spacing * face_view)
    flux_plus[-1] = -moved_field[-1] / (logical_spacing * scale[-1])

    axis_laplacian = (flux_plus - flux_minus) / (logical_spacing * scale_view)
    return np.moveaxis(axis_laplacian, 0, axis)


def apply_legacy_laplacian_operator(
    psi: np.ndarray,
    grid_geometry: StructuredGridGeometry,
) -> np.ndarray:
    """Apply the legacy separable transformed Laplacian to a 3D field."""

    field = validate_orbital_field(psi, grid_geometry=grid_geometry)
    laplacian = np.zeros(grid_geometry.spec.shape, dtype=np.result_type(field.dtype, np.float64))

    laplacian += _apply_axis_laplacian(
        field=field,
        scale_factors=grid_geometry.x_point_jacobian,
        logical_spacing=_logical_spacing(grid_geometry.x_logical, "x"),
        axis=0,
    )
    laplacian += _apply_axis_laplacian(
        field=field,
        scale_factors=grid_geometry.y_point_jacobian,
        logical_spacing=_logical_spacing(grid_geometry.y_logical, "y"),
        axis=1,
    )
    laplacian += _apply_axis_laplacian(
        field=field,
        scale_factors=grid_geometry.z_point_jacobian,
        logical_spacing=_logical_spacing(grid_geometry.z_logical, "z"),
        axis=2,
    )
    return laplacian


def apply_legacy_kinetic_operator(
    psi: np.ndarray,
    grid_geometry: StructuredGridGeometry,
) -> np.ndarray:
    """Apply the legacy separable kinetic operator T = -1/2 Laplacian."""

    return -0.5 * apply_legacy_laplacian_operator(psi=psi, grid_geometry=grid_geometry)


def _validate_monitor_geometry(grid_geometry: MonitorGridGeometry) -> None:
    if np.any(~np.isfinite(grid_geometry.jacobian)):
        raise ValueError("Monitor-grid Jacobian must be finite.")
    if np.any(grid_geometry.jacobian <= 0.0):
        raise ValueError("Monitor-grid Jacobian must stay strictly positive.")
    if np.any(~np.isfinite(grid_geometry.inverse_metric_tensor)):
        raise ValueError("Monitor-grid inverse metric tensor must be finite.")


def apply_monitor_grid_laplacian_operator(
    psi: np.ndarray,
    grid_geometry: MonitorGridGeometry,
) -> np.ndarray:
    """Apply the full curvilinear Laplacian on the 3D monitor grid.

    This path uses

        nabla^2 psi = (1/J) d/dxi_a [ J g^{ab} dpsi/dxi_b ]

    on the uniform logical cube of the monitor grid.
    """

    _validate_monitor_geometry(grid_geometry)
    field = validate_orbital_field(psi, grid_geometry=grid_geometry)
    logical_x = np.asarray(grid_geometry.logical_x, dtype=np.float64)
    logical_y = np.asarray(grid_geometry.logical_y, dtype=np.float64)
    logical_z = np.asarray(grid_geometry.logical_z, dtype=np.float64)
    jacobian = np.asarray(grid_geometry.jacobian, dtype=np.result_type(field.dtype, np.float64))
    inverse_metric = np.asarray(
        grid_geometry.inverse_metric_tensor,
        dtype=np.result_type(field.dtype, np.float64),
    )

    dpsi_dxi = np.gradient(
        field,
        logical_x,
        logical_y,
        logical_z,
        edge_order=2,
    )
    flux_x = jacobian * (
        inverse_metric[..., 0, 0] * dpsi_dxi[0]
        + inverse_metric[..., 0, 1] * dpsi_dxi[1]
        + inverse_metric[..., 0, 2] * dpsi_dxi[2]
    )
    flux_y = jacobian * (
        inverse_metric[..., 1, 0] * dpsi_dxi[0]
        + inverse_metric[..., 1, 1] * dpsi_dxi[1]
        + inverse_metric[..., 1, 2] * dpsi_dxi[2]
    )
    flux_z = jacobian * (
        inverse_metric[..., 2, 0] * dpsi_dxi[0]
        + inverse_metric[..., 2, 1] * dpsi_dxi[1]
        + inverse_metric[..., 2, 2] * dpsi_dxi[2]
    )
    divergence = (
        np.gradient(flux_x, logical_x, axis=0, edge_order=2)
        + np.gradient(flux_y, logical_y, axis=1, edge_order=2)
        + np.gradient(flux_z, logical_z, axis=2, edge_order=2)
    )
    return divergence / jacobian


def apply_monitor_grid_kinetic_operator(
    psi: np.ndarray,
    grid_geometry: MonitorGridGeometry,
) -> np.ndarray:
    """Apply the first A-grid kinetic operator T = -1/2 Laplacian."""

    return -0.5 * apply_monitor_grid_laplacian_operator(psi=psi, grid_geometry=grid_geometry)


def apply_laplacian_operator(psi: np.ndarray, grid_geometry: GridGeometryLike) -> np.ndarray:
    """Apply the appropriate Laplacian on either the legacy or A-grid geometry."""

    if isinstance(grid_geometry, MonitorGridGeometry):
        return apply_monitor_grid_laplacian_operator(psi=psi, grid_geometry=grid_geometry)
    return apply_legacy_laplacian_operator(psi=psi, grid_geometry=grid_geometry)


def apply_kinetic_operator(psi: np.ndarray, grid_geometry: GridGeometryLike) -> np.ndarray:
    """Apply the appropriate kinetic operator on either supported grid family."""

    if isinstance(grid_geometry, MonitorGridGeometry):
        return apply_monitor_grid_kinetic_operator(psi=psi, grid_geometry=grid_geometry)
    return apply_legacy_kinetic_operator(psi=psi, grid_geometry=grid_geometry)
