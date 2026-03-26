"""JAX monitor-grid kinetic kernels for the local A-grid hot path."""

from __future__ import annotations

import numpy as np

from isogrid.config.runtime_jax import get_configured_jax
from isogrid.grid import MonitorGridGeometry


def _logical_spacing(logical_coordinates: np.ndarray, axis_label: str) -> float:
    coordinates = np.asarray(logical_coordinates, dtype=np.float64)
    spacings = np.diff(coordinates)
    if not np.allclose(spacings, spacings[0]):
        raise ValueError(
            f"The current JAX kinetic kernel expects a uniform logical {axis_label}-axis."
        )
    return float(spacings[0])


def _validate_monitor_geometry(grid_geometry: MonitorGridGeometry) -> None:
    if np.any(~np.isfinite(grid_geometry.jacobian)):
        raise ValueError("Monitor-grid Jacobian must be finite.")
    if np.any(grid_geometry.jacobian <= 0.0):
        raise ValueError("Monitor-grid Jacobian must stay strictly positive.")
    if np.any(~np.isfinite(grid_geometry.inverse_metric_tensor)):
        raise ValueError("Monitor-grid inverse metric tensor must be finite.")


def _validate_field(field, grid_geometry: MonitorGridGeometry):
    array = np.asarray(field)
    if array.shape != grid_geometry.spec.shape:
        raise ValueError(
            f"field must match the monitor-grid shape {grid_geometry.spec.shape}; "
            f"received {array.shape}."
        )


def _build_axis_gradient_kernel(
    *,
    spacing: float,
    axis: int,
    use_trial_boundary_fix: bool,
):
    jax = get_configured_jax()
    jnp = jax.numpy
    spacing_value = jnp.asarray(spacing, dtype=jnp.float64)

    @jax.jit
    def _gradient(field):
        moved = jnp.moveaxis(field, axis, 0)
        if moved.shape[0] < 3:
            raise ValueError("Monitor-grid JAX kernels require at least three points per axis.")
        interior = (moved[2:] - moved[:-2]) / (2.0 * spacing_value)
        if use_trial_boundary_fix:
            lower = (moved[1] - 0.0) / (2.0 * spacing_value)
            upper = (0.0 - moved[-2]) / (2.0 * spacing_value)
        else:
            lower = (-3.0 * moved[0] + 4.0 * moved[1] - moved[2]) / (2.0 * spacing_value)
            upper = (3.0 * moved[-1] - 4.0 * moved[-2] + moved[-3]) / (2.0 * spacing_value)
        gradient = jnp.concatenate(
            [
                lower[jnp.newaxis, ...],
                interior,
                upper[jnp.newaxis, ...],
            ],
            axis=0,
        )
        return jnp.moveaxis(gradient, 0, axis)

    return _gradient


def build_monitor_grid_laplacian_operator_jax(
    grid_geometry: MonitorGridGeometry,
    *,
    use_trial_boundary_fix: bool = False,
):
    """Return a jitted monitor-grid Laplacian kernel on one fixed geometry."""

    _validate_monitor_geometry(grid_geometry)
    jax = get_configured_jax()
    jnp = jax.numpy
    jacobian = jnp.asarray(grid_geometry.jacobian, dtype=jnp.float64)
    inverse_metric = jnp.asarray(grid_geometry.inverse_metric_tensor, dtype=jnp.float64)
    gradient_x = _build_axis_gradient_kernel(
        spacing=_logical_spacing(grid_geometry.logical_x, "x"),
        axis=0,
        use_trial_boundary_fix=use_trial_boundary_fix,
    )
    gradient_y = _build_axis_gradient_kernel(
        spacing=_logical_spacing(grid_geometry.logical_y, "y"),
        axis=1,
        use_trial_boundary_fix=use_trial_boundary_fix,
    )
    gradient_z = _build_axis_gradient_kernel(
        spacing=_logical_spacing(grid_geometry.logical_z, "z"),
        axis=2,
        use_trial_boundary_fix=use_trial_boundary_fix,
    )

    @jax.jit
    def _laplacian(field):
        dpsi_dxi = (
            gradient_x(field),
            gradient_y(field),
            gradient_z(field),
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
        divergence = gradient_x(flux_x) + gradient_y(flux_y) + gradient_z(flux_z)
        return divergence / jacobian

    return _laplacian


def apply_monitor_grid_laplacian_operator_jax(
    psi,
    grid_geometry: MonitorGridGeometry,
):
    """Apply the production monitor-grid Laplacian with JAX."""

    _validate_field(psi, grid_geometry)
    laplacian = build_monitor_grid_laplacian_operator_jax(
        grid_geometry,
        use_trial_boundary_fix=False,
    )
    return laplacian(get_configured_jax().numpy.asarray(psi, dtype=get_configured_jax().numpy.float64))


def apply_monitor_grid_laplacian_operator_trial_boundary_fix_jax(
    psi,
    grid_geometry: MonitorGridGeometry,
):
    """Apply the trial-fix monitor-grid Laplacian with JAX."""

    _validate_field(psi, grid_geometry)
    laplacian = build_monitor_grid_laplacian_operator_jax(
        grid_geometry,
        use_trial_boundary_fix=True,
    )
    return laplacian(get_configured_jax().numpy.asarray(psi, dtype=get_configured_jax().numpy.float64))


def build_monitor_grid_kinetic_operator_jax(
    grid_geometry: MonitorGridGeometry,
    *,
    use_trial_boundary_fix: bool = False,
):
    """Return a jitted monitor-grid kinetic kernel on one fixed geometry."""

    jax = get_configured_jax()
    laplacian = build_monitor_grid_laplacian_operator_jax(
        grid_geometry,
        use_trial_boundary_fix=use_trial_boundary_fix,
    )

    @jax.jit
    def _kinetic(field):
        return -0.5 * laplacian(field)

    return _kinetic


def apply_monitor_grid_kinetic_operator_jax(
    psi,
    grid_geometry: MonitorGridGeometry,
):
    """Apply the production monitor-grid kinetic operator with JAX."""

    _validate_field(psi, grid_geometry)
    kinetic = build_monitor_grid_kinetic_operator_jax(
        grid_geometry,
        use_trial_boundary_fix=False,
    )
    return kinetic(get_configured_jax().numpy.asarray(psi, dtype=get_configured_jax().numpy.float64))


def apply_monitor_grid_kinetic_operator_trial_boundary_fix_jax(
    psi,
    grid_geometry: MonitorGridGeometry,
):
    """Apply the trial-fix monitor-grid kinetic operator with JAX."""

    _validate_field(psi, grid_geometry)
    kinetic = build_monitor_grid_kinetic_operator_jax(
        grid_geometry,
        use_trial_boundary_fix=True,
    )
    return kinetic(get_configured_jax().numpy.asarray(psi, dtype=get_configured_jax().numpy.float64))


__all__ = [
    "apply_monitor_grid_kinetic_operator_jax",
    "apply_monitor_grid_kinetic_operator_trial_boundary_fix_jax",
    "apply_monitor_grid_laplacian_operator_jax",
    "apply_monitor_grid_laplacian_operator_trial_boundary_fix_jax",
    "build_monitor_grid_kinetic_operator_jax",
    "build_monitor_grid_laplacian_operator_jax",
]
