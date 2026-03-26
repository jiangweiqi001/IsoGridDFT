"""JAX hot kernels for the monitor-grid open-boundary Poisson path.

This module intentionally migrates only the numerically hot pieces of the
current monitor-grid Poisson route:

- monitor-grid Poisson operator apply
- one small JAX conjugate-gradient solve for the interior unknown

The physical boundary model is unchanged. Boundary values still come from the
existing open-boundary multipole approximation, which remains in the Python
audit/fallback layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config.runtime_jax import get_configured_jax
from isogrid.grid import MonitorGridGeometry
from isogrid.ops.kinetic_jax import apply_monitor_grid_laplacian_operator_jax
from isogrid.ops.kinetic_jax import build_monitor_grid_laplacian_operator_jax

from .hartree import validate_density_field
from .open_boundary import _FOUR_PI
from .open_boundary import _boundary_mask
from .open_boundary import _compute_multipole_boundary_condition
from .open_boundary import OpenBoundaryPoissonResult


@dataclass(frozen=True)
class MonitorPoissonJaxSolveDiagnostics:
    """Compact diagnostics for the JAX monitor-grid Poisson solve."""

    solver_method: str
    iteration_count: int
    residual_max: float
    converged: bool


def apply_monitor_open_boundary_poisson_operator_jax(
    potential,
    *,
    grid_geometry: MonitorGridGeometry,
):
    """Apply the monitor-grid Poisson operator `-L(v)` with JAX."""

    jax = get_configured_jax()
    jnp = jax.numpy
    values = jnp.asarray(potential, dtype=jnp.float64)
    return -apply_monitor_grid_laplacian_operator_jax(values, grid_geometry=grid_geometry)


def _build_monitor_interior_scatter(grid_geometry: MonitorGridGeometry):
    jax = get_configured_jax()
    jnp = jax.numpy
    interior_mask = ~_boundary_mask(grid_geometry.spec.shape)
    interior_indices = np.flatnonzero(interior_mask.reshape(-1))
    shape = grid_geometry.spec.shape
    size = int(interior_indices.size)
    interior_indices_jax = jnp.asarray(interior_indices, dtype=jnp.int32)

    def _scatter(interior_values):
        full = jnp.zeros(shape, dtype=jnp.float64).reshape(-1)
        scattered = full.at[interior_indices_jax].set(interior_values)
        return scattered.reshape(shape)

    def _gather(full_field):
        flattened = jnp.asarray(full_field, dtype=jnp.float64).reshape(-1)
        return flattened[interior_indices_jax]

    return interior_mask, size, _scatter, _gather


def build_monitor_open_boundary_poisson_matvec_jax(
    *,
    grid_geometry: MonitorGridGeometry,
):
    """Return the interior Poisson matvec `x -> -L(x)` for zero-boundary unknowns."""

    jax = get_configured_jax()
    _, _, scatter, gather = _build_monitor_interior_scatter(grid_geometry)
    laplacian = build_monitor_grid_laplacian_operator_jax(
        grid_geometry,
        use_trial_boundary_fix=False,
    )

    @jax.jit
    def _matvec(interior_values):
        full_field = scatter(interior_values)
        action = -laplacian(full_field)
        return gather(action)

    return _matvec


def _run_jax_cg(
    *,
    matvec,
    rhs,
    tolerance: float,
    max_iterations: int,
):
    jax = get_configured_jax()
    jnp = jax.numpy
    rhs_values = jnp.asarray(rhs, dtype=jnp.float64)
    x = jnp.zeros_like(rhs_values)
    r = rhs_values - matvec(x)
    p = r
    rr = jnp.vdot(r, r)

    converged = False
    iteration_count = 0
    residual_max = float(jnp.max(jnp.abs(r)))
    for iteration in range(1, max_iterations + 1):
        ap = matvec(p)
        denominator = jnp.vdot(p, ap)
        if float(jnp.abs(denominator)) <= 1.0e-20:
            break
        alpha = rr / denominator
        x = x + alpha * p
        r = r - alpha * ap
        residual_max = float(jnp.max(jnp.abs(r)))
        iteration_count = iteration
        if residual_max < tolerance:
            converged = True
            break
        rr_new = jnp.vdot(r, r)
        if float(jnp.abs(rr)) <= 1.0e-30:
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    return np.asarray(x, dtype=np.float64), MonitorPoissonJaxSolveDiagnostics(
        solver_method="jax_cg_monitor",
        iteration_count=iteration_count,
        residual_max=residual_max,
        converged=converged,
    )


def solve_open_boundary_poisson_monitor_jax(
    *,
    grid_geometry: MonitorGridGeometry,
    rho: np.ndarray,
    multipole_order: int = 2,
    tolerance: float = 1.0e-8,
    max_iterations: int = 400,
) -> tuple[OpenBoundaryPoissonResult, MonitorPoissonJaxSolveDiagnostics]:
    """Solve the monitor-grid open-boundary Poisson problem with JAX CG."""

    density = validate_density_field(rho, grid_geometry=grid_geometry)
    boundary_condition = _compute_multipole_boundary_condition(
        grid_geometry=grid_geometry,
        rho=density,
        multipole_order=multipole_order,
    )
    interior_mask, _, scatter, gather = _build_monitor_interior_scatter(grid_geometry)
    jax = get_configured_jax()
    jnp = jax.numpy

    boundary_field = np.array(boundary_condition.boundary_values, copy=True)
    boundary_field[interior_mask] = 0.0
    rhs_full = _FOUR_PI * density + np.asarray(
        apply_monitor_grid_laplacian_operator_jax(
            jnp.asarray(boundary_field, dtype=jnp.float64),
            grid_geometry=grid_geometry,
        ),
        dtype=np.float64,
    )
    matvec = build_monitor_open_boundary_poisson_matvec_jax(grid_geometry=grid_geometry)
    interior_solution, diagnostics = _run_jax_cg(
        matvec=matvec,
        rhs=gather(rhs_full),
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    full_interior = np.asarray(scatter(jnp.asarray(interior_solution, dtype=jnp.float64)), dtype=np.float64)
    potential = np.array(boundary_condition.boundary_values, copy=True)
    potential[interior_mask] = full_interior[interior_mask]
    result = OpenBoundaryPoissonResult(
        rho=density,
        potential=potential,
        boundary_condition=boundary_condition,
        solver_method=diagnostics.solver_method,
        solver_iterations=diagnostics.iteration_count,
        residual_max=diagnostics.residual_max,
        description=(
            "Finite-domain Poisson solve on the monitor-driven A-grid with the same "
            "free-space multipole boundary model as the NumPy/SciPy route, but using "
            "a JAX CG interior solve."
        ),
    )
    return result, diagnostics


__all__ = [
    "MonitorPoissonJaxSolveDiagnostics",
    "apply_monitor_open_boundary_poisson_operator_jax",
    "build_monitor_open_boundary_poisson_matvec_jax",
    "solve_open_boundary_poisson_monitor_jax",
]
