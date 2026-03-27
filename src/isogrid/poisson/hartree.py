"""Hartree helpers built on the first-stage open-boundary Poisson slice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.ops import integrate_field
from isogrid.ops import validate_orbital_field

from .open_boundary import OpenBoundaryPoissonResult
from .open_boundary import solve_open_boundary_poisson

_NEGATIVE_DENSITY_TOLERANCE = 1.0e-14
GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry


@dataclass(frozen=True)
class HartreeEvaluation:
    """Resolved Hartree potential, action, and energy for one orbital field."""

    rho: np.ndarray
    potential: np.ndarray
    action: np.ndarray
    energy: float
    poisson_result: OpenBoundaryPoissonResult | None


def validate_density_field(
    rho: np.ndarray,
    grid_geometry: GridGeometryLike,
    name: str = "rho",
) -> np.ndarray:
    """Validate a 3D density field on the current structured grid."""

    values = validate_orbital_field(rho, grid_geometry=grid_geometry, name=name).astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(values < -_NEGATIVE_DENSITY_TOLERANCE):
        raise ValueError(f"{name} must be non-negative up to roundoff.")
    return np.maximum(values, 0.0)


def solve_hartree_potential(
    grid_geometry: GridGeometryLike,
    rho: np.ndarray,
    multipole_order: int = 2,
    tolerance: float = 1.0e-8,
    max_iterations: int = 400,
    solver: str = "auto",
    backend: str = "python",
) -> OpenBoundaryPoissonResult:
    """Solve the first-stage Hartree potential from a total electron density."""

    density = validate_density_field(rho, grid_geometry=grid_geometry)
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"python", "jax"}:
        raise ValueError(
            "backend must be `python` or `jax`; "
            f"received `{backend}`."
        )
    if normalized_backend == "jax":
        if not isinstance(grid_geometry, MonitorGridGeometry):
            raise ValueError(
                "The JAX Hartree backend currently supports only the monitor-grid path."
            )
        from .poisson_jax import solve_open_boundary_poisson_monitor_jax

        poisson_result, _ = solve_open_boundary_poisson_monitor_jax(
            grid_geometry=grid_geometry,
            rho=density,
            multipole_order=multipole_order,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        return poisson_result
    return solve_open_boundary_poisson(
        grid_geometry=grid_geometry,
        rho=density,
        multipole_order=multipole_order,
        tolerance=tolerance,
        max_iterations=max_iterations,
        solver=solver,
    )


def _resolve_hartree_potential_array(
    grid_geometry: GridGeometryLike,
    hartree_potential: OpenBoundaryPoissonResult | np.ndarray,
) -> tuple[np.ndarray, OpenBoundaryPoissonResult | None]:
    if isinstance(hartree_potential, OpenBoundaryPoissonResult):
        return hartree_potential.potential, hartree_potential
    return (
        validate_orbital_field(
            hartree_potential,
            grid_geometry=grid_geometry,
            name="hartree_potential",
        ).astype(np.float64),
        None,
    )


def evaluate_hartree_energy(
    rho: np.ndarray,
    grid_geometry: GridGeometryLike,
    hartree_potential: OpenBoundaryPoissonResult | np.ndarray,
) -> float:
    """Return the first-stage Hartree energy E_H = 1/2 int rho v_H."""

    density = validate_density_field(rho, grid_geometry=grid_geometry)
    potential, _ = _resolve_hartree_potential_array(
        grid_geometry=grid_geometry,
        hartree_potential=hartree_potential,
    )
    return 0.5 * float(integrate_field(density * potential, grid_geometry=grid_geometry))


def build_hartree_action(
    psi: np.ndarray,
    grid_geometry: GridGeometryLike,
    hartree_potential: OpenBoundaryPoissonResult | np.ndarray,
) -> np.ndarray:
    """Return the Hartree action v_H * psi for one orbital field."""

    field = validate_orbital_field(psi, grid_geometry=grid_geometry)
    potential, _ = _resolve_hartree_potential_array(
        grid_geometry=grid_geometry,
        hartree_potential=hartree_potential,
    )
    return potential * field


def evaluate_hartree_terms(
    psi: np.ndarray,
    grid_geometry: GridGeometryLike,
    rho: np.ndarray,
    hartree_potential: OpenBoundaryPoissonResult | np.ndarray | None = None,
    multipole_order: int = 2,
    tolerance: float = 1.0e-8,
    max_iterations: int = 400,
    solver: str = "auto",
) -> HartreeEvaluation:
    """Resolve the Hartree potential, action, and energy for one orbital field."""

    density = validate_density_field(rho, grid_geometry=grid_geometry)
    if hartree_potential is None:
        poisson_result = solve_hartree_potential(
            grid_geometry=grid_geometry,
            rho=density,
            multipole_order=multipole_order,
            tolerance=tolerance,
            max_iterations=max_iterations,
            solver=solver,
        )
        potential = poisson_result.potential
    else:
        potential, poisson_result = _resolve_hartree_potential_array(
            grid_geometry=grid_geometry,
            hartree_potential=hartree_potential,
        )

    action = build_hartree_action(
        psi=psi,
        grid_geometry=grid_geometry,
        hartree_potential=potential,
    )
    energy = evaluate_hartree_energy(
        rho=density,
        grid_geometry=grid_geometry,
        hartree_potential=potential,
    )
    return HartreeEvaluation(
        rho=density,
        potential=potential,
        action=action,
        energy=energy,
        poisson_result=poisson_result,
    )
