"""First-stage open-boundary Poisson solver on the structured adaptive grid.

The current route solves the finite-domain Poisson equation

    nabla^2 v(r) = -4 pi rho(r)

on the existing structured adaptive grid. The boundary values on the finite box
are not periodic. Instead, they are approximated by a free-space multipole
expansion of the density-induced potential about the grid reference center,
truncated at quadrupole order by default.

Inside the box, the same separable flux-form discretization family used by the
current structured-grid kinetic operator is reused for the Poisson operator. The
interior Dirichlet problem is solved with a SciPy BiCGSTAB linear solve when
SciPy is available, and falls back to damped Jacobi iteration otherwise.

This is a first formal open-boundary approximation for the Hartree path. It is
meant to be explicit and auditable, not the final production free-space solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.ops import apply_monitor_grid_laplacian_operator
from isogrid.ops import integrate_field
from isogrid.ops import validate_orbital_field

_FOUR_PI = 4.0 * np.pi
_JACOBI_DAMPING = 0.8
_JACOBI_CHECK_INTERVAL = 25
GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry


@dataclass(frozen=True)
class OpenBoundaryMultipoleBoundary:
    """First-stage free-space boundary approximation data for Poisson."""

    reference_center: tuple[float, float, float]
    multipole_order: int
    total_charge: float
    dipole_moment: np.ndarray
    quadrupole_tensor: np.ndarray
    boundary_values: np.ndarray
    description: str


@dataclass(frozen=True)
class OpenBoundaryPoissonResult:
    """Resolved finite-domain Poisson result with open-boundary approximation."""

    rho: np.ndarray
    potential: np.ndarray
    boundary_condition: OpenBoundaryMultipoleBoundary
    solver_method: str
    solver_iterations: int
    residual_max: float
    description: str


def _logical_spacing(logical_coordinates: np.ndarray, axis_label: str) -> float:
    coordinates = np.asarray(logical_coordinates, dtype=np.float64)
    spacings = np.diff(coordinates)
    if not np.allclose(spacings, spacings[0]):
        raise ValueError(
            f"The current open-boundary Poisson slice expects a uniform logical {axis_label}-axis."
        )
    return float(spacings[0])


def _axis_neighbor_coefficients(
    point_jacobian: np.ndarray,
    logical_coordinates: np.ndarray,
    axis_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    spacing = _logical_spacing(logical_coordinates, axis_label=axis_label)
    scale = np.asarray(point_jacobian, dtype=np.float64)
    face_inverse_scale = 1.0 / (0.5 * (scale[:-1] + scale[1:]))
    center_scale = scale[1:-1]
    coefficient_minus = face_inverse_scale[:-1] / (spacing * spacing * center_scale)
    coefficient_plus = face_inverse_scale[1:] / (spacing * spacing * center_scale)
    return coefficient_minus, coefficient_plus


def _boundary_mask(shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :, :] = True
    mask[-1, :, :] = True
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, 0] = True
    mask[:, :, -1] = True
    return mask


def _compute_multipole_boundary_condition(
    grid_geometry: GridGeometryLike,
    rho: np.ndarray,
    multipole_order: int = 2,
    reference_center: tuple[float, float, float] | None = None,
) -> OpenBoundaryMultipoleBoundary:
    if multipole_order not in (0, 1, 2):
        raise ValueError(
            "The current open-boundary multipole boundary supports only orders 0, 1, and 2; "
            f"received {multipole_order}."
        )

    if reference_center is None:
        if isinstance(grid_geometry, StructuredGridGeometry):
            reference_center = grid_geometry.spec.reference_center
        else:
            reference_center = (
                0.5 * (grid_geometry.spec.box_bounds[0][0] + grid_geometry.spec.box_bounds[0][1]),
                0.5 * (grid_geometry.spec.box_bounds[1][0] + grid_geometry.spec.box_bounds[1][1]),
                0.5 * (grid_geometry.spec.box_bounds[2][0] + grid_geometry.spec.box_bounds[2][1]),
            )

    dx = grid_geometry.x_points - reference_center[0]
    dy = grid_geometry.y_points - reference_center[1]
    dz = grid_geometry.z_points - reference_center[2]
    radius_squared = dx * dx + dy * dy + dz * dz
    radius = np.sqrt(radius_squared, dtype=np.float64)
    weights = grid_geometry.cell_volumes

    total_charge = float(integrate_field(rho, grid_geometry=grid_geometry))
    dipole_moment = np.array(
        [
            integrate_field(rho * dx, grid_geometry=grid_geometry),
            integrate_field(rho * dy, grid_geometry=grid_geometry),
            integrate_field(rho * dz, grid_geometry=grid_geometry),
        ],
        dtype=np.float64,
    )
    quadrupole_tensor = np.array(
        [
            [integrate_field(rho * (3.0 * dx * dx - radius_squared), grid_geometry=grid_geometry), integrate_field(rho * (3.0 * dx * dy), grid_geometry=grid_geometry), integrate_field(rho * (3.0 * dx * dz), grid_geometry=grid_geometry)],
            [integrate_field(rho * (3.0 * dy * dx), grid_geometry=grid_geometry), integrate_field(rho * (3.0 * dy * dy - radius_squared), grid_geometry=grid_geometry), integrate_field(rho * (3.0 * dy * dz), grid_geometry=grid_geometry)],
            [integrate_field(rho * (3.0 * dz * dx), grid_geometry=grid_geometry), integrate_field(rho * (3.0 * dz * dy), grid_geometry=grid_geometry), integrate_field(rho * (3.0 * dz * dz - radius_squared), grid_geometry=grid_geometry)],
        ],
        dtype=np.float64,
    )

    boundary_mask = _boundary_mask(grid_geometry.spec.shape)
    boundary_values = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    boundary_radius = radius[boundary_mask]
    if np.any(boundary_radius <= 0.0):
        raise ValueError("The open-boundary multipole approximation requires positive boundary radii.")

    boundary_dx = dx[boundary_mask]
    boundary_dy = dy[boundary_mask]
    boundary_dz = dz[boundary_mask]
    boundary_potential = total_charge / boundary_radius
    if multipole_order >= 1:
        boundary_potential = boundary_potential + (
            dipole_moment[0] * boundary_dx
            + dipole_moment[1] * boundary_dy
            + dipole_moment[2] * boundary_dz
        ) / (boundary_radius**3)
    if multipole_order >= 2:
        quadrupole_contraction = (
            quadrupole_tensor[0, 0] * boundary_dx * boundary_dx
            + 2.0 * quadrupole_tensor[0, 1] * boundary_dx * boundary_dy
            + 2.0 * quadrupole_tensor[0, 2] * boundary_dx * boundary_dz
            + quadrupole_tensor[1, 1] * boundary_dy * boundary_dy
            + 2.0 * quadrupole_tensor[1, 2] * boundary_dy * boundary_dz
            + quadrupole_tensor[2, 2] * boundary_dz * boundary_dz
        )
        boundary_potential = boundary_potential + 0.5 * quadrupole_contraction / (boundary_radius**5)

    boundary_values[boundary_mask] = boundary_potential
    description = (
        "Finite-domain Dirichlet boundary values from a free-space multipole expansion "
        f"truncated at order {multipole_order} about the grid reference center."
    )
    return OpenBoundaryMultipoleBoundary(
        reference_center=reference_center,
        multipole_order=multipole_order,
        total_charge=total_charge,
        dipole_moment=dipole_moment,
        quadrupole_tensor=quadrupole_tensor,
        boundary_values=boundary_values,
        description=description,
    )


def _interior_operator_coefficients(grid_geometry: StructuredGridGeometry) -> tuple[np.ndarray, ...]:
    ax_minus, ax_plus = _axis_neighbor_coefficients(
        grid_geometry.x_point_jacobian,
        grid_geometry.x_logical,
        axis_label="x",
    )
    ay_minus, ay_plus = _axis_neighbor_coefficients(
        grid_geometry.y_point_jacobian,
        grid_geometry.y_logical,
        axis_label="y",
    )
    az_minus, az_plus = _axis_neighbor_coefficients(
        grid_geometry.z_point_jacobian,
        grid_geometry.z_logical,
        axis_label="z",
    )
    diagonal = (
        ax_minus[:, None, None]
        + ax_plus[:, None, None]
        + ay_minus[None, :, None]
        + ay_plus[None, :, None]
        + az_minus[None, None, :]
        + az_plus[None, None, :]
    )
    return ax_minus, ax_plus, ay_minus, ay_plus, az_minus, az_plus, diagonal


def _solve_monitor_with_scipy_bicgstab(
    rhs: np.ndarray,
    grid_geometry: MonitorGridGeometry,
    interior_mask: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, float, str] | None:
    try:
        from scipy.sparse.linalg import LinearOperator
        from scipy.sparse.linalg import bicgstab
    except ImportError:
        return None

    shape = grid_geometry.spec.shape
    interior_size = int(np.count_nonzero(interior_mask))
    iteration_count = 0

    def matvec(vector: np.ndarray) -> np.ndarray:
        values = np.zeros(shape, dtype=np.float64)
        values[interior_mask] = np.asarray(vector, dtype=np.float64)
        action = -apply_monitor_grid_laplacian_operator(values, grid_geometry=grid_geometry)
        return np.asarray(action[interior_mask], dtype=np.float64)

    def callback(_vector: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1

    operator = LinearOperator((interior_size, interior_size), matvec=matvec, dtype=np.float64)
    solution, info = bicgstab(
        operator,
        rhs[interior_mask],
        x0=np.zeros(interior_size, dtype=np.float64),
        rtol=tolerance,
        atol=0.0,
        maxiter=max_iterations,
        callback=callback,
    )
    if info != 0:
        return None

    interior_solution = np.zeros(shape, dtype=np.float64)
    interior_solution[interior_mask] = np.asarray(solution, dtype=np.float64)
    residual_field = -apply_monitor_grid_laplacian_operator(
        interior_solution,
        grid_geometry=grid_geometry,
    ) - rhs
    residual_max = float(np.max(np.abs(residual_field[interior_mask])))
    return interior_solution, iteration_count, residual_max, "scipy_bicgstab_monitor"


def _monitor_diagonal_estimate(grid_geometry: MonitorGridGeometry) -> np.ndarray:
    dx = _logical_spacing(grid_geometry.logical_x, axis_label="x")
    dy = _logical_spacing(grid_geometry.logical_y, axis_label="y")
    dz = _logical_spacing(grid_geometry.logical_z, axis_label="z")
    inverse_metric = np.asarray(grid_geometry.inverse_metric_tensor, dtype=np.float64)
    diagonal_estimate = 2.0 * (
        inverse_metric[..., 0, 0] / (dx * dx)
        + inverse_metric[..., 1, 1] / (dy * dy)
        + inverse_metric[..., 2, 2] / (dz * dz)
    )
    return np.maximum(diagonal_estimate, 1.0e-8)


def _solve_monitor_with_jacobi(
    rhs: np.ndarray,
    grid_geometry: MonitorGridGeometry,
    interior_mask: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, float, str]:
    values = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    diagonal = _monitor_diagonal_estimate(grid_geometry)
    residual_max = np.inf

    for iteration in range(1, max_iterations + 1):
        residual = rhs + apply_monitor_grid_laplacian_operator(values, grid_geometry=grid_geometry)
        updated = np.array(values, copy=True)
        updated[interior_mask] = values[interior_mask] + _JACOBI_DAMPING * (
            residual[interior_mask] / diagonal[interior_mask]
        )
        values = updated

        if iteration % _JACOBI_CHECK_INTERVAL == 0 or iteration == max_iterations:
            residual = rhs + apply_monitor_grid_laplacian_operator(values, grid_geometry=grid_geometry)
            residual_max = float(np.max(np.abs(residual[interior_mask])))
            if residual_max < tolerance:
                return values, iteration, residual_max, "jacobi_monitor"

    return values, max_iterations, residual_max, "jacobi_monitor_maxiter"


def _solve_open_boundary_poisson_monitor(
    grid_geometry: MonitorGridGeometry,
    rho: np.ndarray,
    multipole_order: int,
    tolerance: float,
    max_iterations: int,
    solver: str,
) -> OpenBoundaryPoissonResult:
    density = validate_orbital_field(rho, grid_geometry=grid_geometry, name="rho").astype(np.float64)
    boundary_condition = _compute_multipole_boundary_condition(
        grid_geometry=grid_geometry,
        rho=density,
        multipole_order=multipole_order,
    )
    boundary_mask = _boundary_mask(grid_geometry.spec.shape)
    interior_mask = ~boundary_mask
    boundary_field = np.array(boundary_condition.boundary_values, copy=True)
    boundary_field[interior_mask] = 0.0
    # Split the full potential as v = u + b, where u vanishes on the boundary
    # and b carries only the prescribed Dirichlet boundary values. The complete
    # monitor-grid Poisson equation is
    #
    #     L(v) = -4 pi rho .
    #
    # With v = u + b and linearity of L, the interior unknown must satisfy
    #
    #     L(u) + L(b) = -4 pi rho
    #     -L(u) = 4 pi rho + L(b) .
    #
    # The monitor path previously used 4 pi rho - L(b), which made the split
    # inconsistent with the full operator identity and yielded
    # L(v) + 4 pi rho ~= 2 L(b). Keep the solver/operator unchanged and repair
    # only this RHS assembly sign.
    rhs = _FOUR_PI * density + apply_monitor_grid_laplacian_operator(
        boundary_field,
        grid_geometry=grid_geometry,
    )

    interior_solution = None
    solver_method = ""
    iteration_count = 0
    residual_max = np.inf

    if solver in {"auto", "scipy_bicgstab"}:
        scipy_result = _solve_monitor_with_scipy_bicgstab(
            rhs=rhs,
            grid_geometry=grid_geometry,
            interior_mask=interior_mask,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        if scipy_result is not None:
            interior_solution, iteration_count, residual_max, solver_method = scipy_result

    if interior_solution is None:
        interior_solution, iteration_count, residual_max, solver_method = _solve_monitor_with_jacobi(
            rhs=rhs,
            grid_geometry=grid_geometry,
            interior_mask=interior_mask,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    potential = np.array(boundary_condition.boundary_values, copy=True)
    potential[interior_mask] = interior_solution[interior_mask]
    description = (
        "Finite-domain Poisson solve on the monitor-driven A-grid with free-space "
        "multipole Dirichlet boundary data. This is the first formal A-grid "
        "open-boundary extension, not the final production solver."
    )
    return OpenBoundaryPoissonResult(
        rho=density,
        potential=potential,
        boundary_condition=boundary_condition,
        solver_method=solver_method,
        solver_iterations=iteration_count,
        residual_max=residual_max,
        description=description,
    )


def _build_poisson_rhs(
    rho: np.ndarray,
    boundary_values: np.ndarray,
    coefficients: tuple[np.ndarray, ...],
) -> np.ndarray:
    ax_minus, ax_plus, ay_minus, ay_plus, az_minus, az_plus, diagonal = coefficients
    rhs = _FOUR_PI * np.asarray(rho[1:-1, 1:-1, 1:-1], dtype=np.float64)
    rhs[0, :, :] += ax_minus[0] * boundary_values[0, 1:-1, 1:-1]
    rhs[-1, :, :] += ax_plus[-1] * boundary_values[-1, 1:-1, 1:-1]
    rhs[:, 0, :] += ay_minus[0] * boundary_values[1:-1, 0, 1:-1]
    rhs[:, -1, :] += ay_plus[-1] * boundary_values[1:-1, -1, 1:-1]
    rhs[:, :, 0] += az_minus[0] * boundary_values[1:-1, 1:-1, 0]
    rhs[:, :, -1] += az_plus[-1] * boundary_values[1:-1, 1:-1, -1]
    return rhs


def _apply_interior_operator(values: np.ndarray, coefficients: tuple[np.ndarray, ...]) -> np.ndarray:
    ax_minus, ax_plus, ay_minus, ay_plus, az_minus, az_plus, diagonal = coefficients
    interior = np.asarray(values, dtype=np.float64)
    action = diagonal * interior
    action[1:, :, :] -= ax_minus[1:, None, None] * interior[:-1, :, :]
    action[:-1, :, :] -= ax_plus[:-1, None, None] * interior[1:, :, :]
    action[:, 1:, :] -= ay_minus[None, 1:, None] * interior[:, :-1, :]
    action[:, :-1, :] -= ay_plus[None, :-1, None] * interior[:, 1:, :]
    action[:, :, 1:] -= az_minus[None, None, 1:] * interior[:, :, :-1]
    action[:, :, :-1] -= az_plus[None, None, :-1] * interior[:, :, 1:]
    return action


def _solve_with_scipy_bicgstab(
    rhs: np.ndarray,
    coefficients: tuple[np.ndarray, ...],
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, float, str] | None:
    try:
        from scipy.sparse.linalg import LinearOperator
        from scipy.sparse.linalg import bicgstab
    except ImportError:
        return None

    diagonal = coefficients[-1]
    shape = rhs.shape
    size = int(np.prod(shape))
    diagonal_flat = diagonal.reshape(-1)
    iteration_count = 0

    def matvec(vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64).reshape(shape)
        return _apply_interior_operator(values, coefficients).reshape(-1)

    def psolve(vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=np.float64) / diagonal_flat

    def callback(_vector: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1

    operator = LinearOperator((size, size), matvec=matvec, dtype=np.float64)
    preconditioner = LinearOperator((size, size), matvec=psolve, dtype=np.float64)
    solution, info = bicgstab(
        operator,
        rhs.reshape(-1),
        x0=np.zeros(size, dtype=np.float64),
        rtol=tolerance,
        atol=0.0,
        maxiter=max_iterations,
        M=preconditioner,
        callback=callback,
    )
    if info != 0:
        return None

    interior = np.asarray(solution, dtype=np.float64).reshape(shape)
    residual = _apply_interior_operator(interior, coefficients) - rhs
    residual_max = float(np.max(np.abs(residual)))
    return interior, iteration_count, residual_max, "scipy_bicgstab"


def _solve_with_jacobi(
    rhs: np.ndarray,
    coefficients: tuple[np.ndarray, ...],
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, float, str]:
    ax_minus, ax_plus, ay_minus, ay_plus, az_minus, az_plus, diagonal = coefficients
    interior = np.zeros(rhs.shape, dtype=np.float64)
    residual_max = np.inf

    for iteration in range(1, max_iterations + 1):
        updated = np.array(rhs, copy=True)
        updated[1:, :, :] += ax_minus[1:, None, None] * interior[:-1, :, :]
        updated[:-1, :, :] += ax_plus[:-1, None, None] * interior[1:, :, :]
        updated[:, 1:, :] += ay_minus[None, 1:, None] * interior[:, :-1, :]
        updated[:, :-1, :] += ay_plus[None, :-1, None] * interior[:, 1:, :]
        updated[:, :, 1:] += az_minus[None, None, 1:] * interior[:, :, :-1]
        updated[:, :, :-1] += az_plus[None, None, :-1] * interior[:, :, 1:]
        updated = updated / diagonal
        interior = _JACOBI_DAMPING * updated + (1.0 - _JACOBI_DAMPING) * interior

        if iteration % _JACOBI_CHECK_INTERVAL == 0 or iteration == max_iterations:
            residual = _apply_interior_operator(interior, coefficients) - rhs
            residual_max = float(np.max(np.abs(residual)))
            if residual_max < tolerance:
                return interior, iteration, residual_max, "jacobi"

    return interior, max_iterations, residual_max, "jacobi_maxiter"


def _solve_open_boundary_poisson_legacy(
    grid_geometry: StructuredGridGeometry,
    rho: np.ndarray,
    multipole_order: int = 2,
    tolerance: float = 1.0e-8,
    max_iterations: int = 400,
    solver: str = "auto",
) -> OpenBoundaryPoissonResult:
    """Solve the first-stage finite-domain Poisson problem with open-boundary data."""

    density = validate_orbital_field(rho, grid_geometry=grid_geometry, name="rho").astype(np.float64)
    boundary_condition = _compute_multipole_boundary_condition(
        grid_geometry=grid_geometry,
        rho=density,
        multipole_order=multipole_order,
    )
    coefficients = _interior_operator_coefficients(grid_geometry)
    rhs = _build_poisson_rhs(
        rho=density,
        boundary_values=boundary_condition.boundary_values,
        coefficients=coefficients,
    )

    interior_solution = None
    solver_method = ""
    iteration_count = 0
    residual_max = np.inf

    if solver not in {"auto", "scipy_bicgstab", "jacobi"}:
        raise ValueError(
            "The current open-boundary Poisson slice supports solver=`auto`, `scipy_bicgstab`, "
            f"or `jacobi`; received `{solver}`."
        )

    if solver in {"auto", "scipy_bicgstab"}:
        scipy_result = _solve_with_scipy_bicgstab(
            rhs=rhs,
            coefficients=coefficients,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        if scipy_result is not None:
            interior_solution, iteration_count, residual_max, solver_method = scipy_result

    if interior_solution is None:
        interior_solution, iteration_count, residual_max, solver_method = _solve_with_jacobi(
            rhs=rhs,
            coefficients=coefficients,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    potential = np.array(boundary_condition.boundary_values, copy=True)
    potential[1:-1, 1:-1, 1:-1] = interior_solution
    description = (
        "Finite-domain Poisson solve on the structured adaptive grid with free-space "
        "multipole Dirichlet boundary data. This is the first formal open-boundary "
        "approximation, not the final production solver."
    )
    return OpenBoundaryPoissonResult(
        rho=density,
        potential=potential,
        boundary_condition=boundary_condition,
        solver_method=solver_method,
        solver_iterations=iteration_count,
        residual_max=residual_max,
        description=description,
    )


def solve_open_boundary_poisson(
    grid_geometry: GridGeometryLike,
    rho: np.ndarray,
    multipole_order: int = 2,
    tolerance: float = 1.0e-8,
    max_iterations: int = 400,
    solver: str = "auto",
) -> OpenBoundaryPoissonResult:
    """Solve the first-stage finite-domain Poisson problem with open-boundary data."""

    if solver not in {"auto", "scipy_bicgstab", "jacobi"}:
        raise ValueError(
            "The current open-boundary Poisson slice supports solver=`auto`, `scipy_bicgstab`, "
            f"or `jacobi`; received `{solver}`."
        )

    if isinstance(grid_geometry, MonitorGridGeometry):
        return _solve_open_boundary_poisson_monitor(
            grid_geometry=grid_geometry,
            rho=rho,
            multipole_order=multipole_order,
            tolerance=tolerance,
            max_iterations=max_iterations,
            solver=solver,
        )
    return _solve_open_boundary_poisson_legacy(
        grid_geometry=grid_geometry,
        rho=rho,
        multipole_order=multipole_order,
        tolerance=tolerance,
        max_iterations=max_iterations,
        solver=solver,
    )
