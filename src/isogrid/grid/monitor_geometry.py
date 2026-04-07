"""3D monitor evaluation and harmonic structured-grid generation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from isogrid.config import BenchmarkCase

from .monitor_model import AtomicMonitorContribution
from .monitor_model import MonitorCellLocalQuadrature
from .monitor_model import GlobalMonitorField
from .monitor_model import MonitorGridGeometry
from .monitor_model import MonitorGridQualityReport
from .monitor_model import MonitorGridSpec
from .monitor_model import MonitorPatchInterface
from .monitor_model import NearCoreElementParameters

_BASELINE_MONITOR = 1.0
_JACOBIAN_FLOOR = 1.0e-12


def _logical_axis(bounds: tuple[float, float], num_points: int) -> np.ndarray:
    lower, upper = bounds
    if upper <= lower:
        raise ValueError("Logical axis bounds must satisfy upper > lower.")
    return np.linspace(lower, upper, num_points, dtype=np.float64)


def build_reference_box_coordinates(
    spec: MonitorGridSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the logical and initial physical coordinates for the monitor grid."""

    logical_x = _logical_axis(spec.logical_bounds[0], spec.nx)
    logical_y = _logical_axis(spec.logical_bounds[1], spec.ny)
    logical_z = _logical_axis(spec.logical_bounds[2], spec.nz)
    physical_x = _logical_axis(spec.box_bounds[0], spec.nx)
    physical_y = _logical_axis(spec.box_bounds[1], spec.ny)
    physical_z = _logical_axis(spec.box_bounds[2], spec.nz)
    x_points, y_points, z_points = np.meshgrid(
        physical_x,
        physical_y,
        physical_z,
        indexing="ij",
    )
    return logical_x, logical_y, logical_z, x_points, y_points, z_points


def evaluate_global_monitor_field(
    case: BenchmarkCase,
    spec: MonitorGridSpec,
    x_points: np.ndarray,
    y_points: np.ndarray,
    z_points: np.ndarray,
) -> GlobalMonitorField:
    """Evaluate the full 3D atom-centered monitor field on current coordinates."""

    raw_total = np.full(spec.shape, _BASELINE_MONITOR, dtype=np.float64)
    values = np.full(spec.shape, _BASELINE_MONITOR, dtype=np.float64)
    atomic_contributions: list[AtomicMonitorContribution] = []

    for atom_index, atom in enumerate(case.geometry.atoms):
        parameters = spec.element_parameters[atom.element]
        dx = x_points - atom.position[0]
        dy = y_points - atom.position[1]
        dz = z_points - atom.position[2]
        radial_distance = np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float64)

        kinetic_component = parameters.kinetic_weight * np.exp(
            -np.power(radial_distance / parameters.near_core_radius, parameters.kinetic_exponent)
        )
        local_component = parameters.local_weight / (
            1.0
            + np.power(radial_distance / parameters.local_radius, parameters.local_exponent)
        )
        if parameters.projector_weight > 0.0:
            projector_component = parameters.projector_weight * np.exp(
                -np.power(
                    radial_distance / parameters.projector_radius,
                    parameters.projector_exponent,
                )
            )
        else:
            projector_component = np.zeros(spec.shape, dtype=np.float64)

        raw_atom = kinetic_component + local_component + projector_component
        capped_atom = parameters.monitor_cap * (1.0 - np.exp(-raw_atom / parameters.monitor_cap))
        raw_total += raw_atom
        values += capped_atom
        atomic_contributions.append(
            AtomicMonitorContribution(
                atom_index=atom_index,
                element=atom.element,
                position=atom.position,
                parameters=parameters,
                kinetic_component=np.asarray(kinetic_component, dtype=np.float64),
                local_component=np.asarray(local_component, dtype=np.float64),
                projector_component=np.asarray(projector_component, dtype=np.float64),
                raw_total=np.asarray(raw_atom, dtype=np.float64),
                capped_total=np.asarray(capped_atom, dtype=np.float64),
            )
        )

    return GlobalMonitorField(
        values=np.asarray(values, dtype=np.float64),
        raw_values=np.asarray(raw_total, dtype=np.float64),
        atomic_contributions=tuple(atomic_contributions),
        baseline_value=_BASELINE_MONITOR,
        minimum_value=float(np.min(values)),
        maximum_value=float(np.max(values)),
    )


def _smooth_monitor_field(values: np.ndarray, smoothing: float) -> np.ndarray:
    if smoothing <= 0.0:
        return np.asarray(values, dtype=np.float64)

    monitor = np.asarray(values, dtype=np.float64)
    smoothed = np.array(monitor, copy=True)
    interior_average = (
        monitor[1:-1, 1:-1, 1:-1]
        + monitor[:-2, 1:-1, 1:-1]
        + monitor[2:, 1:-1, 1:-1]
        + monitor[1:-1, :-2, 1:-1]
        + monitor[1:-1, 2:, 1:-1]
        + monitor[1:-1, 1:-1, :-2]
        + monitor[1:-1, 1:-1, 2:]
    ) / 7.0
    smoothed[1:-1, 1:-1, 1:-1] = (
        (1.0 - smoothing) * monitor[1:-1, 1:-1, 1:-1]
        + smoothing * interior_average
    )
    return np.maximum(smoothed, _BASELINE_MONITOR)


def _logical_spacing(logical_coordinates: np.ndarray) -> float:
    spacings = np.diff(np.asarray(logical_coordinates, dtype=np.float64))
    if not np.allclose(spacings, spacings[0]):
        raise ValueError("The current monitor-grid generator expects uniform logical coordinates.")
    return float(spacings[0])


def _boundary_mask(shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :, :] = True
    mask[-1, :, :] = True
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, 0] = True
    mask[:, :, -1] = True
    return mask


def _subcell_midpoint_samples(
    subcell_divisions: tuple[int, int, int],
) -> tuple[tuple[float, float, float], ...]:
    sx, sy, sz = subcell_divisions
    return tuple(
        (
            (ix + 0.5) / sx,
            (iy + 0.5) / sy,
            (iz + 0.5) / sz,
        )
        for ix in range(sx)
        for iy in range(sy)
        for iz in range(sz)
    )


def _trilinear_sample_nodal_field(
    field: np.ndarray,
    *,
    u: float,
    v: float,
    w: float,
) -> np.ndarray:
    nodal = np.asarray(field, dtype=np.float64)
    c000 = nodal[:-1, :-1, :-1]
    c100 = nodal[1:, :-1, :-1]
    c010 = nodal[:-1, 1:, :-1]
    c110 = nodal[1:, 1:, :-1]
    c001 = nodal[:-1, :-1, 1:]
    c101 = nodal[1:, :-1, 1:]
    c011 = nodal[:-1, 1:, 1:]
    c111 = nodal[1:, 1:, 1:]
    um = 1.0 - u
    vm = 1.0 - v
    wm = 1.0 - w
    return (
        um * vm * wm * c000
        + u * vm * wm * c100
        + um * v * wm * c010
        + u * v * wm * c110
        + um * vm * w * c001
        + u * vm * w * c101
        + um * v * w * c011
        + u * v * w * c111
    )


def build_monitor_cell_local_quadrature(
    grid_geometry: MonitorGridGeometry,
    *,
    subcell_divisions: tuple[int, int, int] = (2, 2, 2),
) -> MonitorCellLocalQuadrature:
    """Build the explicit logical-cell quadrature used by monitor-grid moments."""

    logical_x = np.asarray(grid_geometry.logical_x, dtype=np.float64)
    logical_y = np.asarray(grid_geometry.logical_y, dtype=np.float64)
    logical_z = np.asarray(grid_geometry.logical_z, dtype=np.float64)
    logical_cell_volume = (
        float(np.diff(logical_x)[0])
        * float(np.diff(logical_y)[0])
        * float(np.diff(logical_z)[0])
    )
    sx, sy, sz = subcell_divisions
    sample_count = float(sx * sy * sz)
    sample_points = _subcell_midpoint_samples(subcell_divisions)
    weight_samples = []
    for u, v, w in sample_points:
        jacobian_sample = _trilinear_sample_nodal_field(
            grid_geometry.jacobian,
            u=u,
            v=v,
            w=w,
        )
        weight_samples.append(jacobian_sample * (logical_cell_volume / sample_count))
    return MonitorCellLocalQuadrature(
        subcell_divisions=subcell_divisions,
        logical_cell_volume=logical_cell_volume,
        subcell_logical_volume=logical_cell_volume / sample_count,
        sample_points=sample_points,
        sample_weights=np.stack(weight_samples, axis=-1),
    )


def evaluate_monitor_cell_local_sample_weights(
    grid_geometry: MonitorGridGeometry,
    quadrature: MonitorCellLocalQuadrature,
) -> np.ndarray:
    """Evaluate all cell-local quadrature weights on the mapped monitor grid."""

    del grid_geometry
    return np.asarray(quadrature.sample_weights, dtype=np.float64)


def evaluate_monitor_cell_local_field_samples(
    field: np.ndarray,
    quadrature: MonitorCellLocalQuadrature,
) -> np.ndarray:
    """Evaluate one nodal field on the monitor cell-local quadrature samples."""

    samples = []
    for u, v, w in quadrature.sample_points:
        samples.append(
            _trilinear_sample_nodal_field(
                field,
                u=u,
                v=v,
                w=w,
            )
        )
    return np.stack(samples, axis=-1)


def _solve_weighted_harmonic_coordinates(
    coefficient: np.ndarray,
    logical_x: np.ndarray,
    logical_y: np.ndarray,
    logical_z: np.ndarray,
    boundary_coordinates: np.ndarray,
    initial_coordinates: np.ndarray,
    inner_iterations: int,
    tolerance: float,
    relaxation: float,
) -> np.ndarray:
    """Solve div(M grad X) = 0 for the three physical coordinates."""

    monitor = np.asarray(coefficient, dtype=np.float64)
    coordinates = np.asarray(initial_coordinates, dtype=np.float64).copy()
    boundary_mask = _boundary_mask(monitor.shape)
    coordinates[boundary_mask] = boundary_coordinates[boundary_mask]

    dx = _logical_spacing(logical_x)
    dy = _logical_spacing(logical_y)
    dz = _logical_spacing(logical_z)

    ax_minus = 0.5 * (monitor[1:-1, 1:-1, 1:-1] + monitor[:-2, 1:-1, 1:-1]) / (dx * dx)
    ax_plus = 0.5 * (monitor[1:-1, 1:-1, 1:-1] + monitor[2:, 1:-1, 1:-1]) / (dx * dx)
    ay_minus = 0.5 * (monitor[1:-1, 1:-1, 1:-1] + monitor[1:-1, :-2, 1:-1]) / (dy * dy)
    ay_plus = 0.5 * (monitor[1:-1, 1:-1, 1:-1] + monitor[1:-1, 2:, 1:-1]) / (dy * dy)
    az_minus = 0.5 * (monitor[1:-1, 1:-1, 1:-1] + monitor[1:-1, 1:-1, :-2]) / (dz * dz)
    az_plus = 0.5 * (monitor[1:-1, 1:-1, 1:-1] + monitor[1:-1, 1:-1, 2:]) / (dz * dz)
    diagonal = ax_minus + ax_plus + ay_minus + ay_plus + az_minus + az_plus

    for _ in range(inner_iterations):
        candidate = (
            ax_minus[..., None] * coordinates[:-2, 1:-1, 1:-1, :]
            + ax_plus[..., None] * coordinates[2:, 1:-1, 1:-1, :]
            + ay_minus[..., None] * coordinates[1:-1, :-2, 1:-1, :]
            + ay_plus[..., None] * coordinates[1:-1, 2:, 1:-1, :]
            + az_minus[..., None] * coordinates[1:-1, 1:-1, :-2, :]
            + az_plus[..., None] * coordinates[1:-1, 1:-1, 2:, :]
        ) / diagonal[..., None]

        updated = np.array(coordinates, copy=True)
        updated[1:-1, 1:-1, 1:-1, :] = (
            (1.0 - relaxation) * coordinates[1:-1, 1:-1, 1:-1, :]
            + relaxation * candidate
        )
        updated[boundary_mask] = boundary_coordinates[boundary_mask]
        max_change = float(
            np.max(np.abs(updated[1:-1, 1:-1, 1:-1, :] - coordinates[1:-1, 1:-1, 1:-1, :]))
        )
        coordinates = updated
        if max_change < tolerance:
            break

    return coordinates


def _covariant_basis(
    coordinates: np.ndarray,
    logical_x: np.ndarray,
    logical_y: np.ndarray,
    logical_z: np.ndarray,
) -> np.ndarray:
    basis = np.zeros(coordinates.shape[:-1] + (3, 3), dtype=np.float64)
    for component in range(3):
        basis[..., 0, component] = np.gradient(
            coordinates[..., component],
            logical_x,
            axis=0,
            edge_order=2,
        )
        basis[..., 1, component] = np.gradient(
            coordinates[..., component],
            logical_y,
            axis=1,
            edge_order=2,
        )
        basis[..., 2, component] = np.gradient(
            coordinates[..., component],
            logical_z,
            axis=2,
            edge_order=2,
        )
    return basis


def _basic_geometry_from_coordinates(
    logical_x: np.ndarray,
    logical_y: np.ndarray,
    logical_z: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    basis = _covariant_basis(coordinates, logical_x, logical_y, logical_z)
    metric_tensor = np.einsum("...ai,...bi->...ab", basis, basis)
    inverse_metric_tensor = np.linalg.inv(metric_tensor)
    jacobian = np.linalg.det(basis)
    dx = _logical_spacing(logical_x)
    dy = _logical_spacing(logical_y)
    dz = _logical_spacing(logical_z)
    cell_volumes = jacobian * dx * dy * dz
    spacing_x = np.linalg.norm(basis[..., 0, :], axis=-1) * dx
    spacing_y = np.linalg.norm(basis[..., 1, :], axis=-1) * dy
    spacing_z = np.linalg.norm(basis[..., 2, :], axis=-1) * dz
    return basis, jacobian, metric_tensor, inverse_metric_tensor, cell_volumes, np.stack(
        [spacing_x, spacing_y, spacing_z],
        axis=-1,
    )


def _backtracking_update(
    current_coordinates: np.ndarray,
    solved_coordinates: np.ndarray,
    logical_x: np.ndarray,
    logical_y: np.ndarray,
    logical_z: np.ndarray,
    relaxation: float,
) -> np.ndarray:
    alpha = relaxation
    while alpha >= 1.0e-3:
        trial_coordinates = current_coordinates + alpha * (solved_coordinates - current_coordinates)
        _, jacobian, _, _, _, _ = _basic_geometry_from_coordinates(
            logical_x,
            logical_y,
            logical_z,
            trial_coordinates,
        )
        if float(np.min(jacobian)) > _JACOBIAN_FLOOR:
            return trial_coordinates
        alpha *= 0.5
    return np.asarray(current_coordinates, dtype=np.float64)


def _build_patch_interfaces(
    case: BenchmarkCase,
    element_parameters: dict[str, NearCoreElementParameters],
) -> tuple[MonitorPatchInterface, ...]:
    patches = []
    for atom_index, atom in enumerate(case.geometry.atoms):
        parameters = element_parameters[atom.element]
        patches.append(
            MonitorPatchInterface(
                atom_index=atom_index,
                element=atom.element,
                center=atom.position,
                patch_radius=parameters.patch_radius,
                purposes=("local_gth", "nonlocal_projector", "near_core_integration"),
                implemented_purposes=("local_gth", "near_core_integration"),
                relation_to_main_grid=(
                    "Patch geometry is anchored at the atom center and will sample main-grid "
                    "fields for local GTH / nonlocal projector / near-core integration tasks. "
                    "No global unknowns live on the patch in this first monitor-grid core."
                ),
                implemented=True,
            )
        )
    return tuple(patches)


def _quality_report(
    spec: MonitorGridSpec,
    x_points: np.ndarray,
    y_points: np.ndarray,
    z_points: np.ndarray,
    jacobian: np.ndarray,
    cell_volumes: np.ndarray,
    spacing_measure: np.ndarray,
    monitor_field: GlobalMonitorField,
) -> MonitorGridQualityReport:
    nearest_atom_distance = np.full(spec.shape, np.inf, dtype=np.float64)
    near_mask = np.zeros(spec.shape, dtype=bool)
    max_patch_radius = 0.0

    for contribution in monitor_field.atomic_contributions:
        dx = x_points - contribution.position[0]
        dy = y_points - contribution.position[1]
        dz = z_points - contribution.position[2]
        radius = np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float64)
        nearest_atom_distance = np.minimum(nearest_atom_distance, radius)
        local_threshold = max(
            contribution.parameters.patch_radius,
            contribution.parameters.local_radius,
            contribution.parameters.near_core_radius,
        )
        near_mask |= radius <= local_threshold
        max_patch_radius = max(max_patch_radius, contribution.parameters.patch_radius)

    box_spans = np.array(
        [
            spec.box_bounds[0][1] - spec.box_bounds[0][0],
            spec.box_bounds[1][1] - spec.box_bounds[1][0],
            spec.box_bounds[2][1] - spec.box_bounds[2][0],
        ],
        dtype=np.float64,
    )
    far_threshold = max(3.0 * max_patch_radius, 0.35 * float(np.min(box_spans)))
    far_mask = nearest_atom_distance >= far_threshold
    if not np.any(far_mask):
        far_mask = nearest_atom_distance >= np.quantile(nearest_atom_distance, 0.75)

    mean_near = float(np.mean(spacing_measure[near_mask])) if np.any(near_mask) else float("nan")
    mean_far = float(np.mean(spacing_measure[far_mask])) if np.any(far_mask) else float("nan")
    ratio = mean_near / mean_far if np.isfinite(mean_near) and np.isfinite(mean_far) and mean_far > 0.0 else float("nan")
    return MonitorGridQualityReport(
        min_jacobian=float(np.min(jacobian)),
        max_jacobian=float(np.max(jacobian)),
        has_nonpositive_jacobian=bool(np.any(jacobian <= 0.0)),
        min_cell_volume=float(np.min(cell_volumes)),
        max_cell_volume=float(np.max(cell_volumes)),
        mean_near_atom_spacing=mean_near,
        mean_far_field_spacing=mean_far,
        near_to_far_spacing_ratio=float(ratio),
    )


def generate_monitor_grid_geometry(
    case: BenchmarkCase,
    spec: MonitorGridSpec,
) -> MonitorGridGeometry:
    """Generate a full 3D monitor-driven harmonic structured grid."""

    if case.geometry.unit.lower() != spec.unit.lower():
        raise ValueError(
            "Benchmark geometry and monitor-grid spec must use the same unit system; "
            f"received `{case.geometry.unit}` and `{spec.unit}`."
        )

    logical_x, logical_y, logical_z, x_ref, y_ref, z_ref = build_reference_box_coordinates(spec)
    boundary_coordinates = np.stack([x_ref, y_ref, z_ref], axis=-1)
    coordinates = np.array(boundary_coordinates, copy=True)

    for _ in range(spec.harmonic_outer_iterations):
        monitor_field = evaluate_global_monitor_field(
            case=case,
            spec=spec,
            x_points=coordinates[..., 0],
            y_points=coordinates[..., 1],
            z_points=coordinates[..., 2],
        )
        smoothed_monitor = _smooth_monitor_field(
            monitor_field.values,
            smoothing=spec.monitor_smoothing,
        )
        solved_coordinates = _solve_weighted_harmonic_coordinates(
            coefficient=smoothed_monitor,
            logical_x=logical_x,
            logical_y=logical_y,
            logical_z=logical_z,
            boundary_coordinates=boundary_coordinates,
            initial_coordinates=coordinates,
            inner_iterations=spec.harmonic_inner_iterations,
            tolerance=spec.harmonic_tolerance,
            relaxation=spec.inner_relaxation,
        )
        updated_coordinates = _backtracking_update(
            current_coordinates=coordinates,
            solved_coordinates=solved_coordinates,
            logical_x=logical_x,
            logical_y=logical_y,
            logical_z=logical_z,
            relaxation=spec.harmonic_relaxation,
        )
        max_displacement = float(np.max(np.abs(updated_coordinates - coordinates)))
        coordinates = updated_coordinates
        if max_displacement < spec.harmonic_tolerance:
            break

    final_monitor_field = evaluate_global_monitor_field(
        case=case,
        spec=spec,
        x_points=coordinates[..., 0],
        y_points=coordinates[..., 1],
        z_points=coordinates[..., 2],
    )
    basis, jacobian, metric_tensor, inverse_metric_tensor, cell_volumes, point_spacings = _basic_geometry_from_coordinates(
        logical_x,
        logical_y,
        logical_z,
        coordinates,
    )
    spacing_measure = np.mean(point_spacings, axis=-1)
    quality_report = _quality_report(
        spec=spec,
        x_points=coordinates[..., 0],
        y_points=coordinates[..., 1],
        z_points=coordinates[..., 2],
        jacobian=jacobian,
        cell_volumes=cell_volumes,
        spacing_measure=spacing_measure,
        monitor_field=final_monitor_field,
    )
    return MonitorGridGeometry(
        spec=spec,
        logical_x=logical_x,
        logical_y=logical_y,
        logical_z=logical_z,
        x_points=np.asarray(coordinates[..., 0], dtype=np.float64),
        y_points=np.asarray(coordinates[..., 1], dtype=np.float64),
        z_points=np.asarray(coordinates[..., 2], dtype=np.float64),
        covariant_basis=basis,
        jacobian=jacobian,
        metric_tensor=metric_tensor,
        inverse_metric_tensor=inverse_metric_tensor,
        cell_volumes=cell_volumes,
        spacing_x=point_spacings[..., 0],
        spacing_y=point_spacings[..., 1],
        spacing_z=point_spacings[..., 2],
        spacing_measure=spacing_measure,
        monitor_field=final_monitor_field,
        patch_interfaces=_build_patch_interfaces(case, spec.element_parameters),
        quality_report=quality_report,
    )
