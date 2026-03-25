"""Local GTH pseudopotential evaluation on legacy and monitor-driven grids."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf
from math import pi
from math import sqrt

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.grid import build_default_h2_monitor_grid
from isogrid.ops import integrate_field
from isogrid.ops import validate_orbital_field

from .gth_data import load_case_gth_pseudo_data
from .model import GTHPseudoData

_ERF_VECTOR = np.vectorize(erf, otypes=[np.float64])
GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry


@dataclass(frozen=True)
class AtomicLocalPotentialContribution:
    """One atom's local GTH pseudopotential on the structured grid."""

    atom_index: int
    element: str
    position: tuple[float, float, float]
    pseudo_data: GTHPseudoData
    local_potential: np.ndarray


@dataclass(frozen=True)
class LocalIonicPotentialEvaluation:
    """Local ionic GTH potential assembled over all atoms in one geometry.

    Future nonlocal projector action should attach alongside the per-atom
    contributions collected here.
    """

    pseudo_family: str
    grid_geometry: GridGeometryLike
    atom_contributions: tuple[AtomicLocalPotentialContribution, ...]
    total_local_potential: np.ndarray


@dataclass(frozen=True)
class LocalPotentialPatchParameters:
    """Parameters for the atom-centered local-GTH near-core patch correction."""

    patch_radius_scale: float = 1.0
    patch_grid_shape: tuple[int, int, int] = (25, 25, 25)
    correction_strength: float = 1.0
    interpolation_neighbors: int = 8

    def __post_init__(self) -> None:
        if self.patch_radius_scale <= 0.0:
            raise ValueError("patch_radius_scale must be positive.")
        if self.correction_strength < 0.0:
            raise ValueError("correction_strength must be non-negative.")
        if self.interpolation_neighbors <= 0:
            raise ValueError("interpolation_neighbors must be positive.")
        if len(self.patch_grid_shape) != 3:
            raise ValueError("patch_grid_shape must be a 3-tuple.")
        for axis_points in self.patch_grid_shape:
            if axis_points < 3 or axis_points % 2 == 0:
                raise ValueError(
                    "Each patch-grid direction must use an odd number of points >= 3."
                )


@dataclass(frozen=True)
class AtomicLocalPotentialPatchCorrection:
    """Near-core patch correction for one atom's local GTH contribution."""

    atom_index: int
    element: str
    position: tuple[float, float, float]
    patch_radius_bohr: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    main_grid_patch_energy: float
    patch_integrated_energy: float
    correction_energy: float
    patch_point_count: int


@dataclass(frozen=True)
class LocalIonicPotentialPatchEvaluation:
    """Patch-assisted local-GTH correction on the monitor grid.

    The pointwise main-grid potential is left unchanged. The correction acts on
    the near-core local-GTH energy functional by replacing the main-grid
    quadrature over each atom-centered patch with a finer auxiliary patch
    quadrature.
    """

    base_evaluation: LocalIonicPotentialEvaluation
    density_field: np.ndarray
    patch_parameters: LocalPotentialPatchParameters
    atomic_patch_corrections: tuple[AtomicLocalPotentialPatchCorrection, ...]
    uncorrected_local_energy: float
    corrected_local_energy: float
    total_patch_correction: float


def _screened_coulomb_term(
    radial_distance: np.ndarray,
    ionic_charge: int,
    rloc: float,
) -> np.ndarray:
    scaled_radius = radial_distance / (sqrt(2.0) * rloc)
    erf_values = _ERF_VECTOR(scaled_radius)
    with np.errstate(divide='ignore', invalid='ignore'):
        screened = -ionic_charge * erf_values / radial_distance

    zero_mask = radial_distance == 0.0
    if np.any(zero_mask):
        screened = np.asarray(screened, dtype=np.float64)
        screened[zero_mask] = -ionic_charge * sqrt(2.0 / pi) / rloc
    return np.asarray(screened, dtype=np.float64)


def _local_polynomial_term(
    radial_distance: np.ndarray,
    pseudo_data: GTHPseudoData,
) -> np.ndarray:
    radial_scaled = radial_distance / pseudo_data.local.rloc
    gaussian = np.exp(-0.5 * radial_scaled**2)
    polynomial = np.zeros_like(radial_distance, dtype=np.float64)
    for index, coefficient in enumerate(pseudo_data.local.coefficients):
        polynomial += coefficient * radial_scaled ** (2 * index)
    return gaussian * polynomial


def _evaluate_atomic_local_potential_from_radial_distance(
    radial_distance: np.ndarray,
    pseudo_data: GTHPseudoData,
) -> np.ndarray:
    coulomb_term = _screened_coulomb_term(
        radial_distance=radial_distance,
        ionic_charge=pseudo_data.ionic_charge,
        rloc=pseudo_data.local.rloc,
    )
    polynomial_term = _local_polynomial_term(
        radial_distance=radial_distance,
        pseudo_data=pseudo_data,
    )
    return coulomb_term + polynomial_term


def _evaluate_atomic_local_potential_on_grid(
    position: tuple[float, float, float],
    grid_geometry: GridGeometryLike,
    pseudo_data: GTHPseudoData,
) -> np.ndarray:
    """Evaluate one atom's local GTH potential on one supported grid."""

    grid_unit = grid_geometry.spec.unit.lower()
    if grid_unit != 'bohr':
        raise ValueError(
            "The current local GTH evaluator expects a Bohr-space structured grid; "
            f"received `{grid_geometry.spec.unit}`."
        )

    dx = grid_geometry.x_points - position[0]
    dy = grid_geometry.y_points - position[1]
    dz = grid_geometry.z_points - position[2]
    radial_distance = np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float64)

    return _evaluate_atomic_local_potential_from_radial_distance(
        radial_distance=radial_distance,
        pseudo_data=pseudo_data,
    )


def evaluate_atomic_local_potential_on_legacy_grid(
    position: tuple[float, float, float],
    grid_geometry: StructuredGridGeometry,
    pseudo_data: GTHPseudoData,
) -> np.ndarray:
    """Evaluate one atom's local GTH potential on the legacy structured grid."""

    return _evaluate_atomic_local_potential_on_grid(
        position=position,
        grid_geometry=grid_geometry,
        pseudo_data=pseudo_data,
    )


def evaluate_atomic_local_potential_on_monitor_grid(
    position: tuple[float, float, float],
    grid_geometry: MonitorGridGeometry,
    pseudo_data: GTHPseudoData,
) -> np.ndarray:
    """Evaluate one atom's local GTH potential on the new A-grid.

    This first A-grid local-GTH path uses only the main monitor grid. The patch
    layer is intentionally not used yet; it will be the first follow-up layer
    for near-core local-GTH corrections.
    """

    return _evaluate_atomic_local_potential_on_grid(
        position=position,
        grid_geometry=grid_geometry,
        pseudo_data=pseudo_data,
    )


def evaluate_atomic_local_potential(
    position: tuple[float, float, float],
    grid_geometry: GridGeometryLike,
    pseudo_data: GTHPseudoData,
) -> np.ndarray:
    """Evaluate one atom's local GTH potential on either supported grid family."""

    if isinstance(grid_geometry, MonitorGridGeometry):
        return evaluate_atomic_local_potential_on_monitor_grid(
            position=position,
            grid_geometry=grid_geometry,
            pseudo_data=pseudo_data,
        )
    return evaluate_atomic_local_potential_on_legacy_grid(
        position=position,
        grid_geometry=grid_geometry,
        pseudo_data=pseudo_data,
    )


def evaluate_legacy_local_ionic_potential(
    case: BenchmarkCase,
    grid_geometry: StructuredGridGeometry,
) -> LocalIonicPotentialEvaluation:
    """Evaluate the total local ionic GTH potential on the legacy grid."""

    if case.geometry.unit.lower() != grid_geometry.spec.unit.lower():
        raise ValueError(
            "Benchmark geometry and grid geometry must use the same unit system; "
            f"received `{case.geometry.unit}` and `{grid_geometry.spec.unit}`."
        )

    pseudo_data_by_element = load_case_gth_pseudo_data(case)
    total_local_potential = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    atom_contributions = []

    for atom_index, atom in enumerate(case.geometry.atoms):
        pseudo_data = pseudo_data_by_element[atom.element]
        local_potential = evaluate_atomic_local_potential_on_legacy_grid(
            position=atom.position,
            grid_geometry=grid_geometry,
            pseudo_data=pseudo_data,
        )
        total_local_potential += local_potential
        atom_contributions.append(
            AtomicLocalPotentialContribution(
                atom_index=atom_index,
                element=atom.element,
                position=atom.position,
                pseudo_data=pseudo_data,
                local_potential=local_potential,
            )
        )

    return LocalIonicPotentialEvaluation(
        pseudo_family=case.reference_model.pseudo,
        grid_geometry=grid_geometry,
        atom_contributions=tuple(atom_contributions),
        total_local_potential=total_local_potential,
    )


def evaluate_monitor_grid_local_ionic_potential(
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
) -> LocalIonicPotentialEvaluation:
    """Evaluate the total local ionic GTH potential on the new monitor-driven grid.

    This is the first A-grid local-GTH slice. It is intentionally main-grid only
    and does not yet use the auxiliary patch layer.
    """

    if case.geometry.unit.lower() != grid_geometry.spec.unit.lower():
        raise ValueError(
            "Benchmark geometry and grid geometry must use the same unit system; "
            f"received `{case.geometry.unit}` and `{grid_geometry.spec.unit}`."
        )

    pseudo_data_by_element = load_case_gth_pseudo_data(case)
    total_local_potential = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    atom_contributions = []

    for atom_index, atom in enumerate(case.geometry.atoms):
        pseudo_data = pseudo_data_by_element[atom.element]
        local_potential = evaluate_atomic_local_potential_on_monitor_grid(
            position=atom.position,
            grid_geometry=grid_geometry,
            pseudo_data=pseudo_data,
        )
        total_local_potential += local_potential
        atom_contributions.append(
            AtomicLocalPotentialContribution(
                atom_index=atom_index,
                element=atom.element,
                position=atom.position,
                pseudo_data=pseudo_data,
                local_potential=local_potential,
            )
        )

    return LocalIonicPotentialEvaluation(
        pseudo_family=case.reference_model.pseudo,
        grid_geometry=grid_geometry,
        atom_contributions=tuple(atom_contributions),
        total_local_potential=total_local_potential,
    )


def evaluate_local_ionic_potential(
    case: BenchmarkCase,
    grid_geometry: GridGeometryLike,
) -> LocalIonicPotentialEvaluation:
    """Evaluate the total local ionic GTH potential on either supported grid family."""

    if isinstance(grid_geometry, MonitorGridGeometry):
        return evaluate_monitor_grid_local_ionic_potential(case=case, grid_geometry=grid_geometry)
    return evaluate_legacy_local_ionic_potential(case=case, grid_geometry=grid_geometry)


def evaluate_local_ionic_energy(
    evaluation: LocalIonicPotentialEvaluation,
    density_field: np.ndarray,
) -> float:
    """Evaluate `E_loc,ion = ∫ rho V_loc` on the grid carried by one evaluation."""

    density = validate_orbital_field(
        density_field,
        grid_geometry=evaluation.grid_geometry,
        name="density_field",
    )
    return float(
        integrate_field(
            density * evaluation.total_local_potential,
            grid_geometry=evaluation.grid_geometry,
        )
    )


def _build_patch_grid(
    center: tuple[float, float, float],
    patch_radius: float,
    patch_grid_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    patch_x = np.linspace(
        center[0] - patch_radius,
        center[0] + patch_radius,
        patch_grid_shape[0],
        dtype=np.float64,
    )
    patch_y = np.linspace(
        center[1] - patch_radius,
        center[1] + patch_radius,
        patch_grid_shape[1],
        dtype=np.float64,
    )
    patch_z = np.linspace(
        center[2] - patch_radius,
        center[2] + patch_radius,
        patch_grid_shape[2],
        dtype=np.float64,
    )
    patch_cell_volume = (
        float(patch_x[1] - patch_x[0])
        * float(patch_y[1] - patch_y[0])
        * float(patch_z[1] - patch_z[0])
    )
    x_points, y_points, z_points = np.meshgrid(
        patch_x,
        patch_y,
        patch_z,
        indexing="ij",
    )
    return x_points, y_points, z_points, patch_cell_volume


def _interpolate_field_to_patch_points(
    field: np.ndarray,
    grid_geometry: MonitorGridGeometry,
    patch_points: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "SciPy is required when evaluating the A-grid local-GTH patch correction."
        ) from exc

    point_cloud = np.column_stack(
        (
            grid_geometry.x_points.reshape(-1),
            grid_geometry.y_points.reshape(-1),
            grid_geometry.z_points.reshape(-1),
        )
    )
    values = np.asarray(field, dtype=np.float64).reshape(-1)
    tree = cKDTree(point_cloud)
    distances, indices = tree.query(patch_points, k=neighbors)
    if neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    interpolated = np.empty(patch_points.shape[0], dtype=np.float64)
    zero_mask = distances[:, 0] <= 1.0e-12
    if np.any(zero_mask):
        interpolated[zero_mask] = values[indices[zero_mask, 0]]

    nonzero_mask = ~zero_mask
    if np.any(nonzero_mask):
        local_distances = distances[nonzero_mask]
        local_indices = indices[nonzero_mask]
        weights = 1.0 / np.maximum(local_distances, 1.0e-12) ** 2
        weights /= np.sum(weights, axis=1, keepdims=True)
        interpolated[nonzero_mask] = np.sum(values[local_indices] * weights, axis=1)
    return interpolated


def _evaluate_atomic_local_patch_correction(
    atom_contribution: AtomicLocalPotentialContribution,
    density_field: np.ndarray,
    grid_geometry: MonitorGridGeometry,
    patch_parameters: LocalPotentialPatchParameters,
) -> AtomicLocalPotentialPatchCorrection:
    patch_interface = grid_geometry.patch_interfaces[atom_contribution.atom_index]
    if "local_gth" not in patch_interface.implemented_purposes:
        raise ValueError(
            "This monitor-grid patch interface does not implement local-GTH correction."
        )
    patch_radius = patch_interface.patch_radius * patch_parameters.patch_radius_scale
    x_patch, y_patch, z_patch, patch_cell_volume = _build_patch_grid(
        center=atom_contribution.position,
        patch_radius=patch_radius,
        patch_grid_shape=patch_parameters.patch_grid_shape,
    )
    dx_patch = x_patch - atom_contribution.position[0]
    dy_patch = y_patch - atom_contribution.position[1]
    dz_patch = z_patch - atom_contribution.position[2]
    radial_patch = np.sqrt(
        dx_patch * dx_patch + dy_patch * dy_patch + dz_patch * dz_patch,
        dtype=np.float64,
    )
    patch_mask = radial_patch <= patch_radius
    patch_points = np.column_stack(
        (
            x_patch[patch_mask],
            y_patch[patch_mask],
            z_patch[patch_mask],
        )
    )
    density_patch = _interpolate_field_to_patch_points(
        field=density_field,
        grid_geometry=grid_geometry,
        patch_points=patch_points,
        neighbors=patch_parameters.interpolation_neighbors,
    )
    local_patch = _evaluate_atomic_local_potential_from_radial_distance(
        radial_distance=radial_patch[patch_mask],
        pseudo_data=atom_contribution.pseudo_data,
    )
    patch_integrated_energy = float(
        np.sum(density_patch * local_patch, dtype=np.float64) * patch_cell_volume
    )

    dx_main = grid_geometry.x_points - atom_contribution.position[0]
    dy_main = grid_geometry.y_points - atom_contribution.position[1]
    dz_main = grid_geometry.z_points - atom_contribution.position[2]
    radial_main = np.sqrt(
        dx_main * dx_main + dy_main * dy_main + dz_main * dz_main,
        dtype=np.float64,
    )
    main_patch_mask = radial_main <= patch_radius
    main_grid_patch_energy = float(
        np.sum(
            density_field[main_patch_mask]
            * atom_contribution.local_potential[main_patch_mask]
            * grid_geometry.cell_volumes[main_patch_mask],
            dtype=np.float64,
        )
    )
    correction_energy = patch_parameters.correction_strength * (
        patch_integrated_energy - main_grid_patch_energy
    )
    return AtomicLocalPotentialPatchCorrection(
        atom_index=atom_contribution.atom_index,
        element=atom_contribution.element,
        position=atom_contribution.position,
        patch_radius_bohr=float(patch_radius),
        patch_grid_shape=patch_parameters.patch_grid_shape,
        correction_strength=float(patch_parameters.correction_strength),
        main_grid_patch_energy=main_grid_patch_energy,
        patch_integrated_energy=patch_integrated_energy,
        correction_energy=float(correction_energy),
        patch_point_count=int(patch_points.shape[0]),
    )


def evaluate_monitor_grid_local_ionic_potential_with_patch(
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    density_field: np.ndarray,
    *,
    patch_parameters: LocalPotentialPatchParameters | None = None,
    base_evaluation: LocalIonicPotentialEvaluation | None = None,
) -> LocalIonicPotentialPatchEvaluation:
    """Apply a near-core patch correction to the A-grid local-GTH energy.

    The corrected local energy is

        E_loc^patch = E_loc^main + sum_A lambda [I_A^patch - I_A^main(patch)]

    with one atom-centered spherical patch per atom. `I_A^patch` is evaluated on
    a finer auxiliary Cartesian patch using density interpolated from the main
    grid, while `I_A^main(patch)` is the original main-grid quadrature over the
    same spherical support. The main-grid pointwise potential itself is not
    replaced.
    """

    density = validate_orbital_field(
        density_field,
        grid_geometry=grid_geometry,
        name="density_field",
    )
    if patch_parameters is None:
        patch_parameters = LocalPotentialPatchParameters()
    if base_evaluation is None:
        base_evaluation = evaluate_monitor_grid_local_ionic_potential(
            case=case,
            grid_geometry=grid_geometry,
        )

    uncorrected_local_energy = evaluate_local_ionic_energy(base_evaluation, density)
    atomic_patch_corrections = tuple(
        _evaluate_atomic_local_patch_correction(
            atom_contribution=atom_contribution,
            density_field=density,
            grid_geometry=grid_geometry,
            patch_parameters=patch_parameters,
        )
        for atom_contribution in base_evaluation.atom_contributions
    )
    total_patch_correction = float(
        sum(correction.correction_energy for correction in atomic_patch_corrections)
    )
    return LocalIonicPotentialPatchEvaluation(
        base_evaluation=base_evaluation,
        density_field=np.asarray(density, dtype=np.float64),
        patch_parameters=patch_parameters,
        atomic_patch_corrections=atomic_patch_corrections,
        uncorrected_local_energy=float(uncorrected_local_energy),
        corrected_local_energy=float(uncorrected_local_energy + total_patch_correction),
        total_patch_correction=total_patch_correction,
    )


def build_default_h2_local_ionic_potential(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: StructuredGridGeometry | None = None,
) -> LocalIonicPotentialEvaluation:
    """Evaluate the default H2 local ionic GTH potential on the default grid."""

    if grid_geometry is None:
        grid_geometry = build_default_h2_grid_geometry(case=case)
    return evaluate_local_ionic_potential(case=case, grid_geometry=grid_geometry)


def build_default_h2_monitor_grid_local_ionic_potential(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
) -> LocalIonicPotentialEvaluation:
    """Evaluate the default H2 local ionic GTH potential on the A-grid."""

    if grid_geometry is None:
        grid_geometry = build_default_h2_monitor_grid()
    return evaluate_monitor_grid_local_ionic_potential(case=case, grid_geometry=grid_geometry)


def build_h2_local_patch_development_monitor_grid_local_ionic_potential(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
) -> LocalIonicPotentialEvaluation:
    """Evaluate the H2 A-grid local-GTH potential on the patch-development baseline."""

    if grid_geometry is None:
        grid_geometry = build_h2_local_patch_development_monitor_grid()
    return evaluate_monitor_grid_local_ionic_potential(case=case, grid_geometry=grid_geometry)
