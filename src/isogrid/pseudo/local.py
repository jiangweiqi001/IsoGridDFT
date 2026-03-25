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
from isogrid.grid import build_default_h2_monitor_grid

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
