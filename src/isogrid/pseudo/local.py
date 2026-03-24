"""Local GTH pseudopotential evaluation on the structured adaptive grid."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf
from math import pi
from math import sqrt

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry

from .gth_data import load_case_gth_pseudo_data
from .model import GTHPseudoData

_ERF_VECTOR = np.vectorize(erf, otypes=[np.float64])


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
    grid_geometry: StructuredGridGeometry
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


def evaluate_atomic_local_potential(
    position: tuple[float, float, float],
    grid_geometry: StructuredGridGeometry,
    pseudo_data: GTHPseudoData,
) -> np.ndarray:
    """Evaluate one atom's local GTH potential on the structured grid."""

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


def evaluate_local_ionic_potential(
    case: BenchmarkCase,
    grid_geometry: StructuredGridGeometry,
) -> LocalIonicPotentialEvaluation:
    """Evaluate the total local ionic GTH potential for one benchmark case."""

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
        local_potential = evaluate_atomic_local_potential(
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


def build_default_h2_local_ionic_potential(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: StructuredGridGeometry | None = None,
) -> LocalIonicPotentialEvaluation:
    """Evaluate the default H2 local ionic GTH potential on the default grid."""

    if grid_geometry is None:
        grid_geometry = build_default_h2_grid_geometry(case=case)
    return evaluate_local_ionic_potential(case=case, grid_geometry=grid_geometry)
