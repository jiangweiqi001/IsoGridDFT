"""First-stage real-space GTH nonlocal projector and action slice.

The current implementation is intentionally narrow and audit-friendly:

- only the internal `gth-pade` H/C/N/O data path is supported
- only the real-space projector structure needed by that data is targeted
- the current H/C/N/O `gth-pade` set has no active nonlocal projectors for H
  and one active s-channel projector for C, N, and O

For one atom, the separable GTH nonlocal action is applied as

    V_nl psi = sum_{l,m} sum_{i,j} |p_i^{lm}> h_ij <p_j^{lm} | psi>

with real-space projectors

    p_i^{lm}(r_vec) = p_i^l(r) Y_lm(r_hat)

and the normalized radial part

    p_i^l(r) = sqrt(2) * r^(l + 2 i)
               * exp[-r^2 / (2 r_l^2)]
               / [r_l^(l + (4 i + 3)/2) * sqrt(Gamma(l + (4 i + 3)/2))]

where `i` is zero-based in this implementation and corresponds to the
literature index `i + 1`.

This is a first formal nonlocal slice for the static KS backbone. It is not yet
an optimized production implementation and does not claim general GTH support.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gamma
from math import pi
from math import sqrt

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.ops import integrate_field
from isogrid.ops import validate_orbital_field

from .gth_data import load_case_gth_pseudo_data
from .model import GTHPseudoData

_Y00 = 1.0 / sqrt(4.0 * pi)
_Y1 = sqrt(3.0 / (4.0 * pi))


@dataclass(frozen=True)
class ProjectorFieldEvaluation:
    """One evaluated nonlocal projector field on the structured grid."""

    atom_index: int
    element: str
    position: tuple[float, float, float]
    angular_momentum: int
    magnetic_number: int
    projector_index: int
    radius: float
    projector_values: np.ndarray


@dataclass(frozen=True)
class AtomicNonlocalActionContribution:
    """One atom's resolved nonlocal projector data and action."""

    atom_index: int
    element: str
    position: tuple[float, float, float]
    pseudo_data: GTHPseudoData
    projector_fields: tuple[ProjectorFieldEvaluation, ...]
    projection_coefficients: tuple[complex | float, ...]
    resolved_coefficients: tuple[complex | float, ...]
    nonlocal_action: np.ndarray


@dataclass(frozen=True)
class NonlocalIonicActionEvaluation:
    """Full nonlocal ionic action assembled over all atoms for one orbital."""

    pseudo_family: str
    grid_geometry: StructuredGridGeometry
    atom_contributions: tuple[AtomicNonlocalActionContribution, ...]
    total_nonlocal_action: np.ndarray


def _scalarize(value: complex | float) -> complex | float:
    scalar = complex(value)
    if abs(scalar.imag) < 1.0e-14:
        return float(scalar.real)
    return scalar


def _real_spherical_harmonic(
    angular_momentum: int,
    magnetic_number: int,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    radial_distance: np.ndarray,
) -> np.ndarray:
    if angular_momentum == 0:
        if magnetic_number != 0:
            raise ValueError("The s-like projector supports only m = 0.")
        return np.full(dx.shape, _Y00, dtype=np.float64)

    if angular_momentum == 1:
        with np.errstate(divide='ignore', invalid='ignore'):
            if magnetic_number == -1:
                values = _Y1 * dy / radial_distance
            elif magnetic_number == 0:
                values = _Y1 * dz / radial_distance
            elif magnetic_number == 1:
                values = _Y1 * dx / radial_distance
            else:
                raise ValueError("The p-like real harmonics support only m = -1, 0, 1.")
        values = np.asarray(values, dtype=np.float64)
        values[radial_distance == 0.0] = 0.0
        return values

    raise NotImplementedError(
        "The current real-space GTH nonlocal slice supports only l = 0 and l = 1 "
        f"real harmonics; received l = {angular_momentum}."
    )


def _magnetic_numbers(angular_momentum: int) -> tuple[int, ...]:
    if angular_momentum == 0:
        return (0,)
    if angular_momentum == 1:
        return (-1, 0, 1)
    raise NotImplementedError(
        "The current real-space GTH nonlocal slice supports only l = 0 and l = 1; "
        f"received l = {angular_momentum}."
    )


def _radial_projector(
    radial_distance: np.ndarray,
    angular_momentum: int,
    projector_index: int,
    radius: float,
) -> np.ndarray:
    projector_order = projector_index + 1
    radial_power = angular_momentum + 2 * projector_index
    gamma_argument = angular_momentum + 0.5 * (4 * projector_order - 1)
    prefactor = sqrt(2.0) / (radius ** gamma_argument * sqrt(gamma(gamma_argument)))
    return prefactor * radial_distance**radial_power * np.exp(
        -0.5 * (radial_distance / radius) ** 2
    )


def evaluate_atomic_projector_field(
    position: tuple[float, float, float],
    grid_geometry: StructuredGridGeometry,
    angular_momentum: int,
    magnetic_number: int,
    projector_index: int,
    radius: float,
    atom_index: int = 0,
    element: str = "X",
) -> ProjectorFieldEvaluation:
    """Evaluate one atom-centered GTH nonlocal projector on the grid."""

    grid_unit = grid_geometry.spec.unit.lower()
    if grid_unit != 'bohr':
        raise ValueError(
            "The current GTH nonlocal projector evaluator expects a Bohr-space grid; "
            f"received `{grid_geometry.spec.unit}`."
        )

    dx = grid_geometry.x_points - position[0]
    dy = grid_geometry.y_points - position[1]
    dz = grid_geometry.z_points - position[2]
    radial_distance = np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float64)

    radial_part = _radial_projector(
        radial_distance=radial_distance,
        angular_momentum=angular_momentum,
        projector_index=projector_index,
        radius=radius,
    )
    angular_part = _real_spherical_harmonic(
        angular_momentum=angular_momentum,
        magnetic_number=magnetic_number,
        dx=dx,
        dy=dy,
        dz=dz,
        radial_distance=radial_distance,
    )
    projector_values = radial_part * angular_part
    return ProjectorFieldEvaluation(
        atom_index=atom_index,
        element=element,
        position=position,
        angular_momentum=angular_momentum,
        magnetic_number=magnetic_number,
        projector_index=projector_index,
        radius=radius,
        projector_values=np.asarray(projector_values, dtype=np.float64),
    )


def evaluate_atomic_nonlocal_action(
    position: tuple[float, float, float],
    grid_geometry: StructuredGridGeometry,
    psi: np.ndarray,
    pseudo_data: GTHPseudoData,
    atom_index: int = 0,
    element: str | None = None,
) -> AtomicNonlocalActionContribution:
    """Apply one atom's separable GTH nonlocal term to one orbital field."""

    field = validate_orbital_field(psi, grid_geometry=grid_geometry)
    atom_element = pseudo_data.element if element is None else element
    action_dtype = np.result_type(field.dtype, np.float64)
    atom_action = np.zeros(grid_geometry.spec.shape, dtype=action_dtype)
    projector_fields: list[ProjectorFieldEvaluation] = []
    projection_coefficients: list[complex | float] = []
    resolved_coefficients: list[complex | float] = []

    for channel in pseudo_data.nonlocal_channels:
        if channel.projector_count == 0:
            continue

        h_matrix = np.asarray(channel.h_matrix, dtype=np.float64)
        if h_matrix.shape != (channel.projector_count, channel.projector_count):
            raise ValueError(
                "The nonlocal GTH h-matrix shape must match the declared projector "
                f"count for {pseudo_data.element} l={channel.angular_momentum}; "
                f"received {h_matrix.shape} for projector_count={channel.projector_count}."
            )

        for magnetic_number in _magnetic_numbers(channel.angular_momentum):
            channel_fields = [
                evaluate_atomic_projector_field(
                    position=position,
                    grid_geometry=grid_geometry,
                    angular_momentum=channel.angular_momentum,
                    magnetic_number=magnetic_number,
                    projector_index=projector_index,
                    radius=channel.radius,
                    atom_index=atom_index,
                    element=atom_element,
                )
                for projector_index in range(channel.projector_count)
            ]
            coefficients = np.asarray(
                [
                    integrate_field(
                        np.conjugate(projector.projector_values) * field,
                        grid_geometry=grid_geometry,
                    )
                    for projector in channel_fields
                ],
                dtype=np.result_type(field.dtype, np.float64),
            )
            resolved = h_matrix @ coefficients
            for projector, coefficient, resolved_coefficient in zip(
                channel_fields,
                coefficients,
                resolved,
            ):
                atom_action += resolved_coefficient * projector.projector_values
                projector_fields.append(projector)
                projection_coefficients.append(_scalarize(coefficient))
                resolved_coefficients.append(_scalarize(resolved_coefficient))

    return AtomicNonlocalActionContribution(
        atom_index=atom_index,
        element=atom_element,
        position=position,
        pseudo_data=pseudo_data,
        projector_fields=tuple(projector_fields),
        projection_coefficients=tuple(projection_coefficients),
        resolved_coefficients=tuple(resolved_coefficients),
        nonlocal_action=atom_action,
    )


def evaluate_nonlocal_ionic_action(
    case: BenchmarkCase,
    grid_geometry: StructuredGridGeometry,
    psi: np.ndarray,
) -> NonlocalIonicActionEvaluation:
    """Apply the full ionic GTH nonlocal term to one orbital field."""

    if case.geometry.unit.lower() != grid_geometry.spec.unit.lower():
        raise ValueError(
            "Benchmark geometry and grid geometry must use the same unit system; "
            f"received `{case.geometry.unit}` and `{grid_geometry.spec.unit}`."
        )

    field = validate_orbital_field(psi, grid_geometry=grid_geometry)
    pseudo_data_by_element = load_case_gth_pseudo_data(case)
    total_action = np.zeros(grid_geometry.spec.shape, dtype=np.result_type(field.dtype, np.float64))
    atom_contributions = []

    for atom_index, atom in enumerate(case.geometry.atoms):
        contribution = evaluate_atomic_nonlocal_action(
            position=atom.position,
            grid_geometry=grid_geometry,
            psi=field,
            pseudo_data=pseudo_data_by_element[atom.element],
            atom_index=atom_index,
            element=atom.element,
        )
        total_action += contribution.nonlocal_action
        atom_contributions.append(contribution)

    return NonlocalIonicActionEvaluation(
        pseudo_family=case.reference_model.pseudo,
        grid_geometry=grid_geometry,
        atom_contributions=tuple(atom_contributions),
        total_nonlocal_action=total_action,
    )


def build_default_h2_nonlocal_ionic_action(
    psi: np.ndarray,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: StructuredGridGeometry | None = None,
) -> NonlocalIonicActionEvaluation:
    """Apply the default H2 nonlocal ionic term on the default structured grid."""

    if grid_geometry is None:
        grid_geometry = build_default_h2_grid_geometry(case=case)
    return evaluate_nonlocal_ionic_action(
        case=case,
        grid_geometry=grid_geometry,
        psi=psi,
    )
