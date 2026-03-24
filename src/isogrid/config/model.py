"""Lightweight configuration models for benchmark and audit settings."""

from __future__ import annotations

from dataclasses import dataclass

Coordinate3D = tuple[float, float, float]


@dataclass(frozen=True)
class AtomSpec:
    """Single atom with element symbol and Cartesian position."""

    element: str
    position: Coordinate3D


@dataclass(frozen=True)
class MoleculeGeometry:
    """Molecular geometry for an isolated benchmark case."""

    name: str
    atoms: tuple[AtomSpec, ...]
    unit: str


@dataclass(frozen=True)
class SpinStateSpec:
    """Candidate spin state using the PySCF spin convention."""

    label: str
    spin: int


@dataclass(frozen=True)
class ReferenceModelSettings:
    """Reference-model controls shared by the PySCF audit scripts."""

    mean_field: str
    basis: str
    pseudo: str
    xc: str


@dataclass(frozen=True)
class ScfSettings:
    """Minimal SCF controls for reference-side audit calculations."""

    conv_tol: float
    max_cycle: int


@dataclass(frozen=True)
class BenchmarkCase:
    """Named benchmark case used by audit and later solver development."""

    name: str
    description: str
    geometry: MoleculeGeometry
    charge: int
    spin_states: tuple[SpinStateSpec, ...]
    reference_model: ReferenceModelSettings
    scf: ScfSettings
