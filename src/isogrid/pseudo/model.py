"""Minimal data models for the stage-1 GTH pseudopotential route."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GTHLocalTerm:
    """Local GTH pseudopotential parameters for one element."""

    rloc: float
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.rloc <= 0.0:
            raise ValueError("The local GTH radius rloc must be positive.")
        if not self.coefficients:
            raise ValueError("At least one local GTH coefficient is required.")


@dataclass(frozen=True)
class GTHNonlocalChannel:
    """Nonlocal projector metadata for one angular-momentum channel."""

    angular_momentum: int
    radius: float
    projector_count: int
    h_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class GTHPseudoData:
    """Minimal internal representation of one GTH pseudopotential."""

    family: str
    element: str
    valence_configuration: tuple[int, ...]
    ionic_charge: int
    local: GTHLocalTerm
    nonlocal_channels: tuple[GTHNonlocalChannel, ...]
    source: str
    description: str

    @property
    def valence_electrons(self) -> int:
        """Return the nominal valence electron count for this pseudopotential."""

        return self.ionic_charge
