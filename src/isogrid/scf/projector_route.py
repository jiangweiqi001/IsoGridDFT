"""Experimental projector-route helpers for local-only H2 singlet dry-runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.ks import weighted_orbital_norms
from isogrid.ks import weighted_orthonormalize_orbitals
from isogrid.ks import weighted_overlap_matrix
from isogrid.scf.active_subspace import ActiveSubspaceConfig
from isogrid.scf.active_subspace import ActiveSubspaceSelectionResult
from isogrid.scf.active_subspace import ActiveSubspaceState
from isogrid.scf.active_subspace import initialize_active_subspace
from isogrid.scf.active_subspace import update_active_subspace

GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry


@dataclass(frozen=True)
class ProjectorRouteConfig:
    """Configuration for the experimental singlet projector route."""

    enabled: bool
    active_subspace_size: int
    target_projector_rank: int
    projector_mixing: float
    experimental_route_name: str = "projector_mixing"

    @classmethod
    def local_only_h2_singlet_default(
        cls,
        *,
        projector_mixing: float = 0.35,
    ) -> "ProjectorRouteConfig":
        return cls(
            enabled=True,
            active_subspace_size=2,
            target_projector_rank=1,
            projector_mixing=float(projector_mixing),
            experimental_route_name="projector_mixing",
        )


@dataclass(frozen=True)
class ProjectorRouteState:
    """Persisted projector-route state across singlet SCF iterations."""

    config: ProjectorRouteConfig
    reference_subspace_orbitals: np.ndarray
    reference_projector_matrix: np.ndarray
    reference_occupied_orbitals: np.ndarray


@dataclass(frozen=True)
class ProjectorRouteSelectionResult:
    """One projector-route update result."""

    state: ProjectorRouteState
    active_subspace_selection: ActiveSubspaceSelectionResult
    spectral_projector_matrix: np.ndarray
    mixed_projector_matrix: np.ndarray
    occupied_orbitals: np.ndarray
    projector_response_frobenius_norm: float
    raw_occupied_overlap_abs: float | None
    best_in_subspace_occupied_overlap_abs: float | None
    internal_rotation_angle_deg: float | None
    projector_drift_frobenius_norm: float | None
    subspace_rotation_max_angle_deg: float | None
    verdict: str


def _active_subspace_config(config: ProjectorRouteConfig) -> ActiveSubspaceConfig:
    return ActiveSubspaceConfig(
        enabled=True,
        subspace_size=int(config.active_subspace_size),
        target_occupied_count=int(config.target_projector_rank),
        only_closed_shell_singlet=True,
    )


def _symmetric_projector_from_vector(coefficients: np.ndarray) -> np.ndarray:
    vector = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-16:
        basis = np.zeros_like(vector)
        basis[0] = 1.0
        vector = basis
        norm = 1.0
    vector = vector / norm
    return np.asarray(np.outer(vector, vector), dtype=np.float64)


def _dominant_projector_vector(projector_matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(projector_matrix, dtype=np.float64))
    vector = np.asarray(eigenvectors[:, -1], dtype=np.float64)
    if vector[0] < 0.0:
        vector = -vector
    return vector


def _occupied_orbitals_from_projector(
    *,
    aligned_subspace_orbitals: np.ndarray,
    projector_matrix: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    vector = _dominant_projector_vector(projector_matrix)
    occupied = np.tensordot(vector, np.asarray(aligned_subspace_orbitals, dtype=np.float64), axes=(0, 0))
    occupied_block = np.asarray([occupied], dtype=np.float64)
    return weighted_orthonormalize_orbitals(
        occupied_block,
        grid_geometry=grid_geometry,
        require_full_rank=True,
    )[:1]


def _spectral_projector_in_aligned_basis(
    *,
    aligned_subspace_orbitals: np.ndarray,
    raw_subspace_orbitals: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    overlaps = np.asarray(
        weighted_overlap_matrix(
            aligned_subspace_orbitals,
            grid_geometry=grid_geometry,
            other=np.asarray(raw_subspace_orbitals[:1], dtype=np.float64),
        )[:, 0],
        dtype=np.float64,
    )
    return _symmetric_projector_from_vector(overlaps)


def initialize_projector_route(
    *,
    raw_subspace_orbitals: np.ndarray,
    grid_geometry: GridGeometryLike,
    config: ProjectorRouteConfig,
) -> ProjectorRouteSelectionResult:
    """Initialize the experimental singlet projector route from the current lowest block."""

    active_selection = initialize_active_subspace(
        raw_subspace_orbitals=raw_subspace_orbitals,
        grid_geometry=grid_geometry,
        config=_active_subspace_config(config),
    )
    projector_matrix = np.zeros((config.active_subspace_size, config.active_subspace_size), dtype=np.float64)
    projector_matrix[0, 0] = 1.0
    occupied_orbitals = _occupied_orbitals_from_projector(
        aligned_subspace_orbitals=active_selection.aligned_subspace_orbitals,
        projector_matrix=projector_matrix,
        grid_geometry=grid_geometry,
    )
    state = ProjectorRouteState(
        config=config,
        reference_subspace_orbitals=np.asarray(active_selection.aligned_subspace_orbitals, dtype=np.float64),
        reference_projector_matrix=np.asarray(projector_matrix, dtype=np.float64),
        reference_occupied_orbitals=np.asarray(occupied_orbitals, dtype=np.float64),
    )
    return ProjectorRouteSelectionResult(
        state=state,
        active_subspace_selection=active_selection,
        spectral_projector_matrix=np.asarray(projector_matrix, dtype=np.float64),
        mixed_projector_matrix=np.asarray(projector_matrix, dtype=np.float64),
        occupied_orbitals=np.asarray(occupied_orbitals, dtype=np.float64),
        projector_response_frobenius_norm=0.0,
        raw_occupied_overlap_abs=active_selection.raw_occupied_overlap_abs,
        best_in_subspace_occupied_overlap_abs=active_selection.best_in_subspace_occupied_overlap_abs,
        internal_rotation_angle_deg=active_selection.internal_rotation_angle_deg,
        projector_drift_frobenius_norm=active_selection.projector_drift_frobenius_norm,
        subspace_rotation_max_angle_deg=active_selection.subspace_rotation_max_angle_deg,
        verdict="Initialized projector-route state from the aligned active subspace.",
    )


def update_projector_route(
    *,
    raw_subspace_orbitals: np.ndarray,
    state: ProjectorRouteState,
    grid_geometry: GridGeometryLike,
) -> ProjectorRouteSelectionResult:
    """Update the projector-route state against the current lowest active block."""

    active_state = ActiveSubspaceState(
        config=_active_subspace_config(state.config),
        reference_subspace_orbitals=np.asarray(state.reference_subspace_orbitals, dtype=np.float64),
        reference_occupied_orbitals=np.asarray(state.reference_occupied_orbitals, dtype=np.float64),
    )
    active_selection = update_active_subspace(
        raw_subspace_orbitals=raw_subspace_orbitals,
        state=active_state,
        grid_geometry=grid_geometry,
    )
    spectral_projector = _spectral_projector_in_aligned_basis(
        aligned_subspace_orbitals=active_selection.aligned_subspace_orbitals,
        raw_subspace_orbitals=active_selection.raw_subspace_orbitals,
        grid_geometry=grid_geometry,
    )
    mixed_trial = (
        (1.0 - state.config.projector_mixing) * np.asarray(state.reference_projector_matrix, dtype=np.float64)
        + state.config.projector_mixing * spectral_projector
    )
    mixed_projector = _symmetric_projector_from_vector(_dominant_projector_vector(mixed_trial))
    occupied_orbitals = _occupied_orbitals_from_projector(
        aligned_subspace_orbitals=active_selection.aligned_subspace_orbitals,
        projector_matrix=mixed_projector,
        grid_geometry=grid_geometry,
    )
    current_overlap = np.asarray(
        weighted_overlap_matrix(
            occupied_orbitals,
            grid_geometry=grid_geometry,
            other=state.reference_occupied_orbitals[:1],
        ),
        dtype=np.complex128,
    )
    best_overlap = float(np.abs(current_overlap[0, 0]))
    projector_response_frobenius_norm = float(
        np.linalg.norm(mixed_projector - np.asarray(state.reference_projector_matrix, dtype=np.float64), ord="fro")
    )
    if best_overlap >= 0.95:
        verdict = "The projector-route keeps a continuous occupied projector inside the active subspace."
    else:
        verdict = "The projector-route occupied projector departs materially from the reference occupied direction."
    next_state = ProjectorRouteState(
        config=state.config,
        reference_subspace_orbitals=np.asarray(active_selection.aligned_subspace_orbitals, dtype=np.float64),
        reference_projector_matrix=np.asarray(mixed_projector, dtype=np.float64),
        reference_occupied_orbitals=np.asarray(occupied_orbitals, dtype=np.float64),
    )
    return ProjectorRouteSelectionResult(
        state=next_state,
        active_subspace_selection=active_selection,
        spectral_projector_matrix=np.asarray(spectral_projector, dtype=np.float64),
        mixed_projector_matrix=np.asarray(mixed_projector, dtype=np.float64),
        occupied_orbitals=np.asarray(occupied_orbitals, dtype=np.float64),
        projector_response_frobenius_norm=projector_response_frobenius_norm,
        raw_occupied_overlap_abs=active_selection.raw_occupied_overlap_abs,
        best_in_subspace_occupied_overlap_abs=best_overlap,
        internal_rotation_angle_deg=active_selection.internal_rotation_angle_deg,
        projector_drift_frobenius_norm=active_selection.projector_drift_frobenius_norm,
        subspace_rotation_max_angle_deg=active_selection.subspace_rotation_max_angle_deg,
        verdict=verdict,
    )


def rebuild_density_from_projector_route(
    *,
    selection: ProjectorRouteSelectionResult,
    occupations,
    grid_geometry: GridGeometryLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the closed-shell singlet density from the explicit projector."""

    from isogrid.scf.driver import _renormalize_density

    subspace = np.asarray(selection.active_subspace_selection.aligned_subspace_orbitals, dtype=np.float64)
    projector = np.asarray(selection.mixed_projector_matrix, dtype=np.float64)
    rho_total = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    for i in range(subspace.shape[0]):
        for j in range(subspace.shape[0]):
            rho_total += 2.0 * projector[i, j] * subspace[i] * subspace[j]
    rho_total = np.maximum(np.asarray(rho_total, dtype=np.float64), 0.0)
    rho_up = _renormalize_density(0.5 * rho_total, occupations.n_alpha, grid_geometry=grid_geometry)
    rho_down = _renormalize_density(0.5 * rho_total, occupations.n_beta, grid_geometry=grid_geometry)
    return rho_up, rho_down
