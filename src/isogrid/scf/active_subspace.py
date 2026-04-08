"""Active-subspace tracking helpers for local-only SCF routes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.ks import validate_orbital_block
from isogrid.ks import weighted_orbital_norms
from isogrid.ks import weighted_orthonormalize_orbitals
from isogrid.ks import weighted_overlap_matrix

GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry


@dataclass(frozen=True)
class ActiveSubspaceConfig:
    """Configuration for explicit active-subspace tracking."""

    enabled: bool
    subspace_size: int
    target_occupied_count: int
    only_closed_shell_singlet: bool = True

    @classmethod
    def local_only_h2_singlet_default(cls) -> "ActiveSubspaceConfig":
        return cls(
            enabled=True,
            subspace_size=2,
            target_occupied_count=1,
            only_closed_shell_singlet=True,
        )


@dataclass(frozen=True)
class ActiveSubspaceState:
    """Persisted active-subspace reference state."""

    config: ActiveSubspaceConfig
    reference_subspace_orbitals: np.ndarray
    reference_occupied_orbitals: np.ndarray


@dataclass(frozen=True)
class ActiveSubspaceSelectionResult:
    """One active-subspace selection/update result."""

    state: ActiveSubspaceState
    raw_subspace_orbitals: np.ndarray
    aligned_subspace_orbitals: np.ndarray
    occupied_orbitals: np.ndarray
    raw_occupied_overlap_abs: float | None
    best_in_subspace_occupied_overlap_abs: float | None
    internal_rotation_angle_deg: float | None
    projector_drift_frobenius_norm: float | None
    subspace_overlap_singular_values: tuple[float, ...]
    subspace_rotation_max_angle_deg: float | None
    verdict: str


def _validate_config(config: ActiveSubspaceConfig) -> None:
    if config.subspace_size <= 0:
        raise ValueError("subspace_size must be positive.")
    if config.target_occupied_count <= 0:
        raise ValueError("target_occupied_count must be positive.")
    if config.target_occupied_count > config.subspace_size:
        raise ValueError("target_occupied_count must not exceed subspace_size.")


def _validated_prefix(
    orbitals: np.ndarray,
    *,
    count: int,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    block = validate_orbital_block(orbitals, grid_geometry=grid_geometry, name="orbitals")
    if block.shape[0] < count:
        raise ValueError(
            f"At least {count} orbitals are required; received {block.shape[0]}."
        )
    return np.asarray(block[:count], dtype=np.float64)


def _project_reference_block_onto_subspace(
    *,
    reference_block: np.ndarray,
    current_subspace: np.ndarray,
    grid_geometry: GridGeometryLike,
    require_full_rank: bool,
) -> np.ndarray:
    overlaps = np.asarray(
        weighted_overlap_matrix(
            current_subspace,
            grid_geometry=grid_geometry,
            other=reference_block,
        ),
        dtype=np.float64,
    )
    projected = np.tensordot(overlaps.T, current_subspace, axes=(1, 0))
    projected = np.asarray(projected, dtype=np.float64)
    return weighted_orthonormalize_orbitals(
        projected,
        grid_geometry=grid_geometry,
        require_full_rank=require_full_rank,
    )


def _subspace_overlap_diagnostics(
    *,
    reference_subspace: np.ndarray,
    current_subspace: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> tuple[tuple[float, ...], float, float]:
    overlap = np.asarray(
        weighted_overlap_matrix(
            current_subspace,
            grid_geometry=grid_geometry,
            other=reference_subspace,
        ),
        dtype=np.float64,
    )
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    singular_values = np.clip(np.asarray(singular_values, dtype=np.float64), 0.0, 1.0)
    min_singular = float(np.min(singular_values))
    max_angle_deg = float(np.degrees(np.arccos(min_singular)))
    projector_drift = float(
        np.sqrt(max(0.0, 2.0 * float(np.sum(1.0 - singular_values * singular_values, dtype=np.float64))))
    )
    return tuple(float(value) for value in singular_values), max_angle_deg, projector_drift


def _occupied_overlap_diagnostics(
    *,
    raw_subspace: np.ndarray,
    aligned_occupied: np.ndarray,
    reference_occupied: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> tuple[float, float, float]:
    reference_count = reference_occupied.shape[0]
    raw_overlap = np.asarray(
        weighted_overlap_matrix(
            raw_subspace[:reference_count],
            grid_geometry=grid_geometry,
            other=reference_occupied,
        ),
        dtype=np.complex128,
    )
    aligned_overlap = np.asarray(
        weighted_overlap_matrix(
            aligned_occupied,
            grid_geometry=grid_geometry,
            other=reference_occupied,
        ),
        dtype=np.complex128,
    )
    raw_singular = np.linalg.svd(raw_overlap, compute_uv=False)
    aligned_singular = np.linalg.svd(aligned_overlap, compute_uv=False)
    raw_singular = np.clip(np.asarray(raw_singular, dtype=np.float64), 0.0, 1.0)
    aligned_singular = np.clip(np.asarray(aligned_singular, dtype=np.float64), 0.0, 1.0)
    raw_overlap_abs = float(np.min(raw_singular))
    best_overlap_abs = float(np.min(aligned_singular))
    alignment_overlap = np.asarray(
        weighted_overlap_matrix(
            aligned_occupied,
            grid_geometry=grid_geometry,
            other=raw_subspace[:reference_count],
        ),
        dtype=np.complex128,
    )
    alignment_singular = np.linalg.svd(alignment_overlap, compute_uv=False)
    alignment_singular = np.clip(np.asarray(alignment_singular, dtype=np.float64), 0.0, 1.0)
    internal_rotation = float(np.degrees(np.arccos(float(np.min(alignment_singular)))))
    return raw_overlap_abs, best_overlap_abs, internal_rotation


def initialize_active_subspace(
    *,
    raw_subspace_orbitals: np.ndarray,
    grid_geometry: GridGeometryLike,
    config: ActiveSubspaceConfig,
) -> ActiveSubspaceSelectionResult:
    """Initialize the active-subspace reference from the current raw lowest block."""

    _validate_config(config)
    raw_block = _validated_prefix(
        raw_subspace_orbitals,
        count=config.subspace_size,
        grid_geometry=grid_geometry,
    )
    occupied = np.asarray(
        raw_block[: config.target_occupied_count],
        dtype=np.float64,
    )
    state = ActiveSubspaceState(
        config=config,
        reference_subspace_orbitals=np.asarray(raw_block, dtype=np.float64),
        reference_occupied_orbitals=np.asarray(occupied, dtype=np.float64),
    )
    return ActiveSubspaceSelectionResult(
        state=state,
        raw_subspace_orbitals=np.asarray(raw_block, dtype=np.float64),
        aligned_subspace_orbitals=np.asarray(raw_block, dtype=np.float64),
        occupied_orbitals=np.asarray(occupied, dtype=np.float64),
        raw_occupied_overlap_abs=1.0,
        best_in_subspace_occupied_overlap_abs=1.0,
        internal_rotation_angle_deg=0.0,
        projector_drift_frobenius_norm=0.0,
        subspace_overlap_singular_values=tuple(1.0 for _ in range(config.subspace_size)),
        subspace_rotation_max_angle_deg=0.0,
        verdict="Initialized active-subspace tracking from the current raw subspace.",
    )


def update_active_subspace(
    *,
    raw_subspace_orbitals: np.ndarray,
    state: ActiveSubspaceState,
    grid_geometry: GridGeometryLike,
) -> ActiveSubspaceSelectionResult:
    """Update an active subspace against the stored reference subspace."""

    config = state.config
    _validate_config(config)
    raw_block = _validated_prefix(
        raw_subspace_orbitals,
        count=config.subspace_size,
        grid_geometry=grid_geometry,
    )
    singular_values, max_angle_deg, projector_drift = _subspace_overlap_diagnostics(
        reference_subspace=state.reference_subspace_orbitals,
        current_subspace=raw_block,
        grid_geometry=grid_geometry,
    )
    aligned_subspace = _project_reference_block_onto_subspace(
        reference_block=state.reference_subspace_orbitals,
        current_subspace=raw_block,
        grid_geometry=grid_geometry,
        require_full_rank=False,
    )
    if aligned_subspace.shape[0] < config.subspace_size:
        aligned_subspace = np.asarray(raw_block, dtype=np.float64)
    else:
        aligned_subspace = np.asarray(
            aligned_subspace[: config.subspace_size],
            dtype=np.float64,
        )
    projected_occupied = _project_reference_block_onto_subspace(
        reference_block=state.reference_occupied_orbitals,
        current_subspace=aligned_subspace,
        grid_geometry=grid_geometry,
        require_full_rank=False,
    )
    if projected_occupied.shape[0] < config.target_occupied_count:
        occupied = np.asarray(
            aligned_subspace[: config.target_occupied_count],
            dtype=np.float64,
        )
    else:
        occupied = np.asarray(
            projected_occupied[: config.target_occupied_count],
            dtype=np.float64,
        )
    raw_overlap_abs, best_overlap_abs, internal_rotation = _occupied_overlap_diagnostics(
        raw_subspace=raw_block,
        aligned_occupied=occupied,
        reference_occupied=state.reference_occupied_orbitals,
        grid_geometry=grid_geometry,
    )
    if best_overlap_abs >= 0.95 and raw_overlap_abs <= 0.2 and internal_rotation >= 45.0:
        verdict = (
            "The reference occupied direction stays inside the current active subspace, "
            "but the raw basis rotates strongly inside that subspace."
        )
    elif best_overlap_abs <= 0.2:
        verdict = (
            "The reference occupied direction is no longer well represented by the current "
            "active subspace; the subspace size may be too small."
        )
    else:
        verdict = "The active subspace remains locally continuous under this update."
    next_state = ActiveSubspaceState(
        config=config,
        reference_subspace_orbitals=np.asarray(aligned_subspace, dtype=np.float64),
        reference_occupied_orbitals=np.asarray(occupied, dtype=np.float64),
    )
    return ActiveSubspaceSelectionResult(
        state=next_state,
        raw_subspace_orbitals=np.asarray(raw_block, dtype=np.float64),
        aligned_subspace_orbitals=np.asarray(aligned_subspace, dtype=np.float64),
        occupied_orbitals=np.asarray(occupied, dtype=np.float64),
        raw_occupied_overlap_abs=raw_overlap_abs,
        best_in_subspace_occupied_overlap_abs=best_overlap_abs,
        internal_rotation_angle_deg=internal_rotation,
        projector_drift_frobenius_norm=projector_drift,
        subspace_overlap_singular_values=singular_values,
        subspace_rotation_max_angle_deg=max_angle_deg,
        verdict=verdict,
    )
