"""Unit tests for active-subspace selection helpers."""

import numpy as np

from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def _small_grid_geometry():
    return build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(9, 9, 11),
        box_half_extents=(6.0, 6.0, 8.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )


def test_active_subspace_alignment_recovers_rotated_occupied_direction() -> None:
    from isogrid.scf.active_subspace import ActiveSubspaceConfig
    from isogrid.scf.active_subspace import initialize_active_subspace
    from isogrid.scf.active_subspace import update_active_subspace

    grid_geometry = _small_grid_geometry()
    e0 = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    e1 = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    e0[2, 3, 4] = 1.0
    e1[5, 4, 6] = 1.0

    config = ActiveSubspaceConfig(enabled=True, subspace_size=2, target_occupied_count=1)
    initial = initialize_active_subspace(
        raw_subspace_orbitals=np.asarray([e0, e1], dtype=np.float64),
        grid_geometry=grid_geometry,
        config=config,
    )

    theta = np.deg2rad(89.0)
    rotated = np.asarray(
        [
            np.cos(theta) * e0 + np.sin(theta) * e1,
            -np.sin(theta) * e0 + np.cos(theta) * e1,
        ],
        dtype=np.float64,
    )
    result = update_active_subspace(
        raw_subspace_orbitals=rotated,
        state=initial.state,
        grid_geometry=grid_geometry,
    )

    assert result.raw_occupied_overlap_abs < 0.1
    assert result.best_in_subspace_occupied_overlap_abs > 0.99
    assert result.internal_rotation_angle_deg > 80.0


def test_active_subspace_projector_drift_detects_missing_reference_direction() -> None:
    from isogrid.scf.active_subspace import ActiveSubspaceConfig
    from isogrid.scf.active_subspace import initialize_active_subspace
    from isogrid.scf.active_subspace import update_active_subspace

    grid_geometry = _small_grid_geometry()
    e0 = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    e1 = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    e2 = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    e3 = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    e0[2, 3, 4] = 1.0
    e1[5, 4, 6] = 1.0
    e2[1, 1, 1] = 1.0
    e3[7, 7, 8] = 1.0

    config = ActiveSubspaceConfig(enabled=True, subspace_size=2, target_occupied_count=1)
    initial = initialize_active_subspace(
        raw_subspace_orbitals=np.asarray([e0, e1], dtype=np.float64),
        grid_geometry=grid_geometry,
        config=config,
    )
    result = update_active_subspace(
        raw_subspace_orbitals=np.asarray([e2, e3], dtype=np.float64),
        state=initial.state,
        grid_geometry=grid_geometry,
    )

    assert result.best_in_subspace_occupied_overlap_abs < 0.1
    assert result.projector_drift_frobenius_norm > 1.0
    assert result.verdict.startswith("The reference occupied direction is no longer")
