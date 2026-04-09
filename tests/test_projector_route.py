"""Unit tests for the local-only singlet projector route helpers."""

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


def test_projector_route_update_blends_reference_and_spectral_projectors() -> None:
    from isogrid.scf.projector_route import ProjectorRouteConfig
    from isogrid.scf.projector_route import initialize_projector_route
    from isogrid.scf.projector_route import update_projector_route

    grid_geometry = _small_grid_geometry()
    e0 = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    e1 = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    e0[2, 3, 4] = 1.0
    e1[5, 4, 6] = 1.0

    config = ProjectorRouteConfig.local_only_h2_singlet_default(projector_mixing=0.5)
    initial = initialize_projector_route(
        raw_subspace_orbitals=np.asarray([e0, e1], dtype=np.float64),
        grid_geometry=grid_geometry,
        config=config,
    )

    theta = np.deg2rad(80.0)
    rotated = np.asarray(
        [
            np.cos(theta) * e0 + np.sin(theta) * e1,
            -np.sin(theta) * e0 + np.cos(theta) * e1,
        ],
        dtype=np.float64,
    )
    result = update_projector_route(
        raw_subspace_orbitals=rotated,
        state=initial.state,
        grid_geometry=grid_geometry,
    )

    diagonal = np.diag(np.asarray(result.mixed_projector_matrix, dtype=np.float64))
    assert diagonal[0] > 0.0
    assert diagonal[1] > 0.0
    assert result.projector_response_frobenius_norm > 0.0
    assert result.best_in_subspace_occupied_overlap_abs is not None


def test_projector_route_density_rebuild_preserves_closed_shell_electron_count() -> None:
    from isogrid.scf import resolve_h2_spin_occupations
    from isogrid.scf.projector_route import ProjectorRouteConfig
    from isogrid.scf.projector_route import initialize_projector_route
    from isogrid.scf.projector_route import rebuild_density_from_projector_route

    grid_geometry = _small_grid_geometry()
    e0 = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    e1 = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    e0[2, 3, 4] = 1.0
    e1[5, 4, 6] = 1.0

    config = ProjectorRouteConfig.local_only_h2_singlet_default()
    selection = initialize_projector_route(
        raw_subspace_orbitals=np.asarray([e0, e1], dtype=np.float64),
        grid_geometry=grid_geometry,
        config=config,
    )
    occupations = resolve_h2_spin_occupations(spin_label="singlet", case=H2_BENCHMARK_CASE)
    rho_up, rho_down = rebuild_density_from_projector_route(
        selection=selection,
        occupations=occupations,
        grid_geometry=grid_geometry,
    )

    weights = np.asarray(grid_geometry.cell_volumes, dtype=np.float64)
    assert np.isclose(np.sum(rho_up * weights), occupations.n_alpha)
    assert np.isclose(np.sum(rho_down * weights), occupations.n_beta)
