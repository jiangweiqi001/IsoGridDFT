"""Minimal smoke tests for the triplet reintegration audit."""

from importlib import import_module

from isogrid.audit.h2_jax_triplet_reintegration_smoke_audit import (
    H2JaxTripletReintegrationSmokeParameterSummary,
)
from isogrid.audit.h2_jax_triplet_reintegration_smoke_audit import (
    H2JaxTripletReintegrationSmokeRouteResult,
)
from isogrid.scf import SinglePointEnergyComponents


def test_triplet_reintegration_module_imports() -> None:
    module = import_module("isogrid.audit.h2_jax_triplet_reintegration_smoke_audit")

    assert hasattr(module, "run_h2_jax_triplet_reintegration_smoke_audit")
    assert hasattr(module, "print_h2_jax_triplet_reintegration_smoke_summary")


def test_construct_triplet_reintegration_route_result() -> None:
    route = H2JaxTripletReintegrationSmokeRouteResult(
        path_label="jax-native-eigensolver-triplet-mainline",
        spin_state_label="triplet",
        path_type="monitor_a_grid_plus_patch",
        solver_backend="jax",
        timed_out=False,
        smoke_timeout_seconds=None,
        converged=True,
        iteration_count=18,
        final_total_energy_ha=-1.22,
        final_lowest_eigenvalue_ha=-0.41,
        final_density_residual=4.5e-3,
        final_energy_change_ha=-2.0e-5,
        total_wall_time_seconds=120.0,
        average_iteration_wall_time_seconds=6.0,
        behavior_verdict="converged",
        earliest_issue_sign=None,
        parameter_summary=H2JaxTripletReintegrationSmokeParameterSummary(
            grid_shape=(67, 67, 81),
            box_half_extents_bohr=(8.0, 8.0, 10.0),
            weight_scale=4.0,
            radius_scale=0.70,
            patch_radius_scale=0.75,
            patch_grid_shape=(25, 25, 25),
            correction_strength=1.30,
            interpolation_neighbors=8,
            kinetic_version="trial_fix",
            hartree_backend="jax",
            use_jax_hartree_cached_operator=True,
            jax_hartree_cg_impl="jax_loop",
            jax_hartree_cg_preconditioner="none",
            use_jax_block_kernels=True,
            use_step_local_static_local_reuse=True,
            eigensolver_ncv=20,
            max_iterations=20,
            mixing=0.20,
        ),
        final_energy_components=SinglePointEnergyComponents(
            kinetic=0.1,
            local_ionic=-1.0,
            nonlocal_ionic=0.0,
            hartree=0.3,
            xc=-0.2,
            ion_ion_repulsion=0.7,
            total=-0.1,
        ),
    )

    assert route.solver_backend == "jax"
    assert route.timed_out is False
    assert route.converged is True
    assert route.final_density_residual == 4.5e-3
    assert route.behavior_verdict == "converged"
