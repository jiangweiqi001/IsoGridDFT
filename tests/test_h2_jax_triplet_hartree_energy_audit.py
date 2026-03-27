"""Minimal smoke tests for the H2 triplet JAX Hartree/energy audit."""

from importlib import import_module

from isogrid.audit.h2_jax_triplet_hartree_energy_audit import (
    H2TripletHartreeEnergyAuditResult,
)
from isogrid.audit.h2_jax_triplet_hartree_energy_audit import (
    H2TripletHartreeEnergyRouteResult,
)
from isogrid.audit.h2_jax_triplet_hartree_energy_audit import (
    H2TripletHartreeEnergyTimingBreakdown,
)
from isogrid.scf import SinglePointEnergyComponents


def test_h2_jax_triplet_hartree_energy_module_imports() -> None:
    module = import_module("isogrid.audit.h2_jax_triplet_hartree_energy_audit")

    assert hasattr(module, "run_h2_jax_triplet_hartree_energy_audit")
    assert hasattr(module, "print_h2_jax_triplet_hartree_energy_summary")


def test_construct_h2_jax_triplet_hartree_energy_result() -> None:
    route = H2TripletHartreeEnergyRouteResult(
        path_label="jax-hartree",
        spin_state_label="triplet",
        kinetic_version="trial_fix",
        hartree_backend="jax",
        use_jax_block_kernels=True,
        use_step_local_static_local_reuse=True,
        converged=True,
        iteration_count=18,
        final_total_energy_ha=-1.22,
        lowest_eigenvalue_ha=-0.41,
        final_density_residual=0.005,
        total_wall_time_seconds=600.0,
        average_iteration_wall_time_seconds=33.3,
        hartree_solve_call_count=37,
        timing_breakdown=H2TripletHartreeEnergyTimingBreakdown(
            eigensolver_wall_time_seconds=100.0,
            static_local_prepare_wall_time_seconds=250.0,
            hartree_solve_wall_time_seconds=245.0,
            local_ionic_resolve_wall_time_seconds=4.0,
            xc_resolve_wall_time_seconds=1.0,
            energy_evaluation_wall_time_seconds=240.0,
            kinetic_energy_wall_time_seconds=1.0,
            local_ionic_energy_wall_time_seconds=0.1,
            hartree_energy_wall_time_seconds=0.05,
            xc_energy_wall_time_seconds=0.02,
            ion_ion_energy_wall_time_seconds=0.01,
            density_update_wall_time_seconds=0.1,
            bookkeeping_wall_time_seconds=10.0,
        ),
        parameter_summary="triplet hartree/energy smoke",
        final_energy_components=SinglePointEnergyComponents(
            kinetic=1.0,
            local_ionic=-3.0,
            nonlocal_ionic=0.0,
            hartree=1.5,
            xc=-0.7,
            ion_ion_repulsion=0.714285714286,
            total=-1.22,
        ),
    )
    audit_result = H2TripletHartreeEnergyAuditResult(
        baseline_route=route,
        jax_hartree_route=route,
        note="triplet profiling smoke",
    )

    assert route.hartree_backend == "jax"
    assert route.use_step_local_static_local_reuse is True
    assert route.hartree_solve_call_count == 37
    assert route.timing_breakdown.hartree_solve_wall_time_seconds == 245.0
    assert audit_result.jax_hartree_route.final_total_energy_ha == -1.22
