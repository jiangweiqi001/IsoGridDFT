"""Minimal smoke tests for the H2 triplet JAX Hartree/energy audit."""

from importlib import import_module

from isogrid.audit.h2_jax_triplet_hartree_energy_audit import (
    H2TripletHartreeEnergyAuditResult,
)
from isogrid.audit.h2_jax_triplet_hartree_energy_audit import (
    H2TripletHartreeEnergyRouteResult,
)
from isogrid.audit.h2_jax_triplet_hartree_energy_audit import (
    H2TripletHartreeSingleSolveResult,
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
        path_label="jax-hartree-cgloop",
        spin_state_label="triplet",
        kinetic_version="trial_fix",
        hartree_backend="jax",
        cg_impl="jax_loop",
        use_jax_hartree_cached_operator=True,
        use_jax_block_kernels=True,
        use_step_local_static_local_reuse=True,
        matvec_timing_is_estimated=True,
        converged=True,
        iteration_count=18,
        final_total_energy_ha=-1.22,
        lowest_eigenvalue_ha=-0.41,
        final_density_residual=0.005,
        total_wall_time_seconds=600.0,
        average_iteration_wall_time_seconds=33.3,
        hartree_solve_call_count=37,
        average_hartree_solve_wall_time_seconds=5.0,
        first_hartree_solve_wall_time_seconds=12.0,
        repeated_hartree_solve_average_wall_time_seconds=4.8,
        repeated_hartree_solve_min_wall_time_seconds=4.2,
        repeated_hartree_solve_max_wall_time_seconds=5.1,
        average_hartree_cg_iterations=400.0,
        first_hartree_cg_iterations=400,
        repeated_hartree_cg_iteration_average=400.0,
        average_hartree_boundary_condition_wall_time_seconds=0.2,
        average_hartree_build_wall_time_seconds=0.3,
        average_hartree_rhs_assembly_wall_time_seconds=0.4,
        average_hartree_cg_wall_time_seconds=4.1,
        average_hartree_cg_other_overhead_wall_time_seconds=0.6,
        average_hartree_matvec_call_count=401.0,
        average_hartree_matvec_wall_time_seconds=3.5,
        average_hartree_matvec_wall_time_per_call_seconds=0.0087,
        average_hartree_cg_iteration_wall_time_seconds=0.01025,
        average_hartree_matvec_wall_time_per_iteration_seconds=0.00875,
        average_hartree_other_cg_overhead_wall_time_per_iteration_seconds=0.0015,
        first_hartree_matvec_call_count=401,
        repeated_hartree_matvec_call_count_average=401.0,
        first_hartree_matvec_wall_time_seconds=3.7,
        repeated_hartree_matvec_average_wall_time_seconds=3.45,
        first_hartree_matvec_wall_time_per_call_seconds=0.0092,
        repeated_hartree_matvec_wall_time_per_call_seconds=0.0086,
        hartree_cached_operator_usage_count=37,
        hartree_cached_operator_first_solve_count=1,
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
    single_solve = H2TripletHartreeSingleSolveResult(
        path_label="single-solve-cgloop",
        cg_impl="jax_loop",
        converged=True,
        residual_max=1.0e-8,
        iteration_count=400,
        total_solve_time_seconds=2.5,
        cg_wall_time_seconds=2.0,
        matvec_wall_time_seconds=1.2,
        cg_other_overhead_wall_time_seconds=0.8,
        matvec_call_count=401,
        average_iteration_wall_time_seconds=0.005,
        average_matvec_wall_time_seconds=0.003,
        average_matvec_wall_time_per_call_seconds=0.003,
        matvec_timing_is_estimated=True,
    )
    audit_result = H2TripletHartreeEnergyAuditResult(
        jax_hartree_baseline_route=route,
        jax_hartree_cgloop_route=route,
        single_solve_baseline=single_solve,
        single_solve_cgloop=single_solve,
        note="triplet profiling smoke",
    )

    assert route.hartree_backend == "jax"
    assert route.cg_impl == "jax_loop"
    assert route.use_jax_hartree_cached_operator is True
    assert route.use_step_local_static_local_reuse is True
    assert route.matvec_timing_is_estimated is True
    assert route.hartree_solve_call_count == 37
    assert route.first_hartree_solve_wall_time_seconds == 12.0
    assert route.average_hartree_matvec_call_count == 401.0
    assert route.average_hartree_cg_other_overhead_wall_time_seconds == 0.6
    assert route.first_hartree_matvec_call_count == 401
    assert route.repeated_hartree_matvec_wall_time_per_call_seconds == 0.0086
    assert route.timing_breakdown.hartree_solve_wall_time_seconds == 245.0
    assert single_solve.cg_impl == "jax_loop"
    assert single_solve.average_matvec_wall_time_per_call_seconds == 0.003
    assert audit_result.jax_hartree_cgloop_route.final_total_energy_ha == -1.22
