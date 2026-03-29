from isogrid.audit.h2_jax_triplet_end_to_end_micro_profile_audit import (
    H2JaxTripletEndToEndMicroProfileResult,
)
from isogrid.audit.h2_jax_triplet_end_to_end_micro_profile_audit import (
    H2JaxTripletMicroProfileParameterSummary,
)
from isogrid.audit.h2_jax_triplet_end_to_end_micro_profile_audit import (
    H2JaxTripletMicroProfileStep,
)
from isogrid.scf import SinglePointEnergyComponents


def test_triplet_end_to_end_micro_profile_result_fields() -> None:
    result = H2JaxTripletEndToEndMicroProfileResult(
        path_label="jax-native-eigensolver-triplet-mainline-micro-profile",
        spin_state_label="triplet",
        path_type="monitor_a_grid_plus_patch",
        solver_backend="jax",
        converged=False,
        completed_full_20_steps=False,
        actual_iteration_count=2,
        final_total_energy_ha=-1.0,
        final_lowest_eigenvalue_ha=-0.4,
        final_density_residual=0.1,
        final_energy_change_ha=-1.0e-3,
        total_wall_time_seconds=10.0,
        average_iteration_wall_time_seconds=5.0,
        behavior_verdict="stable_not_converged",
        dominant_timing_bucket="eigensolver",
        dominant_timing_bucket_fraction_of_total=0.9,
        eigensolver_fraction_of_total=0.9,
        step_profiles=(
            H2JaxTripletMicroProfileStep(
                step_index=1,
                solver_backend="jax",
                total_step_wall_time_seconds=5.0,
                eigensolver_wall_time_seconds=4.0,
                static_local_prepare_wall_time_seconds=0.6,
                hartree_solve_wall_time_seconds=0.5,
                energy_eval_wall_time_seconds=0.5,
                density_residual=0.2,
                energy_change_ha=None,
            ),
        ),
        parameter_summary=H2JaxTripletMicroProfileParameterSummary(
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
            kinetic=0.0,
            local_ionic=0.0,
            nonlocal_ionic=0.0,
            hartree=0.0,
            xc=0.0,
            ion_ion_repulsion=0.0,
            total=-1.0,
        ),
    )

    assert result.solver_backend == "jax"
    assert result.converged is False
    assert result.final_density_residual == 0.1
    assert result.dominant_timing_bucket == "eigensolver"
    assert result.step_profiles[0].hartree_solve_wall_time_seconds == 0.5
