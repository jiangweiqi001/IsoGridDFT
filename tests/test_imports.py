"""Import smoke tests for the package skeleton and direct entry points."""


def test_import_isogrid() -> None:
    import isogrid

    assert isogrid.__version__ == "0.1.0"


def test_import_default_h2_config() -> None:
    from isogrid.config import H2_BASIS_CONVERGENCE_BASES
    from isogrid.config import H2_BENCHMARK_CASE
    from isogrid.config import MINIMAL_NONLOCAL_AUDIT_CASES

    assert H2_BENCHMARK_CASE.name == "h2_r1p4_bohr"
    assert H2_BASIS_CONVERGENCE_BASES[0] == "gth-szv"
    assert set(MINIMAL_NONLOCAL_AUDIT_CASES) == {"H2", "N2", "CO", "H2O"}


def test_import_grid_entrypoint() -> None:
    from isogrid.grid import build_default_h2_grid_spec

    spec = build_default_h2_grid_spec()
    assert spec.name == "h2_r1p4_structured_grid"


def test_import_pseudo_entrypoint() -> None:
    from isogrid.pseudo import load_gth_pseudo_data

    pseudo_data = load_gth_pseudo_data("H")
    assert pseudo_data.element == "H"


def test_import_ops_entrypoint() -> None:
    from isogrid.ops import apply_kinetic_operator

    assert callable(apply_kinetic_operator)


def test_import_poisson_entrypoint() -> None:
    from isogrid.poisson import solve_hartree_potential

    assert callable(solve_hartree_potential)


def test_import_local_hamiltonian_entrypoint() -> None:
    from isogrid.ks import apply_local_hamiltonian

    assert callable(apply_local_hamiltonian)


def test_import_nonlocal_entrypoint() -> None:
    from isogrid.pseudo import evaluate_nonlocal_ionic_action

    assert callable(evaluate_nonlocal_ionic_action)


def test_import_lsda_entrypoint() -> None:
    from isogrid.xc import evaluate_lsda_potential

    assert callable(evaluate_lsda_potential)


def test_import_static_ks_entrypoint() -> None:
    from isogrid.ks import apply_static_ks_hamiltonian

    assert callable(apply_static_ks_hamiltonian)


def test_import_fixed_potential_eigensolver_entrypoint() -> None:
    from isogrid.ks import solve_fixed_potential_eigenproblem

    assert callable(solve_fixed_potential_eigenproblem)


def test_import_fixed_potential_static_local_eigensolver_entrypoint() -> None:
    from isogrid.ks import solve_fixed_potential_static_local_eigenproblem

    assert callable(solve_fixed_potential_static_local_eigenproblem)


def test_import_scf_driver_entrypoint() -> None:
    from isogrid.scf import run_h2_minimal_scf
    from isogrid.scf import run_h2_monitor_grid_scf_dry_run

    assert callable(run_h2_minimal_scf)
    assert callable(run_h2_monitor_grid_scf_dry_run)


def test_import_h2_vs_pyscf_audit_entrypoint() -> None:
    from isogrid.audit.h2_vs_pyscf_audit import run_h2_vs_pyscf_audit

    assert callable(run_h2_vs_pyscf_audit)


def test_import_h2_grid_convergence_audit_entrypoint() -> None:
    from isogrid.audit.h2_grid_convergence_audit import run_h2_grid_convergence_audit

    assert callable(run_h2_grid_convergence_audit)


def test_import_h2_regression_baseline() -> None:
    from isogrid.audit.baselines import H2_DEFAULT_PYSCF_REGRESSION_BASELINE
    from isogrid.audit.baselines import H2_HARTREE_TAIL_RECHECK_BASELINE
    from isogrid.audit.baselines import H2_MONITOR_POISSON_REGRESSION_BASELINE
    from isogrid.audit.baselines import H2_STATIC_LOCAL_CHAIN_REGRESSION_BASELINE

    assert H2_DEFAULT_PYSCF_REGRESSION_BASELINE.benchmark_name == "h2_r1p4_bohr"
    assert H2_HARTREE_TAIL_RECHECK_BASELINE.baseline_point.point_label == "baseline"
    assert H2_MONITOR_POISSON_REGRESSION_BASELINE.monitor_shape == (67, 67, 81)
    assert H2_STATIC_LOCAL_CHAIN_REGRESSION_BASELINE.monitor_patch_improvement_vs_monitor_mha == 77.815


def test_import_monitor_grid_entrypoint() -> None:
    from isogrid.grid import build_default_h2_monitor_grid

    assert callable(build_default_h2_monitor_grid)


def test_import_monitor_grid_audit_entrypoint() -> None:
    from isogrid.audit.monitor_grid_audit import run_monitor_grid_audit

    assert callable(run_monitor_grid_audit)


def test_import_monitor_grid_ts_eloc_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_ts_eloc_audit import run_h2_monitor_grid_ts_eloc_audit

    assert callable(run_h2_monitor_grid_ts_eloc_audit)


def test_import_monitor_grid_fair_calibration_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_fair_calibration_audit import (
        run_h2_monitor_grid_fair_calibration_audit,
    )

    assert callable(run_h2_monitor_grid_fair_calibration_audit)


def test_import_monitor_grid_patch_local_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_patch_local_audit import (
        run_h2_monitor_grid_patch_local_audit,
    )

    assert callable(run_h2_monitor_grid_patch_local_audit)


def test_import_monitor_grid_patch_hartree_xc_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_patch_hartree_xc_audit import (
        run_h2_monitor_grid_patch_hartree_xc_audit,
    )

    assert callable(run_h2_monitor_grid_patch_hartree_xc_audit)


def test_import_h2_hartree_poisson_comparison_audit_entrypoint() -> None:
    from isogrid.audit.h2_hartree_poisson_comparison_audit import (
        run_h2_hartree_poisson_comparison_audit,
    )

    assert callable(run_h2_hartree_poisson_comparison_audit)


def test_import_h2_monitor_grid_poisson_operator_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_poisson_operator_audit import (
        run_h2_monitor_grid_poisson_operator_audit,
    )

    assert callable(run_h2_monitor_grid_poisson_operator_audit)


def test_import_h2_hartree_tail_recheck_audit_entrypoint() -> None:
    from isogrid.audit.h2_hartree_tail_recheck_audit import (
        run_h2_hartree_tail_recheck_audit,
    )

    assert callable(run_h2_hartree_tail_recheck_audit)


def test_import_h2_monitor_grid_fixed_potential_eigensolver_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_fixed_potential_eigensolver_audit import (
        run_h2_monitor_grid_fixed_potential_eigensolver_audit,
    )

    assert callable(run_h2_monitor_grid_fixed_potential_eigensolver_audit)


def test_import_h2_fixed_potential_eigensolver_baseline() -> None:
    from isogrid.audit.baselines import H2_FIXED_POTENTIAL_EIGENSOLVER_BASELINE
    from isogrid.audit.baselines import H2_FIXED_POTENTIAL_EIGENSOLVER_TRIAL_FIX_BASELINE

    assert H2_FIXED_POTENTIAL_EIGENSOLVER_BASELINE.monitor_shape == (67, 67, 81)
    assert H2_FIXED_POTENTIAL_EIGENSOLVER_BASELINE.monitor_patch_k1_route.target_orbitals == 1
    assert (
        H2_FIXED_POTENTIAL_EIGENSOLVER_TRIAL_FIX_BASELINE.monitor_patch_trial_fix_k1_route.kinetic_version
        == "trial_fix"
    )


def test_import_h2_monitor_grid_operator_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_operator_audit import (
        run_h2_monitor_grid_operator_audit,
    )

    assert callable(run_h2_monitor_grid_operator_audit)


def test_import_h2_fixed_potential_operator_baseline() -> None:
    from isogrid.audit.baselines import H2_FIXED_POTENTIAL_OPERATOR_AUDIT_BASELINE
    from isogrid.audit.baselines import H2_FIXED_POTENTIAL_OPERATOR_TRIAL_FIX_BASELINE

    assert H2_FIXED_POTENTIAL_OPERATOR_AUDIT_BASELINE.monitor_shape == (67, 67, 81)
    assert (
        H2_FIXED_POTENTIAL_OPERATOR_AUDIT_BASELINE.monitor_patch_route.patch_embedded_correction_mha
        == 77.815
    )
    assert (
        H2_FIXED_POTENTIAL_OPERATOR_TRIAL_FIX_BASELINE.monitor_patch_trial_fix_route.kinetic_version
        == "trial_fix"
    )


def test_import_h2_monitor_grid_kinetic_operator_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_kinetic_operator_audit import (
        run_h2_monitor_grid_kinetic_operator_audit,
    )

    assert callable(run_h2_monitor_grid_kinetic_operator_audit)


def test_import_h2_kinetic_operator_baseline() -> None:
    from isogrid.audit.baselines import H2_KINETIC_OPERATOR_AUDIT_BASELINE

    assert H2_KINETIC_OPERATOR_AUDIT_BASELINE.monitor_shape == (67, 67, 81)
    assert (
        H2_KINETIC_OPERATOR_AUDIT_BASELINE.monitor_patch_baseline_route.eigen_kinetic_ha
        == -3.374908449316
    )


def test_import_h2_monitor_grid_kinetic_form_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_kinetic_form_audit import (
        run_h2_monitor_grid_kinetic_form_audit,
    )

    assert callable(run_h2_monitor_grid_kinetic_form_audit)


def test_import_h2_kinetic_form_baseline() -> None:
    from isogrid.audit.baselines import H2_KINETIC_FORM_AUDIT_BASELINE

    assert H2_KINETIC_FORM_AUDIT_BASELINE.monitor_shape == (67, 67, 81)
    assert (
        H2_KINETIC_FORM_AUDIT_BASELINE.bad_eigen_baseline.production_kinetic_ha
        == -3.374908449316184
    )


def test_import_h2_monitor_grid_geometry_consistency_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_geometry_consistency_audit import (
        run_h2_monitor_grid_geometry_consistency_audit,
    )

    assert callable(run_h2_monitor_grid_geometry_consistency_audit)


def test_import_h2_geometry_consistency_baseline() -> None:
    from isogrid.audit.baselines import H2_GEOMETRY_CONSISTENCY_AUDIT_BASELINE

    assert H2_GEOMETRY_CONSISTENCY_AUDIT_BASELINE.monitor_shape == (67, 67, 81)
    assert (
        H2_GEOMETRY_CONSISTENCY_AUDIT_BASELINE.bad_eigen_baseline.delta_kinetic_mha
        == -7228.047450163502
    )


def test_import_h2_monitor_grid_kinetic_green_identity_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_kinetic_green_identity_audit import (
        run_h2_monitor_grid_kinetic_green_identity_audit,
    )

    assert callable(run_h2_monitor_grid_kinetic_green_identity_audit)


def test_import_h2_kinetic_green_identity_baseline() -> None:
    from isogrid.audit.baselines import H2_KINETIC_GREEN_IDENTITY_AUDIT_BASELINE
    from isogrid.audit.baselines import H2_KINETIC_GREEN_IDENTITY_TRIAL_FIX_BASELINE

    assert H2_KINETIC_GREEN_IDENTITY_AUDIT_BASELINE.monitor_shape == (67, 67, 81)
    assert (
        H2_KINETIC_GREEN_IDENTITY_AUDIT_BASELINE.bad_eigen_baseline.delta_kinetic_mha
        == -7228.047450163502
    )
    assert (
        H2_KINETIC_GREEN_IDENTITY_TRIAL_FIX_BASELINE.bad_eigen_baseline.delta_kinetic_mha
        == -458.64079452697035
    )


def test_import_h2_monitor_grid_orbital_shape_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_orbital_shape_audit import (
        run_h2_monitor_grid_orbital_shape_audit,
    )

    assert callable(run_h2_monitor_grid_orbital_shape_audit)


def test_import_h2_orbital_shape_baseline() -> None:
    from isogrid.audit.baselines import H2_ORBITAL_SHAPE_AUDIT_BASELINE

    assert H2_ORBITAL_SHAPE_AUDIT_BASELINE.monitor_shape == (67, 67, 81)
    assert H2_ORBITAL_SHAPE_AUDIT_BASELINE.monitor_trial_fix_k1_orbital.z_mirror_best_parity == "even"
    assert H2_ORBITAL_SHAPE_AUDIT_BASELINE.monitor_trial_fix_k2_gap_ha == 6.270018720619117e-05


def test_import_h2_monitor_grid_k2_subspace_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_k2_subspace_audit import (
        run_h2_monitor_grid_k2_subspace_audit,
    )

    assert callable(run_h2_monitor_grid_k2_subspace_audit)


def test_import_h2_k2_subspace_baseline() -> None:
    from isogrid.audit.baselines import H2_K2_SUBSPACE_AUDIT_BASELINE

    assert H2_K2_SUBSPACE_AUDIT_BASELINE.monitor_shape == (67, 67, 81)
    assert H2_K2_SUBSPACE_AUDIT_BASELINE.monitor_k2_gap_ha == 6.270018720619386e-05
    assert (
        H2_K2_SUBSPACE_AUDIT_BASELINE.monitor_bonding_rotation.rotated_second_orbital.centerline_sign_changes
        == 68
    )


def test_import_h2_monitor_grid_scf_dry_run_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_scf_dry_run_audit import (
        run_h2_monitor_grid_scf_dry_run_audit,
    )

    assert callable(run_h2_monitor_grid_scf_dry_run_audit)


def test_import_h2_scf_dry_run_baseline() -> None:
    from isogrid.audit.baselines import H2_SCF_DRY_RUN_BASELINE

    assert H2_SCF_DRY_RUN_BASELINE.monitor_shape == (67, 67, 81)
    assert H2_SCF_DRY_RUN_BASELINE.monitor_singlet_route.converged is False
    assert H2_SCF_DRY_RUN_BASELINE.monitor_triplet_route.converged is True


def test_import_h2_monitor_grid_singlet_stability_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_singlet_stability_audit import (
        run_h2_monitor_grid_singlet_stability_audit,
    )

    assert callable(run_h2_monitor_grid_singlet_stability_audit)


def test_import_h2_singlet_stability_baseline() -> None:
    from isogrid.audit.baselines import H2_SINGLET_STABILITY_BASELINE

    assert H2_SINGLET_STABILITY_BASELINE.monitor_shape == (67, 67, 81)
    assert H2_SINGLET_STABILITY_BASELINE.baseline_route.detected_two_cycle is False
    assert H2_SINGLET_STABILITY_BASELINE.smaller_mixing_route.converged is False
    assert H2_SINGLET_STABILITY_BASELINE.diis_prototype_route.scheme_label == "diis-prototype"


def test_import_h2_monitor_grid_diis_scf_audit_entrypoint() -> None:
    from isogrid.audit.h2_monitor_grid_diis_scf_audit import (
        run_h2_monitor_grid_diis_scf_audit,
    )

    assert callable(run_h2_monitor_grid_diis_scf_audit)


def test_import_h2_diis_scf_baseline() -> None:
    from isogrid.audit.baselines import H2_DIIS_SCF_BASELINE

    assert H2_DIIS_SCF_BASELINE.monitor_shape == (67, 67, 81)
    assert H2_DIIS_SCF_BASELINE.singlet.diis_prototype_route.diis_enabled is True
    assert H2_DIIS_SCF_BASELINE.triplet.diis_prototype_route.converged is True
