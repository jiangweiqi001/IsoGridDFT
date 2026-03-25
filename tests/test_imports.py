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


def test_import_scf_driver_entrypoint() -> None:
    from isogrid.scf import run_h2_minimal_scf

    assert callable(run_h2_minimal_scf)


def test_import_h2_vs_pyscf_audit_entrypoint() -> None:
    from isogrid.audit.h2_vs_pyscf_audit import run_h2_vs_pyscf_audit

    assert callable(run_h2_vs_pyscf_audit)


def test_import_h2_grid_convergence_audit_entrypoint() -> None:
    from isogrid.audit.h2_grid_convergence_audit import run_h2_grid_convergence_audit

    assert callable(run_h2_grid_convergence_audit)


def test_import_h2_regression_baseline() -> None:
    from isogrid.audit.baselines import H2_DEFAULT_PYSCF_REGRESSION_BASELINE

    assert H2_DEFAULT_PYSCF_REGRESSION_BASELINE.benchmark_name == "h2_r1p4_bohr"


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
