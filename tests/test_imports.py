"""Import smoke tests for the package skeleton and direct entry points."""


def test_import_isogrid() -> None:
    import isogrid

    assert isogrid.__version__ == "0.1.0"


def test_import_default_h2_config() -> None:
    from isogrid.config import H2_BASIS_CONVERGENCE_BASES
    from isogrid.config import H2_BENCHMARK_CASE

    assert H2_BENCHMARK_CASE.name == "h2_r1p4_bohr"
    assert H2_BASIS_CONVERGENCE_BASES[0] == "gth-szv"


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
