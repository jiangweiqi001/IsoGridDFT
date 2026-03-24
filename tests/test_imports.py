"""Import smoke tests for the package skeleton and config defaults."""


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
