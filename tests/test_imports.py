"""Import smoke tests for the package skeleton and config defaults."""


def test_import_isogrid() -> None:
    import isogrid

    assert isogrid.__version__ == "0.1.0"


def test_import_default_h2_config() -> None:
    from isogrid.config import H2_BENCHMARK_CASE
    from isogrid.config import H2_BASIS_CONVERGENCE_BASES

    assert H2_BENCHMARK_CASE.name == "h2_r1p4_bohr"
    assert H2_BASIS_CONVERGENCE_BASES[0] == "gth-szv"
