"""Placeholder checks for the H2 PySCF audit modules."""

from importlib import import_module
from pathlib import Path


def test_h2_reference_module_imports() -> None:
    module = import_module("isogrid.audit.pyscf_h2_reference")

    assert hasattr(module, "run_reference_case")


def test_h2_basis_convergence_module_imports() -> None:
    module = import_module("isogrid.audit.pyscf_h2_basis_convergence")

    assert hasattr(module, "run_basis_convergence_audit")


def test_h2_audit_scripts_exist() -> None:
    project_root = Path(__file__).resolve().parents[1]

    reference_script = project_root / "src" / "isogrid" / "audit" / "pyscf_h2_reference.py"
    convergence_script = project_root / "src" / "isogrid" / "audit" / "pyscf_h2_basis_convergence.py"

    assert reference_script.is_file()
    assert convergence_script.is_file()
