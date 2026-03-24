"""Basis-sequence audit for the default H2 PySCF reference case."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BASIS_CONVERGENCE_BASES
from isogrid.config import H2_BASIS_CONVERGENCE_PROTOCOL_DESCRIPTION
from isogrid.config import PYSCF_AUDIT_DEFAULT_CASE
from isogrid.audit.pyscf_h2_reference import ReferenceResult
from isogrid.audit.pyscf_h2_reference import run_reference_case


@dataclass(frozen=True)
class BasisScanEntry:
    """Reference results for one basis in the scan sequence."""

    basis: str
    results: tuple[ReferenceResult, ...]


def run_basis_convergence_audit(
    case: BenchmarkCase = PYSCF_AUDIT_DEFAULT_CASE,
    basis_list: tuple[str, ...] = H2_BASIS_CONVERGENCE_BASES,
) -> tuple[BasisScanEntry, ...]:
    """Run the default PySCF basis scan for one benchmark case."""

    entries = []
    for basis in basis_list:
        updated_case = replace(
            case,
            reference_model=replace(case.reference_model, basis=basis),
        )
        entries.append(
            BasisScanEntry(
                basis=basis,
                results=run_reference_case(case=updated_case),
            )
        )
    return tuple(entries)


def print_basis_convergence_summary(entries: tuple[BasisScanEntry, ...]) -> None:
    """Print a compact basis-scan table."""

    print("IsoGridDFT PySCF H2 basis convergence audit")
    print(H2_BASIS_CONVERGENCE_PROTOCOL_DESCRIPTION)
    print(f"{'basis':<12} {'spin':<8} {'converged':<10} {'total energy [Ha]':>20}")
    for entry in entries:
        for result in entry.results:
            print(
                f"{entry.basis:<12} "
                f"{result.spin_state.label:<8} "
                f"{str(result.converged):<10} "
                f"{result.total_energy:>20.12f}"
            )


def main() -> int:
    """Run and print the default H2 basis-sequence audit."""

    entries = run_basis_convergence_audit()
    print_basis_convergence_summary(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
