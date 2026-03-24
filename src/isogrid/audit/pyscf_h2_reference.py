"""PySCF audit baseline for the default H2 benchmark case."""

from __future__ import annotations

from dataclasses import dataclass

from isogrid.config import BenchmarkCase
from isogrid.config import PYSCF_AUDIT_DEFAULT_CASE
from isogrid.config import ReferenceModelSettings
from isogrid.config import SpinStateSpec


@dataclass(frozen=True)
class ReferenceResult:
    """Single-spin-state PySCF audit result."""

    spin_state: SpinStateSpec
    basis: str
    converged: bool
    total_energy: float


def _load_pyscf():
    try:
        from pyscf import dft, gto
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PySCF is required to run the IsoGridDFT audit scripts. "
            "Install it with `pip install pyscf` or `pip install -e .[audit]`."
        ) from exc
    return gto, dft


def build_pyscf_molecule(case: BenchmarkCase, spin_state: SpinStateSpec):
    """Build the PySCF molecule object for one benchmark spin state."""

    gto, _ = _load_pyscf()
    atom_data = [
        (atom.element, atom.position)
        for atom in case.geometry.atoms
    ]
    return gto.M(
        atom=atom_data,
        unit=case.geometry.unit,
        charge=case.charge,
        spin=spin_state.spin,
        basis=case.reference_model.basis,
        pseudo=case.reference_model.pseudo,
        verbose=0,
    )


def build_mean_field(mol, reference_model: ReferenceModelSettings):
    """Build the PySCF mean-field object for the configured audit model."""

    _, dft = _load_pyscf()
    mean_field_name = reference_model.mean_field.lower()
    if mean_field_name != "uks":
        raise ValueError(
            "Only `uks` is implemented in the current PySCF audit layer. "
            "This baseline intentionally uses UKS for both singlet and triplet "
            "to keep one explicit unrestricted code path."
        )

    mf = dft.UKS(mol)
    mf.xc = reference_model.xc
    return mf


def run_reference_spin_state(
    case: BenchmarkCase,
    spin_state: SpinStateSpec,
) -> ReferenceResult:
    """Run the configured PySCF DFT reference for one candidate spin state."""

    mol = build_pyscf_molecule(case=case, spin_state=spin_state)
    mf = build_mean_field(mol=mol, reference_model=case.reference_model)
    mf.conv_tol = case.scf.conv_tol
    mf.max_cycle = case.scf.max_cycle

    total_energy = mf.kernel()
    return ReferenceResult(
        spin_state=spin_state,
        basis=case.reference_model.basis,
        converged=bool(mf.converged),
        total_energy=float(total_energy),
    )


def run_reference_case(case: BenchmarkCase = PYSCF_AUDIT_DEFAULT_CASE) -> tuple[ReferenceResult, ...]:
    """Evaluate all candidate spin states for one configured benchmark case."""

    return tuple(
        run_reference_spin_state(case=case, spin_state=spin_state)
        for spin_state in case.spin_states
    )


def select_lower_energy_state(results: tuple[ReferenceResult, ...]) -> ReferenceResult:
    """Return the lower-energy candidate spin state."""

    return min(results, key=lambda result: result.total_energy)


def print_reference_summary(
    case: BenchmarkCase,
    results: tuple[ReferenceResult, ...],
) -> None:
    """Print a compact summary for one benchmark reference calculation."""

    model = case.reference_model
    print("IsoGridDFT PySCF audit baseline")
    print(f"benchmark: {case.name}")
    print(f"description: {case.description}")
    print(f"molecule: {case.geometry.name}")
    print(f"geometry unit: {case.geometry.unit}")
    print("reference model:")
    print(f"  mean-field: {model.mean_field} (shared UKS path for singlet and triplet)")
    print(f"  pseudopotential: {model.pseudo}")
    print(f"  basis: {model.basis}")
    print(f"  xc: {model.xc}")
    print("scf controls:")
    print(f"  conv_tol: {case.scf.conv_tol:.1e}")
    print(f"  max_cycle: {case.scf.max_cycle}")

    for result in results:
        print(f"spin state: {result.spin_state.label}")
        print(f"  scf converged: {result.converged}")
        print(f"  total energy [Ha]: {result.total_energy:.12f}")

    lowest = select_lower_energy_state(results)
    print(f"lower-energy spin state: {lowest.spin_state.label}")


def main() -> int:
    """Run the default H2 benchmark reference calculation."""

    case = PYSCF_AUDIT_DEFAULT_CASE
    results = run_reference_case(case=case)
    print_reference_summary(case=case, results=results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
