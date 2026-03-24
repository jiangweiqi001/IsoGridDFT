"""Default benchmark and reference configurations for stage-1 auditing."""

from __future__ import annotations

from .model import AtomSpec
from .model import BenchmarkCase
from .model import MoleculeGeometry
from .model import ReferenceModelSettings
from .model import ScfSettings
from .model import SpinStateSpec

H2_BOND_LENGTH_BOHR = 1.4

H2_BASIS_CONVERGENCE_BASES = (
    "gth-szv",
    "gth-dzvp",
    "gth-tzvp",
    "gth-tzv2p",
    "gth-qzv3p",
)

H2_BASIS_CONVERGENCE_PROTOCOL_DESCRIPTION = (
    "Scan a compact-to-larger GTH Gaussian basis sequence on the PySCF side. "
    "Inspect the energies across the sequence; do not assume the largest basis "
    "is already converged without checking."
)

H2_BENCHMARK_CASE = BenchmarkCase(
    name="h2_r1p4_bohr",
    description=(
        "H2 at R = 1.4 Bohr. This is the first formal PySCF audit baseline "
        "for the singlet and triplet candidates before the real-space solver exists."
    ),
    geometry=MoleculeGeometry(
        name="H2",
        atoms=(
            AtomSpec(element="H", position=(0.0, 0.0, -0.5 * H2_BOND_LENGTH_BOHR)),
            AtomSpec(element="H", position=(0.0, 0.0, 0.5 * H2_BOND_LENGTH_BOHR)),
        ),
        unit="Bohr",
    ),
    charge=0,
    spin_states=(
        SpinStateSpec(label="singlet", spin=0),
        SpinStateSpec(label="triplet", spin=2),
    ),
    reference_model=ReferenceModelSettings(
        mean_field="uks",
        basis="gth-dzvp",
        pseudo="gth-pade",
        xc="lda,vwn",
    ),
    scf=ScfSettings(
        conv_tol=1.0e-10,
        max_cycle=100,
    ),
)

PYSCF_AUDIT_DEFAULT_CASE = H2_BENCHMARK_CASE
