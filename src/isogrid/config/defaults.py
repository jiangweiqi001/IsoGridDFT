"""Default benchmark and reference configurations for stage-1 auditing."""

from __future__ import annotations

import math

from .model import AtomSpec
from .model import BenchmarkCase
from .model import MoleculeGeometry
from .model import ReferenceModelSettings
from .model import ScfSettings
from .model import SpinStateSpec

H2_BOND_LENGTH_BOHR = 1.4
N2_BOND_LENGTH_BOHR = 2.074
CO_BOND_LENGTH_BOHR = 2.132
H2O_OH_BOND_LENGTH_BOHR = 1.809
H2O_HOH_ANGLE_DEGREES = 104.5

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

_DEFAULT_STAGE1_REFERENCE_MODEL = ReferenceModelSettings(
    mean_field="uks",
    basis="gth-dzvp",
    pseudo="gth-pade",
    xc="lda,vwn",
)

_DEFAULT_STAGE1_SCF_SETTINGS = ScfSettings(
    conv_tol=1.0e-10,
    max_cycle=100,
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
    reference_model=_DEFAULT_STAGE1_REFERENCE_MODEL,
    scf=_DEFAULT_STAGE1_SCF_SETTINGS,
)

N2_AUDIT_CASE = BenchmarkCase(
    name="n2_neutral_audit",
    description=(
        "Minimal neutral N2 audit geometry for future nonlocal and Hartree-side "
        "expansion beyond H2."
    ),
    geometry=MoleculeGeometry(
        name="N2",
        atoms=(
            AtomSpec(element="N", position=(0.0, 0.0, -0.5 * N2_BOND_LENGTH_BOHR)),
            AtomSpec(element="N", position=(0.0, 0.0, 0.5 * N2_BOND_LENGTH_BOHR)),
        ),
        unit="Bohr",
    ),
    charge=0,
    spin_states=(SpinStateSpec(label="singlet", spin=0),),
    reference_model=_DEFAULT_STAGE1_REFERENCE_MODEL,
    scf=_DEFAULT_STAGE1_SCF_SETTINGS,
)

CO_AUDIT_CASE = BenchmarkCase(
    name="co_neutral_audit",
    description=(
        "Minimal neutral CO audit geometry for future nonlocal and Hartree-side "
        "expansion beyond H2."
    ),
    geometry=MoleculeGeometry(
        name="CO",
        atoms=(
            AtomSpec(element="C", position=(0.0, 0.0, -0.5 * CO_BOND_LENGTH_BOHR)),
            AtomSpec(element="O", position=(0.0, 0.0, 0.5 * CO_BOND_LENGTH_BOHR)),
        ),
        unit="Bohr",
    ),
    charge=0,
    spin_states=(SpinStateSpec(label="singlet", spin=0),),
    reference_model=_DEFAULT_STAGE1_REFERENCE_MODEL,
    scf=_DEFAULT_STAGE1_SCF_SETTINGS,
)

_H2O_HALF_ANGLE_RADIANS = 0.5 * math.radians(H2O_HOH_ANGLE_DEGREES)
_H2O_H_X = H2O_OH_BOND_LENGTH_BOHR * math.sin(_H2O_HALF_ANGLE_RADIANS)
_H2O_H_Z = H2O_OH_BOND_LENGTH_BOHR * math.cos(_H2O_HALF_ANGLE_RADIANS)

H2O_AUDIT_CASE = BenchmarkCase(
    name="h2o_neutral_audit",
    description=(
        "Minimal neutral H2O audit geometry for future nonlocal and Hartree-side "
        "expansion beyond H2."
    ),
    geometry=MoleculeGeometry(
        name="H2O",
        atoms=(
            AtomSpec(element="O", position=(0.0, 0.0, 0.0)),
            AtomSpec(element="H", position=(-_H2O_H_X, 0.0, _H2O_H_Z)),
            AtomSpec(element="H", position=(_H2O_H_X, 0.0, _H2O_H_Z)),
        ),
        unit="Bohr",
    ),
    charge=0,
    spin_states=(SpinStateSpec(label="singlet", spin=0),),
    reference_model=_DEFAULT_STAGE1_REFERENCE_MODEL,
    scf=_DEFAULT_STAGE1_SCF_SETTINGS,
)

MINIMAL_NONLOCAL_AUDIT_CASES = {
    "H2": H2_BENCHMARK_CASE,
    "N2": N2_AUDIT_CASE,
    "CO": CO_AUDIT_CASE,
    "H2O": H2O_AUDIT_CASE,
}

PYSCF_AUDIT_DEFAULT_CASE = H2_BENCHMARK_CASE
