"""Internal GTH data loading for the current stage-1 element set.

This layer intentionally supports only the GTH pseudopotentials needed by the
current project scope: H, C, N, and O with the `gth-pade` family.

The parameter values were transcribed from the CP2K-format GTH data exposed by
PySCF via `pyscf.pbc.gto.pseudo.load("gth-pade", symbol)`.
"""

from __future__ import annotations

from collections.abc import Iterable

from isogrid.config import BenchmarkCase

from .model import GTHLocalTerm
from .model import GTHNonlocalChannel
from .model import GTHPseudoData

SUPPORTED_GTH_FAMILY = "gth-pade"
SUPPORTED_GTH_ELEMENTS = ("H", "C", "N", "O")
GTH_DATA_SOURCE = (
    "Parameters transcribed from PySCF's CP2K-format GTH pseudopotential data "
    "loader for the gth-pade family."
)


_GTH_PADE_DATA = {
    "H": GTHPseudoData(
        family=SUPPORTED_GTH_FAMILY,
        element="H",
        valence_configuration=(1,),
        ionic_charge=1,
        local=GTHLocalTerm(
            rloc=0.2,
            coefficients=(-4.1802368, 0.72507482),
        ),
        nonlocal_channels=(),
        source=GTH_DATA_SOURCE,
        description="GTH-PADE pseudopotential for hydrogen.",
    ),
    "C": GTHPseudoData(
        family=SUPPORTED_GTH_FAMILY,
        element="C",
        valence_configuration=(2, 2),
        ionic_charge=4,
        local=GTHLocalTerm(
            rloc=0.34883045,
            coefficients=(-8.5137711, 1.22843203),
        ),
        nonlocal_channels=(
            GTHNonlocalChannel(
                angular_momentum=0,
                radius=0.30455321,
                projector_count=1,
                h_matrix=((9.52284179,),),
            ),
            GTHNonlocalChannel(
                angular_momentum=1,
                radius=0.2326773,
                projector_count=0,
                h_matrix=(),
            ),
        ),
        source=GTH_DATA_SOURCE,
        description="GTH-PADE pseudopotential for carbon.",
    ),
    "N": GTHPseudoData(
        family=SUPPORTED_GTH_FAMILY,
        element="N",
        valence_configuration=(2, 3),
        ionic_charge=5,
        local=GTHLocalTerm(
            rloc=0.28917923,
            coefficients=(-12.23481988, 1.76640728),
        ),
        nonlocal_channels=(
            GTHNonlocalChannel(
                angular_momentum=0,
                radius=0.25660487,
                projector_count=1,
                h_matrix=((13.55224272,),),
            ),
            GTHNonlocalChannel(
                angular_momentum=1,
                radius=0.27013369,
                projector_count=0,
                h_matrix=(),
            ),
        ),
        source=GTH_DATA_SOURCE,
        description="GTH-PADE pseudopotential for nitrogen.",
    ),
    "O": GTHPseudoData(
        family=SUPPORTED_GTH_FAMILY,
        element="O",
        valence_configuration=(2, 4),
        ionic_charge=6,
        local=GTHLocalTerm(
            rloc=0.24762086,
            coefficients=(-16.58031797, 2.39570092),
        ),
        nonlocal_channels=(
            GTHNonlocalChannel(
                angular_momentum=0,
                radius=0.22178614,
                projector_count=1,
                h_matrix=((18.26691718,),),
            ),
            GTHNonlocalChannel(
                angular_momentum=1,
                radius=0.2568289,
                projector_count=0,
                h_matrix=(),
            ),
        ),
        source=GTH_DATA_SOURCE,
        description="GTH-PADE pseudopotential for oxygen.",
    ),
}


def _normalize_family_name(family: str) -> str:
    return family.strip().lower()


def _normalize_element_symbol(symbol: str) -> str:
    cleaned = ''.join(character for character in symbol.strip() if character.isalpha())
    if not cleaned:
        raise ValueError("An element symbol is required to load GTH data.")
    return cleaned[0].upper() + cleaned[1:].lower()


def load_gth_pseudo_data(symbol: str, family: str = SUPPORTED_GTH_FAMILY) -> GTHPseudoData:
    """Load one supported internal GTH pseudopotential data object."""

    normalized_family = _normalize_family_name(family)
    if normalized_family != SUPPORTED_GTH_FAMILY:
        raise ValueError(
            "The current IsoGridDFT GTH data layer supports only the `gth-pade` "
            f"family; received `{family}`."
        )

    normalized_symbol = _normalize_element_symbol(symbol)
    if normalized_symbol not in _GTH_PADE_DATA:
        supported = ', '.join(SUPPORTED_GTH_ELEMENTS)
        raise ValueError(
            "The current IsoGridDFT GTH data layer supports only "
            f"{supported}; received `{symbol}`."
        )
    return _GTH_PADE_DATA[normalized_symbol]


def load_gth_pseudo_data_for_elements(
    elements: Iterable[str],
    family: str = SUPPORTED_GTH_FAMILY,
) -> dict[str, GTHPseudoData]:
    """Load supported GTH data for a set of element symbols."""

    unique_symbols = []
    seen = set()
    for element in elements:
        symbol = _normalize_element_symbol(element)
        if symbol not in seen:
            unique_symbols.append(symbol)
            seen.add(symbol)

    return {
        symbol: load_gth_pseudo_data(symbol=symbol, family=family)
        for symbol in unique_symbols
    }


def load_case_gth_pseudo_data(case: BenchmarkCase) -> dict[str, GTHPseudoData]:
    """Load the supported GTH data needed for one benchmark case."""

    return load_gth_pseudo_data_for_elements(
        elements=(atom.element for atom in case.geometry.atoms),
        family=case.reference_model.pseudo,
    )
