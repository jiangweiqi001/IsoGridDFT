"""Minimal GTH pseudopotential data and local/nonlocal potential helpers."""

from importlib import import_module

from .gth_data import GTH_DATA_SOURCE
from .gth_data import SUPPORTED_GTH_ELEMENTS
from .gth_data import SUPPORTED_GTH_FAMILY
from .gth_data import load_case_gth_pseudo_data
from .gth_data import load_gth_pseudo_data
from .gth_data import load_gth_pseudo_data_for_elements
from .local import AtomicLocalPotentialContribution
from .local import AtomicLocalPotentialPatchCorrection
from .local import FrozenPatchLocalPotentialEmbedding
from .local import LocalIonicPotentialEvaluation
from .local import LocalIonicPotentialPatchEvaluation
from .local import LocalPotentialPatchParameters
from .local import build_default_h2_local_ionic_potential
from .local import build_default_h2_monitor_grid_local_ionic_potential
from .local import build_h2_local_patch_development_monitor_grid_local_ionic_potential
from .local import evaluate_atomic_local_potential
from .local import evaluate_atomic_local_potential_on_legacy_grid
from .local import evaluate_atomic_local_potential_on_monitor_grid
from .local import evaluate_legacy_local_ionic_potential
from .local import evaluate_local_ionic_energy
from .local import evaluate_local_ionic_potential
from .local import evaluate_monitor_grid_local_ionic_potential
from .local import evaluate_monitor_grid_local_ionic_potential_with_frozen_patch_field
from .local import evaluate_monitor_grid_local_ionic_potential_with_patch
from .model import GTHLocalTerm
from .model import GTHNonlocalChannel
from .model import GTHPseudoData

_nonlocal_module = import_module('.nonlocal', __name__)
AtomicNonlocalActionContribution = _nonlocal_module.AtomicNonlocalActionContribution
NonlocalIonicActionEvaluation = _nonlocal_module.NonlocalIonicActionEvaluation
ProjectorFieldEvaluation = _nonlocal_module.ProjectorFieldEvaluation
build_default_h2_nonlocal_ionic_action = _nonlocal_module.build_default_h2_nonlocal_ionic_action
evaluate_atomic_nonlocal_action = _nonlocal_module.evaluate_atomic_nonlocal_action
evaluate_atomic_projector_field = _nonlocal_module.evaluate_atomic_projector_field
evaluate_nonlocal_ionic_action = _nonlocal_module.evaluate_nonlocal_ionic_action

__all__ = [
    "AtomicLocalPotentialContribution",
    "AtomicLocalPotentialPatchCorrection",
    "AtomicNonlocalActionContribution",
    "FrozenPatchLocalPotentialEmbedding",
    "GTH_DATA_SOURCE",
    "GTHLocalTerm",
    "GTHNonlocalChannel",
    "GTHPseudoData",
    "LocalIonicPotentialEvaluation",
    "LocalIonicPotentialPatchEvaluation",
    "LocalPotentialPatchParameters",
    "NonlocalIonicActionEvaluation",
    "ProjectorFieldEvaluation",
    "SUPPORTED_GTH_ELEMENTS",
    "SUPPORTED_GTH_FAMILY",
    "build_default_h2_local_ionic_potential",
    "build_default_h2_monitor_grid_local_ionic_potential",
    "build_h2_local_patch_development_monitor_grid_local_ionic_potential",
    "build_default_h2_nonlocal_ionic_action",
    "evaluate_atomic_local_potential",
    "evaluate_atomic_local_potential_on_legacy_grid",
    "evaluate_atomic_local_potential_on_monitor_grid",
    "evaluate_atomic_nonlocal_action",
    "evaluate_atomic_projector_field",
    "evaluate_legacy_local_ionic_potential",
    "evaluate_local_ionic_energy",
    "evaluate_local_ionic_potential",
    "evaluate_monitor_grid_local_ionic_potential",
    "evaluate_monitor_grid_local_ionic_potential_with_frozen_patch_field",
    "evaluate_monitor_grid_local_ionic_potential_with_patch",
    "evaluate_nonlocal_ionic_action",
    "load_case_gth_pseudo_data",
    "load_gth_pseudo_data",
    "load_gth_pseudo_data_for_elements",
]
