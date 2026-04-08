"""Post-fix baseline vs freeze-Hartree gap audit on the H2 monitor grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry

from .h2_monitor_grid_scf_amplification_ablation_audit import (
    H2MonitorGridScfAmplificationPairAudit,
)
from .h2_monitor_grid_scf_amplification_ablation_audit import (
    H2MonitorGridScfAmplificationSpinAudit,
)
from .h2_monitor_grid_scf_amplification_ablation_audit import (
    run_h2_monitor_grid_scf_amplification_ablation_audit,
)


@dataclass(frozen=True)
class H2MonitorGridPostFixHartreeGapPair:
    """Pair-level baseline vs freeze-Hartree gap summary."""

    pair_iterations: tuple[int, int]
    baseline_density_residual: float
    freeze_hartree_density_residual: float
    baseline_minus_freeze_hartree_density_residual: float
    baseline_density_residual_ratio: float | None
    freeze_hartree_density_residual_ratio: float | None
    baseline_minus_freeze_hartree_density_residual_ratio: float | None
    baseline_hartree_share: float | None
    freeze_hartree_hartree_share: float | None
    baseline_minus_freeze_hartree_hartree_share: float | None
    baseline_occupied_orbital_overlap_abs: float | None
    freeze_hartree_occupied_orbital_overlap_abs: float | None
    baseline_minus_freeze_hartree_overlap_abs: float | None
    baseline_subspace_rotation_max_angle_deg: float | None
    freeze_hartree_subspace_rotation_max_angle_deg: float | None
    baseline_minus_freeze_hartree_subspace_rotation_max_angle_deg: float | None


@dataclass(frozen=True)
class H2MonitorGridPostFixHartreeGapSpinAudit:
    """Spin-resolved post-fix Hartree gap summary."""

    spin_state_label: str
    pair_gaps: tuple[H2MonitorGridPostFixHartreeGapPair, ...]
    mean_baseline_minus_freeze_hartree_density_residual: float | None
    max_baseline_minus_freeze_hartree_density_residual: float | None
    mean_baseline_minus_freeze_hartree_overlap_abs: float | None
    verdict: str


@dataclass(frozen=True)
class H2MonitorGridPostFixHartreeGapAuditResult:
    """Top-level post-fix baseline vs freeze-Hartree gap audit result."""

    case_name: str
    grid_parameter_summary: str
    singlet: H2MonitorGridPostFixHartreeGapSpinAudit
    triplet: H2MonitorGridPostFixHartreeGapSpinAudit
    note: str


def _difference(current: float | None, reference: float | None) -> float | None:
    if current is None or reference is None:
        return None
    return float(current - reference)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _max(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.max(np.asarray(values, dtype=np.float64)))


def _pair_gap(pair: H2MonitorGridScfAmplificationPairAudit) -> H2MonitorGridPostFixHartreeGapPair:
    baseline = pair.baseline
    freeze_hartree = pair.freeze_hartree
    return H2MonitorGridPostFixHartreeGapPair(
        pair_iterations=pair.pair_iterations,
        baseline_density_residual=float(baseline.density_residual),
        freeze_hartree_density_residual=float(freeze_hartree.density_residual),
        baseline_minus_freeze_hartree_density_residual=float(
            baseline.density_residual - freeze_hartree.density_residual
        ),
        baseline_density_residual_ratio=baseline.density_residual_ratio,
        freeze_hartree_density_residual_ratio=freeze_hartree.density_residual_ratio,
        baseline_minus_freeze_hartree_density_residual_ratio=_difference(
            baseline.density_residual_ratio,
            freeze_hartree.density_residual_ratio,
        ),
        baseline_hartree_share=baseline.hartree_share,
        freeze_hartree_hartree_share=freeze_hartree.hartree_share,
        baseline_minus_freeze_hartree_hartree_share=_difference(
            baseline.hartree_share,
            freeze_hartree.hartree_share,
        ),
        baseline_occupied_orbital_overlap_abs=baseline.occupied_orbital_overlap_abs,
        freeze_hartree_occupied_orbital_overlap_abs=freeze_hartree.occupied_orbital_overlap_abs,
        baseline_minus_freeze_hartree_overlap_abs=_difference(
            baseline.occupied_orbital_overlap_abs,
            freeze_hartree.occupied_orbital_overlap_abs,
        ),
        baseline_subspace_rotation_max_angle_deg=baseline.lowest2_subspace_rotation_max_angle_deg,
        freeze_hartree_subspace_rotation_max_angle_deg=freeze_hartree.lowest2_subspace_rotation_max_angle_deg,
        baseline_minus_freeze_hartree_subspace_rotation_max_angle_deg=_difference(
            baseline.lowest2_subspace_rotation_max_angle_deg,
            freeze_hartree.lowest2_subspace_rotation_max_angle_deg,
        ),
    )


def _spin_gap_audit(
    spin_audit: H2MonitorGridScfAmplificationSpinAudit,
) -> H2MonitorGridPostFixHartreeGapSpinAudit:
    pair_gaps = tuple(_pair_gap(pair) for pair in spin_audit.pair_audits)
    density_gaps = [
        pair.baseline_minus_freeze_hartree_density_residual
        for pair in pair_gaps
    ]
    overlap_gaps = [
        pair.baseline_minus_freeze_hartree_overlap_abs
        for pair in pair_gaps
        if pair.baseline_minus_freeze_hartree_overlap_abs is not None
    ]
    mean_density_gap = _mean(density_gaps)
    max_density_gap = _max(density_gaps)
    mean_overlap_gap = _mean(overlap_gaps)
    if mean_density_gap is None:
        verdict = "No pair-level baseline vs freeze-Hartree comparison was available."
    elif mean_density_gap > 0.0:
        verdict = (
            "Freeze-Hartree still lowers the post-fix density residual on average, "
            "so a material Hartree-feedback gap remains."
        )
    else:
        verdict = (
            "The post-fix baseline is already close to or better than freeze-Hartree "
            "on average, so the remaining Hartree-feedback gap is small or pair-dependent."
        )
    return H2MonitorGridPostFixHartreeGapSpinAudit(
        spin_state_label=spin_audit.spin_state_label,
        pair_gaps=pair_gaps,
        mean_baseline_minus_freeze_hartree_density_residual=mean_density_gap,
        max_baseline_minus_freeze_hartree_density_residual=max_density_gap,
        mean_baseline_minus_freeze_hartree_overlap_abs=mean_overlap_gap,
        verdict=verdict,
    )


def run_h2_monitor_grid_post_fix_hartree_gap_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    source_iteration_count: int = 3,
    controller_name: str = "baseline_linear",
) -> H2MonitorGridPostFixHartreeGapAuditResult:
    """Summarize the remaining post-fix gap between baseline and freeze-Hartree replays."""

    amplification_result = run_h2_monitor_grid_scf_amplification_ablation_audit(
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
        track_lowest_two_states=True,
        controller_name=controller_name,
    )
    return H2MonitorGridPostFixHartreeGapAuditResult(
        case_name=amplification_result.case_name,
        grid_parameter_summary=amplification_result.grid_parameter_summary,
        singlet=_spin_gap_audit(amplification_result.singlet),
        triplet=_spin_gap_audit(amplification_result.triplet),
        note=(
            "This lightweight post-fix audit reuses the early-step amplification replay and "
            "reports the remaining pair-level gap between the baseline singlet/triplet map and "
            "the freeze-Hartree replay on the same small H2 A-grid local-only case. "
            f"controller={controller_name}"
        ),
    )


def print_h2_monitor_grid_post_fix_hartree_gap_summary(
    result: H2MonitorGridPostFixHartreeGapAuditResult,
) -> None:
    """Print the compact post-fix baseline vs freeze-Hartree gap summary."""

    print("IsoGridDFT H2 monitor-grid post-fix baseline vs freeze-Hartree gap audit")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"note: {result.note}")
    for spin_audit in (result.singlet, result.triplet):
        print()
        print(f"spin: {spin_audit.spin_state_label}")
        print(
            "  mean baseline-freezeHartree density residual gap: "
            f"{spin_audit.mean_baseline_minus_freeze_hartree_density_residual}"
        )
        print(
            "  max baseline-freezeHartree density residual gap: "
            f"{spin_audit.max_baseline_minus_freeze_hartree_density_residual}"
        )
        print(
            "  mean baseline-freezeHartree occupied overlap gap: "
            f"{spin_audit.mean_baseline_minus_freeze_hartree_overlap_abs}"
        )
        print(f"  verdict: {spin_audit.verdict}")
        for pair in spin_audit.pair_gaps:
            print(
                f"  pair {pair.pair_iterations[0]}->{pair.pair_iterations[1]}: "
                f"density_gap={pair.baseline_minus_freeze_hartree_density_residual:.12e}, "
                f"overlap_gap={pair.baseline_minus_freeze_hartree_overlap_abs}, "
                f"rotation_gap_deg={pair.baseline_minus_freeze_hartree_subspace_rotation_max_angle_deg}, "
                f"hartree_share_gap={pair.baseline_minus_freeze_hartree_hartree_share}"
            )


def main() -> int:
    result = run_h2_monitor_grid_post_fix_hartree_gap_audit()
    print_h2_monitor_grid_post_fix_hartree_gap_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
