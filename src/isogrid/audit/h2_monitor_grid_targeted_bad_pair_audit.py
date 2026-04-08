"""Targeted audit for early-step pairs that remain Hartree-sensitive after the post-fix update."""

from __future__ import annotations

from dataclasses import dataclass

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry

from .h2_monitor_grid_post_fix_hartree_gap_audit import (
    H2MonitorGridPostFixHartreeGapPair,
)
from .h2_monitor_grid_post_fix_hartree_gap_audit import (
    H2MonitorGridPostFixHartreeGapSpinAudit,
)
from .h2_monitor_grid_post_fix_hartree_gap_audit import (
    run_h2_monitor_grid_post_fix_hartree_gap_audit,
)

_DEFAULT_SOURCE_ITERATION_COUNT = 4
_DEFAULT_DENSITY_GAP_THRESHOLD = 1.0e-2
_DEFAULT_HARTREE_SHARE_THRESHOLD = 0.60


@dataclass(frozen=True)
class H2MonitorGridTargetedBadPair:
    """One targeted early-step pair that still shows a material Hartree gap."""

    pair_iterations: tuple[int, int]
    baseline_minus_freeze_hartree_density_residual: float
    baseline_minus_freeze_hartree_density_residual_ratio: float | None
    baseline_hartree_share: float | None
    baseline_minus_freeze_hartree_overlap_abs: float | None
    baseline_minus_freeze_hartree_subspace_rotation_max_angle_deg: float | None
    bad_pair_score: float
    selection_reason: str


@dataclass(frozen=True)
class H2MonitorGridTargetedBadPairSpinAudit:
    """Spin-resolved targeted bad-pair selection summary."""

    spin_state_label: str
    targeted_pairs: tuple[H2MonitorGridTargetedBadPair, ...]
    density_gap_threshold: float
    hartree_share_threshold: float
    verdict: str


@dataclass(frozen=True)
class H2MonitorGridTargetedBadPairAuditResult:
    """Top-level targeted bad-pair audit result."""

    case_name: str
    grid_parameter_summary: str
    singlet: H2MonitorGridTargetedBadPairSpinAudit
    triplet: H2MonitorGridTargetedBadPairSpinAudit
    note: str


def _pair_score(pair: H2MonitorGridPostFixHartreeGapPair) -> float:
    density_gap = max(float(pair.baseline_minus_freeze_hartree_density_residual), 0.0)
    hartree_share = max(float(pair.baseline_hartree_share or 0.0), 0.0)
    rotation_gap = max(
        float(pair.baseline_minus_freeze_hartree_subspace_rotation_max_angle_deg or 0.0),
        0.0,
    )
    return float(density_gap * (1.0 + hartree_share + rotation_gap / 90.0))


def _selection_reason(
    pair: H2MonitorGridPostFixHartreeGapPair,
    *,
    density_gap_threshold: float,
    hartree_share_threshold: float,
) -> str:
    return (
        "Selected because baseline still exceeds freeze-Hartree by "
        f"{pair.baseline_minus_freeze_hartree_density_residual:.6f} in density residual, "
        f"with baseline Hartree share={pair.baseline_hartree_share} and "
        f"rotation-gap={pair.baseline_minus_freeze_hartree_subspace_rotation_max_angle_deg} deg."
        f" Thresholds: density_gap>{density_gap_threshold}, "
        f"hartree_share>={hartree_share_threshold}."
    )


def _targeted_spin_audit(
    spin_audit: H2MonitorGridPostFixHartreeGapSpinAudit,
    *,
    density_gap_threshold: float,
    hartree_share_threshold: float,
) -> H2MonitorGridTargetedBadPairSpinAudit:
    selected: list[H2MonitorGridTargetedBadPair] = []
    for pair in spin_audit.pair_gaps:
        hartree_share = pair.baseline_hartree_share
        qualifies = (
            pair.baseline_minus_freeze_hartree_density_residual > density_gap_threshold
            and hartree_share is not None
            and hartree_share >= hartree_share_threshold
        )
        if not qualifies:
            continue
        selected.append(
            H2MonitorGridTargetedBadPair(
                pair_iterations=pair.pair_iterations,
                baseline_minus_freeze_hartree_density_residual=float(
                    pair.baseline_minus_freeze_hartree_density_residual
                ),
                baseline_minus_freeze_hartree_density_residual_ratio=(
                    pair.baseline_minus_freeze_hartree_density_residual_ratio
                ),
                baseline_hartree_share=hartree_share,
                baseline_minus_freeze_hartree_overlap_abs=(
                    pair.baseline_minus_freeze_hartree_overlap_abs
                ),
                baseline_minus_freeze_hartree_subspace_rotation_max_angle_deg=(
                    pair.baseline_minus_freeze_hartree_subspace_rotation_max_angle_deg
                ),
                bad_pair_score=_pair_score(pair),
                selection_reason=_selection_reason(
                    pair,
                    density_gap_threshold=density_gap_threshold,
                    hartree_share_threshold=hartree_share_threshold,
                ),
            )
        )
    selected.sort(key=lambda pair: pair.bad_pair_score, reverse=True)
    if selected:
        verdict = (
            "Targeted bad pairs remain after the post-fix update; these pairs are the best "
            "next lightweight probes for the residual Hartree-feedback gap."
        )
    else:
        verdict = (
            "No early-step pair exceeded the targeted Hartree-gap thresholds, so the remaining "
            "post-fix gap is either small or below the current lightweight cutoff."
        )
    return H2MonitorGridTargetedBadPairSpinAudit(
        spin_state_label=spin_audit.spin_state_label,
        targeted_pairs=tuple(selected),
        density_gap_threshold=float(density_gap_threshold),
        hartree_share_threshold=float(hartree_share_threshold),
        verdict=verdict,
    )


def run_h2_monitor_grid_targeted_bad_pair_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    density_gap_threshold: float = _DEFAULT_DENSITY_GAP_THRESHOLD,
    hartree_share_threshold: float = _DEFAULT_HARTREE_SHARE_THRESHOLD,
    controller_name: str = "baseline_linear",
) -> H2MonitorGridTargetedBadPairAuditResult:
    """Select the early-step post-fix pairs that still look materially Hartree-sensitive."""

    post_fix_result = run_h2_monitor_grid_post_fix_hartree_gap_audit(
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
        controller_name=controller_name,
    )
    return H2MonitorGridTargetedBadPairAuditResult(
        case_name=post_fix_result.case_name,
        grid_parameter_summary=post_fix_result.grid_parameter_summary,
        singlet=_targeted_spin_audit(
            post_fix_result.singlet,
            density_gap_threshold=density_gap_threshold,
            hartree_share_threshold=hartree_share_threshold,
        ),
        triplet=_targeted_spin_audit(
            post_fix_result.triplet,
            density_gap_threshold=density_gap_threshold,
            hartree_share_threshold=hartree_share_threshold,
        ),
        note=(
            "This lightweight targeted audit starts from the post-fix baseline vs freeze-Hartree "
            "gap summary, then keeps only the early-step pairs whose baseline residual still "
            "meaningfully exceeds freeze-Hartree under a Hartree-dominated channel share. "
            f"controller={controller_name}"
        ),
    )


def print_h2_monitor_grid_targeted_bad_pair_summary(
    result: H2MonitorGridTargetedBadPairAuditResult,
) -> None:
    """Print the targeted bad-pair summary."""

    print("IsoGridDFT H2 monitor-grid targeted bad-pair audit")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"note: {result.note}")
    for spin_audit in (result.singlet, result.triplet):
        print()
        print(f"spin: {spin_audit.spin_state_label}")
        print(
            "  thresholds: "
            f"density_gap>{spin_audit.density_gap_threshold}, "
            f"hartree_share>={spin_audit.hartree_share_threshold}"
        )
        print(f"  verdict: {spin_audit.verdict}")
        for pair in spin_audit.targeted_pairs:
            print(
                f"  pair {pair.pair_iterations[0]}->{pair.pair_iterations[1]}: "
                f"score={pair.bad_pair_score:.12e}, "
                f"density_gap={pair.baseline_minus_freeze_hartree_density_residual:.12e}, "
                f"ratio_gap={pair.baseline_minus_freeze_hartree_density_residual_ratio}, "
                f"hartree_share={pair.baseline_hartree_share}, "
                f"overlap_gap={pair.baseline_minus_freeze_hartree_overlap_abs}, "
                f"rotation_gap_deg={pair.baseline_minus_freeze_hartree_subspace_rotation_max_angle_deg}"
            )


def main() -> int:
    result = run_h2_monitor_grid_targeted_bad_pair_audit()
    print_h2_monitor_grid_targeted_bad_pair_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
