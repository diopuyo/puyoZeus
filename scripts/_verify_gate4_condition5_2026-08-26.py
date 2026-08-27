"""Gate 4 条件5の8区間sidecarを母数付きで検収する。"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPS = 1e-6
REQUIRED = {
    "t_sec", "state1", "state2", "closed_episode_count", "last_closed_status",
    "last_close_reason",
    "last_closed_generated", "last_closed_canceled", "last_closed_landed",
    "last_closed_unreconciled", "last_closed_has_settlement",
    "closed_unreconciled_total",
    "closed_normal_unreconciled_count",
    "last_closed_oversettled", "gross_inspected_sides",
    "gross_residual_p1", "gross_residual_p2", "is_unresolved",
    "allows_hard_override", "hard_override_target",
    "hard_override_candidate", "hard_override_applied",
    "hard_override_path", "hard_override_hold_reason", "unattributed_settlement_total",
    "duplicate_generated_suppressed_count", "finalize_rejected_count",
    "unbacked_residual_count", "post_close_settlement_dropped_count",
    "post_close_settlement_backfilled_count",
    "simulate_excluded_chain_count", "resolved_chain_count",
    "open_episode_outstanding", "ledger_residual_all",
    "unreconciled",
    "total_generated", "total_canceled", "total_landed",
    "post_close_finalize_backfilled_count",
    "post_close_finalize_dropped_count",
    "post_retire_backfilled_count",
    "post_close_outstanding_delta_total",
    "post_close_growth_backfilled_count",
    "post_close_growth_dropped_count",
    "formula_step_observation_count",
    "provisional_score_decrease_ignored_count",
    "boundary_count", "boundary_settlement_excluded_count",
    "boundary_settlement_excluded_amount",
}


@dataclass
class Totals:
    rows: int = 0
    per_side_settled_rows: int = 0
    both_stable_rows: int = 0
    gross_sides: int = 0
    gross_residual_bad: int = 0
    global_conservation_bad_rows: int = 0
    post_close_backfill_sync_bad: int = 0
    closed: int = 0
    closed_normal: int = 0
    closed_forced: int = 0
    closed_other: int = 0
    max_sec_closed: int = 0
    settlement_backed_closed: int = 0
    no_settlement: int = 0
    conservation_bad: int = 0
    normal_unreconciled_bad: int = 0
    oversettled_bad: int = 0
    segment_end_open: int = 0
    hard_candidates: int = 0
    hard_held: int = 0
    hard_illegal: int = 0
    hard_target_inconsistent: int = 0
    hard_direction_conflict_held: int = 0
    hard_physical_target_capped: int = 0
    hard_live: int = 0
    hard_hold: int = 0
    resolved_chains: int = 0
    simulate_excluded: int = 0
    formula_steps: int = 0
    provisional_decrease_ignored: int = 0
    abnormal_cumulative: int = 0
    unattributed_segments: int = 0
    duplicate_segments: int = 0
    finalize_rejected_segments: int = 0
    unbacked_segments: int = 0
    post_close_segments: int = 0
    finalize_dropped_segments: int = 0
    backfilled_settlements: int = 0
    backfilled_finalizes: int = 0
    retire_backfills: int = 0
    growth_backfills: int = 0
    growth_dropped_segments: int = 0
    boundaries: int = 0
    boundary_settlements_excluded: int = 0
    boundary_settlement_amount: float = 0.0

    def add(self, other: "Totals") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as data:
        missing = REQUIRED - set(data.files)
        if missing:
            raise ValueError(f"{path}: 必須列不足 {sorted(missing)}")
        return {name: data[name] for name in data.files}


def _closed_row_indexes(d: dict[str, np.ndarray], path: Path) -> np.ndarray:
    counts = d["closed_episode_count"].astype(int)
    if not len(counts):
        raise ValueError(f"{path}: sidecar行が0")
    # warmup中のcloseは表示窓の母集団外。最初の記録値をbaselineにする。
    previous = np.concatenate(([counts[0]], counts[:-1]))
    jumps = counts - previous
    if np.any(jumps < 0) or np.any(jumps > 1):
        raise ValueError(f"{path}: close countが非単調または1frameで複数close")
    # max_sec後の遅延決済backfillはclose件数を増やさず、直近要約を更新する。
    # close瞬間でなく、次のclose直前（または区間末尾）の最終値を検査する。
    return np.asarray([
        int(np.flatnonzero(counts == value)[-1])
        for value in range(int(counts[0]) + 1, int(counts[-1]) + 1)
    ], dtype=int)


def _closed_stats(d: dict[str, np.ndarray], indexes: np.ndarray) -> Totals:
    out = Totals(closed=len(indexes))
    if not len(indexes):
        return out
    status = d["last_closed_status"][indexes].astype(str)
    reason = d["last_close_reason"][indexes].astype(str)
    has_input = d["last_closed_has_settlement"][indexes].astype(bool)
    normal = (status == "CLOSED") & (reason == "normal_close")
    forced = status == "CLOSED_FORCED"
    expected = (
        d["last_closed_canceled"][indexes].astype(float)
        + d["last_closed_landed"][indexes].astype(float)
        + d["last_closed_unreconciled"][indexes].astype(float))
    residual = d["last_closed_generated"][indexes].astype(float) - expected
    out.closed_normal = int(np.count_nonzero(normal))
    out.closed_forced = int(np.count_nonzero(forced))
    out.closed_other = int(len(indexes) - out.closed_normal - out.closed_forced)
    out.max_sec_closed = int(np.count_nonzero(reason == "max_sec"))
    out.settlement_backed_closed = int(np.count_nonzero(has_input))
    out.no_settlement = int(np.count_nonzero(normal & ~has_input))
    out.conservation_bad = int(np.count_nonzero(has_input & (abs(residual) > EPS)))
    out.normal_unreconciled_bad = int(np.count_nonzero(
        normal & (d["last_closed_unreconciled"][indexes].astype(float) > EPS)))
    out.oversettled_bad = int(np.count_nonzero(
        d["last_closed_oversettled"][indexes].astype(float) > EPS))
    return out


def _cumulative_abnormal_flags(d: dict[str, np.ndarray]) -> dict[str, int]:
    return {
        "unattributed_segments": int(float(d["unattributed_settlement_total"][-1]) > EPS),
        "duplicate_segments": int(
            float(d["duplicate_generated_suppressed_count"][-1]) > EPS),
        "finalize_rejected_segments": int(
            float(d["finalize_rejected_count"][-1]) > EPS),
        "unbacked_segments": int(float(d["unbacked_residual_count"][-1]) > EPS),
        "post_close_segments": int(
            float(d["post_close_settlement_dropped_count"][-1]) > EPS),
        "finalize_dropped_segments": int(
            float(d["post_close_finalize_dropped_count"][-1]) > EPS),
        "growth_dropped_segments": int(
            float(d["post_close_growth_dropped_count"][-1]) > EPS),
    }


def _post_close_backfill_sync_bad(d: dict[str, np.ndarray]) -> int:
    """後着更新時のglobal未照合量と直近episode要約の増減を突合する。"""
    growth = d.get(
        "post_close_growth_backfilled_count",
        np.zeros_like(d["post_close_finalize_backfilled_count"], dtype=int))
    count = (
        d["post_close_settlement_backfilled_count"].astype(int)
        + d["post_close_finalize_backfilled_count"].astype(int)
        + growth.astype(int))
    count_delta = np.diff(count, prepend=count[0])
    # 後着と同frameに新chainの生成が起き得るため、台帳全体の残量差ではなく
    # close済みchainだけの符号付きoutstanding差をclosed要約と比較する。
    global_delta = np.diff(
        d["post_close_outstanding_delta_total"].astype(float),
        prepend=float(d["post_close_outstanding_delta_total"][0]))
    summary_delta = np.diff(
        d["closed_unreconciled_total"].astype(float),
        prepend=float(d["closed_unreconciled_total"][0]))
    target = count_delta > 0
    return int(np.count_nonzero(
        target & (np.abs(global_delta - summary_delta) > EPS)))


def _segment_stats(path: Path) -> Totals:
    d = _load(path)
    out = _closed_stats(d, _closed_row_indexes(d, path))
    out.rows = len(d["t_sec"])
    stable1 = d["state1"].astype(str) == "STABLE"
    stable2 = d["state2"].astype(str) == "STABLE"
    out.per_side_settled_rows = int(np.count_nonzero(stable1 | stable2))
    out.both_stable_rows = int(np.count_nonzero(stable1 & stable2))
    out.gross_sides = int(d["gross_inspected_sides"].astype(int).sum())
    bad_p1 = np.abs(d["gross_residual_p1"].astype(float)) > EPS
    bad_p2 = np.abs(d["gross_residual_p2"].astype(float)) > EPS
    out.gross_residual_bad = int(np.count_nonzero(bad_p1) + np.count_nonzero(bad_p2))
    global_residual = (
        d["total_generated"].astype(float)
        - d["total_canceled"].astype(float)
        - d["total_landed"].astype(float)
        - d["ledger_residual_all"].astype(float))
    out.global_conservation_bad_rows = int(
        np.count_nonzero(np.abs(global_residual) > EPS))
    out.post_close_backfill_sync_bad = _post_close_backfill_sync_bad(d)
    out.normal_unreconciled_bad = max(
        out.normal_unreconciled_bad,
        int(d["closed_normal_unreconciled_count"].astype(int).max(initial=0)))
    candidate = d["hard_override_candidate"].astype(bool)
    applied = d["hard_override_applied"].astype(bool)
    unresolved = d["is_unresolved"].astype(bool)
    allows = d["allows_hard_override"].astype(bool)
    blocked = unresolved & ~allows
    target = d["hard_override_target"].astype(float)
    out.hard_candidates = int(np.count_nonzero(candidate))
    out.hard_held = int(np.count_nonzero(candidate & ~applied))
    out.hard_illegal = int(np.count_nonzero(candidate & applied & blocked))
    out.hard_target_inconsistent = int(np.count_nonzero(
        (unresolved & allows & (np.abs(target) < 99.999))
        | (unresolved & ~allows & (np.abs(target) > EPS))
        | (~unresolved & (np.abs(target) > EPS))))
    reasons = d["hard_override_hold_reason"].astype(str)
    out.hard_direction_conflict_held = int(np.count_nonzero(
        candidate & ~applied & (reasons == "episode_direction_conflict")))
    out.hard_physical_target_capped = int(np.count_nonzero(
        candidate & ~applied & (reasons == "episode_physical_target_capped")))
    paths = d["hard_override_path"].astype(str)
    out.hard_live = int(np.count_nonzero(np.char.find(paths, "live") >= 0))
    out.hard_hold = int(np.count_nonzero(np.char.find(paths, "hold_") >= 0))
    out.segment_end_open = int(bool(d["is_unresolved"][-1]))
    out.resolved_chains = int(d["resolved_chain_count"].astype(int).max(initial=0))
    out.simulate_excluded = int(d["simulate_excluded_chain_count"][-1])
    out.formula_steps = int(d["formula_step_observation_count"][-1])
    out.provisional_decrease_ignored = int(
        d["provisional_score_decrease_ignored_count"][-1])
    out.backfilled_settlements = int(
        d["post_close_settlement_backfilled_count"][-1])
    out.backfilled_finalizes = int(
        d["post_close_finalize_backfilled_count"][-1])
    out.retire_backfills = int(d["post_retire_backfilled_count"][-1])
    out.growth_backfills = int(d["post_close_growth_backfilled_count"][-1])
    out.boundaries = int(d["boundary_count"][-1])
    out.boundary_settlements_excluded = int(
        d["boundary_settlement_excluded_count"][-1])
    out.boundary_settlement_amount = float(
        d["boundary_settlement_excluded_amount"][-1])
    abnormal = _cumulative_abnormal_flags(d)
    for name, value in abnormal.items():
        setattr(out, name, value)
    out.abnormal_cumulative = sum(abnormal.values())
    return out


def _print(t: Totals, segment_count: int) -> None:
    print(f"sidecar rows={t.rows} / segments={segment_count}")
    print(f"settled-filterable rows: either side={t.per_side_settled_rows}, "
          f"both sides={t.both_stable_rows}")
    print(f"gross conservation residual nonzero={t.gross_residual_bad}/{t.gross_sides} side")
    print(f"closed episodes={t.closed} (normal={t.closed_normal}, forced={t.closed_forced}, "
          f"other={t.closed_other}, max_sec={t.max_sec_closed}, "
          f"normal without settlement={t.no_settlement})")
    print("all closed conservation violations="
          f"{t.conservation_bad}/{t.settlement_backed_closed}")
    print(f"global live conservation violations={t.global_conservation_bad_rows}/{t.rows} rows")
    print(f"post-close backfill global/episode sync violations="
          f"{t.post_close_backfill_sync_bad}/"
          f"{t.backfilled_settlements + t.backfilled_finalizes + t.growth_backfills}")
    print(f"normal CLOSED with unreconciled>0={t.normal_unreconciled_bad}/{t.closed_normal}")
    print(f"oversettled episodes={t.oversettled_bad}/{t.closed}")
    print(f"segment-end OPEN={t.segment_end_open}/{segment_count}")
    print(f"hard override candidates={t.hard_candidates}, held={t.hard_held}, "
          f"illegal applied while unresolved={t.hard_illegal}")
    print(f"hard override direction-state inconsistencies={t.hard_target_inconsistent}/"
          f"{t.rows}")
    print(f"opposite-direction hard overrides held={t.hard_direction_conflict_held}/"
          f"{t.hard_candidates}")
    print(f"physical winner directions capped={t.hard_physical_target_capped}/"
          f"{t.hard_candidates}")
    print(f"hard override path rows: live={t.hard_live}, hold={t.hard_hold}")
    print(f"resolved chains={t.resolved_chains}, simulate excluded={t.simulate_excluded}")
    print("formula cumulative-score decreases ignored="
          f"{t.provisional_decrease_ignored}/{t.formula_steps} observations")
    print(f"max_sec後に要約へ回収した遅延決済={t.backfilled_settlements}")
    print(f"max_sec後に要約へ回収した遅延確定={t.backfilled_finalizes}")
    print(f"side wipe後に退役台帳へ回収した遅延イベント={t.retire_backfills}")
    print(f"close後に旧要約へ回収した遅延成長={t.growth_backfills}")
    print("境界frameの決済二重適用除外="
          f"{t.boundary_settlements_excluded}/{t.boundaries} boundary, "
          f"amount={t.boundary_settlement_amount:.6g}")
    print(f"abnormal cumulative counters nonzero={t.abnormal_cumulative}/{segment_count * 7}")
    print("abnormal segments: "
          f"unattributed={t.unattributed_segments}, duplicate={t.duplicate_segments}, "
          f"finalize_rejected={t.finalize_rejected_segments}, "
          f"unbacked={t.unbacked_segments}, post_close={t.post_close_segments}, "
          f"finalize_dropped={t.finalize_dropped_segments}, "
          f"growth_dropped={t.growth_dropped_segments}")


def main() -> int:
    root = Path(sys.argv[1])
    expected_segments = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    paths = [root] if root.is_file() else sorted(root.glob("seg*_episode.npz"))
    if len(paths) != expected_segments:
        raise ValueError(
            f"{expected_segments}区間必要: found={len(paths)} root={root}")
    total = Totals()
    for path in paths:
        total.add(_segment_stats(path))
    _print(total, len(paths))
    failed = any((
        total.gross_residual_bad, total.conservation_bad,
        total.global_conservation_bad_rows,
        total.post_close_backfill_sync_bad,
        total.normal_unreconciled_bad, total.no_settlement,
        total.oversettled_bad, total.hard_illegal,
        total.hard_target_inconsistent,
        total.abnormal_cumulative,
    ))
    unexercised = any((
        total.gross_sides == 0, total.per_side_settled_rows == 0,
        total.closed_normal == 0,
        total.resolved_chains == 0, total.hard_candidates == 0,
        total.hard_held == 0,
        total.segment_end_open == len(paths),
        (total.max_sec_closed > 0
         and (total.backfilled_settlements + total.backfilled_finalizes
              + total.growth_backfills) == 0),
    ))
    print(f"condition5 ledger verdict={'FAIL' if (failed or unexercised) else 'PASS'} "
          f"(violation={int(failed)}, unexercised={int(unexercised)})")
    return int(failed or unexercised)


if __name__ == "__main__":
    raise SystemExit(main())
