"""Gate 3 実データ検証プローブ v4 (2026-08-25、実装1〜3 反映版)。

## 何を測るか

`docs/agent_coordination` 経由の引き継ぎタスク (2026-08-25 夜) に基づき、
v3 (`scripts/_gate3_episode_probe_v3_2026-08-25.py`) の実測で確定した
3 つの根因に対する修正を実データで再測定する。

1. **実装1 (測定器の是正)**: `D1EpisodeTotals.total_unreconciled` は
   CLOSED/CLOSED_FORCED した episode だけの合算であり、窓の終わりに
   まだ OPEN な episode の残量を含まない。窓を早く切ると
   `total_unreconciled=0` になるが、それは「決着した」ではなく
   「まだ数えていない」だけ (`src/exchange_episode_tracker.py` の
   `D1EpisodeTotals` docstring参照)。`open_episode_outstanding`/
   `ledger_residual_all` を新たに出す。
2. **実装2 (退役 chain の相殺・着弾の転記)**: `retire_side_chains`/
   `_retire_all_chains_at_match_boundary` で `self._chains` から削除
   される chain の相殺・着弾・生成量が `_chains_for_episode` のガードに
   弾かれ episode の集計から丸ごと消えていた (v51 chain6 の相殺 18 個消失、
   327 -> 309)。`retired_canceled`/`retired_landed`/`retired_generated`
   を新たに出す。
3. **実装3 (自己相殺の根治)**: `cancel_own_pending_then_send_surplus`
   (発火した側が自分の受け予定を先に打ち消し、余りだけを相手に送る) を
   FIRE/FINALIZE の登録量に反映する。`raw_generation_total` (自己相殺
   netting 前の生値)・`self_canceled_total` (自己相殺量) を D7 に新たに出す。

## v3 (`_gate3_episode_probe_v3_2026-08-25.py`) からの変更点

`src/exchange_ledger.py`/`src/exchange_episode_tracker.py` 側の変更のみで
本プローブは自動的に新フィールドを受け取る (`dataclasses.asdict(diag)` が
新フィールドを自動的に含むため)。本ファイルは出力の `ledger_extra` に
新フィールドを明示的にも複製し、読みやすくしただけ (処理ロジックは
v3 と同一)。

## 既存資産の流用 (v3 と同じ)

`scripts/collect_indicators_v2.py::_drive_ojama`/`_SideTracker`/
`_update_game_idx` をそのまま import して流用する (v3 と同じ)。

## 制約

- 変更してよいのは本ファイルのみ。
- 既存の出力 (`data/verify/gate3_episode_2026-08-24/`、
  `data/verify/gate3_episode_v3_2026-08-25/` 等) を
  reset / checkout / stash / 削除 / 上書きしない。

使い方 (2026-08-25 再測定):
  python scripts/_gate3_episode_probe_v4_2026-08-25.py \\
      --video data/frames/video_zenchi_c0BQoMJwwQU.mp4 --t0 780.0 --t1 1080.0 \\
      --out-dir data/verify/gate3_episode_v4_2026-08-25/zenchi
出力: <out-dir>/diagnostics.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

cv2.setNumThreads(1)

from scripts.collect_indicators_v2 import (  # noqa: E402
    DEFAULT_FPS,
    TARGET_H,
    TARGET_W,
    _drive_ojama,
    _SideTracker,
    _update_game_idx,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.chain_detector import ChainEvent  # noqa: E402
from src.chain_id_resolver import FINALIZED_SOURCE_SCORE_OCR_DIFF  # noqa: E402
from src.exchange_episode_tracker import (  # noqa: E402
    ChainEventObservation,
    ExchangeEpisodeTracker,
    GenerationObservation,
    PendingUncappedFrame,
    classify_pending_uncapped_delta,
)
from src.exchange_ledger import FINALIZE_DOWNWARD_TOLERANCE  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# 状態機械のウォームアップ (v3 と同じ 30 秒)。
WARMUP_SEC: float = 30.0

# 掛け算式の段を正しく取るために必須の構成 (v3 と同じ)。
FORMULA_FLAGS: dict[str, bool] = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_slide_exit_no_min_display=True,
    enable_formula_step_interlude=True,
)

_SIDE_LABELS: tuple[str, str] = ("1P", "2P")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", type=str,
        default=str(PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"),
    )
    parser.add_argument("--t0", type=float, default=780.0)
    parser.add_argument("--t1", type=float, default=1080.0)
    parser.add_argument(
        "--out-dir", type=str,
        default=str(PROJECT_ROOT / "data/verify/gate3_episode_v4_2026-08-25"),
    )
    return parser.parse_args()


def _chain_event_key(ev: ChainEvent | None) -> tuple | None:
    """ChainEvent の同一性キー (変化検知用。v3 と同じ4値)。"""
    if ev is None:
        return None
    return (round(ev.trigger_sec, 3), ev.mechanism, ev.chain_count, ev.total_score)


def _to_chain_observation(
    side_label: str, ev: ChainEvent, t_sec: float, game_idx: int, elapsed_sec: float,
) -> ChainEventObservation:
    """ChainEvent から tracker が要求する最小情報を抜き出す (v3 と同じ)。"""
    return ChainEventObservation(
        side=side_label, t_sec=t_sec, mechanism=ev.mechanism or "",
        chain_count=ev.chain_count, total_score=ev.total_score,
        ojama_sent=ev.ojama_sent, game_idx=game_idx, elapsed_sec=elapsed_sec,
    )


def _diff_generation(prev_snap, snap) -> dict[str, int]:
    """前フレームとの `total_generated_by_p1/p2` 差分 (v3 から流用)。"""
    return {
        "1P": snap.total_generated_by_p1 - prev_snap.total_generated_by_p1,
        "2P": snap.total_generated_by_p2 - prev_snap.total_generated_by_p2,
    }


def _make_pending_frame(
    t_sec: float, game_idx: int, snap,
    p1_tsumo: bool, p2_tsumo: bool, gen_diff: dict[str, int] | None,
) -> PendingUncappedFrame:
    """1 フレーム分の `PendingUncappedFrame` を組み立てる (v3 と同じ)。"""
    p1_fin = gen_diff is not None and gen_diff["1P"] != 0
    p2_fin = gen_diff is not None and gen_diff["2P"] != 0
    return PendingUncappedFrame(
        t_sec=t_sec, game_idx=game_idx,
        p1_uncapped=float(snap.pending_p1_uncapped),
        p2_uncapped=float(snap.pending_p2_uncapped),
        p1_tsumo_placed=p1_tsumo, p2_tsumo_placed=p2_tsumo,
        p1_chain_finalized=p1_fin, p2_chain_finalized=p2_fin,
    )


def _dump_all_resolved_chains(episode_tracker: ExchangeEpisodeTracker) -> list[dict]:
    """全 resolved chain の生値ダンプ (v3 から流用)。"""
    resolved = episode_tracker._resolver.resolved()  # noqa: SLF001
    return [
        {
            "chain_id": rc.chain_id, "side": rc.side,
            "opened_at_sec": rc.opened_at_sec, "closed_at_sec": rc.closed_at_sec,
            "step_count": rc.step_count, "provisional_score": rc.provisional_score,
            "finalized_score": rc.finalized_score, "was_finalized": rc.was_finalized,
            "force_cut": rc.force_cut, "close_reason": rc.close_reason.name,
            "growth_observed": rc.growth_observed,
            "finalized_source": rc.finalized_source,
        }
        for rc in resolved
    ]


def _dump_rejected_chains(episode_tracker: ExchangeEpisodeTracker) -> list[dict]:
    """`simulate_fallback` で確定した chain の生値ダンプ (v3 から流用)。"""
    resolved = episode_tracker._resolver.resolved()  # noqa: SLF001
    return [
        {
            "chain_id": rc.chain_id, "side": rc.side,
            "growth_observed": rc.growth_observed,
            "provisional_score": rc.provisional_score,
            "finalized_score": rc.finalized_score,
        }
        for rc in resolved
        if rc.was_finalized and rc.finalized_source != FINALIZED_SOURCE_SCORE_OCR_DIFF
    ]


def _process_frame_chain_events(
    result, t_sec: float, t0: float, game_idx: int, elapsed_sec: float,
    episode_tracker: ExchangeEpisodeTracker, last_chain_key: dict[str, tuple | None],
    raw_chain_obs: list[dict],
) -> int:
    """1 フレーム分の ChainEvent 変化を検知して観測を供給する (v3 から流用)。"""
    n = 0
    for side_label, side_result in (("1P", result.p1), ("2P", result.p2)):
        ev = side_result.chain_event
        key = _chain_event_key(ev)
        changed = key != last_chain_key[side_label]
        last_chain_key[side_label] = key
        if changed and t_sec >= t0 and ev is not None:
            episode_tracker.observe(
                _to_chain_observation(side_label, ev, t_sec, game_idx, elapsed_sec),
            )
            n += 1
            raw_chain_obs.append({
                "t_sec": round(t_sec, 3), "side": side_label,
                "mechanism": ev.mechanism, "chain_count": ev.chain_count,
                "total_score": ev.total_score,
            })
    return n


def _run_probe(video: Path, t0: float, t1: float, out_dir: Path) -> None:  # noqa: PLR0915
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {video}", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    start_sec = max(0.0, t0 - WARMUP_SEC)
    start_frame = int(start_sec * fps)
    end_frame = int(t1 * fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
        **FORMULA_FLAGS,
    )
    ojama_tracker = OjamaAccountingTracker()
    ojama_tracker.reset()
    tracker_p1 = _SideTracker()
    tracker_p2 = _SideTracker()
    prev_state_p1 = BoardState.MENU
    prev_state_p2 = BoardState.MENU
    episode_tracker = ExchangeEpisodeTracker(enabled=True)

    last_chain_key: dict[str, tuple | None] = {"1P": None, "2P": None}
    prev_snap = None
    prev_pending_frame: PendingUncappedFrame | None = None
    prev_tsumo = {"1P": None, "2P": None}
    n_observed_frames = 0
    n_chain_observations = 0
    n_settlement_observations = 0
    n_generation_observations = 0
    n_negative_generation_diffs = 0
    n_wipe_observations = 0
    unclassified_drop_total = {"1P": 0.0, "2P": 0.0}
    n_unclassified_frames = 0
    raw_chain_obs: list[dict] = []

    fi = start_frame
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        t_sec = fi / fps
        if fi % 1800 == 0:
            print(f"[progress] fi={fi} t={t_sec:.2f}", flush=True)
        result = pipeline.update(fi, t_sec, frame)
        snap = _drive_ojama(
            ojama_tracker, result.p1, result.p2, prev_state_p1, prev_state_p2, t_sec,
            tracker_p1=tracker_p1, tracker_p2=tracker_p2, pipeline=pipeline,
        )
        _update_game_idx(tracker_p1, result.p1.score)
        _update_game_idx(tracker_p2, result.p2.score)
        prev_state_p1 = result.p1.state
        prev_state_p2 = result.p2.state

        game_idx = max(tracker_p1.game_idx, tracker_p2.game_idx)
        if tracker_p1.game_idx != tracker_p2.game_idx:
            print(
                f"[warn] t={t_sec:.2f} game_idx 不一致: "
                f"p1={tracker_p1.game_idx} p2={tracker_p2.game_idx}", flush=True,
            )
        elapsed_sec = ojama_tracker._elapsed(t_sec)  # noqa: SLF001
        gen_diff = _diff_generation(prev_snap, snap) if prev_snap is not None else None

        n_chain_observations += _process_frame_chain_events(
            result, t_sec, t0, game_idx, elapsed_sec, episode_tracker,
            last_chain_key, raw_chain_obs,
        )

        if gen_diff is not None and t_sec >= t0:
            for side_label in _SIDE_LABELS:
                delta = gen_diff[side_label]
                if delta == 0:
                    continue
                if delta < 0:
                    n_negative_generation_diffs += 1
                    print(
                        f"[warn] t={t_sec:.2f} side={side_label} "
                        f"生成累積カウンタが減少: delta={delta}", flush=True,
                    )
                episode_tracker.observe_generation(GenerationObservation(
                    side=side_label, t_sec=t_sec, game_idx=game_idx,
                    generated_delta=delta,
                ))
                n_generation_observations += 1

        tsumo_1p = pipeline.tsumo_count("1P")
        tsumo_2p = pipeline.tsumo_count("2P")
        p1_tsumo_placed = prev_tsumo["1P"] is not None and tsumo_1p > prev_tsumo["1P"]
        p2_tsumo_placed = prev_tsumo["2P"] is not None and tsumo_2p > prev_tsumo["2P"]
        curr_pending_frame = _make_pending_frame(
            t_sec, game_idx, snap, p1_tsumo_placed, p2_tsumo_placed, gen_diff,
        )
        if prev_pending_frame is not None and t_sec >= t0:
            classification = classify_pending_uncapped_delta(
                prev_pending_frame, curr_pending_frame,
            )
            if classification.settlement is not None:
                episode_tracker.observe_settlement(classification.settlement)
                n_settlement_observations += 1
            for side in classification.wiped_sides:
                episode_tracker.observe_wipe(side, t_sec, game_idx)
                n_wipe_observations += 1
            if (
                classification.unclassified_drop_p1 > 0.0
                or classification.unclassified_drop_p2 > 0.0
            ):
                n_unclassified_frames += 1
                unclassified_drop_total["1P"] += classification.unclassified_drop_p1
                unclassified_drop_total["2P"] += classification.unclassified_drop_p2
                print(
                    f"[warn] t={t_sec:.2f} 未分類の pending 減少: "
                    f"1P={classification.unclassified_drop_p1:.1f} "
                    f"2P={classification.unclassified_drop_p2:.1f}", flush=True,
                )
        prev_pending_frame = curr_pending_frame
        prev_tsumo["1P"], prev_tsumo["2P"] = tsumo_1p, tsumo_2p
        prev_snap = snap
        if t_sec >= t0:
            n_observed_frames += 1
        fi += 1

    cap.release()
    episode_tracker.finish()
    diag = episode_tracker.diagnostics()

    ledger_snapshot = episode_tracker._ledger.snapshot()  # noqa: SLF001
    closed = episode_tracker._ledger.closed_episodes()  # noqa: SLF001
    closed_status_counts: dict[str, int] = {}
    for summary in closed:
        closed_status_counts[summary.status.name] = (
            closed_status_counts.get(summary.status.name, 0) + 1
        )
    d2_divergence_count = len(diag.d2.divergences)
    d2_abs_over_tolerance_count = sum(
        1 for d in diag.d2.divergences if abs(d) > FINALIZE_DOWNWARD_TOLERANCE
    )
    d2_abs_over_tolerance_ratio = (
        d2_abs_over_tolerance_count / d2_divergence_count if d2_divergence_count else None
    )

    report = {
        "video": str(video),
        "window": {"t0": t0, "t1": t1, "warmup_sec": WARMUP_SEC},
        "n_observed_frames": n_observed_frames,
        "n_chain_observations": n_chain_observations,
        "n_settlement_observations": n_settlement_observations,
        "n_wipe_observations": n_wipe_observations,
        "n_generation_observations": n_generation_observations,
        "n_negative_generation_diffs_seen_in_probe": n_negative_generation_diffs,
        "n_unclassified_frames": n_unclassified_frames,
        "unclassified_drop_total": unclassified_drop_total,
        "rejected_chain_dump": _dump_rejected_chains(episode_tracker),
        "all_chains_dump": _dump_all_resolved_chains(episode_tracker),
        "raw_chain_obs": raw_chain_obs,
        "diagnostics": dataclasses.asdict(diag),
        "ledger_extra": {
            "retired_chain_count": ledger_snapshot.retired_chain_count,
            "retired_unreconciled": ledger_snapshot.retired_unreconciled,
            # 実装2 (2026-08-25 追加): 退役 chain の相殺・着弾・生成量の転記。
            "retired_canceled": ledger_snapshot.retired_canceled,
            "retired_landed": ledger_snapshot.retired_landed,
            "retired_generated": ledger_snapshot.retired_generated,
            "duplicate_generated_suppressed_count":
                ledger_snapshot.duplicate_generated_suppressed_count,
            "duplicate_generated_suppressed_amount":
                ledger_snapshot.duplicate_generated_suppressed_amount,
            "finalize_rejected_count": ledger_snapshot.finalize_rejected_count,
            "finalize_rejected_amount": ledger_snapshot.finalize_rejected_amount,
        },
        # 実装1 (2026-08-25 追加): 測定器の是正。「0 は測っていないだけ」を
        # 検出するための2枠 (D1 に同じ値が入っているが、読みやすさのため
        # トップレベルにも複製する)。
        "d1_measurement_honesty": {
            "total_unreconciled": diag.d1.total_unreconciled,
            "open_episode_outstanding": diag.d1.open_episode_outstanding,
            "ledger_residual_all": diag.d1.ledger_residual_all,
        },
        # 実装3 (2026-08-25 追加): 自己相殺の根治。「元の生成」検算比較用。
        "d7_self_cancel": {
            "raw_generation_total": diag.d7.raw_generation_total,
            "self_canceled_total": diag.d7.self_canceled_total,
            "net_generation_total_check": (
                diag.d7.raw_generation_total - diag.d7.self_canceled_total
            ),
        },
        "n_closed_episodes": len(closed),
        "closed_episode_status_counts": closed_status_counts,
        "d2_divergence_count": d2_divergence_count,
        "d2_abs_over_tolerance_count": d2_abs_over_tolerance_count,
        "d2_abs_over_tolerance_ratio": d2_abs_over_tolerance_ratio,
        # デバッグ専用 (2026-08-25 追加、恒久ではない): 自己相殺ルックアップの
        # 生値。chain 単位でどれだけ自己相殺されたかを事後検証するため。
        "self_cancel_lookup_debug": [
            {"side": side.name, "t_sec": t_sec, "amount": amount}
            for (side, t_sec), amount in episode_tracker._build_self_cancel_lookup().items()  # noqa: SLF001
        ],
        "ledger_chain_amounts_debug": [
            {
                "chain_id": cid, "side": rec.side.name,
                "opened_at_sec": rec.opened_at_sec,
                "amount": rec.amount, "canceled": rec.canceled, "landed": rec.landed,
                "outstanding": rec.outstanding,
            }
            for cid, rec in episode_tracker._ledger._chains.items()  # noqa: SLF001
        ],
    }
    out_path = out_dir / "diagnostics.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    args = _parse_args()
    _run_probe(Path(args.video), args.t0, args.t1, Path(args.out_dir))
