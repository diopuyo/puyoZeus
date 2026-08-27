"""測定1 追加計装: `total_generated` がエピソード単位で二重計上されていないか
(2026-08-25、`_diag_gate3_breakdown_2026-08-25.py` の疑いを確認する)。

`D1EpisodeTotals.total_generated` は `ExchangeLedger.closed_episodes()` を
**episode 単位**で合計する。lazy open (`_fire_events_of_open_chains`) は
「既に開いている未決着 chain」を新しい episode へも `FIRE` として引き継ぐため、
同じ chain_id が複数の episode の `events` に現れうる。
`_summarize_episode` は `{e.chain_id for e in ep.events}` で **episode 内**の
重複は除くが、**episode をまたいだ重複は除かない**。

本スクリプトは `episode_tracker._ledger._chains` (read-only, 全 chain_id の
最終状態を chain_id ごとに一意に保持) から**大域で重複排除した**
`sum(rec.amount)` を計算し、D1 の `total_generated` (episode 単位合計、
重複排除なし) と比較する。差があれば「同じ chain の量が複数 episode に
またがって再計上されている」ことの直接証拠になる。

**本番コードは読むだけ。変更しない。**
"""
from __future__ import annotations

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
from src.exchange_episode_tracker import (  # noqa: E402
    ChainEventObservation,
    ExchangeEpisodeTracker,
    SettlementObservation,
)
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# 引数で video/window を切り替える (measurement1 / measurement2 の両方を検査するため)。
import argparse  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--t0", type=float, required=True)
parser.add_argument("--t1", type=float, required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

VIDEO = Path(args.video)
OUT_PATH = Path(args.out)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

T0: float = args.t0
T1: float = args.t1
WARMUP_SEC: float = 30.0

FORMULA_FLAGS: dict[str, bool] = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_slide_exit_no_min_display=True,
    enable_formula_step_interlude=True,
)


def _chain_event_key(ev: ChainEvent | None) -> tuple | None:
    if ev is None:
        return None
    return (round(ev.trigger_sec, 3), ev.mechanism, ev.chain_count, ev.total_score)


def _to_chain_observation(
    side_label: str, ev: ChainEvent, t_sec: float, game_idx: int, elapsed_sec: float,
) -> ChainEventObservation:
    return ChainEventObservation(
        side=side_label, t_sec=t_sec, mechanism=ev.mechanism or "",
        chain_count=ev.chain_count, total_score=ev.total_score,
        ojama_sent=ev.ojama_sent, game_idx=game_idx, elapsed_sec=elapsed_sec,
    )


def _diff_settlement(prev_snap, snap, t_sec: float, game_idx: int):
    d_c1 = snap.total_offset_by_p1 - prev_snap.total_offset_by_p1
    d_c2 = snap.total_offset_by_p2 - prev_snap.total_offset_by_p2
    d_l1 = snap.total_dropped_to_p1 - prev_snap.total_dropped_to_p1
    d_l2 = snap.total_dropped_to_p2 - prev_snap.total_dropped_to_p2
    d_c1, d_c2, d_l1, d_l2 = (max(0.0, v) for v in (d_c1, d_c2, d_l1, d_l2))
    if d_c1 == 0.0 and d_c2 == 0.0 and d_l1 == 0.0 and d_l2 == 0.0:
        return None
    return SettlementObservation(
        t_sec=t_sec, game_idx=game_idx,
        canceled_by_1p=float(d_c1), canceled_by_2p=float(d_c2),
        landed_on_1p=float(d_l1), landed_on_2p=float(d_l2),
    )


def _run() -> None:
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    start_sec = max(0.0, T0 - WARMUP_SEC)
    start_frame = int(start_sec * fps)
    end_frame = int(T1 * fps)
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
        elapsed_sec = ojama_tracker._elapsed(t_sec)  # noqa: SLF001

        for side_label, side_result in (("1P", result.p1), ("2P", result.p2)):
            ev = side_result.chain_event
            key = _chain_event_key(ev)
            changed = key != last_chain_key[side_label]
            last_chain_key[side_label] = key
            if changed and t_sec >= T0 and ev is not None:
                episode_tracker.observe(_to_chain_observation(
                    side_label, ev, t_sec, game_idx, elapsed_sec,
                ))

        if prev_snap is not None and t_sec >= T0:
            settlement = _diff_settlement(prev_snap, snap, t_sec, game_idx)
            if settlement is not None:
                episode_tracker.observe_settlement(settlement)
        prev_snap = snap
        fi += 1

    cap.release()
    episode_tracker.finish()
    diag = episode_tracker.diagnostics()

    ledger = episode_tracker._ledger  # noqa: SLF001
    closed = ledger.closed_episodes()

    # episode 単位合計 (D1 と同じロジックを再現)
    episode_level_total = sum(s.total_generated for s in closed)

    # 大域で chain_id 重複排除した合計 (真の生成量、あるべき値)
    global_chains = ledger._chains  # noqa: SLF001
    global_dedup_total = sum(rec.amount for rec in global_chains.values())

    # chain_id がいくつの episode にまたがって現れたか (episode の events から復元)
    # ClosedEpisodeSummary には chain_id 詳細が無いため、Episode オブジェクト自体は
    # 破棄済み。ここでは episode 数 と chain 数の比較にとどめる (対応表は次段階)。
    report = {
        "video": str(VIDEO), "window": {"t0": T0, "t1": T1},
        "d1_total_generated": diag.d1.total_generated,
        "n_closed_episodes": len(closed),
        "episode_level_total_generated_recomputed": episode_level_total,
        "n_unique_chain_ids_in_ledger": len(global_chains),
        "global_dedup_total_generated": global_dedup_total,
        "gap_episode_minus_global_dedup": episode_level_total - global_dedup_total,
        "closed_episode_summaries": [
            {
                "episode_id": s.episode_id, "status": s.status.name,
                "close_reason": s.close_reason,
                "opened_at_sec": s.opened_at_sec, "closed_at_sec": s.closed_at_sec,
                "total_generated": s.total_generated,
                "total_canceled": s.total_canceled, "total_landed": s.total_landed,
                "unreconciled": s.unreconciled,
                "has_settlement_input": s.has_settlement_input,
            }
            for s in closed
        ],
    }
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[saved] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    _run()
