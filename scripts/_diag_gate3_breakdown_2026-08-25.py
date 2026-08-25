"""Gate 3-2b 実データ検証: `total_generated=45,036` の内訳分解 (2026-08-25)。

コーディネーター指示の測定1に対応する計装スクリプト。
`scripts/_gate3_episode_probe_2026-08-24.py` (2026-08-24、実行1回目) と
**全く同じフレームループ・同じ動画・同じ窓**を使い、以下を追加で行う:

1. `ChainIdResolver` が発行した全 chain_id の生データ (opened/closed/
   close_reason/was_finalized/step_count/score/おじゃま換算) を dump する。
2. A (was_finalized) / B (未確定断片: SUPERSEDED/STEP_DECREASE/STREAM_END/
   FORCE_CUT) / C (その他=MATCH_BOUNDARY) に分解する。
3. 同一 side で時間的に重なる chain_id の組を列挙する。
4. `authoritative_ojama` (`OjamaAccountingTracker.total_generated_by_*` の
   差分) を自己検算に供給する。

**本番コード (`src/` 配下) は読むだけ。変更しない。**
`ExchangeEpisodeTracker` / `ChainIdResolver` の private 属性
(`_resolver`, `_context_by_t`) への read-only アクセスは、既存プローブが
`OjamaAccountingTracker._elapsed` に対して行っている慣習と同じ扱い。

出力: `data/verify/gate3_breakdown_2026-08-25/chain_dump.json` (33件の生データ)
      `data/verify/gate3_breakdown_2026-08-25/diagnostics.json` (集計値)
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
from src.chain_id_resolver import CloseReason  # noqa: E402
from src.exchange_episode_tracker import (  # noqa: E402
    ChainEventObservation,
    ExchangeEpisodeTracker,
    SettlementObservation,
)
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from src.scoring import OJAMA_RATE_STANDARD, score_to_ojama  # noqa: E402

VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
OUT_DIR = PROJECT_ROOT / "data/verify/gate3_breakdown_2026-08-25"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Gate 3-0 / 既存プローブと同じ窓 (再現条件を本番/前回実行と一致させる)。
T0: float = 780.0
T1: float = 1080.0
WARMUP_SEC: float = 30.0

FORMULA_FLAGS: dict[str, bool] = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_slide_exit_no_min_display=True,
    enable_formula_step_interlude=True,
)

# B: 未確定のまま閉じた断片 (コーディネーター指定の4種)。
_B_REASONS = {
    CloseReason.SUPERSEDED, CloseReason.STEP_DECREASE,
    CloseReason.STREAM_END, CloseReason.FORCE_CUT,
}


def _chain_event_key(ev: ChainEvent | None) -> tuple | None:
    if ev is None:
        return None
    return (round(ev.trigger_sec, 3), ev.mechanism, ev.chain_count, ev.total_score)


def _to_chain_observation(
    side_label: str, ev: ChainEvent, t_sec: float, game_idx: int, elapsed_sec: float,
    authoritative_ojama: float | None,
) -> ChainEventObservation:
    return ChainEventObservation(
        side=side_label, t_sec=t_sec, mechanism=ev.mechanism or "",
        chain_count=ev.chain_count, total_score=ev.total_score,
        ojama_sent=ev.ojama_sent, game_idx=game_idx, elapsed_sec=elapsed_sec,
        authoritative_ojama=authoritative_ojama,
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
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {VIDEO}", file=sys.stderr)
        sys.exit(1)
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
    # authoritative_ojama 供給用: 直近のチェインイベント時点での
    # OjamaAccountingTracker 側の累積確定生成量 (side別)。
    last_auth_total: dict[str, float] = {"1P": 0.0, "2P": 0.0}
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

        for side_label, side_result, cum_attr in (
            ("1P", result.p1, "total_generated_by_p1"),
            ("2P", result.p2, "total_generated_by_p2"),
        ):
            ev = side_result.chain_event
            key = _chain_event_key(ev)
            changed = key != last_chain_key[side_label]
            last_chain_key[side_label] = key
            if changed and t_sec >= T0 and ev is not None:
                cum_now = getattr(snap, cum_attr)
                auth_delta = cum_now - last_auth_total[side_label]
                last_auth_total[side_label] = cum_now
                episode_tracker.observe(_to_chain_observation(
                    side_label, ev, t_sec, game_idx, elapsed_sec,
                    authoritative_ojama=float(auth_delta),
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

    # ------------------------------
    # 33本の生データ dump (read-only private アクセス)
    # ------------------------------
    resolved = episode_tracker._resolver.resolved()  # noqa: SLF001
    context_by_t = episode_tracker._context_by_t  # noqa: SLF001

    def _to_ojama(score: int, elapsed: float) -> int:
        return score_to_ojama(
            score, prev_leftover=0, elapsed_sec=elapsed, rate_base=OJAMA_RATE_STANDARD,
        ).ojama_count

    dump = []
    bucket_sums = {"A": 0.0, "B": 0.0, "C": 0.0}
    bucket_counts = {"A": 0, "B": 0, "C": 0}
    for rc in resolved:
        fire_ctx = context_by_t.get(rc.opened_at_sec)
        fire_elapsed = fire_ctx.elapsed_sec if fire_ctx else 0.0
        prov_ojama = _to_ojama(rc.provisional_score, fire_elapsed)
        fin_ojama = None
        fin_elapsed = None
        auth_ojama = None
        if rc.was_finalized and rc.finalized_score is not None:
            fin_ctx = context_by_t.get(rc.closed_at_sec)
            fin_elapsed = fin_ctx.elapsed_sec if fin_ctx else fire_elapsed
            fin_ojama = _to_ojama(rc.finalized_score, fin_elapsed)
            auth_ojama = fin_ctx.authoritative_ojama if fin_ctx else None

        if rc.was_finalized:
            bucket = "A"
        elif rc.close_reason in _B_REASONS:
            bucket = "B"
        else:
            bucket = "C"
        effective_ojama = fin_ojama if fin_ojama is not None else prov_ojama
        bucket_sums[bucket] += effective_ojama
        bucket_counts[bucket] += 1

        dump.append({
            "chain_id": rc.chain_id, "side": rc.side,
            "opened_at_sec": rc.opened_at_sec, "closed_at_sec": rc.closed_at_sec,
            "close_reason": rc.close_reason.name, "was_finalized": rc.was_finalized,
            "growth_observed": rc.growth_observed, "force_cut": rc.force_cut,
            "step_count": rc.step_count,
            "provisional_score": rc.provisional_score,
            "finalized_score": rc.finalized_score,
            "provisional_ojama": prov_ojama, "finalized_ojama": fin_ojama,
            "fire_elapsed_sec": fire_elapsed, "finalize_elapsed_sec": fin_elapsed,
            "authoritative_ojama_at_finalize": auth_ojama,
            "effective_ojama_contribution": effective_ojama,
            "bucket": bucket,
        })

    # ------------------------------
    # 時間的に重なる同一 side の chain_id 組
    # ------------------------------
    overlaps = []
    by_side: dict[str, list[dict]] = {}
    for d in dump:
        by_side.setdefault(d["side"], []).append(d)
    for side, items in by_side.items():
        items_sorted = sorted(items, key=lambda x: x["opened_at_sec"])
        for i in range(len(items_sorted)):
            for j in range(i + 1, len(items_sorted)):
                a, b = items_sorted[i], items_sorted[j]
                if a["closed_at_sec"] >= b["opened_at_sec"]:
                    overlaps.append({
                        "side": side,
                        "chain_id_a": a["chain_id"], "chain_id_b": b["chain_id"],
                        "a_open_close": (a["opened_at_sec"], a["closed_at_sec"]),
                        "b_open_close": (b["opened_at_sec"], b["closed_at_sec"]),
                    })

    report = {
        "video": str(VIDEO),
        "window": {"t0": T0, "t1": T1, "warmup_sec": WARMUP_SEC},
        "diagnostics": dataclasses.asdict(diag),
        "bucket_sums_ojama": bucket_sums,
        "bucket_counts": bucket_counts,
        "total_resolved_chains": len(resolved),
        "overlapping_pairs_same_side": overlaps,
    }
    (OUT_DIR / "diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (OUT_DIR / "chain_dump.json").write_text(
        json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[saved] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    _run()
