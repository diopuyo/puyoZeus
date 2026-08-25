"""測定2: 代表動画 (マスター級・通常対戦・全消し特殊素材ではない) での再測定 (2026-08-25)。

`scripts/_gate3_episode_probe_2026-08-24.py` (実行1回目、全消し素材で不適切と
コーディネーターが判断) の代わりに、`video_51.mp4`
(`data/phase_e_dl_index.tsv` video_idx=51、
【マスター・3ブロック】light vs SAKI 30先、tier確認済み) の
1ラウンドで測り直す。

## 対象区間の選定根拠

粗いスコアリセット走査 (`scripts/_diag_gate3_find_game_boundary_2026-08-25.py`、
`logs/_diag_gate3_find_game_boundary_2026-08-25.log`) で
t=459.00 (light 2743->0, SAKI 14141->0) と t=533.50
(light 55269->0, SAKI 71627->0) にスコアリセットを検出。

実画面フレームで目視確認済み (`data/verify/gate3_episode_repr_2026-08-25/frames/`):
- t=458.00: 前ラウンドの決着 (SAKI が「やった!」、スコア 2743/14141、
  次ラウンドへの WIN 表示は移行前 0-8)
- t=460.00: 新ラウンド開始 (両者 00000000、盤面フェードイン)
- t=531.00: このラウンドの決着 (SAKI が「全消しやった!」、スコア
  55269/71627、消went数 79/96、最大れんさ数 10/10 — 両者とも最大10連鎖を
  経験した実戦的なラウンド)
- t=534.00: 次ラウンド開始 (両者 00000000)

このラウンドの決着が全消しで終わっている点は事実として記録するが、
**動画全体が全消し素材だった前回とは異なる** (通常の30先マスター戦の中で
自然に起きた1ラウンド分の決着)。

**本番コード (`src/` 配下) は読むだけ。変更しない。**
出力: `data/verify/gate3_episode_repr_2026-08-25/diagnostics.json`
      `data/verify/gate3_episode_repr_2026-08-25/chain_dump.json`
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

VIDEO = PROJECT_ROOT / "data/frames/video_51.mp4"
OUT_DIR = PROJECT_ROOT / "data/verify/gate3_episode_repr_2026-08-25"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1ラウンド全体 (目視確認済み、上記 docstring 参照)。
T0: float = 459.0
T1: float = 533.5
WARMUP_SEC: float = 30.0

FORMULA_FLAGS: dict[str, bool] = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_slide_exit_no_min_display=True,
    enable_formula_step_interlude=True,
)

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
    last_auth_total: dict[str, float] = {"1P": 0.0, "2P": 0.0}
    prev_snap = None
    n_observed_frames = 0
    n_chain_observations = 0
    n_settlement_observations = 0

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
                n_chain_observations += 1

        if prev_snap is not None and t_sec >= T0:
            settlement = _diff_settlement(prev_snap, snap, t_sec, game_idx)
            if settlement is not None:
                episode_tracker.observe_settlement(settlement)
                n_settlement_observations += 1
        prev_snap = snap
        if t_sec >= T0:
            n_observed_frames += 1
        fi += 1

    cap.release()
    episode_tracker.finish()
    diag = episode_tracker.diagnostics()

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
            "authoritative_ojama_at_finalize": auth_ojama,
            "effective_ojama_contribution": effective_ojama,
            "bucket": bucket,
        })

    d2_abs = [abs(d) for d in diag.d2.divergences]
    over_tol = sum(1 for v in d2_abs if v > 4.0)

    report = {
        "video": str(VIDEO),
        "video_title": "【マスター・3ブロック】light vs SAKI 30先 (video_idx=51)",
        "window": {"t0": T0, "t1": T1, "warmup_sec": WARMUP_SEC},
        "n_observed_frames": n_observed_frames,
        "n_chain_observations": n_chain_observations,
        "n_settlement_observations": n_settlement_observations,
        "diagnostics": dataclasses.asdict(diag),
        "bucket_sums_ojama": bucket_sums,
        "bucket_counts": bucket_counts,
        "total_resolved_chains": len(resolved),
        "d2_divergence_over_tolerance_count": over_tol,
        "d2_divergence_n": len(d2_abs),
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
