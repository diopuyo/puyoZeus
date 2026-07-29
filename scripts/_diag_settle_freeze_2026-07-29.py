"""指摘1/2 検証用診断: settled ゲートの凍結挙動と kill_override 発火有無をログ出力する。

visualize_advantage_overlay.generate() と同一ロジックを流用するが、動画は書き出さず
毎フレーム診断ログ(state/score/kill関連値)を印字するだけの使い捨てスクリプト。
CPUは動画レンダ1本のみ使用中のため nice -n 19 で実行する想定。

使い方:
    nice -n 19 python -m scripts._diag_settle_freeze_2026-07-29 \
        --video data/frames/video_c56.mp4 --start-sec 320 --end-sec 362 --label c56_g3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama  # noqa: E402
from scripts.visualize_advantage_overlay import (  # noqa: E402
    EarlyFireTracker, HeavyAdvCache, PressureTracker, RealtimeForecastTracker,
    ScoreLeadTracker, _detect_score_reset, _train_model, adv_to_winprob, board_room,
    kill_override, W_PRESSURE, W_FORECAST, W_MODEL, W_THREAT,
)
import src.indicators_v2 as iv  # noqa: E402

DEFAULT_FPS = 30.0


def run(video: Path, start_sec: float, end_sec: float, label: str) -> None:
    """指定区間を診断ログ付きで処理する(動画書き出しなし)。"""
    model = _train_model(None)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    warmup = 30.0
    proc_frame = int(max(0.0, start_sec - warmup) * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, proc_frame)
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
        enable_landing_observed_color=True, enable_drift_resync_match_start_guard=True,
        enable_drift_resync_hsv_gate=True, enable_match_start_full_clear=True,
        enable_recovery_counter_carryover=True, enable_cnn_flicker_hsv_fallback=True,
        enable_initial_confirm_vote=True)
    import re
    m = re.search(r"(v\d+|video_\d+)", video.name)
    if m and hasattr(pipe, "set_video_id"):
        pipe.set_video_id(m.group(1))
    tracker = OjamaAccountingTracker(); tracker.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    ps1 = ps2 = BoardState.MENU
    b1 = b2 = None
    ptracker = PressureTracker()
    fctracker = RealtimeForecastTracker()
    svtracker = ScoreLeadTracker()
    hcache = HeavyAdvCache(model)
    efire = EarlyFireTracker()  # (早期発火) A/B比較用: 新経路の disp_adv も同時出力
    prev_score1 = prev_score2 = None
    adv_ema, p1_last = 0.0, 0.5
    last_settled = False
    last_disp_adv = 0.0
    for fi in range(proc_frame, end_frame):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (720, 1280):
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)
        if _detect_score_reset(r.p1.score, r.p2.score, prev_score1, prev_score2):
            print(f"[{label}] t={t:.2f}s RESET")
            b1 = b2 = None
            adv_ema, p1_last = 0.0, 0.5
            tracker = OjamaAccountingTracker(); tracker.reset()
            tp1, tp2 = _SideTracker(), _SideTracker()
            ptracker, fctracker, svtracker, hcache = (
                PressureTracker(), RealtimeForecastTracker(), ScoreLeadTracker(),
                HeavyAdvCache(model))
        prev_score1, prev_score2 = r.p1.score, r.p2.score
        if r.p1.state == BoardState.STABLE and r.p1.confirmed_board is not None:
            b1 = r.p1.confirmed_board
        if r.p2.state == BoardState.STABLE and r.p2.confirmed_board is not None:
            b2 = r.p2.confirmed_board
        snap = _drive_ojama(tracker, r.p1, r.p2, ps1, ps2, t,
                            tracker_p1=tp1, tracker_p2=tp2, pipeline=pipe)
        ps1, ps2 = r.p1.state, r.p2.state
        # (早期発火) settled ゲートの外側で毎フレーム更新 (本修正の要)
        efire.update(r.p1.chain_event, r.p2.chain_event, b2, b1, tracker._elapsed(t))
        settled = r.p1.state == BoardState.STABLE and r.p2.state == BoardState.STABLE
        if t < start_sec:
            continue
        if settled != last_settled:
            print(f"[{label}] t={t:.2f}s settled {last_settled}->{settled} "
                  f"(1P={r.p1.state.name} 2P={r.p2.state.name})")
            last_settled = settled
        if b1 is not None and b2 is not None and settled:
            model_adv, threat, drivers, ukey1, ukey2, sat1, sat2 = hcache.update(
                b1, b2, snap, r.p1, r.p2, tracker._elapsed(t))
            pres = ptracker.update(iv.board_ojama_count(b1).raw,
                                   iv.board_ojama_count(b2).raw)
            fc = fctracker.update(r.p1.score, r.p2.score,
                                  pipe.tsumo_count("1P"), pipe.tsumo_count("2P"))
            sl_bias = svtracker.update(r.p1.score, r.p2.score)
            adv_pre_kill = max(-100.0, min(100.0,
                (W_PRESSURE * pres + W_FORECAST * fc + W_MODEL * model_adv
                 + W_THREAT * threat) + max(-15.0, min(15.0, sl_bias))))
            room1, room2 = board_room(b1), board_room(b2)
            adv = kill_override(adv_pre_kill, fctracker.inc1, fctracker.inc2, room1, room2)
            killed = abs(adv - adv_pre_kill) > 0.5
            p1 = adv_to_winprob(adv)
            adv_ema = 0.25 * adv + 0.75 * adv_ema
            p1_last = 0.25 * p1 + 0.75 * p1_last
            efire.on_settled()  # 確定計算が入ったので速報バイアスをクリア (二重計上防止)
            print(f"[{label}] t={t:.2f}s SETTLED score1={r.p1.score} score2={r.p2.score} "
                  f"adv_pre_kill={adv_pre_kill:+.1f} adv_post_kill={adv:+.1f} "
                  f"KILL={'YES' if killed else 'no'} inc1={fctracker.inc1:.1f} "
                  f"inc2={fctracker.inc2:.1f} room1={room1} room2={room2} "
                  f"adv_ema={adv_ema:+.1f} p1={p1_last:.2f}")
        # (早期発火) 表示直前にのみ bias を加算(旧経路 adv_ema はそのまま)。
        # 値が前回ログから大きく変化した時だけ印字(冗長ログ抑制)。
        disp_adv = max(-100.0, min(100.0, adv_ema + efire.bias))
        if abs(disp_adv - last_disp_adv) > 3.0:
            print(f"[{label}] t={t:.2f}s EARLYFIRE disp_adv={disp_adv:+.1f} "
                  f"(adv_ema={adv_ema:+.1f} bias={efire.bias:+.1f}) settled={settled}")
            last_disp_adv = disp_adv
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-sec", type=float, required=True)
    ap.add_argument("--end-sec", type=float, required=True)
    ap.add_argument("--label", default="diag")
    a = ap.parse_args()
    run(Path(a.video), a.start_sec, a.end_sec, a.label)


if __name__ == "__main__":
    main()
