"""場面1 (docs/DEMO_REVIEW_2026-08-13.md #1) の振動再発チェック (2026-08-15、

read-only計装, src/ は一切変更しない)。

背景: 場面1 (review_demo_2026-08-12.mp4, source t≈188-189、デモ26-27秒) は
OJAMA_FALL⇔STABLE の 0.15-0.3秒周期振動×15回が実測された区間。 案2
(enable_ojama_fall_placement_override) 導入でこの振動は解消したが、 evidence
一発判定の欠陥 (own_score_delta 単一フレーム即決) が別の実害 (46セル新規
劣化) を持ち込んだため 2026-08-15 に修正した。 本スクリプトは修正版で
同じ区間を再走行し、 (a) 振動が再発していないこと (b) 元の改善
(OJAMA_FALL 張り付き解消) が維持されていることを確認する。

`scripts/_diag_ojama_fall_misclass_2026-08-13.py` と同じ full_prod 構成
(collect_boards_lean.py と同一の RECOGNITION_ADOPTED 相当フラグ) をベースに、
enable_ojama_fall_placement_override=True を追加した構成のみを走らせる
(旧診断は "asis_demo"/"full_prod" の2構成比較が目的だったが、 本スクリプトは
修正版1構成の振動有無チェックが目的のため単純化する)。

Usage:
    PYTHONPATH=. ./venv/bin/python -m \
        scripts._diag_scene1_oscillation_recheck_2026-08-15 \
        --video data/frames/review_demo_2026-08-12.mp4 \
        --start-sec 180 --end-sec 200 --stride 2 \
        --out logs/diag_scene1_oscillation_recheck_2026-08-15.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

DEFAULT_FPS: float = 30.0
SIDES = ("1P", "2P")

# collect_boards_lean.py と同じ resize 方式 (2026-08-13 診断で確認済み:
# resize せず burst guard 系を ON にすると空パッチで cv2.cvtColor がクラッシュ)。
PROD_TARGET_W: int = 1920
PROD_TARGET_H: int = 1080


def _resize_for_prod(frame: Any) -> Any:
    h, w = frame.shape[:2]
    if (h, w) == (PROD_TARGET_H, PROD_TARGET_W):
        return frame
    interp = cv2.INTER_LANCZOS4 if h < PROD_TARGET_H else cv2.INTER_AREA
    return cv2.resize(frame, (PROD_TARGET_W, PROD_TARGET_H), interpolation=interp)


def _build_pipeline() -> RecognitionPipeline:
    """full_prod 相当 + 修正版 placement_override ON の構成."""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
        enable_effect_gate=True,
        enable_burst_guard_v2=True,
        burst_gate_open_threshold=0.954,
        enable_hidden_row_burst_guard=True,
        enable_transition_merge_guard=True,
        enable_match_transition_debounce=True,
        enable_ojama_fall_placement_override=True,
    )


def run(
    video_path: Path, start_sec: float, end_sec: float, out_path: Path,
    decode_from_sec: float, stride: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[diag] cannot open: {video_path}", file=sys.stderr)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    decode_from_frame = int(decode_from_sec * fps)
    end_frame = int(end_sec * fps)
    if decode_from_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, decode_from_frame)

    pipe = _build_pipeline()
    records: list[dict[str, Any]] = []

    n_processed = 0
    for frame_idx in range(decode_from_frame, end_frame):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (frame_idx - decode_from_frame) % stride != 0:
            continue
        t_sec = frame_idx / fps
        frame_in = _resize_for_prod(frame)
        r = pipe.update(frame_idx, t_sec, frame_in.copy())
        if t_sec >= start_sec:
            records.append({
                "frame_idx": frame_idx,
                "t_sec": t_sec,
                "1P_state": r.p1.state.value,
                "2P_state": r.p2.state.value,
                "1P_score": r.p1.score,
                "2P_score": r.p2.score,
            })
        n_processed += 1
        if n_processed % 600 == 0:
            print(f"[diag] processed {n_processed} frames (t={t_sec:.2f}s)",
                  file=sys.stderr)
    cap.release()

    # 振動検出: 同側の state が OJAMA_FALL <-> 他state を短時間で往復した回数。
    osc_events: dict[str, list[dict[str, Any]]] = {"1P": [], "2P": []}
    for side in SIDES:
        key = f"{side}_state"
        prev_state, prev_t = None, None
        for rec in records:
            st, t = rec[key], rec["t_sec"]
            if prev_state is not None and prev_state != st:
                dt = t - prev_t
                if {prev_state, st} & {"OJAMA_FALL"} and dt < 0.35:
                    osc_events[side].append({
                        "t_sec": t, "from": prev_state, "to": st, "dt": dt,
                    })
            prev_state, prev_t = st, t

    result = {
        "video": str(video_path), "fps": fps,
        "start_sec": start_sec, "end_sec": end_sec, "stride": stride,
        "n_frames_processed": n_processed,
        "records": records,
        "oscillation_transitions_lt_0.35s": osc_events,
        "oscillation_count": {s: len(v) for s, v in osc_events.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"[diag] wrote -> {out_path}")
    print(f"[diag] oscillation_count = {result['oscillation_count']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="場面1振動再発チェック (修正版)")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=180.0)
    parser.add_argument("--end-sec", type=float, default=200.0)
    parser.add_argument("--decode-from-sec", type=float, default=0.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.video, args.start_sec, args.end_sec, args.out,
        args.decode_from_sec, args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
