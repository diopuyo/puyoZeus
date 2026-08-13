"""OJAMA_FALL誤分類 修正 (案2+案4-lite+案3) の実データ効果測定 (2026-08-13).

src/_diag_ojama_fall_misclass_2026-08-13.py (根因調査本体) は書き換えず、
「full_prod」設定に新フラグ (enable_ojama_fall_placement_override /
enable_ojama_fall_entry_hardening / enable_chain_gate_raw_fallback) を
ON にした「full_prod_fix」設定を追加して同一フレームに並走させる。

対象:
  - 場面1 (source t=188-190、stride=1): OJAMA_FALL⇔STABLE 振動の解消確認。
  - 場面2 (source t=193-198、stride=2): stride 下で chain_event が瞬間欠落
    した隙に OJAMA_FALL が CHAIN を奪う割り込みの解消確認 (coordinator 追補)。
  - 場面3 (source t=311-335、stride=1): 設置ブロック (阻害) の解消件数確認。

計測方法 (read-only、src/ は一切変更しない):
  各 config・各 side の state 系列から、
    (a) OJAMA_FALL ⇔ STABLE の遷移回数 (振動指標)
    (b) 「直前 frame が CHAIN で当該 frame が OJAMA_FALL」の回数 (CHAIN 割込み指標)
  を frame 単位ログから直接数える。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._verify_ojama_fall_fix_2026-08-13 \
        --video data/frames/review_demo_2026-08-12.mp4 \
        --decode-from-sec 170 --start-sec 185 --end-sec 200 --stride 1 \
        --out logs/verify_ojama_fall_fix_scene1_2026-08-13.json
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
CONFIGS = ("full_prod", "full_prod_fix")

PROD_TARGET_W: int = 1920
PROD_TARGET_H: int = 1080


def _resize_for_prod(frame: Any) -> Any:
    """collect_boards_lean.py と同じ resize 方式 (縮小=INTER_AREA/拡大=LANCZOS4)."""
    h, w = frame.shape[:2]
    if (h, w) == (PROD_TARGET_H, PROD_TARGET_W):
        return frame
    interp = cv2.INTER_LANCZOS4 if h < PROD_TARGET_H else cv2.INTER_AREA
    return cv2.resize(frame, (PROD_TARGET_W, PROD_TARGET_H), interpolation=interp)


def _build_pipeline(config: str) -> RecognitionPipeline:
    """config 名に応じた RecognitionPipeline を構築する.

    full_prod: collect_boards_lean.py 相当の本番構成 (_diag_ojama_fall_
        misclass_2026-08-13.py の full_prod と同一)。
    full_prod_fix: 上記 + 案2/案4-lite/案3 の新フラグ全て ON。
    """
    base_kwargs: dict[str, Any] = dict(
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
    )
    if config == "full_prod":
        return RecognitionPipeline.load_default(**base_kwargs)
    if config == "full_prod_fix":
        return RecognitionPipeline.load_default(
            **base_kwargs,
            enable_ojama_fall_placement_override=True,
            enable_ojama_fall_entry_hardening=True,
            enable_chain_gate_raw_fallback=True,
        )
    raise ValueError(f"unknown config: {config}")


def run(video_path: Path, start_sec: float, end_sec: float, out_path: Path,
        decode_from_sec: float = 0.0, stride: int = 1) -> None:
    """区間を 2 設定 (fix ON/OFF) で並走させ、frame 単位ログ+要約を書き出す."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[verify] cannot open: {video_path}", file=sys.stderr)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    decode_from_frame = int(decode_from_sec * fps)
    end_frame = int(end_sec * fps)
    if decode_from_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, decode_from_frame)

    pipes: dict[str, RecognitionPipeline] = {c: _build_pipeline(c) for c in CONFIGS}
    frame_records: dict[str, list[dict[str, Any]]] = {c: [] for c in CONFIGS}

    n_processed = 0
    for frame_idx in range(decode_from_frame, end_frame):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (frame_idx - decode_from_frame) % stride != 0:
            continue
        t_sec = frame_idx / fps
        for cfg, pipe in pipes.items():
            frame_in = _resize_for_prod(frame)
            r = pipe.update(frame_idx, t_sec, frame_in.copy())
            if t_sec < start_sec:
                continue
            frame_records[cfg].append({
                "frame_idx": frame_idx,
                "t_sec": t_sec,
                "1P_state": r.p1.state.value,
                "2P_state": r.p2.state.value,
            })
        n_processed += 1
        if n_processed % 600 == 0:
            print(f"[verify] processed {n_processed} frames (t={t_sec:.2f}s)",
                  file=sys.stderr)
    cap.release()

    summary = {cfg: _summarize(frame_records[cfg]) for cfg in CONFIGS}
    result = {
        "video": str(video_path), "fps": fps,
        "start_sec": start_sec, "end_sec": end_sec, "stride": stride,
        "n_frames_processed": n_processed,
        "summary": summary,
        "frame_records": frame_records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"[verify] wrote -> {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """side 別に (a) OJAMA_FALL⇔STABLE 遷移回数 (b) CHAIN→OJAMA_FALL 割込み回数
    (c) OJAMA_FALL 滞在 frame 数 を集計する.
    """
    out: dict[str, Any] = {}
    for side in SIDES:
        key = f"{side}_state"
        states = [rec[key] for rec in records]
        osc = 0
        chain_interrupt = 0
        ojama_frames = sum(1 for s in states if s == "ojama_fall")
        for i in range(1, len(states)):
            prev, cur = states[i - 1], states[i]
            if {prev, cur} == {"ojama_fall", "stable"} and prev != cur:
                osc += 1
            if prev == "chain" and cur == "ojama_fall":
                chain_interrupt += 1
        out[side] = {
            "n_frames": len(states),
            "ojama_stable_transitions": osc,
            "chain_to_ojama_interrupts": chain_interrupt,
            "ojama_fall_frame_count": ojama_frames,
        }
    return out


def main() -> int:
    """CLI エントリポイント."""
    parser = argparse.ArgumentParser(
        description="OJAMA_FALL誤分類修正 (案2+案4-lite+案3) の効果測定",
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=150.0)
    parser.add_argument("--end-sec", type=float, default=205.0)
    parser.add_argument("--decode-from-sec", type=float, default=0.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.video, args.start_sec, args.end_sec, args.out, args.decode_from_sec,
        args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
