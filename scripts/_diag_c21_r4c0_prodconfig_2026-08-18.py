"""004_c21 r4c0 (1P) が現行8フラグ全群ON本番構成でも同じ結論 (真値=ojama) を
出すかの軽量再現確認 (2026-08-18)。単一動画・単一side・35秒窓のみ処理
(重い全域測定は避ける、148再収集ジョブとのCPU競合回避)。

production_config.recognition_load_default_kwargs() を単一情報源として使う
(手打ちフラグでなく、本番と完全に同じ構成を保証するため)。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from src.production_config import recognition_load_default_kwargs
from src.recognition_pipeline import RecognitionPipeline

VIDEO_PATH = Path.home() / "frames" / "video_c21.mp4"
OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "verify" / "diag_c21_r4c0_2026-08-18" / "prodconfig_trace.json"
)

SIDE = "1P"
TARGET_R, TARGET_C = 4, 0
START_SEC = 2378.0
END_SEC = 2412.0


def main() -> None:
    kwargs = recognition_load_default_kwargs()
    print("[prod flags]", json.dumps(kwargs, ensure_ascii=False))
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        load_next_detector=True,
        temporal_smoothing=1,
        force_in_match=False,
        **kwargs,
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(START_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    t_sec = START_SEC
    prev_conf: int | None = None
    transitions: list[dict] = []

    while t_sec < END_SEC:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        res = pipeline.update(frame_idx, t_sec, frame)
        side_res = res.p1 if SIDE == "1P" else res.p2
        conf_val = (
            int(side_res.confirmed_board.get(TARGET_R, TARGET_C))
            if side_res.confirmed_board is not None else None
        )
        if conf_val is not None and conf_val != prev_conf:
            transitions.append({
                "frame_idx": frame_idx, "t_sec": round(t_sec, 4),
                "prev_conf": prev_conf, "new_conf": conf_val,
                "state": getattr(side_res.state, "name", str(side_res.state)),
            })
            prev_conf = conf_val
        frame_idx += 1
        t_sec = frame_idx / fps

    cap.release()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({"prod_flags": kwargs, "transitions": transitions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] {OUT_PATH}")
    for tr in transitions:
        print(tr)


if __name__ == "__main__":
    main()
