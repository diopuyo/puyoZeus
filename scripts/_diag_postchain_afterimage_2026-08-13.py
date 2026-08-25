"""連鎖後残像の再発診断 (read-only計装, 2026-08-13 userデモレビュー指摘 場面4).

## 背景
review_demo_2026-08-12.mp4 の 5試合目 (game_idx=4、source t=370-410、
デモ208-248秒) で「連鎖後に残像が残っている」との指摘。

data/verify/fps_stride_ab_2026-08-12/review_demo_stride2.npz の dedup済み
STABLE スナップショットを事前確認したところ、1P 側で t=398.03 (score=287,
puyo=55) → t=407.77 (score=29137, puyo=9) の間、約9.7秒間 STABLE スナップ
ショットが1件も存在しない (= 巨大連鎖と推定、docs 既知知見
「8連鎖実測14.5秒」と整合しうる長さ)。本スクリプトはこの区間を frame 単位で
計装し、以下を判定する:
  1. state (CHAIN/GRAVITY_SETTLE/EFFECT/STABLE) の遷移タイムライン。
  2. 「盤面が視覚的に静止した (cnn 生盤面のぷよ数が連続フレームで不変)」
     最初の時刻 T_visual_settle と、「confirmed_board が新しい盤面に置き
     換わった (STABLE 復帰)」時刻 T_confirmed_update の差分。
  3. 2026-07-23 の既知バグ (「連鎖後の残像」、認識強化フェーズで反映率
     37%→59%に対処済みのはず) の再発か、新規かを classify する。

## 対象
1P 側のみ (該当巨大連鎖は 1P)。t=393-410 (game_idx=4 season, 前後マージン込み)。

## 制約
- read-only 診断。src/ は一切変更しない。
- WSL venv、シングルプロセス。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_postchain_afterimage_2026-08-13 \
        --video data/frames/review_demo_2026-08-12.mp4 \
        --start-sec 393 --end-sec 410 --decode-from-sec 369 \
        --out logs/diag_postchain_afterimage_2026-08-13.json
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
PROD_TARGET_W: int = 1920
PROD_TARGET_H: int = 1080


def _resize_for_prod(frame: Any) -> Any:
    """collect_boards_lean.py と同じ resize 方式 (縮小=INTER_AREA/拡大=LANCZOS4)."""
    h, w = frame.shape[:2]
    if (h, w) == (PROD_TARGET_H, PROD_TARGET_W):
        return frame
    interp = cv2.INTER_LANCZOS4 if h < PROD_TARGET_H else cv2.INTER_AREA
    return cv2.resize(frame, (PROD_TARGET_W, PROD_TARGET_H), interpolation=interp)


def run(video_path: Path, start_sec: float, end_sec: float,
        decode_from_sec: float, out_path: Path) -> None:
    """区間を単一 pipeline (production 相当設定) で走査し JSON に書き出す."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[diag] cannot open: {video_path}", file=sys.stderr)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    decode_from_frame = int(decode_from_sec * fps)
    end_frame = int(end_sec * fps)
    if decode_from_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, decode_from_frame)

    # PRODUCTION_STABLE_FRAME_COUNT=3 (collect_boards_lean.py 等の本番一致値)。
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )

    records: list[dict[str, Any]] = []
    prev_cnn_count_1p: int | None = None
    n_processed = 0
    for frame_idx in range(decode_from_frame, end_frame):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t_sec = frame_idx / fps
        frame_prod = _resize_for_prod(frame)
        r = pipe.update(frame_idx, t_sec, frame_prod)
        cnn_count = int(r.p1.cnn_board.count_puyos())
        confirmed_count = (
            int(r.p1.confirmed_board.count_puyos())
            if r.p1.confirmed_board is not None else None
        )
        if t_sec >= start_sec:
            records.append({
                "frame_idx": frame_idx,
                "t_sec": t_sec,
                "state": r.p1.state.value,
                "score": r.p1.score,
                "chain_event": r.p1.chain_event is not None,
                "cnn_puyo_count": cnn_count,
                "cnn_puyo_diff_prev": (
                    None if prev_cnn_count_1p is None
                    else cnn_count - prev_cnn_count_1p
                ),
                "confirmed_is_none": r.p1.confirmed_board is None,
                "confirmed_puyo_count": confirmed_count,
                "board_provenance": r.p1.board_provenance,
            })
        prev_cnn_count_1p = cnn_count
        n_processed += 1
        if n_processed % 600 == 0:
            print(f"[diag] processed {n_processed} frames (t={t_sec:.2f}s)",
                  file=sys.stderr)
    cap.release()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"video": str(video_path), "fps": fps, "records": records},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[diag] wrote -> {out_path}")


def main() -> int:
    """CLI エントリポイント."""
    parser = argparse.ArgumentParser(description="連鎖後残像の再発診断 (1P固定)")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=393.0)
    parser.add_argument("--end-sec", type=float, default=410.0)
    parser.add_argument("--decode-from-sec", type=float, default=369.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.video, args.start_sec, args.end_sec, args.decode_from_sec, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
