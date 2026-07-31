"""案B検証: A のみ vs A+B で confirmed_board のセル差分を実フレームで比較する
使い捨てスクリプト。既知の誤検出座標 (video_c60 raw row=10, col=0 = x_mark_halo
誤爆) を含む区間を通し、(1,2) 以外のセルで判定が変わるか (=誤検出解消の裏付け)
を確認する。
"""
from __future__ import annotations

import argparse
from collections import Counter

import cv2

from src.recognition_pipeline import RecognitionPipeline
from src.ui_mask import UI_MASK_TARGET_CELLS

TARGET_W, TARGET_H = 1920, 1080


def build_pipeline(enable_b: bool) -> RecognitionPipeline:
    ui_mask_cells = UI_MASK_TARGET_CELLS if enable_b else None
    return RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True,
        enable_chain_tracker=True, temporal_smoothing=1,
        force_in_match=True, ui_mask_cells=ui_mask_cells,
    )


def run(video: str, start_sec: float, n_frames: int, enable_b: bool):
    pipeline = build_pipeline(enable_b)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    boards_1p = []
    boards_2p = []
    cnn_1p = []
    cnn_2p = []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H),
                               interpolation=cv2.INTER_AREA)
        fi = start_frame + i
        result = pipeline.update(fi, fi / fps, frame)
        b1 = result.p1.confirmed_board
        b2 = result.p2.confirmed_board
        boards_1p.append(b1.copy() if b1 is not None else None)
        boards_2p.append(b2.copy() if b2 is not None else None)
        cnn_1p.append(result.p1.cnn_board.copy())
        cnn_2p.append(result.p2.cnn_board.copy())
    cap.release()
    return boards_1p, boards_2p, cnn_1p, cnn_2p


def diff_boards(a_boards, b_boards, side: str, diffs: Counter) -> int:
    n_diff_frames = 0
    for fi, (ba, bb) in enumerate(zip(a_boards, b_boards)):
        if ba is None or bb is None:
            continue
        frame_has_diff = False
        for row in range(13):
            for col in range(6):
                va = int(ba.get(row, col))
                vb = int(bb.get(row, col))
                if va != vb:
                    diffs[(side, row, col)] += 1
                    frame_has_diff = True
        if frame_has_diff:
            n_diff_frames += 1
    return n_diff_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-sec", type=float, default=1490.0)
    ap.add_argument("--frames", type=int, default=400)
    args = ap.parse_args()

    print("Aのみ (ui_mask_cells=None) 実行中...")
    a1, a2, acnn1, acnn2 = run(
        args.video, args.start_sec, args.frames, enable_b=False,
    )
    print("A+B (ui_mask_cells={(1,2)}) 実行中...")
    b1, b2, bcnn1, bcnn2 = run(
        args.video, args.start_sec, args.frames, enable_b=True,
    )

    diffs: Counter = Counter()
    n1 = diff_boards(a1, b1, "1P(confirmed)", diffs)
    n2 = diff_boards(a2, b2, "2P(confirmed)", diffs)
    print(f"\n差分ありフレーム数 (confirmed_board): 1P={n1}/{len(a1)}  2P={n2}/{len(a2)}")

    cnn_diffs: Counter = Counter()
    ncnn1 = diff_boards(acnn1, bcnn1, "1P(cnn_board)", cnn_diffs)
    ncnn2 = diff_boards(acnn2, bcnn2, "2P(cnn_board)", cnn_diffs)
    print(f"差分ありフレーム数 (cnn_board, 平滑化前): 1P={ncnn1}/{len(acnn1)}  "
          f"2P={ncnn2}/{len(acnn2)}")

    if diffs or cnn_diffs:
        print("差分セル内訳 (side, raw_row, col) -> 差分フレーム数:")
        for key, cnt in sorted(
            {**diffs, **cnn_diffs}.items(), key=lambda kv: -kv[1],
        ):
            print(f"  {key}: {cnt}")
    else:
        print("=> セル差分ゼロ (confirmed_board / cnn_board とも完全一致)")


if __name__ == "__main__":
    main()
