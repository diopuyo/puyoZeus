"""Phase U: 本番統合 (HybridClassifier + use_match_state) 動画レンダ。

動画フレームから 0.2 秒間隔で認識し、各セルの認識結果を文字オーバーレイで
表示する。レビュー用の簡易版。

利用例:
    python scripts/phase_u_render.py \\
        data/verify/review_videos/clip_v01_m34.mp4 \\
        data/verify/review_videos/phase_u_v01_m34.mp4 \\
        --cnn-model models/cnn_phase_u_v3.pt --interval 0.2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    VISIBLE_ROWS,
)
from src.hybrid_classifier import HybridClassifier
from src.image_reader import (
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    ImageReader,
)

COLOR_BGR: dict[int, tuple[int, int, int]] = {
    COLOR_EMPTY: (40, 40, 40),
    COLOR_RED: (40, 40, 200),
    COLOR_BLUE: (200, 80, 40),
    COLOR_GREEN: (40, 180, 40),
    COLOR_YELLOW: (40, 200, 220),
    COLOR_PURPLE: (180, 40, 180),
    COLOR_OJAMA: (170, 170, 170),
    COLOR_UNKNOWN: (80, 80, 120),
}

COLOR_LABEL: dict[int, str] = {
    COLOR_EMPTY: "", COLOR_RED: "R", COLOR_BLUE: "B",
    COLOR_GREEN: "G", COLOR_YELLOW: "Y", COLOR_PURPLE: "P",
    COLOR_OJAMA: "O", COLOR_UNKNOWN: "?",
}


def overlay_board(
    frame: np.ndarray, board, region,
) -> None:
    """フレームに各セルの認識色を文字+半透明四角で描画 (in-place)。"""
    for vrow in range(VISIBLE_ROWS):
        for col in range(BOARD_COLS):
            row = vrow + HIDDEN_ROWS
            color = int(board.get(row, col))
            if color == COLOR_EMPTY:
                continue
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            bgr = COLOR_BGR.get(color, (60, 60, 60))
            # 半透明枠
            sub = frame[y1:y2, x1:x2].astype(np.float32)
            overlay = np.full_like(sub, bgr, dtype=np.float32)
            blended = (sub * 0.6 + overlay * 0.4).astype(np.uint8)
            frame[y1:y2, x1:x2] = blended
            # ラベル
            label = COLOR_LABEL.get(color, "?")
            cx = (x1 + x2) // 2 - 8
            cy = (y1 + y2) // 2 + 6
            cv2.putText(
                frame, label, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                2, cv2.LINE_AA,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("output_video")
    parser.add_argument("--cnn-model", default="models/cnn_phase_u_v7.pt")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument(
        "--bg-fp-time", type=float, default=-1.0,
        help="試合開始秒。指定すると周辺フレームから robust BG FP を取得し ImageReader へ設定 (-1 で無効)",
    )
    parser.add_argument(
        "--start-sec", type=float, default=0.0,
        help="動画の何秒からレンダ開始するか (デフォルト 0)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"video open failed: {args.input_video}")
        return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"input: fps={src_fps:.2f} frames={n_frames}")

    # CNN ロード
    classifier_obj = None
    if Path(args.cnn_model).exists():
        try:
            from src.patch_classifier import CnnPatchClassifier
            import torch
            cnn = CnnPatchClassifier()
            state = torch.load(
                args.cnn_model, map_location="cpu", weights_only=True,
            )
            cnn._model.load_state_dict(state)
            cnn._model.eval()
            classifier_obj = HybridClassifier(cnn_classifier=cnn)
            print(f"using HybridClassifier with: {args.cnn_model}")
        except Exception as e:
            print(f"failed to load CNN ({e}), HSV only")

    reader = ImageReader(
        classifier=classifier_obj,
        use_match_state=True,
        use_ui_mask=True,
        use_telop_mask=True,  # V3.1: テロップ被覆セルを UNKNOWN 化
    )

    # 連鎖/相殺アニメ検出フィルタ (P1/P2 別インスタンス)
    from src.animation_filter import AnimationFilter
    af_p1 = AnimationFilter()
    af_p2 = AnimationFilter()

    # V2.4: EnhancedBoardTracker (P1/P2 別) - NextLink + Connectivity + stateful
    from src.enhanced_board_tracker import EnhancedBoardTracker
    tracker_p1 = EnhancedBoardTracker()
    tracker_p2 = EnhancedBoardTracker()

    # V3.2: 試合終了告知検出 (やった!/ばたんきゅー) + ロックダウン
    from src.match_end_detector import MatchEndDetector
    match_end_detector = MatchEndDetector.load_default()

    # NextDetector: V2.1 で必要な next_pair (1P/2P 両側) を取得
    from src.next_detector import NextDetector
    try:
        next_detector: "NextDetector | None" = NextDetector.load_default()
        print("NextDetector loaded")
    except Exception as e:
        print(f"NextDetector load failed ({e}), V2.1 補正は無効")
        next_detector = None
    p1_roi = (
        DEFAULT_P1_REGION.x, DEFAULT_P1_REGION.y,
        DEFAULT_P1_REGION.width, DEFAULT_P1_REGION.height,
    )
    p2_roi = (
        DEFAULT_P2_REGION.x, DEFAULT_P2_REGION.y,
        DEFAULT_P2_REGION.width, DEFAULT_P2_REGION.height,
    )

    # 中央テロップ検出 (検出されたら被覆領域を半透明オーバーレイで明示)
    from src.telop_detector import (
        TelopDetector,
        SEARCH_X as TELOP_X, SEARCH_Y as TELOP_Y,
        SEARCH_W as TELOP_W, SEARCH_H as TELOP_H,
    )
    telop_detector = TelopDetector.load_default()

    # 試合開始時の robust BG FP 取得 (任意)
    if args.bg_fp_time >= 0:
        from src.background_fingerprint import capture_pair_robust
        bg_frames = []
        for offset in (-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5):
            t = max(0.0, args.bg_fp_time + offset)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fb = cap.read()
            if not ok or fb is None:
                continue
            if fb.shape[:2] != (1080, 1920):
                fb = cv2.resize(fb, (1920, 1080), interpolation=cv2.INTER_AREA)
            bg_frames.append(fb)
        if bg_frames:
            p1_t = (
                DEFAULT_P1_REGION.x, DEFAULT_P1_REGION.y,
                DEFAULT_P1_REGION.width, DEFAULT_P1_REGION.height,
            )
            p2_t = (
                DEFAULT_P2_REGION.x, DEFAULT_P2_REGION.y,
                DEFAULT_P2_REGION.width, DEFAULT_P2_REGION.height,
            )
            fp1, fp2 = capture_pair_robust(bg_frames, p1_t, p2_t)
            reader.set_background_fingerprints(fp1, fp2)
            print(f"BG FP set from {len(bg_frames)} frames at t={args.bg_fp_time}s")

    # 開始位置をシーク (start-sec 指定 / BG FP 取得後の位置リセット兼ねる)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        args.output_video, fourcc, src_fps, (1920, 1080),
    )
    if not writer.isOpened():
        print("output writer failed")
        return 1

    interval_frames = max(1, int(src_fps * args.interval))
    max_frames = (
        n_frames if args.max_seconds <= 0
        else int(args.max_seconds * src_fps)
    )
    last_b1, last_b2 = None, None
    telop_visible = False
    match_end_locked = False
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx >= max_frames:
            break
        if frame.shape[:2] != (1080, 1920):
            frame_1080 = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        else:
            frame_1080 = frame.copy()
        # 現在の動画時刻 (ロックダウン管理用)
        t_sec = (args.start_sec + frame_idx / src_fps)
        if frame_idx % interval_frames == 0 or last_b1 is None:
            # V3.2: 試合終了告知検出 + ロックダウン更新
            match_end_locked = match_end_detector.update(frame_1080, t_sec)
            # アニメ判定 (P1/P2 別々)
            anim_p1 = af_p1.is_animation(frame_1080, p1_roi).is_animation
            anim_p2 = af_p2.is_animation(frame_1080, p2_roi).is_animation
            # ロックダウン中は新観測を取らず前盤面保持
            if match_end_locked:
                pass
            else:
                try:
                    b1_obs, b2_obs = reader.read_both_boards(frame_1080)
                    # NextDetector で next_pair 取得 (V2.1 で使用)
                    next_p1, next_p2 = None, None
                    if next_detector is not None:
                        try:
                            both = next_detector.detect_both(frame_1080)
                            next_p1 = both.p1.next_pair
                            next_p2 = both.p2.next_pair
                        except Exception:
                            pass
                    # V2.4: EnhancedBoardTracker で時系列フィルタ
                    # アニメ中は更新スキップ (前盤面保持)
                    if not anim_p1 or last_b1 is None:
                        last_b1 = tracker_p1.update(b1_obs, next_pair=next_p1)
                    if not anim_p2 or last_b2 is None:
                        last_b2 = tracker_p2.update(b2_obs, next_pair=next_p2)
                except Exception:
                    pass
        out_frame = frame_1080.copy()
        if last_b1 is not None:
            overlay_board(out_frame, last_b1, DEFAULT_P1_REGION)
        if last_b2 is not None:
            overlay_board(out_frame, last_b2, DEFAULT_P2_REGION)
        # テロップ被覆を視覚化 (検出時のみ、ピンク半透明枠)
        if frame_idx % interval_frames == 0:
            telop_visible = telop_detector.is_visible(frame_1080)
        if telop_visible:
            sub = out_frame[TELOP_Y:TELOP_Y + TELOP_H,
                            TELOP_X:TELOP_X + TELOP_W].astype(np.float32)
            tint = np.full_like(sub, (180, 80, 200), dtype=np.float32)
            out_frame[TELOP_Y:TELOP_Y + TELOP_H,
                      TELOP_X:TELOP_X + TELOP_W] = (
                sub * 0.85 + tint * 0.15
            ).astype(np.uint8)
            cv2.putText(
                out_frame, "TELOP",
                (TELOP_X + 10, TELOP_Y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 230),
                2, cv2.LINE_AA,
            )
        # 試合終了ロックダウン中であることを画面下部に表示
        if match_end_locked:
            cv2.putText(
                out_frame, "MATCH END LOCKDOWN",
                (700, 1050),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255),
                2, cv2.LINE_AA,
            )
        writer.write(out_frame)
        if frame_idx % 600 == 0:
            print(f"  progress {frame_idx}/{max_frames}")
        frame_idx += 1
    writer.release()
    cap.release()
    print(f"\n[OK] {to_windows_path(args.output_video)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
