"""認識+物理推論結果を盤面に重畳する可視化動画を生成する.

各セルに認識色 (赤/青/緑/黄/紫/お/空/?) を文字で描画し、
盤面外枠に state machine の現在状態 (STABLE/CHAIN/TSUMO_FALL/...) を描画する.

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \\
        --video data/evaluation_videos/v28_clip60s.mp4 \\
        --output data/evaluation_videos/v28_recognition_viz.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402

init_console()

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
    COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN, COLOR_YELLOW, HIDDEN_ROWS, Board,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# ============================
# 描画定数
# ============================
# 盤面 ROI (calibration_video01.json より)
P1_ROI_X = 282
P1_ROI_Y = 160
P2_ROI_X = 1258
P2_ROI_Y = 160
ROI_W = 384
ROI_H = 720
N_VISIBLE_ROWS = 12
CELL_W = ROI_W // BOARD_COLS  # 64 px
CELL_H = ROI_H // N_VISIBLE_ROWS  # 60 px

# 色記号 (コンパクト表示)
COLOR_SYMBOLS = {
    COLOR_EMPTY: "",
    COLOR_RED: "R",
    COLOR_BLUE: "B",
    COLOR_GREEN: "G",
    COLOR_YELLOW: "Y",
    COLOR_PURPLE: "P",
    COLOR_OJAMA: "O",
    COLOR_UNKNOWN: "?",
}
# BGR 色 (cv2)
COLOR_BGR = {
    COLOR_EMPTY: (60, 60, 60),
    COLOR_RED: (50, 50, 240),
    COLOR_BLUE: (240, 100, 50),
    COLOR_GREEN: (50, 220, 50),
    COLOR_YELLOW: (50, 220, 240),
    COLOR_PURPLE: (200, 50, 200),
    COLOR_OJAMA: (180, 180, 180),
    COLOR_UNKNOWN: (255, 255, 255),
}

# State machine state ごとの枠色
STATE_COLOR = {
    BoardState.STABLE: (0, 255, 0),         # green = OK
    BoardState.TSUMO_FALL: (0, 200, 255),   # orange
    BoardState.CHAIN: (200, 100, 255),      # pink/purple
    BoardState.OJAMA_FALL: (255, 200, 0),   # cyan
    BoardState.MENU: (128, 128, 128),       # gray
    BoardState.EFFECT: (255, 0, 255),       # magenta (全消し等)
}

# 描画フォント
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE_CELL = 0.7
FONT_SCALE_STATE = 0.9
FONT_THICKNESS = 2

# サンプリング間隔 (秒)
DEFAULT_SAMPLE_INTERVAL = 0.033  # 30 fps 認識 (= cycle 71p 2026-05-13 ユーザー要望)


def draw_cell_overlay(
    frame: np.ndarray, board: Board, roi_x: int, roi_y: int,
) -> None:
    """盤面 1 つに対し、各 cell の色 symbol を重畳する.

    可視 12 行のみ描画 (隠し段 row 0 は省略)。
    文字色は常に白、黒太縁で puyo 背景と同色化を回避。
    """
    if board is None:
        return
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            color = int(board.get(row, col))
            symbol = COLOR_SYMBOLS.get(color, "?")
            if not symbol:
                continue  # EMPTY は描画しない
            # cell 中心座標 (visible row index = row - HIDDEN_ROWS)
            display_row = row - HIDDEN_ROWS
            cx = roi_x + col * CELL_W + CELL_W // 2
            cy = roi_y + display_row * CELL_H + CELL_H // 2
            # 文字サイズ調整
            (tw, th), _ = cv2.getTextSize(
                symbol, FONT, FONT_SCALE_CELL, FONT_THICKNESS,
            )
            tx = cx - tw // 2
            ty = cy + th // 2
            # 黒太縁 (puyo 背景と同色化を回避するため)
            cv2.putText(
                frame, symbol, (tx, ty), FONT,
                FONT_SCALE_CELL, (0, 0, 0), FONT_THICKNESS + 4, cv2.LINE_AA,
            )
            # 白文字 (常に視認性確保)
            cv2.putText(
                frame, symbol, (tx, ty), FONT,
                FONT_SCALE_CELL, (255, 255, 255),
                FONT_THICKNESS, cv2.LINE_AA,
            )


def draw_state_label(
    frame: np.ndarray, state: BoardState, roi_x: int, roi_y: int,
    score: int = 0, label_prefix: str = "",
) -> None:
    """ROI 上方に state ラベルを描画する."""
    color = STATE_COLOR.get(state, (255, 255, 255))
    text = f"{label_prefix}{state.value}"
    if score > 0:
        text += f" score={score}"
    # 影
    cv2.putText(
        frame, text, (roi_x + 4, roi_y - 12), FONT,
        FONT_SCALE_STATE, (0, 0, 0), FONT_THICKNESS + 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (roi_x + 3, roi_y - 13), FONT,
        FONT_SCALE_STATE, color, FONT_THICKNESS, cv2.LINE_AA,
    )
    # ROI 枠
    cv2.rectangle(
        frame, (roi_x, roi_y), (roi_x + ROI_W, roi_y + ROI_H),
        color, 2,
    )


def draw_global_info(
    frame: np.ndarray, frame_idx: int, t_sec: float,
    p1_state: BoardState, p2_state: BoardState,
) -> None:
    """画面上部に時刻 + 状態を描画."""
    text = f"frame={frame_idx} t={t_sec:.2f}s 1P={p1_state.value} 2P={p2_state.value}"
    cv2.putText(
        frame, text, (20, 30), FONT, 0.8,
        (0, 0, 0), 5, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (20, 30), FONT, 0.8,
        (255, 255, 255), 2, cv2.LINE_AA,
    )


_HSV_DB_ROOT = Path("data/per_video_hsv_ranges")
_HSV_MERGED_DEFAULT = _HSV_DB_ROOT / "_merged_default.json"

# 2範囲以上で定義されている色 → per_video inject 後に DEFAULT の補完範囲を保証する
# (赤は H=0-13 と H=166-180 の循環2範囲。per_video は高側のみ学習しがちなので
#  低側 H=0-13 が失われると赤を系統的に miss する)
_CIRCULAR_GUARD_COLORS: tuple[int, ...] = (COLOR_RED,)


def _ensure_circular_ranges_guard(classifier: object) -> None:
    """per_video inject 後に DEFAULT の循環補完範囲が欠落していないか確認し補完する。

    赤 (COLOR_RED=1) は H=0-13 と H=166-180 の2範囲で循環 Hue をカバーする。
    per_video inject が H=166-180 側のみ学習した場合、append=True でも
    DEFAULT の H=0-13 側が存在する前提。ただし inject 経路のバグや
    将来的な変更で欠落するリスクを guard する。

    Args:
        classifier: ColorClassifier インスタンス (_ranges 属性を持つオブジェクト)。
    """
    from src.image_reader import DEFAULT_COLOR_RANGES, HsvRange
    if not hasattr(classifier, "_ranges"):
        return
    for color in _CIRCULAR_GUARD_COLORS:
        if color not in DEFAULT_COLOR_RANGES:
            continue
        default_rngs = DEFAULT_COLOR_RANGES[color]
        if len(default_rngs) < 2:
            # DEFAULT が1範囲なら循環問題なし
            continue
        current: list[HsvRange] = list(classifier._ranges.get(color, []))
        for dflt in default_rngs:
            already = any(
                r.h_min == dflt.h_min and r.h_max == dflt.h_max
                for r in current
            )
            if not already:
                current.append(dflt)
                print(
                    f"[viz] circular_guard: color={color} "
                    f"H=[{dflt.h_min},{dflt.h_max}] を補完"
                )
        classifier._ranges[color] = current


def resolve_hsv_path(video_path: Path) -> Path:
    """動画ファイル名から動画 ID を抽出し、 per-video HSV JSON を自動選択する。

    優先順位:
      1. video_path のファイル名先頭から "(v[0-9]+)" を抽出
      2. data/per_video_hsv_ranges/{video_id}.json が存在 → それを返す (per-video 直接 inject)
      3. 不在 → _merged_default.json を返す (fallback)

    案 K (2026-05-24): 38 動画 union の背景誤認問題を per-video 直接 inject で回避。
    """
    import re
    m = re.match(r"(v\d+)", video_path.name)
    if m:
        candidate = _HSV_DB_ROOT / f"{m.group(1)}.json"
        if candidate.exists():
            return candidate
    return _HSV_MERGED_DEFAULT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-interval", type=float,
        default=DEFAULT_SAMPLE_INTERVAL,
        help="認識処理する frame 間隔 (秒)。出力は元動画の fps を維持し、未サンプル frame は最後の認識結果を保持。",
    )
    parser.add_argument(
        "--max-sec", type=float, default=0.0,
        help="入力動画の処理最大秒数 (0=全部)",
    )
    parser.add_argument("--cnn-model", type=str, default=None)
    parser.add_argument(
        "--hsv-state", type=Path,
        default=None,
        help="動画別 HSV ranges JSON (per_video_hsv_ranges DB)。 起動時に "
             "ColorClassifier.set_color_ranges_from_simple で注入。 "
             "省略時は resolve_hsv_path() が動画 ID から自動選択 "
             "(= 案 K: per-video JSON 優先、 不在時 _merged_default.json fallback)。 "
             "ファイル不在なら silent skip。 無効化したい場合は存在しないパスを渡す。",
    )
    parser.add_argument(
        "--vote-mode", action="store_true",
        help="ColorClassifier を per-pixel 投票方式に切替 (cycle 71)",
    )
    parser.add_argument(
        "--cnn-override-prob", type=float, default=None,
        help="HybridClassifier CNN 採用閾値 (None で default 0.70).",
    )
    parser.add_argument(
        "--mask-ojama-logit", action="store_true",
        help="cycle 32e: CNN の ojama logit を推論時 mask (= argmax 候補から除外)。 "
             "cycle 32e/32f model 等 ojama を学習対象外にした model で使用。",
    )
    parser.add_argument(
        "--no-online-hsv", action="store_true",
        help="OnlineHsvCalibrator を完全無効化 (= 動画中 HSV 学習なし)。 "
             "Step 0 OnlineHsv 効果定量評価用 (2026-05-24)。",
    )
    parser.add_argument(
        "--use-puyo-gate", action="store_true",
        help="cycle 32e: PuyoPresenceGate を HybridClassifier 前段に挟む。 "
             "gate=False の patch は HSV-only 経路に倒す (= 背景誤認対策)。",
    )
    parser.add_argument(
        "--use-circle-mask", action="store_true",
        help="cycle 32g: 推論時 patch に円形マスク適用 (学習時と必ず揃える)",
    )
    parser.add_argument(
        "--dump-board-log", type=Path, default=None,
        help="cycle 33 (2026-05-19): 各 frame の confirmed_board / state / "
             "chain_event を JSONL 形式で保存 (= 強化アナリスト用)。 "
             "evaluator が後処理で読み込んで自動評価する。",
    )
    args = parser.parse_args()
    # 案 K (2026-05-24): --hsv-state 省略時は動画 ID から自動選択
    if args.hsv_state is None:
        args.hsv_state = resolve_hsv_path(args.video)
        print(f"[viz] HSV auto-resolve: {args.hsv_state} (from {args.video.name})")
    # cycle 32g: 円形マスクを推論前に有効化
    if args.use_circle_mask:
        from src.patch_classifier import set_circle_mask_enabled
        set_circle_mask_enabled(True)
        print("[viz] use_circle_mask=ON (cycle 32g)")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {args.video}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.max_sec > 0:
        n_frames = min(n_frames, int(args.max_sec * fps))
    # 認識処理は 1920x1080 前提、出力もそのサイズで揃える
    out_w, out_h = 1920, 1080
    print(f"[input] {args.video} {width}x{height} fps={fps:.1f} frames={n_frames}")
    print(f"[output] {out_w}x{out_h} (resize から書き出し)")

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), fourcc, fps, (out_w, out_h),
    )

    # Pipeline (force_in_match=True で MENU 判定スキップ)
    pipeline = RecognitionPipeline.load_default(
        # 2 だと一時的 empty 観測で confirmed_board が誤確定する問題あり
        # (= 26s 1P 盤面 fully filled なのに empty 確定)。 6 で慎重に。
        # サイクル70: 6→3 で遅延 50% 短縮 (= 60fps で 50ms)
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=args.cnn_model,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
        vote_mode=args.vote_mode,
        cnn_override_prob=args.cnn_override_prob,
        mask_ojama_logit=args.mask_ojama_logit,
        use_puyo_gate=args.use_puyo_gate,
    )
    # Step 0 (2026-05-24): --no-online-hsv で OnlineHsvCalibrator を無効化
    if args.no_online_hsv:
        pipeline._online_hsv = None
        print("[viz] online_hsv DISABLED (= Step 0 比較用)")
    if args.vote_mode:
        print("[viz] vote_mode=ON (per-pixel HSV voting)")
    if args.cnn_override_prob is not None:
        print(f"[viz] cnn_override_prob={args.cnn_override_prob}")
    if args.mask_ojama_logit:
        print("[viz] mask_ojama_logit=ON (cycle 32e)")
    if args.use_puyo_gate:
        print("[viz] use_puyo_gate=ON (cycle 32e)")
    # 2026-05-11 サイクル63: 元動画解像度を image_reader に通知
    # (image_reader.read_both_boards で 1920x1080 にリサイズされるため、
    # pipeline.update に渡る frame からは元解像度が分からない)。
    if hasattr(pipeline._reader, "set_resolution_aware_s_min"):
        pipeline._reader.set_resolution_aware_s_min(height)
        print(f"[viz] resolution-aware S_min applied for source height={height}")
    # cycle 71r (案 A, 2026-05-13): BoardRegion 自動 calibration.
    # cycle 71u (2026-05-13 副作用対策): 案 A を撤回. 誤った座標補正で
    # ベース認識精度が悪化 (= 「双方とも認識悪化」 ユーザー報告) のため.
    # 必要なら --auto-calibrate 引数で明示的に有効化する.
    # 巻き戻し不要 (= cap は最初から再開).
    pass
    # サイクル4: 動画別 HSV ranges DB を起動時に inject
    if args.hsv_state is not None:
        try:
            import json as _json
            with args.hsv_state.open("r", encoding="utf-8") as _f:
                _state = _json.load(_f)
            _ranges = _state.get("per_video_ranges", {})
            _ranges_int = {
                int(k): tuple(int(x) for x in v) for k, v in _ranges.items()
            }
            from src.hybrid_classifier import HybridClassifier
            _hc = pipeline._reader._classifier
            if (
                isinstance(_hc, HybridClassifier)
                and hasattr(_hc._hsv, "set_color_ranges_from_simple")
                and _ranges_int
            ):
                _hc._hsv.set_color_ranges_from_simple(_ranges_int)
                # 循環 Hue 補完 guard: 赤等の2範囲定義色で per_video inject が
                # 片側 (H=0-13) を欠落させていないか確認し不足分を追加する。
                _ensure_circular_ranges_guard(_hc._hsv)
                # 2026-05-11 サイクル63 #6: 低解像度では pre-inject 後も
                # OnlineHsv の学習を継続させる (= merged_default は generic
                # で動画固有の調整が必要なため). 720p+ は DB が動画別で
                # tight なので従来通り suppress.
                if pipeline._online_hsv is not None and height >= 720:
                    pipeline._online_hsv_injected = True
                print(
                    f"[viz] HSV pre-inject from {args.hsv_state}: "
                    f"{len(_ranges_int)} colors "
                    f"(online_hsv {'suppressed' if height >= 720 else 'continues'})",
                )
        except Exception as _e:
            print(f"[viz] HSV pre-inject failed: {_e}", file=sys.stderr)

    sample_interval_frames = max(1, int(round(args.sample_interval * fps)))
    last_p1_state = BoardState.MENU
    last_p2_state = BoardState.MENU
    # 評価で使う盤面 = STABLE 時の confirmed_board を凍結保持
    # NON-STABLE (chain/tsumo_fall/ojama_fall/effect) では更新せず、前回 STABLE 値維持
    last_p1_eval_board: Board | None = None
    last_p2_eval_board: Board | None = None
    # cycle 33: board log JSONL 出力 (= 強化アナリスト用)
    board_log_fp = None
    if args.dump_board_log is not None:
        args.dump_board_log.parent.mkdir(parents=True, exist_ok=True)
        board_log_fp = open(args.dump_board_log, "w", encoding="utf-8")
        print(f"[viz] board log → {args.dump_board_log}")

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        t_sec = fi / fps
        # 認識実行 (sample_interval_frames ごと)
        if fi % sample_interval_frames == 0:
            result = pipeline.update(fi, t_sec, frame)
            last_p1_state = result.p1.state
            last_p2_state = result.p2.state
            # STABLE 時のみ確定盤面を取得 (= indicator 評価で使うのと同じ条件)
            if (result.p1.state == BoardState.STABLE
                    and result.p1.confirmed_board is not None):
                last_p1_eval_board = result.p1.confirmed_board
            if (result.p2.state == BoardState.STABLE
                    and result.p2.confirmed_board is not None):
                last_p2_eval_board = result.p2.confirmed_board
            # cycle 33: 各 frame の認識結果を JSONL に保存
            if board_log_fp is not None:
                import json as _json
                entry = {
                    "frame_idx": fi,
                    "t_sec": t_sec,
                    "p1_state": result.p1.state.value,
                    "p2_state": result.p2.state.value,
                    "p1_confirmed": (
                        result.p1.confirmed_board.to_dict()["grid"]
                        if result.p1.confirmed_board is not None else None
                    ),
                    "p2_confirmed": (
                        result.p2.confirmed_board.to_dict()["grid"]
                        if result.p2.confirmed_board is not None else None
                    ),
                }
                board_log_fp.write(_json.dumps(entry, ensure_ascii=False) + "\n")
        # 描画用エイリアス
        last_p1_board = last_p1_eval_board
        last_p2_board = last_p2_eval_board

        # 描画
        draw_cell_overlay(frame, last_p1_board, P1_ROI_X, P1_ROI_Y)
        draw_cell_overlay(frame, last_p2_board, P2_ROI_X, P2_ROI_Y)
        draw_state_label(frame, last_p1_state, P1_ROI_X, P1_ROI_Y, label_prefix="1P:")
        draw_state_label(frame, last_p2_state, P2_ROI_X, P2_ROI_Y, label_prefix="2P:")
        draw_global_info(frame, fi, t_sec, last_p1_state, last_p2_state)

        writer.write(frame)
        if fi % 100 == 0:
            print(f"  [progress] {fi}/{n_frames} ({fi*100/max(n_frames,1):.1f}%) "
                  f"1P={last_p1_state.value} 2P={last_p2_state.value}")

    cap.release()
    writer.release()
    if board_log_fp is not None:
        board_log_fp.close()
        print(f"[done] board log saved")
    print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
