"""Phase B レビュー用オーバーレイ動画生成 (B-16).

新方針 pipeline (BoardStateMachine + per-video model + smoothing) を回し、
各 frame の state, confirmed_board, drift, score などをオーバーレイした
動画を出力する。視覚的に「pipeline が正しく動いているか」をレビュー可能に
する目的。

オーバーレイ要素 (1920x1080 にレイヤー):
    - 上部バー: t=, 1P state, 2P state, 1P score, 2P score, frame_idx
    - 1P 盤面右上: 1P drift mismatch_count, last_resync_frame
    - 2P 盤面右上: 2P drift mismatch_count
    - 各 cell: confirmed_board の色を盤面領域に細枠で重ねる (デバッグ表示)

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_b_render_review_video \
        --videos 1,7,13 --duration 30 --out-dir data/review_videos
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402

init_console()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.board import (  # noqa: E402
    BOARD_COLS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN,
    COLOR_OJAMA, COLOR_PURPLE, COLOR_RED, COLOR_YELLOW,
    HIDDEN_ROWS, VISIBLE_ROWS,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION, DEFAULT_P2_REGION, BoardRegion,
)
from src.old.indicators import IndicatorCalculator  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from src.old.scorer import PhaseAwareScorer, ScoreResult  # noqa: E402

# 盤面色 → BGR (薄く表示)
COLOR_BGR: dict[int, tuple[int, int, int]] = {
    COLOR_EMPTY: (60, 60, 60),
    COLOR_RED: (40, 40, 220),
    COLOR_BLUE: (220, 80, 40),
    COLOR_GREEN: (40, 200, 40),
    COLOR_YELLOW: (40, 220, 240),
    COLOR_PURPLE: (200, 40, 200),
    COLOR_OJAMA: (170, 170, 170),
}

STATE_LABEL: dict[BoardState, str] = {
    BoardState.MENU: "MENU",
    BoardState.STABLE: "STABLE",
    BoardState.TSUMO_FALL: "TSUMO",
    BoardState.CHAIN: "CHAIN",
    BoardState.OJAMA_FALL: "OJAMA",
    BoardState.EFFECT: "EFFECT",
}

STATE_COLOR: dict[BoardState, tuple[int, int, int]] = {
    BoardState.MENU: (100, 100, 100),
    BoardState.STABLE: (40, 200, 40),
    BoardState.TSUMO_FALL: (40, 200, 240),
    BoardState.CHAIN: (40, 80, 240),
    BoardState.OJAMA_FALL: (200, 100, 100),
    BoardState.EFFECT: (200, 40, 200),
}


def get_match1(video_id: int) -> tuple[float, float] | None:
    for ver in ("v5", "v4"):
        p = (
            _ROOT
            / f"data/verify/match_boundaries_{ver}/video_{video_id:02d}/"
            f"matches.tsv"
        )
        if not p.exists():
            continue
        with p.open() as f:
            rows = list(csv.reader(f, delimiter="\t"))
        if len(rows) > 1:
            try:
                return float(rows[1][1]), float(rows[1][2])
            except (IndexError, ValueError):
                continue
    return None


def get_match_range(
    video_id: int, n_matches: int = 1,
) -> tuple[float, float] | None:
    """試合 1 〜 試合 n_matches までを連続的に含む時間範囲を返す.

    試合 1 の開始から試合 n_matches の終了まで (= 試合間の遷移区間も含む)。
    """
    for ver in ("v5", "v4"):
        p = (
            _ROOT
            / f"data/verify/match_boundaries_{ver}/video_{video_id:02d}/"
            f"matches.tsv"
        )
        if not p.exists():
            continue
        with p.open() as f:
            rows = list(csv.reader(f, delimiter="\t"))
        if len(rows) > n_matches:
            try:
                start = float(rows[1][1])
                end = float(rows[n_matches][2])
                return start, end
            except (IndexError, ValueError):
                continue
    return None


def select_cnn_model(
    video_id: int, per_video: bool, single_model: Path | None,
) -> Path | None:
    if per_video:
        from src.per_video_model_selector import select_phase_b_model
        m = select_phase_b_model(video_id)
        return Path(m) if m else None
    return single_model


_COLOR_LETTER: dict[int, str] = {
    COLOR_RED: "R", COLOR_BLUE: "B", COLOR_GREEN: "G",
    COLOR_YELLOW: "Y", COLOR_PURPLE: "P", COLOR_OJAMA: "O",
}


def draw_board_overlay(
    canvas: np.ndarray, board, region: BoardRegion,
    border_color: tuple[int, int, int] = (40, 200, 40),
) -> None:
    """盤面 cell の認識色を 1 文字テキストで表示.

    R=赤 / B=青 / G=緑 / Y=黄 / P=紫 / O=おじゃま。
    色背景の小四角に白文字で描画 → 色 + 文字の両方で識別可能。

    SideResult.inferred_board を渡すと:
      - STABLE: confirmed_board と同等 = 静止中の確定盤面
      - CHAIN: ChainSimulator が出した段階的盤面 (連鎖進行に応じて変化)
      - TSUMO/OJAMA/EFFECT: 直近 STABLE 盤面 hold
      - MENU: None (描画しない)
    """
    if board is None:
        return
    overlay = canvas.copy()
    for r in range(HIDDEN_ROWS, HIDDEN_ROWS + VISIBLE_ROWS):
        for c in range(BOARD_COLS):
            cx, cy = region.cell_center(r, c)
            v = int(board.get(r, c))
            if v == COLOR_EMPTY:
                continue
            color = COLOR_BGR.get(v, (200, 200, 200))
            # 色背景小さめ円を overlay に
            cv2.circle(overlay, (cx, cy), 10, color, -1)
    # 半透明合成で元 puyo 画像も透けて見える
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    # 文字は不透明で上に描画
    for r in range(HIDDEN_ROWS, HIDDEN_ROWS + VISIBLE_ROWS):
        for c in range(BOARD_COLS):
            cx, cy = region.cell_center(r, c)
            v = int(board.get(r, c))
            if v == COLOR_EMPTY:
                continue
            letter = _COLOR_LETTER.get(v, "?")
            (tw, th), _ = cv2.getTextSize(
                letter, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2,
            )
            tx = cx - tw // 2
            ty = cy + th // 2
            # 黒縁取り → 白文字 (どの色背景でも見える)
            cv2.putText(
                canvas, letter, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas, letter, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA,
            )


def draw_next_queue_hud(
    canvas: np.ndarray, ctx_next_queue: list[tuple[int, int]],
    side: str, x_anchor: int,
) -> None:
    """next_queue (= ネクストぷよ履歴) を HUD 下部に表示.

    ペア色を 2 つの dot で表現。最新ペアが現在ツモ (推論ベース)。
    """
    if not ctx_next_queue:
        return
    label_y = 80
    cv2.putText(
        canvas, f"{side} next:", (x_anchor, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
    )
    base_x = x_anchor + 110
    for i, (a, b) in enumerate(ctx_next_queue[-3:]):
        # 新しいほど右、最新 = 現在ツモ推定値
        col_a = COLOR_BGR.get(int(a), (100, 100, 100))
        col_b = COLOR_BGR.get(int(b), (100, 100, 100))
        cx = base_x + i * 70
        cv2.circle(canvas, (cx, label_y - 14), 10, col_a, -1)
        cv2.circle(canvas, (cx, label_y - 14), 11, (0, 0, 0), 1)
        cv2.circle(canvas, (cx, label_y + 8), 10, col_b, -1)
        cv2.circle(canvas, (cx, label_y + 8), 11, (0, 0, 0), 1)
        if i == len(ctx_next_queue[-3:]) - 1:
            # 最新 (= 現在ツモ推定) を赤枠で強調
            cv2.rectangle(
                canvas, (cx - 14, label_y - 28),
                (cx + 14, label_y + 22), (40, 40, 220), 2,
            )


def draw_top_hud(
    canvas: np.ndarray, time_sec: float, frame_idx: int,
    p1_state: BoardState, p2_state: BoardState,
    p1_score: int | None, p2_score: int | None,
    p1_drift: int, p2_drift: int,
) -> None:
    """画面上部の HUD バー."""
    cv2.rectangle(canvas, (0, 0), (1920, 50), (10, 10, 10), -1)

    # t=
    cv2.putText(
        canvas, f"t={time_sec:6.2f}s", (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA,
    )

    # 1P state
    p1_color = STATE_COLOR.get(p1_state, (200, 200, 200))
    cv2.putText(
        canvas, f"1P:{STATE_LABEL.get(p1_state, '?')}", (260, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, p1_color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, f"drift={p1_drift}", (490, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 100), 1, cv2.LINE_AA,
    )

    # score
    p1_s = f"{p1_score:08d}" if p1_score is not None else "--------"
    p2_s = f"{p2_score:08d}" if p2_score is not None else "--------"
    cv2.putText(
        canvas, f"1P:{p1_s}", (640, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, f"2P:{p2_s}", (900, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA,
    )

    # 2P state
    p2_color = STATE_COLOR.get(p2_state, (200, 200, 200))
    cv2.putText(
        canvas, f"2P:{STATE_LABEL.get(p2_state, '?')}", (1180, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, p2_color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, f"drift={p2_drift}", (1410, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 100), 1, cv2.LINE_AA,
    )

    # frame_idx
    cv2.putText(
        canvas, f"f={frame_idx:5d}", (1700, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
    )


def _stable_inferred(
    new_board, last_drawn,
):
    """inferred_board が None/空 なら直前 frame の board を hold する.

    state 遷移境界で 1-2 frame 一時的に空に落ちる現象を緩和して
    視覚的な継続性を確保する。
    本物の全消し演出は対象外 (新 inferred_board が EMPTY-only の場合は
    last_drawn が puyo を持っていてもそのまま採用しない、新の方を採用)。
    """
    if new_board is None:
        return last_drawn
    return new_board


def draw_advantage_hud(
    canvas: np.ndarray, score_result: "ScoreResult | None",
    elapsed_sec: float, match_dur_sec: float,
) -> None:
    """画面下部に有利不利スコアバーをオーバーレイ表示."""
    bar_y = 1010
    cv2.rectangle(
        canvas, (700, bar_y - 38), (1220, bar_y + 18), (10, 10, 10), -1,
    )
    if score_result is None:
        cv2.putText(
            canvas, "Advantage: --", (720, bar_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA,
        )
        return
    adv = score_result.advantage_side()
    total = score_result.total_score
    color = (
        (40, 200, 40) if adv == "1P" else
        (40, 40, 220) if adv == "2P" else
        (200, 200, 200)
    )
    cv2.putText(
        canvas, f"Advantage: {adv:>4} {total:+6.1f}", (720, bar_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
    )
    # スコアバー (-100 ... 0 ... +100、中央が 0)
    bar_x_start = 720
    bar_x_end = 1200
    bar_mid = (bar_x_start + bar_x_end) // 2
    bar_y_mid = bar_y - 20
    cv2.line(
        canvas, (bar_x_start, bar_y_mid), (bar_x_end, bar_y_mid),
        (100, 100, 100), 2,
    )
    cv2.line(
        canvas, (bar_mid, bar_y_mid - 8), (bar_mid, bar_y_mid + 8),
        (200, 200, 200), 1,
    )
    # スコア位置を可視化
    sx = bar_mid + int(total / 100.0 * (bar_x_end - bar_mid))
    sx = max(bar_x_start, min(bar_x_end, sx))
    cv2.circle(canvas, (sx, bar_y_mid), 7, color, -1)
    cv2.circle(canvas, (sx, bar_y_mid), 8, (0, 0, 0), 1)


def render_video(
    video_id: int, start_sec: float, end_sec: float,
    fps_sample: float, stable_n: int, smoothing_n: int,
    cnn_model: Path | None, out_dir: Path,
    display_fps: float | None = None,
) -> Path | None:
    """fps_sample で評価、display_fps で動画書出し (= 間引き)。

    display_fps=None なら fps_sample と同じ (従来動作)。
    例: fps_sample=33, display_fps=11 → 評価 0.03秒間隔、表示 0.09秒間隔
    (3 frame ごとに 1 frame だけ動画に書出し)。
    """
    video_path = _ROOT / "data" / "frames" / f"video_{video_id:02d}.mp4"
    if not video_path.exists():
        return None
    if display_fps is None or display_fps >= fps_sample:
        display_fps = fps_sample

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=stable_n,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=cnn_model,
        temporal_smoothing=smoothing_n,
        # force_in_match を撤去: score-based match strengthen で
        # MatchStateDetector の誤判定を補正できるはず。試合外では
        # score=0 / 読めないので false positive は出ない設計。
        force_in_match=False,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"v{video_id:02d}_review_{int(start_sec)}_{int(end_sec)}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path), fourcc, display_fps, (1920, 1080),
    )
    if not writer.isOpened():
        cap.release()
        return None

    interval = 1.0 / fps_sample
    write_every_n = max(1, int(round(fps_sample / display_fps)))
    t = start_sec
    frame_idx = 0
    last_inf_1p = None
    last_inf_2p = None
    # Scorer 統合 (D-1)
    ind_calc = IndicatorCalculator()
    scorer = PhaseAwareScorer(weight_mode="optimal")
    match_dur = max(1.0, end_sec - start_sec)
    last_score_result: ScoreResult | None = None
    last_score_key: tuple[str, str] | None = None
    while t < end_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        result = pipe.update(frame_idx, t, frame)

        # 状態遷移境界で 1-2 frame inferred_board が None になる現象を hold で緩和
        inf_1p = _stable_inferred(result.p1.inferred_board, last_inf_1p)
        inf_2p = _stable_inferred(result.p2.inferred_board, last_inf_2p)
        last_inf_1p = inf_1p
        last_inf_2p = inf_2p

        canvas = frame.copy()
        # 盤面オーバーレイ — inferred_board を表示 (state ごとに切替):
        #   STABLE = confirmed、CHAIN = ChainSim 段階盤面、
        #   TSUMO/OJAMA/EFFECT = 直近 STABLE hold
        draw_board_overlay(canvas, inf_1p, DEFAULT_P1_REGION)
        draw_board_overlay(canvas, inf_2p, DEFAULT_P2_REGION)
        # HUD
        draw_top_hud(
            canvas, t, frame_idx,
            result.p1.state, result.p2.state,
            result.p1.score, result.p2.score,
            result.p1.drift.mismatch_count,
            result.p2.drift.mismatch_count,
        )
        # ネクスト履歴 (現在ツモ推定値 = 最新 next pair)
        sm_1p_ctx = pipe._sm_1p.context  # type: ignore[attr-defined]
        sm_2p_ctx = pipe._sm_2p.context  # type: ignore[attr-defined]
        draw_next_queue_hud(canvas, sm_1p_ctx.next_queue, "1P", 60)
        draw_next_queue_hud(canvas, sm_2p_ctx.next_queue, "2P", 1100)

        # 有利不利スコア計算 (D-1, D-2 厳格化): 両側 STABLE のときのみ計算
        # アクション中 side の評価が変動しないことを構造的に保証
        if (
            result.p1.state == BoardState.STABLE
            and result.p2.state == BoardState.STABLE
            and result.p1.confirmed_board is not None
            and result.p2.confirmed_board is not None
        ):
            try:
                key = (
                    result.p1.confirmed_board.to_json(),
                    result.p2.confirmed_board.to_json(),
                )
                if key != last_score_key:
                    p1_set = ind_calc.compute_all(result.p1.confirmed_board)
                    p2_set = ind_calc.compute_all(result.p2.confirmed_board)
                    elapsed = max(0.0, t - start_sec)
                    last_score_result = scorer.score(
                        p1_set, p2_set, elapsed, match_dur,
                    )
                    last_score_key = key
            except Exception:
                pass
        draw_advantage_hud(
            canvas, last_score_result,
            max(0.0, t - start_sec), match_dur,
        )

        # display_fps に応じて間引き書出し
        if frame_idx % write_every_n == 0:
            writer.write(canvas)
        frame_idx += 1
        t += interval
    cap.release()
    writer.release()
    print(
        f"[done] v{video_id:02d}: {frame_idx} frames -> "
        f"{to_windows_path(out_path)}"
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=str, default="1,7,13")
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="動画の長さ秒。--full-match で試合 1 全体を使う場合は無視",
    )
    parser.add_argument(
        "--full-match", action="store_true",
        help="match_boundaries の試合 1 区間全体を使う (duration を上書き)",
    )
    parser.add_argument(
        "--n-matches", type=int, default=1,
        help="連続 N 試合を抽出 (試合 1 開始 〜 試合 N 終了)。"
             "--full-match と組み合わせて使う",
    )
    parser.add_argument(
        "--random-n", type=int, default=0,
        help="0 なら --videos そのまま、>0 なら全 19 動画から N 本ランダム選定",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--display-fps", type=float, default=None,
        help="動画書出し fps (省略時 fps と同じ)。fps より低くすれば間引き表示",
    )
    parser.add_argument("--stable-n", type=int, default=6)
    parser.add_argument(
        "--smoothing-n", type=int, default=None,
        help="CNN 時系列平均 N (省略時は per_video_model_selector の値、"
             "明示なら CLI 値で上書き、1=OFF)",
    )
    parser.add_argument("--per-video-model", action="store_true", default=True)
    parser.add_argument("--cnn-model", type=Path, default=None)
    parser.add_argument(
        "--out-dir", type=Path,
        default=_ROOT / "data" / "review_videos",
    )
    args = parser.parse_args()

    if args.random_n > 0:
        # 試合 1 区間が 30 秒以上ある動画のみ候補に (短い v05/v15 等は除外検討)
        candidates: list[int] = []
        for vid in range(1, 20):
            m = get_match1(vid)
            if m is None:
                continue
            if (m[1] - m[0]) >= 30.0:
                candidates.append(vid)
        import random
        rng = random.Random(args.seed)
        target_ids = rng.sample(candidates, min(args.random_n, len(candidates)))
        print(
            f"[random] selected {target_ids} from "
            f"{len(candidates)} candidates (seed={args.seed})"
        )
    else:
        target_ids = [int(s) for s in args.videos.split(",") if s.strip()]
    for vid in target_ids:
        if args.n_matches > 1:
            m = get_match_range(vid, args.n_matches)
        else:
            m = get_match1(vid)
        if m is None:
            print(f"[skip] v{vid:02d}")
            continue
        start = m[0]
        if args.full_match or args.n_matches > 1:
            end = m[1]
        else:
            end = min(m[1], start + args.duration)
        cnn_model = select_cnn_model(
            vid, args.per_video_model, args.cnn_model,
        )
        # smoothing 決定: 明示 CLI が最優先、無ければ per-video selector
        if args.smoothing_n is not None:
            smoothing_n = args.smoothing_n
        elif args.per_video_model:
            from src.per_video_model_selector import select_phase_b_smoothing
            smoothing_n = select_phase_b_smoothing(vid)
        else:
            smoothing_n = 1
        tag = f"CNN={cnn_model.name}" if cnn_model else "HSV"
        print(
            f"[run] v{vid:02d} ({tag}, sm={smoothing_n}): "
            f"[{start:.1f}, {end:.1f}]",
            flush=True,
        )
        render_video(
            video_id=vid, start_sec=start, end_sec=end,
            fps_sample=args.fps, stable_n=args.stable_n,
            smoothing_n=smoothing_n,
            cnn_model=cnn_model, out_dir=args.out_dir,
            display_fps=args.display_fps,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
