"""色→別色棄却 (enable_ojama_fall_color_swap_guard) の実測検証 (2026-08-18)。

user発見: 連鎖発火の閃光エフェクトで既設置の色ぷよが別の色へ誤読される
(青→緑、赤→黄 等)。 本スクリプトは 3 動画 (video_36 / video_52 /
video_c100) の対戦区間 (t>=120s) のみで OJAMA_FALL 中の色→別色違反件数を
guard OFF (現行本番デフォルト) / ON (色→別色棄却フラグ有効) の両方で
実測し、以下を報告する:

  1. 違反件数の before/after (result.p{1,2}.cnn_board = 実際に下流へ渡る
     値そのもの、raw ではなく filter 後を見る点が
     _diag_state_transition_physics_2026-08-18.py との違い)。
  2. 固着チェック: guard ON 時、同一セルが同一の誤色に
     OJAMA_REJECT_TIMEOUT_SEC (=1.5秒) を大きく超えて (安全マージン込み
     2.5秒) 張り付いていないか (張り付いていれば固着バグの疑い、
     手動レビュー対象としてフラグを立てる)。

出力: logs/_verify_color_swap_guard_2026-08-18_<tag>_<on|off>.jsonl
      logs/_verify_color_swap_guard_2026-08-18_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recognition_pipeline import RecognitionPipeline, BoardState  # noqa: E402
from src.production_config import recognition_load_default_kwargs  # noqa: E402
from src.ojama_write_accounting import OJAMA_REJECT_TIMEOUT_SEC  # noqa: E402

# 固着判定の安全マージン (実測タイムアウト 1.5秒の 1.5倍超で「固着疑い」)。
STUCK_FLAG_MARGIN_SEC: float = 2.5

MATCH_START_SEC: float = 120.0  # user指定: 3動画とも対戦開始は2分前後


def _grid_of(board) -> list:
    return board.to_dict()["grid"] if board is not None else None


def run_video(
    video_path: str, start_sec: float, end_sec: float, tag: str, guard_on: bool,
) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    kwargs = recognition_load_default_kwargs()
    if guard_on:
        kwargs["enable_ojama_fall_color_swap_guard"] = True
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
        **kwargs,
    )

    suffix = "on" if guard_on else "off"
    log_path = Path(f"logs/_verify_color_swap_guard_2026-08-18_{tag}_{suffix}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "w", encoding="utf-8")

    episode_baseline = {"1P": None, "2P": None}
    prev_state = {"1P": None, "2P": None}
    baseline = {"1P": None, "2P": None}

    violations = {"1P": [], "2P": []}
    # 固着チェック用: (side, r, c) -> (wrong_value, streak_start_t_sec)
    stuck_streak: dict = {}
    flagged_stuck = {"1P": [], "2P": []}

    fi = 0
    frame_idx_out = start_frame
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = start_sec + fi / fps
        if t_sec > end_sec:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        result = pipeline.update(frame_idx_out, t_sec, frame)
        for side, res in (("1P", result.p1), ("2P", result.p2)):
            state = res.state
            cur_grid = _grid_of(res.cnn_board)

            if state != BoardState.STABLE and prev_state[side] != state:
                if baseline[side] is not None:
                    episode_baseline[side] = baseline[side]
            if state == BoardState.STABLE and res.confirmed_board is not None:
                baseline[side] = _grid_of(res.confirmed_board)
                episode_baseline[side] = None

            if (
                state == BoardState.OJAMA_FALL
                and episode_baseline[side] is not None
                and cur_grid is not None
            ):
                base = episode_baseline[side]
                for r in range(13):
                    for c in range(6):
                        b = base[r][c]
                        cur = cur_grid[r][c]
                        key = (side, r, c)
                        is_color_swap = (
                            b in (1, 2, 3, 4, 5) and cur in (1, 2, 3, 4, 5)
                            and cur != b
                        )
                        if is_color_swap:
                            violations[side].append({
                                "t_sec": round(t_sec, 3), "r": r, "c": c,
                                "before": b, "after": cur,
                            })
                            existing = stuck_streak.get(key)
                            if existing is None or existing[0] != cur:
                                stuck_streak[key] = (cur, t_sec)
                            duration = t_sec - stuck_streak[key][1]
                            if duration > OJAMA_REJECT_TIMEOUT_SEC + STUCK_FLAG_MARGIN_SEC:
                                flagged_stuck[side].append({
                                    "r": r, "c": c, "value": cur,
                                    "duration_sec": round(duration, 3),
                                    "t_sec": round(t_sec, 3),
                                })
                        else:
                            stuck_streak.pop(key, None)

            log_fp.write(json.dumps({
                "t_sec": round(t_sec, 3), "side": side, "state": state.value,
                "cnn_grid": cur_grid,
            }, ensure_ascii=False) + "\n")
            prev_state[side] = state
        fi += 1
        frame_idx_out += 1
        if fi % 300 == 0:
            print(f"  [{tag}/{suffix}] progress t={t_sec:.1f}s")

    cap.release()
    log_fp.close()

    summary = {
        "tag": tag, "guard_on": guard_on,
        "violation_count": {s: len(violations[s]) for s in ("1P", "2P")},
        "flagged_stuck_count": {s: len(flagged_stuck[s]) for s in ("1P", "2P")},
        "flagged_stuck_detail": flagged_stuck,
    }
    return summary


def main() -> None:
    jobs = [
        ("data/frames/video_36.mp4", 141.0, 148.0, "video36_spotcheck"),
        ("data/frames/video_36.mp4", MATCH_START_SEC, 160.0, "video36"),
        ("data/frames/video_52.mp4", MATCH_START_SEC, 160.0, "video52"),
        ("data/frames/video_c100.mp4", 570.0, 660.0, "c100"),
    ]
    all_summary = []
    for video_path, s, e, tag in jobs:
        if not Path(video_path).exists():
            print(f"[skip] {video_path} not found")
            continue
        for guard_on in (False, True):
            print(f"=== {tag} guard_on={guard_on}: {video_path} [{s},{e}] ===")
            summ = run_video(video_path, s, e, tag, guard_on)
            all_summary.append(summ)
    out_path = Path("logs/_verify_color_swap_guard_2026-08-18_summary.json")
    out_path.write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
