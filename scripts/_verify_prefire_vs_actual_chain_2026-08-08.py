"""発火前の予測連鎖数と、 発火後の実連鎖数が一致するかを測る (2026-08-08).

user 質問「発火前の想定と発火後の連鎖数はあっていますか?」への回答。

前回の測定 (2026-07-30、 [[project-chain-count-both-untrustworthy-2026-07-30]])
では simulate が一方向に壊滅的過小 (真値 8 連鎖 -> 予測 1 連鎖) で、 原因は
発火直前 STABLE 盤面に浮きぷよ (重力違反) が混じり連結が欠けることだった
([[project-gravity-violation-regen-lead-2026-07-30]])。 その後、 認識は
バーストガード・A'・非対称確定などで大きく変わっているため測り直す。

方法:
  1. デモクリップを認識パイプラインで通す。
  2. 各 side の STABLE 時の confirmed_board を「直前の確定盤面」として保持し
     続ける (評価と同じ凍結ルール)。
  3. ChainEvent が新規に立ったフレームで、 保持していた直前 STABLE 盤面を
     simulate_single にかけて予測連鎖数を求め、 ChainEvent.chain_count
     (掛け算表示由来 = 実測値) と突き合わせる。
  4. 発火直前盤面の重力違反 (浮きぷよ) 数も併記し、 ズレの原因を切り分ける。

出力: data/verify/youtube_demo_2026-08-07/prefire_vs_actual_chain_2026-08-08.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.chain_bitboard import simulate_single  # noqa: E402
from src.console_init import init_console  # noqa: E402

init_console()

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO = _ROOT / "data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4"
MODEL = _ROOT / "models/cnn_finetune_olRyxDGacbg_demo_v3_2026-08-07.pt"
OUT_TSV = (
    _ROOT / "data/verify/youtube_demo_2026-08-07"
    / "prefire_vs_actual_chain_2026-08-08.tsv"
)


def _count_floating(grid: np.ndarray) -> int:
    """浮きぷよ (下が空なのにぷよがあるセル) の数を返す.

    重力違反の直接指標。 これが多いほど連結が欠け simulate が過小になる。
    """
    n = 0
    for col in range(BOARD_COLS):
        for row in range(BOARD_ROWS - 1):
            if grid[row, col] != COLOR_EMPTY and grid[row + 1, col] == COLOR_EMPTY:
                n += 1
    return n


def _predict(board) -> tuple[int, int]:
    """(予測連鎖数, 浮きぷよ数) を返す."""
    if board is None:
        return -1, -1
    try:
        res = simulate_single(board)
        pred = int(getattr(res, "chain_count", -1))
    except Exception:
        pred = -1
    return pred, _count_floating(board._grid)


def main() -> int:
    pipeline = RecognitionPipeline.load_default(
        cnn_model_path=MODEL,
        force_in_match=True,
        enable_effect_gate=True,
        enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=0.954,
        enable_hidden_row_burst_guard=True,
        enable_match_transition_debounce=True,
        enable_asymmetric_recovery_min_frames=True,
        recovery_add_min_frames=3,
        enable_ojama_entry_gravity_settle_guard=True,
        enable_gravity_settle_reset_on_exit=True,
    )
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    last_stable = {"1P": None, "2P": None}
    seen_triggers: set[tuple[str, float]] = set()
    rows: list[tuple] = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)
        for side, sr in (("1P", result.p1), ("2P", result.p2)):
            ev = getattr(sr, "chain_event", None)
            if ev is not None:
                trig = getattr(ev, "trigger_sec", None)
                cnt = getattr(ev, "chain_count", None)
                key = (side, round(float(trig), 3)) if trig is not None else None
                if key is not None and key not in seen_triggers:
                    seen_triggers.add(key)
                    pred, floating = _predict(last_stable[side])
                    rows.append((
                        side, float(trig), int(cnt) if cnt is not None else -1,
                        pred, floating, getattr(ev, "mechanism", ""),
                    ))
            # STABLE の確定盤面を凍結保持 (評価と同じルール)
            if sr.state == BoardState.STABLE and sr.confirmed_board is not None:
                last_stable[side] = sr.confirmed_board
        fi += 1
    cap.release()

    lines = ["side\ttrigger_sec\tactual_chain\tpredicted_chain\tfloating_cells\tmechanism"]
    for side, trig, cnt, pred, fl, mech in rows:
        lines.append(f"{side}\t{trig:.3f}\t{cnt}\t{pred}\t{fl}\t{mech}")
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")

    valid = [(c, p, f) for _, _, c, p, f in
             [(s, t, c, p, f) for s, t, c, p, f, _ in rows] if c > 0 and p >= 0]
    print(f"\n発火イベント {len(rows)} 件 (うち比較可能 {len(valid)} 件)")
    if valid:
        diffs = [p - c for c, p, _ in valid]
        exact = sum(1 for d in diffs if d == 0)
        under = sum(1 for d in diffs if d < 0)
        over = sum(1 for d in diffs if d > 0)
        print(f"  一致 {exact} 件 / 過小 {under} 件 / 過大 {over} 件")
        print(f"  予測-実測の中央値 = {np.median(diffs):+.1f}")
        print(f"  浮きぷよ数 中央値 = {np.median([f for _, _, f in valid]):.1f}")
        print("\n  実測 -> 予測 の内訳 (上位 15 件):")
        for c, p, f in valid[:15]:
            print(f"    実測 {c:2d} 連鎖 -> 予測 {p:2d} 連鎖 (浮きぷよ {f})")
    print(f"\n出力: {OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
