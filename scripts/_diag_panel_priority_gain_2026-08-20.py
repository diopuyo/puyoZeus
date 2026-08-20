"""パネル優先にした場合のラベル増加を、収集し直さずに見積もる (2026-08-20)。

勝者ラベルの付与 (assign_won_labels) は収集の後処理なので、既存 npz の
game_idx / score と、動画から再計算したパネル判定さえあれば、パネル優先
(--enable-winner-panel-priority) にしたときの結果を再現できる。収集は
1本1時間かかるため、フル投入前の確認をこの方法で済ませる。

出力する数値:
  - 現在の欠損試合数/行数
  - パネル優先で救われる試合数/行数 (panel_winner が非 None のもの)
  - 救われずに残るもの (パネルが読めない = 端点など)
  - **パネルと得点が食い違った試合の一覧** (パネルを信じる判断が妥当か
    目視で確かめられるように、得点も並べて出す)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.match_winner import MatchWinnerDetector  # noqa: E402

_NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2"
_FRAMES_DIR = PROJECT_ROOT / "data" / "frames"


def _score_winner(s: np.ndarray, side: np.ndarray, sel: np.ndarray) -> tuple[str | None, int, int]:
    """得点系統の勝者と両者の最終得点 (高い方が勝ちという近似)。"""
    v: dict[str, int] = {}
    for sd in ("1P", "2P"):
        x = s[sel & (side == sd)]
        x = x[~np.isnan(x)]
        if len(x):
            v[sd] = int(np.max(x))
    if len(v) != 2 or v["1P"] == v["2P"]:
        return None, v.get("1P", -1), v.get("2P", -1)
    return ("1P" if v["1P"] > v["2P"] else "2P"), v["1P"], v["2P"]


def _analyze(tid: str, npz_dir: Path) -> None:
    """1本について、パネル優先で増えるラベルを数える。"""
    d = np.load(npz_dir / f"{tid}.npz", allow_pickle=True)
    won = np.asarray(d["won"], dtype=float)
    game = np.asarray(d["game_idx"])
    side = np.asarray(d["side"])
    t = np.asarray(d["t_sec"], dtype=float)
    s = np.asarray(d["score"], dtype=float)
    games = sorted(np.unique(game).tolist())
    starts = [float(t[game == g].min()) for g in games]

    cap = cv2.VideoCapture(str(_FRAMES_DIR / f"video_{tid}.mp4"))
    if not cap.isOpened():
        print(f"[skip] 動画を開けない: {tid}")
        return
    try:
        results = MatchWinnerDetector.load_default().detect_all_winners(
            cap, starts, float(t.max()),
        )
    finally:
        cap.release()

    saved_g, saved_rows, left_g, left_rows = 0, 0, 0, 0
    mismatches: list[str] = []
    for g, r in zip(games, results):
        sel = game == g
        rows = int(sel.sum())
        sw, s1, s2 = _score_winner(s, side, sel)
        pw = None if r.panel_unavailable else r.winner
        if pw is not None and sw is not None and pw != sw:
            mismatches.append(
                f"    試合{g:>3}: パネル={pw} 得点={sw} "
                f"(1P={s1:,} / 2P={s2:,} 差{abs(s1 - s2):,}) "
                f"変化 左{r.left_hamming}/右{r.right_hamming}"
            )
        if not bool(np.isnan(won[sel]).all()):
            continue
        if pw is not None:
            saved_g += 1
            saved_rows += rows
        else:
            left_g += 1
            left_rows += rows

    miss_rows = int(np.isnan(won).sum())
    total = len(won)
    print(f"\n=== {tid}: 全{total}行 現在の欠損{miss_rows}行 ({miss_rows/total*100:.1f}%) ===")
    print(f"  パネル優先で救われる : {saved_g}試合 {saved_rows}行")
    print(f"  救われず残る         : {left_g}試合 {left_rows}行 (パネルが読めない)")
    after = miss_rows - saved_rows
    print(f"  → 欠損 {miss_rows/total*100:.1f}% → {after/total*100:.1f}%")
    print(f"  パネルと得点の食い違い: {len(mismatches)}件"
          f"{' (下記、パネルを信じてよいか目視確認用)' if mismatches else ''}")
    for m in mismatches:
        print(m)


def main() -> int:
    """usage: <npz_dir> <target_id> ..."""
    if len(sys.argv) < 3:
        print("usage: <npz_dir> <target_id> ...")
        return 1
    npz_dir = _NPZ_DIR / sys.argv[1]
    for tid in sys.argv[2:]:
        _analyze(tid, npz_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
