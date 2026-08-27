"""残る勝敗ラベル欠損を A/B に分離する診断 (2026-08-20)。

配線是正後も残る欠損 (39番: 59試合中15試合が丸ごと欠損) の原因を
  A: パネルが**そもそも映っていない** (panel_unavailable) → 見に行く時刻の問題
  B: パネルは読めたが**得点系統と食い違った** → 読み取り精度の問題
に分ける。対策が正反対 (A=探索時刻の修正 / B=認識精度の向上) なので、
どちらが主因かを数値で確定させる。修正はしない (診断専用)。

user 伝授 (2026-08-20): 得点が高い方が勝つのは約98%成立。よって不一致の
うち得点側起因は期待値で全体の2%程度で、残りはパネル側と按分できる
(memory reference_score_winner_98pct_2026-08-20)。

処理は動画の全フレーム走査ではなく、試合ごとの**数フレームのシーク**だけ
なので収集ジョブと並行して走らせても負荷は軽い (cv2 は1スレッド固定)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# 収集ジョブと競合させない (memory project_collect_indicators_v2_perf_2026-07-20)
cv2.setNumThreads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.match_winner import MatchWinnerDetector  # noqa: E402

_NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2"
_FRAMES_DIR = PROJECT_ROOT / "data" / "frames"


def _score_winner(s: np.ndarray, side: np.ndarray, sel: np.ndarray) -> str | None:
    """得点系統の勝者 (得点が高い方が勝ち、collect_boards_lean と同一ロジック)。"""
    vals: dict[str, float] = {}
    for sd in ("1P", "2P"):
        v = s[sel & (side == sd)]
        v = v[~np.isnan(v)]
        if len(v):
            vals[sd] = float(np.max(v))
    if len(vals) != 2 or vals["1P"] == vals["2P"]:
        return None
    return "1P" if vals["1P"] > vals["2P"] else "2P"


def _analyze(npz_path: Path, video_path: Path) -> None:
    """1本について、欠損試合ごとに A/B を判定して出力する。"""
    d = np.load(npz_path, allow_pickle=True)
    won = np.asarray(d["won"], dtype=float)
    game = np.asarray(d["game_idx"])
    t = np.asarray(d["t_sec"], dtype=float)
    score = np.asarray(d["score"], dtype=float)
    side = np.asarray(d["side"])

    games = sorted(np.unique(game).tolist())
    # 試合開始時刻は各試合の最小 t_sec で近似する (npz は記録行の時刻しか持たない)
    starts = [float(t[game == g].min()) for g in games]
    last_obs = float(t.max())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[error] 動画を開けない: {video_path}")
        return
    try:
        det = MatchWinnerDetector.load_default()
        results = det.detect_all_winners(cap, starts, last_obs)
    finally:
        cap.release()

    print(f"\n=== {npz_path.stem} : 試合{len(games)} 行{len(won)} "
          f"欠損{int(np.isnan(won).sum())}行 ===")
    print(f"{'試合':>5} {'状態':>6} {'パネル':>8} {'得点':>6} {'判定':>22}")
    print("-" * 56)

    cnt = {"A_unavailable": 0, "B_mismatch": 0, "agree": 0, "score_none": 0}
    for g, r in zip(games, results):
        sel = game == g
        missing = bool(np.isnan(won[sel]).all())
        sw = _score_winner(score, side, sel)
        pw = None if r.panel_unavailable else r.winner
        if r.panel_unavailable:
            kind, key = "A:映っていない", "A_unavailable"
        elif sw is None:
            kind, key = "得点側が出せない", "score_none"
        elif pw is None:
            kind, key = "A:パネル判定不能", "A_unavailable"
        elif pw != sw:
            kind, key = "B:食い違い", "B_mismatch"
        else:
            kind, key = "一致", "agree"
        if missing:
            cnt[key] += 1
            print(f"{g:>5} {'欠損':>6} {str(pw):>8} {str(sw):>6} {kind:>22}")

    print("-" * 56)
    tot = sum(cnt.values())
    print(f"欠損試合の内訳 (計{tot}):")
    print(f"  A: パネルが映っていない/判定不能 = {cnt['A_unavailable']}")
    print(f"  B: 読めたが得点と食い違い       = {cnt['B_mismatch']}")
    print(f"  得点側が出せない                = {cnt['score_none']}")
    print(f"  (一致しているのに欠損)          = {cnt['agree']}")


def main() -> int:
    """引数: <npzディレクトリ名> <target_id> ... 。"""
    if len(sys.argv) < 3:
        print("usage: <npz_dir> <target_id> [<target_id> ...]")
        return 1
    npz_dir = _NPZ_DIR / sys.argv[1]
    for tid in sys.argv[2:]:
        npz = npz_dir / f"{tid}.npz"
        vid = _FRAMES_DIR / f"video_{tid}.mp4"
        if not npz.exists():
            print(f"[skip] npzなし: {npz}")
            continue
        if not vid.exists():
            print(f"[skip] 動画なし: {vid}")
            continue
        _analyze(npz, vid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
