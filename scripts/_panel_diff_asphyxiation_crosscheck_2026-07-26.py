"""#43 追加検証: WIN★パネル差分の勝者判定を「窒息盤面」でクロスチェックする。

user伝授: 勝敗の本質は「3列目(col=2)の画面内最上段(row=DEATH_ROW=1)を
埋めた側が負け」= 窒息。得点は無関係。

軽量方針:
    新規に動画を再走査して盤面認識をやり直すのではなく、既存の認識済み
    盤面スナップショット (data/indicators_v2/boards_lean_fixed/*.npz,
    data/indicators_v2/boards/*.npz, *_mid.npz) を再利用する。
    これらは既に CNN/state machine で認識済みの grids (t_sec, side, grid)
    を持つ「軽量」データなので、追加の重い計算は発生しない。

制約 (正直に報告する):
    上記 npz は収集時に打ち切り (c系: 先頭20分 t<=1200s、v系:
    先頭5分 t<=300s + 中盤6分 t=[1200,1560]) があるため、その時間内に
    試合終了した試合のみクロスチェック可能。範囲外の試合は「対象外」
    として除外し、カバレッジ (対象外率) を明示する。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from src.board import COLOR_EMPTY, COLOR_UNKNOWN, DEATH_COL, DEATH_ROW

# 試合終了時刻から遡ってスナップショットを探す許容範囲 (秒)。
# これを超えて古いスナップショットしかない場合はクロスチェック対象外とする。
SNAPSHOT_STALENESS_LIMIT_SEC: float = 20.0

# c系: panel-diff-mode 出力 JSON (本検証で新規生成したもの)
C_PANEL_DIFF_JSON_DIR = Path("data/verify/step0_winstar_cseries_2026-07-26")
C_BOARDS_NPZ_DIR = Path("data/indicators_v2/boards_lean_fixed")
C_VIDEOS = ["c1", "c4", "c34", "c82"]

# v系回帰: 既存デフォルトモード winners JSON + 既存 boards npz (5分/中盤6分窓)
V_WINNERS_JSON_DIR = Path("data/indicators_v2/winners")
V_BOARDS_NPZ_DIR = Path("data/indicators_v2/boards")
V_VIDEOS = ["29", "33"]


def is_dead_from_grid(grid: np.ndarray) -> bool:
    """既存 Board.is_dead() と同一ロジック (grid 直接版、再走査なし)。"""
    c = int(grid[DEATH_ROW, DEATH_COL])
    return c != COLOR_EMPTY and c != COLOR_UNKNOWN


def load_side_grids(npz_paths: list[Path]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """side ("1P"/"2P") -> (t_sec配列, grids配列) を返す (複数npzを結合)。"""
    t_by_side: dict[str, list[np.ndarray]] = {"1P": [], "2P": []}
    g_by_side: dict[str, list[np.ndarray]] = {"1P": [], "2P": []}
    for p in npz_paths:
        if not p.exists():
            continue
        d = np.load(str(p), allow_pickle=True)
        side = d["side"]
        for s in ("1P", "2P"):
            mask = side == s
            t_by_side[s].append(d["t_sec"][mask])
            g_by_side[s].append(d["grids"][mask])
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in ("1P", "2P"):
        if not t_by_side[s]:
            out[s] = (np.array([]), np.zeros((0, 13, 6)))
            continue
        t_cat = np.concatenate(t_by_side[s])
        g_cat = np.concatenate(g_by_side[s])
        order = np.argsort(t_cat)
        out[s] = (t_cat[order], g_cat[order])
    return out


def find_last_snapshot_before(
    t_arr: np.ndarray, g_arr: np.ndarray, end_sec: float,
) -> tuple[float | None, np.ndarray | None]:
    """end_sec 以前で最も近いスナップショットを返す (t, grid)。無ければ (None, None)。"""
    idx = np.where(t_arr <= end_sec)[0]
    if idx.size == 0:
        return None, None
    last = idx[-1]
    return float(t_arr[last]), g_arr[last]


def crosscheck_video(
    games: list[dict], side_grids: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    """1動画分のクロスチェック結果を返す。"""
    n_total = len(games)
    n_checkable = 0
    n_match = 0
    n_mismatch = 0
    mismatches: list[dict] = []
    for g in games:
        winner = g.get("winner")
        end_sec = float(g["end_sec"])
        if winner is None:
            continue
        t1, grid1 = find_last_snapshot_before(*side_grids["1P"], end_sec=end_sec)
        t2, grid2 = find_last_snapshot_before(*side_grids["2P"], end_sec=end_sec)
        if grid1 is None or grid2 is None:
            continue
        staleness = max(end_sec - t1, end_sec - t2)
        if staleness > SNAPSHOT_STALENESS_LIMIT_SEC:
            continue
        dead1 = is_dead_from_grid(grid1)
        dead2 = is_dead_from_grid(grid2)
        # 窒息側が負け -> 勝者は窒息していない側。両方/どちらも窒息でなければ判定不能。
        if dead1 and not dead2:
            board_winner = "2P"
        elif dead2 and not dead1:
            board_winner = "1P"
        else:
            continue  # 判定不能 (両方 or どちらも窒息していない)
        n_checkable += 1
        if board_winner == winner:
            n_match += 1
        else:
            n_mismatch += 1
            mismatches.append({
                "game_abs_idx": g["game_abs_idx"], "end_sec": end_sec,
                "panel_winner": winner, "board_winner": board_winner,
                "t1": t1, "t2": t2,
            })
    return {
        "n_total": n_total,
        "n_checkable": n_checkable,
        "coverage": n_checkable / n_total if n_total else 0.0,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "match_rate": (n_match / n_checkable) if n_checkable else float("nan"),
        "mismatches": mismatches,
    }


def run_c_series() -> None:
    print("=== c系4本: 窒息盤面クロスチェック (panel-diff-mode 結果に対して) ===")
    for vid in C_VIDEOS:
        json_path = C_PANEL_DIFF_JSON_DIR / f"panel_diff_video_{vid}.json"
        npz_path = C_BOARDS_NPZ_DIR / f"{vid}.npz"
        if not json_path.exists():
            print(f"  [SKIP] {vid}: panel_diff JSON未生成 (先に本体スキャン完了待ち)")
            continue
        if not npz_path.exists():
            print(f"  [SKIP] {vid}: boards_lean_fixed npz なし")
            continue
        with json_path.open("r", encoding="utf-8") as fp:
            games = json.load(fp)["games"]
        side_grids = load_side_grids([npz_path])
        result = crosscheck_video(games, side_grids)
        print(
            f"  video_{vid}: 全{result['n_total']}試合  "
            f"対象内={result['n_checkable']}件 (カバレッジ{result['coverage']:.1%})  "
            f"一致={result['n_match']}  不一致={result['n_mismatch']}  "
            f"一致率={result['match_rate']:.1%}" if result['n_checkable'] else
            f"  video_{vid}: 全{result['n_total']}試合  対象内0件 (npzのt_sec範囲外)"
        )
        for m in result["mismatches"]:
            print(f"    [不一致] game={m['game_abs_idx']}  end_sec={m['end_sec']:.1f}  "
                  f"panel={m['panel_winner']}  board={m['board_winner']}  "
                  f"t1={m['t1']:.1f} t2={m['t2']:.1f}")
    print()


def run_v_regression() -> None:
    print("=== v系2本回帰: 窒息盤面クロスチェック (デフォルトモード既存結果に対して) ===")
    for vid in V_VIDEOS:
        json_path = V_WINNERS_JSON_DIR / f"video_{vid}.json"
        npz_paths = [
            V_BOARDS_NPZ_DIR / f"v{vid}.npz",
            V_BOARDS_NPZ_DIR / f"v{vid}_mid.npz",
        ]
        if not json_path.exists():
            print(f"  [SKIP] video_{vid}: winners JSONなし")
            continue
        with json_path.open("r", encoding="utf-8") as fp:
            games = json.load(fp)["games"]
        side_grids = load_side_grids(npz_paths)
        result = crosscheck_video(games, side_grids)
        print(
            f"  video_{vid}: 全{result['n_total']}試合  "
            f"対象内={result['n_checkable']}件 (カバレッジ{result['coverage']:.1%})  "
            f"一致={result['n_match']}  不一致={result['n_mismatch']}  "
            + (f"一致率={result['match_rate']:.1%}" if result['n_checkable'] else "一致率=N/A")
        )
        for m in result["mismatches"]:
            print(f"    [不一致] game={m['game_abs_idx']}  end_sec={m['end_sec']:.1f}  "
                  f"panel={m['panel_winner']}  board={m['board_winner']}  "
                  f"t1={m['t1']:.1f} t2={m['t2']:.1f}")
    print()


def main() -> int:
    run_c_series()
    run_v_regression()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
