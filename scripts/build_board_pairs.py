"""board npz 群 + labeled_win.csv からペア配列を構築する軽量ヘルパ。

## 処理概要
1. npz ファイル群を読み込み、(video_id, game_idx, t_sec, side, grid) を取得。
2. labeled_win.csv を読み込み、(video_id, game_idx, t_sec, side, won) を取得。
3. 同一 (video_id, game_idx) 内で 1P/2P を近傍時刻でペア化:
   - 1P と 2P を t_sec でソートし、絶対時刻差が MAX_PAIR_T_DIFF_SEC 以内なら対応付け。
4. 対応する won ラベルを labeled_win.csv から付与 (won は 1P 側の値 0/1 に統一)。
5. ペア配列 (board_1p, board_2p, won, video_id, t_sec, game_idx) を返す。

## 使い方
    python -m scripts.build_board_pairs \\
        --npz-dir data/indicators_v2/boards \\
        --labeled-win data/indicators_v2/study/labeled_win.csv \\
        --out data/indicators_v2/board_pairs.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# プロジェクトルートを import path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 1P/2P を近傍マッチする最大時刻差 (秒)
MAX_PAIR_T_DIFF_SEC: float = 2.0

# npz 内のグリッド shape
EXPECTED_GRID_SHAPE: tuple[int, int] = (13, 6)


class BoardPairResult(NamedTuple):
    """ペア化結果をまとめた NamedTuple。"""
    board_1p: np.ndarray   # (N, 13, 6) int8
    board_2p: np.ndarray   # (N, 13, 6) int8
    won: np.ndarray        # (N,) float32: 1P 視点の won (0/1/NaN)
    video_id: np.ndarray   # (N,) str
    t_sec: np.ndarray      # (N,) float32: 1P 側の t_sec
    game_idx: np.ndarray   # (N,) int32


def load_npz_dir(npz_dir: Path) -> pd.DataFrame:
    """ディレクトリ内の npz を全て読み込み、DataFrame に結合して返す。

    Returns:
        columns: video_id, side, t_sec, game_idx, frame_idx, grid_idx
        ただし grid_idx はグローバル通し番号 (後でグリッド配列を参照するため)。
    """
    records: list[dict] = []
    all_grids: list[np.ndarray] = []
    grid_offset = 0

    for npz_path in sorted(npz_dir.glob("*.npz")):
        data = np.load(str(npz_path), allow_pickle=True)
        grids: np.ndarray = data["grids"]  # (n, 13, 6) int8
        n = len(grids)
        if n == 0:
            continue
        video_ids = data["video_id"].tolist()
        sides = data["side"].tolist()
        t_secs = data["t_sec"].tolist()
        game_idxs = data["game_idx"].tolist()
        frame_idxs = data["frame_idx"].tolist()
        all_grids.append(grids)
        for i in range(n):
            records.append({
                "video_id": str(video_ids[i]),
                "side": str(sides[i]),
                "t_sec": float(t_secs[i]),
                "game_idx": int(game_idxs[i]),
                "frame_idx": int(frame_idxs[i]),
                "grid_idx": grid_offset + i,
            })
        grid_offset += n

    if not all_grids:
        raise ValueError(f"npz_dir に有効なファイルが見つかりません: {npz_dir}")

    grids_all = np.concatenate(all_grids, axis=0)  # (N_total, 13, 6)
    df = pd.DataFrame(records)
    return df, grids_all


def _pair_within_group(
    df_1p: pd.DataFrame,
    df_2p: pd.DataFrame,
    grids_all: np.ndarray,
    won_map: dict[tuple, float],
) -> list[dict]:
    """同一 (video_id, game_idx) 内で 1P/2P を近傍 t_sec でペア化する。

    won_map: (video_id, side, t_sec_rounded) -> won (0/1/NaN)
    """
    pairs: list[dict] = []
    df_1p_sorted = df_1p.sort_values("t_sec").reset_index(drop=True)
    df_2p_sorted = df_2p.sort_values("t_sec").reset_index(drop=True)
    j = 0  # 2P 側ポインタ (単調増加で走査)
    for _, r1 in df_1p_sorted.iterrows():
        t1 = float(r1["t_sec"])
        # j を t1 に最も近い 2P 行へ進める
        while j + 1 < len(df_2p_sorted):
            t_cur = float(df_2p_sorted.at[j, "t_sec"])
            t_nxt = float(df_2p_sorted.at[j + 1, "t_sec"])
            if abs(t_nxt - t1) < abs(t_cur - t1):
                j += 1
            else:
                break
        if j >= len(df_2p_sorted):
            continue
        t2 = float(df_2p_sorted.at[j, "t_sec"])
        if abs(t2 - t1) > MAX_PAIR_T_DIFF_SEC:
            continue
        # won ラベル取得 (1P 視点)
        vid = str(r1["video_id"])
        game = int(r1["game_idx"])
        won_1p = won_map.get((vid, "1P", round(t1, 3)), float("nan"))
        g_1p = grids_all[int(r1["grid_idx"])]
        g_2p = grids_all[int(df_2p_sorted.at[j, "grid_idx"])]
        pairs.append({
            "board_1p": g_1p,
            "board_2p": g_2p,
            "won": won_1p,
            "video_id": vid,
            "t_sec": t1,
            "game_idx": game,
        })
    return pairs


def build_pairs(
    npz_dir: Path,
    labeled_win_csv: Path,
) -> BoardPairResult:
    """ペア配列を構築して返す。

    Args:
        npz_dir: npz ファイルのディレクトリ (collect_indicators_v2 --board-npz 出力先)。
        labeled_win_csv: label_win_from_winners.py 出力 (won 列付き CSV)。

    Returns:
        BoardPairResult (NamedTuple)。
    """
    df_npz, grids_all = load_npz_dir(npz_dir)
    # won_map: (video_id, side, t_sec_rounded) -> won
    # labeled_win.csv を読み込む。空ファイル時は空 DataFrame として扱う
    try:
        df_win = pd.read_csv(str(labeled_win_csv))
    except pd.errors.EmptyDataError:
        df_win = pd.DataFrame(columns=["video_id", "side", "t_sec", "won"])
    won_map: dict[tuple, float] = {}
    for _, r in df_win.iterrows():
        key = (str(r["video_id"]), str(r["side"]), round(float(r["t_sec"]), 3))
        won_val = float(r["won"]) if not pd.isna(r.get("won", float("nan"))) else float("nan")
        won_map[key] = won_val

    all_pairs: list[dict] = []
    for (vid, game), grp in df_npz.groupby(["video_id", "game_idx"]):
        df_1p = grp[grp["side"] == "1P"]
        df_2p = grp[grp["side"] == "2P"]
        if df_1p.empty or df_2p.empty:
            continue
        pairs = _pair_within_group(df_1p, df_2p, grids_all, won_map)
        all_pairs.extend(pairs)

    if not all_pairs:
        # 空のペア結果を返す
        empty = np.zeros((0, 13, 6), dtype=np.int8)
        return BoardPairResult(
            board_1p=empty, board_2p=empty,
            won=np.array([], dtype=np.float32),
            video_id=np.array([]), t_sec=np.array([], dtype=np.float32),
            game_idx=np.array([], dtype=np.int32),
        )

    board_1p = np.array([p["board_1p"] for p in all_pairs], dtype=np.int8)
    board_2p = np.array([p["board_2p"] for p in all_pairs], dtype=np.int8)
    won = np.array([p["won"] for p in all_pairs], dtype=np.float32)
    video_id = np.array([p["video_id"] for p in all_pairs])
    t_sec = np.array([p["t_sec"] for p in all_pairs], dtype=np.float32)
    game_idx = np.array([p["game_idx"] for p in all_pairs], dtype=np.int32)
    return BoardPairResult(
        board_1p=board_1p, board_2p=board_2p, won=won,
        video_id=video_id, t_sec=t_sec, game_idx=game_idx,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="board npz + labeled_win からペア配列を構築")
    parser.add_argument(
        "--npz-dir", type=Path, required=True,
        help="npz ファイルが格納されたディレクトリ",
    )
    parser.add_argument(
        "--labeled-win", type=Path, required=True,
        help="labeled_win.csv のパス (won 列付き)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="出力 npz パス (省略時は保存しない。件数確認のみ)",
    )
    args = parser.parse_args()

    result = build_pairs(args.npz_dir, args.labeled_win)
    n = len(result.board_1p)
    n_labeled = int(np.sum(~np.isnan(result.won)))
    n_won1 = int(np.sum(result.won == 1.0))
    n_won0 = int(np.sum(result.won == 0.0))
    print(f"[build_board_pairs] ペア総数: {n}")
    print(f"  won ラベル付き: {n_labeled} (won=1: {n_won1}, won=0: {n_won0})")

    if args.out is not None and n > 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(args.out),
            board_1p=result.board_1p,
            board_2p=result.board_2p,
            won=result.won,
            video_id=result.video_id,
            t_sec=result.t_sec,
            game_idx=result.game_idx,
        )
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
