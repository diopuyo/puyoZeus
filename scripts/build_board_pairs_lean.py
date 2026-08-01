"""boards_lean_fixed/*.npz (won ラベル内蔵) から 1P/2P ペア配列を構築する。

## 既存 build_board_pairs.py との違い
- labeled_win.csv 不要: npz 内蔵の won フィールドを直接使用。
- 入力: data/indicators_v2/boards_lean_fixed/cN.npz 群。
- 出力: board_pairs_fixed.npz。

## 出力 npz キー
    board_1p  : (M, 13, 6) int8
    board_2p  : (M, 13, 6) int8
    won       : (M,) float32  (1P 視点 1/0/NaN)
    video_id  : (M,) str
    t_sec     : (M,) float32  (1P 側の t_sec)
    game_idx  : (M,) int32

## CLI
    python -m scripts.build_board_pairs_lean \\
        --npz-dir data/indicators_v2/boards_lean_fixed \\
        --out data/indicators_v2/board_pairs_fixed.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 1P/2P を近傍マッチする最大時刻差 (秒) — build_board_pairs.py と同値
MAX_PAIR_T_DIFF_SEC: float = 2.0


class BoardPairResult(NamedTuple):
    """ペア化結果をまとめた NamedTuple。"""
    board_1p: np.ndarray   # (N, 13, 6) int8
    board_2p: np.ndarray   # (N, 13, 6) int8
    won: np.ndarray        # (N,) float32: 1P 視点 1/0/NaN
    video_id: np.ndarray   # (N,) str
    t_sec: np.ndarray      # (N,) float32: 1P 側 t_sec
    game_idx: np.ndarray   # (N,) int32


# 連鎖汚染フィルタ (2026-08-01)。外部正解の物差しで、認識誤りの全てが
# 「連鎖/相殺エフェクトの遷移瞬間の一時汚染」(≤1秒で自己修復) と判明した
# (memory project_yardstick_first_results_2026-07-31)。
# 相殺光は**相手の連鎖でも自陣に被る** (c15 f18294 で実証: 自陣は CHAIN を
# 経ずに光だけ被った) ため、両 side の chain_trigger_sec を統合して使う。
# 窓: 検知の CHAIN_TAINT_PRE_SEC 前 〜 CHAIN_TAINT_POST_SEC 後の行を除外。
# 検知は連鎖開始近く (掛け算表示) で、相殺光は連鎖終了時に出るため後ろを長く取る。
CHAIN_TAINT_PRE_SEC: float = 1.0
CHAIN_TAINT_POST_SEC: float = 5.0


def _load_npz_dir_lean(
    npz_dir: Path,
    exclude_chain_taint: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """boards_lean_fixed ディレクトリ内 npz を全読み込みし DataFrame + grids を返す。

    各 npz は won フィールドを内蔵しているため labeled_win.csv は不要。
    Returns:
        df: columns = [video_id, side, t_sec, game_idx, won, grid_idx]
        grids_all: (N_total, 13, 6) int8
    """
    records: list[dict] = []
    all_grids: list[np.ndarray] = []
    offset = 0
    n_tainted = 0
    n_total_rows = 0

    for npz_path in sorted(npz_dir.glob("*.npz")):
        d = np.load(str(npz_path), allow_pickle=True)
        grids: np.ndarray = d["grids"]  # (n, 13, 6) int8
        n = len(grids)
        if n == 0:
            continue
        all_grids.append(grids)
        video_ids = d["video_id"].tolist()
        sides = d["side"].tolist()
        t_secs = d["t_sec"].tolist()
        game_idxs = d["game_idx"].tolist()
        wons = d["won"].tolist()  # float32 (NaN 含む可)
        # 連鎖汚染フィルタ: 両 side の検知時刻を統合した除外窓を作る
        taint_windows: list[tuple[float, float]] = []
        if exclude_chain_taint and "chain_trigger_sec" in d.files:
            cts = np.asarray(d["chain_trigger_sec"]).astype(float)
            for ct in sorted(set(float(x) for x in cts if not np.isnan(x))):
                taint_windows.append(
                    (ct - CHAIN_TAINT_PRE_SEC, ct + CHAIN_TAINT_POST_SEC),
                )
        for i in range(n):
            n_total_rows += 1
            if taint_windows:
                t = float(t_secs[i])
                if any(lo <= t <= hi for lo, hi in taint_windows):
                    n_tainted += 1
                    continue  # 汚染窓内の行はペア構成から除外
            records.append({
                "video_id": str(video_ids[i]),
                "side": str(sides[i]),
                "t_sec": float(t_secs[i]),
                "game_idx": int(game_idxs[i]),
                "won": float(wons[i]),
                "grid_idx": offset + i,
            })
        offset += n

    if not all_grids:
        raise ValueError(f"有効な npz が見つかりません: {npz_dir}")

    grids_all = np.concatenate(all_grids, axis=0)
    df = pd.DataFrame(records)
    if exclude_chain_taint:
        pct = 100.0 * n_tainted / max(1, n_total_rows)
        print(f"[pairs] 連鎖汚染フィルタ: {n_tainted}/{n_total_rows} 行を除外 ({pct:.1f}%)")
    return df, grids_all


def _pair_within_group(
    df_1p: pd.DataFrame,
    df_2p: pd.DataFrame,
    grids_all: np.ndarray,
) -> list[dict]:
    """同一 (video_id, game_idx) 内で 1P/2P を近傍 t_sec でペア化する。

    won は 1P 側の値をそのまま採用 (2P 側 won = 1-won のはずだが 1P 正とする)。
    貪欲マッチ: 既存 _pair_within_group と同アルゴリズム。
    """
    pairs: list[dict] = []
    df1 = df_1p.sort_values("t_sec").reset_index(drop=True)
    df2 = df_2p.sort_values("t_sec").reset_index(drop=True)
    j = 0
    for _, r1 in df1.iterrows():
        t1 = float(r1["t_sec"])
        while j + 1 < len(df2):
            t_cur = float(df2.at[j, "t_sec"])
            t_nxt = float(df2.at[j + 1, "t_sec"])
            if abs(t_nxt - t1) < abs(t_cur - t1):
                j += 1
            else:
                break
        if j >= len(df2):
            continue
        t2 = float(df2.at[j, "t_sec"])
        if abs(t2 - t1) > MAX_PAIR_T_DIFF_SEC:
            continue
        # won は 1P 側の値を使用 (NaN も保持)
        won_val = float(r1["won"])
        g1 = grids_all[int(r1["grid_idx"])]
        g2 = grids_all[int(df2.at[j, "grid_idx"])]
        pairs.append({
            "board_1p": g1,
            "board_2p": g2,
            "won": won_val,
            "video_id": str(r1["video_id"]),
            "t_sec": t1,
            "game_idx": int(r1["game_idx"]),
        })
    return pairs


def build_pairs_lean(
    npz_dir: Path, exclude_chain_taint: bool = False,
) -> BoardPairResult:
    """boards_lean_fixed ディレクトリからペア配列を構築して返す。"""
    df, grids_all = _load_npz_dir_lean(
        npz_dir, exclude_chain_taint=exclude_chain_taint,
    )
    all_pairs: list[dict] = []

    for (vid, game), grp in df.groupby(["video_id", "game_idx"]):
        df_1p = grp[grp["side"] == "1P"]
        df_2p = grp[grp["side"] == "2P"]
        if df_1p.empty or df_2p.empty:
            continue
        pairs = _pair_within_group(df_1p, df_2p, grids_all)
        all_pairs.extend(pairs)

    if not all_pairs:
        empty: np.ndarray = np.zeros((0, 13, 6), dtype=np.int8)
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


def _print_summary(result: BoardPairResult, out_path: Path | None) -> None:
    """ペア総数・won 内訳・video 別内訳を stdout に表示する。"""
    n = len(result.board_1p)
    nan_mask = np.isnan(result.won)
    n_won1 = int(np.sum(result.won == 1.0))
    n_won0 = int(np.sum(result.won == 0.0))
    n_nan = int(np.sum(nan_mask))

    print(f"[build_board_pairs_lean] ペア総数: {n}")
    print(f"  won=1: {n_won1}, won=0: {n_won0}, won=NaN: {n_nan}")

    # video 別内訳
    vids, counts = np.unique(result.video_id, return_counts=True)
    print(f"  video 別ペア数 ({len(vids)} 動画):")
    for vid, cnt in zip(vids, counts):
        mask = result.video_id == vid
        w1 = int(np.sum(result.won[mask] == 1.0))
        w0 = int(np.sum(result.won[mask] == 0.0))
        wn = int(np.sum(np.isnan(result.won[mask])))
        print(f"    {vid}: {cnt} ペア  (won1={w1}, won0={w0}, NaN={wn})")

    if out_path is not None:
        print(f"  -> 保存先: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="boards_lean_fixed/*.npz から 1P/2P ペア npz を構築する"
    )
    parser.add_argument(
        "--npz-dir", type=Path,
        default=Path("data/indicators_v2/boards_lean_fixed"),
        help="boards_lean_fixed ディレクトリ (既定: data/indicators_v2/boards_lean_fixed)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/indicators_v2/board_pairs_fixed.npz"),
        help="出力 npz パス (既定: data/indicators_v2/board_pairs_fixed.npz)",
    )
    parser.add_argument(
        "--exclude-chain-taint", action="store_true", default=False,
        help="連鎖汚染フィルタ: 両sideの連鎖検知の-1〜+5秒の行を除外 "
             "(2026-08-01。認識誤りは全て遷移瞬間の一時汚染と実証済み)。",
    )
    args = parser.parse_args()

    result = build_pairs_lean(
        args.npz_dir, exclude_chain_taint=args.exclude_chain_taint,
    )
    _print_summary(result, args.out if len(result.board_1p) > 0 else None)

    if len(result.board_1p) > 0:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
