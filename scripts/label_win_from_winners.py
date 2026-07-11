"""
study CSV の各スナップショットに勝敗ラベル (won) を付与する。

## 手法
1. data/indicators_v2/winners/video_NN.json (extract_match_winners.py の出力) を読み込む
2. study CSV の各スナップショットの (video_id, game_idx, t_sec) を JSON の
   試合区間 [start_sec, end_sec) にマップ
3. 対応する試合の winner で won を付与
   - 1P のスナップショット: winner == "1P" なら won=1、"2P" なら won=0
   - 2P のスナップショット: winner == "2P" なら won=1、"1P" なら won=0
4. winner が None / 区間に属さない → won=NaN (除外対象)

## 注意
- study CSV は 先頭5分/中盤6分 窓抜きなので窓内完結試合のみラベル可能
- mid CSV は t_sec が 1200s 以降。JSON の start_sec/end_sec と直接照合
- game_idx は同一動画内の窓内相対インデックス（ JSON のgame_abs_idx とは異なる可能性）
  → t_sec ベースの区間マッチングで対応

## 出力
    data/indicators_v2/study/labeled_win.csv
      (全study行 + won列、ラベル不能は NaN)

## 使い方
    python -m scripts.label_win_from_winners \
        --study data/indicators_v2/study \
        --winners-dir data/indicators_v2/winners \
        --out data/indicators_v2/study/labeled_win.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ラベル付き CSV の出力デフォルトパス
DEFAULT_OUT_PATH: str = "data/indicators_v2/study/labeled_win.csv"

# winners ディレクトリ
DEFAULT_WINNERS_DIR: str = "data/indicators_v2/winners"

# study ディレクトリ
DEFAULT_STUDY_DIR: str = "data/indicators_v2/study"


def load_winners(winners_dir: Path, video_ids: list[str]) -> dict[str, list[dict]]:
    """video_id -> ゲームリスト のマッピングを返す。"""
    winners_map: dict[str, list[dict]] = {}
    for vid in video_ids:
        json_path = winners_dir / f"{vid}.json"
        if not json_path.exists():
            print(f"  [WARN] winners JSON なし: {json_path} -> {vid} はラベル付与スキップ")
            winners_map[vid] = []
            continue
        with json_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        winners_map[vid] = data.get("games", [])
        print(f"  {vid}: {len(winners_map[vid])} 試合を読み込み")
    return winners_map


def find_winner_for_t(
    games: list[dict],
    t_sec: float,
) -> str | None:
    """
    t_sec が属する試合区間を探して winner を返す。

    JSON の start_sec, end_sec は "playing 開始" を使っているので
    [start_sec, end_sec) の半開区間で判定する。
    """
    for game in games:
        s = float(game["start_sec"])
        e = float(game["end_sec"])
        if s <= t_sec < e:
            return game.get("winner")  # "1P" / "2P" / None
    return None  # 区間外


def _won_for_side(winner: str | None, side: str) -> float:
    """winner と side から won (0/1) を返す。区間外・None は NaN。"""
    if winner is None:
        return float("nan")
    if side == "1P":
        return 1.0 if winner == "1P" else 0.0
    elif side == "2P":
        return 1.0 if winner == "2P" else 0.0
    return float("nan")


def attach_won_labels(
    df: pd.DataFrame,
    winners_map: dict[str, list[dict]],
) -> pd.DataFrame:
    """DataFrame に won 列を追加して返す。"""
    won_values: list[float] = []
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        t = float(row["t_sec"])
        side = str(row["side"])
        games = winners_map.get(vid, [])
        winner = find_winner_for_t(games, t)
        won_values.append(_won_for_side(winner, side))
    df = df.copy()
    df["won"] = won_values
    return df


def print_coverage_report(df: pd.DataFrame) -> None:
    """ラベル付与カバレッジを動画別・全体で表示する。"""
    total = len(df)
    labeled = df["won"].notna().sum()
    print()
    print("=== ラベル付与カバレッジ ===")
    print(f"  全行: {total}  ラベル付き: {labeled}  "
          f"カバレッジ: {labeled/total:.1%}")

    # 動画別
    print()
    print(f"  {'video_id':<12}  {'全行':>6}  {'ラベル付き':>9}  {'カバレッジ':>9}  {'won=1':>6}  {'won=0':>6}")
    print("  " + "-" * 62)
    for vid, g in df.groupby("video_id"):
        n_total = len(g)
        n_labeled = g["won"].notna().sum()
        n_won1 = (g["won"] == 1).sum()
        n_won0 = (g["won"] == 0).sum()
        cov = n_labeled / n_total if n_total > 0 else 0.0
        print(f"  {vid:<12}  {n_total:>6}  {n_labeled:>9}  {cov:>9.1%}  {n_won1:>6}  {n_won0:>6}")

    print()
    # won 分布 (ラベル付きのみ)
    df_labeled = df[df["won"].notna()]
    n1 = (df_labeled["won"] == 1.0).sum()
    n0 = (df_labeled["won"] == 0.0).sum()
    print(f"  won=1 (勝ち側): {n1}  won=0 (負け側): {n0}  "
          f"バランス: {n1/(n1+n0):.1%}" if (n1 + n0) > 0 else "  データなし")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="study CSV への勝敗ラベル付与"
    )
    parser.add_argument(
        "--study", default=DEFAULT_STUDY_DIR,
        help=f"study ディレクトリ (デフォルト: {DEFAULT_STUDY_DIR})",
    )
    parser.add_argument(
        "--winners-dir", default=DEFAULT_WINNERS_DIR,
        help=f"winners JSON ディレクトリ (デフォルト: {DEFAULT_WINNERS_DIR})",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT_PATH,
        help=f"出力 CSV (デフォルト: {DEFAULT_OUT_PATH})",
    )
    args = parser.parse_args()

    study_dir = Path(args.study)
    winners_dir = Path(args.winners_dir)
    out_path = Path(args.out)

    # 1. study CSV を読み込む
    csv_paths = sorted(
        p for p in study_dir.glob("*.csv")
        if not p.name.startswith("corr_") and not p.name.startswith("labeled_")
    )
    if not csv_paths:
        print(f"[ERROR] study CSV が存在しない: {study_dir}", file=sys.stderr)
        return 1

    dfs: list[pd.DataFrame] = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df = df.dropna(subset=["video_id", "side"])
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"[label_win] study CSV 読み込み: {len(csv_paths)} ファイル, {len(combined)} 行")

    # 2. winners JSON を読み込む
    video_ids = sorted(combined["video_id"].unique())
    print(f"[label_win] winners JSON 読み込み (動画 {len(video_ids)} 本):")
    winners_map = load_winners(winners_dir, video_ids)

    # 3. ラベル付与
    print("[label_win] ラベル付与中 ...")
    result_df = attach_won_labels(combined, winners_map)

    # 4. カバレッジ報告
    print_coverage_report(result_df)

    # 5. 出力
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(str(out_path), index=False)
    print(f"\n  出力: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
