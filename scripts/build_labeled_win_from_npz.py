"""boards_lean 系 npz (盤面グリッド原情報) → labeled_win 形式 CSV 変換ツール (MVP)。

## 背景 (2026-08-12 user選択肢C確定)
CSV は使い捨ての派生物、npz (boards_lean_phase_l_2026-08-11/*.npz 等) が
恒久資産という前提で設計する。指標セットが増減しても npz 群を再収集せず
(動画も不要)、本ツールで CSV を安く再生成できるようにする。

## 薄い委譲構造 (指標が増減してもこのファイルは触らない)
指標の実計算は 100% `src/indicators_v2.py` の関数群に委譲する。本ツールが
持つのは「npz の 1 行 → Board 再構築 → レジストリ内の関数を呼んで列を書く」
という薄いループだけ。**新指標を追加したい場合は `GRID_ONLY_INDICATORS`
(または `GRID_ONLY_HEAVY_INDICATORS`) に 1 行足すだけで済む** (INDICATOR_
COLUMNS 末尾追加ルールと同じ精神)。center_bulge は他の grid-only 指標と
全く同じ扱いで、このレジストリに登録されているだけで乗る (特別扱い不要)。

## 現状カバー範囲 (正直な記録、詳細は調査報告参照)
- npz に入っている原情報: grids / video_id / side / t_sec / game_idx /
  frame_idx / won / score / next1_a,b / dnext_a,b (--with-next 収集時) /
  chain_trigger_sec / chain_mechanism (--enable-chain-tracker 収集時)。
- **本 MVP が計算するのは「盤面グリッドのみ」から求まる指標のみ**
  (GRID_ONLY_INDICATORS / GRID_ONLY_HEAVY_INDICATORS)。
- **未対応 (既知のギャップ、意図的に列を出力しない)**:
  ojama_net_balance / ojama_forecast — OjamaAccountingTracker は
  「毎フレームの BoardState 遷移 + tsumo_settled タイミング」を要求するが
  npz は STABLE 重複除去済みスナップショットのみで、この遷移列を保持して
  いない。score 列から近似復元する経路は別途検討 (調査報告参照)。
  列を出力しないことで `_resolve_features()` の列存在ガードが自動的に
  除外する (既存の「未収集列」と同じ扱い、後方互換)。
- next_pair 依存指標 (near_future_fire_power 等) は本 MVP では未実装
  (next1_a/b はレジストリの外で読み取り可能、拡張ポイントとしてコメントで
  明示するに留める)。

## 使い方
    python -m scripts.build_labeled_win_from_npz \\
        --npz-dir data/indicators_v2/boards_lean_phase_l_2026-08-11 \\
        --out data/indicators_v2/study/labeled_win_from_npz_2026-08-12.csv \\
        --profile light

--profile light: sub-ms 指標のみ (高速、反復開発向け)。
--profile full : current_max_chain 等の重い連鎖シミュ系も含む (低速)。
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board import Board  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402

# ============================
# 指標レジストリ (薄い委譲構造の核心)
# ============================
# 値は IndicatorV2Value を返す Board -> value の関数のみ (score/raw の2列を
# 自動で書く)。新指標を増やす/減らす場合はこの2つの dict を編集するだけで
# 良く、変換ループ本体 (_compute_row) は触らない。

# light: 実測 <0.15ms/行 (ベンチ scripts/_verify... 2026-08-12、n=500)
GRID_ONLY_INDICATORS: dict[str, Callable[[Board], "iv.IndicatorV2Value"]] = {
    "board_color_puyo_total": iv.board_color_puyo_total,
    "board_puyo_total": iv.board_puyo_total,
    "max_column_height": iv.max_column_height,
    "column_bumpiness": iv.column_bumpiness,
    "death_margin": iv.death_margin,
    "death_margin_neighbor": iv.death_margin_neighbor,
    "center_bulge": iv.center_bulge,
    "board_ojama_count": iv.board_ojama_count,
}
# full: 連鎖シミュレーションを要する重い指標 (実測 1〜19ms/行)。
# --profile full 指定時のみ計算する。
GRID_ONLY_HEAVY_INDICATORS: dict[str, Callable[[Board], "iv.IndicatorV2Value"]] = {
    "current_max_chain": iv.current_max_chain,
    "dig_resistance": iv.dig_resistance,
    "ukeyasusa": iv.ukeyasusa,
    "sub_chain_count": iv.sub_chain_count,
    "saturated_chain_count": iv.saturated_chain_count,
}

VALID_PROFILES: tuple[str, ...] = ("light", "full")


def _resolve_indicator_registry(
    profile: str,
) -> dict[str, Callable[[Board], "iv.IndicatorV2Value"]]:
    """profile に応じて使う指標レジストリを確定する (light はheavy除外)。"""
    if profile == "full":
        return {**GRID_ONLY_INDICATORS, **GRID_ONLY_HEAVY_INDICATORS}
    return dict(GRID_ONLY_INDICATORS)


# CSV メタ列 (labeled_win.csv 既存フォーマットと同じ、tsumo は近似値)
META_COLUMNS: tuple[str, ...] = (
    "video_id", "game_idx", "t_sec", "frame", "tsumo", "side", "won",
)


def _compute_row(
    grid: np.ndarray,
    registry: dict[str, Callable[[Board], "iv.IndicatorV2Value"]],
) -> dict[str, float]:
    """1 スナップショットの grid からレジストリ内の全指標を計算する。

    Board 再構築を 1 回だけ行い、レジストリの全関数に共有する (npz→Board
    復元コストの重複を避ける)。connectivity_observation (score を持たない
    タプル戻り値) は個別に追加する (collect_indicators_v2.py と同じ流儀)。
    """
    board = Board.from_list(grid.tolist())
    row: dict[str, float] = {}
    for name, fn in registry.items():
        v = fn(board)
        row[name] = v.score
        row[f"{name}_raw"] = v.raw
    total_conn, _ = iv.connectivity_observation(board)
    row["conn_pair_count"] = float(total_conn.pair_count)
    row["conn_triple_count"] = float(total_conn.triple_count)
    row["conn_max_group_size"] = float(total_conn.max_group_size)
    return row


def _approx_tsumo(rows_meta: list[dict]) -> None:
    """(video_id, side, game_idx) 内の t_sec 順位で手数を近似する (in-place)。

    実際の tsumo_count (RecognitionPipeline.tsumo_count) とは異なり得る近似
    値 (STABLE スナップショット数 != 常に手数、ojama落下等でも新スナップ
    ショットが生まれるため)。現行 FEATURES/FEATURE_CANDIDATES はどの列も
    tsumo に依存しないため、メタ情報としての近似で実害はない
    (model_indicator_win.META_COLS が特徴量から除外する対象と同じ扱い)。
    """
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows_meta):
        key = (r["video_id"], r["side"], r["game_idx"])
        groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        idxs.sort(key=lambda i: rows_meta[i]["t_sec"])
        for rank, i in enumerate(idxs):
            rows_meta[i]["tsumo"] = rank


def convert_one_npz(
    npz_path: Path, registry: dict[str, Callable[[Board], "iv.IndicatorV2Value"]],
) -> list[dict]:
    """1 npz ファイルを labeled_win 形式の行リストに変換する。

    npz は zlib 圧縮 (np.savez_compressed) のため、`d[key]` は毎回アーカイブ
    メンバを再展開するコストがある。ループの外で各配列を 1 回だけ取り出す
    (2026-08-12 実測: これを怠ると 1本 6693行で 99.8s、修正後は同じ内容が
    数秒で終わる。原情報を安く再変換できることが選択肢C成立の前提のため
    この最適化は必須)。
    """
    d = np.load(str(npz_path), allow_pickle=True)
    grids = d["grids"]
    video_ids = d["video_id"]
    game_idxs = d["game_idx"]
    t_secs = d["t_sec"]
    frame_idxs = d["frame_idx"]
    sides = d["side"]
    wons = d["won"]
    n = len(grids)
    rows: list[dict] = []
    for i in range(n):
        meta = {
            "video_id": str(video_ids[i]),
            "game_idx": int(game_idxs[i]),
            "t_sec": float(t_secs[i]),
            "frame": int(frame_idxs[i]),
            "side": str(sides[i]),
            "won": float(wons[i]),
        }
        rows.append(meta)
    _approx_tsumo(rows)
    for i in range(n):
        rows[i].update(_compute_row(grids[i], registry))
    return rows


def convert_dir(
    npz_dir: Path, out_csv: Path, profile: str = "light",
) -> tuple[int, float]:
    """npz_dir 内の全 npz を変換し out_csv に書き出す。

    Returns:
        (書き出し行数, 所要秒数)。
    """
    registry = _resolve_indicator_registry(profile)
    t0 = time.time()
    all_rows: list[dict] = []
    npz_files = sorted(npz_dir.glob("*.npz"))
    for i, p in enumerate(npz_files):
        rows = convert_one_npz(p, registry)
        all_rows.extend(rows)
        print(f"[{i+1}/{len(npz_files)}] {p.name}: {len(rows)} rows "
              f"(累計 {len(all_rows)}, {time.time()-t0:.1f}s)")
    if not all_rows:
        print("[WARN] 変換対象行が0件でした")
        return 0, time.time() - t0
    indicator_cols: list[str] = []
    for name in registry:
        indicator_cols.extend([name, f"{name}_raw"])
    indicator_cols.extend(["conn_pair_count", "conn_triple_count", "conn_max_group_size"])
    fieldnames = list(META_COLUMNS) + indicator_cols
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    elapsed = time.time() - t0
    print(f"[done] {len(all_rows)} 行 -> {out_csv} ({elapsed:.1f}s, "
          f"{len(npz_files)}本, profile={profile})")
    return len(all_rows), elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", choices=VALID_PROFILES, default="light")
    a = ap.parse_args()
    convert_dir(a.npz_dir, a.out, profile=a.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
