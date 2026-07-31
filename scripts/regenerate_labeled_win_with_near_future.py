"""labeled_win.csv に near_future_fire_k1-5 + fire_stability_k2/4/6 列を追加する (#33)。

## 背景

near_future_fire_power / fire_stability は
scripts/visualize_advantage_overlay.py の FEATURE_CANDIDATES に追加済み
(2026-07-22 統合、PR#19) だが、実際に学習・推論で使われるには
labeled_win.csv (data/indicators_v2/study/labeled_win.csv) 側に列が
存在する必要がある (_resolve_features() の列存在ガード)。列が無い間は
自動的に除外され、オーバーレイにも win-model にも一切効いていなかった。
本スクリプトは列を追加し、下流を自動有効化する。

## スコープ

- **expected_fire は含めない** (COLLECT_EXPECTED_FIRE=False のまま、
  1.7-3.5秒/盤面と重いため。user指示通り)。
- near_future_fire_k1..k5 (score+raw) と fire_stability_k2/4/6 (score+raw)
  のみ追加する。
- 対象行: video_29-38 のうち盤面キャッシュ (data/indicators_v2/boards/*.npz)
  が存在する動画のみ計算可能 (既知の制約、実測4動画=v29/35/36/37)。
  それ以外の行・動画は新列が空欄 (NaN) のままになる
  (列存在ガード配下では fillna(0.0) で中立diffとして扱われる設計、
  scripts/visualize_advantage_overlay.py._train_model 参照)。
- 突合方式は scripts/_tmp_near_future_gen.py / scripts/_tmp_ama_ceiling_gen.py
  と同じ (_match_grid, _compute_active_colors_by_game)。elapsed_sec=0.0
  (先行の win-AUC 検証と同じ簡略化、マージン減衰なし)。next_pair/dnext_pair
  は boards/*.npz に含まれないため常に None (理想ツモ代用、先行検証と同条件)。
- 既存の他列・他行は一切変更しない (新列の追加のみ)。元ファイルはバック
  アップしてから同一パスに上書きする (下流が読むパスをそのまま更新する
  のが目的)。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts.regenerate_labeled_win_with_near_future
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._tmp_validate_build_ceiling_subset import (  # noqa: E402
    _load_npz_for_video, _match_grid, _grid_to_board, TARGET_VIDEO_IDS, BOARDS_DIR,
)
from scripts._tmp_ama_builder import _compute_active_colors_by_game  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402

LABELED_WIN_CSV = Path("data/indicators_v2/study/labeled_win.csv")
BACKUP_DIR = Path("data/indicators_v2/study/backup")

# 追加する新列 (near_future_fire_k1..k5 + fire_stability_k2/4/6、score+raw)。
NEAR_FUTURE_NEW_COLUMNS: "list[str]" = []
for _k in iv.NEAR_FUTURE_K_LEVELS:
    NEAR_FUTURE_NEW_COLUMNS.append(f"near_future_fire_k{_k}")
    NEAR_FUTURE_NEW_COLUMNS.append(f"near_future_fire_k{_k}_raw")
for _k in iv.FIRE_STABILITY_K_LEVELS:
    NEAR_FUTURE_NEW_COLUMNS.append(f"fire_stability_k{_k}")
    NEAR_FUTURE_NEW_COLUMNS.append(f"fire_stability_k{_k}_raw")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _backup_original(path: Path) -> Path:
    """元ファイルをタイムスタンプ付きでバックアップする (上書き前の安全策)。"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"labeled_win_before_near_future_{stamp}.csv"
    shutil.copy2(path, backup_path)
    logger.info("バックアップ作成: %s", backup_path)
    return backup_path


def _compute_indicators_for_matched_rows(df: pd.DataFrame) -> "dict[int, dict[str, float]]":
    """盤面照合できた行 (index -> 新列値dict) を計算する。

    盤面キャッシュ (data/indicators_v2/boards/*.npz) が存在する動画のみ
    照合可能 (既知の制約)。next_pair/dnext_pair は npz に含まれないため
    常に None (理想ツモ代用、先行 win-AUC 検証と同条件)。
    """
    available_videos = [
        vid for vid in TARGET_VIDEO_IDS
        if (BOARDS_DIR / f"{vid.replace('video_', 'v')}.npz").exists()
    ]
    logger.info("盤面キャッシュが存在する動画: %s", available_videos)

    npz_cache = {vid: _load_npz_for_video(vid) for vid in available_videos}
    active_colors_cache: "dict[tuple[str, int], tuple[int, ...]]" = {}
    for vid in available_videos:
        stem = vid.replace("video_", "v")
        active_colors_cache.update(_compute_active_colors_by_game(BOARDS_DIR / f"{stem}.npz"))

    results: "dict[int, dict[str, float]]" = {}
    n_matched = 0
    n_missed = 0
    n_missed_colors = 0
    t_start = time.time()
    total = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        vid = str(row["video_id"])
        if vid not in npz_cache:
            n_missed += 1
            continue
        side = str(row["side"])
        game_idx = int(row["game_idx"])
        t_sec = float(row["t_sec"])
        grid = _match_grid(npz_cache[vid], side, game_idx, t_sec)
        if grid is None:
            n_missed += 1
            continue
        colors = active_colors_cache.get((vid, game_idx))
        if colors is None:
            n_missed_colors += 1
            continue
        board = _grid_to_board(grid)

        nf = iv.near_future_fire_power(board, elapsed_sec=0.0, active_colors=colors)
        fs = iv.fire_stability(board, active_colors=colors)

        row_values: "dict[str, float]" = {}
        for k in iv.NEAR_FUTURE_K_LEVELS:
            row_values[f"near_future_fire_k{k}"] = nf.values[k].score
            row_values[f"near_future_fire_k{k}_raw"] = nf.values[k].raw
        for k in iv.FIRE_STABILITY_K_LEVELS:
            row_values[f"fire_stability_k{k}"] = fs.values[k].score
            row_values[f"fire_stability_k{k}_raw"] = fs.values[k].raw
        results[idx] = row_values
        n_matched += 1

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            logger.info(
                "進捗: %d/%d matched=%d missed=%d missed_colors=%d elapsed=%.0fs eta=%.0fs",
                i + 1, total, n_matched, n_missed, n_missed_colors, elapsed, eta,
            )

    logger.info(
        "照合完了: matched=%d missed=%d missed_colors=%d (total=%d)",
        n_matched, n_missed, n_missed_colors, total,
    )
    return results


def _apply_new_columns(df: pd.DataFrame, results: "dict[int, dict[str, float]]") -> pd.DataFrame:
    """新列を df に追加する (未マッチ行は NaN のまま、既存列・既存行は変更しない)。"""
    for col in NEAR_FUTURE_NEW_COLUMNS:
        df[col] = np.nan
    for idx, values in results.items():
        for col, val in values.items():
            df.at[idx, col] = val
    return df


def main() -> int:
    logger.info("=== labeled_win.csv に near_future_fire/fire_stability 列を追加 (#33) ===")
    df = pd.read_csv(LABELED_WIN_CSV)
    logger.info("読み込み: %d 行 %d 列 (video: %s)", len(df), len(df.columns), sorted(df["video_id"].unique()))

    _backup_original(LABELED_WIN_CSV)

    results = _compute_indicators_for_matched_rows(df)
    df = _apply_new_columns(df, results)

    matched_videos = sorted({str(df.loc[idx, "video_id"]) for idx in results})
    logger.info(
        "新列が埋まった行数: %d / %d (%.1f%%)、対象動画: %s",
        len(results), len(df), 100.0 * len(results) / len(df), matched_videos,
    )

    df.to_csv(LABELED_WIN_CSV, index=False)
    logger.info("上書き保存完了: %s (%d行 %d列)", LABELED_WIN_CSV, len(df), len(df.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
