"""本番実装 (src/indicators_v2.py の near_future_fire_power) の win-AUC 再検証 (v2)。

コーディネータ指示 (2026-07-22、stateless修正後の再検証): active_colors を
外部 (プロトと同じ _compute_active_colors_by_game、試合全体の色頻度) から
明示的に渡した場合に、プロトの数値 (中盤K5=0.776等) が回復するかを確認する。

v1 (このファイルの前バージョン、実行ログは近似の食い違いを特定済み) との違い:
    v1 は active_colors 省略 (=near_future_fire_power 内部の1盤面フォールバック
    _near_future_active_colors) で実行し、中盤で最大-0.11の乖離を検出した。
    本 v2 はその乖離の原因 (stateless近似) を修正すべく、near_future_fire_power
    に active_colors 引数を追加した (2026-07-22) 上で、本検証スクリプト側は
    プロトと全く同じ _compute_active_colors_by_game (試合全体・全フレームの
    色頻度から上位4色を採用) を計算して明示的に渡す。本検証はオフライン
    (全フレームが既にキャッシュ済み) のため、collect_indicators_v2.py の
    因果的 (未来を見ない) _GameColorTracker とは異なり、プロトと全く同じ
    「試合全体の頻度」を使える (=この検証が回復すれば、乖離の原因が
    active_colors 近似であったことの直接証明になる)。

比較の前提 (継続、正直な注記):
    - プロトは raw = 生得点 (お邪魔換算なし)。本番は既存火力系指標と同じ
      「お邪魔換算 (/72)」に変更した (意図的な設計変更、バグではない)。
      elapsed_sec=0.0 (マージン減衰なし、rate=70固定の単純割り算) を使い
      プロトの生得点と単調な関係を保って公平に比較する。

使い方:
    PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        ./venv/bin/python -m scripts._tmp_near_future_prod_verify
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._tmp_validate_build_ceiling_subset import (  # noqa: E402
    _load_npz_for_video, _match_grid, _grid_to_board, TARGET_VIDEO_IDS, BOARDS_DIR,
)
from scripts._tmp_ama_builder import _compute_active_colors_by_game  # noqa: E402
from scripts.model_indicator_win import (  # noqa: E402
    TSUMO_EARLY_RATIO, TSUMO_LATE_RATIO, pair_sides_for_win, build_features,
)
import src.indicators_v2 as iv  # noqa: E402

LABELED_WIN_CSV = Path("data/indicators_v2/study/labeled_win.csv")
OUT_CSV = Path("data/indicators_v2/study/near_future_prod_verify_result.csv")
MAX_TDIFF: float = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _generate() -> pd.DataFrame:
    """本番 iv.near_future_fire_power を matched 行に適用して CSV 化する。

    active_colors はプロトと同じ _compute_active_colors_by_game (試合全体の
    色頻度) を計算して明示的に渡す (stateless修正後の再検証、v2)。
    """
    df = pd.read_csv(LABELED_WIN_CSV)
    df = df[df["video_id"].isin(TARGET_VIDEO_IDS)]
    df = df[df["won"].notna()].reset_index(drop=True)
    logger.info("対象行数 (won付き): %d", len(df))

    npz_cache = {vid: _load_npz_for_video(vid) for vid in TARGET_VIDEO_IDS}
    active_colors_cache: "dict[tuple[str, int], tuple[int, ...]]" = {}
    for vid in TARGET_VIDEO_IDS:
        stem = vid.replace("video_", "v")
        active_colors_cache.update(_compute_active_colors_by_game(BOARDS_DIR / f"{stem}.npz"))
    logger.info("active_colors (試合単位) 計算完了: %d 組", len(active_colors_cache))

    rows_out: "list[dict]" = []
    n_matched = 0
    n_missed = 0
    n_missed_colors = 0
    t_start = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        vid = str(row["video_id"])
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
        t0 = time.perf_counter()
        # elapsed_sec=0.0: プロト (お邪魔換算なし) との公平な比較のため
        # マージン減衰を掛けない (docstring 参照)。active_colors はプロトと
        # 同じ試合全体頻度を明示的に渡す (stateless修正後の再検証)。
        nf = iv.near_future_fire_power(board, elapsed_sec=0.0, active_colors=colors)
        cost = time.perf_counter() - t0

        row_dict = row.to_dict()
        for k in iv.NEAR_FUTURE_K_LEVELS:
            row_dict[f"near_future_fire_k{k}_raw"] = nf.values[k].raw
        row_dict["near_future_cost_sec"] = cost
        rows_out.append(row_dict)
        n_matched += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            logger.info(
                "進捗: %d/%d matched=%d missed=%d missed_colors=%d elapsed=%.0fs",
                i + 1, len(df), n_matched, n_missed, n_missed_colors, elapsed,
            )

    logger.info("完了: matched=%d missed=%d missed_colors=%d", n_matched, n_missed, n_missed_colors)
    out_df = pd.DataFrame(rows_out)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    logger.info("結果 CSV 保存: %s", OUT_CSV)
    cost_ms = out_df["near_future_cost_sec"] * 1000.0
    logger.info(
        "cost: mean=%.1fms p95=%.1fms max=%.1fms", cost_ms.mean(), cost_ms.quantile(0.95),
        cost_ms.max(),
    )
    return out_df


def _phase_masks(paired: pd.DataFrame) -> "dict[str, np.ndarray]":
    tsumo = paired["tsumo_1p"].astype(float).values
    q33 = float(np.quantile(tsumo, TSUMO_EARLY_RATIO))
    q67 = float(np.quantile(tsumo, TSUMO_LATE_RATIO))
    return {
        "序盤": tsumo <= q33,
        "中盤": (tsumo > q33) & (tsumo <= q67),
        "終盤": tsumo > q67,
    }


def _diff_auc(
    paired: pd.DataFrame, feat_col: str, y: np.ndarray, mask: "np.ndarray | None" = None,
) -> "tuple[float, int]":
    feat = build_features(paired, [feat_col])
    diff_col = f"{feat_col}_diff"
    score = feat[diff_col].fillna(0.0).values
    yy = y
    if mask is not None:
        score, yy = score[mask], y[mask]
    if len(score) < 20 or len(np.unique(yy)) < 2:
        return float("nan"), len(score)
    auc = float(roc_auc_score(yy, score))
    auc = max(auc, 1.0 - auc)
    return auc, len(score)


def _verify(df: pd.DataFrame) -> None:
    df = df.dropna(subset=["video_id", "side", "won"]).copy()
    df["won"] = df["won"].astype(int)
    paired = pair_sides_for_win(df, MAX_TDIFF)
    y = paired["won_1p"].astype(int).values
    masks = _phase_masks(paired)

    print()
    print("=" * 100)
    print("  本番実装 (iv.near_future_fire_power) 再検証AUC vs プロト実測値 (単純diff-AUC)")
    print("=" * 100)
    # プロト実測値 (scripts/_tmp_near_future_auc_verify.py の出力、2026-07-22 記録済み)。
    proto_ref = {
        1: {"全体": 0.6718, "序盤": 0.6262, "中盤": 0.7233, "終盤": 0.6080},
        2: {"全体": 0.6848, "序盤": 0.6240, "中盤": 0.7348, "終盤": 0.6198},
        3: {"全体": 0.7063, "序盤": 0.6282, "中盤": 0.7632, "終盤": 0.6274},
        4: {"全体": 0.7087, "序盤": 0.6204, "中盤": 0.7682, "終盤": 0.6340},
        5: {"全体": 0.7160, "序盤": 0.6479, "中盤": 0.7756, "終盤": 0.6424},
    }
    header = f"  {'K':<4}  {'区分':<6}  {'本番AUC':>8}  {'プロトAUC':>9}  {'差':>8}  {'n':>5}"
    print(header)
    for k in iv.NEAR_FUTURE_K_LEVELS:
        col = f"near_future_fire_k{k}_raw"
        auc_all, n_all = _diff_auc(paired, col, y)
        print(f"  K={k:<2}  {'全体':<6}  {auc_all:>8.4f}  {proto_ref[k]['全体']:>9.4f}  "
              f"{auc_all - proto_ref[k]['全体']:>+8.4f}  {n_all:>5d}")
        for phase, mask in masks.items():
            auc_p, n_p = _diff_auc(paired, col, y, mask)
            print(f"  K={k:<2}  {phase:<6}  {auc_p:>8.4f}  {proto_ref[k][phase]:>9.4f}  "
                  f"{auc_p - proto_ref[k][phase]:>+8.4f}  {n_p:>5d}")
    print("\n=== 完了 ===")


def main() -> None:
    df = _generate()
    _verify(df)


if __name__ == "__main__":
    main()
