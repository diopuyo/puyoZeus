"""B3 学習・評価の固定定数 (事前登録 docs/B3_PREREG_2026-10-05.md の数値門・手順をそのまま定数化)。

ここの値は「結果を見る前に固定」した値。変更は事前登録の改訂とセットで行い、
変更理由・版を記録する (§10 末尾)。マジックナンバーを各所に散らさない。
"""
from __future__ import annotations

from dataclasses import dataclass

# --- 乱数・CV (§2.3 / §3.2) ---
OUTER_SEEDS: tuple[int, ...] = (1003, 1004, 1005)      # モデル学習の seed (fold 割当は seed に依存しない)
FINAL_SEED = 1003                                     # 最終配布用 seed (事前固定)
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
OUTER_SALT = "B3-outer-v1|"
INNER_SALT_FMT = "B3-inner-v1|outer{k}|"
PLAYER_CTRL_SALT = "B3-player-ctrl|"
GROUP_JOIN = "+"
MIN_GROUPS_TOTAL = 50                                  # §2.1 本数不足時の扱い

# --- 採点 (§2.4) ---
PROB_CLIP = 1e-6
ECE_BINS = 10
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 1003
CI_LOW_PCT, CI_HIGH_PCT = 2.5, 97.5
NULL_P = 0.5
HOLD_MAX_SEC = 0.5                                     # 推論失敗時に最後の p を保持する上限 (500ms)

# --- 層の最低母数 (§3.2-4) ---
STRATUM_MIN_GAMES = 20
STRATUM_MIN_GROUPS = 5

# --- 局面 (§2.5 user 定義、phase 列は b3_features が計算済み) ---
PHASE_NAMES: tuple[str, ...] = ("early", "middle", "late")
HONSEN_RATIO = 0.6

# --- HGB の基準設定 (R1s 本番: models/exchange_event_v1_r1_hgb) ---
HGB_REFERENCE = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                     early_stopping=False)
INNER_TIE_LL = 0.001                                   # 内側 LL 差がこれ以内なら低費用側を選ぶ (§3.2)
THREADS = 1                                            # BLAS/OpenMP スレッド = 1 (決定性、DECISIONS 10/1)

# --- 副次 CV (§3.3) と単調性事後検査 (§3.5) ---
SECONDARY_MIN_PLAYER_VIDEOS = 3
SECONDARY_HOLD_MEAN = 0.02
SECONDARY_HOLD_PLAYER = 0.05
SECONDARY_HOLD_PLAYER_MIN_GAMES = 20
MONO_MINOR_SHARE = 0.05
MONO_MINOR_WIDTH = 0.03
MONO_HOLD_WIDTH = 0.05
MONO_HOLD_GAMES = 20
MONO_STEPS: tuple[int, ...] = (-6, -4, -2, 2, 4, 6)


@dataclass(frozen=True)
class Stage1Gates:
    """§3.2 の数値門。"""
    overall_point_max: float = 0.0          # 点推定 < 0 (厳密不等号は判定側で扱う)
    overall_upper_max: float = 0.002
    pair_strata_point_max: float = 0.0      # 撃ち合い層・両者構築層: 点推定 < 0
    other_strata_point_max: float = 0.005
    other_strata_upper_max: float = 0.020
    brier_diff_max: float = 0.002
    ece_diff_max: float = 0.010


STAGE1_GATES = Stage1Gates()

# 必須層名 (b3_eval_strata の名前と一致)
PAIR_STRATA: tuple[str, ...] = ("exchange", "both_build")
OTHER_STRATA: tuple[str, ...] = ("phase_early", "phase_middle", "phase_late", "both_chain", "honsen_after")
REPORT_STRATA: tuple[str, ...] = ("static", "honsen_unknown", "chain7plus_after_old")
