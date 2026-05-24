"""
総合スコア算出モジュール

8つの指標を重み付け和として結合し、1P と 2P の差分から
-100〜+100 の有利不利スコアを算出する。

正の値: 1P 有利 / 負の値: 2P 有利 / 0: 互角
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.indicators import (
    ALL_INDICATOR_NAMES,
    INDICATOR_CHAIN_TIMING,
    INDICATOR_COLOR_VARIANCE,
    INDICATOR_DEATH_RISK,
    INDICATOR_EXTENSION,
    INDICATOR_FIELD_EFF,
    INDICATOR_HARASSMENT,
    INDICATOR_INCOMING_OJAMA,
    INDICATOR_KEY_FLEXIBILITY,
    INDICATOR_MAIN_CHAIN,
    INDICATOR_NEXT_ACCEPTANCE,
    INDICATOR_OFFSET,
    INDICATOR_SECOND,
    INDICATOR_SHAPE_SCORE,
    INDICATOR_SUB_CHAIN,
    INDICATOR_SUB_CHAIN_INDEP,
    INDICATOR_TAIL_HEIGHT,
    INDICATOR_TOUCHING_DENSITY,
    IndicatorSet,
)

# ============================
# 定数定義
# ============================

# 総合スコアの出力レンジ
SCORE_RANGE_MIN: float = -100.0
SCORE_RANGE_MAX: float = 100.0

# プレイヤー識別子
PLAYER_1P: str = "1P"
PLAYER_2P: str = "2P"
ADVANTAGE_EVEN: str = "EVEN"
# 互角判定の閾値 (絶対値がこれ以下なら EVEN)
# 旧 5.0 は高過ぎた (実盤面スコアは ±3 程度に収まることが多い) ため 3.0 に引き下げ
EVEN_THRESHOLD: float = 3.0

# 8指標 + 拡張5指標のデフォルト重み
# 正の重み: 高いほど有利 / 負の重み: 高いほど不利 (窒息リスク)
# 拡張4指標 (shape/touching/tail/color_variance) は mayah/puyoai に近い相対重み比率を採用:
#   - puyo link 数 (touching_density) は meatfighter で 25% シェア → 0.5
#   - shape_score (U字+土台) は中盤評価のコア → 0.4
#   - tail_height (発火点低さ) は連鎖未完成時のみ寄与 → 0.3
#   - color_variance (色集中) は補助指標 → 0.2
# tune_weights.py で 50 試合 grid search 後にチューニング可能。
DEFAULT_WEIGHTS: dict[str, float] = {
    INDICATOR_MAIN_CHAIN: 1.5,
    INDICATOR_EXTENSION: 1.0,
    INDICATOR_SUB_CHAIN: 0.8,
    INDICATOR_HARASSMENT: 0.8,
    INDICATOR_DEATH_RISK: -1.5,
    INDICATOR_OFFSET: 1.2,
    INDICATOR_SECOND: 0.6,
    INDICATOR_FIELD_EFF: 0.4,
    # ネクスト受け入れ余地は ALL_INDICATOR_NAMES 外の拡張指標として扱う
    INDICATOR_NEXT_ACCEPTANCE: 0.6,
    # mayah/puyoai 先行研究ベースの拡張4指標
    INDICATOR_SHAPE_SCORE: 0.4,
    INDICATOR_TOUCHING_DENSITY: 0.5,
    INDICATOR_TAIL_HEIGHT: 0.3,
    INDICATOR_COLOR_VARIANCE: 0.2,
    # 高度戦略指標 3 種 (キーぷよ柔軟性 / 副砲独立性 / 連鎖タイミング圧)
    INDICATOR_KEY_FLEXIBILITY: 0.5,
    INDICATOR_SUB_CHAIN_INDEP: 0.4,
    INDICATOR_CHAIN_TIMING: 0.5,
    # 予告お邪魔受け圧 (画像認識から観測した相手の予告お邪魔個数)
    # 高い = 相手の攻撃が大きい = 自分側に不利 → 負の重み
    INDICATOR_INCOMING_OJAMA: -1.0,
}

# DEFAULT_WEIGHTS のうち ALL_INDICATOR_NAMES に属さない拡張指標
# (Scorer は breakdown には含めず、IndicatorSet 属性から別途読み取る)
EXTRA_INDICATOR_NAMES: frozenset[str] = frozenset({
    INDICATOR_NEXT_ACCEPTANCE,
    INDICATOR_SHAPE_SCORE,
    INDICATOR_TOUCHING_DENSITY,
    INDICATOR_TAIL_HEIGHT,
    INDICATOR_COLOR_VARIANCE,
    INDICATOR_KEY_FLEXIBILITY,
    INDICATOR_SUB_CHAIN_INDEP,
    INDICATOR_CHAIN_TIMING,
    INDICATOR_INCOMING_OJAMA,
})


# ==========================================================
# 学習済み重み (scripts/learn_weights_v2.py の出力)
# ==========================================================
# データセット: 3 動画 × 5 時刻フェーズ × ~143 試合 = 715 サンプル
# モデル: LogisticRegression L2 (C=5.0) + StandardScaler 巻き戻し
# split: video_holdout (train=動画01,02 / test=動画03)
# ベースライン (DEFAULT_WEIGHTS): test_acc=0.519
# 学習結果 (best_overall): test_acc=0.510
#
# 既存の DEFAULT_WEIGHTS は触らず、必要時に Scorer(weight_set="LEARNED_GLOBAL")
# のように選択可能とする。LEARNED_WEIGHTS は L1 ノルムを DEFAULT に揃えて
# スコア表示の桁感を維持してある (best_overall_normalized)。

# 全動画統合 (5 時刻フェーズ全て使用) で学習した重み
LEARNED_WEIGHTS_GLOBAL: dict[str, float] = {
    INDICATOR_MAIN_CHAIN: 2.2635,
    INDICATOR_EXTENSION: 1.7826,
    INDICATOR_SUB_CHAIN: -0.2257,
    INDICATOR_HARASSMENT: 0.5172,
    INDICATOR_DEATH_RISK: 0.0535,
    INDICATOR_OFFSET: -0.3474,
    INDICATOR_SECOND: 0.1720,
    INDICATOR_FIELD_EFF: 0.8644,
    INDICATOR_NEXT_ACCEPTANCE: 0.0,
    INDICATOR_SHAPE_SCORE: -0.1397,
    INDICATOR_TOUCHING_DENSITY: -2.7013,
    INDICATOR_TAIL_HEIGHT: -0.2183,
    INDICATOR_COLOR_VARIANCE: 0.4202,
    INDICATOR_KEY_FLEXIBILITY: -1.1969,
    INDICATOR_SUB_CHAIN_INDEP: -0.1765,
    INDICATOR_CHAIN_TIMING: 0.1207,
}

# midpoint フェーズ専用 (試合中央付近のスコアリング向け)
# 全 5 時刻のうち midpoint だけで学習。test_acc=0.548 (DEFAULT 0.519)
LEARNED_WEIGHTS_MIDPOINT: dict[str, float] = {
    INDICATOR_MAIN_CHAIN: 2.0682,
    INDICATOR_EXTENSION: 7.2032,
    INDICATOR_SUB_CHAIN: -1.0264,
    INDICATOR_HARASSMENT: 0.2400,
    INDICATOR_DEATH_RISK: -0.3483,
    INDICATOR_OFFSET: 0.6073,
    INDICATOR_SECOND: 0.1120,
    INDICATOR_FIELD_EFF: 0.9919,
    INDICATOR_NEXT_ACCEPTANCE: 0.0,
    INDICATOR_SHAPE_SCORE: 1.4153,
    INDICATOR_TOUCHING_DENSITY: 1.7432,
    INDICATOR_TAIL_HEIGHT: 0.9631,
    INDICATOR_COLOR_VARIANCE: -1.2165,
    INDICATOR_KEY_FLEXIBILITY: -0.3049,
    INDICATOR_SUB_CHAIN_INDEP: -0.3718,
    INDICATOR_CHAIN_TIMING: 0.3880,
}

# 試合終了 5 秒前専用 (決着直前向け)
# end_minus_5 だけで学習。test_acc=0.548
# 注: ほぼ全指標が負の符号 = 「敗者の方が指標が高い」終盤の特殊現象に最適化
LEARNED_WEIGHTS_END: dict[str, float] = {
    INDICATOR_MAIN_CHAIN: -2.2122,
    INDICATOR_EXTENSION: 3.4751,
    INDICATOR_SUB_CHAIN: -0.4477,
    INDICATOR_HARASSMENT: -1.9583,
    INDICATOR_DEATH_RISK: 0.1326,
    INDICATOR_OFFSET: -1.1143,
    INDICATOR_SECOND: 0.5200,
    INDICATOR_FIELD_EFF: -1.6699,
    INDICATOR_NEXT_ACCEPTANCE: 0.0,
    INDICATOR_SHAPE_SCORE: -2.7570,
    INDICATOR_TOUCHING_DENSITY: -3.6685,
    INDICATOR_TAIL_HEIGHT: -2.3409,
    INDICATOR_COLOR_VARIANCE: -0.3275,
    INDICATOR_KEY_FLEXIBILITY: -1.1415,
    INDICATOR_SUB_CHAIN_INDEP: 0.6892,
    INDICATOR_CHAIN_TIMING: -0.0882,
}

# 試合開始 20 秒後専用 (序盤向け)
# start_plus_20 だけで学習。test_acc=0.571 (最高精度)
LEARNED_WEIGHTS_START: dict[str, float] = {
    INDICATOR_MAIN_CHAIN: 1.7975,
    INDICATOR_EXTENSION: 4.1085,
    INDICATOR_SUB_CHAIN: -0.8092,
    INDICATOR_HARASSMENT: 1.0222,
    INDICATOR_DEATH_RISK: 0.5853,
    INDICATOR_OFFSET: 0.5247,
    INDICATOR_SECOND: 1.7105,
    INDICATOR_FIELD_EFF: 0.1248,
    INDICATOR_NEXT_ACCEPTANCE: 0.0,
    INDICATOR_SHAPE_SCORE: -1.2881,
    INDICATOR_TOUCHING_DENSITY: -0.5739,
    INDICATOR_TAIL_HEIGHT: 0.8744,
    INDICATOR_COLOR_VARIANCE: 0.1267,
    INDICATOR_KEY_FLEXIBILITY: -2.6476,
    INDICATOR_SUB_CHAIN_INDEP: -1.3925,
    INDICATOR_CHAIN_TIMING: -0.1005,
}

# ==========================================================
# V3: 多重共線性除去 + 大規模 (1390 サンプル) + 物理符号尊重
# ==========================================================
# scripts.learn_weights_v3 の出力 (best_overall_normalized → 16 次元へ展開)。
# データセット: 3 動画 × 10 時刻フェーズ × 139 試合 = 1390 サンプル
# 多重共線性: VIF max 18.57 → 3.72 (next_acceptance/offset_power/
# touching_density を削除)
# モデル: RidgeClassifier (alpha=0.1)
# split: video_holdout (train=動画01,02 / test=動画03)
# 学習結果: video_holdout test_acc=0.600 / kfold(k=5)=0.600 ± 0.020
# ベースライン (DEFAULT_WEIGHTS): test_acc=0.523
#
# 削除特徴量 (重み=0.0):
#   next_acceptance        : 盤面差分が常に 0 で無情報 (VIF=inf)
#   offset_power           : main_chain_maturity と r=0.84 重複 (VIF=18.57)
#   touching_density       : offset_power と r=0.76 重複 (VIF=6.91)
LEARNED_WEIGHTS_V3_GLOBAL: dict[str, float] = {
    INDICATOR_MAIN_CHAIN: 0.8756,
    INDICATOR_EXTENSION: 2.1670,
    INDICATOR_SUB_CHAIN: 0.2995,
    INDICATOR_HARASSMENT: 0.4740,
    INDICATOR_DEATH_RISK: 0.5275,
    INDICATOR_OFFSET: 0.0,
    INDICATOR_SECOND: 0.5659,
    INDICATOR_FIELD_EFF: 0.5811,
    INDICATOR_NEXT_ACCEPTANCE: 0.0,
    INDICATOR_SHAPE_SCORE: 0.7921,
    INDICATOR_TOUCHING_DENSITY: 0.0,
    INDICATOR_TAIL_HEIGHT: 0.6030,
    INDICATOR_COLOR_VARIANCE: 0.6941,
    INDICATOR_KEY_FLEXIBILITY: 2.7638,
    INDICATOR_SUB_CHAIN_INDEP: 0.4007,
    INDICATOR_CHAIN_TIMING: 0.4557,
}

# ==========================================================
# RECOMMENDED_WEIGHTS (V3 + ablation 反映、冗長指標を 0 化した最小構成)
# ==========================================================
# scripts.ablation_study (1390 サンプル × video_holdout test) の結果を反映。
# V3 で削除済の 3 指標 (next_acceptance / offset_power / touching_density) に
# 加え、ablation で test_acc 落ち幅が負 (= 除外したら寧ろ精度向上) の 6 指標を
# ゼロ化した、計 7 指標で構成される最小ウェイト。
#
# 削除した指標 (重み=0):
#   next_acceptance        : 学習データで定数列 (std=0、unique=1)
#   offset_power           : main_chain_maturity と高相関 (V3 で除外済)
#   touching_density       : offset_power と高相関 (V3 で除外済)
#   tail_height            : ablation drop=-0.0487 (除外で精度向上)
#   second_chain_potential : ablation drop=-0.0282
#   key_flexibility        : ablation drop=-0.0205
#   shape_score            : ablation drop=-0.0154
#   chain_timing_pressure  : ablation drop=-0.0103
#   harassment_resistance  : ablation drop=-0.0026
#
# 保持した 7 指標 (RidgeClassifier alpha=0.5 で再学習、標準化空間で test_acc=0.641
# 原スケール (生重み) test_acc=0.5487):
LEARNED_WEIGHTS_RECOMMENDED: dict[str, float] = {
    INDICATOR_MAIN_CHAIN: 1.7684,
    INDICATOR_EXTENSION: 4.3765,
    INDICATOR_SUB_CHAIN: 0.6050,
    INDICATOR_HARASSMENT: 0.0,
    INDICATOR_DEATH_RISK: 1.0654,
    INDICATOR_OFFSET: 0.0,
    INDICATOR_SECOND: 0.0,
    INDICATOR_FIELD_EFF: 1.1736,
    INDICATOR_NEXT_ACCEPTANCE: 0.0,
    INDICATOR_SHAPE_SCORE: 0.0,
    INDICATOR_TOUCHING_DENSITY: 0.0,
    INDICATOR_TAIL_HEIGHT: 0.0,
    INDICATOR_COLOR_VARIANCE: 1.4019,
    INDICATOR_KEY_FLEXIBILITY: 0.0,
    INDICATOR_SUB_CHAIN_INDEP: 0.8092,
    INDICATOR_CHAIN_TIMING: 0.0,
}

# ==========================================================
# Phase J 学習済み重み (2026-04-27、21 特徴量、test_acc=0.651)
# ==========================================================
# scripts/learn_weights_v3.py で Phase J 21 特徴量 (incoming_ojama_pressure +
# 凝視 opponent_chain_threat 含む) を v3 reduced 化した CSV で学習。
# best モデル: lr_l1 C=0.5、video holdout test_acc=0.651 (旧 V3=0.600 から +5.1%)
LEARNED_WEIGHTS_PHASE_J_GLOBAL: dict[str, float] = {
    "main_chain_maturity": 2.0877,
    "extension_potential": 3.4672,
    "sub_chain_quality": -0.7562,
    "harassment_resistance": 0.0,
    "death_risk": -0.1088,
    "offset_power": 0.0,
    "second_chain_potential": -0.4453,
    "field_efficiency": 0.0,
    "next_acceptance": 0.0,
    "shape_score": 0.5495,
    "touching_density": 0.0,
    "tail_height": -0.7834,
    "color_variance": 0.0,
    "key_flexibility": -0.8334,
    "sub_chain_independence": 0.1629,
    "chain_timing_pressure": 0.0,
    "incoming_ojama_pressure": -1.0167,
    # Phase J 4 指標
    "opponent_chain_threat": -0.5201,
    "adjacent_height_diff": -1.2934,
    "high_connection_count": 0.0,
    "required_puyo_to_fire": 0.1752,
}


# ==========================================================
# Phase J phase-aware 重み (2026-04-27、phase 別 21 特徴量)
# ==========================================================
# scripts/learn_weights_phase_aware.py で start/mid/end 別に LR L1 学習。
# end phase で test_acc=0.744 (旧 LEARNED_END 0.548 から +19.6%)
LEARNED_WEIGHTS_PHASE_J_START: dict[str, float] = {
    "main_chain_maturity": -0.6415,
    "extension_potential": -1.5600,
    "sub_chain_quality": -0.1565,
    "harassment_resistance": -0.3925,
    "death_risk": 0.3295,
    "offset_power": 1.0974,
    "second_chain_potential": -0.5912,
    "field_efficiency": 0.6193,
    "next_acceptance": 0.6565,
    "shape_score": 1.0869,
    "touching_density": -2.9477,
    "tail_height": -0.3518,
    "color_variance": 0.6349,
    "key_flexibility": 3.9090,
    "sub_chain_independence": 0.0,
    "chain_timing_pressure": 0.0,
    "incoming_ojama_pressure": -0.6682,
    "opponent_chain_threat": -0.5241,
    "adjacent_height_diff": -1.4295,
    "high_connection_count": -0.1406,
    "required_puyo_to_fire": 0.0962,
}
LEARNED_WEIGHTS_PHASE_J_MID: dict[str, float] = {
    "main_chain_maturity": 0.2209,
    "extension_potential": 0.2513,
    "sub_chain_quality": 0.0,
    "harassment_resistance": 0.0,
    "death_risk": 0.0,
    "offset_power": 0.0,
    "second_chain_potential": 0.0,
    "field_efficiency": 0.0,
    "next_acceptance": -0.6089,
    "shape_score": 0.0,
    "touching_density": -0.1157,
    "tail_height": 0.0,
    "color_variance": 0.0,
    "key_flexibility": 0.0,
    "sub_chain_independence": 0.0,
    "chain_timing_pressure": 0.0,
    "incoming_ojama_pressure": -0.7709,
    "opponent_chain_threat": 0.0,
    "adjacent_height_diff": -0.0624,
    "high_connection_count": 0.0,
    "required_puyo_to_fire": 0.0,
}
LEARNED_WEIGHTS_PHASE_J_END: dict[str, float] = {
    "main_chain_maturity": 0.4109,
    "extension_potential": 6.2349,
    "sub_chain_quality": 0.0,
    "harassment_resistance": 2.1756,
    "death_risk": 0.0,
    "offset_power": 0.0,
    "second_chain_potential": -1.6205,
    "field_efficiency": 1.5958,
    "next_acceptance": -0.1281,
    "shape_score": 0.0,
    "touching_density": -4.1523,
    "tail_height": -1.6405,
    "color_variance": 0.0,
    "key_flexibility": 0.0,
    "sub_chain_independence": 0.6315,
    "chain_timing_pressure": 0.0,
    "incoming_ojama_pressure": -0.6449,
    "opponent_chain_threat": 0.0,
    "adjacent_height_diff": -1.8046,
    "high_connection_count": -0.2512,
    "required_puyo_to_fire": 0.0826,
}


# ==========================================================
# Phase E phase-aware 重み (2026-05-05 拡張、新方針 pipeline 推論ベース)
# ==========================================================
# scripts/phase_e_collect_indicator_dataset で state machine 確定盤面
# のみから 7,650 サンプル / 38 動画 / 約 1,549 試合を再生成、E-3 の
# 多重共線性分析で 5 特徴量 (offset_power, touching_density,
# incoming_ojama_pressure, opponent_chain_threat, required_puyo_to_fire)
# を削除した 16 特徴量で scripts/phase_e_learn_phase_aware.py により
# phase 別 LR L2 学習。削除 5 特徴量は 0.0 で埋めて 21 features 形式維持。
#
# LOOV (leave-one-video-out) 結果:
#   start: 0.533 ± 0.076 (n=1553)
#   mid:   0.582 ± 0.047 (n=4544)
#   end:   0.862 ± 0.086 (n=1553)
#   平均:  0.659 (旧 PhaseAware overall 0.578 から +8.1pt)
LEARNED_WEIGHTS_PHASE_E_START: dict[str, float] = {
    "main_chain_maturity": 0.1739,
    "extension_potential": 0.7820,
    "sub_chain_quality": -0.1639,
    "harassment_resistance": 0.1001,
    "death_risk": -0.1220,
    "offset_power": 0.0,  # E-3 削除
    "second_chain_potential": 0.1581,
    "field_efficiency": 0.0042,
    "next_acceptance": 0.2303,
    "shape_score": -0.0559,
    "touching_density": 0.0,  # E-3 削除
    "tail_height": -0.2915,
    "color_variance": 1.3128,
    "key_flexibility": -0.8895,
    "sub_chain_independence": 0.1582,
    "chain_timing_pressure": 0.0602,
    "incoming_ojama_pressure": 0.0,  # E-3 削除
    "opponent_chain_threat": 0.0,  # E-3 削除
    "adjacent_height_diff": -0.0690,
    "high_connection_count": -0.3410,
    "required_puyo_to_fire": 0.0,  # E-3 削除
}
LEARNED_WEIGHTS_PHASE_E_MID: dict[str, float] = {
    "main_chain_maturity": 0.9747,
    "extension_potential": 3.3109,
    "sub_chain_quality": 0.1561,
    "harassment_resistance": 0.0657,
    "death_risk": 0.2939,
    "offset_power": 0.0,
    "second_chain_potential": 0.0700,
    "field_efficiency": -0.2344,
    "next_acceptance": 0.0935,
    "shape_score": -0.2793,
    "touching_density": 0.0,
    "tail_height": -0.2860,
    "color_variance": 0.8066,
    "key_flexibility": -3.7195,
    "sub_chain_independence": 0.0934,
    "chain_timing_pressure": 0.8187,
    "incoming_ojama_pressure": 0.0,
    "opponent_chain_threat": 0.0,
    "adjacent_height_diff": -0.5083,
    "high_connection_count": 0.7494,
    "required_puyo_to_fire": 0.0,
}
LEARNED_WEIGHTS_PHASE_E_END: dict[str, float] = {
    "main_chain_maturity": -2.0212,
    "extension_potential": 9.8729,
    "sub_chain_quality": 1.1538,
    "harassment_resistance": 1.1308,
    "death_risk": -0.7571,
    "offset_power": 0.0,
    "second_chain_potential": -3.1488,
    "field_efficiency": -2.0297,
    "next_acceptance": -1.0845,
    "shape_score": -1.4423,
    "touching_density": 0.0,
    "tail_height": -0.2620,
    "color_variance": 1.1281,
    "key_flexibility": -8.9826,
    "sub_chain_independence": 2.3831,
    "chain_timing_pressure": 2.5029,
    "incoming_ojama_pressure": 0.0,
    "opponent_chain_threat": 0.0,
    "adjacent_height_diff": 0.3520,
    "high_connection_count": 2.4155,
    "required_puyo_to_fire": 0.0,
}


# ==========================================================
# Phase E + Tier B phase-aware 重み (2026-05-06、Tier B 形質指標反映)
# ==========================================================
# 上記 PHASE_E は 38動画 / 7,650行 / 21 features で学習。
# 本 TIER_B variant は scripts/phase_e_collect_indicator_dataset で
# 全 66 動画 / 10,945 行を再生成 + Tier B 形質 3 指標
# (planning_entropy / structure_solidity / base_flatness) 追加で学習。
# 24 features (E-3 削除 5 features は 0.0 で埋め)。
#
# LOOV 結果 (vs 旧 PHASE_E 0.659):
#   start: 0.491 ± 0.115 (旧 0.533、-4.2pt)
#   mid:   0.569 ± 0.069 (旧 0.582、-1.3pt)
#   end:   0.851 ± 0.088 (旧 0.862、-1.1pt)
#   平均:  0.637 (旧 0.659、-2.2pt)
#
# 平均では旧重みが優位だが、GBM permutation importance では
# base_flatness が rank 2 (0.0370)、structure_solidity が rank 7 (0.0234)
# と Tier B 指標が高重要度を占めるため、研究用途で参照可能とする。
LEARNED_WEIGHTS_PHASE_E_TIERB_START: dict[str, float] = {
    "main_chain_maturity": -0.1107,
    "extension_potential": 5.0306,
    "sub_chain_quality": -0.1140,
    "harassment_resistance": 0.2823,
    "death_risk": -0.3029,
    "offset_power": 0.0,
    "second_chain_potential": -0.9071,
    "field_efficiency": -0.0822,
    "next_acceptance": 0.1005,
    "shape_score": 0.0703,
    "touching_density": 0.0,
    "tail_height": 0.1709,
    "color_variance": 0.2141,
    "key_flexibility": -4.4395,
    "sub_chain_independence": 0.2643,
    "chain_timing_pressure": 0.0087,
    "incoming_ojama_pressure": 0.0,
    "opponent_chain_threat": 0.0,
    "adjacent_height_diff": -0.4028,
    "high_connection_count": 0.4795,
    "required_puyo_to_fire": 0.0,
    "planning_entropy": 0.2524,
    "structure_solidity": -0.0898,
    "base_flatness": 0.3215,
}
LEARNED_WEIGHTS_PHASE_E_TIERB_MID: dict[str, float] = {
    "main_chain_maturity": 0.8877,
    "extension_potential": 2.0968,
    "sub_chain_quality": -0.1475,
    "harassment_resistance": 0.1837,
    "death_risk": -0.1157,
    "offset_power": 0.0,
    "second_chain_potential": 0.1542,
    "field_efficiency": 0.1820,
    "next_acceptance": 0.4455,
    "shape_score": -0.1730,
    "touching_density": 0.0,
    "tail_height": -0.2944,
    "color_variance": 0.7302,
    "key_flexibility": -0.5912,
    "sub_chain_independence": 0.1050,
    "chain_timing_pressure": -0.0932,
    "incoming_ojama_pressure": 0.0,
    "opponent_chain_threat": 0.0,
    "adjacent_height_diff": -0.0973,
    "high_connection_count": -0.1640,
    "required_puyo_to_fire": 0.0,
    "planning_entropy": -0.0273,
    "structure_solidity": 0.0698,
    "base_flatness": 0.0239,
}
LEARNED_WEIGHTS_PHASE_E_TIERB_END: dict[str, float] = {
    "main_chain_maturity": 3.1139,
    "extension_potential": 4.9668,
    "sub_chain_quality": 0.9774,
    "harassment_resistance": 0.5082,
    "death_risk": -0.4775,
    "offset_power": 0.0,
    "second_chain_potential": -1.3155,
    "field_efficiency": -1.1560,
    "next_acceptance": -1.2488,
    "shape_score": 0.6236,
    "touching_density": 0.0,
    "tail_height": 0.3466,
    "color_variance": 1.0013,
    "key_flexibility": -2.4103,
    "sub_chain_independence": 1.2714,
    "chain_timing_pressure": 1.5410,
    "incoming_ojama_pressure": 0.0,
    "opponent_chain_threat": 0.0,
    "adjacent_height_diff": -0.5516,
    "high_connection_count": 1.5482,
    "required_puyo_to_fire": 0.0,
    "planning_entropy": 0.3371,
    "structure_solidity": -0.6391,
    "base_flatness": -5.3372,
}


# weight_set 名 → 重み辞書のレジストリ。Scorer(weight_set=...) で参照する。
WEIGHT_SET_DEFAULT: str = "DEFAULT"
WEIGHT_SET_LEARNED_GLOBAL: str = "LEARNED_GLOBAL"
WEIGHT_SET_LEARNED_MIDPOINT: str = "LEARNED_MIDPOINT"
WEIGHT_SET_LEARNED_END: str = "LEARNED_END"
WEIGHT_SET_LEARNED_START: str = "LEARNED_START"
WEIGHT_SET_LEARNED_V3_GLOBAL: str = "LEARNED_V3_GLOBAL"
WEIGHT_SET_LEARNED_PHASE_J_GLOBAL: str = "LEARNED_PHASE_J_GLOBAL"
WEIGHT_SET_LEARNED_PHASE_J_START: str = "LEARNED_PHASE_J_START"
WEIGHT_SET_LEARNED_PHASE_J_MID: str = "LEARNED_PHASE_J_MID"
WEIGHT_SET_LEARNED_PHASE_J_END: str = "LEARNED_PHASE_J_END"
WEIGHT_SET_LEARNED_PHASE_E_START: str = "LEARNED_PHASE_E_START"
WEIGHT_SET_LEARNED_PHASE_E_MID: str = "LEARNED_PHASE_E_MID"
WEIGHT_SET_LEARNED_PHASE_E_END: str = "LEARNED_PHASE_E_END"
WEIGHT_SET_LEARNED_PHASE_E_TIERB_START: str = "LEARNED_PHASE_E_TIERB_START"
WEIGHT_SET_LEARNED_PHASE_E_TIERB_MID: str = "LEARNED_PHASE_E_TIERB_MID"
WEIGHT_SET_LEARNED_PHASE_E_TIERB_END: str = "LEARNED_PHASE_E_TIERB_END"
WEIGHT_SET_RECOMMENDED: str = "RECOMMENDED"

WEIGHT_SET_REGISTRY: dict[str, dict[str, float]] = {
    WEIGHT_SET_DEFAULT: DEFAULT_WEIGHTS,
    WEIGHT_SET_LEARNED_GLOBAL: LEARNED_WEIGHTS_GLOBAL,
    WEIGHT_SET_LEARNED_MIDPOINT: LEARNED_WEIGHTS_MIDPOINT,
    WEIGHT_SET_LEARNED_END: LEARNED_WEIGHTS_END,
    WEIGHT_SET_LEARNED_START: LEARNED_WEIGHTS_START,
    WEIGHT_SET_LEARNED_V3_GLOBAL: LEARNED_WEIGHTS_V3_GLOBAL,
    WEIGHT_SET_LEARNED_PHASE_J_GLOBAL: LEARNED_WEIGHTS_PHASE_J_GLOBAL,
    WEIGHT_SET_LEARNED_PHASE_J_START: LEARNED_WEIGHTS_PHASE_J_START,
    WEIGHT_SET_LEARNED_PHASE_J_MID: LEARNED_WEIGHTS_PHASE_J_MID,
    WEIGHT_SET_LEARNED_PHASE_J_END: LEARNED_WEIGHTS_PHASE_J_END,
    WEIGHT_SET_LEARNED_PHASE_E_START: LEARNED_WEIGHTS_PHASE_E_START,
    WEIGHT_SET_LEARNED_PHASE_E_MID: LEARNED_WEIGHTS_PHASE_E_MID,
    WEIGHT_SET_LEARNED_PHASE_E_END: LEARNED_WEIGHTS_PHASE_E_END,
    WEIGHT_SET_LEARNED_PHASE_E_TIERB_START: LEARNED_WEIGHTS_PHASE_E_TIERB_START,
    WEIGHT_SET_LEARNED_PHASE_E_TIERB_MID: LEARNED_WEIGHTS_PHASE_E_TIERB_MID,
    WEIGHT_SET_LEARNED_PHASE_E_TIERB_END: LEARNED_WEIGHTS_PHASE_E_TIERB_END,
    WEIGHT_SET_RECOMMENDED: LEARNED_WEIGHTS_RECOMMENDED,
}


# ============================
# データクラス
# ============================


@dataclass(frozen=True)
class ScoreResult:
    """
    総合スコア算出結果。

    Attributes:
        total_score: -100〜+100 の総合スコア (正=1P有利)。
        player1_raw: 1P の重み付け和 (正規化前)。
        player2_raw: 2P の重み付け和。
        player1_breakdown: 指標名→1P の重み付け寄与値。
        player2_breakdown: 指標名→2P の重み付け寄与値。
        weights: 使用した重み辞書。
    """
    total_score: float
    player1_raw: float
    player2_raw: float
    player1_breakdown: dict[str, float]
    player2_breakdown: dict[str, float]
    weights: dict[str, float] = field(default_factory=dict)

    def advantage_side(self) -> str:
        """有利側を返す: PLAYER_1P / PLAYER_2P / ADVANTAGE_EVEN。"""
        if abs(self.total_score) <= EVEN_THRESHOLD:
            return ADVANTAGE_EVEN
        return PLAYER_1P if self.total_score > 0 else PLAYER_2P

    def to_dict(self) -> dict[str, Any]:
        """JSON 保存可能な辞書に変換する。"""
        return {
            "total_score": self.total_score,
            "advantage": self.advantage_side(),
            "player1_raw": self.player1_raw,
            "player2_raw": self.player2_raw,
            "player1_breakdown": dict(self.player1_breakdown),
            "player2_breakdown": dict(self.player2_breakdown),
            "weights": dict(self.weights),
        }


# ============================
# Scorer
# ============================


class Scorer:
    """
    8指標から総合スコアを算出するクラス。

    重みは差し替え可能で、将来的に ML モデル (model.py) に置換できる。

    Usage:
        scorer = Scorer()
        result = scorer.score(indicator_set_1p, indicator_set_2p)
        print(result.total_score)   # -100〜+100
        print(result.advantage_side())  # "1P" / "2P" / "EVEN"
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        weight_set: str = WEIGHT_SET_DEFAULT,
    ) -> None:
        """
        Args:
            weights: 指標名→重みの辞書 (None ならデフォルト)。
                     不足している指標はデフォルト値で補完される。
                     ``weights`` を直接指定した場合は ``weight_set`` を無視する。
            weight_set: 重みセット名。WEIGHT_SET_REGISTRY のキーを指定する
                        (例: "DEFAULT", "LEARNED_GLOBAL", "LEARNED_MIDPOINT")。
                        ``weights`` 引数が None の場合のみ参照される。

        Raises:
            ValueError: 未知の指標名や未知の weight_set が含まれている場合。
        """
        if weight_set not in WEIGHT_SET_REGISTRY:
            raise ValueError(
                f"未知の weight_set: {weight_set}. "
                f"利用可能: {sorted(WEIGHT_SET_REGISTRY.keys())}",
            )
        base = WEIGHT_SET_REGISTRY[weight_set]
        merged: dict[str, float] = dict(base)
        # 不足する指標は DEFAULT_WEIGHTS で補完 (互換性確保)
        for name, val in DEFAULT_WEIGHTS.items():
            merged.setdefault(name, val)
        if weights:
            self._validate_weight_keys(weights)
            merged.update(weights)
        self._weights = merged
        self._normalizer = self._compute_normalizer(merged)
        self._weight_set = weight_set

    def score(
        self,
        player1: IndicatorSet,
        player2: IndicatorSet,
    ) -> ScoreResult:
        """
        1P と 2P の指標セットから総合スコアを算出する。

        ALL_INDICATOR_NAMES 外の拡張指標 (例: next_acceptance) は
        IndicatorSet の属性から別途読み込んで合算する。
        breakdown / raw には含めず、互換性を保つ。

        Args:
            player1: 1P の指標セット。
            player2: 2P の指標セット。

        Returns:
            ScoreResult: 総合スコアと内訳。
        """
        p1_break = self._weighted_breakdown(player1)
        p2_break = self._weighted_breakdown(player2)
        p1_raw = sum(p1_break.values())
        p2_raw = sum(p2_break.values())

        # 拡張指標 (next_acceptance 等) の差分寄与を total に加算
        extra_diff = self._extra_diff(player1, player2)

        diff = p1_raw - p2_raw + extra_diff
        if self._normalizer == 0.0:
            total = 0.0
        else:
            total = (diff / self._normalizer) * SCORE_RANGE_MAX

        total = max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, total))

        return ScoreResult(
            total_score=total,
            player1_raw=p1_raw,
            player2_raw=p2_raw,
            player1_breakdown=p1_break,
            player2_breakdown=p2_break,
            weights=dict(self._weights),
        )

    def _extra_diff(
        self, player1: IndicatorSet, player2: IndicatorSet,
    ) -> float:
        """拡張指標 (ALL_INDICATOR_NAMES 外) の差分寄与を返す。"""
        diff = 0.0
        for name in EXTRA_INDICATOR_NAMES:
            if name not in self._weights:
                continue
            weight = self._weights[name]
            p1_val = self._extra_value(player1, name)
            p2_val = self._extra_value(player2, name)
            diff += (p1_val - p2_val) * weight
        return diff

    @staticmethod
    def _extra_value(indicator_set: IndicatorSet, name: str) -> float:
        """IndicatorSet の属性から拡張指標値を取得する。

        next_acceptance は IndicatorSet 属性に格納される歴史的事情があり、
        他の拡張指標は results 辞書 + 同名属性の両方に格納されている。
        results 優先で読み、無ければ属性フォールバックする。
        """
        if name in indicator_set.results:
            return float(indicator_set.results[name].score)
        attr_map = {
            INDICATOR_NEXT_ACCEPTANCE: "next_acceptance",
            INDICATOR_SHAPE_SCORE: "shape_score",
            INDICATOR_TOUCHING_DENSITY: "touching_density",
            INDICATOR_TAIL_HEIGHT: "tail_height_score",
            INDICATOR_COLOR_VARIANCE: "color_variance_score",
            INDICATOR_KEY_FLEXIBILITY: "key_flexibility",
            INDICATOR_SUB_CHAIN_INDEP: "sub_chain_independence",
            INDICATOR_CHAIN_TIMING: "chain_timing_pressure",
            INDICATOR_INCOMING_OJAMA: "incoming_ojama_pressure",
        }
        attr = attr_map.get(name)
        if attr is None:
            return 0.0
        return float(getattr(indicator_set, attr, 0.0))

    # ============================
    # 内部メソッド
    # ============================

    def _weighted_breakdown(self, indicator_set: IndicatorSet) -> dict[str, float]:
        """指標セットを重み付け寄与値の辞書に変換する。

        EXTRA_INDICATOR_NAMES (ALL_INDICATOR_NAMES 外の拡張指標) は
        breakdown には含めず、score() 内で別途処理する。
        """
        breakdown: dict[str, float] = {}
        for name, weight in self._weights.items():
            if name in EXTRA_INDICATOR_NAMES:
                continue
            if name not in indicator_set.results:
                breakdown[name] = 0.0
                continue
            score = indicator_set.score_of(name)
            breakdown[name] = score * weight
        return breakdown

    @staticmethod
    def _validate_weight_keys(weights: dict[str, float]) -> None:
        """重み辞書のキーが既知の指標名であることを確認する。

        ALL_INDICATOR_NAMES と EXTRA_INDICATOR_NAMES の和集合を許容する。
        """
        allowed = set(ALL_INDICATOR_NAMES) | set(EXTRA_INDICATOR_NAMES)
        unknown = set(weights.keys()) - allowed
        if unknown:
            raise ValueError(f"未知の指標名: {sorted(unknown)}")

    @staticmethod
    def _compute_normalizer(weights: dict[str, float]) -> float:
        """
        重み差分の最大値 (絶対値の総和) を計算する。

        各指標の寄与値は 0〜|weight| の範囲で、差分は ±|weight|。
        合計の絶対値最大は sum(|weight|)。
        """
        return sum(abs(w) for w in weights.values())


# ============================
# Phase-aware Scorer
# ============================

# フェーズ識別子
PHASE_START: str = "start"
PHASE_MID: str = "mid"
PHASE_END: str = "end"

# 序盤フェーズの上限秒 (試合開始からこの秒数までを start とする)
PHASE_START_BOUNDARY_SEC: float = 30.0
# 終盤フェーズの幅 (試合終了までの残り秒。これ以下なら end とする)
PHASE_END_BOUNDARY_SEC: float = 15.0
# 線形補間の過渡域幅 (両端からこの秒数だけ重みを混ぜる)
PHASE_BLEND_WIDTH_SEC: float = 10.0

# フェーズ→学習済み重みのマッピング (weight_mode="learned" 用、後方互換)
PHASE_WEIGHT_MAP: dict[str, dict[str, float]] = {
    PHASE_START: LEARNED_WEIGHTS_START,
    PHASE_MID: LEARNED_WEIGHTS_MIDPOINT,
    PHASE_END: LEARNED_WEIGHTS_END,
}

# ==========================================================
# OPTIMAL_PHASE_WEIGHTS (実証ベスト per phase)
# ==========================================================
# 1390 サンプル × 6 戦略の比較結果 (data/verify/phase_aware_eval.json,
# scripts.eval_phase_aware の出力) から実証された最良の組合せ。
#
# | strategy | midpoint | end_minus_5 | video holdout |
# |---|---|---|---|
# | DEFAULT            | 0.604 | 0.108 | 0.523 |
# | LEARNED_GLOBAL     | 0.511 | 0.885 | 0.519 |
# | LEARNED_V3_GLOBAL  | 0.568 | 0.216 | 0.600 |
#
# - start: 序盤は盤面情報が薄く、DEFAULT が最も汎化する
# - mid:   中盤は LEARNED_V3_GLOBAL が holdout 0.600 で V1 系を超える
# - end:   終局は LEARNED_GLOBAL の end_minus_5=0.885 が圧倒的
#
# 既存 LEARNED_WEIGHTS_START/MIDPOINT/END (時刻フェーズ別 LR) は学習しすぎで
# overfit の懸念が強いため、OPTIMAL では採用せず PHASE_WEIGHT_MAP に残置する。
OPTIMAL_PHASE_WEIGHTS: dict[str, dict[str, float]] = {
    PHASE_START: DEFAULT_WEIGHTS,
    PHASE_MID: LEARNED_WEIGHTS_V3_GLOBAL,
    PHASE_END: LEARNED_WEIGHTS_GLOBAL,
}

# Phase J phase-aware (2026-04-27、21 特徴量、scripts/learn_weights_phase_aware.py)
# 各 phase で test_acc=0.590 / 0.631 / 0.744、平均 0.655
# end phase で旧 LEARNED_END 0.548 から +19.6%
PHASE_J_PHASE_WEIGHTS: dict[str, dict[str, float]] = {
    PHASE_START: LEARNED_WEIGHTS_PHASE_J_START,
    PHASE_MID: LEARNED_WEIGHTS_PHASE_J_MID,
    PHASE_END: LEARNED_WEIGHTS_PHASE_J_END,
}

# Phase E phase-aware (2026-05-05、新方針 pipeline 推論ベース、16 特徴量)
# scripts/phase_e_collect_indicator_dataset で state machine 確定盤面のみ
# から 4,893 サンプルを再生成、scripts/phase_e_learn_phase_aware.py で
# phase 別 LR L2 学習。LOOV 平均 0.644 (旧 PhaseAware 0.578 から +6.6pt)
PHASE_E_PHASE_WEIGHTS: dict[str, dict[str, float]] = {
    PHASE_START: LEARNED_WEIGHTS_PHASE_E_START,
    PHASE_MID: LEARNED_WEIGHTS_PHASE_E_MID,
    PHASE_END: LEARNED_WEIGHTS_PHASE_E_END,
}

# Phase E + Tier B phase-aware (2026-05-06)
# 全 66 動画 / 10,945 行 / 24 features (Tier B 形質指標 3 追加)
# LOOV 平均 0.637 (旧 PHASE_E 0.659 と比較で -2.2pt)
PHASE_E_TIERB_PHASE_WEIGHTS: dict[str, dict[str, float]] = {
    PHASE_START: LEARNED_WEIGHTS_PHASE_E_TIERB_START,
    PHASE_MID: LEARNED_WEIGHTS_PHASE_E_TIERB_MID,
    PHASE_END: LEARNED_WEIGHTS_PHASE_E_TIERB_END,
}

# weight_mode 識別子
WEIGHT_MODE_LEARNED: str = "learned"
WEIGHT_MODE_OPTIMAL: str = "optimal"
WEIGHT_MODE_PHASE_J: str = "phase_j"
WEIGHT_MODE_PHASE_E: str = "phase_e"
WEIGHT_MODE_PHASE_E_TIERB: str = "phase_e_tierb"
WEIGHT_MODE_REGISTRY: dict[str, dict[str, dict[str, float]]] = {
    WEIGHT_MODE_LEARNED: PHASE_WEIGHT_MAP,
    WEIGHT_MODE_OPTIMAL: OPTIMAL_PHASE_WEIGHTS,
    WEIGHT_MODE_PHASE_J: PHASE_J_PHASE_WEIGHTS,
    WEIGHT_MODE_PHASE_E: PHASE_E_PHASE_WEIGHTS,
    WEIGHT_MODE_PHASE_E_TIERB: PHASE_E_TIERB_PHASE_WEIGHTS,
}


def classify_phase(elapsed_sec: float, match_duration_sec: float) -> str:
    """試合内経過秒からフェーズを分類する。

    Args:
        elapsed_sec: 試合開始からの経過秒。
        match_duration_sec: 試合全長 (秒)。0 以下なら mid とみなす。

    Returns:
        "start" / "mid" / "end" のいずれか。
    """
    if match_duration_sec <= 0.0:
        return PHASE_MID
    elapsed = max(0.0, elapsed_sec)
    remaining = max(0.0, match_duration_sec - elapsed)
    if elapsed <= PHASE_START_BOUNDARY_SEC:
        return PHASE_START
    if remaining <= PHASE_END_BOUNDARY_SEC:
        return PHASE_END
    return PHASE_MID


def _blend_weights(
    weights_a: dict[str, float],
    weights_b: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    """2 つの重み辞書を ``(1-α)·a + α·b`` で線形補間する。

    補間先 (weights_b) が持たないキーは weights_a の値をそのまま使う。
    """
    alpha_clamped = max(0.0, min(1.0, alpha))
    keys = set(weights_a.keys()) | set(weights_b.keys())
    out: dict[str, float] = {}
    for key in keys:
        va = float(weights_a.get(key, 0.0))
        vb = float(weights_b.get(key, va))
        out[key] = (1.0 - alpha_clamped) * va + alpha_clamped * vb
    return out


def _interpolated_weights(
    elapsed_sec: float,
    match_duration_sec: float,
    phase_map: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """試合内経過秒に基づき隣接 phase 重みを線形補間して返す。

    - start → mid 過渡域: PHASE_START_BOUNDARY_SEC ± PHASE_BLEND_WIDTH_SEC/2
    - mid → end 過渡域:  (duration - PHASE_END_BOUNDARY_SEC) ± PHASE_BLEND_WIDTH_SEC/2
    - 過渡域外は単純に各 phase の重みをそのまま返す。

    Args:
        elapsed_sec: 試合内経過秒。
        match_duration_sec: 試合全長。0 以下なら mid 重みを返す。
        phase_map: phase 名→重み辞書のマップ。None の場合は従来の
            LEARNED_WEIGHTS_* (PHASE_WEIGHT_MAP) を使用する。
    """
    pmap = phase_map if phase_map is not None else PHASE_WEIGHT_MAP
    w_start = pmap[PHASE_START]
    w_mid = pmap[PHASE_MID]
    w_end = pmap[PHASE_END]
    if match_duration_sec <= 0.0:
        return dict(w_mid)
    elapsed = max(0.0, min(elapsed_sec, match_duration_sec))
    half = PHASE_BLEND_WIDTH_SEC / 2.0
    start_center = PHASE_START_BOUNDARY_SEC
    end_center = match_duration_sec - PHASE_END_BOUNDARY_SEC
    if elapsed <= start_center - half:
        return dict(w_start)
    if elapsed < start_center + half:
        alpha = (elapsed - (start_center - half)) / PHASE_BLEND_WIDTH_SEC
        return _blend_weights(w_start, w_mid, alpha)
    if elapsed <= end_center - half or end_center <= start_center + half:
        return dict(w_mid)
    if elapsed < end_center + half:
        alpha = (elapsed - (end_center - half)) / PHASE_BLEND_WIDTH_SEC
        return _blend_weights(w_mid, w_end, alpha)
    return dict(w_end)


class PhaseAwareScorer:
    """試合内経過時刻に応じて重みを切り替える Scorer。

    序盤 / 中盤 / 終盤で異なる学習済み重み (LEARNED_WEIGHTS_START /
    _MIDPOINT / _END) を使用し、interpolate=True の場合は隣接 phase の
    重みを線形補間して過渡域でなめらかに切り替える。

    Usage:
        scorer = PhaseAwareScorer(interpolate=True)
        result = scorer.score(ind1, ind2, elapsed_sec=42.0,
                              match_duration_sec=60.0)
        result.advantage_side()
    """

    # クラス定数 (タスク仕様の PHASE_BOUNDARIES_SEC を踏襲)
    PHASE_BOUNDARIES_SEC: tuple[float, float] = (
        PHASE_START_BOUNDARY_SEC,
        -PHASE_END_BOUNDARY_SEC,
    )

    def __init__(
        self,
        interpolate: bool = True,
        weights_overrides: dict[str, dict[str, float]] | None = None,
        weight_mode: str = WEIGHT_MODE_LEARNED,
    ) -> None:
        """
        Args:
            interpolate: True なら過渡域で隣接 phase 重みを線形補間する。
                False なら離散的に切替 (各 phase の学習済み重みをそのまま使用)。
            weights_overrides: phase 名 ("start"/"mid"/"end") → 重み辞書の
                マップで、特定 phase の重みを差し替えたい場合に使う。
                未指定 phase は weight_mode のベースがデフォルト。
            weight_mode: "learned" (従来の LEARNED_WEIGHTS_START/MIDPOINT/END)
                または "optimal" (OPTIMAL_PHASE_WEIGHTS = DEFAULT/V3/GLOBAL)。
                既定は後方互換のため "learned"。

        Raises:
            ValueError: 未知の weight_mode が指定された場合。
        """
        if weight_mode not in WEIGHT_MODE_REGISTRY:
            raise ValueError(
                f"未知の weight_mode: {weight_mode}. "
                f"利用可能: {sorted(WEIGHT_MODE_REGISTRY.keys())}",
            )
        self._interpolate = bool(interpolate)
        self._weight_mode = weight_mode
        base_map = WEIGHT_MODE_REGISTRY[weight_mode]
        # phase → 重み辞書 (DEFAULT_WEIGHTS で不足分を補完)
        self._phase_weights: dict[str, dict[str, float]] = {}
        for phase, base in base_map.items():
            merged = dict(base)
            for name, val in DEFAULT_WEIGHTS.items():
                merged.setdefault(name, val)
            if weights_overrides and phase in weights_overrides:
                merged.update(weights_overrides[phase])
            self._phase_weights[phase] = merged

    @property
    def interpolate(self) -> bool:
        """補間モードか否か。"""
        return self._interpolate

    @property
    def weight_mode(self) -> str:
        """使用中の重みモード ("learned" / "optimal")。"""
        return self._weight_mode

    def resolve_weights(
        self,
        elapsed_sec: float,
        match_duration_sec: float,
    ) -> dict[str, float]:
        """指定タイミングで実際に使用される重み辞書を返す。

        - 補間モード: ``_interpolated_weights`` の出力に DEFAULT 補完を施す。
        - 離散モード: 該当 phase の重み辞書をそのまま返す (DEFAULT 補完済)。
        """
        if not self._interpolate:
            phase = classify_phase(elapsed_sec, match_duration_sec)
            return dict(self._phase_weights[phase])
        blended = _interpolated_weights(
            elapsed_sec, match_duration_sec, self._phase_weights,
        )
        # DEFAULT_WEIGHTS で不足分を補完
        merged = dict(blended)
        for name, val in DEFAULT_WEIGHTS.items():
            merged.setdefault(name, val)
        return merged

    def score(
        self,
        p1_indicators: IndicatorSet,
        p2_indicators: IndicatorSet,
        elapsed_sec: float,
        match_duration_sec: float,
    ) -> ScoreResult:
        """1P/2P の指標と試合内経過時刻から ScoreResult を返す。

        Args:
            p1_indicators: 1P の IndicatorSet。
            p2_indicators: 2P の IndicatorSet。
            elapsed_sec: 試合開始からの経過秒。
            match_duration_sec: 試合全長 (秒)。

        Returns:
            ScoreResult: 該当 phase (or 補間) の重みでの総合スコア。
        """
        weights = self.resolve_weights(elapsed_sec, match_duration_sec)
        scorer = Scorer(weights=weights)
        return scorer.score(p1_indicators, p2_indicators)

    def current_phase(
        self,
        elapsed_sec: float,
        match_duration_sec: float,
    ) -> str:
        """現在のフェーズ名を返す ("start" / "mid" / "end")。"""
        return classify_phase(elapsed_sec, match_duration_sec)
