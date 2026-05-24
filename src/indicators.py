"""
8つの評価指標モジュール

ぷよぷよeスポーツの競技戦術に基づく8指標を計算する。
各指標は 0.0〜1.0 に正規化されたスコアを返し、差し替え可能な抽象設計とする。

指標一覧:
    1. 本線完成度 (main_chain_maturity)
    2. 伸ばし余地 (extension_potential)
    3. 副砲の質 (sub_chain_quality)
    4. 催促耐性 (harassment_resistance)
    5. 窒息リスク (death_risk)
    6. 相殺力 (offset_power)
    7. セカンド構築力 (second_chain_potential)
    8. フィールド効率 (field_efficiency)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    Board,
    VALID_COLORS,
)
from src.chain import (
    ChainResult,
    ChainSimulator,
    MIN_ERASE_COUNT,
    PROBABILISTIC_DEFAULT_SAMPLES,
    ProbabilisticChainResult,
)
from src.probabilistic_board import ProbabilisticBoard
from src.form_templates import (
    FRON_TEMPLATE,
    GTR_TEMPLATE,
    LLR_TEMPLATE,
    STAIRCASE_TEMPLATE,
    SULLEN_GTR_TEMPLATE,
    ZABUTON_TEMPLATE,
    best_template_score,
)

# ============================
# 指標名定数
# ============================

INDICATOR_MAIN_CHAIN: str = "main_chain_maturity"
INDICATOR_EXTENSION: str = "extension_potential"
INDICATOR_SUB_CHAIN: str = "sub_chain_quality"
INDICATOR_HARASSMENT: str = "harassment_resistance"
INDICATOR_DEATH_RISK: str = "death_risk"
INDICATOR_OFFSET: str = "offset_power"
INDICATOR_SECOND: str = "second_chain_potential"
INDICATOR_FIELD_EFF: str = "field_efficiency"
# ネクスト/ダブルネクスト受け入れ余地 (拡張指標、ALL_INDICATOR_NAMES 外で扱う)
INDICATOR_NEXT_ACCEPTANCE: str = "next_acceptance"

# ----- 拡張指標 (mayah/puyoai 先行研究に基づく追加 4 指標) -----
# 形評価: U字形 + GTR/LLR 風土台らしさ (mayah FIELD_USHAPE_LINEAR を簡易再現)
INDICATOR_SHAPE_SCORE: str = "shape_score"
# 接ぷよ密度: 同色隣接ペア / 全ぷよ数 (mayah CONNECTION_2/3, meatfighter リンク数)
INDICATOR_TOUCHING_DENSITY: str = "touching_density"
# 連鎖発火点高さ: 最終発火点の高さ (mayah IGNITION_HEIGHT)
INDICATOR_TAIL_HEIGHT: str = "tail_height"
# 色分散: 各色の盤面散らばり (meatfighter color variance)
INDICATOR_COLOR_VARIANCE: str = "color_variance"

# ----- 高度戦略指標 (キーぷよ柔軟性 / 副砲独立性 / 連鎖タイミング圧) -----
# キーぷよ柔軟性: 1 個追加で連鎖が伸びる比率
INDICATOR_KEY_FLEXIBILITY: str = "key_flexibility"
# 副砲独立性: 本線4+連結を除いた残骸での副砲存在度
INDICATOR_SUB_CHAIN_INDEP: str = "sub_chain_independence"
# 連鎖タイミング圧: 自盤面の発火近接度
INDICATOR_CHAIN_TIMING: str = "chain_timing_pressure"
# 受け攻撃圧 (incoming ojama vs 自分の相殺力): 予告お邪魔ぷよ列を観測値として与える
INDICATOR_INCOMING_OJAMA: str = "incoming_ojama_pressure"
# Phase J 新指標 (2026-04-27 mayah/ama 先行研究ベース)
INDICATOR_OPPONENT_THREAT: str = "opponent_chain_threat"
INDICATOR_HEIGHT_DIFF: str = "adjacent_height_diff"
INDICATOR_HIGH_CONNECTION: str = "high_connection_count"
INDICATOR_REQUIRED_FIRE: str = "required_puyo_to_fire"
# Phase K 新指標 (2026-04-27 凝視深化 + リソース無駄化)
INDICATOR_OPPONENT_OFFSET: str = "opponent_offset_power"
INDICATOR_POST_OJAMA_HEALTH: str = "post_ojama_chain_health"
INDICATOR_ISOLATED_PUYO: str = "isolated_puyo_count"
# Tier B 新指標 (2026-05-05、key_flexibility 二相性分離)
INDICATOR_PLANNING_ENTROPY: str = "planning_entropy"
INDICATOR_STRUCTURE_SOLIDITY: str = "structure_solidity"
INDICATOR_BASE_FLATNESS: str = "base_flatness"
# I-J 形テンプレ完成度 (2026-05-06、B-1)
INDICATOR_FORM_GTR: str = "form_gtr"
INDICATOR_FORM_LLR: str = "form_llr"
INDICATOR_FORM_STAIRCASE: str = "form_staircase"
INDICATOR_FORM_ZABUTON: str = "form_zabuton"
# B-1.b 拡張: citrus610/ama 由来の Sullen GTR / Fron 派生形 (2026-05-09 追加)
INDICATOR_FORM_SULLEN_GTR: str = "form_sullen_gtr"
INDICATOR_FORM_FRON: str = "form_fron"
# Phase F 新指標 (2026-05-07、B-4) 回し入れ巧拙
# STABLE フレーム間で物理的に説明できないぷよ消失を「回し入れ」候補として
# カウントし、上級者戦術の発生頻度を 0〜1 に正規化する
INDICATOR_ROTATION_SKILL: str = "rotation_skill"

# ============================
# Phase H1 新指標 (2026-05-08、合計 16 個)
# 「形は手段、機能が本質」: 機能能力指標 7 個 + 戦況指標 8 個 + 形分類 1 個。
# 機能 7 個は実装本格、戦況 state-holding 系は中立値 0.5 で API のみ準備
# (phase_e_collect 統合は H2 タスクで実装予定)。
# ============================

# ----- 一次軸: 機能能力指標 7 個 (Capability Indicators) -----
# 即発火可能な連鎖数 (= 現連鎖数)
INDICATOR_READY_CHAIN: str = "ready_chain_count"
# 何手 (puyo) で発火可能か (近いほど高スコア)
INDICATOR_IGNITION_DISTANCE: str = "ignition_distance"
# 即発火した場合の威力 (ojama 換算)
INDICATOR_CURRENT_FIRE_POWER: str = "current_fire_power"
# 完成後の最大威力 (盤面 puyo を理想配置した場合の飽和連鎖量)
INDICATOR_MAXIMUM_FIRE_POWER: str = "maximum_fire_power"
# 中盤応答能力 (sub_chain + 残量) / 1催促消費 puyo 数
INDICATOR_MID_GAME_RESPONSE: str = "mid_game_response_capacity"
# 即催促打てる小連鎖 (2-4 連鎖) の存在度
INDICATOR_HARASS_READINESS: str = "harassment_readiness"
# 仮想 ojama 10/20/30 後の本線残連鎖率 + 掘削可能性
INDICATOR_OJAMA_DEFENSE: str = "ojama_defense_capacity"

# ----- 二次軸: 戦況・タイミング指標 8 個 -----
# 自連鎖時間 (frame、落下マス + クイック判定)
INDICATOR_SELF_CHAIN_DURATION: str = "self_chain_duration_frames"
# 相手連鎖時間 (frame)
INDICATOR_OPP_CHAIN_DURATION: str = "opp_chain_duration_frames"
# 連鎖時間差: (opp_duration - self_duration) を応答可能 puyo 数差で正規化
INDICATOR_CHAIN_DURATION_ADV: str = "chain_duration_advantage"
# 直近 30s の催促回数 (state-holding 必要、現状は 0.5 中立)
INDICATOR_HARASS_COUNT_30S: str = "harass_event_count_30s"
# 試合開始 +30s 内の発火 + ojama 送信 (state-holding、0.5 中立)
INDICATOR_EARLY_AGGRESSION: str = "early_aggression_score"
# 直前 ojama 受信 + 直後本線発火パターン (state-holding、0.5 中立)
INDICATOR_COUNTER_IGNITION: str = "counter_ignition_signal"
# 序盤全消し検出 + 経過秒
INDICATOR_POST_ALL_CLEAR: str = "post_all_clear_state"
# 上部 (10+ 段) puyo 数 / 上 4 段全セル
INDICATOR_UPPER_DENSITY: str = "upper_board_density"

# ----- 三次軸: 形分類指標 1 個 (form_gtr 延長) -----
# GTR 折り返し位置: 0=先折り (1-2 列)、1=後折り (3-4 列)、2=自由形/不明
INDICATOR_GTR_ORIENTATION: str = "gtr_orientation"

ALL_INDICATOR_NAMES: tuple[str, ...] = (
    INDICATOR_MAIN_CHAIN,
    INDICATOR_EXTENSION,
    INDICATOR_SUB_CHAIN,
    INDICATOR_HARASSMENT,
    INDICATOR_DEATH_RISK,
    INDICATOR_OFFSET,
    INDICATOR_SECOND,
    INDICATOR_FIELD_EFF,
)

# 拡張指標 (next_acceptance 含む) のタプル。ALL_INDICATOR_NAMES とは独立に管理し、
# Scorer 側で EXTRA_INDICATOR_NAMES として参照する。
# 高度戦略指標 3 種 (key_flexibility / sub_chain_independence / chain_timing_pressure)
# も末尾に追加。
EXTRA_INDICATOR_NAMES: tuple[str, ...] = (
    INDICATOR_NEXT_ACCEPTANCE,
    INDICATOR_SHAPE_SCORE,
    INDICATOR_TOUCHING_DENSITY,
    INDICATOR_TAIL_HEIGHT,
    INDICATOR_COLOR_VARIANCE,
    INDICATOR_KEY_FLEXIBILITY,
    INDICATOR_SUB_CHAIN_INDEP,
    INDICATOR_CHAIN_TIMING,
    INDICATOR_INCOMING_OJAMA,
    # Phase J 追加 (2026-04-27)
    INDICATOR_OPPONENT_THREAT,
    INDICATOR_HEIGHT_DIFF,
    INDICATOR_HIGH_CONNECTION,
    INDICATOR_REQUIRED_FIRE,
    # Tier B 追加 (2026-05-05、形質指標、二相性分離)
    # planning_entropy は 2026-05-06 高速化で再有効化 (numpy 化 + cache)
    INDICATOR_PLANNING_ENTROPY,
    INDICATOR_STRUCTURE_SOLIDITY,
    INDICATOR_BASE_FLATNESS,
    # I-J 形テンプレ完成度 (2026-05-06、B-1)
    # GTR / LLR / 階段 / 座布団 4 テンプレに対する一致度 (1P/2P best score)
    INDICATOR_FORM_GTR,
    INDICATOR_FORM_LLR,
    INDICATOR_FORM_STAIRCASE,
    INDICATOR_FORM_ZABUTON,
    # Phase F 追加 (2026-05-07、B-4) 回し入れ巧拙
    INDICATOR_ROTATION_SKILL,
    # ※ Phase H1 系列の後ろに B-1.b の Sullen GTR / Fron を追加 (順序保持)
    # Phase H1 追加 (2026-05-08) 機能能力指標 7 個 (Capability)
    INDICATOR_READY_CHAIN,
    INDICATOR_IGNITION_DISTANCE,
    INDICATOR_CURRENT_FIRE_POWER,
    INDICATOR_MAXIMUM_FIRE_POWER,
    INDICATOR_MID_GAME_RESPONSE,
    INDICATOR_HARASS_READINESS,
    INDICATOR_OJAMA_DEFENSE,
    # Phase H1 追加 (2026-05-08) 戦況・タイミング指標 8 個 (Situational)
    INDICATOR_SELF_CHAIN_DURATION,
    INDICATOR_OPP_CHAIN_DURATION,
    INDICATOR_CHAIN_DURATION_ADV,
    INDICATOR_HARASS_COUNT_30S,
    INDICATOR_EARLY_AGGRESSION,
    INDICATOR_COUNTER_IGNITION,
    INDICATOR_POST_ALL_CLEAR,
    INDICATOR_UPPER_DENSITY,
    # Phase H1 追加 (2026-05-08) 形分類指標 1 個 (Form Classification)
    INDICATOR_GTR_ORIENTATION,
    # B-1.b 追加 (2026-05-09) citrus610/ama 由来の Sullen GTR / Fron 形完成度
    # 既存 LEARNED_WEIGHTS_* (dict 形式) は新指標キーが無くても影響なし
    INDICATOR_FORM_SULLEN_GTR,
    INDICATOR_FORM_FRON,
)
# Phase K 指標は IndicatorSet に保持するが、学習 CSV からは除外
# (ablation で他指標と多重共線性が高く、学習精度を下げるため)
# 推論ロジック (ojama 整合性チェック等) では引き続き利用。
PHASE_K_EXTRA_INDICATOR_NAMES: tuple[str, ...] = (
    INDICATOR_OPPONENT_OFFSET,
    INDICATOR_POST_OJAMA_HEALTH,
    INDICATOR_ISOLATED_PUYO,
)

# ============================
# 正規化用定数
# ============================

# 本線完成度: MAX_EXPECTED_CHAIN 連鎖で 1.0
# 10連鎖で満点 (試合中の「完成した本線」は 8-12連鎖が目安)
MAX_EXPECTED_CHAIN: int = 10

# 相殺力: MAX_OJAMA_OFFSET おじゃま相当で 1.0
# 1画面=30個 × 2.4 = 72個で満点 (実戦での「決め手級本線」想定)
MAX_OJAMA_OFFSET: int = 72
# 連鎖ボーナステーブル (ぷよぷよ公式: 1,8,16,32,64,96,128,160,192,224,...)
CHAIN_POWER_TABLE: tuple[int, ...] = (
    0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512,
)
# 連結ボーナス: size→bonus (ぷよぷよ公式: 4=0, 5=2, 6=3, 7=4, 8=5, 9=6, 10=7, 11+=10)
CONNECTION_BONUS_TABLE: dict[int, int] = {
    4: 0, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
}
CONNECTION_BONUS_MAX: int = 10  # 11連結以上で 10 で頭打ち (簡易)
# 色数ボーナス: colors→bonus (ぷよぷよ公式: 1=0, 2=3, 3=6, 4=12, 5=24)
COLOR_BONUS_TABLE: dict[int, int] = {
    1: 0, 2: 3, 3: 6, 4: 12, 5: 24,
}
# ボーナス最小値 1 クランプ (公式式で max(bonus, 1))
MIN_CHAIN_BONUS: int = 1
# 1おじゃま = OJAMA_DIVISOR 点 (ぷよぷよ標準: 70点/個)
OJAMA_DIVISOR: float = 70.0
# 1ぷよあたりの基本得点 (ぷよぷよ: 10点/個)
BASE_SCORE_PER_PUYO: float = 10.0

# 催促耐性: 10〜30個を 5個刻みでテスト
HARASSMENT_OJAMA_MIN: int = 10
HARASSMENT_OJAMA_MAX: int = 30
HARASSMENT_OJAMA_STEP: int = 5
# 本線未構築 (chain=0) は中立値 0.5 を返す (旧: col2高さで過大評価)
HARASSMENT_NEUTRAL_SCORE: float = 0.5

# 窒息リスク: 各列への重み (致命列=2 が最大)
DEATH_RISK_WEIGHTS: dict[int, float] = {
    0: 0.3,
    1: 0.6,
    2: 1.0,
    3: 0.6,
    4: 0.3,
    5: 0.2,
}
# 危険帯(高さ≥DEATH_RISK_DANGER_HEIGHT) で指数的上昇する非線形マップ
# risk = 1 - exp(-((h - thr)^2) / scale) を合成して線形分を補強する
DEATH_RISK_DANGER_HEIGHT: int = 9
DEATH_RISK_EXP_SCALE: float = 8.0
# 線形寄与と非線形寄与のブレンド比 (線形 + 非線形 = クランプ後 max)
DEATH_RISK_NONLINEAR_WEIGHT: float = 0.7
DEATH_RISK_LINEAR_WEIGHT: float = 0.3

# 伸ばし余地: ツモ探索で本線延長に寄与する placement 数
# 単一ツモ (1puyo) を色×列で試行し、連鎖数が伸びた placement の割合で評価
EXTENSION_IMPROVEMENT_WEIGHT: float = 0.7  # 連鎖延長による貢献度
EXTENSION_RESERVE_WEIGHT: float = 0.3      # 空きセル余裕の貢献度
EXTENSION_MIN_RESERVE_CELLS: int = 24      # 4段分の空きを確保できれば満点
MAX_COLORS_EXPECTED: int = 4               # ぷよぷよeスポーツ標準

# 副砲の質: 独立発火シミュレーションの合格ライン
# viable判定: 連鎖 >=2 かつ 消去 >=8 (単発4個消しは副砲と呼べない)
SUB_CHAIN_MIN_ERASE_CREDIT: int = 8
SUB_CHAIN_MIN_CHAIN_COUNT: int = 2
SUB_CHAIN_MAX_CANDIDATES: int = 6          # 探索するグループ上限
SUB_CHAIN_TRIGGER_SIZE: int = 3            # 「もう1つで消える」予備軍

# フィールド効率: 連鎖発火条件 (連結サイズ>=4) のぷよ数 / 全ぷよ数。
# I-G (2026-05-05): 旧 2 だと既に発火済みではない連結まで拾うため、
# 連鎖発火に直接寄与する連結サイズ >=4 に修正。指標が「実際に発火する
# ぷよの効率」を表すよう意味的整合を確保。
FIELD_EFF_MIN_CLUSTER_SIZE: int = 4

# セカンド構築力: 本線発火後の追加ツモ連鎖探索
SECOND_CHAIN_SEARCH_PUYOS: int = 2         # 追加投下する puyo 数
SECOND_CHAIN_MAX_EXPECTED: int = 3         # 期待連鎖数 (分母)
SECOND_CHAIN_EMPTY_BOARD_VIABILITY: float = 0.1  # 空盤面ボーナス (構築余地)

# ネクスト受け入れ余地 (next_acceptance) 用定数
# next_pair / dnext_pair が None の場合の中立値
NEXT_ACCEPTANCE_NEUTRAL: float = 0.5
# ペアの回転は 4 通り: 0=縦TOP上, 1=横TOP左, 2=縦TOP下, 3=横TOP右
PAIR_ROTATIONS: tuple[int, ...] = (0, 1, 2, 3)
# 探索時の連鎖伸長量を best_chain - base_chain で計算し
# MAX_EXPECTED_CHAIN で割って 0〜1 に正規化
NEXT_ACCEPTANCE_MAX_DELTA: int = MAX_EXPECTED_CHAIN

# ============================
# 拡張指標用定数 (先行研究ベース)
# ============================

# 形評価: U字形理想プロファイル (中央 col2,3 が低く 端 col0,5 が高い)
# mayah FIELD_USHAPE_LINEAR の簡易再現。理想偏差 0 で score=1.0
SHAPE_IDEAL_HEIGHTS: tuple[int, ...] = (10, 9, 7, 7, 9, 10)
# 偏差合計の正規化分母 (経験的に 30 程度で 0 寄り)
SHAPE_MAX_DEVIATION: float = 24.0

# 接ぷよ密度: ペア数 / (全ぷよ数 - 1) を 0〜1 で正規化
# 上限値: 平均 1.5 隣接 (ぷよ1個あたり水平+垂直で2ペア寄与)
TOUCHING_MAX_RATIO: float = 1.5

# 連鎖発火点高さ: 最終発火列高さ ÷ BOARD_ROWS
# 0=最下段(理想)、1=最上段(危険)。スコアは 1 - height_ratio で「低いほど高評価」
TAIL_HEIGHT_MAX_ROW: int = BOARD_ROWS

# 色分散: 各色の重心からの平均距離 ÷ MAX_DISPERSION で正規化
# 上限経験値: 13×6 盤面で対角約 14、平均距離 4 でほぼ最大分散
COLOR_VARIANCE_MAX_DIST: float = 4.0
# 1 色のサンプル数が COLOR_VARIANCE_MIN_COUNT 未満なら集計対象外
COLOR_VARIANCE_MIN_COUNT: int = 3

# ----- 高度戦略指標用定数 -----
# 探索対象色 (eスポーツ標準4色 + 紫を含めた標準5色)
KEY_FLEX_TRIAL_COLORS: tuple[int, ...] = (
    COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE,
)

# 副砲独立性: 期待最大副砲連鎖数 (これで割って 0〜1 正規化)
SUB_CHAIN_INDEP_MAX_EXPECTED: int = 3

# 連鎖タイミング圧: 「あと N 個で発火可能」の N を [0, MAX] にマッピング。
# N=0 (即発火) で 1.0、N=MAX 以上で 0.0。
CHAIN_TIMING_MAX_PUYO_GAP: int = 6
# 発火探索で試す追加 puyo 数の上限
CHAIN_TIMING_TRIAL_LIMIT: int = 6

# スコアクランプ範囲
SCORE_MIN: float = 0.0
SCORE_MAX: float = 1.0

# ----- Phase F (C-3) 相手盤面コンテキスト調整用定数 -----
# SecondChainPotential: 相手が大型本線 (chain_count >= この値) を持つとき
# 第二連鎖の必要性が上がる閾値。SUB_CHAIN_INDEP_MAX_EXPECTED と同水準を採用。
SECOND_CHAIN_OPP_THREAT_CHAIN_THRESHOLD: int = 3
# 相手脅威時の score 強化倍率 (1.2x、ただし最終 clamp で 1.0 上限)
SECOND_CHAIN_OPP_THREAT_BOOST: float = 1.2

# ExtensionPotential: 相手連鎖数による improvement_ratio 減衰係数の最大値
# decay = 1.0 - 0.3 * min(opp_chain / MAX_EXPECTED_CHAIN, 1.0)
EXTENSION_OPP_DECAY_MAX: float = 0.3

# ChainTimingPressure: 相手 min_n を考慮した相対 pressure の中央値 (0.5)
CHAIN_TIMING_RELATIVE_CENTER: float = 0.5

# Phase F (B-4) 回し入れ判定の履歴長 (RotationTracker と共有)
ROTATION_TRACKER_MAX_HISTORY: int = 20

# ============================
# Phase H1 用定数 (2026-05-08)
# ============================

# ----- 機能能力指標 (Capability) -----
# IgnitionDistance: 探索する追加 puyo 数の上限 (近いほど高スコア)
IGNITION_DISTANCE_MAX_PUYOS: int = 6
# IgnitionDistance: 「届かない」(MAX 超え) を表すマーカー値
IGNITION_DISTANCE_UNREACHABLE: int = IGNITION_DISTANCE_MAX_PUYOS + 1
# MaximumFirePower: ぷよ 4 個で 1 連鎖が組める ratio の経験値
# (4 puyo / 1 chain。色分散があれば連鎖数増)
MAX_FIRE_PUYO_PER_CHAIN: float = 4.0
# MaximumFirePower: 色多様性ボーナス (色数 × 0.5 連鎖追加期待)
MAX_FIRE_COLOR_DIVERSITY_BONUS: float = 0.5
# MaximumFirePower: 1 puyo の理想配置で削れる係数の上限
# (実盤面でも MAX_EXPECTED_CHAIN を超える連鎖は稀のため上限)
MAX_FIRE_CHAIN_CEIL: int = MAX_EXPECTED_CHAIN
# MidGameResponseCapacity: 1 催促で消費する puyo 数の経験値
MID_GAME_HARASS_PUYO_COST: float = 6.0
# HarassmentReadiness: 即催促可能と認める最小連鎖数 (sub_chain と整合)
HARASS_READINESS_MIN_CHAIN: int = 2
# HarassmentReadiness: 即催促可能と認める最大連鎖数 (これを超えると本線扱い)
HARASS_READINESS_MAX_CHAIN: int = 4
# HarassmentReadiness: 候補ojama 数の正規化分母 (経験値)
HARASS_READINESS_NORM_OJAMA: float = 12.0
# OjamaDefenseCapacity: 仮想 ojama テスト個数 (10/20/30)
OJAMA_DEFENSE_TEST_COUNTS: tuple[int, ...] = (10, 20, 30)
# OjamaDefenseCapacity: 受け切り (掘削) 可能と判定する基準 (連鎖発火可能)
OJAMA_DEFENSE_DIG_MIN_CHAIN: int = 1

# ----- 戦況・タイミング指標 (Situational) -----
# SelfChainDuration / OppChainDuration: 1 連鎖あたりの平均 frame 数
# (公式: ぷよ落下 + アニメ込み 1 連鎖 ≈ 80-86 frame)
CHAIN_DURATION_FRAMES_PER_CHAIN: float = 84.0
# 60 fps 想定で frame 上限 (10 連鎖で 840 frame ≒ 14 秒)
CHAIN_DURATION_NORM_MAX_FRAMES: float = (
    CHAIN_DURATION_FRAMES_PER_CHAIN * MAX_EXPECTED_CHAIN
)
# ChainDurationAdvantage: ツモ間隔 (フィールド到達まで ≈ 2 秒 = 120 frame)
CHAIN_DURATION_TUMO_INTERVAL_FRAMES: float = 120.0
# 応答可能 puyo 数差の正規化幅 (経験値、左右 6 個程度の幅で 0..1 マップ)
CHAIN_DURATION_ADV_NORM_PUYOS: float = 6.0
# state-holding 系の中立値
SITUATIONAL_NEUTRAL_SCORE: float = 0.5
# PostAllClear: 序盤と判定する経過秒上限 (これ以降は中立)
POST_ALL_CLEAR_EARLY_GAME_MAX_SEC: float = 60.0
# UpperBoardDensity: 上部とみなす row 数 (上 4 段)
UPPER_DENSITY_TOP_ROWS: int = 4
# UpperBoardDensity: 高密度危険ライン (10+ 段 = row >= 10)
UPPER_DENSITY_HIGH_ROW_START: int = BOARD_ROWS - UPPER_DENSITY_TOP_ROWS  # 13-4=9

# ----- 形分類指標 (Form Classification) -----
# GtrOrientation: 出力コード (0=先折り, 1=後折り, 2=自由形/不明)
GTR_ORIENTATION_FRONT: int = 0   # 先折り (左下 col 0-1 で折り返し)
GTR_ORIENTATION_BACK: int = 1    # 後折り (col 4-5 で折り返し)
GTR_ORIENTATION_FREE: int = 2    # 自由形/不明
# スコア化: front=1.0, back=0.5, free=0.0 で「組み戦略の確立度」を表現
GTR_ORIENTATION_SCORE_FRONT: float = 1.0
GTR_ORIENTATION_SCORE_BACK: float = 0.5
GTR_ORIENTATION_SCORE_FREE: float = 0.0
# 折り返し検出のための form_gtr 最低スコア (この値未満は free 判定)
GTR_ORIENTATION_FORM_THRESHOLD: float = 0.4

# ============================
# Phase G (C-1) 確率版 indicator 用定数
# ============================

# ExtensionPotential 確率版の Monte Carlo サンプル数 (探索コスト高いので少なめ)
EXTENSION_PROB_SAMPLES: int = 5
# DeathRisk 確率版で「セルに ぷよが存在する」とみなす確率閾値の補完計算では
# P(non-empty) = 1 - P(EMPTY) を直接使う (個別閾値不要)。


# ============================
# データクラス
# ============================


@dataclass(frozen=True)
class IndicatorResult:
    """
    1つの指標の計算結果。

    Attributes:
        name: 指標名 (ALL_INDICATOR_NAMES のいずれか)。
        score: 0.0〜1.0 に正規化されたスコア。
        raw_value: 正規化前の生値 (デバッグ・可視化用)。
        detail: 指標固有の追加情報。
        next_acceptance: ネクスト受け入れ余地スコア (0.0〜1.0)。
            指標毎には付与せず、IndicatorCalculator 経由でメイン指標に
            付加情報として記録するためのオプショナル領域。デフォルト 0.0。
    """
    name: str
    score: float
    raw_value: float
    detail: dict[str, Any] = field(default_factory=dict)
    next_acceptance: float = 0.0


@dataclass
class IndicatorSet:
    """
    盤面1つに対する全指標の結果セット。

    Attributes:
        results: 指標名→IndicatorResult の辞書 (拡張指標もここに格納される)。
        next_acceptance: ネクスト/ダブルネクストを考慮した受け入れ余地スコア
            (0.0〜1.0)。next_pair が未指定なら NEXT_ACCEPTANCE_NEUTRAL=0.5。
        shape_score: 形評価 (U字形+土台らしさ) 0〜1。
        touching_density: 接ぷよ密度 0〜1。
        tail_height_score: 連鎖発火点低さ 0〜1 (発火点低いほど高い)。
        color_variance_score: 色集中度 0〜1 (まとまりが良いほど高い)。
        key_flexibility: キーぷよ柔軟性 (0〜1)。
        sub_chain_independence: 副砲独立性 (0〜1)。
        chain_timing_pressure: 連鎖タイミング圧 (0〜1、発火近接度)。
        incoming_ojama_pressure: 受け攻撃圧 (相手の予告お邪魔と
            自分の相殺力比較、0〜1)。Scorer 側で負の重みとして使用。
    """
    results: dict[str, IndicatorResult]
    next_acceptance: float = NEXT_ACCEPTANCE_NEUTRAL
    shape_score: float = 0.0
    touching_density: float = 0.0
    tail_height_score: float = 0.0
    color_variance_score: float = 0.0
    key_flexibility: float = 0.0
    sub_chain_independence: float = 0.0
    chain_timing_pressure: float = 0.0
    incoming_ojama_pressure: float = 0.0
    # Phase J 追加
    opponent_chain_threat: float = 0.0
    adjacent_height_diff: float = 0.0
    high_connection_count: float = 0.0
    required_puyo_to_fire: float = 0.0
    # Phase K 追加 (凝視深化)
    opponent_offset_power: float = 0.0
    post_ojama_chain_health: float = 0.0
    isolated_puyo_count: float = 0.0
    # Tier B 追加 (2026-05-05、形質 3 指標、二相性分離)
    planning_entropy: float = 0.0
    structure_solidity: float = 0.0
    base_flatness: float = 0.0
    # I-J 追加 (2026-05-06、B-1) 形テンプレ完成度
    form_gtr: float = 0.0
    form_llr: float = 0.0
    form_staircase: float = 0.0
    form_zabuton: float = 0.0
    # Phase F 追加 (2026-05-07、B-4) 回し入れ巧拙 (デフォルト neutral 0.5)
    rotation_skill: float = 0.5
    # Phase H1 追加 (2026-05-08) 機能能力 7 個
    ready_chain_count: float = 0.0
    ignition_distance: float = 0.0
    current_fire_power: float = 0.0
    maximum_fire_power: float = 0.0
    mid_game_response_capacity: float = 0.0
    harassment_readiness: float = 0.0
    ojama_defense_capacity: float = 0.0
    # Phase H1 追加 (2026-05-08) 戦況 8 個 (state-holding 系は 0.5 中立)
    self_chain_duration_frames: float = 0.0
    opp_chain_duration_frames: float = 0.0
    chain_duration_advantage: float = SITUATIONAL_NEUTRAL_SCORE
    harass_event_count_30s: float = SITUATIONAL_NEUTRAL_SCORE
    early_aggression_score: float = SITUATIONAL_NEUTRAL_SCORE
    counter_ignition_signal: float = SITUATIONAL_NEUTRAL_SCORE
    post_all_clear_state: float = 0.0
    upper_board_density: float = 0.0
    # Phase H1 追加 (2026-05-08) 形分類 1 個
    gtr_orientation: float = GTR_ORIENTATION_SCORE_FREE
    # B-1.b 追加 (2026-05-09) citrus610/ama 由来の Sullen GTR / Fron
    form_sullen_gtr: float = 0.0
    form_fron: float = 0.0

    def get(self, name: str) -> IndicatorResult:
        """指標名から結果を取得する。"""
        return self.results[name]

    def score_of(self, name: str) -> float:
        """指標名から正規化スコアを取得する。"""
        return self.results[name].score

    def to_dict(self) -> dict[str, Any]:
        """JSON 保存可能な辞書に変換する。"""
        return {
            name: {
                "score": r.score,
                "raw_value": r.raw_value,
                "detail": r.detail,
            }
            for name, r in self.results.items()
        }


# ============================
# BaseIndicator
# ============================


class BaseIndicator(ABC):
    """
    指標計算の抽象基底クラス。

    各指標は本クラスを継承し name プロパティと compute() を実装する。
    IndicatorCalculator が差し替え可能に利用する。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """指標名 (ALL_INDICATOR_NAMES のいずれか)。"""
        ...

    @abstractmethod
    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
    ) -> IndicatorResult:
        """
        指標を計算する。

        Args:
            board: 評価対象の盤面。
            chain_result: 事前計算済みの連鎖結果 (再利用で高速化)。
            simulator: 仮想発火等に利用するシミュレータ。

        Returns:
            IndicatorResult: 計算結果。
        """
        ...

    @staticmethod
    def _clamp(value: float) -> float:
        """値を SCORE_MIN〜SCORE_MAX にクランプする。"""
        return max(SCORE_MIN, min(SCORE_MAX, value))

    @staticmethod
    def _ensure_chain(
        board: Board,
        chain_result: ChainResult | None,
        simulator: ChainSimulator | None,
    ) -> tuple[ChainResult, ChainSimulator]:
        """chain_result/simulator を必要に応じて生成する。"""
        sim = simulator or ChainSimulator()
        result = chain_result if chain_result is not None else sim.simulate(board)
        return result, sim

    def compute_probabilistic(
        self,
        prob_board: ProbabilisticBoard,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        **kwargs: Any,
    ) -> IndicatorResult:
        """確率版指標計算 (Phase G、C-1).

        デフォルトは MLE 盤面に変換し通常の compute() に委譲する。
        各 indicator が真の確率分布消費を実装する場合は本メソッドを
        オーバーライドすること。**kwargs は incoming_ojama や
        opponent_board 等を透過的に渡すため。
        """
        board = prob_board.to_max_likelihood_board()
        return self.compute(
            board, chain_result=chain_result, simulator=simulator, **kwargs,
        )


# ============================
# ユーティリティ関数
# ============================


def _count_cells(board: Board, predicate) -> int:
    """predicate(color) が True のセルを数える。"""
    count = 0
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if predicate(board.get(row, col)):
                count += 1
    return count


def _count_ojama(board: Board) -> int:
    """盤面のおじゃま数。"""
    return _count_cells(board, lambda c: c == COLOR_OJAMA)


def _color_counts(board: Board) -> dict[int, int]:
    """盤面の色別ぷよ数 (空・おじゃま除く)。"""
    counts: dict[int, int] = {}
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            c = board.get(row, col)
            if c != COLOR_EMPTY and c != COLOR_OJAMA:
                counts[c] = counts.get(c, 0) + 1
    return counts


def _present_colors(board: Board) -> list[int]:
    """盤面に存在する通常色 (おじゃま・空除く) を返す。"""
    return list(_color_counts(board).keys())


def _drop_row(board: Board, col: int) -> int | None:
    """
    指定列に puyo を 1 つ落とした時に着地する row を返す。

    Args:
        board: 対象盤面。
        col: 列番号 (0-5)。

    Returns:
        int | None: 着地 row (0 が最上段)。埋まっていれば None。
    """
    height = board.height_of(col)
    if height >= BOARD_ROWS:
        return None
    return BOARD_ROWS - 1 - height


def _simulate_with_placement(
    simulator: ChainSimulator,
    board: Board,
    col: int,
    color: int,
) -> ChainResult | None:
    """
    1 つ puyo を指定列に落下させた盤面で連鎖シミュレーションする。

    Args:
        simulator: シミュレータ。
        board: 元盤面 (変更しない)。
        col: 落下列。
        color: 落下ぷよの色。

    Returns:
        ChainResult | None: 置けなかった場合 None。
    """
    row = _drop_row(board, col)
    if row is None:
        return None
    work = board.copy()
    work.set(row, col, color)
    return simulator.simulate(work)


def _place_pair(
    board: Board,
    pair: tuple[int, int],
    col: int,
    rotation: int,
) -> Board | None:
    """
    指定列・回転で puyo ペアを盤面に下から積む。

    回転は 0=縦TOP上(下にBOT), 1=横TOP左(右にBOT), 2=縦TOP下(下にTOP),
    3=横TOP右(左にBOT)。横配置時は col が左 puyo の列。

    Args:
        board: 元盤面 (破壊しない)。
        pair: (TOP色, BOT色) の組。
        col: 配置列 (横回転時はペアの左 puyo 列)。
        rotation: 回転コード (0〜3)。

    Returns:
        Board | None: 不可なら None。新しい Board を返す。
    """
    top, bot = pair
    if top == COLOR_EMPTY or bot == COLOR_EMPTY:
        return None
    work = board.copy()
    if rotation in (0, 2):
        # 縦配置 (同列で 2 puyo 積む)
        if not (0 <= col < BOARD_COLS):
            return None
        upper, lower = (top, bot) if rotation == 0 else (bot, top)
        return _drop_two_in_column(work, col, upper, lower)
    # 横配置: col と col+1 にそれぞれ 1 puyo
    if not (0 <= col < BOARD_COLS - 1):
        return None
    left, right = (top, bot) if rotation == 1 else (bot, top)
    if _drop_one(work, col, left) is None:
        return None
    if _drop_one(work, col + 1, right) is None:
        return None
    return work


def _drop_one(board: Board, col: int, color: int) -> Board | None:
    """1 つ puyo を col の最下空セルに落下 (in-place)。失敗なら None。"""
    row = _drop_row(board, col)
    if row is None:
        return None
    board.set(row, col, color)
    return board


def _drop_two_in_column(
    board: Board, col: int, upper: int, lower: int,
) -> Board | None:
    """同一列に 2 puyo を積む (lower が下、upper が上、in-place)。"""
    if board.height_of(col) > BOARD_ROWS - 2:
        return None
    if _drop_one(board, col, lower) is None:
        return None
    if _drop_one(board, col, upper) is None:
        return None
    return board


def _compute_next_acceptance(
    board: Board,
    next_pair: tuple[int, int],
    dnext_pair: tuple[int, int],
    simulator: ChainSimulator,
) -> tuple[float, dict[str, Any]]:
    """
    ネクスト・ダブルネクスト受け入れ余地を計算する。

    next/dnext を全 (列×回転) で仮置きし、その後 dnext を全 (列×回転) で
    仮置きした際に達成できる最大連鎖数を base_chain との差分で評価。

    Args:
        board: 評価対象の盤面 (破壊しない)。
        next_pair: 次のツモ (TOP, BOT 色)。
        dnext_pair: ダブルネクストのツモ。
        simulator: 連鎖シミュレータ。

    Returns:
        (score, detail): score は 0.0〜1.0、detail はデバッグ情報。
    """
    base_chain = simulator.simulate(board).chain_count
    best_chain = base_chain
    for nrot in PAIR_ROTATIONS:
        ncols = range(BOARD_COLS) if nrot in (0, 2) else range(BOARD_COLS - 1)
        for ncol in ncols:
            board_n = _place_pair(board, next_pair, ncol, nrot)
            if board_n is None or board_n.is_dead():
                continue
            best_chain = max(
                best_chain,
                _best_after_dnext(board_n, dnext_pair, simulator),
            )
    delta = max(0, best_chain - base_chain)
    score = min(SCORE_MAX, delta / NEXT_ACCEPTANCE_MAX_DELTA)
    return score, {
        "base_chain": base_chain,
        "best_chain": best_chain,
        "delta": delta,
    }


def _best_after_dnext(
    board_after_next: Board,
    dnext_pair: tuple[int, int],
    simulator: ChainSimulator,
) -> int:
    """next 配置済み盤面に dnext を仮置きした際の最良連鎖数を返す。"""
    best = simulator.simulate(board_after_next).chain_count
    for drot in PAIR_ROTATIONS:
        dcols = range(BOARD_COLS) if drot in (0, 2) else range(BOARD_COLS - 1)
        for dcol in dcols:
            board_d = _place_pair(board_after_next, dnext_pair, dcol, drot)
            if board_d is None or board_d.is_dead():
                continue
            chain = simulator.simulate(board_d).chain_count
            if chain > best:
                best = chain
    return best


# ============================
# 具象 Indicator: 単純系
# ============================


class FieldEfficiencyIndicator(BaseIndicator):
    """
    フィールド効率: 連鎖参加ぷよ数 ÷ 通常ぷよ総数。

    未発火盤面 (chain=0) では participating=0 で常に 0 になってしまい、
    土台評価に寄与しない。そこで連鎖が発生しないケースは
    「連結サイズ >= FIELD_EFF_MIN_CLUSTER_SIZE のぷよ数 / 通常ぷよ総数」を
    代替指標として返す。孤立ぷよが少ないほど (= まとまっているほど) 高評価。
    """

    @property
    def name(self) -> str:
        return INDICATOR_FIELD_EFF

    def compute(self, board, chain_result=None, simulator=None):
        chain_result, sim = self._ensure_chain(board, chain_result, simulator)

        ojama = _count_ojama(board)
        normal_total = board.count_puyos() - ojama
        participating = chain_result.participating_cells

        if normal_total == 0:
            ratio = 0.0
        elif participating > 0:
            ratio = participating / normal_total
        else:
            # 未発火盤面: 連結サイズ>=閾値のぷよ数で代替評価
            clustered = self._count_clustered_puyos(board, sim)
            ratio = clustered / normal_total

        return IndicatorResult(
            name=self.name,
            score=self._clamp(ratio),
            raw_value=ratio,
            detail={
                "participating": participating,
                "normal_puyos": normal_total,
                "ojama_puyos": ojama,
                "used_fallback": participating == 0 and normal_total > 0,
            },
        )

    @staticmethod
    def _count_clustered_puyos(board: Board, sim: ChainSimulator) -> int:
        """連結サイズ>=FIELD_EFF_MIN_CLUSTER_SIZE のグループに属するぷよ数を返す。"""
        total = 0
        for group in sim.find_groups(board):
            if group.size >= FIELD_EFF_MIN_CLUSTER_SIZE:
                total += group.size
        return total


class DeathRiskIndicator(BaseIndicator):
    """窒息リスク: 致命列・近傍の重み付き高さ平均 + 非線形な危険帯補正。

    線形成分 (平均高/行数) と、致命列 col=2 の高さに対する指数的上昇成分
    `1 - exp(-(h - thr)^2 / scale)` を重み付け合成する。
    これにより危険帯 (高さ 9-12) で指標が急激に増加する。
    """

    @property
    def name(self) -> str:
        return INDICATOR_DEATH_RISK

    def compute(self, board, chain_result=None, simulator=None):
        total_weight = sum(DEATH_RISK_WEIGHTS.values())
        weighted_height = 0.0
        col_heights: dict[int, int] = {}
        for col, weight in DEATH_RISK_WEIGHTS.items():
            h = board.height_of(col)
            col_heights[col] = h
            weighted_height += h * weight

        avg_height = weighted_height / total_weight
        linear_score = avg_height / BOARD_ROWS

        # 致命列 (col=2) の高さに基づく非線形成分
        col2_height = col_heights.get(2, 0)
        nonlinear_score = self._danger_curve(col2_height)

        blended = (
            DEATH_RISK_LINEAR_WEIGHT * linear_score
            + DEATH_RISK_NONLINEAR_WEIGHT * nonlinear_score
        )
        score = self._clamp(blended)

        # 窒息中なら最大リスク
        if board.is_dead():
            score = SCORE_MAX

        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=avg_height,
            detail={
                "col_heights": col_heights,
                "is_dead": board.is_dead(),
                "linear": linear_score,
                "nonlinear": nonlinear_score,
            },
        )

    @staticmethod
    def _danger_curve(height: int) -> float:
        """致命列高さ→非線形リスク (高さ<閾値で0, >=閾値で指数的上昇)。"""
        if height < DEATH_RISK_DANGER_HEIGHT:
            return 0.0
        delta = height - DEATH_RISK_DANGER_HEIGHT
        return 1.0 - math.exp(-(delta * delta) / DEATH_RISK_EXP_SCALE)

    def compute_probabilistic(
        self,
        prob_board: ProbabilisticBoard,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        **kwargs: Any,
    ) -> IndicatorResult:
        """確率版窒息リスク: 各列の期待高さを確率分布から直接計算.

        各セルの P(non-empty) = 1 - P(EMPTY) を行方向に積み上げ、
        列の期待高さ E[h(col)] を求める。
        DEATH_RISK_WEIGHTS で重み付き平均を取り、致命列 col=2 の
        期待高さで非線形成分を計算 (_danger_curve は連続値も受理可)。
        """
        col_expected_heights = self._compute_expected_heights(prob_board)
        total_weight = sum(DEATH_RISK_WEIGHTS.values())
        weighted_height = 0.0
        for col, weight in DEATH_RISK_WEIGHTS.items():
            weighted_height += col_expected_heights[col] * weight
        avg_height = weighted_height / total_weight
        linear_score = avg_height / BOARD_ROWS
        col2_height = col_expected_heights.get(2, 0.0)
        nonlinear_score = self._danger_curve_continuous(col2_height)
        blended = (
            DEATH_RISK_LINEAR_WEIGHT * linear_score
            + DEATH_RISK_NONLINEAR_WEIGHT * nonlinear_score
        )
        score = self._clamp(blended)
        # 確率版窒息判定: P(col=2 の row=0 が non-empty) が高い場合
        col2_top_cell = prob_board.cell(0, 2)
        p_dead = 1.0 - col2_top_cell.get(COLOR_EMPTY)
        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=avg_height,
            detail={
                "col_expected_heights": col_expected_heights,
                "p_dead": p_dead,
                "linear": linear_score,
                "nonlinear": nonlinear_score,
                "probabilistic": True,
            },
        )

    @staticmethod
    def _compute_expected_heights(
        prob_board: ProbabilisticBoard,
    ) -> dict[int, float]:
        """各列の期待高さ E[h(col)] を P(non-empty) の合計で計算する."""
        heights: dict[int, float] = {}
        for col in DEATH_RISK_WEIGHTS.keys():
            expected = 0.0
            for row in range(BOARD_ROWS):
                cell = prob_board.cell(row, col)
                p_non_empty = 1.0 - cell.get(COLOR_EMPTY)
                expected += p_non_empty
            heights[col] = expected
        return heights

    @staticmethod
    def _danger_curve_continuous(height: float) -> float:
        """致命列期待高さ→非線形リスクの連続版."""
        if height < DEATH_RISK_DANGER_HEIGHT:
            return 0.0
        delta = height - DEATH_RISK_DANGER_HEIGHT
        return 1.0 - math.exp(-(delta * delta) / DEATH_RISK_EXP_SCALE)


class MainChainMaturityIndicator(BaseIndicator):
    """本線完成度: 連鎖数 ÷ MAX_EXPECTED_CHAIN。"""

    @property
    def name(self) -> str:
        return INDICATOR_MAIN_CHAIN

    def compute(self, board, chain_result=None, simulator=None):
        chain_result, _ = self._ensure_chain(board, chain_result, simulator)

        count = chain_result.chain_count
        ratio = count / MAX_EXPECTED_CHAIN

        return IndicatorResult(
            name=self.name,
            score=self._clamp(ratio),
            raw_value=float(count),
            detail={
                "chain_count": count,
                "max_expected": MAX_EXPECTED_CHAIN,
                "total_erased": chain_result.total_erased,
            },
        )

    def compute_probabilistic(
        self,
        prob_board: ProbabilisticBoard,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        n_samples: int = PROBABILISTIC_DEFAULT_SAMPLES,
        **kwargs: Any,
    ) -> IndicatorResult:
        """確率版本線完成度: N サンプルの mean_chain_count を採用.

        Monte Carlo で得た平均連鎖数を MAX_EXPECTED_CHAIN で正規化する。
        chain_count の標準偏差・サンプル数を detail に含める。
        """
        sim = simulator or ChainSimulator()
        prob_result = sim.simulate_probabilistic(
            prob_board, n_samples=n_samples,
        )
        mean_chain = prob_result.mean_chain_count
        ratio = mean_chain / MAX_EXPECTED_CHAIN
        return IndicatorResult(
            name=self.name,
            score=self._clamp(ratio),
            raw_value=float(mean_chain),
            detail={
                "mean_chain_count": mean_chain,
                "std_chain_count": prob_result.std_chain_count,
                "n_samples": prob_result.n_samples,
                "max_expected": MAX_EXPECTED_CHAIN,
                "mean_erased": prob_result.mean_erased_puyos,
                "probabilistic": True,
            },
        )


# ============================
# 具象 Indicator: シミュレーション系
# ============================


class OffsetPowerIndicator(BaseIndicator):
    """相殺力: 即時発火で送れるおじゃま相当数 ÷ MAX_OJAMA_OFFSET。

    ぷよぷよ公式得点式を採用:
        step_score = 10 * erased * max(chain_power + group_bonus + color_bonus, 1)
    step 毎に集計し、おじゃま相当数 = total_score / OJAMA_DIVISOR。
    """

    @property
    def name(self) -> str:
        return INDICATOR_OFFSET

    def compute(self, board, chain_result=None, simulator=None):
        chain_result, _ = self._ensure_chain(board, chain_result, simulator)

        count = chain_result.chain_count
        erased = chain_result.total_erased

        total_score = self._total_chain_score(chain_result)
        ojama_equiv = total_score / OJAMA_DIVISOR if total_score > 0 else 0.0

        # 対数スケール: 旧 線形 ratio = ojama / 72 は 4連鎖で早期飽和し +1.00 張り付き多発。
        # log1p 変換で小〜中連鎖の差が取れる。参照点 72個で score=1.0 に揃える。
        # ratio_log = log1p(ojama) / log1p(MAX_OJAMA_OFFSET)
        if ojama_equiv > 0:
            ratio = math.log1p(ojama_equiv) / math.log1p(MAX_OJAMA_OFFSET)
        else:
            ratio = 0.0

        return IndicatorResult(
            name=self.name,
            score=self._clamp(ratio),
            raw_value=ojama_equiv,
            detail={
                "chain_count": count,
                "erased_puyos": erased,
                "estimated_ojama": int(ojama_equiv),
                "total_score": int(total_score),
                "scale": "log1p",
            },
        )

    @staticmethod
    def _total_chain_score(chain_result: ChainResult) -> float:
        """ChainResult から連結/色数/連鎖ボーナスを含む累計得点を計算する。"""
        total = 0.0
        for step in chain_result.steps:
            step_erased = step.erased_count
            if step_erased == 0:
                continue
            # 連鎖パワー (chain_index は 1-indexed)
            idx = step.chain_index
            if idx < len(CHAIN_POWER_TABLE):
                chain_power = CHAIN_POWER_TABLE[idx]
            else:
                chain_power = CHAIN_POWER_TABLE[-1]
            # 連結ボーナス (全グループの size→bonus 合算)
            group_bonus = 0
            colors_in_step: set[int] = set()
            for group in step.erased_groups:
                bonus = CONNECTION_BONUS_TABLE.get(group.size, CONNECTION_BONUS_MAX)
                group_bonus += bonus
                colors_in_step.add(group.color)
            # 色数ボーナス
            color_bonus = COLOR_BONUS_TABLE.get(
                len(colors_in_step), COLOR_BONUS_TABLE[max(COLOR_BONUS_TABLE)]
            )
            bonus_sum = max(chain_power + group_bonus + color_bonus, MIN_CHAIN_BONUS)
            total += BASE_SCORE_PER_PUYO * step_erased * bonus_sum
        return total


class HarassmentResistanceIndicator(BaseIndicator):
    """催促耐性: おじゃま 10〜30個落下後の本線残存度平均.

    C-3 (I-H、2026-05-06): incoming_ojama 指定時は実際の予告おじゃま量で
    重み付き平均を取る (2x の重み)。相手の現実的な攻撃量に基づく評価。
    """

    @property
    def name(self) -> str:
        return INDICATOR_HARASSMENT

    def compute(
        self, board, chain_result=None, simulator=None,
        incoming_ojama: int = 0,
    ):
        chain_result, sim = self._ensure_chain(board, chain_result, simulator)
        base_chain = chain_result.chain_count

        survival_scores: list[float] = []
        weights: list[float] = []
        details: dict[int, dict[str, Any]] = {}

        for count in range(
            HARASSMENT_OJAMA_MIN,
            HARASSMENT_OJAMA_MAX + 1,
            HARASSMENT_OJAMA_STEP,
        ):
            retention, is_dead, post_chain = self._evaluate_drop(
                board, sim, count, base_chain
            )
            survival_scores.append(retention)
            weights.append(1.0)
            details[count] = {"dead": is_dead, "chain": post_chain}

        # C-3: 実際の予告おじゃま量で追加評価 (2x 重み)
        if incoming_ojama > 0 and incoming_ojama not in details:
            retention, is_dead, post_chain = self._evaluate_drop(
                board, sim, incoming_ojama, base_chain
            )
            survival_scores.append(retention)
            weights.append(2.0)
            details[incoming_ojama] = {
                "dead": is_dead, "chain": post_chain, "incoming": True,
            }

        if survival_scores:
            wsum = sum(weights)
            avg = sum(s * w for s, w in zip(survival_scores, weights)) / wsum
        else:
            avg = 0.0

        return IndicatorResult(
            name=self.name,
            score=self._clamp(avg),
            raw_value=avg,
            detail={
                "base_chain": base_chain,
                "survival_by_count": details,
                "incoming_ojama": incoming_ojama,
            },
        )

    @staticmethod
    def _evaluate_drop(
        board: Board,
        sim: ChainSimulator,
        ojama_count: int,
        base_chain: int,
    ) -> tuple[float, bool, int]:
        """1段階のおじゃま落下を評価し (残存度, 窒息, 連鎖数) を返す。

        本線未構築 (base_chain=0) の場合は「守る本線がない」ため
        中立値 HARASSMENT_NEUTRAL_SCORE を返す (過大評価・過小評価を避ける)。
        窒息した場合のみ 0 を返す。
        """
        ojama_board = sim.drop_ojama(board, ojama_count)
        if ojama_board.is_dead():
            return 0.0, True, 0
        post = sim.simulate(ojama_board)
        if base_chain == 0:
            retention = HARASSMENT_NEUTRAL_SCORE
        else:
            retention = post.chain_count / base_chain
        return min(SCORE_MAX, retention), False, post.chain_count

    def compute_probabilistic(
        self,
        prob_board: ProbabilisticBoard,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        incoming_ojama: int = 0,
        n_samples: int = PROBABILISTIC_DEFAULT_SAMPLES,
        **kwargs: Any,
    ) -> IndicatorResult:
        """確率版催促耐性: N サンプル盤面で個別評価し score 平均.

        各サンプルで通常 compute() を実行し、平均 score / 平均 base_chain を取る。
        per-sample sub-scores を detail に格納する。
        """
        sim = simulator or ChainSimulator()
        rng = np.random.default_rng()
        per_sample_scores: list[float] = []
        per_sample_base_chains: list[int] = []
        for _ in range(n_samples):
            sampled = prob_board.sample_board(rng=rng)
            chain_res = sim.simulate(sampled)
            sub = self.compute(
                sampled,
                chain_result=chain_res,
                simulator=sim,
                incoming_ojama=incoming_ojama,
            )
            per_sample_scores.append(sub.score)
            per_sample_base_chains.append(chain_res.chain_count)
        if per_sample_scores:
            avg = sum(per_sample_scores) / len(per_sample_scores)
        else:
            avg = HARASSMENT_NEUTRAL_SCORE
        return IndicatorResult(
            name=self.name,
            score=self._clamp(avg),
            raw_value=avg,
            detail={
                "n_samples": n_samples,
                "per_sample_scores": per_sample_scores,
                "per_sample_base_chains": per_sample_base_chains,
                "incoming_ojama": incoming_ojama,
                "probabilistic": True,
            },
        )


# ============================
# 具象 Indicator: 探索系(簡易版)
# ============================


class ExtensionPotentialIndicator(BaseIndicator):
    """
    伸ばし余地: 1〜2 ツモ単純探索で本線を伸ばせる placement 比率 + 空きセル余裕。

    各 (色, 列) の単一ぷよ落下を試行し、現在の連鎖数を超える placement を
    "伸ばし貢献" と数える (1-step)。
    1-step で発火しなかった placement について、もう 1 ツモ追加して再試行する
    (2-step、B-2、2026-05-06、半重み)。
    空きセル数も加味し、長期構築余力を評価する。

    Phase F (C-3): opponent_board が与えられ相手が高 chain_maturity の場合、
    伸ばしの意義が減るため improvement_ratio に減衰係数 decay を掛ける。
        decay = 1.0 - EXTENSION_OPP_DECAY_MAX
                * min(opp_chain_count / MAX_EXPECTED_CHAIN, 1.0)
    opponent_board=None なら従来挙動を維持。
    """

    @property
    def name(self) -> str:
        return INDICATOR_EXTENSION

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        chain_result, sim = self._ensure_chain(board, chain_result, simulator)
        base_chain = chain_result.chain_count

        empty_cells = _count_cells(board, lambda c: c == COLOR_EMPTY)
        empty_reserve = min(
            SCORE_MAX, empty_cells / EXTENSION_MIN_RESERVE_CELLS,
        )

        improvement_ratio, n1, n2 = self._search_improvement_2step(
            board, sim, base_chain,
        )

        # Phase F (C-3): 相手脅威減衰
        opp_chain, decay = self._compute_opp_decay(opponent_board, sim)
        attenuated_ratio = improvement_ratio * decay

        score = self._clamp(
            EXTENSION_IMPROVEMENT_WEIGHT * attenuated_ratio
            + EXTENSION_RESERVE_WEIGHT * empty_reserve
        )

        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=improvement_ratio,
            detail={
                "base_chain": base_chain,
                "improvement_ratio": improvement_ratio,
                "improved_1step": n1,
                "improved_2step": n2,
                "empty_reserve": empty_reserve,
                "empty_cells": empty_cells,
                "opp_chain_count": opp_chain,
                "opp_decay": decay,
            },
        )

    @staticmethod
    def _compute_opp_decay(
        opponent_board: Board | None,
        sim: ChainSimulator,
    ) -> tuple[int, float]:
        """相手連鎖数に基づく減衰係数 decay を計算する.

        Returns:
            (opp_chain_count, decay)。opponent_board=None なら (0, 1.0)。
        """
        if opponent_board is None:
            return 0, 1.0
        try:
            opp_result = sim.simulate(opponent_board)
        except Exception:
            return 0, 1.0
        opp_chain = opp_result.chain_count
        ratio = min(opp_chain / float(MAX_EXPECTED_CHAIN), 1.0)
        decay = 1.0 - EXTENSION_OPP_DECAY_MAX * ratio
        return opp_chain, decay

    @staticmethod
    def _search_improvement_2step(
        board: Board,
        sim: ChainSimulator,
        base_chain: int,
    ) -> tuple[float, int, int]:
        """1〜2 ツモ探索で base_chain を超える比率を返す.

        Returns:
            (improvement_ratio, improved_1step, improved_2step)
        """
        colors = _present_colors(board)
        if not colors:
            return 0.0, 0, 0
        total = 0
        improved_1step = 0
        improved_2step = 0
        # 1-step 探索で発火しなかった盤面を保存して 2-step 探索する
        unfired_boards: list[Board] = []
        for color in colors:
            for col in range(BOARD_COLS):
                row = _drop_row(board, col)
                if row is None:
                    continue
                total += 1
                trial1 = board.copy()
                trial1.set(row, col, color)
                result = sim.simulate(trial1)
                if result.chain_count > base_chain:
                    improved_1step += 1
                else:
                    unfired_boards.append(trial1)
        # 2-step: 1-step で発火しなかった盤面のうち、追加 1 ぷよで発火するか
        # ChainSimulator のキャッシュで重複盤面は高速化される。
        for trial1 in unfired_boards:
            for color in colors:
                fired_2step = False
                for col in range(BOARD_COLS):
                    row2 = _drop_row(trial1, col)
                    if row2 is None:
                        continue
                    trial2 = trial1.copy()
                    trial2.set(row2, col, color)
                    if sim.simulate(trial2).chain_count > base_chain:
                        improved_2step += 1
                        fired_2step = True
                        break
                if fired_2step:
                    break
        if total == 0:
            return 0.0, 0, 0
        # 2-step は 1-step より確度が低いので半重み
        ratio = (improved_1step + 0.5 * improved_2step) / total
        return ratio, improved_1step, improved_2step

    def compute_probabilistic(
        self,
        prob_board: ProbabilisticBoard,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
        n_samples: int = EXTENSION_PROB_SAMPLES,
        **kwargs: Any,
    ) -> IndicatorResult:
        """確率版伸ばし余地: MLE 盤面 + N サンプルの improvement_ratio 平均.

        探索コストが高いため n_samples=EXTENSION_PROB_SAMPLES (=5) と
        小さめ。MLE 代表盤面でも 1 回計算し、サンプル平均と統合する。
        """
        sim = simulator or ChainSimulator()
        # MLE 盤面と N サンプルを結合して improvement_ratio 平均を取る
        mle_board = prob_board.to_max_likelihood_board()
        ratios: list[float] = []
        details_per_sample: list[dict[str, int]] = []
        # MLE 盤面の評価 (重み 1)
        mle_chain = sim.simulate(mle_board).chain_count
        mle_ratio, n1, n2 = self._search_improvement_2step(
            mle_board, sim, mle_chain,
        )
        ratios.append(mle_ratio)
        details_per_sample.append({"n1": n1, "n2": n2, "base": mle_chain})
        # 残り N-1 個のランダムサンプル
        rng = np.random.default_rng()
        for _ in range(max(0, n_samples - 1)):
            sampled = prob_board.sample_board(rng=rng)
            base_chain = sim.simulate(sampled).chain_count
            ratio, n1, n2 = self._search_improvement_2step(
                sampled, sim, base_chain,
            )
            ratios.append(ratio)
            details_per_sample.append(
                {"n1": n1, "n2": n2, "base": base_chain}
            )
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        if len(ratios) > 1:
            arr = np.array(ratios, dtype=np.float64)
            std_ratio = float(arr.std())
        else:
            std_ratio = 0.0
        # 空きセルは MLE 盤面ベースで計算 (確定的近似)
        empty_cells = _count_cells(mle_board, lambda c: c == COLOR_EMPTY)
        empty_reserve = min(
            SCORE_MAX, empty_cells / EXTENSION_MIN_RESERVE_CELLS,
        )
        opp_chain, decay = self._compute_opp_decay(opponent_board, sim)
        attenuated_ratio = mean_ratio * decay
        score = self._clamp(
            EXTENSION_IMPROVEMENT_WEIGHT * attenuated_ratio
            + EXTENSION_RESERVE_WEIGHT * empty_reserve
        )
        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=mean_ratio,
            detail={
                "improvement_ratio_mean": mean_ratio,
                "improvement_ratio_std": std_ratio,
                "n_samples": len(ratios),
                "empty_reserve": empty_reserve,
                "empty_cells": empty_cells,
                "opp_chain_count": opp_chain,
                "opp_decay": decay,
                "per_sample": details_per_sample,
                "probabilistic": True,
            },
        )


class SubChainQualityIndicator(BaseIndicator):
    """
    副砲の質: 現盤面から独立して発火可能な小連鎖の強さを評価する。

    手順:
        1. 窒息盤面は副砲評価不可 (score=0 で即時終了)
        2. size>=3 のグループごとに「あと 1 個で消える」候補を列挙
        3. 候補に対し 1 puyo 追加の placement を試行し、
           連鎖 >= 2 かつ 消去 >= 8 なら viable (単発消しは副砲と呼べない)
        4. viable placement 数 / SUB_CHAIN_MAX_CANDIDATES を品質とする
    """

    @property
    def name(self) -> str:
        return INDICATOR_SUB_CHAIN

    def compute(self, board, chain_result=None, simulator=None):
        _, sim = self._ensure_chain(board, chain_result, simulator)

        # 窒息盤面では副砲の議論が成立しない (危険盤面を副砲扱いしない)
        if board.is_dead():
            return IndicatorResult(
                name=self.name,
                score=SCORE_MIN,
                raw_value=0.0,
                detail={
                    "viable_count": 0,
                    "candidate_count": 0,
                    "best_chain": 0,
                    "board_dead": True,
                },
            )

        candidates = self._collect_candidates(board, sim)
        viable: list[dict[str, Any]] = []
        for color, col in candidates:
            result = _simulate_with_placement(sim, board, col, color)
            if result is None:
                continue
            if (
                result.chain_count >= SUB_CHAIN_MIN_CHAIN_COUNT
                and result.total_erased >= SUB_CHAIN_MIN_ERASE_CREDIT
            ):
                viable.append({
                    "color": color,
                    "col": col,
                    "chain": result.chain_count,
                    "erased": result.total_erased,
                })

        score = self._clamp(len(viable) / SUB_CHAIN_MAX_CANDIDATES)

        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=float(len(viable)),
            detail={
                "viable_count": len(viable),
                "candidate_count": len(candidates),
                "best_chain": max((v["chain"] for v in viable), default=0),
                "board_dead": False,
            },
        )

    @staticmethod
    def _collect_candidates(
        board: Board, sim: ChainSimulator,
    ) -> list[tuple[int, int]]:
        """発火候補となる (色, 列) ペアを列挙する。"""
        candidates: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        groups = sim.find_groups(board)
        for group in groups:
            if group.size < SUB_CHAIN_TRIGGER_SIZE:
                continue
            # グループに隣接する空きセルのある列を発火候補にする
            for (row, col) in group.cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS):
                        continue
                    if board.get(nr, nc) != COLOR_EMPTY:
                        continue
                    key = (group.color, nc)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(key)
                    if len(candidates) >= SUB_CHAIN_MAX_CANDIDATES * 2:
                        return candidates
        return candidates


class SecondChainPotentialIndicator(BaseIndicator):
    """
    セカンド構築力: 本線発火後の残ぷよから第二連鎖を組める余地を探索する。

    手順:
        1. 現盤面の連鎖シミュレーション後の final_board を取得
        2. final_board に全色 × 全列で単発 puyo を試し、連鎖が発生する placement を計測
        3. 最良連鎖数と placement 成功率の重み付け平均で評価

    Phase F (C-3): opponent_board が与えられ、相手の chain_count が
    SECOND_CHAIN_OPP_THREAT_CHAIN_THRESHOLD 以上なら第二連鎖の必要性が
    上昇するため score を SECOND_CHAIN_OPP_THREAT_BOOST 倍に強化する
    (clamp で 1.0 上限)。opponent_board=None なら従来挙動。
    """

    @property
    def name(self) -> str:
        return INDICATOR_SECOND

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        chain_result, sim = self._ensure_chain(board, chain_result, simulator)

        remaining = chain_result.final_board
        present = _present_colors(remaining)

        # 空盤面は「これから構築可能」として最小ボーナス付与
        if not present:
            base_score = SECOND_CHAIN_EMPTY_BOARD_VIABILITY
            opp_chain, boosted = self._apply_opp_boost(
                base_score, opponent_board, sim,
            )
            return IndicatorResult(
                name=self.name,
                score=self._clamp(boosted),
                raw_value=0.0,
                detail={
                    "viable_placements": 0, "best_chain": 0,
                    "remaining_empty": True,
                    "opp_chain_count": opp_chain,
                    "opp_threat_boost_applied": opp_chain
                    >= SECOND_CHAIN_OPP_THREAT_CHAIN_THRESHOLD,
                },
            )

        viable_placements = 0
        best_chain = 0
        total_trials = 0
        for color in present:
            for col in range(BOARD_COLS):
                result = _simulate_with_placement(sim, remaining, col, color)
                if result is None:
                    continue
                total_trials += 1
                if result.chain_count >= 1:
                    viable_placements += 1
                    best_chain = max(best_chain, result.chain_count)

        placement_ratio = (
            viable_placements / total_trials if total_trials > 0 else 0.0
        )
        chain_ratio = best_chain / SECOND_CHAIN_MAX_EXPECTED

        base_score = 0.5 * placement_ratio + 0.5 * chain_ratio
        opp_chain, boosted = self._apply_opp_boost(
            base_score, opponent_board, sim,
        )
        score = self._clamp(boosted)

        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=float(viable_placements),
            detail={
                "viable_placements": viable_placements,
                "best_chain": best_chain,
                "total_trials": total_trials,
                "remaining_empty": False,
                "opp_chain_count": opp_chain,
                "opp_threat_boost_applied": opp_chain
                >= SECOND_CHAIN_OPP_THREAT_CHAIN_THRESHOLD,
            },
        )

    @staticmethod
    def _apply_opp_boost(
        base_score: float,
        opponent_board: Board | None,
        sim: ChainSimulator,
    ) -> tuple[int, float]:
        """相手脅威に応じて base_score を BOOST 倍に強化する.

        Returns:
            (opp_chain_count, boosted_score)。opponent_board=None なら (0, base_score)。
        """
        if opponent_board is None:
            return 0, base_score
        try:
            opp_result = sim.simulate(opponent_board)
        except Exception:
            return 0, base_score
        opp_chain = opp_result.chain_count
        if opp_chain >= SECOND_CHAIN_OPP_THREAT_CHAIN_THRESHOLD:
            return opp_chain, base_score * SECOND_CHAIN_OPP_THREAT_BOOST
        return opp_chain, base_score


# ============================
# 拡張 Indicator: 形 / 接ぷよ / 発火点 / 色分散 (先行研究ベース)
# ============================


class ShapeScoreIndicator(BaseIndicator):
    """
    形評価: U字形理想高さプロファイルへの近さを 0〜1 で返す。

    mayah/puyoai の FIELD_USHAPE_LINEAR を簡易化。GTR / LLR / だぁ積みは
    どれも「中央 col2-3 が低めで 端 col0,5 が高い」U字に類似するため、
    SHAPE_IDEAL_HEIGHTS との列高さ偏差合計を 1 - dev/MAX で正規化する。
    """

    @property
    def name(self) -> str:
        return INDICATOR_SHAPE_SCORE

    def compute(self, board, chain_result=None, simulator=None):
        # 列ごとの実高さ (UNKNOWN は除外、Board.height_of がケア)
        heights = [board.height_of(c) for c in range(BOARD_COLS)]
        deviation = 0.0
        for col, ideal in enumerate(SHAPE_IDEAL_HEIGHTS):
            deviation += abs(heights[col] - ideal)
        # 偏差大ほどスコア低 (1 - dev/MAX)
        score = 1.0 - min(SCORE_MAX, deviation / SHAPE_MAX_DEVIATION)
        return IndicatorResult(
            name=self.name,
            score=self._clamp(score),
            raw_value=deviation,
            detail={
                "heights": heights,
                "ideal": list(SHAPE_IDEAL_HEIGHTS),
                "deviation": deviation,
            },
        )


class TouchingDensityIndicator(BaseIndicator):
    """
    接ぷよ密度: 同色隣接ペア数 ÷ 全通常ぷよ数。

    meatfighter の puyo link 数指標 + mayah CONNECTION_2/3 の簡易版。
    高いほど連鎖伸長余地が大きい。水平・垂直の両方をカウント、
    重複しないよう (右下方向のみ) 走査する。
    """

    @property
    def name(self) -> str:
        return INDICATOR_TOUCHING_DENSITY

    def compute(self, board, chain_result=None, simulator=None):
        ojama = _count_ojama(board)
        normal = board.count_puyos() - ojama
        if normal <= 1:
            return IndicatorResult(
                name=self.name, score=SCORE_MIN, raw_value=0.0,
                detail={"pairs": 0, "normal_puyos": normal},
            )
        pairs = _count_touching_pairs(board)
        ratio = pairs / normal
        score = self._clamp(ratio / TOUCHING_MAX_RATIO)
        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=float(pairs),
            detail={"pairs": pairs, "normal_puyos": normal,
                    "ratio": ratio},
        )


class TailHeightIndicator(BaseIndicator):
    """
    連鎖発火点低さ: 最終発火点の高さを 1 - h/13 で評価する。

    mayah IGNITION_HEIGHT を反転 (低いほど高評価)。connection が無い盤面は
    中立値 0.5 を返し、発火点 = 1連鎖目で消えたグループの最大行とする。
    発火点が低い (= 下段) ほど安定した連鎖構造であり、催促耐性も高い。
    """

    @property
    def name(self) -> str:
        return INDICATOR_TAIL_HEIGHT

    def compute(self, board, chain_result=None, simulator=None):
        chain_result, _ = self._ensure_chain(board, chain_result, simulator)
        if not chain_result.steps:
            # 連鎖が組めていない盤面は中立評価 (発火点未定義)
            return IndicatorResult(
                name=self.name, score=NEXT_ACCEPTANCE_NEUTRAL, raw_value=0.0,
                detail={"trigger_row": None, "no_chain": True},
            )
        first = chain_result.steps[0]
        # 1連鎖目で消えたグループのうち row が最大 (= 最下段) のセルを発火点扱い
        max_row = 0
        for grp in first.erased_groups:
            for (r, _c) in grp.cells:
                if r > max_row:
                    max_row = r
        # 高さ = 13 - row (row=12 が最下段)
        trigger_height = TAIL_HEIGHT_MAX_ROW - max_row
        score = 1.0 - (trigger_height / TAIL_HEIGHT_MAX_ROW)
        return IndicatorResult(
            name=self.name,
            score=self._clamp(score),
            raw_value=float(trigger_height),
            detail={"trigger_row": max_row, "trigger_height": trigger_height},
        )


class ColorVarianceIndicator(BaseIndicator):
    """
    色集中度: 各色の重心からの平均距離 → 1 - mean/MAX で評価する。

    meatfighter 7 メトリクスのうち color variance を反転 (集中度が高い=連鎖
    組みやすい)。色ごとに重心 (mean row, mean col) を計算し、各セルからの
    Manhattan 距離平均を取る。サンプル数 < COLOR_VARIANCE_MIN_COUNT の色は
    集計から除外し、全色平均を返す。
    """

    @property
    def name(self) -> str:
        return INDICATOR_COLOR_VARIANCE

    def compute(self, board, chain_result=None, simulator=None):
        per_color = _compute_per_color_dispersion(board)
        if not per_color:
            return IndicatorResult(
                name=self.name, score=NEXT_ACCEPTANCE_NEUTRAL, raw_value=0.0,
                detail={"per_color": {}, "no_colors": True},
            )
        mean_dist = sum(per_color.values()) / len(per_color)
        score = 1.0 - min(SCORE_MAX, mean_dist / COLOR_VARIANCE_MAX_DIST)
        return IndicatorResult(
            name=self.name,
            score=self._clamp(score),
            raw_value=mean_dist,
            detail={"per_color": per_color, "mean_dist": mean_dist},
        )


# ============================
# 拡張指標用ヘルパー関数
# ============================


def _count_touching_pairs(board: Board) -> int:
    """同色ペアを (右隣 + 下隣) のみで走査して数える (重複防止)。"""
    pairs = 0
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            c = board.get(row, col)
            if c == COLOR_EMPTY or c == COLOR_OJAMA:
                continue
            if c not in VALID_COLORS:
                continue
            # 右隣
            if col + 1 < BOARD_COLS and board.get(row, col + 1) == c:
                pairs += 1
            # 下隣
            if row + 1 < BOARD_ROWS and board.get(row + 1, col) == c:
                pairs += 1
    return pairs


def _try_drop_one(board: Board, col: int, color: int) -> Board | None:
    """
    指定列に 1 puyo を落とした新盤面を返す (元盤面は変更しない)。

    Args:
        board: 元盤面。
        col: 落下列。
        color: 落下ぷよの色。

    Returns:
        Board | None: 置けない場合は None、置けた場合は新しい Board。
    """
    row = _drop_row(board, col)
    if row is None:
        return None
    work = board.copy()
    work.set(row, col, color)
    return work


def _strip_main_chain_groups(board: Board, sim: ChainSimulator) -> Board:
    """
    本線候補となる「サイズ >= MIN_ERASE_COUNT」グループを取り除いた盤面を返す。

    本線として消えるであろう連結を撤去することで、副砲側が独立に発火可能か
    評価する用途に使う。撤去後は重力を再適用して密度を保つ。

    Args:
        board: 元盤面。
        sim: グループ検出用シミュレータ。

    Returns:
        Board: 大グループを除いた残骸盤面。
    """
    work = board.copy()
    groups = sim.find_groups(work)
    for grp in groups:
        if grp.size < MIN_ERASE_COUNT:
            continue
        for (r, c) in grp.cells:
            work.set(r, c, COLOR_EMPTY)
    sim.apply_gravity(work)
    return work


def _min_puyos_to_ignite(
    board: Board,
    sim: ChainSimulator,
    base_chain: int,
    trial_limit: int,
) -> int:
    """
    `あと N 個 puyo を追加すると本線連鎖数を超える発火が可能` の最小 N を探索。

    N=1 で base_chain を超える placement があれば 1 を返す。trial_limit まで
    試し見つからなければ trial_limit + 1 を返す (= 発火困難)。
    探索は色 × 列の全試行 (浅い BFS)。N=2 までは多重ループで検査する。

    Args:
        board: 評価盤面 (破壊しない)。
        sim: 連鎖シミュレータ。
        base_chain: 元盤面の連鎖数 (これを超えるかで判定)。
        trial_limit: 探索する追加 puyo 数の上限。

    Returns:
        int: 最小発火 N (発火不能なら trial_limit + 1)。
    """
    # N=1
    for color in KEY_FLEX_TRIAL_COLORS:
        for col in range(BOARD_COLS):
            res = _simulate_with_placement(sim, board, col, color)
            if res is not None and res.chain_count > base_chain:
                return 1
    # N=2 (色×列を 2 段ネスト、計算量制御で trial_limit >= 2 のときのみ)
    if trial_limit >= 2:
        for color1 in KEY_FLEX_TRIAL_COLORS:
            for col1 in range(BOARD_COLS):
                board1 = _try_drop_one(board, col1, color1)
                if board1 is None or board1.is_dead():
                    continue
                for color2 in KEY_FLEX_TRIAL_COLORS:
                    for col2 in range(BOARD_COLS):
                        res = _simulate_with_placement(sim, board1, col2, color2)
                        if res is not None and res.chain_count > base_chain:
                            return 2
    return trial_limit + 1


def _compute_per_color_dispersion(board: Board) -> dict[int, float]:
    """色ごとに (重心からの Manhattan 距離平均) を返す (少数色は除外)。"""
    cells_by_color: dict[int, list[tuple[int, int]]] = {}
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            c = board.get(row, col)
            if c == COLOR_EMPTY or c == COLOR_OJAMA:
                continue
            if c not in VALID_COLORS:
                continue
            cells_by_color.setdefault(c, []).append((row, col))
    result: dict[int, float] = {}
    for color, cells in cells_by_color.items():
        if len(cells) < COLOR_VARIANCE_MIN_COUNT:
            continue
        mr = sum(r for r, _ in cells) / len(cells)
        mc = sum(c for _, c in cells) / len(cells)
        avg = sum(abs(r - mr) + abs(c - mc) for r, c in cells) / len(cells)
        result[color] = avg
    return result


# ============================
# 高度戦略指標: KeyFlexibility / SubChainIndependence / ChainTimingPressure
# ============================


class KeyFlexibilityIndicator(BaseIndicator):
    """
    キーぷよ柔軟性: 1 puyo 追加で連鎖が伸びる placement の比率 (0〜1)。

    各列 × 各色 (5色) の単発落下を試し、置けた placement のうち
    元連鎖数を超える比率を返す。値が高いほど「あと 1 個で発火可能」な
    キーぷよ位置の選択肢が多く、戦略上有利。

    Phase H1 拡張 (2026-05-08): キーぷよ外し本線戦略の視点強化。
    1 puyo 追加 (1-step) に加え、2-3 puyo 追加 (multi-step) で発火可能な
    placement も半重みで加算する。相手連鎖時間中に追加 puyo を入れる
    余裕がある状況での連鎖伸長度を表現する。
    旧 1-step 比率は detail に保持し、後方互換性を維持する。
    """

    # 旧式相当の重み (1-step 主体評価)
    _STEP_WEIGHTS: tuple[float, ...] = (1.0, 0.5, 0.25)
    _MULTI_STEP_LIMIT: int = 3

    @property
    def name(self) -> str:
        return INDICATOR_KEY_FLEXIBILITY

    def compute(self, board, chain_result=None, simulator=None):
        chain_result, sim = self._ensure_chain(board, chain_result, simulator)
        base_chain = chain_result.chain_count

        # 1-step 評価 (従来ロジック、互換性のため raw_value/detail に保持)
        trial_count, extension_count = self._count_extensions(
            board, sim, base_chain,
        )
        ratio_1step = (
            extension_count / trial_count if trial_count > 0 else 0.0
        )

        # multi-step (2-3 puyo) 評価: 1-step で発火しなかった placement のうち、
        # 追加 1-2 puyo で発火する盤面を半重みで加算する。
        ratio_multi, ext_multi = self._compute_multistep_ratio(
            board, sim, base_chain,
        )
        # 1-step 1.0 + 2-step 0.5 + 3-step 0.25 を合成 (clip 1.0)
        combined = ratio_1step * self._STEP_WEIGHTS[0] + ratio_multi
        ratio = self._clamp(combined)

        return IndicatorResult(
            name=self.name,
            score=ratio,
            raw_value=float(extension_count),
            detail={
                "base_chain": base_chain,
                "extension_count": extension_count,
                "trial_count": trial_count,
                # H1 拡張: 互換性のため 1step 比率を別フィールドに保持
                "ratio_1step": ratio_1step,
                "ratio_multi": ratio_multi,
                "extension_count_multi": ext_multi,
            },
        )

    @staticmethod
    def _count_extensions(
        board: Board,
        sim: ChainSimulator,
        base_chain: int,
    ) -> tuple[int, int]:
        """1-step 探索: 1 puyo 追加で base_chain を超える placement 数を返す."""
        trial_count = 0
        extension_count = 0
        for col in range(BOARD_COLS):
            for color in KEY_FLEX_TRIAL_COLORS:
                new_board = _try_drop_one(board, col, color)
                if new_board is None or new_board.is_dead():
                    continue
                trial_count += 1
                result = sim.simulate(new_board)
                if result.chain_count > base_chain:
                    extension_count += 1
        return trial_count, extension_count

    def _compute_multistep_ratio(
        self,
        board: Board,
        sim: ChainSimulator,
        base_chain: int,
    ) -> tuple[float, int]:
        """2-step 発火比率を半重み、3-step 発火比率を 1/4 重みで合成する.

        相手連鎖時間中に追加 puyo を入れる戦略余裕度を表現。
        計算量制御のため、1-step で「発火直前」と判定された盤面のみを
        2-step 候補とし、さらに 3-step は最初の 2-step 候補のみ深掘りする。
        Returns: (combined_ratio, total_multi_extension_count)
        """
        # 浅い BFS: 1 puyo 追加 → 発火しないが close な盤面リストを取得
        first_step_unfired: list[Board] = []
        for col in range(BOARD_COLS):
            for color in KEY_FLEX_TRIAL_COLORS:
                new_b = _try_drop_one(board, col, color)
                if new_b is None or new_b.is_dead():
                    continue
                if sim.simulate(new_b).chain_count <= base_chain:
                    first_step_unfired.append(new_b)
        if not first_step_unfired:
            return 0.0, 0
        # 2-step
        n_total_2 = 0
        n_fire_2 = 0
        two_step_unfired: list[Board] = []
        for b1 in first_step_unfired:
            for col in range(BOARD_COLS):
                for color in KEY_FLEX_TRIAL_COLORS:
                    nb = _try_drop_one(b1, col, color)
                    if nb is None or nb.is_dead():
                        continue
                    n_total_2 += 1
                    if sim.simulate(nb).chain_count > base_chain:
                        n_fire_2 += 1
                    elif len(two_step_unfired) < self._MULTI_STEP_LIMIT * 2:
                        two_step_unfired.append(nb)
        # 3-step (計算量制限のため two_step_unfired の先頭のみ)
        n_total_3 = 0
        n_fire_3 = 0
        for b2 in two_step_unfired[: self._MULTI_STEP_LIMIT]:
            for col in range(BOARD_COLS):
                for color in KEY_FLEX_TRIAL_COLORS:
                    nb = _try_drop_one(b2, col, color)
                    if nb is None or nb.is_dead():
                        continue
                    n_total_3 += 1
                    if sim.simulate(nb).chain_count > base_chain:
                        n_fire_3 += 1
        ratio_2 = n_fire_2 / n_total_2 if n_total_2 > 0 else 0.0
        ratio_3 = n_fire_3 / n_total_3 if n_total_3 > 0 else 0.0
        combined = (
            ratio_2 * self._STEP_WEIGHTS[1] + ratio_3 * self._STEP_WEIGHTS[2]
        )
        return combined, n_fire_2 + n_fire_3


class SubChainIndependenceIndicator(BaseIndicator):
    """
    副砲独立性: 本線 (4+連結) を撤去した残骸で発火可能な副砲の質を評価。

    手順:
        1. 現盤面の 4+ 連結グループを撤去 + 重力適用 → 残骸盤面
        2. 残骸盤面で 1 puyo 追加 (色 × 列) を試行し、最大連鎖数を取得
        3. max_sub_chain / SUB_CHAIN_INDEP_MAX_EXPECTED で 0〜1 正規化
    """

    @property
    def name(self) -> str:
        return INDICATOR_SUB_CHAIN_INDEP

    def compute(self, board, chain_result=None, simulator=None):
        _, sim = self._ensure_chain(board, chain_result, simulator)

        if board.is_dead():
            return IndicatorResult(
                name=self.name, score=SCORE_MIN, raw_value=0.0,
                detail={"board_dead": True, "best_sub_chain": 0},
            )

        residual = _strip_main_chain_groups(board, sim)
        best_sub_chain = sim.simulate(residual).chain_count
        # 残骸に 1 puyo 追加して発火を試す
        for col in range(BOARD_COLS):
            for color in KEY_FLEX_TRIAL_COLORS:
                trial = _try_drop_one(residual, col, color)
                if trial is None or trial.is_dead():
                    continue
                chain = sim.simulate(trial).chain_count
                if chain > best_sub_chain:
                    best_sub_chain = chain

        ratio = best_sub_chain / SUB_CHAIN_INDEP_MAX_EXPECTED
        return IndicatorResult(
            name=self.name,
            score=self._clamp(ratio),
            raw_value=float(best_sub_chain),
            detail={
                "best_sub_chain": best_sub_chain,
                "max_expected": SUB_CHAIN_INDEP_MAX_EXPECTED,
            },
        )


class ChainTimingPressureIndicator(BaseIndicator):
    """
    連鎖タイミング圧: 自盤面の発火近接度 (あと N puyo で本線が伸びる)。

    旧式 (opponent_board=None): score = 1 - min(1, N / CHAIN_TIMING_MAX_PUYO_GAP)
    N=0 (即発火可能) で 1.0、N=MAX 以上で 0.0。
    base_chain を超える発火を 1〜2 puyo 追加で探し、見つからなければ MAX+1 扱い。

    Phase F (C-3): opponent_board が与えられた場合は相手の min_n も計算し、
    相対 pressure を返す:
        relative = clamp(0.5 + (opp_min_n - self_min_n)
            / (2 * CHAIN_TIMING_MAX_PUYO_GAP), 0, 1)
    自分が相手より早く発火可能なら 1 寄り、逆なら 0 寄り。
    opponent_board=None で従来挙動を保持する。
    """

    @property
    def name(self) -> str:
        return INDICATOR_CHAIN_TIMING

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        chain_result, sim = self._ensure_chain(board, chain_result, simulator)
        base_chain = chain_result.chain_count

        if board.is_dead():
            return IndicatorResult(
                name=self.name, score=SCORE_MIN, raw_value=0.0,
                detail={"board_dead": True, "min_n": None},
            )

        min_n = _min_puyos_to_ignite(
            board, sim, base_chain, CHAIN_TIMING_TRIAL_LIMIT,
        )

        if opponent_board is None:
            # 旧挙動: 自盤面のみで N → score
            gap = min_n / CHAIN_TIMING_MAX_PUYO_GAP
            score = self._clamp(1.0 - gap)
            return IndicatorResult(
                name=self.name,
                score=score,
                raw_value=float(min_n),
                detail={
                    "base_chain": base_chain,
                    "min_n": min_n,
                    "trial_limit": CHAIN_TIMING_TRIAL_LIMIT,
                    "opp_min_n": None,
                },
            )

        # Phase F: 相手 min_n を計算して相対 pressure を返す
        opp_min_n = self._safe_opp_min_n(opponent_board, sim)
        relative = (
            CHAIN_TIMING_RELATIVE_CENTER
            + (opp_min_n - min_n) / float(2 * CHAIN_TIMING_MAX_PUYO_GAP)
        )
        score = self._clamp(relative)
        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=float(min_n),
            detail={
                "base_chain": base_chain,
                "min_n": min_n,
                "opp_min_n": opp_min_n,
                "trial_limit": CHAIN_TIMING_TRIAL_LIMIT,
                "relative_mode": True,
            },
        )

    @staticmethod
    def _safe_opp_min_n(
        opponent_board: Board, sim: ChainSimulator,
    ) -> int:
        """相手盤面の min_n を安全に計算する (例外時は MAX+1)。"""
        try:
            opp_chain = sim.simulate(opponent_board).chain_count
        except Exception:
            return CHAIN_TIMING_TRIAL_LIMIT + 1
        if opponent_board.is_dead():
            return CHAIN_TIMING_TRIAL_LIMIT + 1
        return _min_puyos_to_ignite(
            opponent_board, sim, opp_chain, CHAIN_TIMING_TRIAL_LIMIT,
        )


# ============================
# 予告お邪魔受け圧 (画像認識の OjamaWarningDetector 結果を入力)
# ============================


class IncomingOjamaPressureIndicator(BaseIndicator):
    """予告お邪魔ぷよと自分の相殺力の比較から「不利方向の圧力」を算出する。

    画像認識で得た incoming_ojama (相手から飛んでくる予告おじゃま個数) を
    MAX_OJAMA_OFFSET (=72) で正規化し、0〜1 の圧力値を返す。
    自分側が完全に相殺できる範囲を超えると 1.0 になる。

    score = clamp(incoming_ojama / MAX_OJAMA_OFFSET, 0, 1)

    ※「不利方向の指標」のため、Scorer 側では負の重みで取り込む想定。
    """

    @property
    def name(self) -> str:
        return INDICATOR_INCOMING_OJAMA

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        incoming_ojama: int = 0,
    ) -> IndicatorResult:
        """incoming_ojama を 0〜1 に正規化したスコアを返す。"""
        ojama = max(0, int(incoming_ojama))
        ratio = ojama / float(MAX_OJAMA_OFFSET)
        score = self._clamp(ratio)
        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=float(ojama),
            detail={
                "incoming_ojama": ojama,
                "max_offset": MAX_OJAMA_OFFSET,
            },
        )


# ============================
# Phase F 新指標 (2026-05-07、B-4) 回し入れ巧拙
# ============================


class RotationSkillIndicator(BaseIndicator):
    """回し入れ (rotation maneuver) 検出スコアを 0〜1 で返す degenerate 指標。

    実体は src.rotation_tracker.RotationTracker が STABLE 連続 frame の
    cell 消失と物理推論 (連鎖シミュレーション) の整合性を比較し、
    物理的に説明できない消失を「回し入れ候補」としてカウントする。
    本指標クラスはそのスコアを引数 rotation_score として受け取り、
    そのまま IndicatorResult に詰めるだけのアダプタ役。

    パターンは IncomingOjamaPressureIndicator (incoming_ojama 引数) と同じく、
    IndicatorCalculator._compute_extra で個別 instantiate + 引数渡し。
    rotation_score=0.5 (neutral) のデフォルトで未指定時の挙動を担保する。
    """

    @property
    def name(self) -> str:
        return INDICATOR_ROTATION_SKILL

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        rotation_score: float = 0.5,
    ) -> IndicatorResult:
        """rotation_score を 0〜1 に clamp してそのまま score として返す."""
        clamped = self._clamp(float(rotation_score))
        return IndicatorResult(
            name=self.name,
            score=clamped,
            raw_value=float(rotation_score),
            detail={"rotation_score": float(rotation_score)},
        )


# ============================
# Phase J 新指標 (2026-04-27 mayah/ama 先行研究ベース)
# ============================

# 隣接列高さ差ペナルティ用パラメータ
HEIGHT_DIFF_NORM_MAX: float = 30.0  # 全列差分合計の正規化分母
# 高連結数ペナルティ用パラメータ (4 連結以上 = 即発火 = 致命的)
HIGH_CONNECTION_NORM_MAX: float = 6.0
# 必要ぷよ数の正規化上限
REQUIRED_PUYO_NORM_MAX: float = 8.0
# 相手連鎖威力の正規化上限 (60 個 = 2 段以上)
OPPONENT_THREAT_NORM_MAX: float = 60.0


def _column_heights(board: Board) -> list[int]:
    """各列の高さ (0-13) を返す。"""
    return [board.height_of(c) for c in range(BOARD_COLS)]


# 同色連結成分キャッシュ (Tier B 高速化、2026-05-06)
# StructureSolidity / HighConnection / IsolatedPuyo 等が同一盤面に対して
# それぞれ独立に呼ぶため、盤面 grid hash でメモ化することで
# 重複計算を排除する。サイズ上限を設定し OOM を防ぐ。
_CC_CACHE: dict[bytes, list[tuple[int, list[tuple[int, int]]]]] = {}
_CC_CACHE_MAX_SIZE: int = 50_000


def _connected_components(board: Board) -> list[tuple[int, list[tuple[int, int]]]]:
    """同色の連結成分を全列挙する。Returns: [(color, [(r,c),...]), ...]

    盤面 grid hash でキャッシュ化済 (重複計算回避)。
    """
    grid = board._grid  # numpy.ndarray (BOARD_ROWS, BOARD_COLS)
    key = grid.tobytes()
    cached = _CC_CACHE.get(key)
    if cached is not None:
        return cached

    visited = [[False] * BOARD_COLS for _ in range(BOARD_ROWS)]
    components: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if visited[r][c]:
                continue
            color = int(grid[r, c])
            # 空セル / おじゃま / unknown は除外
            if color == 0 or color == 9 or color == 10:
                visited[r][c] = True
                continue
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                if not (0 <= rr < BOARD_ROWS and 0 <= cc < BOARD_COLS):
                    continue
                if visited[rr][cc]:
                    continue
                if int(grid[rr, cc]) != color:
                    continue
                visited[rr][cc] = True
                comp.append((rr, cc))
                stack.extend([(rr + 1, cc), (rr - 1, cc),
                              (rr, cc + 1), (rr, cc - 1)])
            if comp:
                components.append((color, comp))
    if len(_CC_CACHE) < _CC_CACHE_MAX_SIZE:
        _CC_CACHE[key] = components
    return components


class AdjacentHeightDiffIndicator(BaseIndicator):
    """隣接列高さ差の総和で「凸凹さ」を評価する (mayah 山谷ペナルティ + ちぎり代替)。

    score = 1 - clamp(diff_sum / HEIGHT_DIFF_NORM_MAX, 0, 1)
    凸凹が少ない盤面ほど score 1.0 に近い。
    """

    @property
    def name(self) -> str:
        return INDICATOR_HEIGHT_DIFF

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
    ) -> IndicatorResult:
        heights = _column_heights(board)
        diff_sum = sum(
            abs(heights[i] - heights[i + 1]) for i in range(BOARD_COLS - 1)
        )
        score = 1.0 - min(1.0, diff_sum / HEIGHT_DIFF_NORM_MAX)
        return IndicatorResult(
            name=self.name,
            score=float(score),
            raw_value=float(diff_sum),
            detail={"heights": heights, "diff_sum": diff_sum},
        )


class HighConnectionCountIndicator(BaseIndicator):
    """3 連結以上の総数を評価する。

    3 連結 = 良い積み (連鎖の素材)。4+ 連結は即発火・致命的。
    score = clamp(n_3 / 6, 0, 1) - 0.5 * clamp(n_4plus / 4, 0, 1)
    最終的に 0-1 にクランプ。
    """

    @property
    def name(self) -> str:
        return INDICATOR_HIGH_CONNECTION

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
    ) -> IndicatorResult:
        components = _connected_components(board)
        n_3 = sum(1 for _, comp in components if len(comp) == 3)
        n_4plus = sum(1 for _, comp in components if len(comp) >= 4)
        # 3 連結はプラス、4+ はマイナス
        bonus = min(1.0, n_3 / HIGH_CONNECTION_NORM_MAX)
        penalty = 0.5 * min(1.0, n_4plus / 4.0)
        score = max(0.0, min(1.0, bonus - penalty + 0.5))
        return IndicatorResult(
            name=self.name,
            score=float(score),
            raw_value=float(n_3),
            detail={"n_3": n_3, "n_4plus": n_4plus},
        )


class RequiredPuyoToFireIndicator(BaseIndicator):
    """本線発火に必要な補完ぷよ数を評価する (少ないほど高スコア = 即発火可能)。

    既存 _min_puyos_to_ignite を活用。
    score = 1 - clamp(min_n / REQUIRED_PUYO_NORM_MAX, 0, 1)
    """

    @property
    def name(self) -> str:
        return INDICATOR_REQUIRED_FIRE

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
    ) -> IndicatorResult:
        sim = simulator or ChainSimulator()
        try:
            base_chain = chain_result.chain_count if chain_result else 0
            min_n = _min_puyos_to_ignite(
                board, sim, base_chain=base_chain,
                trial_limit=CHAIN_TIMING_TRIAL_LIMIT,
            )
        except Exception:
            min_n = int(REQUIRED_PUYO_NORM_MAX)
        score = 1.0 - min(1.0, float(min_n) / REQUIRED_PUYO_NORM_MAX)
        return IndicatorResult(
            name=self.name,
            score=float(score),
            raw_value=float(min_n),
            detail={"min_n": min_n},
        )


class OpponentChainThreatIndicator(BaseIndicator):
    """相手フィールドの最大連鎖を ChainSimulator で計算し、ojama 換算で脅威度を評価する。

    opponent_board が None なら中立値 (0.5) を返す。
    score = clamp(opp_ojama / OPPONENT_THREAT_NORM_MAX, 0, 1)

    ※「不利方向の指標」のため、Scorer 側では負の重みで取り込む想定。
    """

    @property
    def name(self) -> str:
        return INDICATOR_OPPONENT_THREAT

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        if opponent_board is None:
            return IndicatorResult(
                name=self.name,
                score=0.5,
                raw_value=0.0,
                detail={"reason": "no_opponent_board"},
            )
        sim = simulator or ChainSimulator()
        try:
            opp_result = sim.simulate(opponent_board)
        except Exception:
            return IndicatorResult(
                name=self.name, score=0.5, raw_value=0.0,
                detail={"reason": "simulate_error"},
            )
        # 簡易換算: 連鎖参加ぷよ数 × 連鎖係数 / OJAMA_RATE_STANDARD (=70)
        # mayah の精緻な式まで踏み込まず、連鎖数ベース概算
        chain_n = opp_result.chain_count
        cleared = opp_result.total_erased if hasattr(opp_result, "total_erased") else 0
        # 連鎖ボーナス簡易: 1連鎖=0、2連鎖=8、以降+32
        chain_bonus_total = sum(max(0, 8 + 32 * (i - 1)) for i in range(2, chain_n + 1))
        score_est = cleared * 10 * max(1, chain_bonus_total)
        opp_ojama = score_est // 70
        score = min(1.0, opp_ojama / OPPONENT_THREAT_NORM_MAX)
        return IndicatorResult(
            name=self.name,
            score=float(score),
            raw_value=float(opp_ojama),
            detail={
                "opp_chain_n": chain_n,
                "opp_cleared": cleared,
                "opp_score_est": score_est,
                "opp_ojama_est": opp_ojama,
            },
        )


# ============================
# Phase K 新指標 (2026-04-27 凝視深化 + リソース無駄化)
# ============================

# Phase K 正規化パラメータ
OPPONENT_OFFSET_NORM_MAX: float = 60.0  # 相手の即時相殺力上限
ISOLATED_PUYO_NORM_MAX: float = 12.0    # 孤立ぷよ数上限


class OpponentOffsetPowerIndicator(BaseIndicator):
    """相手側の即時相殺力 (相手が現在発火可能な連鎖の威力 - 自分の予告ojama)。

    自分が今 ojama 攻撃を仕掛けても、相手が即発火で相殺してくる脅威度を測る。

    score = clamp(opp_immediate_ojama / OPPONENT_OFFSET_NORM_MAX, 0, 1)
    高いほど「相手が相殺してくる」= 自分が攻撃しても無効化される = 不利
    Scorer 側で負の重みで取り込む。
    """

    @property
    def name(self) -> str:
        return INDICATOR_OPPONENT_OFFSET

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        if opponent_board is None:
            return IndicatorResult(
                name=self.name, score=0.5, raw_value=0.0,
                detail={"reason": "no_opponent_board"},
            )
        sim = simulator or ChainSimulator()
        # 相手フィールドを「即発火」した場合の威力 = ChainSimulator(opponent_board)
        try:
            opp_result = sim.simulate(opponent_board)
        except Exception:
            return IndicatorResult(
                name=self.name, score=0.5, raw_value=0.0,
                detail={"reason": "simulate_error"},
            )
        # 即発火 = chain_count が 0 でない場合のみ相殺力ありと判定
        # 0 連鎖 (連鎖発火しない) = 即時相殺力なし
        if opp_result.chain_count == 0:
            return IndicatorResult(
                name=self.name, score=0.0, raw_value=0.0,
                detail={"opp_chain_n": 0},
            )
        # 即発火の威力換算: 簡易ojama個数
        chain_n = opp_result.chain_count
        cleared = opp_result.total_erased
        chain_bonus_total = sum(max(0, 8 + 32 * (i - 1)) for i in range(2, chain_n + 1))
        score_est = cleared * 10 * max(1, chain_bonus_total)
        opp_offset = score_est // 70
        score = min(1.0, opp_offset / OPPONENT_OFFSET_NORM_MAX)
        return IndicatorResult(
            name=self.name,
            score=float(score),
            raw_value=float(opp_offset),
            detail={
                "opp_chain_n": chain_n,
                "opp_cleared": cleared,
                "opp_offset_ojama": opp_offset,
            },
        )


class PostOjamaChainHealthIndicator(BaseIndicator):
    """おじゃま 30 個 (1 ターン落下上限) を受けた後の本線生存判定。

    現状盤面に ojama 30 個を仮想的に降らせて再シミュレーション、
    元の連鎖数の何割が生存するかを測る。

    score = post_chain_n / max(1, original_chain_n)
    高いほど「ojama 受けても本線生存」= 強靭
    """

    @property
    def name(self) -> str:
        return INDICATOR_POST_OJAMA_HEALTH

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
    ) -> IndicatorResult:
        if chain_result is None:
            chain_result = (simulator or ChainSimulator()).simulate(board)
        original_chain = chain_result.chain_count
        if original_chain == 0:
            return IndicatorResult(
                name=self.name, score=0.5, raw_value=0.0,
                detail={"reason": "no_original_chain"},
            )
        # 30 個の ojama を空セルに降らせる (上から順に)
        sim = simulator or ChainSimulator()
        ojama_board = self._drop_ojama(board, 30)
        try:
            post_result = sim.simulate(ojama_board)
        except Exception:
            return IndicatorResult(
                name=self.name, score=0.0, raw_value=0.0,
                detail={"reason": "simulate_error"},
            )
        post_chain = post_result.chain_count
        score = min(1.0, post_chain / max(1, original_chain))
        return IndicatorResult(
            name=self.name,
            score=float(score),
            raw_value=float(post_chain),
            detail={
                "original_chain": original_chain,
                "post_chain": post_chain,
            },
        )

    @staticmethod
    def _drop_ojama(board: Board, count: int) -> Board:
        """空セルに上から順に ojama を落として新しい Board を返す。"""
        # board のコピーを作る
        new_grid = [
            [int(board.get(r, c)) for c in range(BOARD_COLS)]
            for r in range(BOARD_ROWS)
        ]
        dropped = 0
        # 各列に均等に分散して落とす
        col_targets = [count // BOARD_COLS] * BOARD_COLS
        for i in range(count % BOARD_COLS):
            col_targets[i] += 1
        for col in range(BOARD_COLS):
            target = col_targets[col]
            if target == 0:
                continue
            # 列の上から空セルを探して下方向に積む
            placed = 0
            for r in range(BOARD_ROWS - 1, -1, -1):
                if placed >= target:
                    break
                if new_grid[r][col] == 0:
                    new_grid[r][col] = 9  # COLOR_OJAMA
                    placed += 1
                    dropped += 1
        return Board.from_list(new_grid)


class IsolatedPuyoCountIndicator(BaseIndicator):
    """連鎖参加しない孤立ぷよ数 (リソース無駄化評価)。

    連鎖完了後の盤面に残ったぷよ数 ÷ 全ぷよ数で「無駄なリソース割合」を測る。
    実際は ChainResult の participating_cells と全ぷよ数の差を使う。

    score = 1 - clamp(isolated / ISOLATED_PUYO_NORM_MAX, 0, 1)
    孤立ぷよが少ないほど高スコア。
    """

    @property
    def name(self) -> str:
        return INDICATOR_ISOLATED_PUYO

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
    ) -> IndicatorResult:
        # 全ぷよ数
        total_puyo = sum(
            1
            for r in range(BOARD_ROWS)
            for c in range(BOARD_COLS)
            if int(board.get(r, c)) not in (0, 9, 10)
        )
        if total_puyo == 0:
            return IndicatorResult(
                name=self.name, score=1.0, raw_value=0.0,
                detail={"reason": "empty_board"},
            )
        if chain_result is None:
            chain_result = (simulator or ChainSimulator()).simulate(board)
        # 連鎖参加ぷよ数
        participating = chain_result.participating_cells
        isolated = max(0, total_puyo - participating)
        score = 1.0 - min(1.0, isolated / ISOLATED_PUYO_NORM_MAX)
        return IndicatorResult(
            name=self.name,
            score=float(score),
            raw_value=float(isolated),
            detail={
                "total_puyo": total_puyo,
                "participating": participating,
                "isolated": isolated,
            },
        )


# ============================
# Tier B 新指標 (2026-05-05): 「連鎖未確定の二相性」分離
# ============================
#
# `key_flexibility` の負係数は、雑然型 (構造未確立) と柔軟保留型 (土台堅実+
# 上層未確定) が同じ値で重なるため。以下 3 指標で 2 状態を分離する:
#   - planning_entropy: 1 ツモ追加で発火する連鎖の多様性 (両方高い)
#   - structure_solidity: 下半分の連結品質 (雑然=低、柔軟保留=高)
#   - base_flatness: 下層 3 段の高さ標準偏差 (雑然=悪、柔軟保留=良)


# 形質指標用定数
# planning_entropy: 5色×6列=30回の ChainSimulator が重く E-1 拡張時間が
# 5-8 倍に膨らむため、計算量を抑えた軽量版を採用。
# 軽量化: 盤面に既存する色のみ + 各色×3列 (列 0,2,4) = 最大 12 試行
PLANNING_ENTROPY_TRIAL_COLORS: tuple[int, ...] = (
    COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE,
)
PLANNING_ENTROPY_TRIAL_COLS: tuple[int, ...] = (0, 2, 4)
# エントロピー正規化分母 (ビン数の log2)
PLANNING_ENTROPY_MAX_BINS: int = 11  # 連鎖 0..10 連鎖

# structure_solidity: 下半分の row 範囲
STRUCTURE_BOTTOM_HALF_START_ROW: int = BOARD_ROWS // 2  # row 6 から下

# base_flatness: 下層 3 段の row 範囲 (10, 11, 12)
BASE_FLATNESS_BOTTOM_ROWS: int = 3
BASE_FLATNESS_MAX_STD: float = 4.0  # 標準偏差の上限経験値

def _column_top_row(board: Board, col: int) -> int:
    """指定列の最上段 puyo の row。EMPTY 列は BOARD_ROWS。"""
    for row in range(BOARD_ROWS):
        c = board.get(row, col)
        if c != COLOR_EMPTY and c != COLOR_UNKNOWN:
            return row
    return BOARD_ROWS


def _column_height(board: Board, col: int) -> int:
    """列の puyo 高さ (BOARD_ROWS から最上段 row を引いた値)。"""
    return BOARD_ROWS - _column_top_row(board, col)


class PlanningEntropyIndicator(BaseIndicator):
    """1 ツモ追加で発火する連鎖サイズ分布のエントロピー.

    高速版 (2026-05-06):
        - 各列に「既存色のうち最頻」1 色のみ試行 (= 最大 6 sim / call)。
        - 列毎の最頻色計算は numpy で一括 (BOARD_ROWS × BOARD_COLS の py loop 削除)。
        - ChainSimulator は同一 grid に対しキャッシュ済 (異なる trial では効かないが、
          同 frame 内の連続 indicator 呼び出しでは効く)。
    """

    @property
    def name(self) -> str:
        return INDICATOR_PLANNING_ENTROPY

    def compute(
        self, board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
    ) -> IndicatorResult:
        _, sim = self._ensure_chain(board, chain_result, simulator)
        # 各列で最頻色を numpy で一括計算
        # grid: (BOARD_ROWS, BOARD_COLS), uint8
        grid = board._grid
        col_top_color: list[int] = []
        # COLOR_RED..COLOR_PURPLE = 1..5
        color_min = min(PLANNING_ENTROPY_TRIAL_COLORS)
        color_max = max(PLANNING_ENTROPY_TRIAL_COLORS)
        for col in range(BOARD_COLS):
            column = grid[:, col]
            mask = (column >= color_min) & (column <= color_max)
            if mask.any():
                # bincount で最頻色 (色 1..5 の範囲)
                counts = np.bincount(
                    column[mask], minlength=color_max + 1
                )[color_min:color_max + 1]
                top_color = int(color_min + np.argmax(counts))
                col_top_color.append(top_color)
            else:
                col_top_color.append(COLOR_RED)
        chain_sizes: list[int] = []
        for col in range(BOARD_COLS):
            top = _column_top_row(board, col)
            if top <= HIDDEN_ROWS:
                continue
            trial = board.copy()
            trial.set(top - 1, col, col_top_color[col])
            cr = sim.simulate(trial)
            chain_sizes.append(cr.chain_count)
        if not chain_sizes:
            return IndicatorResult(
                name=self.name, score=0.0, raw_value=0.0,
                detail={"reason": "no_trial"},
            )
        # 連鎖サイズ分布 (ビン: 0..10 連鎖)
        bins: dict[int, int] = {}
        for s in chain_sizes:
            b = min(s, PLANNING_ENTROPY_MAX_BINS - 1)
            bins[b] = bins.get(b, 0) + 1
        total = sum(bins.values())
        # Shannon entropy
        h = 0.0
        for n in bins.values():
            if n == 0:
                continue
            p = n / total
            h -= p * math.log2(p)
        # 最大エントロピー = log2(11) ≈ 3.46
        max_h = math.log2(PLANNING_ENTROPY_MAX_BINS)
        norm = h / max_h if max_h > 0 else 0.0
        return IndicatorResult(
            name=self.name, score=self._clamp(norm), raw_value=h,
            detail={"n_bins": len(bins), "n_trials": total},
        )


class StructureSolidityIndicator(BaseIndicator):
    """下半分の連結 ≥3 ぷよ数 / 下半分 puyo 数.

    高速化 (2026-05-06): _connected_components はモジュールレベル
    キャッシュ済 (board hash で重複計算回避)。下半分おじゃま数集計を
    numpy ベクトル化。
    """

    @property
    def name(self) -> str:
        return INDICATOR_STRUCTURE_SOLIDITY

    def compute(
        self, board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
    ) -> IndicatorResult:
        comps = _connected_components(board)
        # 下半分にあるセルだけカウント (連結 ≥3 を solid として加算)
        bottom_total = 0
        bottom_solid = 0
        for _, cells in comps:
            n_in_bottom = sum(
                1 for r, _c in cells
                if r >= STRUCTURE_BOTTOM_HALF_START_ROW
            )
            if n_in_bottom == 0:
                continue
            bottom_total += n_in_bottom
            if len(cells) >= 3:
                bottom_solid += n_in_bottom
        # 下半分のおじゃまも総数に含める (構造未確立を反映) → numpy 集計
        bottom_grid = board._grid[STRUCTURE_BOTTOM_HALF_START_ROW:]
        bottom_total += int((bottom_grid == COLOR_OJAMA).sum())
        if bottom_total == 0:
            return IndicatorResult(
                name=self.name, score=0.0, raw_value=0.0,
                detail={"reason": "empty_bottom"},
            )
        ratio = bottom_solid / bottom_total
        return IndicatorResult(
            name=self.name, score=self._clamp(ratio),
            raw_value=float(bottom_solid),
            detail={"bottom_total": bottom_total, "solid": bottom_solid},
        )


class BaseFlatnessIndicator(BaseIndicator):
    """下層 3 段の高さ標準偏差 (低い=平ら=高評価)."""

    @property
    def name(self) -> str:
        return INDICATOR_BASE_FLATNESS

    def compute(
        self, board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
    ) -> IndicatorResult:
        # numpy ベクトル化: 下層 3 段の各列で空/unknown 以外の puyo 数を一括計算。
        bottom_start = BOARD_ROWS - BASE_FLATNESS_BOTTOM_ROWS
        bottom = board._grid[bottom_start:]  # shape: (3, BOARD_COLS)
        mask = (bottom != COLOR_EMPTY) & (bottom != COLOR_UNKNOWN)
        heights_arr = mask.sum(axis=0)  # shape: (BOARD_COLS,)
        heights = [int(h) for h in heights_arr]
        mean = float(heights_arr.mean())
        std = float(np.sqrt(((heights_arr - mean) ** 2).mean()))
        # 1 - normalized_std で「平ら=高評価」
        norm_std = min(std / BASE_FLATNESS_MAX_STD, 1.0)
        score = 1.0 - norm_std
        return IndicatorResult(
            name=self.name, score=self._clamp(score), raw_value=std,
            detail={"heights": heights, "std": std},
        )


# Tier B 新指標は EXTRA とは別タプルで管理
TIER_B_INDICATOR_NAMES: tuple[str, ...] = (
    INDICATOR_PLANNING_ENTROPY,
    INDICATOR_STRUCTURE_SOLIDITY,
    INDICATOR_BASE_FLATNESS,
)


# ============================
# I-J 形テンプレ完成度 (B-1、2026-05-06)
# ============================
#
# 既知の戦法テンプレート (GTR / LLR / 階段 / 座布団) との一致度を
# 0〜1 で評価する 4 指標。等価クラス (A/B/C/D) で色対称性を考慮し、
# 1P/2P 両側のうち最良スコアを採用 (mirror=True で水平反転して評価)。
#
# key_flexibility 二相性 (雑然 vs 柔軟保留) で、後者が定石組み中で
# あることを分離するための指標。Tier B 形質指標 (planning_entropy /
# structure_solidity / base_flatness) と組み合わせて使用する。


class _FormTemplateIndicator(BaseIndicator):
    """形テンプレ完成度の共通基底."""

    def __init__(self, template, indicator_name: str) -> None:
        self._template = template
        self._name = indicator_name

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self, board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
    ) -> IndicatorResult:
        score, mirror_used = best_template_score(board, self._template)
        return IndicatorResult(
            name=self._name, score=self._clamp(score),
            raw_value=float(score),
            detail={"mirror": mirror_used, "template": self._template.name},
        )


class GtrCompletenessIndicator(_FormTemplateIndicator):
    """GTR (じーてぃーあーる) テンプレ完成度."""

    def __init__(self) -> None:
        super().__init__(GTR_TEMPLATE, INDICATOR_FORM_GTR)


class LlrCompletenessIndicator(_FormTemplateIndicator):
    """LLR (えるえるあーる) テンプレ完成度."""

    def __init__(self) -> None:
        super().__init__(LLR_TEMPLATE, INDICATOR_FORM_LLR)


class StaircaseCompletenessIndicator(_FormTemplateIndicator):
    """階段テンプレ完成度."""

    def __init__(self) -> None:
        super().__init__(STAIRCASE_TEMPLATE, INDICATOR_FORM_STAIRCASE)


class ZabutonCompletenessIndicator(_FormTemplateIndicator):
    """座布団テンプレ完成度."""

    def __init__(self) -> None:
        super().__init__(ZABUTON_TEMPLATE, INDICATOR_FORM_ZABUTON)


class SullenGtrCompletenessIndicator(_FormTemplateIndicator):
    """Sullen GTR (フキゲン GTR) テンプレ完成度. (B-1.b、2026-05-09 追加)

    citrus610/ama (Puyo Puyo Tsu AI) 由来の派生形。GTR と異なり col 1 に
    同等価クラスが縦 2 連立つ点で「不機嫌」と呼ばれる。
    """

    def __init__(self) -> None:
        super().__init__(SULLEN_GTR_TEMPLATE, INDICATOR_FORM_SULLEN_GTR)


class FronCompletenessIndicator(_FormTemplateIndicator):
    """Fron (フロン積み) テンプレ完成度. (B-1.b、2026-05-09 追加)

    citrus610/ama 由来の LLR 派生圧縮形。col 2 中央に C 挟みの B/B 縦連が特徴。
    """

    def __init__(self) -> None:
        super().__init__(FRON_TEMPLATE, INDICATOR_FORM_FRON)


FORM_TEMPLATE_INDICATOR_NAMES: tuple[str, ...] = (
    INDICATOR_FORM_GTR,
    INDICATOR_FORM_LLR,
    INDICATOR_FORM_STAIRCASE,
    INDICATOR_FORM_ZABUTON,
    # B-1.b 追加 (2026-05-09)
    INDICATOR_FORM_SULLEN_GTR,
    INDICATOR_FORM_FRON,
)


# ============================
# Phase H1 新指標 (2026-05-08): 機能能力 + 戦況 + 形分類 = 16 個
#
# 「形は手段、機能が本質」: 火力・厚み・お邪魔体制を直接測定する 7 指標を
# 一次軸とし、戦況 8 個 + GTR 折り返し 1 個を二次軸とする。
# state-holding が必要な戦況 3 指標 (harass_count_30s / early_aggression /
# counter_ignition) は API のみ準備し中立値 0.5 を返す (phase_e_collect 統合は
# H2 タスクで実装予定)。
# ============================


class ReadyChainCountIndicator(BaseIndicator):
    """ready_chain_count: 即発火可能な連鎖数 (現連鎖数) を 0〜1 に正規化.

    score = chain_count / MAX_EXPECTED_CHAIN
    main_chain_maturity と同等値 (連鎖数ベース)。chain_power_full と
    並列保持し、本指標は「数」、main_chain は「成熟度」の意味分離を意識する。
    """

    @property
    def name(self) -> str:
        return INDICATOR_READY_CHAIN

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        chain_result, _ = self._ensure_chain(board, chain_result, simulator)
        count = chain_result.chain_count
        score = self._clamp(count / float(MAX_EXPECTED_CHAIN))
        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=float(count),
            detail={
                "chain_count": count,
                "max_expected": MAX_EXPECTED_CHAIN,
            },
        )


class IgnitionDistanceIndicator(BaseIndicator):
    """ignition_distance: 発火寸前度合 (1-step / 2-step 探索).

    既存 _min_puyos_to_ignite を活用し、最小 N が小さいほど高スコア。
    score = 1 - clamp(N / IGNITION_DISTANCE_MAX_PUYOS, 0, 1)
    N=0 (即発火) で 1.0、N=MAX 以上で 0.0。RequiredPuyoToFireIndicator と
    意味は近いが、本指標は MAX をより小さく取り (6 vs 8) 鋭敏に反応する。
    """

    @property
    def name(self) -> str:
        return INDICATOR_IGNITION_DISTANCE

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        chain_result, sim = self._ensure_chain(board, chain_result, simulator)
        if board.is_dead():
            return IndicatorResult(
                name=self.name, score=SCORE_MIN, raw_value=0.0,
                detail={"board_dead": True},
            )
        base_chain = chain_result.chain_count
        try:
            min_n = _min_puyos_to_ignite(
                board, sim, base_chain=base_chain,
                trial_limit=IGNITION_DISTANCE_MAX_PUYOS,
            )
        except Exception:
            min_n = IGNITION_DISTANCE_UNREACHABLE
        score = 1.0 - min(1.0, float(min_n) / IGNITION_DISTANCE_MAX_PUYOS)
        return IndicatorResult(
            name=self.name,
            score=self._clamp(score),
            raw_value=float(min_n),
            detail={
                "min_n": min_n,
                "base_chain": base_chain,
                "max_puyos": IGNITION_DISTANCE_MAX_PUYOS,
            },
        )


class CurrentFirePowerIndicator(BaseIndicator):
    """current_fire_power: 即発火した場合の威力 (ojama 換算).

    chain_result.score (= 公式得点) を OJAMA_DIVISOR で割って ojama 数に換算し、
    MAX_OJAMA_OFFSET で正規化。OffsetPower と意味は近いが、本指標は
    対数変換を行わず線形ratio で「現威力の絶対量」を表現する。
    """

    @property
    def name(self) -> str:
        return INDICATOR_CURRENT_FIRE_POWER

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        chain_result, _ = self._ensure_chain(board, chain_result, simulator)
        # OffsetPower と同じ得点計算ロジックを再利用
        total_score = OffsetPowerIndicator._total_chain_score(chain_result)
        ojama_equiv = (
            total_score / OJAMA_DIVISOR if total_score > 0 else 0.0
        )
        ratio = ojama_equiv / float(MAX_OJAMA_OFFSET)
        return IndicatorResult(
            name=self.name,
            score=self._clamp(ratio),
            raw_value=float(ojama_equiv),
            detail={
                "chain_count": chain_result.chain_count,
                "estimated_ojama": int(ojama_equiv),
                "total_score": int(total_score),
            },
        )


class MaximumFirePowerIndicator(BaseIndicator):
    """maximum_fire_power: 完成後の最大威力 (理想配置の飽和連鎖量).

    全 puyo を理想的に配置した場合の最大連鎖威力を heuristic 推定:
        chain_estimate = (puyo_count + color_count * BONUS) / PUYO_PER_CHAIN
        capped at MAX_FIRE_CHAIN_CEIL
        ojama = chain_power_table[chain] × erased_proxy / OJAMA_DIVISOR
    色の多様性が高いほど色数ボーナスで威力が増す (色数 4 で +12 ボーナス)。
    """

    @property
    def name(self) -> str:
        return INDICATOR_MAXIMUM_FIRE_POWER

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        # 色別 puyo 数 (ojama / empty 除く)
        color_counts = _color_counts(board)
        total_puyo = sum(color_counts.values())
        n_colors = len(color_counts)

        # 連鎖数 heuristic: puyo 数 / 1連鎖必要数 + 色多様性補正
        chain_est_raw = (
            total_puyo / MAX_FIRE_PUYO_PER_CHAIN
            + n_colors * MAX_FIRE_COLOR_DIVERSITY_BONUS
        )
        chain_est = int(min(chain_est_raw, float(MAX_FIRE_CHAIN_CEIL)))

        # CHAIN_POWER_TABLE で連鎖パワー、erased_proxy = total_puyo で得点換算
        if chain_est <= 0:
            score_est = 0.0
        else:
            chain_power_total = sum(
                CHAIN_POWER_TABLE[i]
                for i in range(1, chain_est + 1)
                if i < len(CHAIN_POWER_TABLE)
            )
            color_bonus = COLOR_BONUS_TABLE.get(
                min(n_colors, max(COLOR_BONUS_TABLE)),
                COLOR_BONUS_TABLE[max(COLOR_BONUS_TABLE)],
            )
            bonus_total = max(
                chain_power_total + color_bonus, MIN_CHAIN_BONUS,
            )
            # 連鎖参加 puyo 想定: 全 puyo の 80%
            erased_proxy = total_puyo * 0.8
            score_est = BASE_SCORE_PER_PUYO * erased_proxy * bonus_total
        ojama_equiv = score_est / OJAMA_DIVISOR
        ratio = ojama_equiv / float(MAX_OJAMA_OFFSET)
        return IndicatorResult(
            name=self.name,
            score=self._clamp(ratio),
            raw_value=float(ojama_equiv),
            detail={
                "puyo_count": total_puyo,
                "color_count": n_colors,
                "chain_estimate": chain_est,
                "score_est": int(score_est),
                "ojama_est": int(ojama_equiv),
            },
        )


class MidGameResponseCapacityIndicator(BaseIndicator):
    """mid_game_response_capacity: 中盤応答能力 (催促打ち合いの余裕).

    sub_chain_quality (副砲威力) + maximum_fire_power 残量を
    1 催促消費 puyo 数 (経験値 6) で正規化する。
    値が高いほど「催促打ち合いに何度耐えられるか」を表現。
    """

    @property
    def name(self) -> str:
        return INDICATOR_MID_GAME_RESPONSE

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        chain_result, sim = self._ensure_chain(board, chain_result, simulator)
        # sub_chain 評価
        sub_ind = SubChainQualityIndicator()
        sub_res = sub_ind.compute(
            board, chain_result=chain_result, simulator=sim,
        )
        sub_score = sub_res.score
        # max_fire_power 評価
        max_ind = MaximumFirePowerIndicator()
        max_res = max_ind.compute(
            board, chain_result=chain_result, simulator=sim,
        )
        max_score = max_res.score
        # 残量 = (current の補集合): max - current の正の差で「未使用余力」
        cur_ind = CurrentFirePowerIndicator()
        cur_res = cur_ind.compute(
            board, chain_result=chain_result, simulator=sim,
        )
        cur_score = cur_res.score
        residual = max(0.0, max_score - cur_score)

        # 1 催促消費 puyo 数で正規化 (低 cost で同等 sub_score なら高評価)
        # heuristic: combined = (sub_score + residual) を MID_GAME 係数で線形圧縮
        combined = (sub_score + residual) / 2.0
        # 1 催促消費を 6 puyo として、MAX_OJAMA_OFFSET / 6 ≒ 12 回の応答可能性
        # 直接 0..1 に圧縮済 (combined 自体が 0..1)
        return IndicatorResult(
            name=self.name,
            score=self._clamp(combined),
            raw_value=float(combined),
            detail={
                "sub_chain_score": sub_score,
                "max_fire_power": max_score,
                "current_fire_power": cur_score,
                "residual": residual,
                "harass_puyo_cost": MID_GAME_HARASS_PUYO_COST,
            },
        )


class HarassmentReadinessIndicator(BaseIndicator):
    """harassment_readiness: 即催促打てる小連鎖 (2-4 連鎖) の存在度.

    sub_chain_quality 拡張版。SubChainQuality は viable count を測るが、
    本指標は「即催促可能な小連鎖の威力 (ojama 換算)」を直接評価する。
    候補グループから 1 puyo 追加発火を試行し、2-4 連鎖の威力合計 (ojama)
    を HARASS_READINESS_NORM_OJAMA で正規化。
    """

    @property
    def name(self) -> str:
        return INDICATOR_HARASS_READINESS

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        _, sim = self._ensure_chain(board, chain_result, simulator)
        if board.is_dead():
            return IndicatorResult(
                name=self.name, score=SCORE_MIN, raw_value=0.0,
                detail={"board_dead": True, "viable_count": 0},
            )
        candidates = SubChainQualityIndicator._collect_candidates(board, sim)
        viable_ojama_total = 0.0
        viable_count = 0
        best_chain = 0
        for color, col in candidates:
            res = _simulate_with_placement(sim, board, col, color)
            if res is None:
                continue
            n = res.chain_count
            if HARASS_READINESS_MIN_CHAIN <= n <= HARASS_READINESS_MAX_CHAIN:
                # 公式得点 → ojama 換算
                total = OffsetPowerIndicator._total_chain_score(res)
                ojama = total / OJAMA_DIVISOR
                viable_ojama_total += ojama
                viable_count += 1
                best_chain = max(best_chain, n)
        ratio = viable_ojama_total / HARASS_READINESS_NORM_OJAMA
        return IndicatorResult(
            name=self.name,
            score=self._clamp(ratio),
            raw_value=float(viable_ojama_total),
            detail={
                "viable_count": viable_count,
                "best_chain": best_chain,
                "candidate_count": len(candidates),
                "ojama_total": viable_ojama_total,
                "min_chain": HARASS_READINESS_MIN_CHAIN,
                "max_chain": HARASS_READINESS_MAX_CHAIN,
            },
        )


class OjamaDefenseCapacityIndicator(BaseIndicator):
    """ojama_defense_capacity: お邪魔体制 (受けて掘れる量).

    仮想 ojama 10/20/30 個を降らせた後、本線生存率と掘削可能性を評価。
    HarassmentResistance は連鎖数残存率のみだが、本指標は
    「掘削 (= 1 連鎖以上発火可能か)」も加味する。
    score = 平均 (生存率 + 掘削可能性 0.5 重み) を 0..1 に正規化。
    """

    @property
    def name(self) -> str:
        return INDICATOR_OJAMA_DEFENSE

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        chain_result, sim = self._ensure_chain(board, chain_result, simulator)
        base_chain = max(1, chain_result.chain_count)
        scores: list[float] = []
        details: dict[int, dict[str, Any]] = {}
        for n_ojama in OJAMA_DEFENSE_TEST_COUNTS:
            try:
                ojama_board = sim.drop_ojama(board, n_ojama)
            except Exception:
                scores.append(0.0)
                details[n_ojama] = {"error": "drop_failed"}
                continue
            if ojama_board.is_dead():
                scores.append(0.0)
                details[n_ojama] = {"dead": True}
                continue
            try:
                post = sim.simulate(ojama_board)
            except Exception:
                scores.append(0.0)
                details[n_ojama] = {"error": "simulate_failed"}
                continue
            survival = min(1.0, post.chain_count / float(base_chain))
            # 掘削可能性 = 1 連鎖以上残せるか
            dig = 1.0 if post.chain_count >= OJAMA_DEFENSE_DIG_MIN_CHAIN else 0.0
            combined = 0.7 * survival + 0.3 * dig
            scores.append(combined)
            details[n_ojama] = {
                "post_chain": post.chain_count,
                "survival": survival,
                "dig": dig,
            }
        avg = sum(scores) / len(scores) if scores else 0.0
        return IndicatorResult(
            name=self.name,
            score=self._clamp(avg),
            raw_value=float(avg),
            detail={
                "base_chain": chain_result.chain_count,
                "by_ojama_count": details,
                "test_counts": list(OJAMA_DEFENSE_TEST_COUNTS),
            },
        )


# ============================
# Phase H1: 戦況・タイミング指標 8 個
# ============================


def _estimate_chain_duration_frames(
    chain_count: int,
) -> float:
    """連鎖数から frame 数を概算する.

    1 連鎖 ≈ 84 frame (公式アニメーション基準) として線形換算。
    chain_count=0 で 0 frame, chain_count=10 で 840 frame ≈ 14 秒。
    """
    return float(chain_count) * CHAIN_DURATION_FRAMES_PER_CHAIN


class SelfChainDurationIndicator(BaseIndicator):
    """self_chain_duration_frames: 自連鎖 frame 数を 0..1 に正規化.

    score = clamp(frames / CHAIN_DURATION_NORM_MAX_FRAMES, 0, 1)
    値が高いほど「長い連鎖時間 = 相手にとって応答時間が長く与えられる = 不利」
    という意味なので、Scorer 側では負の重みで取り込む想定。
    ただし「長く時間を稼げる」とも解釈でき、学習で重要度を判定する。
    """

    @property
    def name(self) -> str:
        return INDICATOR_SELF_CHAIN_DURATION

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        chain_result, _ = self._ensure_chain(board, chain_result, simulator)
        frames = _estimate_chain_duration_frames(chain_result.chain_count)
        score = min(1.0, frames / CHAIN_DURATION_NORM_MAX_FRAMES)
        return IndicatorResult(
            name=self.name,
            score=self._clamp(score),
            raw_value=float(frames),
            detail={
                "chain_count": chain_result.chain_count,
                "frames_per_chain": CHAIN_DURATION_FRAMES_PER_CHAIN,
            },
        )


class OppChainDurationIndicator(BaseIndicator):
    """opp_chain_duration_frames: 相手連鎖 frame 数 (opponent_board 必須).

    opponent_board=None なら中立値 0.0 を返す。
    """

    @property
    def name(self) -> str:
        return INDICATOR_OPP_CHAIN_DURATION

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        if opponent_board is None:
            return IndicatorResult(
                name=self.name, score=0.0, raw_value=0.0,
                detail={"reason": "no_opponent_board"},
            )
        sim = simulator or ChainSimulator()
        try:
            opp_result = sim.simulate(opponent_board)
        except Exception:
            return IndicatorResult(
                name=self.name, score=0.0, raw_value=0.0,
                detail={"reason": "simulate_error"},
            )
        frames = _estimate_chain_duration_frames(opp_result.chain_count)
        score = min(1.0, frames / CHAIN_DURATION_NORM_MAX_FRAMES)
        return IndicatorResult(
            name=self.name,
            score=self._clamp(score),
            raw_value=float(frames),
            detail={
                "opp_chain_count": opp_result.chain_count,
                "frames_per_chain": CHAIN_DURATION_FRAMES_PER_CHAIN,
            },
        )


class ChainDurationAdvantageIndicator(BaseIndicator):
    """chain_duration_advantage: 応答可能 puyo 数差 (相手連鎖時間 - 自連鎖時間).

    (opp_frames - self_frames) / TUMO_INTERVAL で「相手連鎖中に自分が
    余分に置けるツモ数」を概算し、CHAIN_DURATION_ADV_NORM_PUYOS で正規化。
    score = clamp(0.5 + adv / (2 * NORM), 0, 1)。0.5 中央 (互角)。
    opponent_board=None なら 0.5 (中立)。
    """

    @property
    def name(self) -> str:
        return INDICATOR_CHAIN_DURATION_ADV

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        if opponent_board is None:
            return IndicatorResult(
                name=self.name, score=SITUATIONAL_NEUTRAL_SCORE,
                raw_value=0.0,
                detail={"reason": "no_opponent_board"},
            )
        chain_result, sim = self._ensure_chain(board, chain_result, simulator)
        self_frames = _estimate_chain_duration_frames(chain_result.chain_count)
        try:
            opp_chain = sim.simulate(opponent_board).chain_count
        except Exception:
            opp_chain = 0
        opp_frames = _estimate_chain_duration_frames(opp_chain)
        # 応答可能 puyo 差: 相手連鎖中に自分が置けるツモ数
        delta_frames = opp_frames - self_frames
        delta_puyos = delta_frames / CHAIN_DURATION_TUMO_INTERVAL_FRAMES
        # 正規化: 0..1 に圧縮、center=0.5
        ratio = (
            SITUATIONAL_NEUTRAL_SCORE
            + delta_puyos / (2.0 * CHAIN_DURATION_ADV_NORM_PUYOS)
        )
        return IndicatorResult(
            name=self.name,
            score=self._clamp(ratio),
            raw_value=float(delta_puyos),
            detail={
                "self_frames": self_frames,
                "opp_frames": opp_frames,
                "delta_puyos": delta_puyos,
                "tumo_interval": CHAIN_DURATION_TUMO_INTERVAL_FRAMES,
            },
        )


class HarassEventCount30sIndicator(BaseIndicator):
    """harass_event_count_30s: 直近 30s の催促回数 (state-holding stub).

    現状 stateless 実装のため中立値 0.5 を返す。
    H2 タスクで phase_e_collect 統合後、外部 wrapper から値を注入可能。
    API のみ準備し、harass_count パラメータ受け取りに対応。
    """

    @property
    def name(self) -> str:
        return INDICATOR_HARASS_COUNT_30S

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
        harass_count: int | None = None,
    ) -> IndicatorResult:
        # H2 統合まで中立値 (パラメータが None でない場合だけ正規化値を返す)
        if harass_count is None:
            score = SITUATIONAL_NEUTRAL_SCORE
            raw = 0.0
        else:
            # 経験値: 30s 内に 5 催促で max 想定
            raw = float(max(0, harass_count))
            score = min(1.0, raw / 5.0)
        return IndicatorResult(
            name=self.name,
            score=self._clamp(score),
            raw_value=raw,
            detail={
                "stateful": harass_count is not None,
                "harass_count": harass_count,
            },
        )


class EarlyAggressionScoreIndicator(BaseIndicator):
    """early_aggression_score: 試合開始 +30s 内の発火 + ojama 送信 (state-holding).

    現状 stateless で中立値 0.5。H2 で外部 wrapper から早期攻撃量を注入。
    """

    @property
    def name(self) -> str:
        return INDICATOR_EARLY_AGGRESSION

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
        early_aggression: float | None = None,
    ) -> IndicatorResult:
        if early_aggression is None:
            score = SITUATIONAL_NEUTRAL_SCORE
            raw = 0.0
        else:
            raw = float(early_aggression)
            score = self._clamp(raw)
        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=raw,
            detail={
                "stateful": early_aggression is not None,
                "early_aggression": early_aggression,
            },
        )


class CounterIgnitionSignalIndicator(BaseIndicator):
    """counter_ignition_signal: 直前 ojama 受信 + 直後本線発火パターン (state-holding).

    現状 stateless で中立値 0.5。H2 で時系列パターン検出と統合。
    """

    @property
    def name(self) -> str:
        return INDICATOR_COUNTER_IGNITION

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
        counter_signal: float | None = None,
    ) -> IndicatorResult:
        if counter_signal is None:
            score = SITUATIONAL_NEUTRAL_SCORE
            raw = 0.0
        else:
            raw = float(counter_signal)
            score = self._clamp(raw)
        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=raw,
            detail={
                "stateful": counter_signal is not None,
                "counter_signal": counter_signal,
            },
        )


class PostAllClearStateIndicator(BaseIndicator):
    """post_all_clear_state: 序盤全消し検出 + 経過秒で判定.

    現盤面のぷよが全消し可能 (全 puyo 連鎖参加 + 残ぷよゼロ) かを検出。
    elapsed_sec が None / 序盤閾値以下なら「全消しボーナス使用可能」として
    score を 1.0 (検出) or 0.0 (未検出) で返す。
    serag (シリアル) state-holding が必要だが、stateless でも盤面検出のみ可。
    """

    @property
    def name(self) -> str:
        return INDICATOR_POST_ALL_CLEAR

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
        elapsed_sec: float | None = None,
    ) -> IndicatorResult:
        chain_result, _ = self._ensure_chain(board, chain_result, simulator)
        # 連鎖後盤面に通常 puyo がゼロかつ chain_count >= 1 で全消し
        final_board = chain_result.final_board
        normal_puyo_count = _count_cells(
            final_board,
            lambda c: c != COLOR_EMPTY and c != COLOR_OJAMA
            and c != COLOR_UNKNOWN,
        )
        is_all_clear = (
            chain_result.chain_count >= 1 and normal_puyo_count == 0
        )
        # elapsed_sec で序盤判定
        if elapsed_sec is None:
            in_early_game = True  # 経過秒不明 → 中立的に序盤扱い
        else:
            in_early_game = elapsed_sec <= POST_ALL_CLEAR_EARLY_GAME_MAX_SEC
        if is_all_clear and in_early_game:
            score = 1.0
        elif is_all_clear:
            score = 0.5  # 序盤外の全消し (ボーナス効果は薄い)
        else:
            score = 0.0
        return IndicatorResult(
            name=self.name,
            score=self._clamp(score),
            raw_value=float(is_all_clear),
            detail={
                "is_all_clear": is_all_clear,
                "in_early_game": in_early_game,
                "elapsed_sec": elapsed_sec,
                "normal_puyo_count": normal_puyo_count,
                "chain_count": chain_result.chain_count,
            },
        )


class UpperBoardDensityIndicator(BaseIndicator):
    """upper_board_density: 上部 (10+ 段) の puyo 密度.

    上 4 段 (row 0..3) の puyo 数 / (4 行 × 6 列) で 0..1 正規化。
    上級者は配置リズムが速く高密度でも回せる (advantage)。
    中級者は窒息リスク大 (penalty)。学習で重要度を判定する。
    死亡判定 (DeathRiskIndicator) とは独立して、純粋な「上部の積み上げ
    速度」を測る。
    """

    @property
    def name(self) -> str:
        return INDICATOR_UPPER_DENSITY

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        # 上 UPPER_DENSITY_TOP_ROWS 行を集計 (= row 0..UPPER_DENSITY_TOP_ROWS-1)
        upper_grid = board._grid[:UPPER_DENSITY_TOP_ROWS]
        # 通常ぷよ + おじゃまをカウント (空・unknown 除く)
        mask = (upper_grid != COLOR_EMPTY) & (upper_grid != COLOR_UNKNOWN)
        n_puyo = int(mask.sum())
        n_cells = UPPER_DENSITY_TOP_ROWS * BOARD_COLS
        density = n_puyo / float(n_cells) if n_cells > 0 else 0.0
        return IndicatorResult(
            name=self.name,
            score=self._clamp(density),
            raw_value=float(n_puyo),
            detail={
                "upper_puyo_count": n_puyo,
                "upper_cells_total": n_cells,
                "top_rows": UPPER_DENSITY_TOP_ROWS,
            },
        )


# ============================
# Phase H1: 形分類指標 1 個 (GTR 折り返し位置)
# ============================


class GtrOrientationIndicator(BaseIndicator):
    """gtr_orientation: GTR 折り返し位置の検出 (先折り / 後折り / 自由形).

    既存 form_gtr テンプレ評価結果から、1P / 2P (mirror) どちらに合致したかで
    先折り / 後折りを判定する。形が GTR と十分一致しない場合は「自由形/不明」。
    score: 先折り=1.0, 後折り=0.5, 自由形=0.0 (戦略確立度の表現)。
    detail.orientation_code: 0=先折り, 1=後折り, 2=自由形/不明
    """

    @property
    def name(self) -> str:
        return INDICATOR_GTR_ORIENTATION

    def compute(
        self,
        board: Board,
        chain_result: ChainResult | None = None,
        simulator: ChainSimulator | None = None,
        opponent_board: Board | None = None,
    ) -> IndicatorResult:
        gtr_score, mirror_used = best_template_score(board, GTR_TEMPLATE)
        if gtr_score < GTR_ORIENTATION_FORM_THRESHOLD:
            code = GTR_ORIENTATION_FREE
            score = GTR_ORIENTATION_SCORE_FREE
            label = "free"
        elif mirror_used:
            # mirror = 右側 (col 4-5) で折り返し → 後折り
            code = GTR_ORIENTATION_BACK
            score = GTR_ORIENTATION_SCORE_BACK
            label = "back"
        else:
            # 通常 (col 0-1) で折り返し → 先折り
            code = GTR_ORIENTATION_FRONT
            score = GTR_ORIENTATION_SCORE_FRONT
            label = "front"
        return IndicatorResult(
            name=self.name,
            score=self._clamp(score),
            raw_value=float(code),
            detail={
                "orientation_code": code,
                "orientation_label": label,
                "gtr_score": gtr_score,
                "mirror_used": mirror_used,
                "form_threshold": GTR_ORIENTATION_FORM_THRESHOLD,
            },
        )


# Phase H1 指標タプル (機能 + 戦況 + 形分類)
PHASE_H1_CAPABILITY_INDICATOR_NAMES: tuple[str, ...] = (
    INDICATOR_READY_CHAIN,
    INDICATOR_IGNITION_DISTANCE,
    INDICATOR_CURRENT_FIRE_POWER,
    INDICATOR_MAXIMUM_FIRE_POWER,
    INDICATOR_MID_GAME_RESPONSE,
    INDICATOR_HARASS_READINESS,
    INDICATOR_OJAMA_DEFENSE,
)
PHASE_H1_SITUATIONAL_INDICATOR_NAMES: tuple[str, ...] = (
    INDICATOR_SELF_CHAIN_DURATION,
    INDICATOR_OPP_CHAIN_DURATION,
    INDICATOR_CHAIN_DURATION_ADV,
    INDICATOR_HARASS_COUNT_30S,
    INDICATOR_EARLY_AGGRESSION,
    INDICATOR_COUNTER_IGNITION,
    INDICATOR_POST_ALL_CLEAR,
    INDICATOR_UPPER_DENSITY,
)
PHASE_H1_FORM_ORIENTATION_INDICATOR_NAMES: tuple[str, ...] = (
    INDICATOR_GTR_ORIENTATION,
)
PHASE_H1_INDICATOR_NAMES: tuple[str, ...] = (
    PHASE_H1_CAPABILITY_INDICATOR_NAMES
    + PHASE_H1_SITUATIONAL_INDICATOR_NAMES
    + PHASE_H1_FORM_ORIENTATION_INDICATOR_NAMES
)


# ============================
# IndicatorCalculator (ファサード)
# ============================


class IndicatorCalculator:
    """
    8指標を一括計算するファサード。

    差し替え可能な指標リストを受け取り、compute_all() で全指標を計算する。

    Usage:
        calc = IndicatorCalculator()
        indicator_set = calc.compute_all(board)
        print(indicator_set.score_of(INDICATOR_MAIN_CHAIN))
    """

    def __init__(
        self,
        indicators: list[BaseIndicator] | None = None,
        simulator: ChainSimulator | None = None,
    ) -> None:
        """
        Args:
            indicators: 使用する指標のリスト (None ならデフォルト8指標)。
            simulator: 共有シミュレータ (None なら内部生成)。
        """
        self._simulator = simulator or ChainSimulator()
        inds = indicators if indicators is not None else self._default_indicators()
        self._indicators: dict[str, BaseIndicator] = {i.name: i for i in inds}

    def compute_all(
        self,
        board: Board,
        next_pair: tuple[int, int] | None = None,
        dnext_pair: tuple[int, int] | None = None,
        incoming_ojama: int = 0,
        opponent_board: Board | None = None,
        rotation_score: float = 0.5,
        elapsed_sec: float | None = None,
        harass_count: int | None = None,
        early_aggression: float | None = None,
        counter_signal: float | None = None,
    ) -> IndicatorSet:
        """
        全指標を計算して IndicatorSet を返す。

        Args:
            board: 評価対象の盤面。
            next_pair: 次のツモ (TOP, BOT 色)。None なら受け入れ余地は中立値。
            dnext_pair: ダブルネクストのツモ。None なら中立値。
            incoming_ojama: 相手から飛来する予告お邪魔個数 (0 なら無し)。
            opponent_board: 相手フィールド (Phase J: 凝視 opponent_chain_threat
                + Phase F (C-3): ExtensionPotential / SecondChainPotential /
                ChainTimingPressure に opponent context を付与)。
            rotation_score: Phase F (B-4) RotationTracker.score を直接受け取る。
                0.5 (neutral) がデフォルト、 IndicatorSet.rotation_skill にも反映。
            elapsed_sec: Phase H1 (2026-05-08) post_all_clear_state 用、
                試合開始からの経過秒。None なら序盤扱い。
            harass_count: Phase H1 state-holding 系の注入用。None で中立値 0.5。
            early_aggression: 同上、序盤攻撃量 (0..1)。None で中立値 0.5。
            counter_signal: 同上、カウンター発火信号 (0..1)。None で中立値 0.5。

        Returns:
            IndicatorSet: 全指標 (8 主 + 拡張) を含む結果セット。
        """
        chain_result = self._simulator.simulate(board)
        results: dict[str, IndicatorResult] = {}
        for name, ind in self._indicators.items():
            # C-3 (I-H): HarassmentResistance に incoming_ojama を渡す
            if name == INDICATOR_HARASSMENT and incoming_ojama > 0:
                results[name] = ind.compute(
                    board,
                    chain_result=chain_result,
                    simulator=self._simulator,
                    incoming_ojama=incoming_ojama,
                )
            elif name in (INDICATOR_EXTENSION, INDICATOR_SECOND):
                # Phase F (C-3): opponent_board を opt-in で渡す
                # (None 時は backwards compat 維持)
                results[name] = ind.compute(
                    board,
                    chain_result=chain_result,
                    simulator=self._simulator,
                    opponent_board=opponent_board,
                )
            else:
                results[name] = ind.compute(
                    board,
                    chain_result=chain_result,
                    simulator=self._simulator,
                )
        next_score = self._calc_next_acceptance(board, next_pair, dnext_pair)
        ext = self._compute_extra(
            board, chain_result, incoming_ojama, opponent_board,
            rotation_score=rotation_score,
            elapsed_sec=elapsed_sec,
            harass_count=harass_count,
            early_aggression=early_aggression,
            counter_signal=counter_signal,
        )
        results.update(ext)
        return self._build_indicator_set(results, ext, next_score)

    def _build_indicator_set(
        self,
        results: dict[str, IndicatorResult],
        ext: dict[str, IndicatorResult],
        next_score: float,
    ) -> IndicatorSet:
        """results / ext から IndicatorSet を組み立てる共通ヘルパ.

        compute_all と compute_all_probabilistic で共有することで
        Phase H1 16 指標の field 反映を一箇所で管理する。
        """
        return IndicatorSet(
            results=results,
            next_acceptance=next_score,
            shape_score=ext[INDICATOR_SHAPE_SCORE].score,
            touching_density=ext[INDICATOR_TOUCHING_DENSITY].score,
            tail_height_score=ext[INDICATOR_TAIL_HEIGHT].score,
            color_variance_score=ext[INDICATOR_COLOR_VARIANCE].score,
            key_flexibility=ext[INDICATOR_KEY_FLEXIBILITY].score,
            sub_chain_independence=ext[INDICATOR_SUB_CHAIN_INDEP].score,
            chain_timing_pressure=ext[INDICATOR_CHAIN_TIMING].score,
            incoming_ojama_pressure=ext[INDICATOR_INCOMING_OJAMA].score,
            opponent_chain_threat=ext[INDICATOR_OPPONENT_THREAT].score,
            adjacent_height_diff=ext[INDICATOR_HEIGHT_DIFF].score,
            high_connection_count=ext[INDICATOR_HIGH_CONNECTION].score,
            required_puyo_to_fire=ext[INDICATOR_REQUIRED_FIRE].score,
            opponent_offset_power=ext[INDICATOR_OPPONENT_OFFSET].score,
            post_ojama_chain_health=ext[INDICATOR_POST_OJAMA_HEALTH].score,
            isolated_puyo_count=ext[INDICATOR_ISOLATED_PUYO].score,
            planning_entropy=ext[INDICATOR_PLANNING_ENTROPY].score,
            structure_solidity=ext[INDICATOR_STRUCTURE_SOLIDITY].score,
            base_flatness=ext[INDICATOR_BASE_FLATNESS].score,
            form_gtr=ext[INDICATOR_FORM_GTR].score,
            form_llr=ext[INDICATOR_FORM_LLR].score,
            form_staircase=ext[INDICATOR_FORM_STAIRCASE].score,
            form_zabuton=ext[INDICATOR_FORM_ZABUTON].score,
            # B-1.b 追加 (2026-05-09)
            form_sullen_gtr=ext[INDICATOR_FORM_SULLEN_GTR].score,
            form_fron=ext[INDICATOR_FORM_FRON].score,
            rotation_skill=ext[INDICATOR_ROTATION_SKILL].score,
            # Phase H1 (2026-05-08) capability 7
            ready_chain_count=ext[INDICATOR_READY_CHAIN].score,
            ignition_distance=ext[INDICATOR_IGNITION_DISTANCE].score,
            current_fire_power=ext[INDICATOR_CURRENT_FIRE_POWER].score,
            maximum_fire_power=ext[INDICATOR_MAXIMUM_FIRE_POWER].score,
            mid_game_response_capacity=ext[INDICATOR_MID_GAME_RESPONSE].score,
            harassment_readiness=ext[INDICATOR_HARASS_READINESS].score,
            ojama_defense_capacity=ext[INDICATOR_OJAMA_DEFENSE].score,
            # Phase H1 situational 8
            self_chain_duration_frames=ext[INDICATOR_SELF_CHAIN_DURATION].score,
            opp_chain_duration_frames=ext[INDICATOR_OPP_CHAIN_DURATION].score,
            chain_duration_advantage=ext[INDICATOR_CHAIN_DURATION_ADV].score,
            harass_event_count_30s=ext[INDICATOR_HARASS_COUNT_30S].score,
            early_aggression_score=ext[INDICATOR_EARLY_AGGRESSION].score,
            counter_ignition_signal=ext[INDICATOR_COUNTER_IGNITION].score,
            post_all_clear_state=ext[INDICATOR_POST_ALL_CLEAR].score,
            upper_board_density=ext[INDICATOR_UPPER_DENSITY].score,
            # Phase H1 form orientation 1
            gtr_orientation=ext[INDICATOR_GTR_ORIENTATION].score,
        )

    def compute_all_probabilistic(
        self,
        prob_board: ProbabilisticBoard,
        next_pair: tuple[int, int] | None = None,
        dnext_pair: tuple[int, int] | None = None,
        incoming_ojama: int = 0,
        opponent_board: Board | None = None,
        rotation_score: float = 0.5,
        n_samples: int = PROBABILISTIC_DEFAULT_SAMPLES,
        elapsed_sec: float | None = None,
        harass_count: int | None = None,
        early_aggression: float | None = None,
        counter_signal: float | None = None,
    ) -> IndicatorSet:
        """確率版 compute_all (Phase G、C-1).

        各 indicator が compute_probabilistic をオーバーライドしている
        場合はそれを呼び、未オーバーライドなら BaseIndicator のデフォルト
        実装 (= MLE 盤面に変換 → 通常 compute) で計算する。
        backwards compat のため通常 compute_all は維持。

        Args:
            prob_board: ProbabilisticBoard。
            next_pair: 次のツモ (TOP, BOT 色)。
            dnext_pair: ダブルネクストのツモ。
            incoming_ojama: 相手から飛来する予告お邪魔個数。
            opponent_board: 相手フィールド (確定盤面)。
            rotation_score: RotationTracker.score。
            n_samples: Monte Carlo サンプル数。
            elapsed_sec / harass_count / early_aggression / counter_signal:
                Phase H1 stateful 系の注入引数 (None で中立値)。

        Returns:
            IndicatorSet: 全指標の結果セット (compute_all と同じ形)。
        """
        # MLE 盤面を 1 回作って共有 (chain_result / extra 用)
        mle_board = prob_board.to_max_likelihood_board()
        chain_result = self._simulator.simulate(mle_board)
        results: dict[str, IndicatorResult] = {}
        for name, ind in self._indicators.items():
            kwargs: dict[str, Any] = {}
            if name == INDICATOR_HARASSMENT and incoming_ojama > 0:
                kwargs["incoming_ojama"] = incoming_ojama
            if name in (INDICATOR_EXTENSION, INDICATOR_SECOND):
                kwargs["opponent_board"] = opponent_board
            # n_samples は確率版を持つ indicator のみ対象 (デフォルト引数で吸収)
            if self._has_prob_override(ind):
                kwargs["n_samples"] = n_samples
            results[name] = ind.compute_probabilistic(
                prob_board,
                chain_result=chain_result,
                simulator=self._simulator,
                **kwargs,
            )
        next_score = self._calc_next_acceptance(
            mle_board, next_pair, dnext_pair,
        )
        ext = self._compute_extra(
            mle_board, chain_result, incoming_ojama, opponent_board,
            rotation_score=rotation_score,
            elapsed_sec=elapsed_sec,
            harass_count=harass_count,
            early_aggression=early_aggression,
            counter_signal=counter_signal,
        )
        results.update(ext)
        return self._build_indicator_set(results, ext, next_score)

    @staticmethod
    def _has_prob_override(ind: BaseIndicator) -> bool:
        """indicator が compute_probabilistic をオーバーライドしているかを判定."""
        own = type(ind).compute_probabilistic
        base = BaseIndicator.compute_probabilistic
        return own is not base

    def _compute_extra(
        self,
        board: Board,
        chain_result: ChainResult,
        incoming_ojama: int = 0,
        opponent_board: Board | None = None,
        rotation_score: float = 0.5,
        elapsed_sec: float | None = None,
        harass_count: int | None = None,
        early_aggression: float | None = None,
        counter_signal: float | None = None,
    ) -> dict[str, IndicatorResult]:
        """拡張指標を計算する (Phase J で 4 指標追加、Phase F で B-4/C-3 拡張、
        Phase H1 で機能 7 + 戦況 8 + 形分類 1 = 16 指標追加).
        """
        # Phase F (C-3): chain_timing_pressure は opponent_board を取るため
        # 通常リストから外して個別 instantiate する。
        ext_inds: list[BaseIndicator] = [
            ShapeScoreIndicator(),
            TouchingDensityIndicator(),
            TailHeightIndicator(),
            ColorVarianceIndicator(),
            KeyFlexibilityIndicator(),
            SubChainIndependenceIndicator(),
            # Phase J 追加 (3 指標は board のみ)
            AdjacentHeightDiffIndicator(),
            HighConnectionCountIndicator(),
            RequiredPuyoToFireIndicator(),
            # Phase K 追加 (board のみ)
            PostOjamaChainHealthIndicator(),
            IsolatedPuyoCountIndicator(),
            # Tier B 追加 (2026-05-05、形質指標 3 個)。
            # planning_entropy は 2026-05-06 numpy 化 + cache 高速化で再有効化。
            PlanningEntropyIndicator(),
            StructureSolidityIndicator(),
            BaseFlatnessIndicator(),
            # I-J 追加 (2026-05-06、B-1) 形テンプレ完成度 4 指標
            GtrCompletenessIndicator(),
            LlrCompletenessIndicator(),
            StaircaseCompletenessIndicator(),
            ZabutonCompletenessIndicator(),
            # B-1.b 追加 (2026-05-09) Sullen GTR / Fron 派生形 2 指標
            SullenGtrCompletenessIndicator(),
            FronCompletenessIndicator(),
            # Phase H1 機能能力 (board のみ、5 指標は対戦無依存)
            ReadyChainCountIndicator(),
            IgnitionDistanceIndicator(),
            CurrentFirePowerIndicator(),
            MaximumFirePowerIndicator(),
            MidGameResponseCapacityIndicator(),
            HarassmentReadinessIndicator(),
            OjamaDefenseCapacityIndicator(),
            # Phase H1 戦況 (board のみで成立する 3 指標)
            SelfChainDurationIndicator(),
            UpperBoardDensityIndicator(),
            # Phase H1 形分類 (board のみ)
            GtrOrientationIndicator(),
        ]
        out: dict[str, IndicatorResult] = {}
        for ind in ext_inds:
            out[ind.name] = ind.compute(
                board,
                chain_result=chain_result,
                simulator=self._simulator,
            )
        # Phase F (C-3): chain_timing_pressure に opponent_board を渡す
        timing = ChainTimingPressureIndicator()
        out[INDICATOR_CHAIN_TIMING] = timing.compute(
            board,
            chain_result=chain_result,
            simulator=self._simulator,
            opponent_board=opponent_board,
        )
        # incoming_ojama_pressure は incoming_ojama 引数を取るため別扱い
        incoming = IncomingOjamaPressureIndicator()
        out[INDICATOR_INCOMING_OJAMA] = incoming.compute(
            board,
            chain_result=chain_result,
            simulator=self._simulator,
            incoming_ojama=incoming_ojama,
        )
        # opponent_chain_threat は opponent_board 引数を取るため別扱い
        opp = OpponentChainThreatIndicator()
        out[INDICATOR_OPPONENT_THREAT] = opp.compute(
            board,
            chain_result=chain_result,
            simulator=self._simulator,
            opponent_board=opponent_board,
        )
        # opponent_offset_power も opponent_board 引数
        opp_off = OpponentOffsetPowerIndicator()
        out[INDICATOR_OPPONENT_OFFSET] = opp_off.compute(
            board,
            chain_result=chain_result,
            simulator=self._simulator,
            opponent_board=opponent_board,
        )
        # Phase F (B-4): rotation_skill は外部 Tracker のスコアを直接受け取る
        rot = RotationSkillIndicator()
        out[INDICATOR_ROTATION_SKILL] = rot.compute(
            board,
            chain_result=chain_result,
            simulator=self._simulator,
            rotation_score=rotation_score,
        )
        # Phase H1: opponent_board 必要な戦況 2 指標
        opp_dur = OppChainDurationIndicator()
        out[INDICATOR_OPP_CHAIN_DURATION] = opp_dur.compute(
            board,
            chain_result=chain_result,
            simulator=self._simulator,
            opponent_board=opponent_board,
        )
        chain_adv = ChainDurationAdvantageIndicator()
        out[INDICATOR_CHAIN_DURATION_ADV] = chain_adv.compute(
            board,
            chain_result=chain_result,
            simulator=self._simulator,
            opponent_board=opponent_board,
        )
        # Phase H1: state-holding 系 3 指標 (注入値が None なら中立 0.5)
        h_count = HarassEventCount30sIndicator()
        out[INDICATOR_HARASS_COUNT_30S] = h_count.compute(
            board,
            chain_result=chain_result,
            simulator=self._simulator,
            harass_count=harass_count,
        )
        early = EarlyAggressionScoreIndicator()
        out[INDICATOR_EARLY_AGGRESSION] = early.compute(
            board,
            chain_result=chain_result,
            simulator=self._simulator,
            early_aggression=early_aggression,
        )
        counter = CounterIgnitionSignalIndicator()
        out[INDICATOR_COUNTER_IGNITION] = counter.compute(
            board,
            chain_result=chain_result,
            simulator=self._simulator,
            counter_signal=counter_signal,
        )
        # Phase H1: post_all_clear_state は elapsed_sec を取る
        pac = PostAllClearStateIndicator()
        out[INDICATOR_POST_ALL_CLEAR] = pac.compute(
            board,
            chain_result=chain_result,
            simulator=self._simulator,
            elapsed_sec=elapsed_sec,
        )
        return out

    def _calc_next_acceptance(
        self,
        board: Board,
        next_pair: tuple[int, int] | None,
        dnext_pair: tuple[int, int] | None,
    ) -> float:
        """next/dnext が揃っていれば受け入れ余地を計算、未指定なら中立値。"""
        if next_pair is None or dnext_pair is None:
            return NEXT_ACCEPTANCE_NEUTRAL
        score, _ = _compute_next_acceptance(
            board, next_pair, dnext_pair, self._simulator,
        )
        return score

    def indicator_names(self) -> list[str]:
        """登録済み指標名のリストを返す。"""
        return list(self._indicators.keys())

    @staticmethod
    def _default_indicators() -> list[BaseIndicator]:
        """CLAUDE.md 指定順の8指標を返す。"""
        return [
            MainChainMaturityIndicator(),
            ExtensionPotentialIndicator(),
            SubChainQualityIndicator(),
            HarassmentResistanceIndicator(),
            DeathRiskIndicator(),
            OffsetPowerIndicator(),
            SecondChainPotentialIndicator(),
            FieldEfficiencyIndicator(),
        ]
