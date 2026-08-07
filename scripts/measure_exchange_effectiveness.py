"""#24 打ち合い計測器 Step2「有効性判定MC」のオーケストレーション層 (2026-07-29)。

Step1 (scripts/measure_exchange_dynamics.py、OppCoverageStatus) と
src/indicators_v2.py の counter_reach_probability / _fast (#24 Step2の本体、
既存 expected_fire_power の rollout 機構を「平均得点」から「閾値到達確率」
に差し替えた新規関数) を接続する。

設計方針 (userとのすり合わせ済み、reference_saisoku_exchange_model_2026-07-22):
    - 有効な催促 = 着弾までに相手が返せる見込みが50%以下
      (COUNTER_REACH_EFFECTIVE_THRESHOLD_PROB、src/indicators_v2.py)。
    - 相手の応手の返り量 (お邪魔換算) によって有利/やや不利/不利を判定する。
    - 相手が OPP_CHAINING (連鎖中=既に発火済み、応手不能) の場合、応手確率は
      定義上ほぼ0として扱う (2026-07-29の重要な発見)。

⚠️ 着弾遅延 (何手先まで数えるか) の見積もりについて (正直な注記):
    memory `project_exchange_measurement_foundation_2026-07-22` の実測は
    n が極小 (連鎖1以上の着弾検出は20件のみ) で確定的な値ではない。
    既存 TIME_PER_CHAIN_SEC (0.30秒/連鎖) は大幅な過小評価と判明済みのため
    使わず、本モジュール専用の2アンカー点線形補間を使う (要再計測、
    定数は独立して調整可能)。

本モジュールは「やる」範囲 (Step2) のみ:
    1. 応手確率計算 (counter_reach_probability系への薄い接続)。
    2. 有効性判定 (<=50%で有効) + 有利/やや不利/不利の3値判定。
    3. OppCoverageStatus接続 (OPP_CHAININGは確率0扱い)。
    「やらない」: collectへの本配線 (Step3)、AUC検証 (Step4)。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.board import Board
from src.indicators_v2 import (
    COUNTER_REACH_EFFECTIVE_THRESHOLD_PROB,
    CounterReachResult,
    counter_reach_probability,
    counter_reach_probability_fast,
    estimate_chain_anim_duration_sec,
    expected_fire_power,
    ojama_damage,
)
from src.indicators_v2 import SEC_PER_HAND as EXISTING_SEC_PER_HAND
from scripts.measure_exchange_dynamics import (
    OJAMA_PER_DAN,
    RETURN_ONE_DAN_UPPER,
    RETURN_TWO_DAN_LOWER,
    OppCoverageStatus,
)

# ============================
# 着弾遅延 -> 相手の手数見積もり (物差し一本化、2026-08-01 Step0)
# ============================
#
# 旧実装は本モジュール専用の2アンカー点線形補間 (memory
# project_exchange_measurement_foundation_2026-07-22、n=2の暫定値) を
# 使っていたが、23動画418イベント実測ベースで最も検証件数が多い
# src.indicators_v2.estimate_chain_anim_duration_sec (CHAIN_ANIM_PER_STEP_SEC
# =0.4秒/連鎖) に一本化した (2026-08-01、user/アーキ確定)。

# counter_reach_probability系がサポートするK水準の上限 (src/indicators_v2.py
# EXPECTED_FIRE_K_LEVELS と同じ 1..4)。着弾遅延から逆算した手数がこれを
# 超える場合はこの上限にクランプする (K=5以上のMCは本Step2の対象外)。
MAX_SUPPORTED_K_HANDS: int = 4


def estimate_landing_delay_sec(chain_count: int) -> float:
    """連鎖数から着弾遅延秒数を推定する (物差し一本化版)。

    src.indicators_v2.estimate_chain_anim_duration_sec (CHAIN_ANIM_PER_STEP_SEC
    =0.4秒/連鎖、23動画418イベント実測ベース) に委譲する。連鎖数が負・0の
    場合は0.0にクランプする (委譲先の関数が保証)。
    """
    return estimate_chain_anim_duration_sec(float(chain_count))


def estimate_available_hands(chain_count: int) -> int:
    """着弾までに相手が打てる手数 (K) を概算する。

    user伝授 (reference_ojama_landing_gated_by_placement_2026-07-29):
    「おじゃまは連鎖完了後に受け側のツモが着地した時に降る」ため、
    floor(連鎖アニメ時間 ÷ 1手時間) + 1 手 (受け側の着地1手分) が正しい
    見積もりとなる。+1 により最低手数は常に1以上になる
    (旧実装の「0手=応手する暇がない」を許容する設計は、この修正で
    到達しなくなる: delay_sec>=0 なら floor(delay/hand)>=0 なので
    +1 後は必ず>=1)。
    counter_reach_probability の対応レンジ (K=1..4) に上限クランプする。
    """
    delay_sec = estimate_landing_delay_sec(chain_count)
    hands = int(delay_sec // EXISTING_SEC_PER_HAND) + 1
    return min(MAX_SUPPORTED_K_HANDS, hands)


# ============================
# 有効性判定 (<=50%で有効)
# ============================


def is_effective_saisoku(reach_probability: float) -> bool:
    """有効な催促 = 着弾までに相手が返せる見込みが50%以下 (user確定定義)。

    reach_probability が NaN (判定不能、下記 EffectivenessJudgement 参照)
    の場合は False を返さず ValueError を送出する (誤って「有効」と
    誤認させないため、呼び出し側で NaN を先に弾く設計とする)。
    """
    if reach_probability != reach_probability:  # NaN チェック (math.isnan 相当)
        raise ValueError("reach_probability が NaN です (判定不能ケースは呼び出し側で先に弾くこと)")
    return reach_probability <= COUNTER_REACH_EFFECTIVE_THRESHOLD_PROB


# ============================
# 有利/やや不利/不利 (返り量ベース3値判定)
# ============================

# Step1 の RETURN_ONE_DAN_UPPER (1.2段) / RETURN_TWO_DAN_LOWER (1.8段) を
# お邪魔換算個数に変換した閾値 (OJAMA_PER_DAN=6個/段、Step1定数をそのまま
# 流用しマジックナンバーを増やさない)。
RETURN_ONE_DAN_OJAMA: float = RETURN_ONE_DAN_UPPER * OJAMA_PER_DAN  # = 7.2個
RETURN_TWO_DAN_OJAMA: float = RETURN_TWO_DAN_LOWER * OJAMA_PER_DAN  # = 10.8個


class ExchangeAdvantageLabel(Enum):
    """催促の有利/やや不利/不利 (user定義: 相手の応手の返り量で判定)。"""
    ADVANTAGE = "有利"       # 相手はほぼ返せない (小さい返りへの到達確率も低い)
    SLIGHT_DISADVANTAGE = "やや不利"  # 中間
    DISADVANTAGE = "不利"    # 相手は大きく返せる見込みが高い


def classify_exchange_advantage(
    prob_return_one_dan: float, prob_return_two_dan: float,
) -> ExchangeAdvantageLabel:
    """有利/やや不利/不利の3値判定。

    Args:
        prob_return_one_dan: 相手の反撃が RETURN_ONE_DAN_OJAMA (1.2段相当)
            以上に届く確率 (counter_reach_probability(threshold_ojama=
            RETURN_ONE_DAN_OJAMA) の結果)。
        prob_return_two_dan: 相手の反撃が RETURN_TWO_DAN_OJAMA (1.8段相当)
            以上に届く確率。

    Returns:
        ExchangeAdvantageLabel: 小さい返りにすら届きにくい (<=50%) なら
        有利、大きい返りにまで届きやすい (>50%) なら不利、その中間は
        やや不利。
    """
    if prob_return_one_dan <= COUNTER_REACH_EFFECTIVE_THRESHOLD_PROB:
        return ExchangeAdvantageLabel.ADVANTAGE
    if prob_return_two_dan > COUNTER_REACH_EFFECTIVE_THRESHOLD_PROB:
        return ExchangeAdvantageLabel.DISADVANTAGE
    return ExchangeAdvantageLabel.SLIGHT_DISADVANTAGE


# ============================
# OppCoverageStatus 接続
# ============================


@dataclass(frozen=True)
class EffectivenessJudgement:
    """1発火イベント分の有効性判定結果。

    Attributes:
        reach_probability: 応手到達確率 (RETURN_ONE_DAN_OJAMA基準)。
            OPP_CHAINING なら0.0固定、判定不能なら NaN。
        is_effective: is_effective_saisoku の結果 (NaN時は None)。
        advantage_label: classify_exchange_advantage の結果 (NaN時は None)。
        coverage_status: 判定に使った OppCoverageStatus (デバッグ用)。
    """
    reach_probability: float
    is_effective: "bool | None"
    advantage_label: "ExchangeAdvantageLabel | None"
    coverage_status: OppCoverageStatus


def judge_exchange_effectiveness(
    opp_board: "Board | None",
    coverage_status: OppCoverageStatus,
    chain_count: int,
    elapsed_sec: float = 0.0,
    mode: str = "precise",
) -> EffectivenessJudgement:
    """1発火イベント分の有効性 (有効/無効) + 有利不利ラベルを判定する。

    2026-07-29の重要な発見: 相手が OPP_CHAINING (連鎖中=既に発火済み、
    応手不能) の場合は counter_reach_probability を計算せず確率0固定と
    する (発火済みなので着弾に対して原理的に応手不可能)。それ以外で
    opp_board が None (観測不能: UNOBSERVED/MATCH_END/UNKNOWN) の場合は
    判定不能として NaN を返す (0に丸めて「有効」と誤認させない)。

    Args:
        opp_board: 相手側の STABLE 確定盤面 (観測できていれば)。
        coverage_status: Step1 の OppCoverageStatus。
        chain_count: 攻撃側の連鎖数 (着弾までの手数見積もりに使う)。
        elapsed_sec: 試合相対経過秒 (マージンタイム換算用)。
        mode: "precise" (ChainSimulator厳密) または "fast" (chain_bitboard
            バッチ近似)。二層設計 (project_dual_mode_indicator_design_
            2026-07-22): オフライン動画評価は precise、リアルタイムは
            fast を使う想定。

    Returns:
        EffectivenessJudgement。
    """
    if coverage_status == OppCoverageStatus.OPP_CHAINING:
        return EffectivenessJudgement(
            reach_probability=0.0, is_effective=True,
            advantage_label=ExchangeAdvantageLabel.ADVANTAGE,
            coverage_status=coverage_status,
        )
    if opp_board is None or opp_board.is_dead():
        return EffectivenessJudgement(
            reach_probability=float("nan"), is_effective=None,
            advantage_label=None, coverage_status=coverage_status,
        )

    k_hands = estimate_available_hands(chain_count)
    # 2026-08-01 Step0: estimate_available_hands は
    # floor(遅延/1手時間)+1 のため必ず 1 以上を返す (受け側の着地1手分を
    # 必ず含むため)。旧実装は k_hands<=0 (=応手する暇がない) を許容する
    # 分岐を持っていたが、+1 修正後はこの分岐に到達しない。dead code を
    # 放置せず assert で到達不能を明示する (回帰テスト:
    # tests/test_landing_delay_unification.py
    # test_k_hands_never_reaches_zero_branch)。
    assert k_hands >= 1, "estimate_available_hands は常に1以上を返す設計 (Step0)"

    fn = counter_reach_probability_fast if mode == "fast" else counter_reach_probability
    result_one: CounterReachResult = fn(
        opp_board, threshold_ojama=RETURN_ONE_DAN_OJAMA, elapsed_sec=elapsed_sec,
    )
    result_two: CounterReachResult = fn(
        opp_board, threshold_ojama=RETURN_TWO_DAN_OJAMA, elapsed_sec=elapsed_sec,
    )
    prob_one = result_one.probabilities[k_hands]
    prob_two = result_two.probabilities[k_hands]
    return EffectivenessJudgement(
        reach_probability=prob_one,
        is_effective=is_effective_saisoku(prob_one),
        advantage_label=classify_exchange_advantage(prob_one, prob_two),
        coverage_status=coverage_status,
    )


# ============================
# Step5: 修正シミュの評価用関数 (2026-08-01、2026-08-02符号バグ修正)
# ============================
#
# 既存資産の組み替えのみ (新規ロジック最小): estimate_available_hands
# (Step0で+1修正済み) で相手の残り手数を見積もり、expected_fire_power で
# 相手の期待反撃量を求め、攻撃側が送ったお邪魔から差し引いた正味を
# ojama_damage (折れ点12個/18個の非線形構造、user伝授の暫定実装) に通して
# 最終ダメージスコアを返す。較正定数の再フィットは行わない
# (CHAIN_ANIM_PER_STEP_SEC=0.4 を外部較正値としてそのまま流用)。
#
# ⚠️ 2026-08-02 バグ修正 (main精査済み・実バグ確定):
# net_expected は相殺ルール上「相手側に着弾する」正味おじゃまなのに、旧実装は
# ojama_damage(attacker_board_after_fire, ...) と**攻撃側自身の盤面**で威力評価
# していた。user伝授ドメインルール (memory reference_ojama_damage_nonlinear:
# 威力は受け側の残り容量に依存) に反するため、ojama_damage(opp_board, ...) に
# 修正した (受け側=相手の盤面で評価)。


def estimate_expected_net_damage(
    attacker_ojama_sent: float,
    opp_board: Board,
    opp_coverage_status: OppCoverageStatus,
    attacker_chain_count: int,
    attacker_board_after_fire: Board,
    elapsed_sec: float = 0.0,
    mode: str = "precise",
) -> float:
    """発火1件分の「相手の反撃を差し引いた正味ダメージ」スコアを計算する。

    手順 (既存資産の組み替えのみ):
      1. k_hands = estimate_available_hands(attacker_chain_count)
         (着弾までに相手が打てる手数、Step0で+1修正済み=常に1以上)。
      2. 相手が OPP_CHAINING (連鎖中=応手不能) なら期待反撃量は0固定。
      3. それ以外は expected_fire_power(opp_board, k_levels=(k_hands,)) の
         raw (お邪魔換算の平均ツモ期待火力) を期待反撃量とする。
      4. net_expected = attacker_ojama_sent − 期待反撃量 (負値は0にクランプ、
         「相手が攻撃側より多く返す見込み」を負のダメージにしない)。
      5. ojama_damage(opp_board, net_expected) のスコア (0〜1、折れ点
         12個/18個の非線形構造は再利用・再実装しない) を返す。net_expected
         は相手側に着弾する正味おじゃまのため、受け側=相手の盤面
         (opp_board) で評価する (2026-08-02 修正、旧実装は attacker_board_
         after_fire で評価しておりバグだった)。

    ⚠️ 正直な注記: mode="fast" は counter_reach_probability_fast のような
    高速版が expected_fire_power にはまだ存在しない (2026-08-01時点)。
    interface 統一 (project_dual_mode_indicator_design_2026-07-22) のため
    引数は受け取るが、現状は "precise"/"fast" どちらでも同じ計算になる
    (将来 expected_fire_power_fast が実装されたら差し替える窓口として残す)。

    Args:
        attacker_ojama_sent: 攻撃側が実際に送ったお邪魔量 (個数)。
        opp_board: 相手側の STABLE 確定盤面 (発火時点、破壊しない)。
            net_expected の ojama_damage 評価にもこの盤面を使う
            (2026-08-02修正、受け側=相手基準に統一)。
        opp_coverage_status: Step1 の OppCoverageStatus。
        attacker_chain_count: 攻撃側の連鎖数 (着弾遅延見積もり用)。
        attacker_board_after_fire: 現在未使用 (2026-08-02のバグ修正で
            ojama_damage の評価基準を opp_board に変更したため)。
            backwards compat のため引数は削除せず保持する (既存呼び出し元
            scripts/augment_exchange_labels_with_sim.py 等のシグネチャを
            壊さないため)。
        elapsed_sec: 試合相対経過秒 (マージンタイム換算用)。
        mode: "precise" または "fast" (現状は挙動同一、上記注記参照)。

    Returns:
        float: 正味ダメージスコア (0〜1、大きいほど攻撃側に有利
            [相手が受ける期待正味ダメージが大きいほど攻撃側に有利]。
            2026-08-02修正: 旧docstringは「大きいほど攻撃側に不利」と
            誤記していた、実装意図 [Step5=期待正味ダメージ=攻撃の価値]
            に照らして訂正)。
    """
    k_hands = estimate_available_hands(attacker_chain_count)
    if opp_coverage_status == OppCoverageStatus.OPP_CHAINING:
        expected_counter_ojama = 0.0
    else:
        result = expected_fire_power(
            opp_board, k_levels=(k_hands,), elapsed_sec=elapsed_sec,
        )
        expected_counter_ojama = result.values[k_hands].raw
    net_expected = max(0.0, attacker_ojama_sent - expected_counter_ojama)
    damage = ojama_damage(opp_board, ojama_count=net_expected)
    return float(damage.score)
