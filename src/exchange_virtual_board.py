"""発火イベントから「発火後の仮想盤面ペア」を再構成する純関数群 (Step2)。

## 背景 (アーキ設計 案C: 仮想盤面2回評価)
発火検知の瞬間 (`ChainEvent.trigger_sec`) に、攻撃側の連鎖消化後盤面と
相手側のおじゃま着弾後盤面を **決定論的に** 再構成できれば、実際に連鎖の
アニメーションが終わって認識が STABLE 復帰するのを待たずに前後盤面の
指標を計算できる (Step0 検証: `logs/step0_diag/aggregate_result.log` で
機能D早期発火の凍結盤面はオフライン `board_ref_index` と92.4%一致確認済み)。

## 設計方針 (CLAUDE.md 準拠)
- **stateless**: 本モジュールは state を持たない。`ChainSimulator` インスタンス
  はオプション引数として受け取るのみ (呼び出し側がキャッシュ管理する外部
  wrapper)。
- **既存資産の再利用・再実装禁止**: 連鎖消化は `src.chain.ChainSimulator.simulate`、
  おじゃま着弾は `src.chain.ChainSimulator.drop_ojama`
  (6列均等 floor(N/6) + 端数ランダム、`reference_ojama_landing_pattern`
  仕様通り) をそのまま使う。窒息判定は `Board.is_dead()` を使う。
- **端数配置の再現性**: `drop_ojama` の `seed` 引数に、盤面内容から導出した
  決定論的シードを渡す (`src.indicators_v2._expected_fire_seed` と同じ思想:
  `zlib.crc32` はプロセス非依存の決定的ハッシュ、同一入力には常に同一結果)。
  本モジュールには「行キー」(video_id/t_sec 等) を受け取る引数が無い
  (純関数として盤面とおじゃま数のみを入力にする設計のため)、その代替として
  攻撃側盤面・相手側盤面・おじゃま数を組み合わせてシードを作る
  (同一イベントには常に同一結果、異なるイベントではほぼ確実に異なる結果)。
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

from src.board import Board
from src.chain import ChainResult, ChainSimulator
from src.ojama_accounting import cancel_own_pending_then_send_surplus
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.scoring import OJAMA_MAX_DROP_PER_TURN

# ============================
# 定数定義
# ============================

# おじゃまシード計算時に攻撃側/相手側で異なるシードにするための salt。
# 同一盤面 (自分自身に降らせる/相手に降らせる) でも端数列パターンが
# 一致しないようにするためのマジックナンバー回避定数。
_OJAMA_SEED_SALT_TO_OPPONENT: int = 0xA11
_OJAMA_SEED_SALT_TO_ATTACKER: int = 0xB22

# 2026-08-03 指摘 (欠陥E-1): 仮想着弾の1ターン上限。
# 実ゲームでは1ターンに降るおじゃまは最大 OJAMA_MAX_DROP_PER_TURN (=30個、
# 5段×6列、src/scoring.py で既定義・docs/PUYO_RULES_CONFIRMED_2026-07-22.md
# L17で確定済み) であり、超過分は複数ターンに分割して降る (即死ではない)。
# 旧実装は net_ojama_after_pred (連続値、時に300〜450個規模) を1回で全量
# 投下しており、単独盤面容量(72セル)を超える大型連鎖ほど不当に
# opponent_dead を True 判定していた (main実測 match_02 2992.93s イベント:
# 予測448個を2Pの空き36セルへ全量投下→即窒息判定、実際は2Pは生存し続けた)。
# 本定数を超える分は「まだ空中(次ターン以降に降る予定)」として扱い、
# 本関数の1回評価では物理配置しない (VirtualBoardPair の ojama_to_* は
# 実際に配置した個数=上限適用後の値を返す)。

# net_ojama_after_pred の丸め方式: 予測値は連続値 (期待値) だが実際の着弾は
# 整数個のため四捨五入する。0.5 を挟む境界での偏りは無視できる規模
# (おじゃま数は通常 6 以上、四捨五入誤差は最大 0.5 個分のみ)。


@dataclass(frozen=True)
class VirtualBoardPair:
    """発火イベント後の仮想盤面ペア (Step2 出力)。

    Attributes:
        attacker_board_after: 攻撃側の連鎖消化後盤面 (ChainSimulator.simulate
            の final_board)。
        opponent_board_after: 相手側のおじゃま着弾後仮想盤面。
        ojama_to_opponent: 相手側に着弾させたおじゃま数 (0以上)。
        ojama_to_attacker: 攻撃側に着弾させたおじゃま数 (net_ojama_after_pred
            が負 = 相手が返し勝つ予測の場合のみ 0 より大きくなる)。
        attacker_dead: attacker_board_after が窒息判定 (Board.is_dead()) なら True。
        opponent_dead: opponent_board_after が窒息判定なら True。
        chain_result: attacker側 simulate の生結果 (chain_count 等を下流で
            使う場合のために保持する)。
    """
    attacker_board_after: Board
    opponent_board_after: Board
    ojama_to_opponent: int
    ojama_to_attacker: int
    attacker_dead: bool
    opponent_dead: bool
    chain_result: ChainResult


def _deterministic_ojama_seed(
    attacker_board: Board, opponent_board: Board, ojama_count: int, salt: int,
) -> int:
    """盤面ペア + おじゃま数 + salt から決定論的乱数シードを導出する。

    `src.indicators_v2._expected_fire_seed` と同じ思想 (zlib.crc32、
    プロセス非依存の決定的ハッシュ) を、盤面2枚 + おじゃま数の組に拡張した
    もの。同一の (attacker_board, opponent_board, ojama_count) には常に
    同一のシードを返す (stateless, 再現性確保)。

    Args:
        attacker_board: 攻撃側盤面 (連鎖消化後)。
        opponent_board: 相手側盤面。
        ojama_count: 着弾させるおじゃま数。
        salt: 攻撃側/相手側で異なるシードにするための固定値。

    Returns:
        int: `ChainSimulator.drop_ojama` の `seed` 引数にそのまま渡せる値。
    """
    crc = zlib.crc32(attacker_board._grid.tobytes())
    crc = zlib.crc32(opponent_board._grid.tobytes(), crc)
    crc = zlib.crc32(ojama_count.to_bytes(8, "big", signed=True), crc)
    return crc ^ salt


def reconstruct_virtual_board_pair(
    before_board: Board,
    opponent_board: Board,
    net_ojama_after_pred: float,
    simulator: "ChainSimulator | None" = None,
) -> VirtualBoardPair:
    """発火イベントから仮想盤面ペアを再構成する (Step2 メイン API)。

    攻撃側は `before_board` (= ChainEvent.before_board、発火直前確定盤面)
    を `ChainSimulator.simulate()` で連鎖消化し、その `final_board` を採用する
    (連鎖ロジックは一切再実装しない)。

    相手側は `net_ojama_after_pred` (正味おじゃま予測値、相手へ送る数を正、
    相手が返し勝って攻撃側が受け取る数を負とする符号規約) を
    `ChainSimulator.drop_ojama()` で着弾させる。正値なら相手側盤面へ、
    負値なら攻撃側 (連鎖消化後) 盤面へ着弾させる。

    Args:
        before_board: 発火直前の攻撃側盤面 (ChainEvent.before_board)。
        opponent_board: 発火時点の相手側盤面 (連鎖消化前、時刻最近傍で
            復元したもの)。
        net_ojama_after_pred: 予測正味おじゃま数。正値=相手へ送る数、
            負値=相手が返し勝って攻撃側が受け取る数。NaN は不可
            (呼び出し側が事前に欠損行を除外すること、本関数は ValueError で
            拒否する。0 への silent fallback は行わない = fail-silent 回避)。
        simulator: 再利用する ChainSimulator インスタンス (省略時は関数内で
            新規生成)。呼び出し側が多数のイベントに対して繰り返し呼ぶ場合は
            同一インスタンスを渡すと simulate() の内部キャッシュが効く。

    Returns:
        VirtualBoardPair: 前後盤面・着弾おじゃま数・窒息フラグを含む結果。

    Raises:
        ValueError: net_ojama_after_pred が NaN の場合。
    """
    if net_ojama_after_pred != net_ojama_after_pred:  # NaN 検出 (math.isnan 相当)
        raise ValueError(
            "net_ojama_after_pred が NaN です。呼び出し側で欠損行を除外する"
            "か、明示的な既定値を選んでから渡してください (silent fallback 禁止)。"
        )

    # 幽霊連鎖ルール (2026-08-10 本番ON採用): production_config.py が単一情報源。
    sim = simulator if simulator is not None else ChainSimulator(
        exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    )
    chain_result = sim.simulate(before_board)
    attacker_after = chain_result.final_board.copy()
    opponent_after = opponent_board.copy()

    # 欠陥E-1: 1ターンで物理配置するのは OJAMA_MAX_DROP_PER_TURN (=30個) まで
    # (超過分はまだ空中、本関数の1回評価では配置しない、モジュール冒頭の
    # 定数コメント参照)。seed もこの「実際に配置する個数」から導出する
    # (物理的に同じ配置結果になる入力は同じシードになるのが自然なため)。
    ojama_count = min(int(round(abs(net_ojama_after_pred))), OJAMA_MAX_DROP_PER_TURN)
    ojama_to_opponent = 0
    ojama_to_attacker = 0
    if ojama_count > 0:
        if net_ojama_after_pred >= 0:
            seed = _deterministic_ojama_seed(
                attacker_after, opponent_after, ojama_count,
                _OJAMA_SEED_SALT_TO_OPPONENT,
            )
            opponent_after = sim.drop_ojama(opponent_after, ojama_count, seed=seed)
            ojama_to_opponent = ojama_count
        else:
            seed = _deterministic_ojama_seed(
                attacker_after, opponent_after, ojama_count,
                _OJAMA_SEED_SALT_TO_ATTACKER,
            )
            attacker_after = sim.drop_ojama(attacker_after, ojama_count, seed=seed)
            ojama_to_attacker = ojama_count

    return VirtualBoardPair(
        attacker_board_after=attacker_after,
        opponent_board_after=opponent_after,
        ojama_to_opponent=ojama_to_opponent,
        ojama_to_attacker=ojama_to_attacker,
        attacker_dead=attacker_after.is_dead(),
        opponent_dead=opponent_after.is_dead(),
        chain_result=chain_result,
    )


# ============================
# 両者同時発火の決着計算 (2026-08-13、デモレビュー #9 対処)
# ============================
#
# `reconstruct_virtual_board_pair` は「攻撃側1名 + 予測正味おじゃま」の
# 1方向モデル用 (スタッキングモデルの予測値を受け取る設計)。デモレビュー #9
# (両者発火後の勝率乱高下) は「両側が同時に連鎖を撃っている」実況面での
# 決着計算が必要で、双方の `ChainEvent.before_board` が既知 (=予測不要、
# 双方とも実際に確定した起点盤面から直接 simulate できる) という点が
# 上記関数の前提と異なるため、専用関数を追加する。ただし内部で使う部品
# (ChainSimulator.simulate/drop_ojama、_deterministic_ojama_seed、
# OJAMA_MAX_DROP_PER_TURN 上限) は本モジュール既存資産をそのまま再利用する
# (再実装しない)。


@dataclass(frozen=True)
class MutualExchangeResult:
    """両者同時発火の決着計算結果 (#9 対処)。

    Attributes:
        board_p1_after: 1P の連鎖消化 + 着弾後仮想盤面。
        board_p2_after: 2P の連鎖消化 + 着弾後仮想盤面。
        dropped_to_p1: 1P へ実際に配置したおじゃま数 (1ターン上限適用後)。
        dropped_to_p2: 2P へ実際に配置したおじゃま数。
        leftover_p1: 1P へまだ配置していない残り (次ターン繰越=forecast相当)。
        leftover_p2: 2P へまだ配置していない残り。
        p1_dead: board_p1_after が窒息判定なら True。
        p2_dead: board_p2_after が窒息判定なら True。
        chain_result_p1: 1P simulate の生結果 (chain_count 等の下流利用のため保持)。
        chain_result_p2: 2P simulate の生結果。
    """
    board_p1_after: Board
    board_p2_after: Board
    dropped_to_p1: int
    dropped_to_p2: int
    leftover_p1: int
    leftover_p2: int
    p1_dead: bool
    p2_dead: bool
    chain_result_p1: ChainResult
    chain_result_p2: ChainResult


def _cancel_mutual_pending(
    gen_p1_ojama: int, gen_p2_ojama: int, pending_p1: int, pending_p2: int,
) -> tuple[int, int]:
    """両者同時発火の相殺会計 (純関数、stateless)。

    実ゲーム仕様 (reference_ojama_landing_pattern) の相殺規則「自分の生成量で
    まず自分への予告を相殺し、余剰を相手へ送る」を、双方が同時に発火した
    場合に拡張する。同時発火のため、一方の余剰が他方の予告をさらに
    相殺する交差処理は行わない (両者とも「発火した瞬間」に手持ちの生成量
    だけを使い切る単純化。1フレーム内の同時発生を扱うための近似であり、
    厳密な時系列上の連鎖的相殺までは再現しない)。
    既存 `cancel_own_pending_then_send_surplus` (ojama_accounting.py) を
    other_pending=0 固定で2回呼び、各々の surplus を手動で交換して
    合算する (再実装しない)。

    Returns:
        (最終的に p1 へ向かう予告個数, 最終的に p2 へ向かう予告個数)。
    """
    own1, surplus1 = cancel_own_pending_then_send_surplus(gen_p1_ojama, pending_p1, 0)
    own2, surplus2 = cancel_own_pending_then_send_surplus(gen_p2_ojama, pending_p2, 0)
    return own1 + surplus2, own2 + surplus1


def resolve_mutual_exchange(
    before_p1: Board, before_p2: Board,
    gen_p1_ojama: int, gen_p2_ojama: int,
    pending_p1: int, pending_p2: int,
    simulator: "ChainSimulator | None" = None,
) -> MutualExchangeResult:
    """両者同時発火 (#9 デモレビュー対処) の決着を1回で計算する。

    双方の発火直前盤面 (`ChainEvent.before_board`) から連鎖を完走
    シミュレーションし (幽霊連鎖ルールは `GHOST_CHAIN_RULE_ENABLED` に従う、
    本番採用フラグと同じ経路)、得点→おじゃま換算済みの生成量
    (`gen_p1_ojama`/`gen_p2_ojama`、換算は呼び出し側が `score_to_ojama` で
    行う設計 — `reconstruct_virtual_board_pair` と同じ役割分担) を相殺し、
    1ターン上限 (`OJAMA_MAX_DROP_PER_TURN`) まで着弾させた仮想盤面ペアを
    返す。

    Args:
        before_p1: 1P の発火直前盤面。
        before_p2: 2P の発火直前盤面。
        gen_p1_ojama: 1P の連鎖が生成したおじゃま数 (お邪魔換算済み、0以上)。
        gen_p2_ojama: 2P の連鎖が生成したおじゃま数。
        pending_p1: 交換前に既に確定している 1P への予告おじゃま数。
        pending_p2: 交換前に既に確定している 2P への予告おじゃま数。
        simulator: 再利用する ChainSimulator (省略時は関数内で新規生成)。

    Returns:
        MutualExchangeResult。

    Raises:
        ValueError: gen_p1_ojama/gen_p2_ojama が負の場合。
    """
    if gen_p1_ojama < 0 or gen_p2_ojama < 0:
        raise ValueError("gen_p1_ojama/gen_p2_ojama は0以上である必要があります")
    sim = simulator if simulator is not None else ChainSimulator(
        exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    )
    chain_result_p1 = sim.simulate(before_p1)
    chain_result_p2 = sim.simulate(before_p2)
    board_p1 = chain_result_p1.final_board.copy()
    board_p2 = chain_result_p2.final_board.copy()
    total_p1, total_p2 = _cancel_mutual_pending(
        int(gen_p1_ojama), int(gen_p2_ojama), int(pending_p1), int(pending_p2))
    drop_p1 = min(total_p1, OJAMA_MAX_DROP_PER_TURN)
    drop_p2 = min(total_p2, OJAMA_MAX_DROP_PER_TURN)
    if drop_p1 > 0:
        seed = _deterministic_ojama_seed(
            board_p1, board_p2, drop_p1, _OJAMA_SEED_SALT_TO_OPPONENT)
        board_p1 = sim.drop_ojama(board_p1, drop_p1, seed=seed)
    if drop_p2 > 0:
        seed = _deterministic_ojama_seed(
            board_p2, board_p1, drop_p2, _OJAMA_SEED_SALT_TO_ATTACKER)
        board_p2 = sim.drop_ojama(board_p2, drop_p2, seed=seed)
    return MutualExchangeResult(
        board_p1_after=board_p1, board_p2_after=board_p2,
        dropped_to_p1=drop_p1, dropped_to_p2=drop_p2,
        leftover_p1=total_p1 - drop_p1, leftover_p2=total_p2 - drop_p2,
        p1_dead=board_p1.is_dead(), p2_dead=board_p2.is_dead(),
        chain_result_p1=chain_result_p1, chain_result_p2=chain_result_p2,
    )
