"""案C: 既知3組の全列挙から脅威の差を測る純粋関数（既定OFF）。

Phase 4 の割当済みqueueを外部から受け取る。履歴・未来ツモ・勝敗は読まない。
欠測値はNaNで保持する。未知の4組目以降は推測せず、応手は既知範囲の最大量。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import prefire_v5_search as base
from src import prefire_v5b_search as exact
from src.board import BOARD_COLS, BOARD_ROWS
from src.exchange_event_features import validate_elapsed
from src.exchange_event_landing import remaining_hands
from src.indicators_v2 import SEC_PER_HAND
from src.prefire_v5c_search import TransitionTable, side_options

KNOWN_HANDS = exact.MAX_KNOWN_HANDS
CAPACITY = BOARD_ROWS * BOARD_COLS
RAW_COLUMNS = ('attack', 'attack_hands', 'counter', 'difference', 'rows')
FLAG_COLUMNS = ('attack_missing', 'counter_missing', 'beyond_known_horizon')
SIDE_COLUMNS = RAW_COLUMNS + FLAG_COLUMNS
# 非線形の形を採点前に固定: 1段刻み、盤面の高さで最終区分。
ROW_BOUNDS = tuple(float(n) for n in range(1, BOARD_ROWS + 1))
ENCODED_SIDE_COLUMNS = SIDE_COLUMNS + tuple(f'rows_ge_{int(n)}' for n in ROW_BOUNDS)
COLUMNS = tuple(f'{side}_{column}' for side in ('own', 'opp', 'diff')
                for column in ENCODED_SIDE_COLUMNS)


@dataclass(frozen=True)
class Threat:
    """片側の攻撃・その攻撃に対する相手の応手（個数は未正規化）。"""

    attack: float = np.nan
    attack_hands: float = np.nan
    counter: float = np.nan
    difference: float = np.nan
    rows: float = np.nan
    attack_missing: bool = True
    counter_missing: bool = True
    beyond_known_horizon: bool = False

    def raw(self) -> np.ndarray:
        """監査用の量と欠測理由を固定順で返す。"""
        return np.asarray([getattr(self, name) for name in SIDE_COLUMNS], dtype=float)


@dataclass(frozen=True)
class Fire:
    """全終端のうち発火終端の、脅威計算に必要な3量。"""

    score: int
    chains: int
    consumed: int


def _options(position: base.Position, side: int) -> tuple[Fire | base.Position, ...] | None:
    """同じ全列挙の縮約APIを使用。不在なら5Cの全盤面返却へフォールバック。"""
    if exact.known_depth(position.queue) != KNOWN_HANDS:
        return None
    if base.status(position.board, position.queue, KNOWN_HANDS) != base.OK:
        return None
    engine = base.native._native
    if engine is not None and hasattr(engine, 'prefire_threat_maxima_py'):
        pairs = [position.queue[i:i+base.PAIR_SIZE] for i in
                 range(0, KNOWN_HANDS*base.PAIR_SIZE, base.PAIR_SIZE)]
        return tuple(Fire(*entry) for entry in engine.prefire_threat_maxima_py(
            list(position.board), pairs, base.GHOST_CHAIN_RULE_ENABLED))
    options = side_options(position, KNOWN_HANDS, side)
    return options or None


def _side(options: tuple, side: int, elapsed: float, table: TransitionTable) -> Threat:
    """最大送量、同量なら最少手数、なお同じなら既存の列挙順で選ぶ。"""
    attacks = options[side]
    if attacks is None:
        return Threat()
    fires = [p for p in attacks if p.chains]
    if not fires:
        return Threat(0., 0., 0., 0., 0., False, False, False)
    def sent(position: Fire | base.Position) -> int:
        return table.generated(position.score, elapsed + (position.consumed - 1) * SEC_PER_HAND)
    attack = max(fires, key=lambda p: (sent(p), -p.consumed))
    amount = sent(attack)
    horizon = remaining_hands(attack.chains, 0., 0.) + attack.consumed - 1
    responses = options[1 - side]
    beyond = horizon > KNOWN_HANDS
    if responses is None:
        return Threat(attack=float(amount), attack_hands=float(attack.consumed),
                      attack_missing=False, beyond_known_horizon=beyond)
    # 同じ攻撃の発火時点レートで5Cの得点換算を共用する。
    fire_time = elapsed + (attack.consumed - 1) * SEC_PER_HAND
    counter = max((table.generated(p.score, fire_time) for p in responses
                   if p.chains and p.consumed <= horizon), default=0)
    difference = max(0, amount - counter)
    return Threat(float(amount), float(attack.consumed), float(counter), float(difference),
                  difference / BOARD_COLS, False, False, beyond)


def threat_features(positions: tuple[base.Position, base.Position], elapsed: float,
                    enabled: bool = False) -> tuple[Threat, Threat] | None:
    """両側を独立に攻撃側として評価する。OFFでは探索も入力参照も行わない。"""
    if not enabled:
        return None
    validate_elapsed(elapsed)
    if len(positions) != 2:
        raise ValueError('両側の盤面とqueueが必要')
    if any(len(p.board) != CAPACITY or p.consumed or p.score for p in positions):
        raise ValueError('探索前の78セル盤面が必要')
    options = tuple(_options(p, side) for side, p in enumerate(positions))
    table = TransitionTable()
    return tuple(_side(options, side, elapsed, table) for side in range(2))


def encode(threats: tuple[Threat, Threat], source_side: int = 0) -> np.ndarray:
    """絶対量は0〜1へ、側は交換だけ、側の差のみ符号反転。欠測は維持。"""
    if source_side not in (0, 1):
        raise ValueError('source_sideは0または1')
    sides = []
    for threat in threats:
        values = threat.raw()
        for index in (0, 2, 3):
            values[index] /= CAPACITY + values[index]
        values[1] /= KNOWN_HANDS
        values[4] /= BOARD_ROWS + values[4]
        bins = [float(threat.rows >= bound) if np.isfinite(threat.rows) else np.nan
                for bound in ROW_BOUNDS]
        sides.append(np.r_[values, bins])
    own, opp = np.asarray(sides)[[source_side, 1-source_side]]
    return np.r_[own, opp, own - opp]
