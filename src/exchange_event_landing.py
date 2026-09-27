"""確定送り量を仮想着弾盤面へ反映する、ON経路専用の外部ラッパー。"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Any

import numpy as np

from src.board import Board, BOARD_COLS, BOARD_ROWS, DEATH_COL, DEATH_ROW
from src.chain import ChainSimulator
from src.exchange_event_evaluator import evaluate_exchange_event
from src.exchange_event_tracker import SIDE_LABELS
from src.exchange_virtual_board import land_pending_ojama_onto_board
from src.indicators_v2 import (SEC_PER_HAND, estimate_chain_anim_duration_sec,
                               near_future_fire_power, NEAR_FUTURE_KNOWN_HAND_SLOTS)
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.scoring import (OJAMA_MAX_DROP_PER_TURN, score_to_ojama, calculate_chain_score,
                         BASE_SCORE_PER_PUYO)

UNAVOIDABLE_DEATH_PROBABILITY = 0.02
LOGIT_EPSILON = 1e-6
PROJECTION_CACHE_SIZE = 2048
ANIMATION_CALIBRATION = "empirical_table_2026_08_14"
# 縦2向き×6列＋横2向き×5列。最初の1手の全配置を保持する応手探索幅。
RESPONSE_BEAM_WIDTH = 2 * BOARD_COLS + 2 * (BOARD_COLS - 1)
# 端数列の最悪配置を死の確定根拠にせず、最低でも丸1段の超過を要求する。
DEATH_OVERFLOW_ROWS = 1
PUYO_COLORS = (1, 2, 3, 4, 5)


def logit_mean(left: float, right: float) -> float:
    """同じ1P視点の確率を等重みのlogit平均で合成する。"""
    values = np.clip([left, right], LOGIT_EPSILON, 1 - LOGIT_EPSILON)
    mean = float(np.mean(np.log(values / (1 - values))))
    return 1 / (1 + math.exp(-mean))


def remaining_hands(chain_count: int, trigger: float, now: float, busy_sec: float = 0.0) -> int:
    """演出残時間を手数へ切り捨て、着地を起こす最後の1手を加える。"""
    duration = estimate_chain_anim_duration_sec(chain_count, ANIMATION_CALIBRATION)
    return math.floor(max(0.0, duration - max(0.0, now - trigger) - busy_sec) / SEC_PER_HAND) + 1


def minimum_cancel(board: Board, opponent: Board, incoming: int) -> int:
    """既存の着弾パターンで生存できる最大残量まで相殺する個数を返す。"""
    for remaining in range(min(incoming, OJAMA_MAX_DROP_PER_TURN), -1, -1):
        landed, _, _ = land_pending_ojama_onto_board(board, opponent, remaining)
        if not landed.is_dead():
            return incoming - remaining
    return incoming


@lru_cache(maxsize=PROJECTION_CACHE_SIZE)
def future_send(raw: bytes, shape: tuple, dtype: str, queue: tuple,
                hands: int, elapsed: float) -> float:
    """同じ確定盤面・NEXT・時間予算では近未来探索を一度だけ行う。"""
    board = Board()
    board._grid = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    # Kは既知NEXT2手を除く追加手数。K=-1/0も内部チェックポイントとして有効。
    level = hands - NEAR_FUTURE_KNOWN_HAND_SLOTS
    result = near_future_fire_power(board, queue[:2], queue[2:], elapsed_sec=elapsed,
                                    k_levels=(level,), resolve_before_death=True,
                                    beam_width=RESPONSE_BEAM_WIDTH)
    return result.values[level].raw


class ExchangeLandingProjection:
    """元のS3を保存して合成の累積を防ぎ、実着地以降の二重投下を防ぐ。"""

    def __init__(self) -> None:
        self.identity: tuple | None = None
        self.key: tuple | None = None
        self.drops = (0, 0)
        self.counts = [0, 0]
        self.last: dict | None = None
        self.latest: tuple = ()
        self.amount_key: tuple | None = None
        self.death: dict | None = None
        self.death_record: Any = None
        self.response_id: int | None = None
        self.board_changed = False
        self.rejected_boards: list[dict] = []
        self.evaluated_drops: tuple = ()
        self.simulator = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)

    def update(self, overlay: Any, result: Any, snapshot: Any, t_sec: float) -> None:
        """両側の通知とSTABLE履歴の更新後、確定送り量が変われば再評価する。"""
        tracker = overlay.tracker
        if self._refresh_death(overlay, t_sec):
            return
        record = tracker.current or self.death_record
        if record is None or overlay._m0 is None or not all(overlay._history):
            self._hold(tracker)
            return
        identity = (record.game_idx, record.exchange_id)
        dropped = (snapshot.total_dropped_to_p1, snapshot.total_dropped_to_p2)
        if identity != self.identity:
            self.identity, self.key, self.last = identity, None, None
            before = next((s for t, s in reversed(overlay._snapshots)
                           if t < record.trigger_sec), snapshot)
            self.drops = (before.total_dropped_to_p1, before.total_dropped_to_p2)
            self.counts = [0, 0]
            self.amount_key = None
        for idx, side in enumerate((result.p1, result.p2)):
            if side.chain_event is not None:
                self.counts[idx] = side.chain_event.chain_count
        base = next((v for v in reversed(record.values)
                     if v["source"] in ("S3_provisional", "S3")), None)
        if base is None:
            self._hold(tracker)
            return
        self._select_boards(overlay, t_sec)
        incoming = self._incoming(tracker, dropped, record)
        if not any(incoming):
            tracker.source, tracker.probability = base["source"], base["p1"]
            self.amount_key = None
            if self.board_changed or dropped == self.evaluated_drops:
                self.death, self.death_record, self.response_id = None, None, None
            self._hold(tracker)
            return
        amount_key = (tuple(incoming), base["t_sec"], tuple(c.chain_id for c in record.chains))
        self._project(overlay, snapshot, incoming, amount_key, base, record, t_sec)

    def _project(self, overlay: Any, snapshot: Any, incoming: list[int], amount_key: tuple,
                 base: dict, record: Any, t_sec: float) -> None:
        """物理量・確定盤面・残手数が変わった時だけ合成と保持の再判定を行う。"""
        tracker = overlay.tracker
        dropped = (snapshot.total_dropped_to_p1, snapshot.total_dropped_to_p2)
        reduced = self.amount_key is not None and any(a < b for a, b in zip(incoming, self.amount_key[0]))
        reassess = self.board_changed or (reduced and dropped == self.evaluated_drops)
        self.amount_key = amount_key
        self.evaluated_drops = dropped
        hands = tuple(self._hands(tracker, 1-i, t_sec) for i in range(2))
        key = (amount_key, hands, tuple((s.board._grid.tobytes(), s.queue.tobytes()) for s in self.latest))
        if key != self.key:
            self.last = self._evaluate(overlay, snapshot, self.latest, incoming, hands, base, t_sec)
            self.key = key
            if reassess and self.last["source"] != "unavoidable_death":
                self.death, self.death_record, self.response_id = None, None, None
            self._latch(tracker)
            value = self.last if self.death is None else dict(self.last,
                source="unavoidable_death", p1=self.death["p1"],
                dead_sides=self.death["dead_sides"], held_from_sec=self.death["t_sec"])
            record.values.append(value)
        tracker.source, tracker.probability = self.last["source"], self.last["p1"]
        self._hold(tracker)

    def _select_boards(self, overlay: Any, t_sec: float) -> None:
        """設置・消去で確定値が変われば再判定し、得点で不可能な色消失を除く。"""
        latest = list(h[-1] for h in overlay._history)
        self.board_changed = False
        self.rejected_boards = []
        if self.death is not None and self.latest:
            idx = SIDE_LABELS.index(self.death["dead_sides"][0])
            before, after = self.latest[idx].board, latest[idx].board
            changed = not np.array_equal(before._grid, after._grid)
            chain = overlay.tracker.latest_chain(SIDE_LABELS[idx])
            if changed and chain is not None:
                score = max(chain.formula_total or 0, chain.score_delta or 0)
                lost = sum(max(0, int(np.count_nonzero(before._grid == c))
                    - int(np.count_nonzero(after._grid == c))) for c in PUYO_COLORS)
                # 連鎖倍率は1以上。得点/10を超える色ぷよ消失は物理的に不可能。
                if lost > math.floor(score / BASE_SCORE_PER_PUYO):
                    self.rejected_boards.append(dict(side=SIDE_LABELS[idx], t_sec=t_sec,
                        lost_colors=lost, max_cleared=math.floor(score / BASE_SCORE_PER_PUYO)))
                    latest[idx] = self.latest[idx]
                    changed = False
            self.board_changed = changed
        self.latest = tuple(latest)

    def _latch(self, tracker: Any) -> None:
        """最初に成立した回避不能死と、その時点の受け側連鎖を保持する。"""
        if self.death is not None or self.last["source"] != "unavoidable_death":
            return
        self.death, self.death_record = self.last, tracker.current
        side = self.death["dead_sides"][0]
        chain = tracker.latest_chain(side)
        self.response_id = chain.chain_id if chain else None

    def _hold(self, tracker: Any) -> None:
        """終了通知や会計更新では固定を解除しない。"""
        if self.death is not None:
            tracker.source, tracker.probability = "unavoidable_death", self.death["p1"]

    def _refresh_death(self, overlay: Any, t_sec: float) -> bool:
        """受け側の新発火、試合境界、着地後の両側確定で保持を解除する。"""
        if self.death is None:
            return False
        tracker, record = overlay.tracker, self.death_record
        side = self.death["dead_sides"][0]
        chain = tracker.latest_chain(side)
        response = chain is not None and chain.chain_id != self.response_id
        boundary = tracker._game_idx != record.game_idx
        confirmed = tuple(h[-1].t_sec for h in overlay._history if h)
        ready = (len(confirmed) == 2 and tracker.ready_for_static(t_sec, confirmed)
                 and all(c.score_ready_sec is not None for c in record.chains))
        cutoff = max((c.score_ready_sec or record.trigger_sec for c in record.chains),
                     default=record.trigger_sec)
        landed = ready and any(r["side"] == side and r["t_sec"] is not None
            and r["t_sec"] >= self.death["t_sec"]
            and min(confirmed) > cutoff and min(confirmed) >= r["t_sec"]
            for r in record.landings)
        if response or boundary or landed:
            self.death, self.death_record, self.response_id = None, None, None
            self.key, self.amount_key = None, None
        return bool(landed)

    def _incoming(self, tracker: Any, dropped: tuple, record: Any = None) -> list[int]:
        """段ごとの累積得点を既存換算し、相殺と既着地分を控除する。"""
        record = record or tracker.current
        totals = [sum(c.score_delta if c.score_ready_sec is not None else (c.formula_total or 0)
                      for c in record.chains if c.side == label) for label in SIDE_LABELS]
        sent = [math.floor(score_to_ojama(s, elapsed_sec=tracker._score_elapsed).ojama_count)
                for s in totals]
        return [max(0, sent[1-i] - sent[i] - max(0, dropped[i] - self.drops[i]))
                for i in range(2)]

    def _hands(self, tracker: Any, attacker: int, t_sec: float) -> int:
        """自側連鎖中の操作不能時間を差し引き、着地前の最後の1手を加える。"""
        chain = tracker.latest_chain(SIDE_LABELS[attacker])
        if chain is None or chain.end_signal_sec is not None:
            return 1
        receiver = tracker.latest_chain(SIDE_LABELS[1-attacker])
        busy = 0.0
        if receiver is not None and receiver.end_signal_sec is None:
            duration = estimate_chain_anim_duration_sec(self.counts[1-attacker], ANIMATION_CALIBRATION)
            busy = max(0.0, duration - max(0.0, t_sec-receiver.trigger_sec))
        return remaining_hands(self.counts[attacker], chain.trigger_sec, t_sec, busy)

    def _evaluate(self, overlay: Any, snapshot: Any, latest: tuple, incoming: list,
                  hands: tuple, base: dict, t_sec: float) -> dict:
        """仮想受け盤面と相手の現在確定盤面をG_feへ渡し、応手不足なら固定する。"""
        boards = tuple(s.board for s in latest)
        responses, credit = self._receivers(overlay.tracker, latest, incoming)
        virtual = tuple(land_pending_ojama_onto_board(b, boards[1-i], incoming[i])[0]
                        for i, b in enumerate(boards))
        elapsed = t_sec - overlay._start
        m0 = overlay._m0(np.stack([b._grid for b in virtual]), np.stack([s.queue for s in latest]))
        event = overlay._build_static(virtual, snapshot, elapsed, m0)
        gfe = evaluate_exchange_event(event, overlay.tracker.models)
        probability = logit_mean(base["p1"], gfe)
        dead, required, available = [], [0, 0], [None, None]
        margins, evidence, optimistic = [None, None], [False, False], [None, None]
        for i, landed in enumerate(virtual):
            if incoming[i] <= 0 or not landed.is_dead():
                continue
            required[i] = minimum_cancel(boards[i], boards[1-i], incoming[i])
            grid = responses[i]._grid
            available[i] = future_send(grid.tobytes(), grid.shape, grid.dtype.str,
                tuple(int(v) for v in latest[i].queue), hands[i], overlay.tracker._score_elapsed)
            available[i] += credit[i]
            margins[i] = (boards[i].height_of(DEATH_COL)
                + min(incoming[i], OJAMA_MAX_DROP_PER_TURN) // BOARD_COLS - (BOARD_ROWS-DEATH_ROW))
            held = self.death is not None and SIDE_LABELS[i] in self.death["dead_sides"]
            evidence[i] = self._verified_attack(overlay.tracker, 1-i)
            candidate = available[i] < required[i] and (held or
                (evidence[i] and margins[i] >= DEATH_OVERFLOW_ROWS))
            if candidate and not held:
                optimistic[i] = self._optimistic_response(responses[i], latest[i].queue,
                    hands[i], overlay.tracker._score_elapsed) + credit[i]
                candidate = optimistic[i] < required[i]
            if candidate:
                dead.append(SIDE_LABELS[i])
        if len(dead) == 1:
            probability = (UNAVOIDABLE_DEATH_PROBABILITY if dead[0] == "1P"
                           else 1 - UNAVOIDABLE_DEATH_PROBABILITY)
        return dict(source="unavoidable_death" if len(dead) == 1 else "S3_landing",
                    t_sec=t_sec, p1=probability, base_p1=base["p1"], gfe_p1=gfe,
                    incoming=incoming, hands=hands, required_cancel=required,
                    near_future_send=available, resolving_send=credit, dead_sides=dead,
                    overflow_rows=margins, verified_attack=evidence,
                    rejected_boards=self.rejected_boards, optimistic_send=optimistic)

    def _optimistic_response(self, board: Board, queue: np.ndarray, hands: int, elapsed: float) -> float:
        """初回断定は既存指標の候補集合でも拒否できた場合だけに限定する。"""
        # 既存指標は既知NEXT2枠+Kを探索する。n手の探索下界だけで不可能とは証明できない。
        # 追加2枠は実際に置ける手数へ加算せず、未検知設置・枝刈りへの楽観的な拒否条件に使う。
        grid = board._grid
        return future_send(grid.tobytes(), grid.shape, grid.dtype.str,
            tuple(int(v) for v in queue), hands + NEAR_FUTURE_KNOWN_HAND_SLOTS, elapsed)

    def _verified_attack(self, tracker: Any, attacker: int) -> bool:
        """累積表示得点の差だけでは、連鎖への帰属・相殺の欠落を確定できない。"""
        record = tracker.current or self.death_record
        chains = [c for c in record.chains if c.side == SIDE_LABELS[attacker]]
        return bool(chains) and all(c.formula_total is not None or
            c.score_ready_reason == "score_finalize" for c in chains)

    def _receivers(self, tracker: Any, latest: tuple, incoming: list) -> tuple:
        """応手探索では既発火群を解消し、未観測の自側火力を加える。"""
        boards, credit = [], []
        for i, side in enumerate(latest):
            if incoming[i] <= 0:
                boards.append(side.board)
                credit.append(0)
                continue
            result = self.simulator.simulate(side.board)
            boards.append(result.final_board)
            chain = tracker.latest_chain(SIDE_LABELS[i])
            observed = (chain.formula_total or 0) if chain and chain.end_signal_sec is None else 0
            remaining = max(0, calculate_chain_score(result).total_score - observed)
            credit.append(math.floor(score_to_ojama(remaining, elapsed_sec=tracker._score_elapsed).ojama_count))
        return tuple(boards), credit
