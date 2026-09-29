"""確定送り量を仮想着弾盤面へ反映する、ON経路専用の外部ラッパー。"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Any

import numpy as np

from src.board import Board, BOARD_COLS, BOARD_ROWS, DEATH_COL, DEATH_ROW
from src.chain import ChainSimulator
from src.board_state_machine import BoardState
from src.exchange_event_evaluator import evaluate_exchange_event
from src.exchange_event_tracker import SIDE_LABELS
from src.exchange_virtual_board import land_pending_ojama_onto_board
from src.indicators_v2 import (SEC_PER_HAND, estimate_chain_anim_duration_sec,
                               near_future_fire_power, NEAR_FUTURE_KNOWN_HAND_SLOTS)
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.ojama_accounting import cancel_own_pending_then_send_surplus
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

    def __init__(self, counter_response: bool = False, counter_probability_model: Any = None,
                 hands_spec: bool = False, multi_landing_death: bool = False,
                 landing_state_safety: bool = False, pending_ledger: bool = False,
                 color_score_safety: bool = False, completion_recovery: bool = False,
                 death_pending_ledger: bool = False, hidden_row_death: bool = False,
                 single_death_proof_guard: bool = False,
                 single_death_proof_negative_only: bool = False) -> None:
        self.counter_response = counter_response
        self.death_pending_ledger = death_pending_ledger
        self.death_only_inputs = death_pending_ledger or hidden_row_death
        self.multi_landing_death = multi_landing_death
        self.single_death_proof_guard = single_death_proof_guard
        # D5b: D5併用時のみ有効。生存枝・相殺可能の証明があるときだけ取り消す。
        self.single_death_proof_negative_only = single_death_proof_guard and single_death_proof_negative_only
        from src.exchange_landing_safety import LandingStateSafety
        self.safety = LandingStateSafety(landing_state_safety or pending_ledger,
            landing_state_safety or color_score_safety or single_death_proof_guard,
            landing_state_safety or completion_recovery) if any((landing_state_safety,
                pending_ledger, color_score_safety, completion_recovery,
                death_pending_ledger, hidden_row_death, single_death_proof_guard)) else None
        self.multi_landing_cache: dict[tuple, dict] = {}
        self.counter_probability_model = counter_probability_model
        from src.exchange_event_hands import LandingHandsObservation
        self.hands_observation = LandingHandsObservation() if hands_spec else None
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
        self.busy = (False, False)
        self.simulator = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)

    def update(self, overlay: Any, result: Any, snapshot: Any, t_sec: float) -> None:
        """両側の通知とSTABLE履歴の更新後、確定送り量が変われば再評価する。"""
        tracker = overlay.tracker
        self._observe_frame(overlay, result, snapshot)
        if self.safety is not None:
            self.safety.observe(overlay, result, snapshot, t_sec)
        if self.hands_observation is not None:
            self.hands_observation.observe_chains(tracker, self.counts, t_sec)
        if self._refresh_death(overlay, t_sec):
            return
        record = tracker.current or self.death_record
        if record is None or overlay._m0 is None or not all(overlay._history):
            self._hold(tracker)
            return
        dropped = (snapshot.total_dropped_to_p1, snapshot.total_dropped_to_p2)
        base = next((v for v in reversed(record.values)
                     if v["source"] in ("S3_provisional", "S3")), None)
        if base is None:
            self._hold(tracker)
            return
        self._select_boards(overlay, t_sec)
        incoming = self._incoming(tracker, dropped, record)
        if not any(incoming) and not (self.death_pending_ledger and any(self.safety.ledger.pending)):
            tracker.source, tracker.probability = base["source"], base["p1"]
            self.amount_key = None
            if self.board_changed or dropped == self.evaluated_drops:
                self.death, self.death_record, self.response_id = None, None, None
            self._hold(tracker)
            return
        amount_key = (tuple(incoming), base["t_sec"], tuple(c.chain_id for c in record.chains))
        self._project(overlay, snapshot, incoming, amount_key, base, record, t_sec)

    def _observe_frame(self, overlay: Any, result: Any, snapshot: Any) -> None:
        """当該フレームの段数と物理状態を揃えてから死保持を再判定する。"""
        self.busy = tuple(getattr(s, "state", None) in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE)
                          for s in (result.p1, result.p2))
        record = overlay.tracker.current or self.death_record
        if record is not None and overlay._m0 is not None and all(overlay._history):
            identity = (record.game_idx, record.exchange_id)
            if identity != self.identity:
                self.identity, self.key, self.last = identity, None, None
                before = next((s for t, s in reversed(overlay._snapshots)
                               if t < record.trigger_sec), snapshot)
                self.drops = (before.total_dropped_to_p1, before.total_dropped_to_p2)
                self.counts, self.amount_key = [0, 0], None
        for idx, side in enumerate((result.p1, result.p2)):
            if side.chain_event is not None:
                self.counts[idx] = side.chain_event.chain_count

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
        active = tuple(self._chaining(tracker, i) for i in range(2))
        boards = tuple((s.board._grid.tobytes(), s.queue.tobytes()) for s in self.latest)
        evidence = tuple(self._verified_attack(tracker, i) for i in range(2))
        completion = tuple(self._completion_board(tracker, i) is not None for i in range(2))
        key = (amount_key, hands, active, boards, evidence, completion)
        if getattr(self, 'post_counter_bound', None) is not None:
            key += (self.post_counter_bound.revision,)
        if self.safety is not None and self.safety.guard_enabled:
            key += (self.safety.signature(self, tracker, t_sec),)
            reassess = reassess or self.key != key
        if self.death_only_inputs:
            key += (tuple(self.safety.ledger.pending), getattr(getattr(self, 'hidden_death', None), 'revision', 0))
            if getattr(self, 'prefire', None) is not None:
                key += (self.prefire.revision,)
            reassess = reassess or self.key != key
        reassess = reassess or (self.key is not None
            and (self.key[1] != hands or self.key[3] != boards or self.key[5] != completion))
        if key != self.key:
            self.last = self._evaluate(overlay, snapshot, self.latest, incoming, hands, base, t_sec)
            if getattr(overlay, '_midchain', None) is not None:
                self.last['midchain_prediction'] = overlay._midchain.provenance(overlay._game, record.chains)
            if self.hands_observation is not None:
                self.last["hands_spec"] = [self._spec_budget(tracker, 1-i, t_sec) for i in range(2)]
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
        idx = SIDE_LABELS.index(side)
        recheck = self._chaining(tracker, idx) and (
            (self._completion_board(tracker, idx) is None and not self._hidden_completion(chain))
            or side not in self.death.get("completion_sides", []))
        if response or boundary or landed or recheck:
            self.death, self.death_record, self.response_id = None, None, None
            self.key, self.amount_key = None, None
            if tracker.source == "unavoidable_death":
                tracker.probability = tracker._static_probability
                tracker.source = "G_fe" if tracker.probability is not None else "waiting_confirmed"
        return bool(landed)

    def _hidden_completion(self, chain: Any) -> bool:
        """死亡保持の再検査だけで隠し段上限の有効性を参照する。"""
        from src.exchange_death_inputs import maximum_response
        return maximum_response(self, chain) is not None

    def _incoming(self, tracker: Any, dropped: tuple, record: Any = None) -> list[int]:
        """段ごとの累積得点を既存換算し、相殺と既着地分を控除する。"""
        if self.safety is not None and self.safety.ledger_enabled:
            return self.safety.ledger.pending
        record = record or tracker.current
        totals = [sum(c.provisional_score
                      for c in record.chains if c.side == label) for label in SIDE_LABELS]
        sent = [math.floor(score_to_ojama(s, elapsed_sec=tracker._score_elapsed).ojama_count)
                for s in totals]
        if getattr(self, 'prefire', None) is not None:
            sent = self.prefire.sends(record.chains, tracker._score_elapsed) or sent
        return [max(0, sent[1-i] - sent[i] - max(0, dropped[i] - self.drops[i]))
                for i in range(2)]

    def _hands(self, tracker: Any, attacker: int, t_sec: float) -> int:
        """自側連鎖中の操作不能時間を差し引き、着地前の最後の1手を加える。"""
        if self.hands_observation is not None:
            return self._spec_budget(tracker, attacker, t_sec)["hands"]
        chain = tracker.latest_chain(SIDE_LABELS[attacker])
        if chain is None or (chain.end_signal_sec is not None and chain.end_confirmed is not False):
            return 1
        receiver = tracker.latest_chain(SIDE_LABELS[1-attacker])
        busy = 0.0
        if receiver is not None and (receiver.end_signal_sec is None or receiver.end_confirmed is False):
            duration = estimate_chain_anim_duration_sec(
                max(self.counts[1-attacker], receiver.predicted_chain_count or 0), ANIMATION_CALIBRATION)
            busy = max(0.0, duration - max(0.0, t_sec-receiver.trigger_sec))
        return remaining_hands(max(self.counts[attacker], chain.predicted_chain_count or 0),
                               chain.trigger_sec, t_sec, busy)

    def _spec_budget(self, tracker: Any, attacker: int, t_sec: float) -> dict:
        """NEXT確定を最優先し、残演出を受け側の観測中央値で手数に変換する。"""
        return self.hands_observation.estimate(tracker.latest_chain(SIDE_LABELS[attacker]),
                                             attacker, t_sec, self.counts[attacker])

    def _chaining(self, tracker: Any, idx: int) -> bool:
        """物理連鎖中または終了未確認なら、完走後盤面での判定を要求する。"""
        chain = tracker.latest_chain(SIDE_LABELS[idx])
        return self.busy[idx] or (chain is not None
            and (chain.end_signal_sec is None or chain.end_confirmed is False))

    def _completion_board(self, tracker: Any, idx: int) -> Board | None:
        """観測と矛盾しない予測だけから完走後盤面を復元する。"""
        chain = tracker.latest_chain(SIDE_LABELS[idx])
        if chain is None or chain.predicted_final_board is None:
            return None
        observed = max(chain.formula_total or 0, (chain.score_delta or 0) - chain.drop_bonus_score)
        if (observed > (chain.predicted_final_score or 0)
                or self.counts[idx] > (chain.predicted_chain_count or 0)):
            return None
        board = Board()
        board._grid = np.array(chain.predicted_final_board, dtype=board._grid.dtype)
        return board

    def _death_boards(self, tracker: Any, boards: tuple, responses: tuple,
                      credit: list) -> tuple:
        """連鎖中は完走盤面を使い、既に純受け量へ算入した予測火力を重ねない。"""
        targets, replies, certain = list(boards), list(responses), [True, True]
        for idx in range(2):
            if not self._chaining(tracker, idx):
                continue
            completion = self._completion_board(tracker, idx)
            certain[idx] = completion is not None
            if completion is not None:
                targets[idx] = replies[idx] = completion
                credit[idx] = 0
        return tuple(targets), tuple(replies), certain

    def _landing_gfe(self, overlay: Any, snapshot: Any, latest: tuple,
                     incoming: list, t_sec: float) -> float:
        """E12の確率合成に使う仮想着弾評価を維持する。"""
        boards = tuple(s.board for s in latest)
        if getattr(self, 'prefire', None) is not None:
            boards = tuple(self._completion_board(overlay.tracker, i) or b
                if self.prefire.active(overlay.tracker.latest_chain(SIDE_LABELS[i])) else b
                for i, b in enumerate(boards))
        virtual = tuple(land_pending_ojama_onto_board(b, boards[1-i], incoming[i])[0]
                        for i, b in enumerate(boards))
        elapsed = t_sec - overlay._start
        m0 = overlay._m0(np.stack([b._grid for b in virtual]), np.stack([s.queue for s in latest]))
        event = overlay._build_static(virtual, snapshot, elapsed, m0)
        return evaluate_exchange_event(event, overlay.tracker.models)

    def _evaluate(self, overlay: Any, snapshot: Any, latest: tuple, incoming: list,
                  hands: tuple, base: dict, t_sec: float) -> dict:
        """既定はE22そのまま。明示ONだけ複数着弾の保守的証明を追加する。"""
        from src.exchange_death_inputs import death_inputs
        context = death_inputs(self, overlay, latest, incoming) if self.death_only_inputs else None
        value = self._evaluate_single(overlay, snapshot, latest, incoming, hands, base, t_sec, context)
        if context is not None:
            value['death_incoming'] = context['incoming']
            value['hidden_death_scores'] = [v['score'] if v else None for v in context['hidden']]
        if self.safety is not None:
            value['state_safety'] = self.safety.signature(self, overlay.tracker, t_sec)
            value['pending_ledger'] = self.safety.ledger.pending
        if self.multi_landing_death:
            from src.exchange_event_multilanding import evaluate_multilanding
            value = evaluate_multilanding(self, overlay, latest, incoming, hands, t_sec, value, context)
        if context is not None and not any(incoming) and value['source'] != 'unavoidable_death':
            value.update(source=base['source'], p1=base['p1'])
        return value

    def _evaluate_single(self, overlay: Any, snapshot: Any, latest: tuple, incoming: list,
                         hands: tuple, base: dict, t_sec: float, context: dict | None = None) -> dict:
        """完走後の受け盤面で窒息と応手不足を確認し、回避不能なら固定する。"""
        boards = tuple(s.board for s in latest)
        responses, credit = self._receivers(overlay.tracker, latest, incoming)
        gfe, counter = self._probability_inputs(overlay, snapshot, latest, incoming, hands, t_sec)
        probability = logit_mean(base["p1"], gfe)
        if getattr(overlay.tracker, 'hidden_row_belief', None) is not None:
            from src.exchange_hidden_row_probability import weighted_landing
            weighted = weighted_landing(self, overlay, snapshot, latest, hands, base, t_sec)
            if weighted is not None:
                probability, gfe, counter = weighted
        boards, responses, certain = self._death_boards(overlay.tracker, boards, responses, credit)
        probability_incoming = incoming
        if context is not None:
            incoming, boards, responses, certain, credit = (context[k] for k in
                ('incoming', 'boards', 'replies', 'certain', 'credit'))
        metrics = self._single_death_metrics(overlay, latest, incoming, hands, t_sec,
                                              boards, responses, certain, credit, context)
        dead = metrics['dead_sides']
        if len(dead) == 1:
            probability = (UNAVOIDABLE_DEATH_PROBABILITY if dead[0] == "1P"
                           else 1 - UNAVOIDABLE_DEATH_PROBABILITY)
        return dict(source="unavoidable_death" if len(dead) == 1 else "S3_landing",
                    t_sec=t_sec, p1=probability, base_p1=base["p1"], gfe_p1=gfe,
                    incoming=probability_incoming, hands=hands, required_cancel=metrics['required_cancel'],
                    near_future_send=metrics['near_future_send'], resolving_send=credit, dead_sides=dead,
                    overflow_rows=metrics['overflow_rows'], verified_attack=metrics['verified_attack'],
                    rejected_boards=self.rejected_boards, optimistic_send=metrics['optimistic_send'],
                    completion_certain=certain, completion_sides=[SIDE_LABELS[i] for i in range(2)
                        if certain[i] and self._chaining(overlay.tracker, i)], **counter,
                    **({'single_death_proof': metrics['proofs']} if self.single_death_proof_guard else {}))

    def _single_death_metrics(self, overlay: Any, latest: tuple, incoming: list, hands: tuple,
                              t_sec: float, boards: tuple, responses: tuple, certain: list,
                              credit: list, context: dict | None) -> dict:
        """死亡専用受け量で単発着弾を判定し、確率の計算には触れない。"""
        landed_boards = tuple(land_pending_ojama_onto_board(b, boards[1-i], incoming[i])[0]
                              for i, b in enumerate(boards))
        dead, required, available = [], [0, 0], [None, None]
        margins, evidence, optimistic = [None, None], [False, False], [None, None]
        proofs = [None, None]
        for i, landed in enumerate(landed_boards):
            if (incoming[i] <= 0 or not landed.is_dead() or not certain[i]
                    or (context is not None and context['hidden'][i] is not None)
                    or not self._known_budget(overlay.tracker, 1-i, t_sec)):
                continue
            held = self.death is not None and SIDE_LABELS[i] in self.death["dead_sides"]
            evidence[i] = self._verified_attack(overlay.tracker, 1-i)
            if not evidence[i] and not held:
                continue  # 予測火力は確率へ反映するが、死を証明する探索には使わない。
            required[i] = minimum_cancel(boards[i], boards[1-i], incoming[i])
            grid = responses[i]._grid
            available[i] = future_send(grid.tobytes(), grid.shape, grid.dtype.str,
                tuple(int(v) for v in latest[i].queue), hands[i], overlay.tracker._score_elapsed)
            available[i] += credit[i]
            margins[i] = (boards[i].height_of(DEATH_COL)
                + min(incoming[i], OJAMA_MAX_DROP_PER_TURN) // BOARD_COLS - (BOARD_ROWS-DEATH_ROW))
            candidate = available[i] < required[i] and (held or
                (evidence[i] and margins[i] >= DEATH_OVERFLOW_ROWS))
            if candidate and not held:
                optimistic[i] = self._optimistic_response(responses[i], latest[i].queue,
                    hands[i], overlay.tracker._score_elapsed) + credit[i]
                candidate = optimistic[i] < required[i]
            if candidate and self.single_death_proof_guard:
                from src.exchange_single_death_proof import prove_single_candidate
                proofs[i] = prove_single_candidate(self, overlay, latest[i], responses[i],
                    incoming[i], hands[i], credit[i], i, t_sec, self.single_death_proof_negative_only)
                candidate = proofs[i]['dead']
            if candidate:
                dead.append(SIDE_LABELS[i])
        return dict(required_cancel=required, near_future_send=available, dead_sides=dead,
                    overflow_rows=margins, verified_attack=evidence, optimistic_send=optimistic, proofs=proofs)

    def _probability_inputs(self, overlay: Any, snapshot: Any, latest: tuple,
                            incoming: list, hands: tuple, stamp: float) -> tuple:
        """死亡専用台帳・隠し段候補を受け取らず、従来の確率入力だけを計算する。"""
        gfe = self._landing_gfe(overlay, snapshot, latest, incoming, stamp)
        counter = self._counter_projection(overlay, snapshot, latest, incoming, hands, gfe, stamp)
        gfe = counter.get('gfe_response_p1', gfe) if counter.get('response_selected') else gfe
        return counter.get('gfe_weighted_p1', gfe), counter

    def _counter_projection(self, overlay: Any, snapshot: Any, latest: tuple,
                            incoming: list, hands: tuple, gfe: float, t_sec: float) -> dict:
        """確定履歴から探索した応手を予測層だけへ反映し、二つの仮定を保存する。"""
        if not self.counter_response and self.counter_probability_model is None:
            return {}
        replies, credit = self._receivers(overlay.tracker, latest, incoming)
        net, sends, surplus = list(incoming), [None, None], [0, 0]
        for idx in range(2):
            if incoming[idx] <= 0:
                continue
            chain = overlay.tracker.latest_chain(SIDE_LABELS[idx])
            stale = chain is not None and chain.end_signal_sec is not None and latest[idx].t_sec <= chain.end_signal_sec
            board = replies[idx]
            if self._chaining(overlay.tracker, idx) or stale:
                board = self._completion_board(overlay.tracker, idx)
                if board is None:
                    continue  # 未解消の既発火火力を、追加の応手として二重に数えない。
                credit[idx] = 0
            grid = board._grid
            sends[idx] = future_send(grid.tobytes(), grid.shape, grid.dtype.str,
                tuple(int(v) for v in latest[idx].queue), hands[idx], overlay.tracker._score_elapsed)
            sends[idx] = max(0, math.floor(sends[idx] + credit[idx]))
            net[idx], net[1-idx] = cancel_own_pending_then_send_surplus(sends[idx], net[idx], net[1-idx])
            surplus[idx] = max(0, sends[idx] - incoming[idx])
        response_gfe = (self._landing_gfe(overlay, snapshot, latest, net, t_sec)
                        if net != incoming else gfe)
        selected = any(amount > 0 for amount in incoming) and all(
            amount <= 0 or (sends[idx] is not None and sends[idx] >= amount)
            for idx, amount in enumerate(incoming))
        value = dict(gfe_no_response_p1=gfe, gfe_response_p1=response_gfe,
            response_selected=selected, response_layer="prediction", response_send=sends,
            response_incoming=net, response_surplus=surplus,
            response_board_sec=[s.t_sec for s in latest])
        if self.counter_probability_model is not None:
            value.update(self._counter_probability(overlay, latest, incoming, hands, value, t_sec))
        return value

    def _known_budget(self, tracker: Any, attacker: int, t_sec: float) -> bool:
        """未観測時の1手は下限なので、それだけで回避不能とは断定しない。"""
        return self.hands_observation is None or not self._spec_budget(
            tracker, attacker, t_sec)["reason"].startswith("missing_")

    def _counter_probability(self, overlay: Any, latest: tuple, incoming: list, hands: tuple,
                             value: dict, t_sec: float) -> dict:
        """学習対象の未発火側だけ確率化し、識別対象外を確定応手として補完しない。"""
        from src.landing_counter_probability import response_features
        probabilities, features, reasons = [None, None], [None, None], ["no_incoming", "no_incoming"]
        weight = 0.
        for idx, amount in enumerate(incoming):
            if amount <= 0:
                continue
            probabilities[idx] = 0.
            reasons[idx] = "below_incoming"
            if not value["response_selected"]:
                continue
            if self._chaining(overlay.tracker, idx):
                reasons[idx] = "active_receiver_outside_training_support"
                continue
            side = latest[idx]
            x = response_features(side.board._grid, side.queue, value["response_send"][idx], amount,
                                  hands[idx], t_sec-side.t_sec, False)
            weight = float(self.counter_probability_model.predict(x))
            if not np.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError("応手確率が0〜1の範囲外")
            probabilities[idx], features[idx], reasons[idx] = weight, x.tolist(), "model"
        mixed = weight*value["gfe_response_p1"]+(1-weight)*value["gfe_no_response_p1"]
        return dict(gfe_weighted_p1=mixed, counter_probability=probabilities,
                    counter_probability_features=features, counter_probability_reasons=reasons)

    def _optimistic_response(self, board: Board, queue: np.ndarray, hands: int, elapsed: float) -> float:
        """初回断定は既存指標の候補集合でも拒否できた場合だけに限定する。"""
        # 既存指標は既知NEXT2枠+Kを探索する。n手の探索下界だけで不可能とは証明できない。
        # 追加2枠は実際に置ける手数へ加算せず、未検知設置・枝刈りへの楽観的な拒否条件に使う。
        grid = board._grid
        return future_send(grid.tobytes(), grid.shape, grid.dtype.str,
            tuple(int(v) for v in queue), hands + NEAR_FUTURE_KNOWN_HAND_SLOTS, elapsed)

    def _verified_attack(self, tracker: Any, attacker: int) -> bool:
        """帰属確認済みの実測得点だけで裏付けられる攻撃量に死の断定を限る。"""
        if self.safety is not None:
            return self.safety.ledger.verified(1-attacker)
        record = tracker.current or self.death_record
        chains = [c for c in record.chains if c.side == SIDE_LABELS[attacker]]
        return bool(chains) and all(
            (c.formula_total is not None or c.score_ready_reason == "score_finalize")
            and c.provisional_score <= max(c.formula_total or 0, c.score_delta or 0)
            for c in chains)

    def _receivers(self, tracker: Any, latest: tuple, incoming: list) -> tuple:
        """応手探索では既発火群を解消し、未観測の自側火力を加える。"""
        boards, credit = [], []
        for i, side in enumerate(latest):
            completion = self._completion_board(tracker, i) if self._chaining(tracker, i) else None
            if completion is not None:
                boards.append(completion)
                credit.append(0)
                continue
            if incoming[i] <= 0:
                boards.append(side.board)
                credit.append(0)
                continue
            try:
                result = self.simulator.simulate(side.board)
            except (ValueError, TypeError, FloatingPointError):
                if not self._chaining(tracker, i):
                    raise
                boards.append(side.board)
                credit.append(0)
                continue
            boards.append(result.final_board)
            chain = tracker.latest_chain(SIDE_LABELS[i])
            observed = (chain.provisional_score if chain and
                        (chain.end_signal_sec is None or chain.end_confirmed is False) else 0)
            remaining = max(0, calculate_chain_score(result).total_score - observed)
            credit.append(math.floor(score_to_ojama(remaining, elapsed_sec=tracker._score_elapsed).ojama_count))
        return tuple(boards), credit
