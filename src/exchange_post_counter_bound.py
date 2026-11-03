"""E35を死亡専用入力と予測層へ接続し、証明できない場合は従来探索へ戻す。"""
from __future__ import annotations

from time import perf_counter
from typing import Any

from src.board import Board
from src.match_color_evidence import MatchColorEvidence
from src.post_counter_death_bound import prove_post_counter
from src.scoring import compute_effective_rate

CACHE_LIMIT = 2048


class PostCounterDeathBound:
    """盤面確率は変更せず、全候補の死亡証明と所要時間だけを保存する。"""

    def __init__(self, early_exit: bool = False, *, input_integrity: bool = False,
                 input_integrity_v2: bool = False, current_puyo_integrity: bool = False,
                 current_puyo_integrity_v2c: bool = False, completion_prefix_integrity: bool = False,
                 completion_step_integrity: bool = False, completion_recent_integrity: bool = False,
                 strict_integrity: bool = False, cap_completed: bool = False,
                 retain_audit: bool = False) -> None:
        # 既定OFF (2026-09-30): 全候補が死ぬ場合だけ「死亡」と判定するので、最初に死なない候補が
        # 見つかった時点で判定は確定する。ON では以降の候補の証明を省く (判定は同一、監査の proofs だけ短くなる)。
        self.early_exit = early_exit
        self.input_integrity = input_integrity
        self.input_integrity_v2 = input_integrity_v2
        self.current_puyo_integrity = current_puyo_integrity
        self.current_puyo_integrity_v2c = current_puyo_integrity_v2c
        self.completion_prefix_integrity = completion_prefix_integrity
        self.completion_step_integrity = completion_step_integrity
        self.completion_recent_integrity = completion_recent_integrity
        self.strict_integrity = strict_integrity
        self.cap_completed = cap_completed
        self.step_observations: dict = {}
        self.formula_observations: dict = {}
        self.base_predictions: dict = {}
        self.states = (None, None)
        self.current_observation_key: tuple = ()
        self.next_motion = (False, False)
        self.colors = [MatchColorEvidence(), MatchColorEvidence()]
        self.cache: dict = {}
        self.board_cache: dict = {}
        self.audit: list[dict] = []
        self.retain_audit = retain_audit
        self.game: int | None = None
        self.queues: tuple = ((), ())
        self.revision = 0

    def observe(self, result: Any, game: int) -> None:
        """試合境界で色情報を初期化する。未来の観測は参照しない。"""
        motion = tuple(bool(getattr(s, 'next_slide_motion', False)) for s in (result.p1, result.p2))
        states = tuple(getattr(getattr(s, 'state', None), 'name', None) for s in (result.p1, result.p2))
        if ((self.input_integrity or self.input_integrity_v2 or self.current_puyo_integrity_v2c)
                and motion != self.next_motion) or (self.current_puyo_integrity_v2c and states != self.states):
            self.revision += 1
        self.next_motion = motion
        self.states = states
        if self.game != game:
            if not self.retain_audit:
                self.audit.clear()
            self.game = game
            self.colors = [MatchColorEvidence(), MatchColorEvidence()]
            self.cache.clear()
            self.board_cache.clear()
            self.formula_observations.clear()
            self.base_predictions.clear()
            self.step_observations.clear()
        for evidence, side in zip(self.colors, (result.p1, result.p2)):
            evidence.observe(side)
        queues = tuple(tuple(int(c) for c in (*(getattr(s, 'next_pair', None) or (0, 0)),
                                              *(getattr(s, 'dnext_pair', None) or (0, 0))))
                       for s in (result.p1, result.p2))
        if queues != self.queues:
            self.queues = queues
            self.revision += 1

    def prove(self, projection: Any, overlay: Any, idx: int, latest: tuple,
              incoming: int, hands: int, context: dict, stamp: float) -> dict:
        """複数候補は全て判定する。単一予測の盤面欠測は補完しない。"""
        if self.input_integrity or self.input_integrity_v2:
            reason = self.input_blocker(projection, overlay.tracker, idx, stamp, context)
            if reason is not None:
                return dict(dead=False, reason=reason)
            # 相手が終了後の次ツモを操作した実観測に限り、最後の設置一手を使う。
            from src.exchange_event_hands import FINAL_PLACEMENT_HANDS
            hands = FINAL_PLACEMENT_HANDS
        chain = overlay.tracker.latest_chain(f'{idx+1}P')
        hidden = context['hidden'][idx]
        if chain is None or not context['certain'][idx] or context['credit'][idx]:
            return dict(dead=False, reason='missing_post_counter_completion')
        if not projection._known_budget(overlay.tracker, 1-idx, stamp) or not context['verified'][idx]:
            return dict(dead=False, reason='unverified_attack_or_budget')
        if projection._chaining(overlay.tracker, idx) and hidden is None and chain.predicted_final_board is None:
            return dict(dead=False, reason='missing_majority_board')
        boards = ([Board.from_list(v['board']) for v in hidden['options']] if hidden
                  else [context['replies'][idx]])
        palette = self.colors[idx].active()
        queue, evidence = self.current_queue(overlay, chain, idx)
        if (self.completion_prefix_integrity or self.completion_step_integrity or self.completion_recent_integrity) and evidence.get('unknown_current_pair_omitted_self_chain'):
            evidence['completion_prefix_check'] = self.prefix_check(projection, overlay, chain, hidden, stamp)
            if not evidence['completion_prefix_check']['accepted']:
                reason = ('post_counter_completion_recent_blocked' if self.completion_recent_integrity else
                          'post_counter_completion_step_blocked' if self.completion_step_integrity else 'post_counter_completion_prefix_blocked')
                value = dict(dead=False, reason=reason, **evidence)
                self.audit.append(dict(game=overlay._game, side=f'{idx+1}P', chain_id=chain.chain_id,
                    t_sec=stamp, incoming=incoming, hands=hands, queue=queue, **value))
                return value
        return self._prove_boards(overlay, chain, idx, boards, queue, evidence, incoming,
                                  hands, palette, hidden, stamp)

    def _prove_boards(self, overlay: Any, chain: Any, idx: int, boards: list[Board],
                      queue: tuple, evidence: dict, incoming: int, hands: int,
                      palette: tuple, hidden: dict | None, stamp: float) -> dict:
        """照合を通過した候補だけを探索し、既存の証明キャッシュを共有する。"""
        elapsed = overlay.tracker._score_elapsed
        key = (tuple(b._grid.tobytes() for b in boards), incoming, queue, hands, palette,
               compute_effective_rate(elapsed))
        cached = key in self.cache
        started = perf_counter()
        if not cached:
            if len(self.cache) >= CACHE_LIMIT:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = self.proofs_for(boards, queue, incoming, hands, elapsed, palette)
        proofs = self.cache[key]
        dead = bool(proofs) and all(p['dead'] for p in proofs)
        value = dict(dead=dead, reason='post_counter_upper_bound' if dead else 'fallback_search',
                     candidates=len(boards), proofs=proofs, prediction_included=True)
        value.update(evidence)
        self.audit.append(dict(game=overlay._game, side=f'{idx+1}P', chain_id=chain.chain_id,
            t_sec=stamp, incoming=incoming, hands=hands, palette=palette, queue=queue,
            completion_score=hidden['score'] if hidden else chain.predicted_final_score,
            elapsed_sec=perf_counter()-started, cached=cached, **value))
        return value

    def current_queue(self, overlay: Any, chain: Any, idx: int) -> tuple[tuple, dict]:
        """既存のCHAIN・設置・NEXT終了観測だけで未知手持ちの前置を決める。"""
        queue, evidence = self.queues[idx], {}
        if not (self.current_puyo_integrity or self.current_puyo_integrity_v2c):
            return queue, evidence
        omit = False
        if self.current_puyo_integrity_v2c:
            signal = overlay._signals.get(chain.chain_id)
            observe = getattr(signal, 'current_puyo_observation', None)
            observed = {} if observe is None else observe(queue)
            omit = (self.states[idx] == 'CHAIN' and observed.get('firing_placed', False)
                    and observed.get('next_unchanged', False) and not self.next_motion[idx]
                    and not observed.get('ended', True) and chain.end_signal_sec is None)
            evidence['current_puyo_observation'] = dict(observed, state=self.states[idx],
                next_motion=self.next_motion[idx], end_signal_sec=chain.end_signal_sec,
                trigger_sec=chain.trigger_sec, next_queue=queue)
        evidence['unknown_current_pair_prepended'] = not omit
        evidence['unknown_current_pair_omitted_self_chain'] = omit
        if not omit:
            from src.exchange_death_integrity_v2 import UNKNOWN_CURRENT_PAIR
            queue = (*UNKNOWN_CURRENT_PAIR, *queue)
        return queue, evidence

    def prefix_check(self, projection: Any, overlay: Any, chain: Any, hidden: dict | None,
                     stamp: float | None = None) -> dict:
        """現在の予測と観測だけを照合し、後続の死亡観測は参照しない。"""
        if self.completion_recent_integrity:
            from src.exchange_post_counter_recent import check_recent
            return check_recent(projection, chain, hidden,
                                self.step_observations.get((overlay._game, chain.chain_id)),
                                strict_integrity=self.strict_integrity, stamp=stamp,
                                cap_completed=self.cap_completed)
        if self.completion_step_integrity:
            from src.exchange_post_counter_steps import check_steps
            return check_steps(projection, chain, hidden,
                               self.step_observations.get((overlay._game, chain.chain_id)))
        from src.exchange_post_counter_prefix import check_prefix
        return check_prefix(projection, chain, hidden,
                            self.formula_observations.get((overlay._game, chain.chain_id)))

    def refresh_current_observation(self, overlay: Any, result: Any = None,
                                    stamp: float | None = None) -> None:
        """NEXT基準確定・終了観測の変化でも、外側の評価キャッシュを更新する。"""
        if not self.current_puyo_integrity_v2c:
            return
        if self.completion_prefix_integrity and result is not None:
            from src.exchange_post_counter_prefix import observe_formula
            observe_formula(overlay, result, self.formula_observations)
        if (self.completion_step_integrity or self.completion_recent_integrity) and result is not None:
            from src.exchange_post_counter_steps import observe_steps
            observe_steps(overlay, result, self.step_observations)
        key = tuple(None if (chain := overlay.tracker.latest_chain(f'{idx+1}P')) is None
                    else (chain.chain_id, repr(self.current_queue(overlay, chain, idx)))
                    for idx in range(2))
        if self.completion_prefix_integrity or self.completion_step_integrity or self.completion_recent_integrity:
            from src.exchange_death_inputs import maximum_response
            projection = overlay._landing_projection
            key += tuple(None if (chain := overlay.tracker.latest_chain(f'{idx+1}P')) is None
                else repr(self.prefix_check(projection, overlay, chain, maximum_response(projection, chain), stamp))
                for idx in range(2))
        if key != self.current_observation_key:
            self.current_observation_key = key
            self.revision += 1

    def input_blocker(self, projection: Any, tracker: Any, idx: int, stamp: float,
                      context: dict | None = None) -> str | None:
        """予測時間は応手数の上限でない。終了後の操作観測まで証明を保留する。"""
        if self.next_motion[idx]:
            return 'next_transition'
        if projection.safety is None:
            return 'missing_input_safety'
        reason = projection.safety.blocker(projection, tracker, idx, stamp)
        if self.input_integrity_v2 and reason == 'completion_prefix_unverified':
            from src.exchange_death_integrity_v2 import completion_blocker
            reason = completion_blocker(projection, tracker, idx, context)
        if reason is not None:
            return reason
        attacker = tracker.latest_chain(f'{2-idx}P')
        if (attacker is None or attacker.end_signal_sec is None
                or attacker.post_end_drop_sec is None or attacker.end_confirmed is not True):
            return 'unverified_response_upper_bound'
        return None

    def proofs_for(self, boards: list[Board], queue: tuple, incoming: int, hands: int,
                   elapsed: float, palette: tuple) -> list[dict]:
        """候補一覧の変化を跨いで同一入力の全証明を共有する（試合内・有限）。"""
        inputs = (queue, incoming, hands, compute_effective_rate(elapsed), palette)
        proofs = []
        for board in boards:
            key = (board._grid.tobytes(), *inputs)
            if key not in self.board_cache:
                if len(self.board_cache) >= CACHE_LIMIT:
                    self.board_cache.pop(next(iter(self.board_cache)))
                self.board_cache[key] = prove_post_counter(board, queue, incoming, hands, elapsed, palette)
            proofs.append(self.board_cache[key])
            if self.early_exit and not proofs[-1]['dead']:
                break
        return proofs

    def evaluate(self, projection: Any, overlay: Any, latest: tuple, hands: tuple,
                 stamp: float, value: dict, context: dict | None) -> None:
        """新規証明だけを回避不能死へ追加し、既存の予測込み表示を利用する。"""
        if context is None:
            return
        rows = []
        for idx, incoming in enumerate(context['incoming']):
            side = f'{idx+1}P'
            if incoming <= 0 or side in value['dead_sides']:
                continue
            result = self.prove(projection, overlay, idx, latest, incoming, hands[idx], context, stamp)
            rows.append(dict(side=side, **result))
            if result['dead']:
                value['dead_sides'].append(side)
        value['post_counter_bound'] = rows
