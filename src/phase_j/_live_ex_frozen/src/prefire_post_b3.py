"""v4.5専用。未来を持たない通知から連鎖終了盤を固定B3へ渡す。既定では無効。"""
from __future__ import annotations
from src.phase_j._live_ex_frozen.src import prefire_work_budget as work_budget

from copy import copy, deepcopy
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Protocol
import numpy as np
from src.phase_j._live_ex_frozen.src.b3_state import GameState
from src.phase_j._live_ex_frozen.src.b3_features import build_features
from src.phase_j._live_ex_frozen.src.b3_timing import TimingTable
from src.phase_j._live_ex_frozen.src.exchange_count_ledger import gross, STEP_SEC
from src.phase_j._live_ex_frozen.src.exchange_native_chain import native_chains
from src.phase_j._live_ex_frozen.src.prefire_chain_board import initial, advance, error
from src.phase_j._live_ex_frozen.src.prefire_worker import LOW_SPEC_RATIO
from src.phase_j._live_ex_frozen.src.production_config import PREFIRE_V4_POST_ENABLED_DEFAULT

MS = 1000.
LIMIT_MS = 500.
HISTORY_SIZE = 3
ROWS, COLS = 13, 6
INPUT_FPS = 30
ACTIVE_STATES = frozenset(('CHAIN', 'GRAVITY_SETTLE'))


class Predictor(Protocol):
    def predict(self, row: dict) -> float:
        """固定B3の列順と推論を実装する。"""
        ...


def cancelled_pending(pending: list[float], remaining: list[float]) -> list[float]:
    """両側の未反映送りを同じ台帳上で一度だけ相殺する。"""
    result = list(pending)
    for side, sent in enumerate(remaining):
        offset = min(result[side], sent)
        result[side] -= offset
        result[1-side] += sent-offset
    return result


def validate_publication(value: dict) -> None:
    if any(t > value['input_sec'] for t in value['evidence_times']):
        raise ValueError('未来の入力')
    if not work_budget.enabled() and value['ready_sec']+1e-9 < value['input_sec']+value['low_spec_ms']/MS:
        raise ValueError('計算完了前の公開')


def forecast_state(current: GameState) -> GameState:
    """特徴量生成が変更する側状態と予告辞書だけを複製する。履歴は読取専用。"""
    state = copy(current)
    state.sides = [copy(side) for side in current.sides]
    state.sl_snap = copy(current.sl_snap)
    return state


def forecast_features(current: GameState, boards: list[list[int]], pending: list[float],
                      table: TimingTable, native: Any, fast_copy: bool = False) -> dict:
    """予測コピーだけを変更。特徴変換・差分・正規化はB3本体を通す。"""
    state = forecast_state(current) if fast_copy else deepcopy(current)
    for i, side in enumerate(state.sides):
        side.board = np.asarray(boards[i], dtype=np.int8).reshape(ROWS, COLS)
        side.board_t, side.state, side.episode = state.t, 'STABLE', None
        state.sl_snap[f'pending_p{i+1}'] = pending[i]
        state.sl_snap[f'pending_confirmed_p{i+1}'] = pending[i]
    with native_chains(native):
        return build_features(state, table)


class PostB3:
    """観測状態は外部wrapperに保持し、B3のモデル・設定は変更しない。"""

    def __init__(self, predictor: Predictor, table: TimingTable, policy: dict,
                 native: Any, enabled: bool = PREFIRE_V4_POST_ENABLED_DEFAULT) -> None:
        self.predictor, self.table, self.policy, self.native = predictor, table, policy, native
        self.enabled = enabled
        self.state = GameState(SimpleNamespace(rows_of=lambda side: ()))
        self.chains: list[dict | None] = [None, None]
        self.histories: list[list[dict]] = [[], []]
        self.ledger_history: list[tuple] = []
        self.fight: float | None = None
        self.last_key: tuple | None = None
        self.busy = 0.
        self.pending_since: float | None = None
        self.coalesced_inputs = 0

    def observe(self, stamp: float, game: int, raw: list[dict],
                forecasts: list[dict] | None = None) -> dict | None:
        if not self.enabled:
            return None
        began = perf_counter()
        if np.isfinite(self.state.t) and stamp <= self.state.t:
            raise ValueError('時刻順違反')
        self._boundary(stamp, game)
        self._current(stamp, game, raw)
        self._accept_forecasts(forecasts or [], stamp, game)
        self._chains(stamp, raw)
        stable = all(s.state == 'STABLE' for s in self.state.sides) and not any(
            c and c.get('external') and not c['finished'] for c in self.chains)
        if stable and self.fight is not None:
            result = dict(game=game, input_sec=stamp, ready_sec=stamp, closed=self.fight,
                          evidence_times=[stamp], low_spec_ms=0.,
                          fired=[c is not None for c in self.chains])
            self.fight, self.chains, self.last_key = None, [None, None], None
            self.pending_since = None
            return result
        if self.fight is None or any(s.board is None for s in self.state.sides):
            return None
        extra_ms = sum(r['low_spec_ms'] for r in forecasts or [])
        return self._evaluate(stamp, began, extra_ms)

    def _accept_forecasts(self, forecasts: list[dict], stamp: float, game: int) -> None:
        for forecast in forecasts:
            if forecast['input_sec'] > stamp or any(t > forecast['input_sec'] for t in forecast['evidence_times']):
                raise ValueError('未来のv3.4/v3.10候補')
            if forecast['game'] != game or 'board' not in forecast:
                continue
            side, trigger = forecast['side'], forecast['trigger']
            chain = self.chains[side]
            if chain and chain['signature'][0] > trigger:
                continue
            same = chain is not None and chain['signature'][0] == trigger
            baseline = next((r for r in reversed(self.ledger_history) if r[0] < trigger), None)
            baseline = baseline or (self.state.game_start_t, 0., 0., 0., 0.)
            chain = dict(signature=(trigger, 0, 0), external=True, external_sent=forecast['sent'],
                origin_input_sec=forecast['input_sec'],
                finished=forecast.get('completed', False) or (chain['finished'] if same else False),
                generated_at_fire=baseline[1+side], carry_at_fire=baseline[3+side], started=trigger,
                result=dict(board=forecast['board'], evidence_times=forecast['evidence_times']+[baseline[0]]),
                best=dict(scores=forecast['scores']))
            self.chains[side] = chain
            if self.fight is None:
                self.fight = stamp

    def _boundary(self, stamp: float, game: int) -> None:
        if self.state.game_idx != game:
            self.state.t = stamp
            self.state.reset_game(game)
            self.chains, self.histories = [None, None], [[], []]
            self.ledger_history = []
            self.fight, self.last_key, self.busy = None, None, stamp
            self.pending_since = None

    def _current(self, stamp: float, game: int, raw: list[dict]) -> None:
        state, sides = self.state, []
        state.t, state.frame, state.in_match = stamp, round(stamp*INPUT_FPS), True
        for i, (side, source) in enumerate(zip(state.sides, raw)):
            board = source.get('confirmed_board')
            board = board['board']['array'] if board is not None else None
            status = source['state']['state']
            side.update_frame(stamp, dict(state=status, score=source.get('score'),
                confirmed_board=board if status == 'STABLE' else None,
                next_slide_motion=source.get('next_slide_motion', False)), True)
            side.queue = tuple(c for k in ('next_pair', 'dnext_pair')
                               for c in (source.get(k) or {}).get('tuple', (0, 0)))
            side.queue_t = stamp
            state.colors.add_queue(side.queue)
            if side.first_place_t is None and source.get('next_slide_motion'):
                side.first_place_t = stamp
            if side.board_changed:
                state.colors.add_board(side.board)
                item = dict(t=stamp, board=side.board.ravel().tolist())
                self.histories[i] = (self.histories[i]+[item])[-HISTORY_SIZE:]
            sides.append(dict(side=side.name, state=status, score=source.get('score'),
                confirmed_board=board, next_slide_motion=source.get('next_slide_motion', False)))
            if self.state.sl.core.defer_all_clear:
                sides[-1]['chain_info'] = (source.get('chain_event') or {}).get('namespace')
        state.sl_snap = state.sl.update(dict(t_sec=stamp, frame=state.frame, game_idx=game,
                                             match_active=True, sides=sides))
        self._remember_ledger()
        for i, source in enumerate(raw):
            self._finish_chain(i, stamp, source)

    def _finish_chain(self, side: int, stamp: float, source: dict) -> None:
        """単発STABLEへの誤遷移で、配送中の式付き連鎖を終了しない。"""
        chain = self.chains[side]
        if chain is None:
            return
        stable = source['state']['state'] == 'STABLE'
        if not chain.get('external'):
            chain['finished'] |= stable
            return
        reflected = self.state.sl_snap[f'generated_p{side+1}']-chain['generated_at_fire']
        bonus = self.state.sl.core.all_clear[side].forecast(chain['started']) if self.state.sl.core.defer_all_clear else 0
        chain['finished'] |= stable and reflected >= chain['external_sent']+bonus

    def _resolving(self, side: int) -> bool:
        chain = self.chains[side]
        return chain is not None and not chain['finished'] and (
            chain.get('external', False) or self.state.sides[side].state in ACTIVE_STATES)

    def _defer_inference(self, stamp: float, began: float, extra_ms: float) -> bool:
        """推論中は観測だけ進め、未処理のB3要求を最新状態へまとめる。"""
        if work_budget.enabled() or self.busy <= stamp:
            return False
        self.busy += (perf_counter()-began)*LOW_SPEC_RATIO+extra_ms/MS
        self.pending_since = stamp if self.pending_since is None else self.pending_since
        self.coalesced_inputs += 1
        return True

    def _remember_ledger(self) -> None:
        snap = self.state.sl_snap
        if snap is not None:
            values = tuple(snap[f'{key}_p{i+1}'] for key in ('generated', 'carry') for i in range(2))
            if not self.ledger_history or values != self.ledger_history[-1][1:]:
                self.ledger_history.append((self.state.t, *values))

    def _chains(self, stamp: float, raw: list[dict]) -> None:
        from src.phase_j._live_ex_frozen.src.prefire_chain_identity import normalize, remember
        colors = self.state.colors.known()[1]
        for i, source in enumerate(raw):
            event = source.get('chain_event')
            if not event or source['state']['state'] != 'CHAIN' or len(colors) != 4:
                continue
            observed = event['namespace']
            event = normalize(self.chains[i], observed, stamp, source)
            signature = (event['trigger_sec'], event['chain_count'], event['total_score'])
            if signature[0] > stamp:
                raise ValueError('未来の発火通知')
            chain = self.chains[i]
            if chain and chain.get('external') and (not chain['finished'] or chain['signature'][0] == signature[0]):
                continue
            if chain and chain['signature'] == signature:
                continue
            if chain is None or chain['signature'][0] != signature[0]:
                chain = self._initial(i, event, stamp, colors)
            chain['signature'] = signature
            chain['result'] = advance(chain, int(signature[1]), int(signature[2]), stamp,
                                      list(colors), self.policy, self.native)
            chain['best'] = min(chain['pool'], key=lambda v: (error(v, chain['observed']), -v['log_weight']))
            self.chains[i] = chain
            remember(chain, observed, stamp, source)
            if self.fight is None:
                self.fight = stamp

    def _initial(self, side: int, event: dict, stamp: float, colors: tuple) -> dict:
        board = np.asarray(event['before_board']['board']['array']).ravel().tolist()
        queue = self.state.sides[side].queue
        known = tuple(tuple(queue[k:k+2]) for k in range(0, len(queue), 2))
        chain = initial(board, self.histories[side], known, list(colors), stamp, self.policy, self.native)
        # 通知が遅れた場合も、既に反映された初段以降を将来送りへ重複加算しない。
        baseline = next((r for r in reversed(self.ledger_history) if r[0] < event['trigger_sec']), None)
        if baseline is None:
            baseline = (self.state.game_start_t, 0., 0., 0., 0.)
        chain['evidence'].append(baseline[0])
        chain.update(generated_at_fire=baseline[1+side], carry_at_fire=baseline[3+side],
                     started=event['trigger_sec'], finished=False)
        return chain

    def projection(self) -> dict:
        snap, boards, remaining, evidence = self.state.sl_snap, [], [], [self.state.t]
        if not snap:
            raise ValueError('得点台帳未到達')
        evidence.append(snap['available_at'])
        for i, side in enumerate(self.state.sides):
            chain = self.chains[i]
            resolving = self._resolving(i)
            board = chain['result']['board'] if resolving else side.board.ravel().tolist()
            boards.append(board)
            if chain:
                scores = chain['best']['scores']
                times = [chain['started']+k*STEP_SEC for k in range(len(scores))]
                first = self.state.first_move_t() or self.state.game_start_t
                total, _ = gross(scores, times, int(chain['carry_at_fire']), first)
                total = chain.get('external_sent', total)
                reflected = snap[f'generated_p{i+1}']-chain['generated_at_fire']
                remaining.append(max(0., total-reflected) if not chain['finished'] else 0.)
                evidence.extend(chain['result']['evidence_times'])
            else:
                remaining.append(0.)
        pending = [float(snap[f'pending_p{i+1}']) for i in range(2)]
        post = cancelled_pending(pending, remaining)
        return dict(boards=boards, pending=post, current_pending=pending, remaining=remaining,
                    now=pending[1]-pending[0], post=post[1]-post[0], evidence_times=evidence)

    def _evaluate(self, stamp: float, began: float, extra_ms: float = 0.) -> dict | None:
        if self.state.sl_snap is None:
            return None
        if any(not np.isfinite(self.state.sl_snap[f'pending_p{i+1}']) for i in range(2)):
            return dict(game=self.state.game_idx, input_sec=stamp, ready_sec=stamp,
                        fight=self.fight, missing='予告未確定', evidence_times=[stamp], low_spec_ms=0.)
        projected = self.projection()
        if any(c not in (0, 1, 2, 3, 4, 5, 9) for board in projected['boards'] for c in board):
            return dict(game=self.state.game_idx, input_sec=stamp, ready_sec=stamp,
                        fight=self.fight, missing='未確定色を含む盤面',
                        evidence_times=projected['evidence_times'], low_spec_ms=0.)
        key = (tuple(map(tuple, projected['boards'])), tuple(projected['pending']),
               tuple(s.queue for s in self.state.sides), tuple(projected['current_pending']))
        if key == self.last_key:
            spent = (perf_counter()-began)*LOW_SPEC_RATIO+extra_ms/MS
            self.busy = max(self.busy, stamp)+spent
            self.pending_since = None
            return None
        if self._defer_inference(stamp, began, extra_ms):
            return None
        self.last_key = key
        infer = perf_counter()
        row = forecast_features(self.state, projected['boards'], projected['pending'], self.table, self.native)
        probability = self.predictor.predict(row)
        inference_ms = (perf_counter()-infer)*MS
        measured = (perf_counter()-began)*MS
        elapsed = measured*LOW_SPEC_RATIO+extra_ms
        ready = max(stamp, self.busy)+elapsed/MS
        self.busy = ready
        requested = stamp if self.pending_since is None else self.pending_since
        self.pending_since = None
        value = dict(projected, game=self.state.game_idx, input_sec=stamp, ready_sec=ready,
            fight=self.fight, v_post=probability, inference_ms=inference_ms, measured_ms=measured,
            low_spec_ms=elapsed, over_budget=ready-requested > LIMIT_MS/MS,
            requested_sec=requested, request_latency_ms=(ready-requested)*MS,
            fired=[c is not None for c in self.chains],
            forecast_sources=['v3.4/v3.10' if c and c.get('external') else '通知候補' for c in self.chains],
            forecast_sent=[c.get('external_sent') if c else None for c in self.chains],
            forecast_input=[c.get('origin_input_sec') if c else None for c in self.chains],
            board_prediction_active=[self._resolving(i) for i in range(2)],
            chain_ids=[c['signature'][0] if c else None for c in self.chains])
        work_budget.publication(value)
        validate_publication(value)
        return value
