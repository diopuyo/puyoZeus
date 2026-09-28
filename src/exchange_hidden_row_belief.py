"""E32: 隠し段の履歴分布を発火前画像へ結合する予測専用層。"""
from __future__ import annotations
from collections import Counter, deque
from copy import deepcopy
from time import perf_counter
from typing import Any
import numpy as np
from src.board import Board, COLOR_UNKNOWN
from src.chain import ChainSimulator
from src.chain_detector import CHAIN_MECHANISM_FORMULA_READ
from src.exchange_prefire_snapshot import PrefireSnapshot
from src.exchange_prefire_candidates import completion, statistics
from src.hidden_row_belief import HiddenRowBelief, combinations
from src.prefire_snapshot_reader import validate_snapshot

HISTORY_FRAMES = 128
MAJORITY = 0.5


def validate_visible(origin: Board | None, snapshot: dict | None, colors: tuple) -> tuple:
    """E31と同じ可視段検査を適用し、不可視セルだけを検査対象から外す。"""
    if not snapshot or snapshot.get('board') is None:
        return validate_snapshot(origin, snapshot, colors)
    grid = np.asarray(snapshot['board']).copy()
    if grid.shape != Board()._grid.shape:
        return 'invalid_shape', None
    grid[0] = 0
    before = origin.copy() if origin is not None else None
    if before is not None:
        before._grid[0] = 0
    return validate_snapshot(before, dict(snapshot, board=grid.tolist()), colors)


def majority_statistics(options: list[dict], elapsed: float) -> dict:
    """終端は最大重みが全候補の過半を占める時だけ公開する。"""
    value = statistics(options, elapsed)
    if options:
        weights: Counter = Counter()
        for option in options:
            weights[tuple(map(tuple, option['board']))] += option['weight']
        if max(weights.values()) <= MAJORITY*sum(weights.values()):
            value['board'] = None
    return value


class HiddenRowPrefire(PrefireSnapshot):
    """可視盤面を固定し、不可視段だけを因果分布から列挙する。"""

    def __init__(self, simulator: ChainSimulator) -> None:
        super().__init__(simulator)
        self.beliefs = [HiddenRowBelief(simulator) for _ in range(2)]
        self.timelines = [deque(maxlen=HISTORY_FRAMES) for _ in range(2)]
        self.calibration: list[dict] = []
        self.game: int | None = None

    def reset(self) -> None:
        """試合境界で推定履歴を消し、検証原票のみ保存する。"""
        super().reset()
        self.beliefs = [HiddenRowBelief(self.simulator) for _ in range(2)]
        self.timelines = [deque(maxlen=HISTORY_FRAMES) for _ in range(2)]

    def observe_history(self, sides: tuple, stamp: float, game: int) -> None:
        """表示時刻以前の履歴を保持し、古い画像窓には当時の分布だけを使う。"""
        self.game = game
        colors = self.evidence.active()
        for idx, (belief, side) in enumerate(zip(self.beliefs, sides)):
            belief.observe(side, colors, stamp)
            self.calibration.extend(dict(row, game=game, side=f'{idx+1}P') for row in belief.audit)
            belief.audit.clear()
            self.timelines[idx].append((stamp, deepcopy(belief, {id(self.simulator): self.simulator})))

    def _options(self, chain: Any, board: Board, snapshot: dict) -> tuple:
        """発火時刻や未来の色を使わず、取得窓の終了時までの履歴から生成する。"""
        idx = int(chain.side == '2P')
        end = snapshot.get('end_sec', chain.trigger_sec)
        saved = next((b for t, b in reversed(self.timelines[idx]) if t <= end), None)
        belief = (deepcopy(saved, {id(self.simulator): self.simulator}) if saved is not None
                  else HiddenRowBelief(self.simulator))
        colors = self.evidence.active()
        belief.advance(board._grid, colors, end, belief.pair)
        cells = belief.snapshot(board._grid, colors)
        choices, mass = combinations(cells)
        options = []
        for hidden, weight in choices:
            placed = board.copy()
            placed._grid[0] = hidden
            options.append(dict(completion(placed, self.simulator), weight=weight))
        return options, mass, [c.probs for c in cells]

    def seed(self, chain: Any, event: Any, history: list, game: int, elapsed: float) -> None:
        """全発火に一度だけ採否・打切り確率・処理時間を記録する。"""
        if chain.chain_id in self.entries:
            return
        started = perf_counter()
        saved = next((s for s in reversed(history) if s.t_sec < chain.trigger_sec), None)
        origin = getattr(event, 'before_board', None)
        origin = origin if origin is not None else saved.board if saved else None
        snapshot = self.snapshots.get(chain.side)
        reason, board = validate_visible(origin, snapshot, self.evidence.active())
        options, mass, cells = self._options(chain, board, snapshot) if reason is None else ([], 0., [])
        enumerated = len(options)
        if reason is None and (event.mechanism != CHAIN_MECHANISM_FORMULA_READ or event.chain_count != 1):
            reason = 'missing_first_formula'
        if reason is None:
            options = [v for v in options if v['prefix'] and v['prefix'][0] == event.total_score]
            reason = None if options else 'first_score'
        options = [] if reason else options
        self._normalize(options)
        stats = majority_statistics(options, elapsed)
        row = dict(game=game, side=chain.side, chain_id=chain.chain_id, trigger_sec=chain.trigger_sec,
            reason=reason, accepted=reason is None, withdrawn=False, snapshot=snapshot, used=False, uses=0,
            predicted_score=stats['mean_score'], observations=[], hidden_distributions=cells,
            enumerated=enumerated, retained_mass=mass, enumeration_sec=perf_counter()-started)
        self.audit.append(row)
        self.entries[chain.chain_id] = dict(chain=chain, options=options, elapsed=elapsed,
            original=(chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board),
            published=None, last=None, audit=row, stats=stats)

    @staticmethod
    def _normalize(options: list[dict]) -> None:
        """観測と不一致の候補を捨てた後で事後分布を正規化する。"""
        total = sum(v['weight'] for v in options)
        for value in options:
            value['weight'] /= total

    def observe(self, overlay: Any, result: Any, stamp: float) -> None:
        """各段の式で候補を除外し、消滅時はE27の同フレーム予測へ戻す。"""
        for idx, side in enumerate((result.p1, result.p2)):
            chain = overlay.tracker.latest_chain(f'{idx+1}P')
            entry = self.entries.get(chain.chain_id) if chain else None
            if entry is None or not entry['options']:
                continue
            entry['original'] = (chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board)
            event = side.chain_event
            if event and event.mechanism == CHAIN_MECHANISM_FORMULA_READ and event.trigger_sec >= chain.trigger_sec:
                key = (event.chain_count, event.total_score)
                if key != entry['last']:
                    self._filter(entry, key, stamp)
            self._publish(entry)

    def _filter(self, entry: dict, key: tuple, stamp: float) -> None:
        """段欠落も永久撤回とし、同じ段の再通知を二重計上しない。"""
        started = perf_counter()
        count, score = key
        previous = entry['last'][0] if entry['last'] else 0
        continuous = previous <= count <= previous+1
        entry['options'] = [v for v in entry['options'] if continuous and 0 < count <= len(v['prefix'])
                            and v['prefix'][count-1] == score]
        self._normalize(entry['options'])
        entry['audit']['observations'].append(dict(t_sec=stamp, count=count, score=score,
            remaining=len(entry['options']), filter_sec=perf_counter()-started))
        if not entry['options']:
            entry['audit'].update(withdrawn=True, withdraw_sec=stamp,
                withdraw_reason='stage_score' if continuous else 'stage_gap')
        entry['last'] = key
        self.revision += 1

    def _publish(self, entry: dict) -> dict:
        """確率集約用の得点と、過半を満たした完走盤面だけを公開する。"""
        value = majority_statistics(entry['options'], entry['elapsed'])
        predicted = ((value['mean_score'], value['predicted_count'], value['board'])
                     if entry['options'] else entry['original'])
        chain = entry['chain']
        chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board = predicted
        entry['published'] = predicted if entry['options'] else None
        entry['stats'] = value
        return dict(value)

    def maximum(self, chain: Any) -> dict | None:
        """全生存候補の最大火力と全終端で最も楽観的な死亡反証を許す。"""
        bound = super().maximum(chain)
        if bound is not None:
            entry = self.active(chain)
            unique = {tuple(map(tuple, v['board'])): v for v in entry['options']}
            bound['options'] = list(unique.values())
        return bound

    def summary(self) -> dict:
        """発火監査と自己較正原票を返す。"""
        return dict(rows=self.audit, calibration=self.calibration)
