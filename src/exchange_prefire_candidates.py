"""保存起点から発火する全配置を保持し、段ごとの観測だけで絞る予測層。"""
from __future__ import annotations

from collections import Counter
from itertools import accumulate, combinations_with_replacement
from time import perf_counter
from typing import Any
import numpy as np

from src.board import Board, COLOR_UNKNOWN
from src.chain import ChainSimulator
from src.chain_detector import CHAIN_MECHANISM_FORMULA_READ
from src.match_color_evidence import MatchColorEvidence, GAME_COLOR_COUNT, COLORS
from src import puyo_core_bridge as native
from src.scoring import calculate_chain_score, score_to_ojama

PAIR_SIZE = 2
MAX_HANDS = 2
PERCENTILES = (50, 95)


def pairs_for(colors: tuple, queue: tuple) -> tuple:
    """4色10組に読取NEXTを加える。順序違いは22配置が既に網羅する。"""
    pairs = set(combinations_with_replacement(colors, PAIR_SIZE))
    for offset in range(0, len(queue), PAIR_SIZE):
        pair = tuple(sorted(queue[offset:offset+PAIR_SIZE]))
        if len(pair) == PAIR_SIZE and set(pair) <= COLORS:
            pairs.add(pair)
    return tuple(sorted(pairs))


def placements(board: Board, pairs: tuple, simulator: ChainSimulator) -> list:
    """発火で窒息を解消できる配置も含め、全22通りを省略せず計算する。"""
    if native.NATIVE_AVAILABLE:
        return [v for pair in pairs for v in native.enumerate_and_simulate_placements(
            board, pair, filter_dead=False,
            exclude_hidden_row_from_pop=simulator._exclude_hidden_row_from_pop)]
    from types import SimpleNamespace
    return [SimpleNamespace(placed_board=b, chain_result=simulator.simulate(b), is_dead=b.is_dead())
            for pair in pairs for _, _, b in native.enumerate_placements(board, pair, filter_dead=False)]


def completion(placed: Board, simulator: ChainSimulator) -> dict:
    """各段の厳密得点・消去色・終端を一緒に保存する。"""
    result = simulator.simulate(placed)
    score = calculate_chain_score(result)
    return dict(prefix=tuple(accumulate(s.score for s in score.steps)),
        colors=tuple(tuple(sorted({g.color for g in s.erased_groups})) for s in result.steps),
        score=score.total_score, board=result.final_board._grid.tolist(), weight=0)


def enumerate_candidates(board: Board, colors: tuple, queue: tuple,
                         simulator: ChainSimulator) -> tuple[list[dict], int]:
    """同じ中間盤面は計算を共用し、全設置経路の多重度を平均へ反映する。"""
    if len(colors) != GAME_COLOR_COUNT or np.any(board._grid == COLOR_UNKNOWN):
        return [], 0
    pairs, options, frontier, trials = pairs_for(colors, queue), {}, {}, 0
    for value in placements(board, pairs, simulator):
        trials += 1
        placed = value.placed_board
        raw = placed._grid.tobytes()
        if value.chain_result.chain_count:
            option = options.setdefault(raw, None)
            if option is None:
                options[raw] = completion(placed, simulator)
            options[raw]['weight'] += 1
        elif not value.is_dead:
            previous = frontier.get(raw, (placed, 0))
            frontier[raw] = (placed, previous[1]+1)
    for first, weight in frontier.values():
        for value in placements(first, pairs, simulator):
            trials += weight
            if not value.chain_result.chain_count:
                continue
            raw = value.placed_board._grid.tobytes()
            if raw not in options:
                options[raw] = completion(value.placed_board, simulator)
            options[raw]['weight'] += weight
    return list(options.values()), trials


def statistics(options: list[dict], elapsed: float) -> dict:
    """得点・送り量を各候補から平均し、終端は一意の最頻だけを採る。"""
    total = sum(v['weight'] for v in options)
    if not total:
        return dict(candidates=0, unique=0, mean_score=None, mean_send=None, board=None, predicted_count=None)
    boards: Counter = Counter()
    for value in options:
        boards[tuple(map(tuple, value['board']))] += value['weight']
    ranked = boards.most_common()
    mode = ranked[0][0] if len(ranked) == 1 or ranked[0][1] > ranked[1][1] else None
    return dict(candidates=total, unique=len(options),
        mean_score=sum(v['score']*v['weight'] for v in options)/total,
        mean_send=sum(score_to_ojama(v['score'], elapsed_sec=elapsed).ojama_count*v['weight']
                      for v in options)/total,
        board=list(map(list, mode)) if mode is not None else None,
        predicted_count=max(len(v['prefix']) for v in options))


class PrefireCandidates:
    """発火単位で一度列挙し、未来の最終得点を採否に使わない。"""

    def __init__(self, simulator: ChainSimulator) -> None:
        self.simulator = simulator
        self.entries: dict[int, dict] = {}
        self.audit: list[dict] = []
        self.evidence = MatchColorEvidence()
        self.revision = 0

    def reset(self) -> None:
        """試合をまたぐ候補・試合色を消去し、監査だけ残す。"""
        self.entries.clear()
        self.evidence = MatchColorEvidence()
        self.revision += 1

    def observe_colors(self, sides: tuple) -> None:
        """両者のSTABLE盤面から試合共通の4色を因果的に取得する。"""
        for side in sides:
            self.evidence.observe(side)

    def seed(self, chain: Any, event: Any, history: list, game: int, elapsed: float) -> None:
        """0連鎖の保存起点だけを補完し、既に発火を再現できた起点は維持する。"""
        if chain.chain_id in self.entries or chain.predicted_chain_count:
            return
        started = perf_counter()
        saved = next((s for s in reversed(history) if s.t_sec < chain.trigger_sec), None)
        board = getattr(event, 'before_board', None)
        board = board if board is not None else (saved.board if saved else None)
        colors = self.evidence.active()
        reason = ('missing_origin' if board is None else 'unknown_origin' if
            np.any(board._grid == COLOR_UNKNOWN) else 'unknown_palette' if len(colors) != GAME_COLOR_COUNT else None)
        queue = tuple(int(v) for v in saved.queue) if saved else ()
        options, trials = enumerate_candidates(board, colors, queue, self.simulator) if reason is None else ([], 0)
        row = dict(game=game, side=chain.side, chain_id=chain.chain_id, trigger_sec=chain.trigger_sec,
            colors=colors, queue=queue, trials=trials, skipped=reason, enumeration_sec=perf_counter()-started,
            initial=statistics(options, elapsed), observations=[], used=False, uses=0)
        row['initial'].pop('board')
        self.audit.append(row)
        self.entries[chain.chain_id] = dict(chain=chain, options=options, elapsed=elapsed,
            original=(chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board),
            published=None, last=None, audit=row)
        self._publish(self.entries[chain.chain_id])

    def observe(self, overlay: Any, result: Any, stamp: float) -> None:
        """同じ式の再通知では再計算せず、段ごとの累積得点と消去色を照合する。"""
        for idx, side in enumerate((result.p1, result.p2)):
            chain = overlay.tracker.latest_chain(f'{idx+1}P')
            entry = self.entries.get(chain.chain_id) if chain else None
            event = side.chain_event
            if (entry is None or event is None or event.mechanism != CHAIN_MECHANISM_FORMULA_READ
                    or event.trigger_sec < chain.trigger_sec or event.chain_count < 1):
                continue
            colors = getattr(event, 'erased_colors', None)
            key = (event.chain_count, event.total_score, tuple(sorted(colors)) if colors else None)
            if key == entry['last']:
                continue
            started = perf_counter()
            count, observed, colors = key
            entry['options'] = [v for v in entry['options'] if len(v['prefix']) >= count
                and v['prefix'][count-1] == observed and (colors is None or v['colors'][count-1] == colors)]
            entry['last'] = key
            stats = self._publish(entry)
            stats.pop('board')
            entry['audit']['observations'].append(dict(t_sec=stamp, count=count, observed=observed,
                erased_colors=colors, filter_sec=perf_counter()-started, **stats))
            self.revision += 1

    def _publish(self, entry: dict) -> dict:
        """平均を予測欄にだけ公開し、候補消滅時は直前の独立予測へ撤回する。"""
        chain = entry['chain']
        current = (chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board)
        if entry['published'] is not None and current != entry['published']:
            entry['original'] = current
        value = statistics(entry['options'], entry['elapsed'])
        predicted = ((value['mean_score'], value['predicted_count'], value['board']) if entry['options'] else entry['original'])
        chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board = predicted
        entry['published'] = predicted if entry['options'] else None
        entry['stats'] = value
        return dict(value)

    def active(self, chain: Any) -> dict | None:
        """実得点確定後は候補予測を返さない。"""
        if chain is None or (chain.end_signal_sec is not None and chain.score_ready_sec is not None
                             and chain.end_confirmed is not False):
            return None
        entry = self.entries.get(chain.chain_id)
        return entry if entry and entry['options'] else None

    def maximum(self, chain: Any) -> dict | None:
        """最大得点の全同点終端を返し、死亡の断定には全盤面の証明を要求する。"""
        entry = self.active(chain)
        if entry is None:
            return None
        maximum = max(v['score'] for v in entry['options'])
        unique = {tuple(map(tuple, v['board'])): v for v in entry['options'] if v['score'] == maximum}
        return dict(score=maximum, options=list(unique.values()), audit=entry['audit'], prefire=True)

    def provenance(self, chains: list) -> list[dict]:
        """確率出力へ候補数と予測由来を記録する。"""
        return [dict(side=c.side, chain_id=c.chain_id, **{k: e['stats'][k]
            for k in ('candidates', 'mean_score', 'mean_send')}) for c in chains if (e := self.active(c))]

    def sends(self, chains: list, elapsed: float) -> list[float] | None:
        """着弾層は切捨て後の候補送り量の平均を使い、平均得点から再切捨てしない。"""
        if not any(self.active(c) for c in chains):
            return None
        values = []
        for side in ('1P', '2P'):
            selected = [c for c in chains if c.side == side]
            predicted = [(c, self.active(c)) for c in selected]
            fallback = sum(c.provisional_score for c, e in predicted if e is None)
            values.append(float(score_to_ojama(fallback, elapsed_sec=elapsed).ojama_count)
                + sum(e['stats']['mean_send'] for _, e in predicted if e is not None))
        return values

    def summary(self) -> dict:
        """最終得点照合は監査だけで行い、予測器の採否には戻さない。"""
        return dict(rows=self.audit)
