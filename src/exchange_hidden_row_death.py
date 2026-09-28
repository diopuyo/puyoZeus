"""隠し段の全色組合せから、死亡判定専用の最大打ち返しを保持する。"""
from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Any
import numpy as np

from src.board import Board, COLOR_UNKNOWN
from src.board_state_machine import BoardState
from src.chain import ChainSimulator
from src.chain_detector import CHAIN_MECHANISM_FORMULA_READ
from src.exchange_event_tracker import valid_nonnegative, TIME_EPSILON_SEC
from src.exchange_midchain_completion import remaining, SETTLE_SAMPLES, MAX_SETTLE_GAP_SEC
from src.match_color_evidence import MatchColorEvidence

GAME_COLOR_COUNT = 4
MAX_HIDDEN_UNKNOWN = 3
COLORS = frozenset((1, 2, 3, 4, 5))


def enumerate_hidden(board: Board, colors: tuple, count: int, score: float,
                     simulator: ChainSimulator) -> tuple[list[dict], int]:
    """最大125通りを省略せず計算し、可視UNKNOWNと浮遊を補正しない。"""
    grid = board._grid
    columns = np.flatnonzero(grid[0] == COLOR_UNKNOWN)
    if (len(set(colors)) != GAME_COLOR_COUNT or not set(colors) <= COLORS
            or not 1 <= len(columns) <= MAX_HIDDEN_UNKNOWN
            or np.any(grid[1:] == COLOR_UNKNOWN)):
        return [], 0
    options, trials = [], 0
    for values in product((0, *colors), repeat=len(columns)):
        filled = board.copy()
        filled._grid[0, columns] = values
        trials += 1
        value = remaining(filled, count, score, simulator)
        if value is not None:
            options.append(dict(value, hidden=list(values)))
    return options, trials


class HiddenRowDeathCompletion:
    """確率層の連鎖レコードへ一切書き込まない、死亡専用の候補台帳。"""

    def __init__(self, simulator: ChainSimulator) -> None:
        self.simulator = simulator
        self.entries: dict[tuple, dict] = {}
        self.color_evidence = [MatchColorEvidence(), MatchColorEvidence()]
        self.palettes: tuple = ((), ())
        self.audit: list[dict] = []
        self.skipped: Counter = Counter()
        self.revision = 0

    def reset(self) -> None:
        """試合間で色と有効候補を持ち越さず、監査原票だけを保持する。"""
        self.entries.clear()
        self.color_evidence = [MatchColorEvidence(), MatchColorEvidence()]
        self.palettes = ((), ())
        self.revision += 1

    def observe(self, overlay: Any, result: Any, stamp: float) -> None:
        """因果的に確認した4色と、現在の式・落ち切りだけを入力する。"""
        sides = (result.p1, result.p2)
        for evidence, side in zip(self.color_evidence, sides):
            evidence.observe(side)
        palettes = tuple(e.active() for e in self.color_evidence)
        if palettes != self.palettes:
            self.palettes = palettes
            self.revision += 1
        for idx, side in enumerate(sides):
            chain = overlay.tracker.latest_chain(f'{idx+1}P')
            if chain is None:
                continue
            key = (overlay._game, idx, chain.chain_id)
            entry = self.entries.setdefault(key, dict(chain=chain, formula=None, pending=None,
                active=None, settle=None, samples=0, used_count=None))
            self._formula(key, entry, side.chain_event, stamp)
            self._sample(key, entry, side, stamp)

    def _formula(self, key: tuple, entry: dict, event: Any, stamp: float) -> None:
        """次段と後続式の一致候補だけを残し、空集合になれば失効する。"""
        if (event is None or event.mechanism != CHAIN_MECHANISM_FORMULA_READ
                or event.trigger_sec < entry['chain'].trigger_sec or event.chain_count < 1
                or not valid_nonnegative(event.total_score)):
            return
        observed = (event.chain_count, event.total_score)
        if observed == entry['formula']:
            return
        previous, entry['formula'] = entry['formula'], observed
        entry['settle'], entry['samples'] = None, 0
        active = entry['active']
        if active is not None:
            active['options'] = [v for v in active['options'] if v['prefix'].get(observed[0]) == observed[1]]
            if not active['options']:
                active['audit']['revoked_sec'] = stamp
                entry['active'] = None
            self.revision += 1
        pending = entry['pending']
        if pending is None or (previous is not None and observed[0] <= previous[0]):
            return
        entry['pending'] = None
        matching = [v for v in pending['options'] if v['next_count'] == observed[0]
                    and v['prefix'].get(observed[0]) == observed[1]]
        row = pending['audit']
        row.update(verified_sec=stamp, observed_count=observed[0], observed_score=observed[1],
            matching=len(matching), outcome='accepted' if matching else 'next_mismatch')
        if matching:
            entry['active'] = dict(options=matching, audit=row)
            row['maximum_score'] = max(v['score'] for v in matching)
        self.revision += 1

    def _sample(self, key: tuple, entry: dict, side: Any, stamp: float) -> None:
        """隠し段以外の欠測がなく、同一盤面が二観測続いた時だけ全列挙する。"""
        board = getattr(side, 'midchain_board', None)
        if side.state in (BoardState.MENU, BoardState.TSUMO_FALL, BoardState.OJAMA_FALL):
            entry['pending'] = None
        if side.state != BoardState.GRAVITY_SETTLE or board is None:
            entry['settle'], entry['samples'] = None, 0
            return
        grid = board._grid
        colors = self.palettes[key[1]]
        count = int(np.count_nonzero(grid[0] == COLOR_UNKNOWN))
        if (not 1 <= count <= MAX_HIDDEN_UNKNOWN or np.any(grid[1:] == COLOR_UNKNOWN)
                or np.any((grid[:-1] != 0) & (grid[1:] == 0))
                or entry['formula'] is None or len(colors) != GAME_COLOR_COUNT):
            entry['settle'], entry['samples'] = None, 0
            self.skipped['ineligible'] += 1
            return
        identity = (entry['formula'], grid.tobytes(), colors)
        gap = stamp-entry.get('stamp', -float('inf'))
        continuous = TIME_EPSILON_SEC < gap <= MAX_SETTLE_GAP_SEC+TIME_EPSILON_SEC
        entry['samples'] = entry['samples']+1 if entry['settle'] == identity and continuous else 1
        entry['settle'], entry['stamp'] = identity, stamp
        if entry['samples'] < SETTLE_SAMPLES or entry['used_count'] == entry['formula'][0]:
            return
        options, trials = enumerate_hidden(board, colors, *entry['formula'], self.simulator)
        row = dict(game=key[0], side=entry['chain'].side, chain_id=key[2], candidate_sec=stamp,
            unknown=count, colors=list(colors), trials=trials, candidates=len(options),
            outcome='awaiting_formula', used=False, uses=0)
        self.audit.append(row)
        entry['pending'] = dict(options=options, audit=row)
        entry['used_count'] = entry['formula'][0]

    def maximum(self, chain: Any) -> dict | None:
        """最大火力の同点候補を全て返し、恣意的に死にやすい終端を選ばない。"""
        if chain is None:
            return None
        if chain.end_signal_sec is not None and chain.score_ready_sec is not None and chain.end_confirmed is not False:
            return None
        entry = next((e for e in self.entries.values() if e['chain'] is chain), None)
        active = entry['active'] if entry else None
        if active is None:
            return None
        if tuple(active['audit']['colors']) != self.palettes[int(chain.side == '2P')]:
            return None
        maximum = max(v['score'] for v in active['options'])
        unique = {np.asarray(v['board']).tobytes(): v for v in active['options'] if v['score'] == maximum}
        return dict(score=maximum, options=list(unique.values()), audit=active['audit'])

    def mark_used(self, value: dict, stamp: float) -> None:
        """死亡証明の探索へ渡した候補数と探索呼出回数を区別して記録する。"""
        row = value['audit']
        row.setdefault('first_used_sec', stamp)
        row['used'], row['uses'] = True, row['uses']+1

    def summary(self) -> dict:
        """列挙・式一致・実使用を別母数で集計する。"""
        return dict(enumerated=len(self.audit), combinations=sum(r['trials'] for r in self.audit),
            accepted=sum(r['outcome'] == 'accepted' for r in self.audit),
            next_mismatch=sum(r['outcome'] == 'next_mismatch' for r in self.audit),
            used=sum(r['used'] for r in self.audit), uses=sum(r['uses'] for r in self.audit),
            skipped=dict(self.skipped), rows=self.audit)
