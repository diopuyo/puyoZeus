"""落ち切り観測を次段の実測得点で検証する、既定OFFの予測専用層。"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any
import numpy as np

from src.board import Board, COLOR_UNKNOWN, BOARD_ROWS, BOARD_COLS
from src.board_state_machine import BoardState
from src.chain import ChainSimulator
from src.chain_detector import CHAIN_MECHANISM_FORMULA_READ
from src.scoring import calculate_step_score
from src.exchange_event_tracker import valid_nonnegative, OBSERVATION_FPS, TIME_EPSILON_SEC

SETTLE_SAMPLES = 2
MAX_SETTLE_GAP_SEC = SETTLE_SAMPLES / OBSERVATION_FPS
PREDICTION_FIELDS = ('predicted_final_score', 'predicted_chain_count', 'predicted_final_board')


def compact(board: Board) -> bool:
    """UNKNOWNや支持のないセルを補正せず拒否する。"""
    grid = board._grid
    return (grid.shape == (BOARD_ROWS, BOARD_COLS) and not np.any(grid == COLOR_UNKNOWN)
            and not np.any((grid[:-1] != 0) & (grid[1:] == 0)))


def remaining(board: Board, count: int, score: float, simulator: ChainSimulator) -> dict | None:
    """残り一段目にも元連鎖の段数ボーナスを使い、実測累積点へ加算する。"""
    if not compact(board):
        return None
    result = simulator.simulate(board)
    if not result.steps:
        return None
    prefix, total = {}, score
    for step in result.steps:
        absolute = count+step.chain_index
        total += calculate_step_score(replace(step, chain_index=absolute)).score
        prefix[absolute] = total
    return dict(prefix=prefix, score=total, count=count+result.chain_count,
                board=result.final_board._grid.tolist(), next_count=count+1)


class MidchainCompletion:
    """盤面履歴を変更せず、連鎖IDごとに候補・採用・最終照合を管理する。"""

    def __init__(self, simulator: ChainSimulator) -> None:
        self.simulator = simulator
        self.entries: dict[tuple, dict] = {}
        self.audit: list[dict] = []
        self.skipped: Counter = Counter()
        self.candidates = 0

    def reset(self) -> None:
        """試合境界で未採用候補だけ失効させ、監査用の連鎖参照は保持する。"""
        for entry in self.entries.values():
            entry['pending'], entry['settle'] = None, None

    def observe(self, overlay: Any, result: Any, stamp: float) -> None:
        """得点通知の更新後、モデル予測の更新前にだけ呼ぶ。"""
        for idx, side in enumerate((result.p1, result.p2)):
            chain = overlay.tracker.latest_chain(f'{idx+1}P')
            if chain is None:
                continue
            key = (overlay._game, idx, chain.chain_id)
            entry = self.entries.setdefault(key, dict(chain=chain, formula=None,
                pending=None, active=None, settle=None, samples=0, used_count=None))
            self._formula(key, entry, side.chain_event, stamp)
            self._sample(key, entry, side, stamp)

    def _formula(self, key: tuple, entry: dict, event: Any, stamp: float) -> None:
        """次段の段数と累積点を同じ通知で照合し、後続矛盾では予測を撤回する。"""
        chain = entry['chain']
        if (event is None or event.mechanism != CHAIN_MECHANISM_FORMULA_READ
                or event.trigger_sec < chain.trigger_sec or event.chain_count < 1
                or not valid_nonnegative(event.total_score)):
            return
        observed = (event.chain_count, event.total_score)
        if observed == entry['formula']:
            return
        previous, entry['formula'] = entry['formula'], observed
        entry['settle'], entry['samples'] = None, 0
        pending, active = entry['pending'], entry['active']
        if active and active['prefix'].get(observed[0]) != observed[1]:
            self._restore(entry)
            active['audit']['revoked_sec'] = stamp
            active['audit']['revoke_reason'] = 'later_formula_mismatch'
        if pending is None or (previous is not None and observed[0] <= previous[0]):
            return
        entry['pending'] = None
        matched = observed[0] == pending['next_count'] and pending['prefix'][observed[0]] == observed[1]
        row = dict(game=key[0], side=chain.side, chain_id=chain.chain_id,
            candidate_sec=pending['stamp'], t_sec=stamp, next_count=pending['next_count'],
            observed_count=observed[0], observed_score=observed[1],
            expected_score=pending['prefix'][pending['next_count']],
            predicted_final_score=pending['score'], outcome='accepted' if matched else 'next_mismatch')
        self.audit.append(row)
        if matched:
            self._accept(entry, pending, row)

    def _accept(self, entry: dict, pending: dict, row: dict) -> None:
        """実測一致後だけ完走予測を書き替え、由来を監査記録へ残す。"""
        chain = entry['chain']
        if entry['active'] is None:
            entry['original'] = tuple(getattr(chain, name) for name in PREDICTION_FIELDS)
        for name, value in zip(PREDICTION_FIELDS, (pending['score'], pending['count'], pending['board'])):
            setattr(chain, name, value)
        pending['audit'] = row
        entry['active'] = pending

    def _restore(self, entry: dict) -> None:
        """矛盾した途中予測を残さず、採用前の予測へ戻す。"""
        for name, value in zip(PREDICTION_FIELDS, entry['original']):
            setattr(entry['chain'], name, value)
        entry['active'] = None

    def _sample(self, key: tuple, entry: dict, side: Any, stamp: float) -> None:
        """重力待ち中の連続二観測一致・支持ありだけを落ち切り候補にする。"""
        board = getattr(side, 'midchain_board', None)
        if side.state in (BoardState.MENU, BoardState.TSUMO_FALL, BoardState.OJAMA_FALL):
            entry['pending'] = None
        if side.state != BoardState.GRAVITY_SETTLE or board is None:
            entry['settle'], entry['samples'] = None, 0
            if side.state == BoardState.GRAVITY_SETTLE:
                self.skipped['missing_board'] += 1
            return
        if not compact(board) or entry['formula'] is None:
            entry['settle'], entry['samples'] = None, 0
            self.skipped['unknown_or_floating' if not compact(board) else 'missing_formula'] += 1
            return
        count, score = entry['formula']
        identity = (count, board._grid.tobytes())
        previous_stamp = entry.get('stamp', -float('inf'))
        continuous = TIME_EPSILON_SEC < stamp-previous_stamp <= MAX_SETTLE_GAP_SEC+TIME_EPSILON_SEC
        entry['samples'] = entry['samples']+1 if entry['settle'] == identity and continuous else 1
        entry['settle'], entry['stamp'] = identity, stamp
        if entry['samples'] < SETTLE_SAMPLES or entry['used_count'] == count:
            return
        value = remaining(board, count, score, self.simulator)
        if value is None:
            self.skipped['no_remaining_chain'] += 1
            return
        value['stamp'] = stamp
        entry['pending'], entry['used_count'] = value, count
        self.candidates += 1

    def provenance(self, game: int, chains: list | None = None) -> list[dict]:
        """現在有効な予測を、実測確定とは別の由来として表示・記録へ渡す。"""
        return [dict(side=e['chain'].side, chain_id=e['chain'].chain_id,
            accepted_sec=e['active']['audit']['t_sec'], layer='prediction')
            for key, e in self.entries.items() if key[0] == game and self._predicting(e)
            and (chains is None or any(c is e['chain'] for c in chains))]

    @staticmethod
    def _predicting(entry: dict) -> bool:
        """完了済み連鎖を新しい交換の予測由来として表示しない。"""
        chain = entry['chain']
        finalized = (chain.end_signal_sec is not None and chain.score_ready_sec is not None
                     and chain.end_confirmed is not False)
        return entry['active'] is not None and not finalized

    def verified(self, chain: Any) -> bool:
        """式一致で採用済みの同一連鎖だけを整合条件へ伝える。"""
        return any(e['chain'] is chain and self._predicting(e) for e in self.entries.values())

    def summary(self) -> dict:
        """採用一回を一件とし、最終得点が欠測なら不一致ゼロへ紛れ込ませない。"""
        chains = {(key[0], e['chain'].side, key[2]): e['chain'] for key, e in self.entries.items()}
        for row in self.audit:
            if row['outcome'] != 'accepted':
                continue
            chain = chains[(row['game'], row['side'], row['chain_id'])]
            ready = (chain.score_ready_sec is not None and chain.end_signal_sec is not None
                     and chain.end_confirmed is not False and chain.score_delta is not None)
            row['final_score'] = chain.score_delta if ready else None
            row['final_mismatch'] = chain.score_delta != row['predicted_final_score'] if ready else None
        accepted = [r for r in self.audit if r['outcome'] == 'accepted']
        return dict(candidates=self.candidates, accepted=len(accepted),
            accepted_chains=len({(r['game'], r['side'], r['chain_id']) for r in accepted}),
            next_mismatch=sum(r['outcome'] == 'next_mismatch' for r in self.audit),
            final_mismatch=sum(r['final_mismatch'] is True for r in accepted),
            final_unresolved=sum(r['final_mismatch'] is None for r in accepted),
            skipped=dict(self.skipped), rows=self.audit)
