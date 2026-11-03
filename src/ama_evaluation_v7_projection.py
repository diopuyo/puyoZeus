"""v7予測層（明示使用専用）。観測を時刻順に受け、実盤とは別に終了盤を持つ。"""
from __future__ import annotations
from copy import deepcopy
from typing import Any
from src.exchange_count_prediction import simulate
from src.exchange_count_ledger import gross
from src.ama_rules_v5 import landing_proof
from src.b3_timing import TimingTable

SIDES = (0, 1)
VALID_COLORS = (0, 1, 2, 3, 4, 5, 9)


def board_of(raw: dict, key: str = 'confirmed_board') -> list | None:
    value = raw.get(key)
    return [int(c) for row in value['board']['array'] for c in row] if value else None


def net_pending(pending: list, remaining: list) -> list:
    """双方の未反映送りを自己相殺してから相手予告へ移す。"""
    own = [max(0, p-s) for p, s in zip(pending, remaining)]
    sent = [max(0, s-p) for p, s in zip(pending, remaining)]
    cancel = min(sent)
    return [own[i]+sent[1-i]-cancel for i in SIDES]


class Projection:
    """mutableな履歴は予測ラッパー内だけ。確定盤・本番設定を変更しない。"""

    def __init__(self, native: Any, stable_origin: bool = False) -> None:
        self.native = native
        self.boards = [None, None]
        self.times = [None, None]
        self.states = [None, None]
        self.active = [None, None]
        self.bonus = [False, False]
        self.records: list[dict] = []
        self.last_t = float('-inf')
        self.chain_ended = [float('-inf'), float('-inf')]
        self.stable_origin = stable_origin
        self.stable_boards = [None, None]
        self.stable_times = [None, None]
        self.stable_history: list[list] = [[], []]

    def chain(self, side: int, raw: dict, stamp: float, ledger: dict) -> bool:
        event = (raw.get('chain_event') or {}).get('namespace')
        if not event:
            return False
        if event['trigger_sec'] > stamp:
            raise ValueError('未来の発火時刻')
        previous = self.active[side]
        # 同じCHAIN区間内でも原票のtrigger_secは更新される。撃った瞬間の盤は保持する。
        if previous and previous['kind'] == 'chain':
            return False
        if event['trigger_sec'] <= self.chain_ended[side]:
            return False
        saved = next((r for r in reversed(self.stable_history[side])
                      if r['t'] < event['trigger_sec']), None) if self.stable_origin else None
        board = saved['board'] if saved else None if self.stable_origin else board_of(event, 'before_board')
        if board is None or any(c not in VALID_COLORS for c in board):
            return False
        value = simulate(board, self.native)
        table = TimingTable()
        times = [stamp+(i+1)*table.stage_sec for i in range(len(value['scores']))]
        origin = stamp if ledger['origin'] is None else ledger['origin']
        amount, carry = gross(value['scores'], times, ledger['carry'][side],
                              origin, self.bonus[side])
        record = dict(id=len(self.records), side=side, kind='chain', start=stamp,
            trigger=event['trigger_sec'], board=value['final'], origin=board,
            scores=value['scores'], gross=amount, carry=carry,
            generated=ledger['generated'][side], used_bonus=self.bonus[side],
            next_bonus=not any(value['final']), evidence_times=[stamp, ledger['t']],
            end=None, mismatch=None)
        if self.stable_origin:
            record.update(origin_kind='prefire_stable', origin_sec=saved['t'])
            record['evidence_times'].append(saved['t'])
        self.records.append(record)
        self.active[side] = record
        self.boards[side], self.times[side] = value['final'], stamp
        return True

    def landing(self, side: int, raw: dict, stamp: float, ledger: dict) -> bool:
        previous = self.active[side]
        if self.states[side] == 'OJAMA_FALL' or (previous and previous['kind'] == 'ojama'):
            return False
        board = self.boards[side] if self.active[side] else (
            self.stable_boards[side] if self.stable_origin else board_of(raw))
        if board is None or any(c not in VALID_COLORS for c in board):
            return False
        pending = ledger['pending'][side]
        proof = landing_proof(board, pending, 'carry')
        parent = self.active[side]
        record = dict(id=len(self.records), side=side, kind='ojama', start=stamp,
            board=proof['board'], origin=board, landing=proof,
            pending=pending-proof['landed'], dropped=ledger['dropped'][side],
            parent=parent['id'] if parent else None,
            evidence_times=[stamp, ledger['t']], end=None, mismatch=None)
        self.records.append(record)
        if parent:
            # 連鎖直後にSTABLEが無い場合は着地込みの予測を最初の確定盤と照合する。
            parent['validation_board'] = proof['board']
            parent['validation_scope'] = 'chain_and_landing'
            if parent['kind'] == 'chain':
                self.bonus[side] = parent['next_bonus']
        self.active[side] = record
        self.boards[side], self.times[side] = proof['board'], stamp
        return True

    def stable(self, side: int, raw: dict, stamp: float) -> bool:
        board = board_of(raw)
        if board is None:
            return False
        previous = self.active[side]
        changed = board != self.boards[side] or self.states[side] != 'STABLE'
        if previous:
            self.finish(previous, board, stamp)
            self.active[side] = None
        self.boards[side], self.times[side] = board, stamp
        self.stable_boards[side], self.stable_times[side] = list(board), stamp
        if self.stable_origin:
            self.stable_history[side].append(dict(t=stamp, board=list(board)))
        return changed

    def finish(self, record: dict, board: list, stamp: float) -> None:
        """答え合わせだけ。保存済み評価入力の盤・予告を修正しない。"""
        expected = record.get('validation_board', record['board'])
        record.update(end=stamp, actual=board, mismatch=expected != board,
                      different_cells=sum(a != b for a, b in zip(expected, board)))
        if record['kind'] == 'chain':
            self.chain_ended[record['side']] = stamp
            self.bonus[record['side']] = (record['next_bonus'] if record.get('validation_scope')
                                           == 'chain_and_landing' else not any(board))
        elif record.get('parent') is not None:
            self.finish(self.records[record['parent']], board, stamp)

    def observe(self, raw: list, stamp: float, ledger: dict) -> bool:
        if stamp < self.last_t or ledger['t'] > stamp:
            raise ValueError('未来参照または時刻逆転')
        self.last_t = stamp
        changed, predicted_before = False, any(self.active)
        states = [r['state']['state'] for r in raw]
        for side, state in enumerate(states):
            if state == 'CHAIN':
                changed |= self.chain(side, raw[side], stamp, ledger)
            elif state == 'OJAMA_FALL':
                changed |= self.landing(side, raw[side], stamp, ledger)
            elif state == 'STABLE':
                changed |= self.stable(side, raw[side], stamp)
        self.states = states
        # 観測・確定台帳は進めるが、両側連鎖中は評価を一切発行しない。
        if states == ['CHAIN', 'CHAIN'] or 'MENU' in states:
            return False
        return changed and (predicted_before or any(self.active))

    def snapshot(self, ledger: dict) -> dict:
        if ledger['t'] > self.last_t:
            raise ValueError('未来の台帳')
        if self.states == ['CHAIN', 'CHAIN']:
            raise ValueError('両側連鎖中は評価しない')
        remaining, carries, bonuses = [0, 0], list(ledger['carry']), list(self.bonus)
        pending = list(ledger['pending'])
        for side, record in enumerate(self.active):
            if record and record['kind'] == 'chain':
                reflected = max(0, ledger['generated'][side]-record['generated'])
                remaining[side] = max(0, record['gross']-reflected)
                carries[side], bonuses[side] = record['carry'], record['next_bonus']
            elif record and record['kind'] == 'ojama':
                reflected = max(0, ledger['dropped'][side]-record['dropped'])
                pending[side] = max(0, pending[side]-max(0, record['landing']['landed']-reflected))
        return deepcopy(dict(boards=self.boards, pending=net_pending(pending, remaining),
            carry=carries, all_clear=bonuses, remaining=remaining,
            prediction_ids=[r['id'] for r in self.active if r],
            kinds=[r['kind'] if r else 'stable' if s == 'STABLE' else 'held'
                   for r, s in zip(self.active, self.states)],
            evidence_times=[ledger['t']]+[t for t in self.times if t is not None]))
