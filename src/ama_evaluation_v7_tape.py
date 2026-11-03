"""追加入力のツモ。入力時刻までに届いた履歴だけから種を特定する。"""
from __future__ import annotations
from dataclasses import asdict
from src.exchange_tsumo_tape import TsumoTape
from src.ama_tsumo_seed_v5 import identify, build_tapes, merged_known
from src.exchange_tsumo_tape import OVERLAP_MIN

VISIBLE = 3


class CausalTape:
    """共通番号の確定前や可視3組不足は欠測。未来原票を受け取らない。"""

    def __init__(self, table: object, inherit_counter: bool = False) -> None:
        self.tape = TsumoTape()
        self.table = table
        self.identify = identify
        self.last = float('-inf')
        self.cached: dict = {}
        self.inherit_counter = inherit_counter
        self.inheritance: list[dict] = []
        self.states = [None, None]
        self.counter_history: list[list] = [[], []]
        self.pinned = [None, None]

    def observe(self, stamp: float, queues: list, states: list | None = None,
                triggers: list | None = None) -> None:
        if stamp < self.last:
            raise ValueError('ツモの時刻逆転')
        self.last = stamp
        self.states = states or ['STABLE', 'STABLE']
        for side, queue in enumerate(queues):
            if self.inherit_counter and self.states[side] == 'CHAIN':
                if self.pinned[side] is None:
                    trigger = (triggers or [None, None])[side]
                    before = stamp if trigger is None else trigger
                    saved = next((r for r in reversed(self.counter_history[side]) if r['t'] < before), None)
                    self.pinned[side] = saved['counter'] if saved else self.tape.counter(side)
            else:
                self.pinned[side] = None
            # 入力の番号は固定するが、NEXT観測履歴は整合確認のため継続する。
            # 演出区間の観測まで捨てると、終了時の繰上がりを取りこぼす。
            was_aligned = self.tape.aligned
            restarts = [s.restarts for s in self.tape.sides]
            bases = [s.base for s in self.tape.sides]
            self.tape.observe(side, stamp, *queue)
            if self.inherit_counter and was_aligned and not self.tape.aligned:
                self.preserve_alignment(stamp, side, restarts, bases)
            if self.inherit_counter and self.states[side] == 'STABLE':
                self.counter_history[side].append(dict(t=stamp, counter=self.tape.counter(side)))

    def preserve_alignment(self, stamp: float, side: int, restarts: list, bases: list) -> None:
        """手数差だけで共通番号を失わない。矛盾・restart・補正時は引き継がない。"""
        tape = self.tape
        if restarts != [s.restarts for s in tape.sides] or bases != [s.base for s in tape.sides]:
            return
        history = dict(counters=bases, tapes=[s.own for s in tape.sides])
        if merged_known(history) is None:
            return
        overlap = set(range(bases[0], bases[0]+len(tape.sides[0].own))) & set(
            range(bases[1], bases[1]+len(tape.sides[1].own)))
        if len(overlap) >= OVERLAP_MIN:
            tape.aligned = True
            self.inheritance.append(dict(t=stamp, side=side, reason='確定済み共通番号・全履歴無矛盾'))

    def bundle(self, colors: list) -> dict:
        tape = self.tape
        if not tape.aligned:
            raise ValueError('共通ツモ番号未確定')
        history = dict(counters=[s.base for s in tape.sides], tapes=[s.own for s in tape.sides])
        known = merged_known(history)
        if known is None:
            raise ValueError('観測ツモ履歴の欠落または矛盾')
        signature = repr((history, colors))
        if signature not in self.cached:
            self.cached[signature] = self.identify(self.table, history, colors)
        ident = self.cached[signature]
        counters = [self.pinned[s] if self.pinned[s] is not None else tape.counter(s) for s in (0, 1)]
        if None in counters:
            raise ValueError('現在ツモ番号欠測')
        # 観測列の再開後は、連鎖前に固定した番号が保存範囲より古いことがある。
        # 未観測の過去を生成せず、通常の欠測として次の有効入力を待つ。
        if min(counters) < min(known):
            raise ValueError('連鎖前ツモ番号の観測履歴不足')
        tails = [[known[i] for i in range(c, max(known)+1)] for c in counters]
        # 未知の3組目は補完して可視と称さない。特定済みの種なら生成できる。
        if min(map(len, tails)) < VISIBLE and not ident.identified:
            raise ValueError('開始3組不足・種未特定')
        all_bundle = build_tapes(history, colors, ident, self.table)
        result = dict(all_bundle)
        result['tapes'] = [t[c-base:] for t, c, base in
                           zip(all_bundle['tapes'], counters, history['counters'])]
        if min(map(len, result['tapes'])) < 40:
            raise ValueError('種生成テープ末尾')
        result.update(counters=counters, identification=asdict(ident), evidence_time=self.last,
                      observed_history=history, input_known=[t[:VISIBLE] for t in result['tapes']])
        if self.inherit_counter:
            result['counter_policy'] = '連鎖中は発火直前番号を保持'
            result['counter_states'] = list(self.states)
        return result
