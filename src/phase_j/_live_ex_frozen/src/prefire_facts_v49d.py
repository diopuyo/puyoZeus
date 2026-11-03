"""死亡証明の入力契約へ渡す観測事実を、通知の到着ごとに蓄積する外部wrapper (v4.9d)。

観測指標は stateless が原則のため、状態 (盤面と得点の整合ガード・全消し帰属・連鎖物理照合・
攻撃照合の履歴) はここに閉じ込め、B3本体・台帳 (ConnectedLedger) は変更しない。
通知は時刻順に1回ずつ `observe` へ渡す。未来の観測は参照しない。
"""
from __future__ import annotations

from typing import Any
import numpy as np
from src.phase_j._live_ex_frozen.src.board import Board
from src.phase_j._live_ex_frozen.src.exchange_landing_safety import ColorScoreGuard
from src.phase_j._live_ex_frozen.src.exchange_native_chain import completion
from src.phase_j._live_ex_frozen.src.b3_score_ledger import KIND_ALL_CLEAR, FORMULA_DROP_SLACK
from src.phase_j._live_ex_frozen.src.prefire_proof_contract import (
    ProofFacts, ATTACK_RECONCILE_MISMATCH, ATTACK_CANDIDATE_MISSING,
    ATTACK_ALL_CLEAR_AMBIGUOUS, ATTACK_CHAIN_OPEN)

SIDES = (0, 1)
COLOR_UNKNOWN = 10
# 台帳の連鎖記録と送り候補の発火時刻を結び付ける許容 (式の段間隔 1.0〜1.5秒 + 読み取り遅れ)。
CHAIN_LINK_SEC = 1.5
# 全消し後の盤面を確認する窓 (連鎖終了からこの秒数以内に空盤面なら全消しとみなす)。
ALL_CLEAR_WINDOW_SEC = 10.0
# 物理照合の許容: 式の得点に同時に乗りうる落下点 (台帳の定数を再利用)。
PHYSICS_SLACK = FORMULA_DROP_SLACK
NATIVE_CACHE_SIZE = 4096


class FactsTracker:
    """通知ごとの観測を蓄積し、受け側ごとの ProofFacts を作る。"""

    def __init__(self, native: Any) -> None:
        self.native = native
        self.reset()

    def reset(self) -> None:
        self.guards = [ColorScoreGuard(), ColorScoreGuard()]
        self.guard_pass_t: list[float | None] = [None, None]
        self.tainted: set[tuple[int, int]] = set()
        self.verdicts: dict[tuple[int, float], dict] = {}
        self.physics: list[dict | None] = [None, None]
        self.seen_events: set[tuple] = set()
        self.native_cache: dict[bytes, dict | None] = {}
        self.placements = [0, 0]
        self.slide_prev = [False, False]
        self.counters = dict(physics_checked=0, physics_mismatch=0, physics_unknown=0,
                             all_clear_events=0, empty_board_taints=0)

    # ---------- 通知ごとの蓄積 ----------

    def observe(self, state: Any, raw: list[dict], stamp: float) -> None:
        """B3状態の更新後に呼ぶ。盤面ガード・全消し・連鎖物理照合を更新する。"""
        core = state.sl.core
        self._placements(state)
        self._guards(state, stamp)
        self._all_clear_events(state, core)
        self._empty_boards(state, core, stamp)
        for side in SIDES:
            self._physics(side, raw[side], stamp)

    def _placements(self, state: Any) -> None:
        """NEXTが動く立ち上がり = 1手の置き終わり。"""
        for i, side in enumerate(state.sides):
            if side.slide and not self.slide_prev[i]:
                self.placements[i] += 1
            self.slide_prev[i] = bool(side.slide)

    def _guards(self, state: Any, stamp: float) -> None:
        for i, side in enumerate(state.sides):
            if side.state != 'STABLE' or side.board is None or side.board_t != stamp:
                continue
            board = Board.from_list(np.asarray(side.board).astype(int).tolist())
            self.guards[i].observe(board, tuple(side.queue), side.score, stamp)
            if self.guards[i].reason is None:
                self.guard_pass_t[i] = stamp

    def _open_or_last_chain(self, core: Any, side: int, stamp: float) -> Any:
        chain = core.st[side].chain
        if chain is not None:
            return chain
        done = [c for c in core.chains if c.side == side and c.t_end is not None
                and stamp - c.t_end <= ALL_CLEAR_WINDOW_SEC]
        return done[-1] if done else None

    def _all_clear_events(self, state: Any, core: Any) -> None:
        snap = state.sl_snap
        for event in (snap['events'] if snap else ()):
            if event.kind == 'score_attr' and event.get('kind') == KIND_ALL_CLEAR:
                chain = self._open_or_last_chain(core, event.side, state.t)
                if chain is not None:
                    self.tainted.add((chain.side, chain.chain_id))
                self.counters['all_clear_events'] += 1

    def _empty_boards(self, state: Any, core: Any, stamp: float) -> None:
        """全消しは盤面が空になる。直近の連鎖の予告を、全消しボーナス帰属が曖昧な予告として記録する。"""
        for i, side in enumerate(state.sides):
            if side.state != 'STABLE' or side.board is None or side.board_t != stamp:
                continue
            if not np.any(side.board):
                chain = self._open_or_last_chain(core, i, stamp)
                key = (i, chain.chain_id) if chain is not None else None
                if key is not None and key not in self.tainted:
                    self.tainted.add(key)
                    self.counters['empty_board_taints'] += 1

    def _solve(self, board: list[int]) -> dict | None:
        raw = bytes(board)
        if raw not in self.native_cache:
            if len(self.native_cache) >= NATIVE_CACHE_SIZE:
                self.native_cache.pop(next(iter(self.native_cache)))
            self.native_cache[raw] = (None if COLOR_UNKNOWN in board else completion(board, self.native))
        return self.native_cache[raw]

    def _physics(self, side: int, raw: dict, stamp: float) -> None:
        """連鎖前の盤面を物理で解いた得点が、報告された式の得点と食い違えば、その側の盤面を疑う。"""
        event = raw.get('chain_event')
        if not event:
            return
        ns = event['namespace']
        key = (side, ns['trigger_sec'], ns['chain_count'], ns['total_score'])
        if key in self.seen_events:
            return
        self.seen_events.add(key)
        board = [int(c) for c in np.asarray(ns['before_board']['board']['array']).ravel()]
        solved = self._solve(board)
        self.counters['physics_checked'] += 1
        if solved is None:
            self.counters['physics_unknown'] += 1
            return
        prefix = solved['prefix']
        physical = prefix[min(int(ns['chain_count']), len(prefix))-1] if prefix else 0
        if abs(int(ns['total_score'])-physical) > PHYSICS_SLACK:
            self.counters['physics_mismatch'] += 1
            self.physics[side] = dict(trigger=ns['trigger_sec'], reported=int(ns['total_score']),
                                      physical=int(physical), seen=stamp)

    # ---------- 攻撃の照合 ----------

    def refresh_verdicts(self, chains: list[dict | None], reconciliation: list[dict | None],
                         core: Any) -> None:
        """送り候補の総量と台帳の反映量を、候補の連鎖ごとに記録する。

        候補の最終段の後に新しい連鎖が始まったら凍結する (新しい連鎖の生成を、古い候補の反映量に混ぜない)。
        """
        for side in SIDES:
            chain, detail = chains[side], reconciliation[side]
            if chain is None or detail is None or not detail['times']:
                continue
            key, end = (side, chain['started']), max(detail['times'])
            if self._newer_chain(core, side, end):
                continue
            self.verdicts[key] = dict(total=detail['total'], reflected=detail['reflected'],
                                      finished=bool(chain['finished']), start=chain['started'], end=end)

    @staticmethod
    def _newer_chain(core: Any, side: int, end: float) -> bool:
        records = [c for c in core.chains if c.side == side]
        if core.st[side].chain is not None:
            records.append(core.st[side].chain)
        return any(c.t_start > end+CHAIN_LINK_SEC for c in records)

    def _ledger_chain(self, core: Any, side: int, chain_id: int) -> Any:
        found = [c for c in core.chains if c.side == side and c.chain_id == chain_id]
        if core.st[side].chain is not None and core.st[side].chain.chain_id == chain_id:
            found.append(core.st[side].chain)
        return found[-1] if found else None

    def _verdict_for(self, side: int, record: Any) -> dict | None:
        """台帳の連鎖記録を含む期間の送り候補。台帳が1つの候補を複数の連鎖へ分けた場合も同じ候補へ結ぶ。"""
        near = [(abs(v['start']-record.t_start), v) for (s, _), v in self.verdicts.items()
                if s == side and v['start']-CHAIN_LINK_SEC <= record.t_start <= v['end']+CHAIN_LINK_SEC]
        return min(near, key=lambda item: item[0])[1] if near else None

    def attack_reason(self, core: Any, receiver: int) -> str | None:
        """受け側の確定予告の出どころ (送り側の連鎖) がすべて照合済みなら None。"""
        for batch in core.batches[receiver]:
            if not batch.confirmed or batch.n <= 0:
                continue
            record = self._ledger_chain(core, batch.src_side, batch.chain_id)
            if record is None:
                return ATTACK_CANDIDATE_MISSING
            if (batch.src_side, batch.chain_id) in self.tainted:
                credit = core.all_clear[batch.src_side] if getattr(core, 'defer_all_clear', False) else None
                repaired = credit is not None and batch.chain_id in credit.recognized_chain_ids and not credit.uncertain_points
                if not repaired:
                    return ATTACK_ALL_CLEAR_AMBIGUOUS
                self.counters['all_clear_guard_bypassed'] = self.counters.get('all_clear_guard_bypassed', 0)+1
            verdict = self._verdict_for(batch.src_side, record)
            if verdict is None:
                return ATTACK_CANDIDATE_MISSING
            if not verdict['finished']:
                return ATTACK_CHAIN_OPEN
            if verdict['total'] != verdict['reflected']:
                return ATTACK_RECONCILE_MISMATCH
        return None

    # ---------- 受け側の事実 ----------

    def _physics_taint(self, core: Any, side: int, stamp: float) -> bool:
        """不一致だった連鎖が終わり、その後に盤面ガードを通る観測が来るまで盤面を疑う。"""
        mark = self.physics[side]
        if mark is None:
            return False
        ended = core.st[side].chain is None and any(
            c.side == side and c.t_end is not None and c.t_start >= mark['trigger']-CHAIN_LINK_SEC
            for c in core.chains)
        end_t = max((c.t_end for c in core.chains if c.side == side and c.t_end is not None
                     and c.t_start >= mark['trigger']-CHAIN_LINK_SEC), default=None)
        passed = self.guard_pass_t[side]
        if ended and end_t is not None and passed is not None and passed > end_t:
            self.physics[side] = None
            return False
        return True

    def board_resting(self, board: np.ndarray) -> bool:
        """STABLEの確定盤面は静止盤面のはず。連鎖を含むなら発火直前の盤面で、受け側の返しが始まる。"""
        solved = self._solve([int(c) for c in np.asarray(board).ravel()])
        return solved is not None and solved['chain_count'] == 0

    def facts(self, side: int, state: Any, core: Any, value: dict, active: list[bool]) -> ProofFacts:
        """受け側 `side` の事実を作る。証明の対象かどうか (予告の有無) は ProofFacts.pending が持つ。"""
        receiver, stamp = state.sides[side], state.t
        board = value['boards'][side]
        unknown = COLOR_UNKNOWN in board
        current = receiver.state == 'STABLE' and receiver.board_t == stamp
        resting = (not unknown) and self.board_resting(np.asarray(board))
        opponent_open = core.st[1-side].chain is not None
        return ProofFacts(
            side=side, pending=int(value['pending'][side]),
            confirmed=int(value['confirmed_observed'][side]),
            unknown_board=unknown,
            receiver_chain_open=core.st[side].chain is not None,
            opponent_unfinished=bool(any(active)) or opponent_open,
            receiver_current=current, board_resting=resting,
            guard_reason=self.guards[side].blocker(stamp),
            attack_reason=self.attack_reason(core, side),
            physics_mismatch=self._physics_taint(core, side, stamp),
            origin_known=value['margin_origin_sec'] is not None)
