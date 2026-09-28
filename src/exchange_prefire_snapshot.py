"""一つの発火前観測盤面を段別得点で検証する予測層。"""
from __future__ import annotations
from typing import Any
from src.exchange_prefire_candidates import PrefireCandidates, completion, statistics
from src.prefire_snapshot_reader import validate_snapshot
from src.chain_detector import CHAIN_MECHANISM_FORMULA_READ


class PrefireSnapshot(PrefireCandidates):
    """配置列挙を行わず、受理・棄却・撤回を全発火で記録する。"""

    def observe_colors(self, sides: tuple) -> None:
        """確定色の観測と予測専用の画像入力を分けて保存する。"""
        self.restore()
        super().observe_colors(sides)
        self.snapshots = {f'{i+1}P': getattr(s, 'prefire_snapshot', None) for i, s in enumerate(sides)}

    def reset(self) -> None:
        """旧試合の公開予測を戻してから窓と試合色を破棄する。"""
        self.restore()
        super().reset()

    def restore(self) -> None:
        """既存E27の更新前に独立予測を戻し、前フレームの予測を入力にしない。"""
        for entry in self.entries.values():
            if entry['published'] is not None:
                chain = entry['chain']
                chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board = entry['original']
                entry['published'] = None

    def seed(self, chain: Any, event: Any, history: list, game: int, elapsed: float) -> None:
        """1段目の式が一致するまで予測を一度も公開しない。"""
        if chain.chain_id in self.entries:
            return
        saved = next((s for s in reversed(history) if s.t_sec < chain.trigger_sec), None)
        origin = getattr(event, 'before_board', None)
        origin = origin if origin is not None else saved.board if saved else None
        snapshot = self.snapshots.get(chain.side)
        reason, board = validate_snapshot(origin, snapshot, self.evidence.active())
        option = completion(board, self.simulator) if board is not None else None
        if reason is None and (event.mechanism != CHAIN_MECHANISM_FORMULA_READ or event.chain_count != 1):
            reason = 'missing_first_formula'
        if reason is None and (not option['prefix'] or option['prefix'][0] != event.total_score):
            reason = 'first_score'
        options = [] if reason else [dict(option, weight=1)]
        row = dict(game=game, side=chain.side, chain_id=chain.chain_id, trigger_sec=chain.trigger_sec,
            reason=reason, accepted=reason is None, withdrawn=False, snapshot=snapshot, used=False, uses=0,
            predicted_score=option['score'] if options else None, observations=[])
        self.audit.append(row)
        self.entries[chain.chain_id] = dict(chain=chain, options=options, elapsed=elapsed,
            original=(chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board),
            published=None, last=None, audit=row, stats=statistics(options, elapsed))

    def observe(self, overlay: Any, result: Any, stamp: float) -> None:
        """段ごとの累積得点不一致で永久撤回し、その時点のE27予測へ戻す。"""
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
                    count, score = key
                    prefix = entry['options'][0]['prefix']
                    previous_count = entry['last'][0] if entry['last'] is not None else 0
                    good = previous_count <= count <= previous_count+1 and 0 < count <= len(prefix) and prefix[count-1] == score
                    entry['audit']['observations'].append(dict(t_sec=stamp, count=count, score=score, match=good))
                    if not good:
                        entry['options'] = []
                        entry['audit']['withdrawn'] = True
                        entry['audit']['withdraw_sec'] = stamp
                        entry['audit']['withdraw_reason'] = 'stage_gap' if count > previous_count+1 else 'stage_score'
                    entry['last'] = key
                    self.revision += 1
            self._publish(entry)

    def provenance(self, chains: list) -> list[dict]:
        """CSV・画面・原票に同一の予測由来を渡す。"""
        return [dict(row, method='prefire_snapshot', label='発火前盤面から予測')
                for row in super().provenance(chains)]
