"""採用済み通常予測と隠し段上限を、試合境界を越えて最終得点へ照合する。"""
from __future__ import annotations
from typing import Any
from scripts.run_e23_multilanding_20260928 import ProgressTrace


class PredictionTrace(ProgressTrace):
    """監査用参照だけを保持し、予測器と確定盤面には書き込まない。"""

    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.hidden_chains: dict[tuple, Any] = {}
        self.hidden_engine: Any = None

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        super().__call__(overlay, inputs)
        self.hidden_engine = overlay._hidden_death
        if self.hidden_engine is not None:
            self.hidden_chains.update({(key[0], entry['chain'].side, key[2]): entry['chain']
                for key, entry in self.hidden_engine.entries.items()})

    def hidden_summary(self) -> dict:
        """撤回済み採用も除外せず、未確定を一致扱いしない。"""
        value = self.hidden_engine.summary()
        for row in value['rows']:
            if row['outcome'] != 'accepted':
                continue
            chain = self.hidden_chains[(row['game'], row['side'], row['chain_id'])]
            ready = (chain.score_ready_sec is not None and chain.end_signal_sec is not None
                and chain.end_confirmed is not False and chain.score_delta is not None)
            row['final_score'] = chain.score_delta if ready else None
            row['final_mismatch'] = chain.score_delta != row['maximum_score'] if ready else None
        accepted = [r for r in value['rows'] if r['outcome'] == 'accepted']
        value.update(final_mismatch=sum(r['final_mismatch'] is True for r in accepted),
            final_unresolved=sum(r['final_mismatch'] is None for r in accepted),
            accepted_chains=len({(r['game'], r['side'], r['chain_id']) for r in accepted}))
        return value


def counts(summary: dict) -> dict:
    """通常予測と死亡専用上限を同じ四母数で比較する。"""
    names = ('accepted', 'accepted_chains', 'next_mismatch', 'final_mismatch', 'final_unresolved')
    return {name: summary[name] for name in names}
