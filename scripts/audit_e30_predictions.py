"""候補の予測時点と実際の最終得点を、採否へ逆流させず照合する。"""
from __future__ import annotations
from typing import Any
from scripts.run_e23_multilanding_20260928 import ProgressTrace


class CandidateTrace(ProgressTrace):
    """試合境界をまたいで監査対象の連鎖参照だけを保つ。"""

    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.chains: dict[tuple, Any] = {}
        self.engine: Any = None

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        super().__call__(overlay, inputs)
        self.engine = overlay._prefire
        if self.engine is not None:
            self.chains.update({(e['audit']['game'], e['chain'].side, identity): e['chain']
                for identity, e in self.engine.entries.items()})

    def summary(self) -> dict:
        """初回式一致時と最後に残った候補の双方を照合し、生存者だけの誤差にしない。"""
        value = self.engine.summary()
        for row in value['rows']:
            chain = self.chains[(row['game'], row['side'], row['chain_id'])]
            ready = (chain.score_ready_sec is not None and chain.end_signal_sec is not None
                and chain.end_confirmed is not False and chain.score_delta is not None)
            row['final_score'] = chain.score_delta if ready else None
            predictions = [r for r in row['observations'] if r['candidates']]
            row['remaining'] = (row['observations'][-1] if row['observations'] else row['initial'])['candidates']
            for name, prediction in (('first', predictions[0] if predictions else None),
                                     ('last', predictions[-1] if predictions else None)):
                row[name+'_mean_score'] = prediction['mean_score'] if prediction else None
                row[name+'_error'] = prediction['mean_score']-chain.score_delta if ready and prediction else None
            row['became_zero'] = bool(row['initial']['candidates'] and not row['remaining'])
        return value
