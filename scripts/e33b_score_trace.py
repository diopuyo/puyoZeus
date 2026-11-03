"""表示を作った評価時点の得点を記録し、保持中の予測を後知恵で更新しない。"""
from __future__ import annotations
import gzip
import json
from pathlib import Path
from typing import Any
import numpy as np
from scripts.run_e32 import BeliefTrace
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

OUT = Path('logs/e33b')
PREDICTION_SOURCES = ('S3_provisional', 'S3_landing')


def effective_scores(engine: Any, chains: list) -> list[dict]:
    """モデルが各候補へ適用する観測下限を先に適用してから平均する。"""
    result = []
    for chain in chains:
        entry = engine.active(chain) if engine is not None else None
        if entry:
            score = sum(v['weight']*max(v['score'], chain.formula_total or 0.,
                        chain.score_delta or 0.) for v in entry['options'])
        else:
            score = chain.provisional_score
        result.append(dict(side=chain.side, chain_id=chain.chain_id, score=float(score)))
    return result


def value_key(record: Any, value: dict) -> tuple:
    """複製された保持値も、評価時刻・出所・確率で元の計算へ対応付ける。"""
    return record.game_idx, record.exchange_id, value['source'], value['t_sec'], value['p1']


class ScoreTrace(BeliefTrace):
    """評価器を変更せず、表示に選ばれた評価の入力得点を外部へ記録する。"""
    def __init__(self, source: str, out: Path) -> None:
        super().__init__(source)
        self.stream = gzip.open(out/'score_rows.jsonl.gz', 'wt', encoding='utf-8')
        self.values: dict[tuple, list[dict]] = {}
        self.display_count = 0
        self.install_tracker()
        self.install_projection()

    def install_tracker(self) -> None:
        """S3の評価時点を保存し、再計算されない表示行にも同じ値を供給する。"""
        from src.exchange_event_tracker import ExchangeEventTracker
        original = ExchangeEventTracker._evaluate
        def traced(tracker: Any, event: Any, source: str, stamp: float) -> bool:
            record = tracker.current
            scores = effective_scores(getattr(tracker, 'hidden_row_belief', None),
                record.chains) if record is not None and source.startswith('S3') else []
            accepted = original(tracker, event, source, stamp)
            if accepted and record is not None and scores:
                self.values[value_key(record, record.values[-1])] = scores
            return accepted
        ExchangeEventTracker._evaluate = traced

    def install_projection(self) -> None:
        """候補加重で再評価した得点と、既存S3を保持した得点を区別する。"""
        from src.exchange_event_landing import ExchangeLandingProjection
        original = ExchangeLandingProjection._evaluate
        def traced(projection: Any, overlay: Any, snapshot: Any, latest: tuple,
                   incoming: list, hands: tuple, base: dict, stamp: float) -> dict:
            record = overlay.tracker.current or projection.death_record
            engine = overlay._prefire
            base_scores = self.values.get(value_key(record, base), [])
            scores = (effective_scores(engine, record.chains)
                if any(engine.active(c) for c in record.chains) else base_scores)
            value = original(projection, overlay, snapshot, latest, incoming, hands, base, stamp)
            if value['source'] == base['source'] and value['p1'] == base['p1']:
                scores = base_scores
            self.values[value_key(record, value)] = scores
            return value
        ExchangeLandingProjection._evaluate = traced

    def selected_value(self, overlay: Any, row: Any) -> tuple[list[dict], float | None]:
        """死亡保持・着弾保持・S3の順で、実際の出力確率と一致する値を探す。"""
        projection, tracker = overlay._landing_projection, overlay.tracker
        record = tracker.current or projection.death_record
        if record is None:
            return [], None
        candidates = [projection.death, projection.last]
        candidates.extend(reversed(record.values))
        for value in candidates:
            if value is None or value['source'] != row.source or value['p1'] != row.display_p1:
                continue
            key = value_key(record, value)
            if key in self.values:
                return self.values[key], value['t_sec']
        return [], None

    def observe_display(self, overlay: Any, row: Any) -> None:
        """実在する表示行だけを記録する。ウォームアップ更新は母数に含めない。"""
        scores, value_sec = self.selected_value(overlay, row)
        value = dict(frame=self.display_count, t_sec=row.t_sec, game=row.game_idx,
            source=row.source, value_sec=value_sec, scores=scores)
        self.stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n')
        self.display_count += 1

    def summary(self) -> dict:
        """採点真値は原票へ別記し、予測の保存値へ逆流させない。"""
        self.stream.close()
        return super().summary()


def baseline_directory(source: str) -> Path:
    """元のE32成果物を参照する。"""
    return Path('logs/e32/on/renders')/source/'on'


def prepare_cohort() -> dict:
    """従来の73連鎖から、E32の連鎖中の予測表示行を先に固定する。"""
    result = {}
    for source in SOURCES:
        root = baseline_directory(source)
        audit = json.loads((root/'snapshot_final_audit.json').read_text())['rows']
        accepted = {(r['game'], r['side'], r['chain_id']): r for r in audit
                    if r['accepted'] and r['error'] is not None}
        records = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        chains = {(r['game_idx'], c['side'], c['chain_id']): c for r in records for c in r['chains']}
        data, rows = np.load(root/'display.npz'), []
        for key, adoption in accepted.items():
            chain = chains[key]
            mask = ((data['game_idx'] == key[0]) & (data['t_sec'] >= chain['observed_sec']) &
                (data['t_sec'] < chain['end_signal_sec']) & np.isin(data['source'], PREDICTION_SOURCES))
            rows.extend(dict(frame=int(i), t_sec=float(data['t_sec'][i]), game=key[0],
                side=key[1], chain_id=key[2], actual=adoption['final_score']) for i in np.flatnonzero(mask))
        rows.sort(key=lambda r: (r['frame'], r['side'], r['chain_id']))
        save_json(OUT/'cohort'/f'{source}.json', rows)
        result[source] = dict(chains=len(accepted), chain_rows=len(rows),
            display_rows=len({r['frame'] for r in rows}))
    save_json(OUT/'COHORT.json', result)
    return result
