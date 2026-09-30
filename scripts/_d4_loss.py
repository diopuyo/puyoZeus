"""qの固定6526行を試合・連鎖・時間帯へ加法分解する。"""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from scripts.aggregate_e3_exchange_eval_20260926 import outcomes, FPS, PROBABILITY_EPSILON
from scripts.r1_measure_helpers import lines

OUT = Path('logs/d4')
Q = 'q_7gc4TgFig'
BIN_SEC = 5.
COHORT_SIZE = 6526


def directory(mode: str, source: str = Q) -> Path:
    """R1の元評価出力を読み取り専用で選ぶ。"""
    suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
    return Path('logs/r1')/mode/suffix


def save(name: str, value: object) -> None:
    """D4の出力だけを保存する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist()), encoding='utf-8')


def event_at(events: list[dict], stamp: float, game: int) -> dict:
    """閉鎖後の保持値にも直前イベントの由来を付ける。"""
    eligible = [e for e in events if e['game_idx'] == game and e['trigger_sec'] <= stamp]
    if not eligible:
        return dict(exchange=None, chains=[])
    event = eligible[-1]
    return dict(exchange=event['exchange_id'], chains=[c['chain_id'] for c in event['chains'] if c['observed_sec'] <= stamp])


def build_rows() -> list[dict]:
    """既存M3と同じ勝敗ラベル・clipで行損失を求める。"""
    data = {m: dict(np.load(directory(m)/'display.npz')) for m in ('off', 'on')}
    events = {m: list(lines(directory(m)/'events.jsonl')) for m in data}
    np.testing.assert_array_equal(data['off']['t_sec'], data['on']['t_sec'])
    windows, provenance = outcomes(Q)
    save('q_labels.json', provenance)
    result = []
    for match, window in enumerate(windows):
        frames = np.rint(data['off']['t_sec']*FPS).astype(int)
        indexes = np.flatnonzero((frames >= window['start']) & (frames < window['end']))
        for i in indexes:
            stamp, game = float(data['off']['t_sec'][i]), int(data['off']['game_idx'][i])
            row = dict(i=int(i), t=stamp, match=match, game=game, y=int(window['winner']=='1P'))
            for mode, values in data.items():
                p = float(np.clip(values['display_p1'][i], PROBABILITY_EPSILON, 1-PROBABILITY_EPSILON))
                row[mode] = dict(p=p, loss=-np.log(p if row['y'] else 1-p),
                    source=str(values['source'][i]), **event_at(events[mode], stamp, game))
            row['delta'] = row['on']['loss']-row['off']['loss']
            row['bin'] = int(stamp//BIN_SEC)
            result.append(row)
    assert len(result) == COHORT_SIZE
    return result


def aggregate(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    """各区分の総損失差を全体母数で割り、寄与の総和を保つ。"""
    groups: dict = defaultdict(list)
    for row in rows:
        key = tuple(row[f] if f in row else tuple(row['off']['chains']) for f in fields)
        groups[key].append(row)
    result = []
    for key, values in groups.items():
        delta = float(sum(r['delta'] for r in values))
        peak = max(values, key=lambda r: r['delta'])
        result.append(dict(zip(fields, key), n=len(values), start=values[0]['t'], end=values[-1]['t'],
            delta_sum=delta, contribution=delta/COHORT_SIZE, delta_mean=delta/len(values),
            off=float(np.mean([r['off']['loss'] for r in values])),
            on=float(np.mean([r['on']['loss'] for r in values])), peak=peak))
    return sorted(result, key=lambda r: -r['delta_sum'])


def main() -> None:
    """上位区間の開始と最大悪化行へ直前修正を対応付ける。"""
    rows = build_rows()
    corrections = [r for r in lines(Path('logs/r1/capture')/Q/'placement_signal_reconcile.jsonl') if r['corrections']]
    intervals = aggregate(rows, ('match', 'bin'))
    for interval in intervals:
        for key, stamp in (('preceding', interval['start']), ('peak_preceding', interval['peak']['t'])):
            prior = [r for r in corrections if r['t_sec'] <= stamp]
            last = prior[-1] if prior else None
            interval[key] = None if last is None else dict(t=last['t_sec'], age=stamp-last['t_sec'],
                signal=last['signal'], side=last['side'], cells=last['corrections'])
        interval['within'] = [r for r in corrections if interval['start'] <= r['t_sec'] <= interval['end']]
    save('q_rows.json', rows)
    save('q_intervals.json', intervals)
    save('q_top10.json', intervals[:10])
    seconds = [dict(r, second=int(r['t'])) for r in rows]
    fine = aggregate(seconds, ('match', 'second'))[:10]
    for interval in fine:
        prior = [r for r in corrections if r['t_sec'] <= interval['start']]
        last = prior[-1] if prior else None
        interval['preceding'] = None if last is None else dict(t=last['t_sec'],
            age=interval['start']-last['t_sec'], signal=last['signal'], side=last['side'])
    save('q_top10_seconds.json', fine)
    save('q_matches.json', aggregate(rows, ('match', 'game')))
    save('q_chains.json', aggregate(rows, ('match', 'chains')))
    summary = dict(n=len(rows), off=float(np.mean([r['off']['loss'] for r in rows])),
        on=float(np.mean([r['on']['loss'] for r in rows])), delta_sum=sum(r['delta'] for r in rows),
        worsened=sum(r['delta']>0 for r in rows), improved=sum(r['delta']<0 for r in rows),
        top10_sum=sum(r['delta_sum'] for r in intervals[:10]))
    save('q_summary.json', summary)
    print(summary)
    print([(r['start'], r['end'], r['delta_sum'], r['preceding']) for r in intervals[:10]])


if __name__ == '__main__':
    main()
