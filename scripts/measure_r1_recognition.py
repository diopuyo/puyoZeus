"""D3の同じ3,958置きを選び直さず、R1の残差・遅延・代理正解率を採点する。"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scripts._d3_candidates import load
from scripts._d3_inventory import SOURCES
from scripts._d3_measure import observations as d3_observations
from scripts.r1_measure_helpers import OUT, records, observations, contact, first_reflection, quantiles, lines, REPORT_FPS

PLACEMENT_COUNT = 3958
BASELINE_RESIDUAL = 412
MAX_ERROR_RATE = .01
REMAINING_RATIO = .5
FRAME_TOLERANCE = 1e-6


def score_event(event: dict, old: list, new: list, reference: np.ndarray) -> dict:
    """D3の保存行・基準フレーム・誤り座標を固定したまま両条件を採点する。"""
    side, index = event['side'], event['i']
    before, after = (rows[index]['sides'][side]['confirmed_board'] for rows in (old, new))
    mask = reference >= 0
    mask[0] = False
    off = np.asarray(before)
    on = np.asarray(after) if after is not None else np.full_like(off, -1)
    cells = event['cells'] if event['early'] else []
    return dict(source=event['source'], t_sec=event['t'], side=side,
        old_residual=sum(off[c['row'],c['col']] != c['reference'] for c in cells),
        new_residual=sum(on[c['row'],c['col']] != c['reference'] for c in cells),
        proxy_cells=int(mask.sum()), old_correct=int((off[mask] == reference[mask]).sum()),
        new_correct=int((on[mask] == reference[mask]).sum()), missing_board=after is None)


def measure_source(source: str, events: list[dict]) -> list[dict]:
    """保存行の時刻・試合が全一致することを先に確認し、同じ位置で比較する。"""
    old, new = load(source), records(source)
    assert len(old) == len(new), source
    assert all(a['t'] == b['t'] and a['game'] == b['game'] for a, b in zip(old, new)), source
    frozen, _ = d3_observations(source)
    native = observations(source)
    result = []
    for event in events:
        if event['source'] != source or event['status'] != '対応候補':
            continue
        reference = frozen[event['reference_frame'], event['side']]
        row = score_event(event, old, new, reference)
        landed = contact(event, native)
        off_first, on_first = first_reflection(event, old), first_reflection(event, new)
        row.update(contact_sec=landed, old_first=off_first, new_first=on_first)
        for name, stamp in (('old', off_first), ('new', on_first)):
            row[name+'_delay'] = max(0., (stamp-landed)*REPORT_FPS) if stamp is not None and landed is not None else None
        result.append(row)
    return result


def correction_counts() -> dict:
    """合図別の回数・セル数と、誤修正・確認不能を全5記録で集計する。"""
    counts, cells, statuses = Counter(), Counter(), Counter()
    for source in SOURCES:
        for row in lines(OUT/'capture'/source/'placement_signal_reconcile.jsonl'):
            if row.get('corrections'):
                counts[row['signal']] += 1
                cells[row['signal']] += len(row['corrections'])
        statuses.update(row['status'] for row in lines(OUT/'followup'/f'{source}.jsonl'))
    assert sum(cells.values()) == sum(statuses.values())
    total = sum(cells.values())
    return dict(events=dict(counts), cells=dict(cells), total=total, statuses=dict(statuses),
                error_rate=statuses['wrong']/total if total else None,
                worst_case_rate=(statuses['wrong']+statuses['unverified'])/total if total else None)


def summarize(rows: list[dict]) -> dict:
    """欠測を母数から落とさず、判定できない遅延は合格扱いしない。"""
    numbers = ('old_residual','new_residual','proxy_cells','old_correct','new_correct','missing_board')
    sums = {key: int(sum(row[key] for row in rows)) for key in numbers}
    delays = {mode: quantiles([r[mode+'_delay'] for r in rows if r[mode+'_delay'] is not None])
              for mode in ('old', 'new')}
    anchored = sum(r['contact_sec'] is not None for r in rows)
    complete = sum(r['old_delay'] is not None and r['new_delay'] is not None for r in rows)
    correction = correction_counts()
    assert len(rows) == PLACEMENT_COUNT and sums['old_residual'] == BASELINE_RESIDUAL, (len(rows), sums)
    same_delay = complete == anchored > 0 and all(
        np.isclose(delays['old'][key], delays['new'][key], atol=FRAME_TOLERANCE, rtol=0) for key in ('p50','p95'))
    gates = dict(a=sums['new_residual'] <= sums['old_residual']*REMAINING_RATIO,
        b=correction['error_rate'] is not None and correction['error_rate'] <= MAX_ERROR_RATE,
        c=same_delay, d=sums['new_correct'] >= sums['old_correct'])
    return dict(placements=len(rows), **sums, delays=delays, paired_delays=complete, anchored_delays=anchored,
                corrections=correction, gates=gates)


def main() -> None:
    """事前固定した母数の明細と合否を両方保存する。"""
    events = json.loads(Path('logs/d3/measurements.json').read_text())
    rows = [row for source in SOURCES for row in measure_source(source, events)]
    result = summarize(rows)
    (OUT/'RECOGNITION_ROWS.json').write_text(json.dumps(rows, ensure_ascii=False, default=int))
    (OUT/'RECOGNITION_METRICS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
