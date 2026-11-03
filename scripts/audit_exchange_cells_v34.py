"""v3.3代表盤の全誤読セルを候補支持・順位別に記録する事後監査。"""
from __future__ import annotations

from collections import Counter
import json
from scripts.audit_exchange_counts_v33 import OUT as V33
from scripts.exchange_review_v32_data import BASE, source, video_path
from scripts.exchange_counts_v33_collect import ImageSources, extract_sources
from scripts.exchange_review_io import save
from scripts.exchange_review_v3_fit import write_csv
from scripts.replay_exchange_projection_shadow import updates
from src.exchange_count_prediction import cell_probabilities

OUT = BASE/'video_80_g7_v3_4'
GAME, COLS = 7, 6


def collect() -> list[dict]:
    path = OUT/'inputs_80.json'
    if path.exists():
        return json.loads(path.read_text())
    items = json.loads((BASE/'video_80_g7_v3_2/events.json').read_text())['origins']
    targets = {r['fire']['t_sec']: r for r in items}
    images, rows = ImageSources(video_path(80)), []
    if images.cap is None:
        raise FileNotFoundError('動画80の元映像が必要です')
    for args in updates(source(80)/'inputs.jsonl.gz', {GAME}):
        if args[3] not in targets:
            continue
        item = targets[args[3]]
        raw = (args[0].p1, args[0].p2)[item['side']]
        sources = extract_sources(raw, args[3], item['fire']['trigger_sec'], item['side'], images)
        rows.append(dict(event_id=item['event_id'], sources=sources))
    images.cap.release()
    save(path, rows)
    return rows


def main() -> None:
    inputs = {r['event_id']: r for r in collect()}
    truths = json.loads((V33/'truth_probe.json').read_text())
    predictions = json.loads((V33/'predictions.json').read_text())['rows']
    predictions = {r['event_id']: r for r in predictions}
    scores = {r['event_id']: r['variants']['board_candidates']
              for r in json.loads((V33/'scored_events.json').read_text())}
    policy = json.loads((V33/'policy.json').read_text())
    rows = []
    for truth in truths:
        if 'duplicate_of' in truth:
            continue
        eid = truth['event_id']
        values = predictions[eid]['variants']['board_candidates']
        best = max(values, key=lambda v: v['probability'])
        if all(scores[eid][key] for key in ('event_stage_match', 'sent_match', 'remaining_match')):
            continue
        probs = cell_probabilities(inputs[eid]['sources'], [1, 2, 3, 5], policy['image_weight'])
        for cell, (predicted, correct) in enumerate(zip(best['origin'], truth['oracle_origin'])):
            if predicted == correct:
                continue
            votes = probs[cell]
            rank = 1+sum(p > votes[correct] for p in votes.values()) if correct in votes else None
            support = [i+1 for i, v in enumerate(values) if v['origin'][cell] == correct]
            rows.append(dict(event_id=eid, row=cell//COLS, col=cell%COLS,
                predicted=predicted, correct=correct, color_rank=rank,
                reason='候補にない' if rank is None else '候補にはあるが順位が低い',
                probability=votes.get(correct, 0.), cell_candidates=votes,
                board_candidate_rank=min(support, default=None),
                source_votes=[dict(kind=s['kind'], t=s['t'], color=s['board'][cell]) for s in inputs[eid]['sources']]))
    save(OUT/'cell_audit.json', dict(rows=rows, counts=dict(Counter(r['reason'] for r in rows))))
    write_csv(OUT/'cell_audit.csv', rows)
    print(Counter(r['reason'] for r in rows))
    for row in rows:
        print({k: v for k, v in row.items() if k != 'source_votes'})


if __name__ == '__main__':
    main()
