"""保存Jの返却状態と起点交替を集計する。現在runや物理終了の代用にはしない。"""
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'video38_split_tail_candidate_v22' / 'atomic_journal.jsonl'
FIRST, LAST, SIDE = 34702, 36298, '2P'


def main() -> None:
    started, count, segments, last = time.perf_counter(), 0, [], None
    with SOURCE.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('kind') != 'step' or row['side'] != SIDE or not FIRST <= row['frame_idx'] <= LAST:
                continue
            count += 1
            returned = row.get('returned') or {}
            origin = returned.get('active_origin') or {}
            generation = row['generation_after']
            key = (returned.get('state'), origin.get('object_id'), origin.get('trigger_sec'),
                   origin.get('mechanism'), generation['reset_epoch'], generation['action_revision'])
            if key == last:
                segments[-1]['last'] = row['frame_idx']
                segments[-1]['rows'] += 1
                continue
            segments.append(dict(first=row['frame_idx'], last=row['frame_idx'], rows=1,
                state=key[0], origin_id=key[1], trigger=key[2], mechanism=key[3], generation=generation))
            last = key
    result = dict(source=str(SOURCE), side=SIDE, rows=count, segments=segments,
                  same_run_live_verified=False, physical_end_verified=False, quality_gate_clear=False,
                  seconds=time.perf_counter() - started)
    with (ROOT / 'RETURNED_ORIGIN_TIMELINE_v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(rows=count, segments=len(segments), first=segments[:2], last=segments[-8:],
                          seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
