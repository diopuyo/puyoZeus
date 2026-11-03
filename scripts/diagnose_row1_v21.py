"""PC2の保存内部原票を分解する。認識・評価を再実行した結果とは区別する。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path('D:/puyo_analyzer/share/tasks/done/el_row1_guard_v2_short')
GRID_CELLS = 13*6
EARLY_END, GAME36_START = 6000, 66000


def read_rows(path: Path) -> dict[int, dict]:
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return {r['frame']: r for r in map(json.loads, stream)}


def diagnose(segment: str) -> dict:
    rows = {m: read_rows(ROOT/segment/m/'internal_history.jsonl.gz') for m in ('off', 'on')}
    differences = {}
    for side in ('p1', 'p2'):
        keys = set(next(iter(rows['off'].values()))['result'][side]) - {'prob_board'}
        for key in keys:
            frames = [f for f, a in rows['off'].items()
                      if a['result'][side][key] != rows['on'][f]['result'][side][key]]
            if frames:
                differences[f'{side}.{key}'] = dict(count=len(frames), first=frames[0])
    trace = []
    if segment == 'game36':
        for f in range(70765, 70802):
            r = rows['off'][f]
            s, v = r['sides']['2P'], r['result']['p2']
            g = s['guard']
            trace.append(dict(frame=f, signals=g['signals'],
                before=g['before'][1][2] if g['before'] else None,
                raw=g['raw_cnn_board'][1][2], state=v['state'],
                next=v['next_pair'], dnext=v['dnext_pair'], landing=v['landing_diag']))
    updates = {m: {r['frame']: r for r in map(json.loads,
        (ROOT/segment/m/'b3_updates.jsonl').read_text().splitlines())} for m in ('off', 'on')}
    shared = sorted(updates['off'].keys() & updates['on'].keys())
    raw_diff = [f for f in shared if updates['off'][f].get('raw_probability') !=
                updates['on'][f].get('raw_probability')]
    focused = {str(f): {m: dict(
        side_changes={s: [k for k in rows['off'][f]['result'][s] if k != 'prob_board'
            and rows['off'][f]['result'][s][k] != rows['on'][f]['result'][s][k]]
            for s in ('p1', 'p2')},
        latest_prediction={k: v for k, v in updates[m][max(x for x in updates[m] if x <= f)].items()
                           if k in ('frame', 'raw_probability')}) for m in ('off', 'on')}
        for f in (70721, 70734) if f in rows['off']}
    return dict(differences=differences, green_trace=trace,
        focused=focused,
        note='prob_boardはelのpackがreprで保存しており、アドレス差を実体差に数えない。',
        probability_diff=dict(count=len(raw_diff), first=raw_diff[:1]),
        update_only={m: sorted(updates[m].keys()-updates[n].keys())
                     for m, n in (('off', 'on'), ('on', 'off'))},
        guard_replay=replay(rows))


def replay(rows: dict) -> dict:
    """保存されたガード入力をそのまま再投下。動画認識の再走とは区別する。"""
    from src.board import Board
    from src.row1_guard import Row1Guard
    guards = {s: Row1Guard() for s in ('1P', '2P')}
    outside = denominator = v2_mismatch = 0
    old_guards = {s: Row1Guard() for s in guards}
    green, purple, phantom = [], [], []
    for f, r in rows['off'].items():
        for side, guard in guards.items():
            if f < EARLY_END or f >= GAME36_START:
                denominator += GRID_CELLS-1
            data = r['sides'][side]['guard']
            b = Board.from_list(data['before']) if data['before'] is not None else None
            raw = Board.from_list(data['raw_cnn_board']) if data['raw_cnn_board'] is not None else None
            v = r['result']['p1' if side == '1P' else 'p2']
            signals = dict(in_chain=data['signals']['formula'], next_moved=data['signals']['next_moved'],
                           ojama_falling=data['signals']['ojama_falling'])
            if b is None:
                guard.reset()
                old_guards[side].reset()
                continue
            old = old_guards[side].step(b, raw, r['time_sec'], **signals)
            actual = rows['on'][f]['sides'][side]['guard']['after']
            v2_mismatch += old._grid.tolist() != actual
            out = guard.step(b, raw, r['time_sec'], **signals,
                             settled=v['state'] == 'STABLE' and v['landing_diag'] is not None)
            diff = out._grid != b._grid
            diff[1, 2] = False
            if f < EARLY_END or f >= GAME36_START:
                outside += int(diff.sum())
            if side == '2P':
                top = int(out.get(1, 2))
                if 70778 <= f <= 70810 and top == 3:
                    green.append(f)
                if 68682 <= f <= 68720 and top == 5:
                    purple.append(f)
                if 69864 <= f <= 69930:
                    phantom.append(top)
    return dict(v2_saved_output_mismatches=v2_mismatch, outside=outside, outside_cells=denominator,
        green_first=green[:1], purple_first=purple[:1], phantom_nonempty=sum(v != 0 for v in phantom),
        phantom_frames=len(phantom))


def main() -> None:
    result = {s: diagnose(s) for s in ('early', 'game36')}
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
