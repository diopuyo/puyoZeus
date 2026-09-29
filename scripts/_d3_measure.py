"""D3の独立色一致で、合図候補・保存書込み・残差を対応付ける。"""
from __future__ import annotations

from collections import Counter
import csv
import json
import os
from typing import Any
import numpy as np
from scripts._d3_candidates import load
from scripts._d3_inventory import OUT, SOURCES

UNKNOWN = -1
COLORS = (1, 2, 3, 4, 5)
VISIBLE_ROWS = 12
COLS = 6
ACCEPT_FRAMES = 8
FPS = 30
TIME_EPSILON = 1e-6


def observations(source: str) -> tuple[dict, float]:
    """単独CNNと単独HSVの一致だけを代理基準にする。"""
    name = 'zenchi' if source == 'review' else source
    result: dict = {}
    fps = 0.0
    with (OUT/f'{name}_observations.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            fps = row['fps']
            for side, data in row['sides'].items():
                cnn, hsv = np.asarray(data['cnn']), np.asarray(data['hsv'])
                grid = np.full((VISIBLE_ROWS+1, COLS), UNKNOWN, dtype=np.int8)
                grid[1:] = np.where(cnn == hsv, cnn, UNKNOWN).reshape(VISIBLE_ROWS, COLS)
                result[(row['frame'], int(side))] = grid
    return result, fps


def point(obs: dict, stamp: float, side: int, fps: float) -> np.ndarray | None:
    """存在しない原フレームを補間しない。"""
    return obs.get((round(stamp*fps), side))


def reference(event: dict, obs: dict, fps: float) -> tuple[np.ndarray | None, int]:
    """式は原動画の1〜4フレーム前で、色ぷよが最多のフレームを採る。"""
    frame, side = round(event['t']*fps), event['side']
    offsets = range(1, 5) if event['type'] == 'formula' else (0,)
    values = [(obs[(frame-d, side)], frame-d) for d in offsets if (frame-d, side) in obs]
    if not values:
        return None, frame
    return max(values, key=lambda item: (int(np.isin(item[0], COLORS).sum()),
                                         int((item[0] >= 0).sum()), item[1]))


def supported(grid: np.ndarray, cells: list) -> bool:
    """少なくとも一つの追加色ぷよが底か既存の積みに接している。"""
    for r, c in cells:
        below = r+1
        while below <= VISIBLE_ROWS and [below, c] in cells:
            below += 1
        if below > VISIBLE_ROWS or grid[below, c] > 0:
            return True
    return False


def plausible_write(write: dict, obs: dict, fps: float, side: int,
                    baseline: np.ndarray) -> dict | None:
    """前周期画像では空だった位置への、画像支持付き色書込みを対応する。"""
    visual = point(obs, write['t'], side, fps)
    if visual is None:
        return None
    additions = [[r, c] for r, c in write['diff'] if baseline[r, c] == 0
        and write['board'][r][c] in COLORS and visual[r, c] == write['board'][r][c]]
    if not 1 <= len(additions) <= 2 or not supported(visual, additions):
        return None
    return dict(write, original_additions=write['additions'], additions=additions)


def signal_check(event: dict, rows: list[dict], target: np.ndarray, obs: dict, fps: float) -> str:
    """保存合図の補助証拠を残し、状態名だけの落下を確定扱いしない。"""
    side, i = event['side'], event['i']
    if event['type'] == 'formula':
        return '原映像×記号出現' if 'ncc' in event else '保存式出現'
    if event['type'] == 'NEXT':
        if 'moving' in event:
            return '原映像NEXT絵柄移動'
        queue = {(tuple(r['sides'][side]['next_pair'] or []),
                  tuple(r['sides'][side]['dnext_pair'] or []))
                 for r in rows[max(0, i-4):i+9]}
        return 'NEXT色列変化あり' if len(queue) > 1 else 'NEXT移動フラグのみ'
    before = point(obs, rows[max(0, i-8)]['t'], side, fps)
    if before is not None and np.any((target == 9) & (before >= 0) & (before != 9)):
        return 'おじゃま画像増加あり'
    return 'おじゃま状態のみ'


def error_cells(write: dict, target: np.ndarray, visual: np.ndarray,
                saved: np.ndarray, landing_cells: list) -> list[dict]:
    """既存の誤りを除き、保存後に画像が変わったセルだけを抽出する。"""
    board = np.asarray(write['board'])
    fresh = np.zeros_like(target, dtype=bool)
    for r, c in landing_cells:
        fresh[r, c] = True
    linked_missing = fresh & (visual == target) if int(fresh.sum()) <= 2 else np.zeros_like(fresh)
    mask = (target >= 0) & (visual >= 0) & ((visual == board) | linked_missing) & (target != board)
    # 合図に伴って流入したおじゃま自体を、組ぷよの早期確定誤りへ混ぜない。
    mask &= (target != 9) & (board != 9) & (target != 10) & (board != 10)
    # 早期書込みした組の旧位置と、合図で確認した新位置に帰属を限定する。
    related = np.zeros_like(mask)
    for r, c in write['additions']:
        related[r, c] = True
    if int(fresh.sum()) <= 2:
        related |= fresh
    mask &= related
    cells = []
    for r, c in np.argwhere(mask):
        cells.append(dict(row=int(r), col=int(c), written=int(board[r, c]),
            reference=int(target[r, c]), persists_at_signal=bool(saved[r, c] != target[r, c])))
    return cells


def actual_errors(event: dict, rows: list[dict], obs: dict, fps: float) -> None:
    """初回保存との差だけでなく、実際に保存値が誤った観測時点を要求する。"""
    result, side = [], event['side']
    for cell in event['cells']:
        index = event['write_i']
        for frame in range(round(event['write_t']*fps), round(event['t']*fps)+1):
            stamp = frame/fps
            while index+1 < len(rows) and rows[index+1]['t'] <= stamp+TIME_EPSILON:
                index += 1
            row = rows[index]
            visual = obs.get((frame, side))
            saved = row['sides'][side]['confirmed_board']
            if visual is None or saved is None:
                continue
            r, c, correct = cell['row'], cell['col'], cell['reference']
            if visual[r, c] == correct and saved[r][c] != correct:
                cell.update(first_error_i=index, first_error_sec=stamp)
                result.append(cell)
                break
    event['initial_comparison_cells'] = len(event['cells'])
    event['cells'] = result


def durations(event: dict, rows: list[dict], horizon: float | None) -> None:
    """初回誤り観測から訂正までを測り、盤面過程の終了は打切りにする。"""
    side = event['side']
    for cell in event['cells']:
        ended, reason = rows[-1]['t'], '記録末尾'
        for row in rows[cell['first_error_i']+1:]:
            if horizon is not None and row['t'] >= horizon:
                ended, reason = horizon, '次の置き合図'
                break
            value = row['sides'][side]
            board = value['confirmed_board']
            if row['game'] != event['game'] or value['state']['state'] in ('MENU', 'CHAIN', 'GRAVITY_SETTLE'):
                ended, reason = row['t'], '盤面過程終了'
                break
            if board is not None and board[cell['row']][cell['col']] == cell['reference']:
                ended, reason = row['t'], '訂正'
                break
        cell.update(duration_sec=max(0.0, ended-cell['first_error_sec']), censored=reason != '訂正', end_reason=reason, end=ended)


def measure(event: dict, group: dict, rows: list[dict], obs: dict, fps: float) -> dict:
    """連続する合図の画像から置きを識別し、その最初の書込みを対応する。"""
    result = dict(source='', **event, status='書込み対応なし', fps=fps,
        write_t=None, write_i=None, early=None, stage=None, cells=[])
    target, reference_frame = reference(event, obs, fps)
    if target is None:
        return dict(result, status='画像未取得')
    baseline = point(obs, event['cycle_start'], event['side'], fps)
    if baseline is None:
        return dict(result, status='前周期画像未取得')
    added = np.argwhere((baseline == 0) & np.isin(target, COLORS)).tolist()
    if not 1 <= len(added) <= 2 or not supported(target, added):
        return dict(result, status='画像上の置き対応不明', landing_cells=added)
    result.update(placement_confirmed=True, landing_cells=added, stage=min(p[0] for p in added))
    writes = placement_writes(event, group, obs, fps, baseline, target, added)
    if not writes:
        return result
    write = writes[0]
    visual = point(obs, write['t'], event['side'], fps)
    saved_list = rows[event['i']]['sides'][event['side']]['confirmed_board']
    if saved_list is None:
        return dict(result, status='合図時保存盤面なし')
    signal = signal_check(event, rows, target, obs, fps)
    result.update(status='対応候補',
        signal_evidence=signal, write_t=write['t'], write_i=write['i'],
        write_additions=write['additions'], landing_cells=added,
        stage=min(p[0] for p in added),
        written_stage=min(p[0] for p in write['additions']),
        early=write['t'] < event['t'], reference_frame=reference_frame,
        added_delay_native_frames=max(0, round((event['t']-write['t'])*fps)),
        added_delay_30fps=max(0, (event['t']-write['t'])*FPS),
        cells=error_cells(write, target, visual, np.asarray(saved_list), added),
        observed_cells=int((target[1:] >= 0).sum()), write_board=write['board'])
    if not result['early']:
        result['cells'] = []
    actual_errors(result, rows, obs, fps)
    later = [e['t'] for e in group['events'] if e['t'] > event['t']]
    durations(result, rows, min(later) if later else None)
    return result


def placement_writes(event: dict, group: dict, obs: dict, fps: float,
                     baseline: np.ndarray, target: np.ndarray, landing_cells: list) -> list[dict]:
    """保存盤面に既存の欠落があっても、画像基準で現在の組の書込みを探す。"""
    writes = []
    expected = Counter(int(target[r, c]) for r, c in landing_cells)
    for number in event['write_indices']:
        original = group['writes'][number]
        if original['game'] != event['game']:
            continue
        candidate = plausible_write(original, obs, fps, event['side'], baseline)
        if candidate is not None:
            observed = Counter(candidate['board'][r][c] for r, c in candidate['additions'])
            if len(landing_cells) == 2 and any(count > expected[color] for color, count in observed.items()):
                continue
            writes.append(candidate)
    return writes


def summarize(events: list[dict]) -> dict:
    """母数・未対応・画像差分を区別して集計する。"""
    paired = [e for e in events if e['status'] == '対応候補']
    early = [e for e in paired if e['early']]
    errors = [e for e in early if e['cells']]
    delays = [e['added_delay_native_frames'] for e in early]
    cells = [c for e in errors for c in e['cells']]
    closed = [c['duration_sec'] for c in cells if not c['censored']]
    event_closed = [max(c['duration_sec'] for c in e['cells']) for e in errors if not any(c['censored'] for c in e['cells'])]
    lower_durations = [c['duration_sec'] for c in cells]
    return dict(signals=len(events), physical_placements=sum(e.get('placement_confirmed', False) for e in events),
        paired=len(paired), early=len(early),
        comparison_events=sum(e.get('initial_comparison_cells', 0) > 0 for e in early),
        comparison_cells=sum(e.get('initial_comparison_cells', 0) for e in early),
        error_events=len(errors), error_cells=len(cells),
        persistent_events=sum(any(c['persists_at_signal'] for c in e['cells']) for e in errors),
        persistent_cells=sum(c['persists_at_signal'] for c in cells),
        unresolved=len(events)-len(paired), statuses=dict(Counter(e['status'] for e in events)),
        delay_n=len(delays), delay_p50=float(np.percentile(delays, 50)) if delays else None,
        delay_p95=float(np.percentile(delays, 95)) if delays else None,
        delay_over8=sum(d > ACCEPT_FRAMES for d in delays),
        closed_n=len(closed), duration_p50=float(np.percentile(closed, 50)) if closed else None,
        duration_p95=float(np.percentile(closed, 95)) if closed else None,
        event_closed_n=len(event_closed),
        event_duration_p50=float(np.percentile(event_closed, 50)) if event_closed else None,
        event_duration_p95=float(np.percentile(event_closed, 95)) if event_closed else None,
        duration_lower_p50=float(np.percentile(lower_durations, 50)) if lower_durations else None,
        duration_lower_p95=float(np.percentile(lower_durations, 95)) if lower_durations else None,
        censored=sum(c['censored'] for c in cells))


def overlap(events: list[dict], all_rows: dict) -> list[dict]:
    """同じ記録・side・座標・保存値の継続をD1の96セルと照合する。"""
    cells = json.loads((OUT.parent/'d1/classified.json').read_text())['cells']
    result = []
    for cell in cells:
        if not cell['category'].startswith('経路未確定'):
            continue
        matches = []
        for event in events:
            if event['source'] != cell['source'] or event['side'] != cell['side'] or not event.get('early'):
                continue
            if event['t'] > cell['trigger'] or event['game'] != cell['game']:
                continue
            for found in event.get('cells', []):
                if (found['row'], found['col'], found['written'], found['reference']) != (cell['row'], cell['col'], cell['origin'], cell['snapshot']):
                    continue
                rows = all_rows[cell['source']]
                span = [r for r in rows[event['i']:] if r['t'] <= cell['trigger']]
                uninterrupted = all(r['sides'][cell['side']]['confirmed_board'] is not None and r['sides'][cell['side']]['confirmed_board'][cell['row']][cell['col']] == cell['origin'] for r in span)
                if uninterrupted:
                    matches.append(dict(t=event['t'], write_t=event['write_t'], type=event['type'], status=event['status']))
        result.append(dict(**cell, matches=matches))
    return result


def cross_table(events: list[dict]) -> None:
    """段×合図の全組合せを、未確定段も含めて保存する。"""
    records = []
    for stage in (*range(1, 13), None):
        for signal in ('NEXT', 'ojama', 'formula'):
            data = summarize([e for e in events if e.get('stage') == stage and e['type'] == signal])
            records.append(dict(stage=stage if stage is not None else '未確定', signal=signal,
                **{k: data[k] for k in ('signals', 'paired', 'early', 'comparison_events', 'comparison_cells', 'error_events', 'error_cells',
                    'persistent_events', 'persistent_cells', 'unresolved', 'delay_p50', 'delay_p95', 'delay_over8')}))
    with (OUT/'stage_signal_counts.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(records)


def reject_duplicate_writes(events: list[dict]) -> None:
    """同じ書込みが二つの周期へ対応する場合は、両方とも未確定へ戻す。"""
    counts = Counter((e['source'], e['side'], e['write_t']) for e in events if e['status'] == '対応候補')
    conflicts = []
    for event in events:
        key = event['source'], event['side'], event['write_t']
        if event['status'] == '対応候補' and counts[key] > 1:
            conflicts.append(dict(source=event['source'], side=event['side'],
                write_t=event['write_t'], signal_t=event['t']))
            event.update(status='書込み対応競合')
    (OUT/'ambiguous_write_cycles.json').write_text(json.dumps(conflicts, indent=2))


def main() -> None:
    """5記録を直列集計し、明細と母数付き集計を保存する。"""
    os.nice(19)
    events, all_rows = [], {}
    for source in SOURCES:
        name = 'zenchi' if source == 'review' else source
        if not (OUT/f'{name}_observations_complete.json').exists():
            continue
        rows = load(source)
        all_rows[source] = rows
        obs, fps = observations(source)
        groups = json.loads((OUT/f'{source}_candidates.json').read_text())
        for group in groups:
            for event in group['events']:
                value = measure(event, group, rows, obs, fps)
                value['source'] = source
                events.append(value)
        print(source, summarize([e for e in events if e['source'] == source]), flush=True)
    reject_duplicate_writes(events)
    result = dict(total=summarize(events),
        sources={s: summarize([e for e in events if e['source'] == s]) for s in all_rows},
        stages={str(r): summarize([e for e in events if e.get('stage') == r]) for r in range(1, 13)},
        unknown_stage=summarize([e for e in events if e.get('stage') is None]),
        signals={s: summarize([e for e in events if e['type'] == s]) for s in ('NEXT', 'ojama', 'formula')})
    (OUT/'measurements.json').write_text(json.dumps(events, ensure_ascii=False, separators=(',', ':')))
    (OUT/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (OUT/'d1_overlap.json').write_text(json.dumps(overlap(events, all_rows), ensure_ascii=False, indent=2))
    cross_table(events)


if __name__ == '__main__':
    main()
