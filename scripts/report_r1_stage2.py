"""R1段2の事前固定4セルを全数判定し、失敗なら後段を許可しない。"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
from scripts._d3_candidates import load
from scripts.inspect_r1_stage2 import CELLS, OUT

EXPECTED = [1, 0, 3, 1]
SIGNAL_FIRST, SIGNAL_LAST = 2750.5, 2751.0
SCENE_FIRST, SCENE_LAST = 2740., 2760.


def read_lines(path: Path) -> list[dict]:
    """中断行を黙って捨てず、完了したJSONLだけを読む。"""
    return [json.loads(line) for line in path.read_text().splitlines()]


def values(board: list | None) -> list | None:
    """対象の欠測は空盤面で補完しない。"""
    return [board[r][c] for r, c in CELLS] if board is not None else None


def judge_signal(audit: dict, trace: list[dict], observations: dict) -> dict:
    """合図直後の公開盤面と原画像2枚が全4セルで一致することを要求する。"""
    board = audit['before']
    after = [row[:] for row in board] if board is not None else None
    for cell in audit['corrections']:
        after[cell['row']][cell['col']] = cell['after']
    public = next((r for r in trace if r['frame'] >= audit['frame']), None)
    current = observations.get(audit['observed_frame'])
    previous = observations.get(audit['observed_frame']-1) if current else None
    evidence = all(row is not None and row['target_cnn'] == row['target_hsv'] == EXPECTED
                   for row in (current, previous))
    observed = values(public['sides'][1]['board']) if public else None
    return dict(t_sec=audit['t_sec'], signal=audit['signal'], reason=audit['reason'],
                before=values(board), after=values(after), published=observed,
                corrections=audit['corrections'], independent_two_frames=evidence,
                passed=evidence and values(after) == observed == EXPECTED)


def baseline_check(trace: list[dict]) -> dict:
    """段2の全認識行を元固定入力の同時刻に対応付ける。"""
    old = {round(row['t'], 6): row for row in load('review')}
    matches = 0
    mismatches = []
    for row in trace:
        saved = old.get(round(row['t'], 6))
        same = saved is not None and all(
            left['board'] == right['confirmed_board'] and left['state'] == right['state']['state']
            and left['score'] == right['score'] for left, right in zip(row['sides'], saved['sides']))
        matches += same
        if not same:
            mismatches.append(row['t'])
    return dict(rows=len(trace), matches=matches, mismatch_times=mismatches)


def main() -> None:
    """両走行の完了を確認してから、合図別の修正件数と停止理由を保存する。"""
    assert all((OUT/mode/'completed.json').exists() for mode in ('off', 'on'))
    audit = read_lines(OUT/'on/capture/review/placement_signal_reconcile.jsonl')
    scene = [r for r in audit if SCENE_FIRST <= r['t_sec'] <= SCENE_LAST]
    trace = read_lines(OUT/'on/trace.jsonl')
    observations = {r['frame']: r for r in json.loads((OUT/'independent_observations.json').read_text())}
    target = [judge_signal(r, trace, observations) for r in scene
              if r['side'] == 1 and SIGNAL_FIRST <= r['t_sec'] < SIGNAL_LAST and r['signal'] == 'next']
    corrections = Counter()
    events = Counter()
    for row in scene:
        corrections[row['signal']] += len(row['corrections'])
        events[row['signal']] += bool(row['corrections'])
    result = dict(passed=any(r['passed'] for r in target), target=target,
        baseline=baseline_check(read_lines(OUT/'off/trace.jsonl')),
        correction_cells=dict(corrections), correction_events=dict(events),
        reasons=dict(Counter(r['reason'] for r in scene)), signals=len(scene),
        expected=EXPECTED, target_cells=CELLS)
    (OUT/'SUMMARY.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
