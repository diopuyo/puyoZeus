"""同じ記録入力をOFF/ONで再生し、offlineの評価列と交換記録を厳密照合する。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from unittest.mock import patch, Mock

from src.phase_j.live_cpu import configure_environment, apply_runtime

CACHE_PROBE_CALLS = 1000
RECOGNITION_COLUMNS = ('frame', 't_sec', 'boards', 'raw', 'stable', 'states', 'active', 'ready', 'scores')


def compare_live(reference: Path, candidate: Path) -> dict:
    import numpy as np
    with np.load(reference/'recognition.npz') as left, np.load(candidate/'recognition.npz') as right:
        count = len(right['frame'])
        assert count > 0
        for key in RECOGNITION_COLUMNS:
            np.testing.assert_array_equal(left[key][:count], right[key], err_msg=key)
    with np.load(reference/'display.npz') as left, np.load(candidate/'display.npz') as right:
        count_display = len(right['t_sec'])
        for key in right.files:
            if key != 'video_id':
                np.testing.assert_array_equal(left[key][:count_display], right[key], err_msg=key)
    result = dict(recognition_frames=count, recognition_columns=list(RECOGNITION_COLUMNS),
                  display_rows=count_display, display_all_columns='exact')
    (candidate/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def cache_probe(output: Path) -> dict:
    from src.board import Board
    from src.chain import ChainSimulator
    from src.phase_j.live_cache import RecentCache
    board = Board()
    result = {}
    for name in ('off', 'on'):
        simulator = ChainSimulator()
        cached = simulator.simulate(board)
        if name == 'off':
            simulator._cache = {i.to_bytes(8, 'little'): cached for i in range(simulator._CACHE_MAX_SIZE)}
        else:
            simulator._cache = RecentCache()
            for i in range(simulator._CACHE_MAX_SIZE):
                simulator._cache[i.to_bytes(8, 'little')] = cached
        with patch.object(simulator, '_simulate_uncached', Mock(wraps=simulator._simulate_uncached)) as call:
            for _ in range(CACHE_PROBE_CALLS):
                actual = simulator.simulate(board)
                assert actual.final_board.grid_bytes() == cached.final_board.grid_bytes()
            result[name] = dict(requests=CACHE_PROBE_CALLS, uncached_calls=call.call_count,
                                retained=len(simulator._cache))
    output.mkdir(parents=True, exist_ok=True)
    (output/'cache_probe.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def replay_mode(record: Path, output: Path, bounded: bool) -> dict:
    configure_environment(1, 0)
    apply_runtime('evaluation')
    import scripts.replay_exchange_event_20260926 as replay
    from src.phase_j.live_cache import bounded_chain_caches
    from src.phase_j.live_retention import bounded_overlay
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if bounded:
            stack.enter_context(bounded_chain_caches())
            cls = bounded_overlay(replay.ExchangeEventOverlay, output/'spool', stack)
            stack.enter_context(patch.object(replay, 'ExchangeEventOverlay', cls))
        return replay.replay(record, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, default=Path('logs/live_b11/baseline/inputs.jsonl.gz'))
    parser.add_argument('--output', type=Path, default=Path('logs/live_b11/replay'))
    parser.add_argument('--mode', choices=('off', 'on', 'compare', 'cache', 'live'), required=True)
    parser.add_argument('--reference', type=Path, default=Path('logs/live_b11/baseline'))
    parser.add_argument('--candidate', type=Path, default=Path('logs/live_b11/candidate'))
    options = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if options.mode == 'live':
        result = compare_live(options.reference, options.candidate)
    elif options.mode == 'cache':
        result = cache_probe(options.output)
    elif options.mode == 'compare':
        from scripts.replay_exchange_event_20260926 import compare
        result = compare(options.output/'off', options.output/'on')
        (options.output/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    else:
        result = replay_mode(options.record, options.output/options.mode, options.mode == 'on')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
