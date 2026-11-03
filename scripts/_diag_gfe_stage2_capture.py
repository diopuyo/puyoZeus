"""段2と同じ再生のモデル境界入力を診断専用出力へ保存する。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scripts.gfe_phase_stage1_20261002 import memory_guard, save_json
from scripts.gfe_phase_stage2_replay_20261002 import (
    bound_adapter, record_path, production_directory, model_directory,
)

OUT = Path('logs/gfe_phase_stage2_diag_20261002')
PROGRESS_INTERVAL = 1000
S2_FRAME_LIMIT = 2000


def observer(destination: Path) -> Any:
    """再生の区間内時刻と資源条件を定期記録する。"""
    frames = 0
    def observe(overlay: Any, inputs: tuple) -> None:
        nonlocal frames
        memory_guard()
        frames += 1
        if frames % PROGRESS_INTERVAL == 0:
            save_json(destination/'PROGRESS.json', dict(frames=frames, time=inputs[3]))
    return observe


def limit_replay(source: str) -> None:
    """s2だけ先頭約66.7秒の一部再生に限定し、診断範囲を明示する。"""
    if source != 's2':
        return
    from scripts import replay_exchange_event_20260926 as module
    original = module.read_records
    def limited(path: Path) -> Any:
        frames = 0
        for row in original(path):
            if row['kind'] == 'update':
                if frames == S2_FRAME_LIMIT:
                    yield dict(kind='complete', frames=frames)
                    return
                frames += 1
            yield row
    module.read_records = limited


def identity(destination: Path, source: str, variant: str) -> dict:
    """全再生はバイト一致、部分再生は表示全列の接頭区間一致を確認する。"""
    old = Path('logs/gfe_phase_stage2_20261002')/variant/'replay'/source
    if source != 's2':
        result = {n: (destination/n).read_bytes() == (old/n).read_bytes()
                  for n in ('display.npz', 'events.jsonl')}
    else:
        with np.load(destination/'display.npz') as actual, np.load(old/'display.npz') as prior:
            n = len(actual['t_sec'])
            for key in actual.files:
                np.testing.assert_array_equal(actual[key], prior[key][:n] if prior[key].ndim else prior[key])
        result = dict(display_prefix_all_columns=True)
    assert all(result.values())
    return result


def select_variant(variant: str) -> None:
    """V3確認再生でも段2と同じ読み込み境界だけを差し替える。"""
    if variant == 'baseline':
        return
    from src.exchange_event_evaluator import FileExchangeModels
    original = FileExchangeModels.load
    def load(cls: type, directory: Path, lightweight: bool = True, **kwargs: Any) -> Any:
        return original(directory, lightweight, gfe_directory=model_directory(variant), **kwargs)
    FileExchangeModels.load = classmethod(load)


def capture(source: str, variant: str = 'baseline') -> None:
    """一回の本番再生で54列と59列を同じ入力から記録する。"""
    from scripts import d5_runtime
    destination = OUT/('capture' if variant == 'baseline' else 'capture_'+variant)/source
    destination.mkdir(parents=True, exist_ok=False)
    d5_runtime.OUT = destination/'runtime'
    d5_runtime.install()
    bound = sys.modules['src.exchange_post_counter_bound']
    bound.PostCounterDeathBound = bound_adapter(bound.PostCounterDeathBound)
    from scripts.run_e36b import options
    from scripts.replay_exchange_event_20260926 import replay
    from src import exchange_event_tracker as tracker
    from src import exchange_event_landing as landing, exchange_event_layers as layers
    from src.exchange_event_evaluator import StaticInput, build_features
    from src.exchange_event_user_phase import static_user_features
    from src.exchange_event_overlay import ExchangeEventOverlay
    original, original_update = tracker.evaluate_exchange_event, ExchangeEventOverlay.update
    values, context = [], [np.nan, -1]

    def evaluate(event: Any, models: Any) -> float:
        if isinstance(event, StaticInput):
            x59 = build_features(event, models.elapsed_thresholds)[1]
            x54 = static_user_features(event.d_features, event.m0_probability_1p, event.source_side)
            caller = sys._getframe(1).f_code.co_name
            values.append((context.copy(), event.elapsed_sec, event.d_features, x54, x59, caller))
        return original(event, models)

    def update(overlay: Any, *args: Any, **kwargs: Any) -> None:
        context[:] = [args[3], args[4]]
        return original_update(overlay, *args, **kwargs)

    for module in (tracker, landing, layers):
        module.evaluate_exchange_event = evaluate
    ExchangeEventOverlay.update = update
    opts = dict(options(), switch_smoothing=True)
    select_variant(variant)
    limit_replay(source)
    result = replay(record_path(source), destination, production_directory(opts), True,
                    observer(destination), **opts)
    np.savez_compressed(destination/'inputs.npz', context=[r[0] for r in values],
        elapsed=[r[1] for r in values], raw=[r[2] for r in values],
        x54=[r[3] for r in values], x59=[r[4] for r in values], caller=[r[5] for r in values])
    result['calls'] = len(values)
    result['byte_identity'] = identity(destination, source, variant)
    result['scope'] = f'first_{S2_FRAME_LIMIT}_updates' if source == 's2' else 'full'
    save_json(destination/'DONE.json', result)
    print(result, flush=True)


def main() -> None:
    """低優先度・6GiB以上の余裕を確認して再生する。"""
    os.nice(max(0, 19-os.nice(0)))
    memory_guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--variant', choices=('baseline','V3'), default='baseline')
    args = parser.parse_args()
    capture(args.source, args.variant)


if __name__ == '__main__':
    main()
