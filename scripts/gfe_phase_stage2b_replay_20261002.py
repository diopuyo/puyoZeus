"""段2と同じ再生器で静止専用候補を測り、用途・モデル・通知を記録する。"""
from __future__ import annotations

import argparse
import json
import os
from time import perf_counter
from typing import Any

from scripts import gfe_phase_stage2_replay_20261002 as previous
from scripts.gfe_phase_stage2b_train_20261002 import OUT, model_directory
from scripts.gfe_phase_stage1_20261002 import save_json
import numpy as np

PURPOSES = ('static', 'landing')


class Trace:
    """評価器の外側だけで計測状態を保持する。"""

    def __init__(self) -> None:
        self.total: list[float] = []
        self.inference: list[float] = []
        self.calls: list[dict] = []
        self.notification = -1
        self.time = 0.
        self.game = -1
        self.models: dict[int, str] = {}
        self.metadata: dict[str, dict] = {}
        self.purpose_totals: dict[str, list[float]] = {p: [] for p in PURPOSES}
        self.purpose_inference: dict[str, list[float]] = {p: [] for p in PURPOSES}

    def install(self, mode: str) -> tuple[list, list]:
        """読込み指定と呼出元別ラッパーだけを追加する。"""
        from src import exchange_event_tracker as tracker, exchange_event_landing as landing
        from src import exchange_event_layers as layers
        from src.exchange_event_evaluator import FileExchangeModels, StaticInput
        original_load = FileExchangeModels.load
        original_predict = FileExchangeModels.predict_source_probability
        original_evaluate = tracker.evaluate_exchange_event

        def load(cls: type, directory: Any, lightweight: bool = True, **kwargs: Any) -> Any:
            if mode != 'baseline':
                kwargs.update(gfe_directory=model_directory(mode), gfe_static_only=True)
            result = original_load(directory, lightweight, **kwargs)
            self.register(result, mode)
            return result

        def predict(models: Any, name: str, values: np.ndarray) -> float:
            started = perf_counter()
            value = original_predict(models, name, values)
            if name == 'G_fe':
                self.inference.append((perf_counter()-started)*previous.MILLISECONDS)
            return value

        def wrapper(purpose: str) -> Any:
            def evaluate(event: Any, models: Any) -> float:
                started = perf_counter()
                value = original_evaluate(event, models)
                if isinstance(event, StaticInput):
                    elapsed = (perf_counter()-started)*previous.MILLISECONDS
                    self.total.append(elapsed)
                    self.calls.append(dict(notification=self.notification, t_sec=self.time,
                        game_idx=self.game, purpose=purpose, model=self.models[id(models)],
                        total_ms=elapsed, inference_ms=self.inference[-1], p1=value))
                return value
            return evaluate

        FileExchangeModels.load = classmethod(load)
        FileExchangeModels.predict_source_probability = predict
        tracker.evaluate_exchange_event = layers.evaluate_exchange_event = wrapper('static')
        landing.evaluate_exchange_event = wrapper('landing')
        return self.total, self.inference

    def register(self, models: Any, mode: str) -> None:
        """呼び出した実モデルをファイルハッシュへ結び付ける。"""
        from pathlib import Path
        base = Path('models/exchange_event_v1')
        for label, model, directory in (
                (mode, models, base if mode == 'baseline' else model_directory(mode)),
                ('baseline', models.landing_models, base)):
            if model is None:
                continue
            entry = json.loads((directory/'manifest.json').read_text())['models']['G_fe']
            self.models[id(model)] = label
            self.metadata[label] = dict(directory=str(directory), sha256=entry['sha256'],
                estimator=type(model.estimators['G_fe']).__name__, columns=len(entry['columns']))

    def notifications(self, total: list, inference: list) -> tuple[list, list, list]:
        """総計・用途別とも1通知内の呼出し時間を合算する。"""
        from src.exchange_event_overlay import ExchangeEventOverlay
        original = ExchangeEventOverlay.update
        totals, inferences, counts = [], [], []

        def update(overlay: Any, *args: Any, **kwargs: Any) -> None:
            self.notification += 1
            self.time = float(args[3] if len(args) > 3 else kwargs['t_sec'])
            self.game = int(args[4] if len(args) > 4 else kwargs['game_idx'])
            start = len(self.calls)
            original(overlay, *args, **kwargs)
            calls = self.calls[start:]
            if calls:
                totals.append(sum(c['total_ms'] for c in calls))
                inferences.append(sum(c['inference_ms'] for c in calls))
                counts.append(len(calls))
            for purpose in PURPOSES:
                selected = [c for c in calls if c['purpose'] == purpose]
                if selected:
                    self.purpose_totals[purpose].append(sum(c['total_ms'] for c in selected))
                    self.purpose_inference[purpose].append(sum(c['inference_ms'] for c in selected))

        ExchangeEventOverlay.update = update
        return totals, inferences, counts

    def save(self, mode: str, source: str) -> None:
        """表示・eventsを変更せず独立した監査原票へ保存する。"""
        destination = OUT/mode/'replay'/source
        keys = ('notification', 't_sec', 'game_idx', 'purpose', 'model', 'total_ms', 'inference_ms', 'p1')
        values = {key: np.asarray([c[key] for c in self.calls]) for key in keys}
        for purpose in PURPOSES:
            values[purpose+'_notification_ms'] = self.purpose_totals[purpose]
            values[purpose+'_inference_ms'] = self.purpose_inference[purpose]
        np.savez_compressed(destination/'gfe_usage.npz', **values)
        counts = {p: {m: sum(c['purpose'] == p and c['model'] == m for c in self.calls)
                      for m in self.metadata} for p in PURPOSES}
        assert all(c['model'] == ('baseline' if c['purpose'] == 'landing' else mode) for c in self.calls)
        save_json(destination/'GFE_USAGE.json', dict(models=self.metadata, calls=counts,
            nice=os.nice(0), optional_default=False, gfe_static_only=mode != 'baseline'))


def main() -> None:
    """既存再生と証明器アダプタを変更せず再利用する。"""
    os.nice(max(0, 19-os.nice(0)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('baseline', 'V3s', 'R1s'))
    parser.add_argument('source', choices=previous.SOURCES)
    args = parser.parse_args()
    trace = Trace()
    previous.OUT = OUT
    previous.install_timing = trace.install
    previous.install_notifications = trace.notifications
    previous.run(args.mode, args.source)
    trace.save(args.mode, args.source)


if __name__ == '__main__':
    main()
