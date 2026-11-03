"""本番と同じ記録・構成でG_feのみを差し替え、通知単位の時間を保存する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from scripts.gfe_phase_stage2_train_20261002 import OUT, model_directory
from scripts.gfe_phase_stage1_20261002 import memory_guard, save_json
import numpy as np

PREFIRE = Path("/mnt/d/puyo_analyzer/wt_prefire")
EXEV = Path(_public_dev_text('<repo>'))
EVALSET = Path("/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set")
SET1 = tuple(f"c{i}" for i in range(1, 7)) + ("zenchi",)
SET2 = tuple(f"s{i}" for i in range(7))
EXISTING = ("q_7gc4TgFig", "fcXG83vInDY", "mia8KCjr52g", "review", "zenchi")
SOURCES = tuple(dict.fromkeys((*SET1, *SET2, *EXISTING)))
MILLISECONDS = 1000
PROGRESS_INTERVAL = 5000


def record_path(source: str) -> Path:
    """案Cと同じ固定記録だけを読む。"""
    if source in SET1 and source != "zenchi":
        return EVALSET / "collect/records" / f"{source}.jsonl.gz"
    if source in SET2:
        return EVALSET / "set2/collect/records" / f"{source}.jsonl.gz"
    return EXEV / "logs/pending_expiry/full/records" / f"{source}.jsonl.gz"


def baseline(source: str) -> Path:
    """案Cが検証済みの補正ゼロ本番再生を対照として参照する。"""
    mode = "train" if source in SET1 else "replay"
    return PREFIRE / "logs/prefire_prediction/v6" / mode / source


def bound_adapter(original: type) -> type:
    """旧証明器に既定Falseの新引数だけを受け付けさせ、証明処理は完全継承する。"""
    class CompatibleBound(original):
        def __init__(self, early_exit: bool = False) -> None:
            if early_exit:
                raise ValueError("固定版証明器ではearly_exit=Trueは禁止")
            super().__init__()
    return CompatibleBound


def install_timing(mode: str) -> tuple[list[float], list[float]]:
    """特徴生成込み評価と推定器だけの時間を別々に実測する。"""
    from src import exchange_event_tracker as tracker
    from src import exchange_event_landing as landing, exchange_event_layers as layers
    from src.exchange_event_evaluator import FileExchangeModels, StaticInput

    original_load = FileExchangeModels.load
    original_predict = FileExchangeModels.predict_source_probability
    original_evaluate = tracker.evaluate_exchange_event
    total, inference = [], []

    def load(cls: type, directory: Path, lightweight: bool = True, **kwargs: Any) -> Any:
        if mode != "baseline":
            kwargs["gfe_directory"] = model_directory(mode)
        return original_load(directory, lightweight, **kwargs)

    def predict(models: Any, name: str, values: np.ndarray) -> float:
        started = perf_counter()
        result = original_predict(models, name, values)
        if name == "G_fe":
            inference.append((perf_counter()-started)*MILLISECONDS)
        return result

    def evaluate(event: Any, models: Any) -> float:
        started = perf_counter()
        try:
            return original_evaluate(event, models)
        finally:
            if isinstance(event, StaticInput):
                total.append((perf_counter()-started)*MILLISECONDS)

    FileExchangeModels.load = classmethod(load)
    FileExchangeModels.predict_source_probability = predict
    for module in (tracker, landing, layers):
        module.evaluate_exchange_event = evaluate
    return total, inference


def install_notifications(total: list[float], inference: list[float]) -> tuple[list, list, list]:
    """1更新通知内で複数回G_feを呼ぶ場合も、計算時間を足して一通知として測る。"""
    from src.exchange_event_overlay import ExchangeEventOverlay
    original = ExchangeEventOverlay.update
    totals, inferences, counts = [], [], []

    def update(overlay: Any, *args: Any, **kwargs: Any) -> None:
        start, before = len(total), len(inference)
        original(overlay, *args, **kwargs)
        if len(total) > start:
            totals.append(sum(total[start:]))
            inferences.append(sum(inference[before:]))
            counts.append(len(total)-start)

    ExchangeEventOverlay.update = update
    return totals, inferences, counts


def production_directory(options: dict) -> Path:
    """参照実装model_directoryと同じ構成条件でv3/v4を選ぶ。"""
    version = "v4" if options.get("count_sync") or options.get("e16") else "v3"
    return Path("models/exchange_event_" + version)


def progress_observer(destination: Path) -> Any:
    """出力値には触れず、長い再生の進捗保存時にもメモリ条件を確認する。"""
    frames = 0
    def observe(overlay: Any, inputs: tuple) -> None:
        nonlocal frames
        frames += 1
        if frames % PROGRESS_INTERVAL == 0:
            memory_guard()
            save_json(destination/"PROGRESS.json", dict(frames=frames, t_sec=inputs[3]))
    return observe


def run(mode: str, source: str) -> None:
    """独立プロセスで1記録を再生し、既定OFFでは保存本番とバイト照合する。"""
    memory_guard()
    destination = OUT / mode / "replay" / source
    if (destination / "DONE.json").exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    from scripts import d5_runtime
    d5_runtime.OUT = destination / "runtime"
    d5_runtime.install()
    bound = sys.modules["src.exchange_post_counter_bound"]
    bound.PostCounterDeathBound = bound_adapter(bound.PostCounterDeathBound)
    from scripts.run_e36b import options
    from scripts.replay_exchange_event_20260926 import replay
    total, inference = install_timing(mode)
    totals, inferences, calls = install_notifications(total, inference)
    opts = dict(options(), switch_smoothing=True)
    result = replay(record_path(source), destination, production_directory(opts), True,
                    progress_observer(destination), **opts)
    np.savez_compressed(destination / "latency.npz", total_ms=totals, inference_ms=inferences,
                        call_ms=total, call_inference_ms=inference, calls_per_notification=calls)
    assert len(total) == len(inference)
    from threadpoolctl import threadpool_info
    result.update(options=opts, notifications=len(totals), gfe_calls=len(total), threadpools=threadpool_info())
    assert all(pool["num_threads"] == 1 for pool in result["threadpools"] if pool["user_api"] == "blas")
    if mode == "baseline":
        result["byte_identity"] = verify_baseline(destination, source)
    save_json(destination / "DONE.json", result)
    print(json.dumps(dict(mode=mode, source=source, **result)), flush=True)


def verify_baseline(destination: Path, source: str) -> dict:
    """既存の表示・イベントの全バイトを照合し、版差を比較へ混入させない。"""
    expected = baseline(source)
    result = {name: (destination / name).read_bytes() == (expected / name).read_bytes()
              for name in ("display.npz", "events.jsonl")}
    save_json(destination / "BYTE_IDENTITY.json", result)
    if not all(result.values()):
        raise AssertionError(f"既定OFFと保存本番のバイト不一致: {source}: {result}")
    return result


def main() -> None:
    """低優先度を強制してから実行する。"""
    os.nice(max(0, 19-os.nice(0)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "V3", "V1b"))
    parser.add_argument("source", choices=SOURCES)
    args = parser.parse_args()
    run(args.mode, args.source)


if __name__ == "__main__":
    main()
