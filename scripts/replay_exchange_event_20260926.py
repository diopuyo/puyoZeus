"""記録された評価入力だけで撃ち合いを再生し、E4互換の出力を作る。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.exchange_event_evaluator import FileExchangeModels, StaticInput
from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_event_record import read_records, static_key


def static_builder(record: Path) -> Any:
    """保存Dを入力キーで再利用する。変更後に新しい盤面組が必要なら再計算する。"""
    from scripts.visualize_advantage_overlay import _exchange_static_input
    cache = {row["key"]: row["input"]["d_features"] for row in read_records(record)
             if row["kind"] == "static"}

    def build(boards: tuple, snapshot: Any, elapsed: float, m0: float) -> StaticInput:
        key = static_key(boards, snapshot)
        if key not in cache:
            cache[key] = _exchange_static_input(boards, snapshot, elapsed, m0).d_features
        return StaticInput(cache[key], m0, elapsed, source_side=0)
    return build


def display_row(overlay: ExchangeEventOverlay, inputs: tuple, context: dict,
                smoothing: Any = None) -> Any:
    """表示値と由来を今回の評価から生成し、旧経路の補助列だけ記録から使う。"""
    from scripts.visualize_advantage_overlay import (
        DisplayTimelineRow, TIMELINE_DUMP_SCORE_NONE_SENTINEL, _exchange_display,
    )
    result = inputs[0]
    adv, probability = _exchange_display(overlay, context["fallback_adv"], context["fallback_p1"],
                                         smoothing, context["t_sec"])
    if overlay.tracker.probability is not None:
        probability = overlay.tracker.probability
    scores = [TIMELINE_DUMP_SCORE_NONE_SENTINEL if s.score is None else int(s.score)
              for s in (result.p1, result.p2)]
    return DisplayTimelineRow(t_sec=context["t_sec"], game_idx=context["game_idx"],
        state1=result.p1.state.name, state2=result.p2.state.name, display_adv=adv,
        display_p1=probability, adv_raw_last=context["adv_raw_last"], source=overlay.tracker.source,
        resolved_active=context["resolved_active"], settled_ran=context["settled_ran"],
        score1=scores[0], score2=scores[1])


def replay(record: Path, out: Path, model_dir: Path | None = None) -> dict:
    """認識器も動画も開かず、tracker・終了判定・全評価器を新規生成する。"""
    from scripts.visualize_advantage_overlay import (
        _ExchangeEventEndSignals, _ExchangeDisplayEMA, _exchange_display, save_display_timeline,
    )
    from src.exchange_event_m0 import FileM0Predictor
    start = time.perf_counter()
    stream = read_records(record)
    header = next(stream)
    directory = model_dir or Path(header["model_dir"])
    overlay = ExchangeEventOverlay(FileExchangeModels.load(directory, lightweight=True),
        static_builder(record), _ExchangeEventEndSignals, FileM0Predictor(directory / "M0"),
        per_side_settled=header["per_side_settled"])
    rows, frames, inputs = [], 0, None
    smoothing = _ExchangeDisplayEMA()
    for item in stream:
        if item["kind"] == "update":
            inputs = item["args"]
            overlay.update(*inputs)
            _exchange_display(overlay, 0.0, 0.5, smoothing, inputs[3])
            frames += 1
        elif item["kind"] == "display":
            if inputs is None or inputs[3:5] != (item["t_sec"], item["game_idx"]):
                raise ValueError("表示文脈と入力フレームが不一致")
            rows.append(display_row(overlay, inputs, item, smoothing))
        elif item["kind"] == "complete" and item["frames"] != frames:
            raise ValueError("記録フレーム数が不一致")
    save_display_timeline(out / "display.npz", header["video_id"], rows)
    overlay.tracker.save(out / "events.jsonl")
    status = dict(state="completed", frames=frames, display_frames=len(rows),
                  elapsed_seconds=time.perf_counter() - start, record_bytes=record.stat().st_size)
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def compare(expected: Path, actual: Path) -> dict:
    """NPZの全列・dtype・NaN位置とJSONLの全バイトを厳密照合する。"""
    with np.load(expected / "display.npz") as left, np.load(actual / "display.npz") as right:
        if left.files != right.files:
            raise AssertionError("display.npzの列が不一致")
        for name in left.files:
            if left[name].dtype != right[name].dtype:
                raise AssertionError(f"display.npz dtype: {name}")
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)
            if left[name].tobytes() != right[name].tobytes():
                raise AssertionError(f"display.npz bits: {name}")
    for name in ("events.jsonl", "events.diagnostics.json"):
        if name.endswith("jsonl"):
            assert (expected / name).read_bytes() == (actual / name).read_bytes(), name
        else:
            assert json.loads((expected / name).read_text()) == json.loads((actual / name).read_text()), name
    if (expected / "display.npz").read_bytes() != (actual / "display.npz").read_bytes():
        raise AssertionError("display.npzのファイルバイトが不一致")
    return dict(display="byte_identical", events="byte_identical", diagnostics="identical")


def main() -> None:
    """単独再生と同じレンダ出力との等価性検査を提供する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--compare", type=Path)
    options = parser.parse_args()
    result = replay(options.record, options.out, options.model_dir)
    if options.compare:
        result["equivalence"] = compare(options.compare, options.out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
