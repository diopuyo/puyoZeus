"""prechain shadowへlanding疑似イベント開始metadataの原子初期化だけを足す。"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
from pathlib import Path
from typing import Any

from scripts import diagnose_video38_prechain_shadow_v1 as prechain


base = prechain.base
FORMAT = "video38-atomic-landing-shadow/v1"
LAUNCHER = base.ROOT / "scripts/launch_video38_atomic_landing_shadow_v1.sh"
TEST = base.ROOT / "tests/test_diagnose_video38_atomic_landing_shadow_v1.py"
ORIGINAL_PREPARE = prechain.shadow_prepare
ORIGINAL_INSTRUMENT = prechain.shadow_instrument
ORIGINAL_SUMMARY = prechain._shadow_summary


def _metadata(pipe: Any, side: str) -> dict[str, Any]:
    """side別の連鎖開始metadataと非対象状態を比較可能にする。"""
    suffix = "1p" if side == "1P" else "2p"
    names = ("active_chain", "chain_until", "chain_entry_t",
             "chain_start_next", "chain_event_max_until",
             "last_chain_event_for_settle")
    return {name: base.json_value(getattr(pipe, f"_{name}_{suffix}", None))
            for name in names}


def _initialize_landing_metadata(pipe: Any, side: str, event: Any) -> None:
    """既存score/formula登録と同じ開始metadataだけをlandingへ適用する。"""
    if side not in ("1P", "2P"):
        raise ValueError(f"不正side: {side}")
    suffix = "1p" if side == "1P" else "2p"
    setattr(pipe, f"_chain_entry_t_{suffix}", float(event.trigger_sec))
    current_next = getattr(pipe, f"_last_seen_next_{suffix}")
    setattr(pipe, f"_chain_start_next_{suffix}", current_next)
    if bool(getattr(pipe, "_enable_game_event_chain_exit")):
        max_until = float(event.trigger_sec) + float(pipe._chain_max_hold_sec)
        setattr(pipe, f"_chain_event_max_until_{suffix}", max_until)


class AtomicRecorder(prechain.ShadowRecorder):
    """prechain証拠へlanding metadata更新だけを追記する。"""

    def __init__(self, stream: Any, target: dict[str, Any]) -> None:
        super().__init__(stream, target)
        self.atomic_metadata_rows: list[dict[str, Any]] = []

    def record_atomic(self, value: dict[str, Any]) -> None:
        row = base.json_value({"kind": "atomic_landing_metadata", **value})
        self.atomic_metadata_rows.append(row)
        self.emit(row)


def instrument_atomic_metadata(stack: contextlib.ExitStack, cls: Any,
                               rec: AtomicRecorder) -> None:
    """landingの_start_chain_estimate完了時だけ開始metadataを揃える。"""
    original = cls._start_chain_estimate

    @functools.wraps(original)
    def start(pipe: Any, side: str, event: Any,
              precomputed_result: Any = None) -> Any:
        if getattr(event, "mechanism", None) != "landing":
            return original(pipe, side, event, precomputed_result)
        before = _metadata(pipe, side)
        try:
            result = original(pipe, side, event, precomputed_result)
        except Exception as exc:
            rec.record_atomic({"status": "original_exception", "side": side,
                               "event": prechain.details.event_value(event),
                               "before": before, "exception_type": type(exc).__name__})
            raise
        _initialize_landing_metadata(pipe, side, event)
        rec.record_atomic({"status": "applied", "frame_idx": rec.frame,
                           "side": side, "event": prechain.details.event_value(event),
                           "before": before, "after": _metadata(pipe, side)})
        return result

    base.patch(stack, cls, "_start_chain_estimate", start)


def atomic_prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """prechain全guardへ新規3ファイルと単一軸契約を追加する。"""
    receipt, config = ORIGINAL_PREPARE(args)
    for path in (Path(__file__), LAUNCHER, TEST):
        receipt["input_and_code_sha256"][str(path)] = base.sha256(path)
    receipt["format_version"] = FORMAT
    receipt["atomic_landing_policy"] = {
        "adoption_status": "diagnostic_only_not_adopted",
        "one_axis": "initialize landing pseudo entry/start_next/max_until metadata",
        "baseline": "prechain return-board shadow v1",
        "unchanged": ["active event", "chain_until", "C-6", "slide guard",
                      "accounting", "pending/confirmed rollback"],
        "slide_guard_enabled": False, "production_modified": False,
        "rollback_problem_fixed": False}
    return receipt, config


def atomic_instrument(stack: contextlib.ExitStack, collector: Any,
                      rec: AtomicRecorder) -> None:
    """prechain/details計装を維持してlanding metadata wrapperだけ追加する。"""
    ORIGINAL_INSTRUMENT(stack, collector, rec)
    instrument_atomic_metadata(stack, collector.RecognitionPipeline, rec)


def atomic_summary(receipt: dict[str, Any], rec: AtomicRecorder) -> dict[str, Any]:
    """prechain summaryを保持し、原子metadata証拠を加える。"""
    summary = ORIGINAL_SUMMARY(receipt, rec)
    summary.update({
        "format_version": FORMAT,
        "atomic_landing_policy": receipt["atomic_landing_policy"],
        "atomic_metadata_count": len(rec.atomic_metadata_rows),
        "atomic_metadata_rows": rec.atomic_metadata_rows,
        "slide_guard_enabled": False,
        "pending_confirmed_rollback_fixed": False,
        "production_adoption": False,
    })
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    """fresh process内だけprechain shadowの構成点を差し替える。"""
    with contextlib.ExitStack() as stack:
        base.patch(stack, prechain, "shadow_prepare", atomic_prepare)
        base.patch(stack, prechain, "ShadowRecorder", AtomicRecorder)
        base.patch(stack, prechain, "shadow_instrument", atomic_instrument)
        base.patch(stack, prechain, "_shadow_summary", atomic_summary)
        return prechain.run(args)


def main() -> int:
    """新規rootとnative差の明示許可を必須にする。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=base.START_SEC)
    parser.add_argument("--end-sec", type=float, default=base.END_SEC)
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
