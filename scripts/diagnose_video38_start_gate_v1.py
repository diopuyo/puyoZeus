"""凍結履歴を変えず、開始時gateの既存呼出と返却だけを観測する。"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from scripts import diagnose_video38_accounting_history_v1 as history


base = history.base
FORMAT = "video38-start-gate-observation/v1"
FIRST_OBSERVED_FRAME, LAST_OBSERVED_FRAME = 32100, 32880
EXPECTED_OBSERVED_UPDATES = 391
HISTORY_SHA = "23836d917a97c4667c2a23b31ff863bc4e848c26d450506aa51134345b53768e"
REFERENCE_COMPLETE_SHA = "a706c178db8b807502a994c1b8584349832cb3708ae6ed627810126df7f42535"
REFERENCE = base.VERIFY / "video38_accounting_history_2026-09-07_v1"
TEST = base.ROOT / "tests/test_diagnose_video38_start_gate_v1.py"
LAUNCHER = base.ROOT / "scripts/launch_video38_start_gate_v1.sh"
EXTRA_NAME = "START_GATE_SUMMARY.json"
END_MODULE = base.ROOT / "scripts/chain_final_boundary_observation_v1.py"
END_TEST = base.ROOT / "tests/test_chain_final_boundary_observation_v1.py"
END_HASHES = {str(END_MODULE.resolve()): "183bf0b4ea95687cfa4f078a573b2a4f757a6a5a80c53aeb8b16a5fb83457984",
              str(END_TEST.resolve()): "804d2d253543787ed75883add2c5c51f8fe0452bf3fc275c6eea5eb04fc447d3"}
END_PREFIX = "end_boundary_"
END_FIRST_FRAME, END_LAST_FRAME = 35770, 35812
GATE_KINDS = frozenset(("start_gate_latch", "start_gate_board_return",
                        "start_gate_reset", "start_gate_result"))
LATCH_FIELDS = ("post_match_lockdown_active", "post_match_lockdown_prev_end_locked",
                "post_match_lockdown_started_time", "post_match_lockdown_raw_active_since",
                "last_match_end_locked", "match_end_locked_since", "last_active_frame_time",
                "match_active_started_time")
ORIGINAL_PREPARE = history.prepare
ORIGINAL_INSTRUMENT = history.instrument_pipeline
ORIGINAL_FINISH = history.finish


def in_window(rec: Any) -> bool:
    """535–548秒の両端を含む391 updateだけを追加観測する。"""
    return FIRST_OBSERVED_FRAME <= rec.frame <= LAST_OBSERVED_FRAME


def gate_snapshot(pipeline: Any) -> dict[str, Any]:
    """既存値を複製し、detectorや判定器を再評価しない。"""
    values = {name: getattr(pipeline, "_" + name) for name in LATCH_FIELDS}
    detector = pipeline._match_end_detector
    values["match_end_detector_last_detected_t"] = (
        None if detector is None else detector.last_detected_t)
    values["side_states"] = {
        side: str(getattr(getattr(pipeline, "_sm_" + side.lower()).context.state,
                          "value", getattr(pipeline, "_sm_" + side.lower()).context.state))
        for side in history.SIDES}
    return base.json_value(values)


def caller_site() -> dict[str, Any]:
    """wrapperの呼出元だけを読む。line traceを起動しない。"""
    current = inspect.currentframe()
    try:
        caller = current.f_back.f_back
        return {"source": caller.f_code.co_filename, "line": caller.f_lineno,
                "function": caller.f_code.co_name}
    finally:
        del current


def instrument_latch(stack: contextlib.ExitStack, cls: Any, rec: Any) -> None:
    """実引数・実変更を記録し、元のlatch更新を一回だけ呼ぶ。"""
    original = cls._update_post_match_lockdown_latch

    def latch(self: Any, frame: Any, time_sec: float, match_end_locked: bool,
              score_zero_both: bool, cur_score_1p: int | None,
              cur_score_2p: int | None, score_actively_moving: bool) -> Any:
        if not in_window(rec):
            return original(self, frame, time_sec, match_end_locked, score_zero_both,
                            cur_score_1p, cur_score_2p, score_actively_moving)
        values = dict(time_sec=time_sec, match_end_locked=match_end_locked,
                      score_zero_both=score_zero_both, cur_score_1p=cur_score_1p,
                      cur_score_2p=cur_score_2p, score_actively_moving=score_actively_moving)
        before, site, failure = gate_snapshot(self), caller_site(), None
        try:
            return original(self, frame, time_sec, match_end_locked, score_zero_both,
                            cur_score_1p, cur_score_2p, score_actively_moving)
        except BaseException as exc:
            failure = type(exc).__name__
            raise
        finally:
            rec.emit({"kind": "start_gate_latch", "arguments": values, "caller": site,
                      "before": before, "after": gate_snapshot(self), "exception": failure})

    base.patch(stack, cls, "_update_post_match_lockdown_latch", latch)


def instrument_board(stack: contextlib.ExitStack, cls: Any, rec: Any) -> None:
    """実際に呼ばれたpixel gateの返却だけを保存する。追加呼出は禁止。"""
    original = cls._board_shows_real_gameplay

    def board(self: Any, frame: Any) -> Any:
        if not in_window(rec):
            return original(self, frame)
        site = caller_site()
        try:
            result = original(self, frame)
        except BaseException as exc:
            rec.emit({"kind": "start_gate_board_return", "caller": site,
                      "exception": type(exc).__name__, "returned": None})
            raise
        rec.emit({"kind": "start_gate_board_return", "caller": site,
                  "exception": None, "returned": result})
        return result

    base.patch(stack, cls, "_board_shows_real_gameplay", board)


def instrument_reset(stack: contextlib.ExitStack, cls: Any, rec: Any) -> None:
    """実resetの前後を保存し、実引数・戻り値・例外を維持する。"""
    original = cls.reset

    def reset(self: Any, match_start_sec: float | None = None) -> Any:
        if not in_window(rec):
            return original(self, match_start_sec=match_start_sec)
        before, site, failure = gate_snapshot(self), caller_site(), None
        try:
            return original(self, match_start_sec=match_start_sec)
        except BaseException as exc:
            failure = type(exc).__name__
            raise
        finally:
            rec.emit({"kind": "start_gate_reset", "caller": site,
                      "arguments": {"match_start_sec": match_start_sec},
                      "before": before, "after": gate_snapshot(self), "exception": failure})

    base.patch(stack, cls, "reset", reset)


def instrument(stack: contextlib.ExitStack, collector: Any, rec: Any) -> None:
    """旧instrumentをそのまま通した後、追加hookだけを接続する。"""
    ORIGINAL_INSTRUMENT(stack, collector, rec)
    cls = collector.RecognitionPipeline
    original = cls.update
    instrument_latch(stack, cls, rec)
    instrument_board(stack, cls, rec)
    instrument_reset(stack, cls, rec)

    def update(self: Any, frame_idx: int, time_sec: float, frame: Any) -> Any:
        result = original(self, frame_idx, time_sec, frame)
        if in_window(rec):
            rec.emit({"kind": "start_gate_result", "is_match_active": result.is_match_active,
                      "match_end_locked": result.match_end_locked, "gate": gate_snapshot(self)})
        return result

    base.patch(stack, cls, "update", update)
    install_end(stack, collector, rec)


def install_end(stack: contextlib.ExitStack, collector: Any, rec: Any) -> None:
    """親所有moduleを凍結srcロード後、開始側hookの後に一度だけ接続する。"""
    # snapshotへcwd/pathを変えた後のscripts namespaceへ探索させない。
    name = "_video38_start_gate_end_observer"
    spec = importlib.util.spec_from_file_location(name, END_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError("親の終端観測moduleをロードできません")
    boundary = importlib.util.module_from_spec(spec)
    existed, previous = name in sys.modules, sys.modules.get(name)
    def restore() -> None:
        if existed:
            sys.modules[name] = previous
        else:
            sys.modules.pop(name, None)
    stack.callback(restore)
    sys.modules[name] = boundary
    spec.loader.exec_module(boundary)
    boundary.install(stack, collector, rec)


def reference_hashes() -> dict[str, str]:
    """旧COMPLETEの3artifactとreceipt自身を実照合する。"""
    if base.sha256(REFERENCE / "COMPLETE") != REFERENCE_COMPLETE_SHA:
        raise ValueError("旧history COMPLETEの固定SHAが違います")
    complete = base.read_json(REFERENCE / "COMPLETE")
    names = {"PLAN.json", "frames.jsonl", "SUMMARY.json"}
    if set(complete["sha256"]) != names:
        raise ValueError("旧history COMPLETEのartifact契約が違います")
    result = {str((REFERENCE / name).resolve()): digest
              for name, digest in complete["sha256"].items()}
    result[str((REFERENCE / "COMPLETE").resolve())] = base.sha256(REFERENCE / "COMPLETE")
    base.assert_unchanged(result)
    return result


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """固定historyと実参照を確認し、新規3fileを開始終了guardへ含める。"""
    if base.sha256(Path(history.__file__)) != HISTORY_SHA:
        raise ValueError("凍結history script SHAが一致しません")
    references = reference_hashes()
    base.assert_unchanged(END_HASHES)
    receipt, config = ORIGINAL_PREPARE(args)
    paths = (Path(__file__), TEST, LAUNCHER)
    receipt["input_and_code_sha256"].update(references)
    receipt["input_and_code_sha256"].update(END_HASHES)
    receipt["input_and_code_sha256"].update({str(p.resolve()): base.sha256(p) for p in paths})
    base.assert_unchanged(receipt["input_and_code_sha256"])
    receipt.update(format_version=FORMAT, observation_frame_range_inclusive=[
        FIRST_OBSERVED_FRAME, LAST_OBSERVED_FRAME], expected_observed_updates=EXPECTED_OBSERVED_UPDATES,
        observation_only=True, extra_detector_calls=False, start_gate_reference=str(REFERENCE),
        end_boundary_observation_frame_range_inclusive=[END_FIRST_FRAME, END_LAST_FRAME],
        legacy_comparison="全legacy行をkind/frame/side/occurrenceと全順序で文字列一致要求",
        decoded_clock_caveat="target_decoded_frame外側clockは直前update。actual_frameと混同しない")
    return receipt, config


def read_rows(path: Path, allow_gate: bool) -> tuple[list[Any], list[dict[str, Any]]]:
    """旧行の文字列を維持し、追加観測行だけを明示的に分離する。"""
    legacy, gate, occurrence = [], [], Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] in GATE_KINDS or row["kind"].startswith(END_PREFIX):
                if not allow_gate:
                    raise ValueError("旧対照に新gate行が混入しています")
                gate.append(row)
                continue
            key = (row["kind"], row["frame_idx"], row.get("side"))
            occurrence[key] += 1
            legacy.append(((*key, occurrence[key]), line.rstrip("\r\n")))
    return legacy, gate


def compare_legacy(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """追加計装が公開値/会計/採録を変えていないことを全旧行で拒否型検査する。"""
    old_path = REFERENCE / "frames.jsonl"
    before = {str(p): base.sha256(p) for p in (old_path, path)}
    old, _ = read_rows(old_path, False)
    new, gate = read_rows(path, True)
    if old != new:
        first = next((i for i, pair in enumerate(zip(old, new)) if pair[0] != pair[1]),
                     min(len(old), len(new)))
        raise ValueError(f"legacy bit exact不一致: index={first}, old={len(old)}, new={len(new)}")
    base.assert_unchanged(before)
    counts = Counter(key[0] for key, _ in old)
    return {"status": "all_legacy_rows_bit_exact", "legacy_row_count": len(old),
            "legacy_counts": dict(counts), "compared_sha256": before,
            "key": ["kind", "frame_idx", "side", "occurrence"], "global_order_equal": True,
            "decoded_clock_caveat": "target_decoded_frameは文字列一致のみ。clock母数に混ぜない"}, gate


def gate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """結果行391件と追加窓を検査する。呼ばれなかったresetを捏造しない。"""
    expected = set(range(FIRST_OBSERVED_FRAME, LAST_OBSERVED_FRAME + 1, history.STRIDE))
    result_frames = []
    for row in rows:
        frame = row["frame_idx"]
        if frame not in expected or abs(row["time_sec"] - frame / history.FPS) > 1e-8:
            raise ValueError("追加gate観測が窓/clock外です")
        if row.get("exception") is not None:
            raise ValueError("gate観測中に例外がありました")
        if row["kind"] == "start_gate_result":
            result_frames.append(frame)
    if len(result_frames) != EXPECTED_OBSERVED_UPDATES or set(result_frames) != expected:
        raise ValueError("追加resultの欠落/重複があります")
    counts = Counter(row["kind"] for row in rows)
    if counts["start_gate_latch"] != EXPECTED_OBSERVED_UPDATES or counts["start_gate_board_return"] == 0:
        raise ValueError("latch/pixelの実呼出観測がありません")
    return {"counts": dict(counts), "result_update_count": len(result_frames),
            "frame_range_inclusive": [FIRST_OBSERVED_FRAME, LAST_OBSERVED_FRAME],
            "reset_zero_means_no_observed_call": True, "individual_gate_cause_verified": False}


def end_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """親の終端窓は別母数で検査し、未呼出detector/helperを補完しない。"""
    expected = {(frame, side) for frame in range(END_FIRST_FRAME, END_LAST_FRAME + 1, history.STRIDE)
                for side in history.SIDES}
    keys: dict[str, list[Any]] = {"step_enter": [], "step_return": []}
    suffixes = {"step_enter", "step_return", "detector_return", "exit_helper_return"}
    for row in rows:
        frame, suffix = row["frame_idx"], row["kind"][len(END_PREFIX):]
        if suffix not in suffixes or not END_FIRST_FRAME <= frame <= END_LAST_FRAME:
            raise ValueError("終端観測kind/窓が不正です")
        if frame % history.STRIDE or abs(row["time_sec"] - frame / history.FPS) > 1e-8:
            raise ValueError("終端観測clockが不正です")
        if suffix in keys:
            keys[suffix].append((frame, row["side"]))
    if any(len(values) != len(expected) or set(values) != expected for values in keys.values()):
        raise ValueError("終端stepの欠落/重複があります")
    return {"counts": dict(Counter(row["kind"] for row in rows)), "step_frame_side_count": len(expected),
            "frame_range_inclusive": [END_FIRST_FRAME, END_LAST_FRAME],
            "uncalled_detectors_or_helpers_not_filled": True, "normal_release_verified": False}


def finish(output: Path, receipt: dict[str, Any], rec: Any, elapsed: float) -> dict[str, Any]:
    """旧finishのCOMPLETEを保留し、対照/追加receiptとSHAが揃ってから公開する。"""
    comparison, rows = compare_legacy(output / "frames.jsonl")
    start_rows = [row for row in rows if row["kind"] in GATE_KINDS]
    end_rows = [row for row in rows if row["kind"].startswith(END_PREFIX)]
    extra = {"format_version": FORMAT, "comparison": comparison, "observation": gate_summary(start_rows),
             "end_boundary_observation": end_summary(end_rows),
             "production_adoption": False, "accounting_basis_verified": False}
    original_write, pending = base.write_json, []

    def defer(path: Path, value: Any) -> None:
        if path == output / "COMPLETE":
            pending.append(value)
        else:
            original_write(path, value)

    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "write_json", defer)
        summary = ORIGINAL_FINISH(output, receipt, rec, elapsed)
    if len(pending) != 1:
        raise RuntimeError("旧finishの完了receiptが一意ではありません")
    base.write_json(output / EXTRA_NAME, extra)
    base.assert_unchanged(comparison["compared_sha256"])
    base.assert_unchanged(receipt["input_and_code_sha256"])
    hashes = dict(pending[0]["sha256"])
    hashes[EXTRA_NAME] = base.sha256(output / EXTRA_NAME)
    base.assert_unchanged({str(output / name): digest for name, digest in hashes.items()})
    base.write_json(output / "COMPLETE", {"format_version": FORMAT,
        "status": "observation_complete_not_adopted", "sha256": hashes})
    return {**summary, "start_gate": extra}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """既存runの排他/同一pipeline/復元を保ち、一時接続を必ず戻す。"""
    with contextlib.ExitStack() as stack:
        base.patch(stack, history, "prepare", prepare)
        base.patch(stack, history, "instrument_pipeline", instrument)
        base.patch(stack, history, "finish", finish)
        return history.run(args)


def main() -> int:
    """GPU起動は親のCPU検収後のみ。数値ライブラリは2 threadsに限定する。"""
    import cv2
    import torch

    cv2.setNumThreads(history.THREADS)
    torch.set_num_threads(history.THREADS)
    torch.set_num_interop_threads(history.THREADS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=history.START_SEC)
    parser.add_argument("--end-sec", type=float, default=history.END_SEC)
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
