"""C-6の未検証finalを内部状態へ書かず、既存transactionへ候補保存するshadow。"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import functools
import hashlib
import inspect
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

from scripts import diagnose_video38_prediction_ledger_shadow_v1 as prediction


base, completion = prediction.base, prediction.completion
FORMAT = "video38-c6-pending-commit-shadow/v1"
REFERENCE = base.VERIFY / "video38_prediction_ledger_shadow_2026-09-07_v1"
LAUNCHER = base.ROOT / "scripts/launch_video38_c6_pending_commit_shadow_v1.sh"
TEST = base.ROOT / "tests/test_diagnose_video38_c6_pending_commit_shadow_v1.py"
PENDING_NAME = "PENDING_RECEIPT.json"
ORIGINAL_PREPARE = prediction.ledger_prepare
ORIGINAL_INSTRUMENT = prediction.ledger_instrument
ORIGINAL_FINISH = prediction.ledger_finish


def _grid_equal(value: Any, grid: Any) -> bool:
    """Board系の公開値と候補tuple gridをcopyだけで比較する。"""
    if value is None:
        return False
    board = value.to_board() if hasattr(value, "to_board") else value
    actual = board.copy().to_dict()["grid"]
    return actual == [list(row) for row in grid]


def _public_board_value(value: Any) -> dict[str, Any] | None:
    """Board/ProbabilisticBoardを内部観測receipt用のBoardへ複製する。"""
    if value is None:
        return None
    board = value.to_board() if hasattr(value, "to_board") else value
    return base.board_value(board)


def _erased_counts(result: Any) -> Counter[int]:
    """既存simulate結果の通常色消去差分を再計算なしで集計する。"""
    counts: Counter[int] = Counter()
    for step in result.steps:
        for group in step.erased_groups:
            counts[int(group.color)] += int(group.size)
    return counts


def _side_snapshot(pipe: Any, side: str, ctx: Any) -> dict[str, Any]:
    """C-6が従来変更した対象と公開汚染先を同時に保存する。"""
    suffix = "1p" if side == "1P" else "2p"
    memory = getattr(pipe, f"_stable_color_memory_{suffix}")
    stable_memory = sorted((int(r), int(c), int(v)) for (r, c), v in memory.items())
    return {"state": base.json_value(ctx.state),
            "confirmed": base.board_value(ctx.confirmed_board),
            "pending": base.board_value(ctx.pending_board),
            "counter": dict(getattr(pipe, f"_tsumo_count_{suffix}")),
            "constraint_valid": getattr(pipe, f"_constraint_valid_{suffix}"),
            "old_verifier": base.json_value(getattr(pipe, f"_chain_verify_pending_{suffix}")),
            "prev_stable": base.board_value(getattr(pipe, f"_prev_stable_confirmed_{suffix}")),
            "stable_color_memory": stable_memory}


class PendingCommitRecorder(prediction.PredictionLedgerRecorder):
    """既存ledger行に、未適用transactionと公開隔離の証拠だけを加える。"""

    def __init__(self, stream: Any, target: dict[str, Any]) -> None:
        super().__init__(stream, target)
        self.commit_module: ModuleType | None = None
        self.active_candidates: dict[str, dict[str, Any]] = {}
        self.candidate_records: list[dict[str, Any]] = []
        self.c6_rows: list[dict[str, Any]] = []
        self.publication_rows: list[dict[str, Any]] = []
        self.transform_receipt: dict[str, Any] = {}
        self.board_state_module: ModuleType | None = None

    def bind_commit_runtime(self, board_state_module: ModuleType | None = None) -> None:
        """prediction adapterがguard-loadした同一commit moduleへだけ接続する。"""
        module = sys.modules.get("src.chain_commit_candidate_v1")
        if module is None or Path(str(module.__file__)).resolve() != prediction.COMMIT_SOURCE.resolve():
            raise RuntimeError("guard済みcommit moduleへ接続できません")
        board_state_module = board_state_module or sys.modules.get("src.board_state_machine")
        if board_state_module is None or not callable(
            getattr(board_state_module, "_apply_gravity_filter", None)
        ):
            raise RuntimeError("board_state_machine実行時moduleへ接続できません")
        self.commit_module = module
        self.board_state_module = board_state_module

    def record_generation(self, row: dict[str, Any]) -> None:
        """既存software行を保持し、世代差ではtransactionも一回だけ破棄する。"""
        super().record_generation(row)
        side = str(row["side"])
        record = self.active_candidates.get(side)
        if record is None:
            return
        identity = record["candidate"].identity
        changed = (row.get("reset_epoch") != identity.reset_epoch
                   or row.get("action_revision") != identity.action_revision)
        if changed:
            self._discard_candidate(
                side, "software_generation_changed_not_release_proof", row,
            )

    def record_c6_pending(
        self, pipe: Any, side: str, frame_idx: int, time_sec: float,
        ctx: Any, event: Any,
    ) -> None:
        """C-6を一度だけsimulateし、内部反映なしで候補とtransactionを保存する。"""
        before = _side_snapshot(pipe, side, ctx)
        result = pipe._chain_sim.simulate(event.before_board)
        final = result.final_board.copy() if result.final_board is not None else None
        if final is not None:
            self.board_state_module._apply_gravity_filter(final)
        old_episode_count = self._episode_count(side)
        self.record_c6_prediction(side, event, event.before_board, result, "_step_side/C-6")
        secured = self._store_candidate(side, event, result, final, old_episode_count)
        self._consume_stash(pipe, side)
        after = _side_snapshot(pipe, side, ctx)
        row = {"kind": "c6_pending_block", "side": side, "secured": secured,
               "before": before, "after": after,
               "blocked_effects": ["confirmed", "pending", "old_verifier",
                                   "color_counter", "constraint_valid"],
               "state_machine_rolled_back": False, "commit_permission_issued": False}
        self.c6_rows.append(base.json_value(row))
        self.emit(row)

    def isolate_side_result(self, side: str, result: Any) -> Any:
        """pending中は内部観測を保存し、公開2盤面だけを実際に隔離する。"""
        record = self.active_candidates.get(side)
        if record is None:
            return result
        candidate = record["candidate"]
        row = {"kind": "c6_pending_publication", "side": side,
               "candidate_id": candidate.candidate_id, "state": base.json_value(result.state),
               "confirmed_equals_candidate": _grid_equal(result.confirmed_board, candidate.final_grid),
               "prob_equals_candidate": _grid_equal(result.prob_board, candidate.final_grid),
               "pre_isolation_confirmed": _public_board_value(result.confirmed_board),
               "pre_isolation_prob": _public_board_value(result.prob_board),
               "returned_confirmed": None, "returned_prob": None,
               "hold_reason": "pending_candidate_not_verified",
               "state_machine_observation_preserved": True,
               "commit_permission_issued": False}
        self.publication_rows.append(base.json_value(row))
        self.emit(row)
        return dataclasses.replace(
            result, confirmed_board=None, prob_board=None,
            board_none_reason="pending_candidate_not_verified",
        )

    def isolate_pipeline_result(self, result: Any) -> Any:
        """collectorへ返すPipelineResultの左右を候補所有sideごとに隔離する。"""
        return dataclasses.replace(
            result,
            p1=self.isolate_side_result("1P", result.p1),
            p2=self.isolate_side_result("2P", result.p2),
        )

    def isolate_old_verifier(self, pipe: Any, side: str) -> tuple[str, None]:
        """pending中は旧断続票の補正を行わず、残留状態も隔離する。"""
        suffix = "1p" if side == "1P" else "2p"
        name = f"_chain_verify_pending_{suffix}"
        residual = getattr(pipe, name)
        setattr(pipe, name, None)
        row = {"kind": "c6_pending_old_verifier_isolated", "side": side,
               "candidate_id": self.active_candidates[side]["candidate"].candidate_id,
               "residual_present": residual is not None,
               "correction_applied": False, "commit_permission_issued": False}
        self.c6_rows.append(base.json_value(row))
        self.emit(row)
        return "pending_candidate_not_verified", None

    def _episode_count(self, side: str) -> int:
        handle = self.active_handles.get(side)
        return 0 if handle is None else len(self.ledger.snapshot(handle).episodes)

    def _store_candidate(
        self, side: str, event: Any, result: Any, final: Any, old_count: int,
    ) -> bool:
        """同instanceの最初のC-6候補だけに単一transactionを発行する。"""
        handle = self.active_handles.get(side)
        if handle is None or final is None or result.chain_count <= 0:
            self._candidate_reject(side, "landing_handle_or_physical_final_missing")
            return False
        snapshot = self.ledger.snapshot(handle)
        if self.active_handles.get(side) is not handle or snapshot.status.value != "provisional":
            self._candidate_reject(side, "ledger_handle_invalidated_during_c6_capture")
            return False
        if len(snapshot.episodes) != old_count + 1:
            self._candidate_reject(side, "c6_episode_was_not_exactly_appended")
            return False
        episode = snapshot.episodes[-1]
        if not _prediction_matches_call(snapshot, episode, event, result):
            self._candidate_reject(side, "c6_prediction_not_bound_to_actual_call")
            return False
        existing = self.active_candidates.get(side)
        if existing is not None and existing["instance_id"] == handle.instance_id:
            self._candidate_row("candidate_reused_without_reconstruction", existing)
            return True
        return self._create_candidate(side, event, result, final, snapshot)

    def _create_candidate(
        self, side: str, event: Any, result: Any, final: Any, snapshot: Any,
    ) -> bool:
        """既知software世代と最新episodeだけから未検証transactionを作る。"""
        generation = snapshot.generation
        if generation.reset_epoch is None or generation.action_revision is None:
            self._candidate_reject(side, "software_generation_unknown")
            return False
        episode = snapshot.episodes[-1]
        try:
            candidate = self.commit_module.create_chain_commit_candidate(
                side=side, reset_epoch=generation.reset_epoch,
                event_revision=episode.episode_revision,
                action_revision=generation.action_revision,
                created_frame_idx=self.frame, created_time_sec=self.time_sec,
                event=event, final_board=final,
                erased_color_count=_erased_counts(result),
            )
        except Exception as exc:
            self._candidate_reject(side, f"{type(exc).__name__}:{exc}")
            return False
        record = {"instance_id": snapshot.handle.instance_id, "candidate": candidate,
                  "transaction": self.commit_module.ChainCommitTransaction(candidate),
                  "status": "pending", "discard": None}
        self.active_candidates[side] = record
        self.candidate_records.append(record)
        self._candidate_row("candidate_created_pending_unverified", record)
        return True

    def _handle_ledger_error(
        self, stage: str, side: str, exc: Exception, detail: Any = None,
    ) -> None:
        """ledger失効に単一所有transactionも追従させる。"""
        super()._handle_ledger_error(stage, side, exc, detail)
        if side in self.active_candidates and side not in self.active_handles:
            active = self._clock_active()
            self._discard_candidate(side, "ledger_handle_invalidated", {
                "frame_idx": self.frame if active else None,
                "time_sec": self.time_sec if active else None,
                "clock_source": ("pipeline_update_input" if active
                                 else "outside_update_time_unknown"),
            })

    def _discard_candidate(
        self, side: str, reason: str, observation: dict[str, Any] | None = None,
    ) -> None:
        """世代差で未適用transactionを一回だけdiscardする。"""
        record = self.active_candidates.pop(side)
        receipt = record["transaction"].discard(reason)
        record["status"], record["discard"] = "discarded", receipt
        self._candidate_row("candidate_discarded", record, observation)

    def _candidate_reject(self, side: str, reason: str) -> None:
        row = {"kind": "c6_pending_candidate_rejected", "side": side, "reason": reason,
               "commit_permission_issued": False}
        self.c6_rows.append(base.json_value(row))
        self.emit(row)

    def _candidate_row(
        self, kind: str, record: dict[str, Any],
        observation: dict[str, Any] | None = None,
    ) -> None:
        row = {"kind": f"c6_pending_{kind}", "side": record["candidate"].identity.side,
               "instance_id": record["instance_id"],
               "candidate": prediction._json_value(record["candidate"]),
               "transaction_state": record["transaction"].state.value,
               "discard": prediction._json_value(record["discard"]),
               "generation_observation": base.json_value(observation),
               "commit_permission_issued": False}
        self.c6_rows.append(base.json_value(row))
        self.emit(row)

    @staticmethod
    def _consume_stash(pipe: Any, side: str) -> None:
        suffix = "1p" if side == "1P" else "2p"
        setattr(pipe, f"_last_chain_event_for_settle_{suffix}", None)

    def pending_receipt(self) -> dict[str, Any]:
        """未適用候補と全公開隔離行をowner objectなしで保存する。"""
        candidates = [{"instance_id": row["instance_id"],
            "candidate": prediction._json_value(row["candidate"]),
            "transaction_state": row["transaction"].state.value,
            "status": row["status"], "discard": prediction._json_value(row["discard"])}
            for row in self.candidate_records]
        return {"format_version": FORMAT, "status": "diagnostic_only_not_adopted",
                "candidates": candidates, "c6_rows": self.c6_rows,
                "publication_rows": self.publication_rows,
                "transform_receipt": self.transform_receipt,
                "normal_sm_and_raw_observation_preserved": True,
                "completion_or_release_policy_connected": False,
                "commit_permission_issued": False}


def _board_grid(board: Any) -> tuple[tuple[int, ...], ...]:
    """Boardを可変参照のないledger比較形式へ変換する。"""
    return tuple(tuple(int(value) for value in row)
                 for row in board.copy().to_dict()["grid"])


def _prediction_matches_call(
    snapshot: Any, episode: Any, event: Any, result: Any,
) -> bool:
    """今回のevent/input/resultが最新revisionと同一と検査する。"""
    if not snapshot.predictions or result.final_board is None:
        return False
    value = snapshot.predictions[-1]
    return (
        value.episode_revision == episode.episode_revision
        and episode.before_grid == _board_grid(event.before_board)
        and value.input_grid == _board_grid(event.before_board)
        and value.final_grid == _board_grid(result.final_board)
    )


def _is_c6_block(node: ast.If) -> bool:
    """C-6成功blockを名前・代入先・simulateの複合条件で一意に識別する。"""
    test = ast.unparse(node.test)
    body = "\n".join(ast.unparse(item) for item in node.body)
    markers = ("prev_state in", "BoardState.CHAIN", "BoardState.GRAVITY_SETTLE",
               "ctx.state == BoardState.STABLE", "_effective_chain_event is not None")
    effects = ("self._chain_sim.simulate", "ctx.confirmed_board = final",
               "ctx.pending_board = final.copy()", "_chain_verify_pending_1p",
               "target_tsumo", "_constraint_valid_1p")
    return all(marker in test for marker in markers) and all(effect in body for effect in effects)


def _hook_statement() -> ast.stmt:
    """C-6 block全体を一つの候補保存callへ置換するASTを作る。"""
    names = ("self", "side", "frame_idx", "time_sec", "ctx", "_effective_chain_event")
    call = ast.Call(func=ast.Name(id="__c6_pending_shadow_hook", ctx=ast.Load()),
                    args=[ast.Name(id=name, ctx=ast.Load()) for name in names], keywords=[])
    return ast.Expr(value=call)


def _build_transformed_step(original: Any, hook: Any) -> tuple[Any, dict[str, Any]]:
    """元methodのC-6 block一件だけを置換し、元行番号を保ってcompileする。"""
    lines, first = inspect.getsourcelines(original)
    source = textwrap.dedent("".join(lines))
    tree = ast.parse(source)
    targets = [node for node in ast.walk(tree)
               if isinstance(node, ast.If) and _is_c6_block(node)]
    if len(targets) != 1:
        raise RuntimeError(f"C-6 AST patternは1件必要です: {len(targets)}")
    target = targets[0]
    target.body = [ast.copy_location(_hook_statement(), target.body[0])]
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, first - 1)
    namespace = dict(original.__globals__)
    namespace["__c6_pending_shadow_hook"] = hook
    exec(compile(tree, original.__code__.co_filename, "exec"), namespace)
    transformed = functools.update_wrapper(namespace[original.__name__], original)
    receipt = {"source_file": original.__code__.co_filename,
               "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "first_line": first, "matched_blocks": 1,
               "replacement": "C6 success block -> pending candidate hook",
               "state_machine_rollback": False}
    return transformed, receipt


def install_pending_transform(
    stack: contextlib.ExitStack, cls: Any, rec: PendingCommitRecorder,
) -> None:
    """診断process内だけ_step_sideを一意AST変換し、終了時に復元する。"""
    original = cls._step_side

    def hook(pipe: Any, side: str, frame_idx: int, time_sec: float,
             ctx: Any, event: Any) -> None:
        rec.record_c6_pending(pipe, side, frame_idx, time_sec, ctx, event)

    transformed, receipt = _build_transformed_step(original, hook)
    rec.transform_receipt = receipt
    base.patch(stack, cls, "_step_side", transformed)
    rec.step_code = transformed.__code__


def install_pending_publication(
    stack: contextlib.ExitStack, cls: Any, rec: PendingCommitRecorder,
) -> None:
    """frame_side計装より内側で、実返却PipelineResultを隔離する。"""
    original_update = cls.update
    @functools.wraps(original_update)
    def update(pipe: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_update(pipe, *args, **kwargs)
        return rec.isolate_pipeline_result(result)
    base.patch(stack, cls, "update", update)


def install_pending_verifier(
    stack: contextlib.ExitStack, cls: Any, rec: PendingCommitRecorder,
) -> None:
    """pending sideの旧断続票を破棄し、補正盤面を返さない。"""
    original = cls._update_chain_estimate_verification

    @functools.wraps(original)
    def verify(pipe: Any, side: str, *args: Any, **kwargs: Any) -> Any:
        if side in rec.active_candidates:
            return rec.isolate_old_verifier(pipe, side)
        return original(pipe, side, *args, **kwargs)

    base.patch(stack, cls, "_update_chain_estimate_verification", verify)


def pending_prepare(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """既存全guardと直前ledger rootへ新規3資産を追加する。"""
    receipt, config = ORIGINAL_PREPARE(args)
    reference_receipt, hashes = completion.prechain._reference_receipt(REFERENCE)
    paths = (Path(__file__), TEST, LAUNCHER)
    receipt["input_and_code_sha256"].update(hashes)
    receipt["input_and_code_sha256"].update(
        {str(path): base.sha256(path) for path in paths},
    )
    receipt["c6_pending_commit_shadow"] = {
        "format_version": FORMAT, "reference": reference_receipt,
        "ast_expected_match_count": 1,
        "one_axis": "block unverified C6 internal/public/accounting mutation",
        "preserved": ["SM.update", "raw/filtered observation", "stable_color_memory before C6",
                      "normal landing Counter increment"],
        "blocked": ["confirmed", "pending", "old verifier", "color Counter decrement",
                    "constraint re-enable", "candidate publication via confirmed/prob"],
        "state_machine_rollback": False, "release_policy_connected": False,
        "production_modified": False, "commit_permission_issued": False}
    return receipt, config


def pending_instrument(
    stack: contextlib.ExitStack, collector: ModuleType, rec: PendingCommitRecorder,
) -> None:
    """既存shadowのglobal差替え後にC-6変換と公開隔離を重ねる。"""
    cls = collector.RecognitionPipeline
    install_pending_publication(stack, cls, rec)
    ORIGINAL_INSTRUMENT(stack, collector, rec)
    install_pending_transform(stack, cls, rec)
    install_pending_verifier(stack, cls, rec)
    rec.bind_commit_runtime(sys.modules.get("src.board_state_machine"))


def _intentional_prefix_comparison(candidate: Path) -> dict[str, Any]:
    """旧atomic prefixの差を例外化せず、欠落と内容差に分ける。"""
    reference = completion.ATOMIC_REFERENCE / "frames.jsonl"
    results: dict[str, Any] = {}
    for kind in ("frame_side", "collector_snapshot"):
        old = _window_index(reference, kind, 560.0, 583.0)
        new = _window_index(candidate, kind, 560.0, 583.0)
        common = sorted(set(old) & set(new))
        changed = [key for key in common if old[key] != new[key]]
        results[kind] = {
            "reference_count": len(old), "candidate_prefix_count": len(new),
            "missing_keys": sorted(set(old) - set(new)),
            "added_keys": sorted(set(new) - set(old)),
            "changed_count": len(changed),
            "first_changed_key": changed[0] if changed else None,
            "prefix_bit_exact": old == new,
        }
    return {"scope": "560<=time<583; intentional shadow differences retained",
            "comparison_bypassed": False, "kinds": results}


def _is_pending_kind(kind: Any) -> bool:
    return str(kind).startswith("c6_pending_")


def _kind_rows(path: Path, *, candidate: bool) -> dict[str, list[str]]:
    """全kindを除外なしの分母へ分け、pending追加行だけ比較外にする。"""
    values: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            kind = str(row.get("kind"))
            if candidate and _is_pending_kind(kind):
                continue
            values.setdefault(kind, []).append(json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ))
    return values


def _indexed_rows(path: Path, *, candidate: bool = False) -> dict[str, dict[str, Any]]:
    """kind/frame/side/同frame順で全行を一意化する。"""
    values: dict[str, dict[str, Any]] = {}
    seen: Counter[tuple[Any, ...]] = Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if candidate and _is_pending_kind(row.get("kind")):
                continue
            base_key = (row.get("kind"), row.get("frame_idx"), row.get("side"))
            occurrence = seen[base_key]
            seen[base_key] += 1
            key = json.dumps((*base_key, occurrence), ensure_ascii=False)
            values[key] = row
    return values


def _window_index(
    path: Path, kind: str, start_sec: float, end_sec: float,
) -> dict[str, str]:
    """行数sliceでなく実time窓でprefixを切り出す。"""
    rows = _indexed_rows(path)
    return {key: json.dumps(row, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"))
            for key, row in rows.items()
            if row.get("kind") == kind and row.get("time_sec") is not None
            and start_sec <= float(row["time_sec"]) < end_sec}


def _window_stats(
    rows: dict[str, dict[str, Any]], side: str, start: int, end: int,
) -> dict[str, Any]:
    """正常対照窓の実返却行とcollector採録を数える。"""
    side_rows = [row for row in rows.values() if row.get("kind") == "frame_side"
                 and row.get("side") == side
                 and start <= int(row.get("frame_idx", -1)) <= end]
    collectors = [row for row in rows.values() if row.get("kind") == "collector_snapshot"
                  and row.get("side") == side
                  and start <= int(row.get("frame_idx", -1)) <= end]
    return {"side": side, "start_frame": start, "end_frame": end,
            "frame_side_count": len(side_rows),
            "stable_count": sum(str(row.get("state")).lower() == "stable"
                                for row in side_rows),
            "collector_count": len(collectors),
            "collector_frames": [row.get("frame_idx") for row in collectors]}


def _coverage(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """固定45秒の必須左右行に欠落・重複がないか検査する。"""
    result: dict[str, Any] = {}
    expected = {(frame, side) for frame in range(33600, 36300, 2)
                for side in ("1P", "2P")}
    for kind in ("frame_side", "raw_score_ocr", "completion_frame_observation"):
        selected = [row for row in rows.values() if row.get("kind") == kind]
        keys = [(row.get("frame_idx"), row.get("side")) for row in selected]
        actual = set(keys)
        result[kind] = {"expected": 2700, "actual": len(selected),
                        "unique_frame_side": len(actual),
                        "duplicate_count": len(keys) - len(actual),
                        "missing_frame_side": sorted(expected - actual),
                        "extra_frame_side": sorted(actual - expected),
                        "complete": len(selected) == len(actual) == 2700
                                    and actual == expected}
    return result


def _raw_field(row: dict[str, Any]) -> Any:
    """会計filteredと分け、各kindの独立raw正本だけを返す。"""
    kind = row.get("kind")
    if kind == "raw_score_ocr":
        return row.get("raw_value")
    if kind == "frame_side":
        return (row.get("accounting_capture") or {}).get("raw")
    if kind == "completion_frame_observation":
        return row.get("raw_before_accounting")
    return None


def _difference_receipt(candidate: Path) -> dict[str, Any]:
    """直前runとの差を全kind・全位置で数え、意図差も失敗も除外しない。"""
    expected = _indexed_rows(REFERENCE / "frames.jsonl")
    actual = _indexed_rows(candidate, candidate=True)
    common = sorted(set(expected) & set(actual))
    changed = [key for key in common if expected[key] != actual[key]]
    raw_kinds = ("raw_score_ocr", "frame_side", "completion_frame_observation")
    raw_changed = {kind: sum(_raw_field(expected[key]) != _raw_field(actual[key])
                             for key in changed if expected[key].get("kind") == kind)
                   for kind in raw_kinds}
    windows = {"normal_1p_cold": ("1P", 34080, 34380),
               "normal_1p_landing": ("1P", 34700, 34792),
               "target_2p_completion": ("2P", 35704, 36298)}
    return {"reference": str(REFERENCE), "excluded_new_pending_rows_only": True,
            "all_reference_and_candidate_rows_counted": True,
            "reference_count": len(expected), "actual_count": len(actual),
            "missing_keys": sorted(set(expected) - set(actual)),
            "added_keys": sorted(set(actual) - set(expected)),
            "changed_count": len(changed), "changed_keys": changed,
            "independent_raw_field_changed": raw_changed,
            "coverage": _coverage(actual),
            "windows": {name: {"reference": _window_stats(expected, *spec),
                               "candidate": _window_stats(actual, *spec)}
                        for name, spec in windows.items()},
            "release_policy_exercised": False}


def pending_finish(
    output: Path, receipt: dict[str, Any], rec: PendingCommitRecorder, elapsed: float,
) -> dict[str, Any]:
    """候補receiptと全差分を保存し、既存7 artifact SHA連鎖へ追加する。"""
    result = rec.pending_receipt()
    result["full_difference_from_ledger_shadow"] = _difference_receipt(
        output / "frames.jsonl",
    )
    base.assert_unchanged(receipt["input_and_code_sha256"])
    base.write_json(output / PENDING_NAME, result)
    original_write = base.write_json

    def write(path: Path, value: Any) -> None:
        if path.name == "COMPLETE":
            value = dict(value)
            value["sha256"] = {**value["sha256"], PENDING_NAME:
                               base.sha256(output / PENDING_NAME)}
        original_write(path, value)

    def intentional(path: Path) -> dict[str, Any]:
        return {"all_legacy_rows_bit_exact": False,
                "expected_intentional_c6_difference": True,
                "full_difference_receipt": PENDING_NAME,
                "row_count": sum(len(rows) for rows in _kind_rows(path, candidate=True).values())}

    with contextlib.ExitStack() as stack:
        base.patch(stack, base, "write_json", write)
        base.patch(stack, prediction, "_full_legacy_comparison", intentional)
        base.patch(stack, completion, "_prefix_comparison", _intentional_prefix_comparison)
        summary = ORIGINAL_FINISH(output, receipt, rec, elapsed)
    return {**summary, "c6_pending_commit_receipt": result}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """既存prediction診断の4拡張点だけを一時差替えする。"""
    with contextlib.ExitStack() as stack:
        base.patch(stack, prediction, "ledger_prepare", pending_prepare)
        base.patch(stack, prediction, "PredictionLedgerRecorder", PendingCommitRecorder)
        base.patch(stack, prediction, "ledger_instrument", pending_instrument)
        base.patch(stack, prediction, "ledger_finish", pending_finish)
        return prediction.run(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=prediction.START_SEC)
    parser.add_argument("--end-sec", type=float, default=prediction.END_SEC)
    parser.add_argument("--allow-native-runtime-mismatch", action="store_true")
    completion._configure_torch_threads()
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
