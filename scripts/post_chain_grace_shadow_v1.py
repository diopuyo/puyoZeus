"""連鎖終了世代の landing grace 再注入だけを止める既定OFF shadow。"""

from __future__ import annotations

import ast
import copy
import contextlib
import functools
import inspect
import math
import textwrap
from dataclasses import dataclass
from typing import Any, Callable

from scripts import diagnose_video38_c6_pending_commit_shadow_v1 as pending


SIDES = ("1P", "2P")
GUARD_PATHS: tuple[str, ...] = (
    "scripts/diagnose_video38_c6_pending_commit_shadow_v1.py",
)


@dataclass(frozen=True)
class GraceOwner:
    """連鎖終了時の既存ledger/software世代を固定する。"""

    side: str
    instance_id: int
    reset_epoch: int
    action_revision: int


@dataclass
class _StashScope:
    """実stash呼出中だけmutation候補を保持する。"""

    pipe: Any
    side: str
    active: Any
    frame_idx: int
    time_sec: float
    handle: Any
    instance_id: int
    generation: tuple[int, int]
    candidate: dict[str, Any]
    transaction: Any
    mutation: dict[str, Any] | None = None


class PostChainGraceController:
    """成功した連鎖終了から新actionまで、旧graceだけを隔離する。"""

    def __init__(self, rec: Any, write: Callable[[dict[str, Any]], Any]) -> None:
        self.rec, self._write = rec, write
        self.owners: dict[str, GraceOwner] = {}
        self._released_at: dict[str, tuple[int, float]] = {}
        self.transform_receipt: dict[str, Any] = {}
        self._stash_scope: _StashScope | None = None
        self._pipe: Any = None

    def observe(self, row: dict[str, Any]) -> None:
        """同一handleの成功chain-end mutationだけをownerにする。"""
        scope = self._stash_scope
        parsed = self._success_identity(row)
        if scope is None or parsed is None or parsed[0] != scope.side:
            return
        if scope.mutation is not None:
            raise RuntimeError("同じstash呼出に複数の成功mutationがあります")
        scope.mutation = dict(row)

    def run_stash(self, original: Any, pipe: Any, side: str) -> Any:
        """実activeを持つ同期stashの正常完了後だけownerを確定する。"""
        if self._stash_scope is not None:
            raise RuntimeError("stashの再入を許可しません")
        active = self._active(pipe, side)
        point = self._clock_point()
        identity = self._stash_identity(side)
        if (active is None or point is None or identity is None
                or getattr(self.rec.generation_recorder, "_pipeline", None) is not pipe
                or (self._pipe is not None and pipe is not self._pipe)):
            return original(pipe, side)
        scope = _StashScope(pipe, side, active, *point, *identity)
        self._stash_scope = scope
        try:
            result = original(pipe, side)
            if self._active(pipe, side) is None and self._clock_point() == point:
                self._activate(scope)
            return result
        finally:
            self._stash_scope = None

    def _activate(self, scope: _StashScope) -> None:
        """同じ呼出中のmutationとpending ownerを再照合する。"""
        parsed = self._success_identity(scope.mutation or {})
        if (parsed is None or parsed != (scope.side, scope.instance_id)
                or not self._scope_identity_current(scope)):
            return
        side, instance_id = parsed
        self.before_step(scope.pipe, side, scope.frame_idx, scope.time_sec)
        generation = self._bound_generation(side, instance_id, scope)
        if generation is None:
            self._emit(side, "owner_generation_unavailable", instance_id)
            return
        owner = GraceOwner(side, instance_id, *generation)
        existing = self.owners.get(side)
        if existing is not None and existing != owner:
            self._emit(side, "owner_conflict_fail_closed", instance_id)
            return
        if self._pipe is None:
            self._pipe = scope.pipe
        self.owners[side] = owner
        self._emit(side, "post_chain_owner_started", instance_id)

    def _stash_identity(self, side: str) -> tuple[Any, int, tuple[int, int], dict[str, Any], Any] | None:
        """stash前のhandle/candidate実参照を一組として固定する。"""
        handle = self.rec.active_handles.get(side)
        instance_id = getattr(handle, "instance_id", None)
        generation = self._bound_generation(side, instance_id)
        candidate = self.rec.active_candidates.get(side)
        if generation is None or not isinstance(candidate, dict):
            return None
        return handle, instance_id, generation, candidate, candidate["transaction"]

    def _scope_identity_current(self, scope: _StashScope) -> bool:
        """stash途中のhandle/candidate差替えを同値偽装でも拒否する。"""
        candidate = self.rec.active_candidates.get(scope.side)
        return (self.rec.active_handles.get(scope.side) is scope.handle
                and candidate is scope.candidate
                and candidate.get("transaction") is scope.transaction
                and self._bound_generation(scope.side, scope.instance_id, scope)
                == scope.generation)

    def _active(self, pipe: Any, side: str) -> Any:
        """左右以外を推測補完せず実active参照だけを返す。"""
        if side not in SIDES:
            return None
        return getattr(pipe, f"_active_chain_{side.lower()}", None)

    def _clock_point(self) -> tuple[int, float] | None:
        """recorderの実update時計だけを同期呼出へ束縛する。"""
        if hasattr(self.rec, "_clock_active") and not self.rec._clock_active():
            return None
        frame, value = getattr(self.rec, "frame", None), getattr(self.rec, "time_sec", None)
        if (type(frame) is not int or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value)):
            return None
        return frame, float(value)

    def before_step(self, pipe: Any, side: str, frame_idx: int, time_sec: float) -> None:
        """実step入口で世代を同期し、owner中の旧graceを失効する。"""
        if (side not in SIDES or side not in self.owners
                or (self._pipe is not None and pipe is not self._pipe)):
            return
        if self._generation_changed(side):
            owner = self.owners.pop(side)
            self._released_at[side] = (frame_idx, float(time_sec))
            self._clear_grace(pipe, side)
            self._emit(side, "software_generation_changed", owner.instance_id)
            return
        if self._clear_grace(pipe, side):
            self._emit(side, "stale_grace_invalidated", self.owners[side].instance_id)

    def allow_generation(
        self, pipe: Any, target_side: str, runtime_side: str,
        frame_idx: int, time_sec: float,
    ) -> bool:
        """owner中または解除同frameのgrace生成だけを拒否する。"""
        if target_side != runtime_side or target_side not in SIDES:
            raise RuntimeError("grace代入sideが実step sideと一致しません")
        if self._pipe is not None and pipe is not self._pipe:
            return True
        self.before_step(pipe, target_side, frame_idx, time_sec)
        released = self._released_at.get(target_side) == (frame_idx, float(time_sec))
        blocked = target_side in self.owners or released
        if blocked:
            instance = getattr(self.owners.get(target_side), "instance_id", None)
            self._emit(target_side, "post_chain_grace_generation_blocked", instance)
        return not blocked

    def _success_identity(self, row: dict[str, Any]) -> tuple[str, int] | None:
        """観測行を自由boolでなく構造化済み成功mutationとして検査する。"""
        if row.get("kind") != "boundary_repair_mutation" or row.get("repair") != "chain_end":
            return None
        side, evidence = row.get("side"), row.get("evidence")
        instance_id = evidence.get("instance_id") if isinstance(evidence, dict) else None
        before, after = row.get("before"), row.get("after")
        if (side not in SIDES or type(instance_id) is not int
                or not isinstance(before, dict) or before.get("active") is None
                or not isinstance(after, dict) or after.get("active", object()) is not None):
            return None
        return side, instance_id

    def _bound_generation(
        self, side: str, instance_id: int, scope: _StashScope | None = None,
    ) -> tuple[int, int] | None:
        """現handleとledger snapshotが同一instanceの時だけ世代を得る。"""
        if hasattr(self.rec, "_clock_active") and not self.rec._clock_active():
            return None
        handle = self.rec.active_handles.get(side)
        if handle is None or getattr(handle, "instance_id", None) != instance_id:
            return None
        if scope is not None and handle is not scope.handle:
            return None
        snapshot = self.rec.ledger.snapshot(handle)
        status = getattr(snapshot.status, "value", snapshot.status)
        candidate = self.rec.active_candidates.get(side)
        candidate_state = getattr(getattr(candidate.get("transaction"), "state", None),
                                  "value", None) if isinstance(candidate, dict) else None
        if (getattr(handle, "side", None) != side or status != "provisional"
                or not isinstance(candidate, dict)
                or candidate.get("instance_id") != instance_id
                or candidate.get("status") != "pending" or candidate_state != "pending"):
            return None
        if (scope is not None and (candidate is not scope.candidate
                                   or candidate.get("transaction") is not scope.transaction)):
            return None
        generation = snapshot.generation
        reset, action = generation.reset_epoch, generation.action_revision
        if type(reset) is not int or type(action) is not int:
            return None
        current = self.rec.generation_recorder.generation(side)
        if (current.reset_epoch, current.action_revision) != (reset, action):
            return None
        return reset, action

    def _generation_changed(self, side: str) -> bool:
        """UNKNOWN補完なしで現在software世代との差だけを見る。"""
        owner = self.owners[side]
        current = self.rec.generation_recorder.generation(side)
        reset, action = current.reset_epoch, current.action_revision
        if type(reset) is not int or (action is not None and type(action) is not int):
            return False
        if reset != owner.reset_epoch:
            return True
        return type(action) is int and action != owner.action_revision

    def _clear_grace(self, pipe: Any, side: str) -> bool:
        """landing_pending等へ触れず対象sideのgraceだけを消す。"""
        name = f"_landing_grace_{side.lower()}"
        existed = getattr(pipe, name, None) is not None
        setattr(pipe, name, None)
        return existed

    def _emit(self, side: str, reason: str, instance_id: int | None) -> None:
        """既存recorder時計で診断行だけを保存する。"""
        self._write({"kind": "post_chain_grace_observation", "side": side,
                     "instance_id": instance_id, "reason": reason,
                     "state_or_release_modified": False,
                     "commit_permission_issued": False})


class _GraceTransformer(ast.NodeTransformer):
    """landing grace代入2箇所だけへ生成許可guardを付ける。"""

    def __init__(self) -> None:
        self.matched: list[str] = []

    def visit_Assign(self, node: ast.Assign) -> ast.stmt:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Attribute):
            return node
        name = node.targets[0].attr
        side = {"_landing_grace_1p": "1P", "_landing_grace_2p": "2P"}.get(name)
        if side is None or not isinstance(node.value, ast.Tuple):
            return node
        self.matched.append(side)
        call = ast.Call(ast.Name("__post_chain_grace_allow", ast.Load()),
                        [ast.Name("self", ast.Load()), ast.Constant(side),
                         ast.Name("side", ast.Load()), ast.Name("frame_idx", ast.Load()),
                         ast.Name("time_sec", ast.Load())], [])
        # 同じtarget nodeを共有するとincrement_linenoが二重適用され、実trace行が壊れる。
        clear = ast.Assign(targets=[copy.deepcopy(node.targets[0])], value=ast.Constant(None))
        return ast.copy_location(ast.If(test=call, body=[node], orelse=[clear]), node)


def _composite_step(current: Any, controller: PostChainGraceController) -> tuple[Any, dict[str, Any]]:
    """既存C6 pending置換を保持し、grace guardを同じASTへ合成する。"""
    hook = current.__globals__.get("__c6_pending_shadow_hook")
    if not callable(hook):
        raise RuntimeError("C6 pending transformを先に接続してください")
    lines, first = inspect.getsourcelines(current)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    c6 = [node for node in ast.walk(tree) if isinstance(node, ast.If)
          and pending._is_c6_block(node)]
    if len(c6) != 1:
        raise RuntimeError(f"C6 pending AST patternは1件必要です: {len(c6)}")
    c6[0].body = [ast.copy_location(pending._hook_statement(), c6[0].body[0])]
    transformer = _GraceTransformer()
    tree = transformer.visit(tree)
    if sorted(transformer.matched) != ["1P", "2P"]:
        raise RuntimeError(f"landing grace代入は左右1件ずつ必要です: {transformer.matched}")
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    prelude = ast.Expr(ast.Call(ast.Name("__post_chain_grace_before_step", ast.Load()),
                               [ast.Name("self", ast.Load()), ast.Name("side", ast.Load()),
                                ast.Name("frame_idx", ast.Load()),
                                ast.Name("time_sec", ast.Load())], []))
    index = 1 if (function.body and isinstance(function.body[0], ast.Expr)
                  and isinstance(function.body[0].value, ast.Constant)) else 0
    function.body.insert(index, prelude)
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, first - 1)
    namespace = dict(current.__globals__)
    namespace.update({"__c6_pending_shadow_hook": hook,
                      "__post_chain_grace_before_step": controller.before_step,
                      "__post_chain_grace_allow": controller.allow_generation})
    exec(compile(tree, current.__code__.co_filename, "exec"), namespace)
    result = functools.update_wrapper(namespace[current.__name__], current)
    return result, {"c6_pending_blocks": 1, "grace_assignments": transformer.matched,
                    "landing_pending_modified": False, "counter_modified": False}


def _patch_instance(stack: contextlib.ExitStack, obj: Any, name: str, value: Any) -> None:
    """instance属性の有無を含めて終了時に復元する。"""
    existed, old = name in vars(obj), vars(obj).get(name)
    setattr(obj, name, value)
    if existed:
        stack.callback(setattr, obj, name, old)
    else:
        stack.callback(vars(obj).pop, name, None)


def install(stack: contextlib.ExitStack, collector: Any, rec: Any) -> PostChainGraceController:
    """pending+chain-end計装後へ一軸shadowを接続する。"""
    required = ("active_handles", "active_candidates", "ledger",
                "generation_recorder", "emit")
    if any(not hasattr(rec, name) for name in required) or rec.generation_recorder is None:
        raise RuntimeError("ledger/generation/pending runtimeが先に必要です")
    original_emit = rec.emit
    controller = PostChainGraceController(rec, original_emit)

    def emit(row: dict[str, Any]) -> Any:
        result = original_emit(row)
        controller.observe(row)
        return result

    _patch_instance(stack, rec, "emit", emit)
    cls = collector.RecognitionPipeline
    current = cls._step_side
    transformed, receipt = _composite_step(current, controller)
    controller.transform_receipt = receipt
    pending.base.patch(stack, cls, "_step_side", transformed)
    _patch_instance(stack, rec, "step_code", transformed.__code__)
    original_stash = cls._stash_and_clear_active_chain

    @functools.wraps(original_stash)
    def stash(pipe: Any, side: str) -> Any:
        return controller.run_stash(original_stash, pipe, side)

    pending.base.patch(stack, cls, "_stash_and_clear_active_chain", stash)
    return controller


__all__ = ["GUARD_PATHS", "GraceOwner", "PostChainGraceController", "install"]
