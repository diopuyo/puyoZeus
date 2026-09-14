"""既存pipelineのソフトウェア世代を、挙動を変えず外側へ記録する。

reset成功とstate machineのTSUMO_FALL進入を記録するだけで、物理的な連鎖終了・
着手の正しさ・全resync経路の網羅を証明しない。現在pipelineや本番設定からは未接続。
初回の着手進入を観測するまではaction_revision=None。推定で0補完しない。
新しい台帳へ配線する前に、固定診断の全公開行が不変であることを別途検収する。
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable

from scripts.diagnose_video38_confirmed_collapse_v1 import patch


SIDES = ("1P", "2P")
TSUMO_FALL_STATE = "tsumo_fall"
IDENTITY_SCOPE = "software_observation_only_not_physical_identity"


@dataclass(frozen=True)
class SoftwareGeneration:
    """追跡開始後に観測したソフトウェアの版。物理的な認証tokenではない。"""

    side: str
    reset_epoch: int
    action_revision: int | None
    identity_scope: str = IDENTITY_SCOPE


class PipelineGenerationRecorder:
    """一つのfresh pipelineだけに対応する、外側の世代記録器。"""

    def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self._emit = emit
        self._pipeline: Any = None
        self._machines: dict[str, Any] = {}
        self._generation: dict[str, SoftwareGeneration] = {}
        self._frame: int | None = None
        self._time: float | None = None
        self._in_frame = False
        self._sequence = 0

    def bind_pipeline(self, pipeline: Any) -> None:
        """fresh loadの成功後に左右を対応付ける。履歴を復元したとは扱わない。"""
        if self._pipeline is not None:
            raise ValueError("記録器は一つのfresh pipelineにのみ接続できます")
        machines = {side: getattr(pipeline, f"_sm_{side.lower()}") for side in SIDES}
        if any(value is None for value in machines.values()):
            raise ValueError("両sideのstate machineが必要です")
        if machines["1P"] is machines["2P"]:
            raise ValueError("左右のstate machineは独立である必要があります")
        self._pipeline, self._machines = pipeline, machines
        for side in SIDES:
            self._generation[side] = SoftwareGeneration(side, 0, None)
            self._record(side, "fresh_pipeline_bound")

    def begin_frame(self, pipeline: Any, frame_idx: int, time_sec: float) -> None:
        """元updateより先に時計を固定し、同frame内reset/actionの順序を保存する。"""
        if pipeline is not self._pipeline or pipeline is None:
            raise ValueError("未接続または別のpipelineです")
        if self._in_frame:
            raise ValueError("pipeline updateの再入を拒否します")
        if type(frame_idx) is not int or frame_idx < 0:
            raise ValueError("frameは0以上の整数が必要です")
        if isinstance(time_sec, bool) or not isinstance(time_sec, (int, float)):
            raise ValueError("時刻は有限数値が必要です")
        if not math.isfinite(time_sec) or time_sec < 0:
            raise ValueError("時刻は有限かつ0以上が必要です")
        if self._frame is not None and (frame_idx < self._frame or time_sec < self._time):
            raise ValueError("観測時計の逆行を拒否します")
        self._frame, self._time = frame_idx, float(time_sec)
        self._in_frame = True

    def end_frame(self) -> None:
        """例外時も現在時計を非activeにし、外部resetへ流用しない。"""
        self._in_frame = False

    def generation(self, side: str) -> SoftwareGeneration:
        """不変snapshotを返す。UNKNOWN actionを既知へ変換しない。"""
        if side not in self._generation:
            raise ValueError("未接続または不正なsideです")
        return self._generation[side]

    def record_reset(self, machine: Any) -> None:
        """元resetが成功した場合だけ、そのsideの版を進める。"""
        side = self._side_for(machine)
        if side is None:
            return
        previous = self.generation(side)
        self._generation[side] = SoftwareGeneration(side, previous.reset_epoch + 1, None)
        self._record(side, "state_machine_reset_succeeded")

    def record_transition(self, machine: Any, before: Any, after: Any) -> None:
        """実ソフトウェア遷移のみを記録し、NEXT値やtimeoutから着手を補完しない。"""
        side = self._side_for(machine)
        if side is None:
            return
        old, new = _state_value(before), _state_value(after)
        if old == new or new != TSUMO_FALL_STATE:
            return
        if not self._in_frame:
            raise ValueError("時計なしの着手進入を記録できません")
        previous = self.generation(side)
        revision = 1 if previous.action_revision is None else previous.action_revision + 1
        self._generation[side] = SoftwareGeneration(side, previous.reset_epoch, revision)
        self._record(side, "state_machine_tsumo_fall_entry", before=old, after=new)

    def _side_for(self, machine: Any) -> str | None:
        """オブジェクト同一性だけでsideを決め、未登録を1Pへ補完しない。"""
        return next((side for side, known in self._machines.items() if known is machine), None)

    def _record(self, side: str, reason: str, **extra: Any) -> None:
        """記録失敗を握り潰さず、診断全体を失敗にする。"""
        generation = self.generation(side)
        self._sequence += 1
        self._emit({"kind": "software_generation", "generation_sequence": self._sequence,
                    "frame_idx": self._frame if self._in_frame else None,
                    "time_sec": self._time if self._in_frame else None, "side": side,
                    "clock_source": ("pipeline_update_input" if self._in_frame
                                     else "outside_update_time_unknown"),
                    "reset_epoch": generation.reset_epoch,
                    "action_revision": generation.action_revision, "reason": reason,
                    "identity_scope": IDENTITY_SCOPE, "commit_permission_issued": False,
                    **extra})


def _state_value(state: Any) -> str:
    """既存Enumまたは文字列のstateを、検証用の小文字へ揃える。"""
    return str(getattr(state, "value", state)).lower()


def _install_pipeline_hooks(
    stack: contextlib.ExitStack, cls: Any, rec: PipelineGenerationRecorder,
) -> None:
    """既存load/updateのdescriptor・引数・戻り値を保持する。"""
    original_load = inspect.getattr_static(cls, "load_default")
    original_update = cls.update

    def load(inner_cls: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load.__func__(inner_cls, *args, **kwargs)
        rec.bind_pipeline(result)
        return result

    @functools.wraps(original_update)
    def update(pipeline: Any, frame_idx: int, time_sec: float, *args: Any, **kwargs: Any) -> Any:
        rec.begin_frame(pipeline, frame_idx, time_sec)
        try:
            return original_update(pipeline, frame_idx, time_sec, *args, **kwargs)
        finally:
            rec.end_frame()

    patch(stack, cls, "load_default", classmethod(load))
    patch(stack, cls, "update", update)


def _install_machine_hooks(
    stack: contextlib.ExitStack, cls: Any, rec: PipelineGenerationRecorder,
) -> None:
    """元処理の成功後だけ世代を記録し、contextを変更しない。"""
    original_update, original_reset = cls.update, cls.reset

    @functools.wraps(original_update)
    def update(machine: Any, *args: Any, **kwargs: Any) -> Any:
        before = machine.context.state
        result = original_update(machine, *args, **kwargs)
        rec.record_transition(machine, before, machine.context.state)
        return result

    @functools.wraps(original_reset)
    def reset(machine: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_reset(machine, *args, **kwargs)
        rec.record_reset(machine)
        return result

    patch(stack, cls, "update", update)
    patch(stack, cls, "reset", reset)


def install_generation_hooks(
    stack: contextlib.ExitStack, pipeline_class: Any, machine_class: Any,
    recorder: PipelineGenerationRecorder,
) -> None:
    """fresh診断内でのみ使い、例外終了でも既存descriptorを復元する。"""
    _install_pipeline_hooks(stack, pipeline_class, recorder)
    _install_machine_hooks(stack, machine_class, recorder)
