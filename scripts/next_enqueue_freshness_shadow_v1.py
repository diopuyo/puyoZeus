"""案A: NEXTの会計専用受理履歴。CPUのみ、実global/slide/Counterへ書き戻さない。

静止観測の受理と物理着手の認証を分離する。同色A→Aの実着手識別は未解決。
原本enqueue/settleを私有facadeで使い、初手修復/歴史会計/実update接続は行わない。
actual_call/run_shaはCPU fixtureの整合入力であり、保存本文の自動認証や
実戻りobjectの捕捉ではない。実update adapterは未接続。
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

from scripts import initial_placement_accounting_shadow_v1 as base


SCOPE = "cpu_accounting_next_history_not_live_or_physical_placement_proof"
INPUT_SCOPE = "cpu_fixture_consistency_not_saved_body_authentication_or_actual_return_capture"
STATES = frozenset(("menu", "stable", "tsumo_fall", "chain", "gravity_settle", "ojama_fall", "effect"))
CALL_LINES = {"1P": 5077, "2P": 5089}
SAMPLING_STRIDE = base.OBSERVATION_FRAMES[1] - base.OBSERVATION_FRAMES[0]
REQUIRED_INPUT_SHA256 = {
    str(base.PIPELINE): base.PIPELINE_SHA,
    str(base.ROOT / "scripts/initial_placement_accounting_shadow_v1.py"):
        "7f569e0d8dbde2ef2e46333eb3985b4d0fbb43035260c18e5780e42e77697b83",
    str(base.ROOT / "tests/test_initial_placement_accounting_shadow_v1.py"):
        "0f5a0a433433604d3838602a6b0cf556a22966743293fc8b6c5aa53a7209b77b",
}


def _canonical_path(value: str) -> str:
    """同じWindows/WSL絶対path表記だけを対応させる。"""
    value = value.replace("\\", "/")
    if value.startswith("/mnt/") and len(value) > 7 and value[6] == "/":
        value = value[5].upper() + ":/" + value[7:]
    return value.casefold()


def _numeric(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class SlideEvidence:
    """実戻り値の保存record。真の物理発進を証明する型ではない。"""

    side: str
    clock: base.Clock
    pulse: bool | None
    diff_score: float | None
    threshold_used: float | None
    cooldown_after: int
    actual_call: bool
    exception: str | None
    caller_source: str
    caller_line: int

    def __post_init__(self) -> None:
        if self.side not in base.SIDES or type(self.clock) is not base.Clock:
            raise ValueError("slide side/clockが不正です")
        if type(self.actual_call) is not bool:
            raise ValueError("actual_callは観測boolが必要です")
        if type(self.cooldown_after) is not int or self.cooldown_after < 0:
            raise ValueError("cooldownが不正です")
        if type(self.caller_source) is not str or type(self.caller_line) is not int:
            raise ValueError("実callerの型が不正です")
        if self.exception is not None and type(self.exception) is not str:
            raise ValueError("例外recordは不変文字列が必要です")
        if self.exception is not None:
            if any(value is not None for value in (self.pulse, self.diff_score, self.threshold_used)):
                raise ValueError("例外の戻り値を数値/falseで補完しません")
            return
        if type(self.pulse) is not bool:
            raise ValueError("slide boolの型が不正です")
        if not _numeric(self.diff_score) or not _numeric(self.threshold_used):
            raise ValueError("slide数値は有限値が必要です")
        if self.diff_score < 0 or self.threshold_used <= 0:
            raise ValueError("diff/thresholdの範囲が不正です")

    @classmethod
    def from_saved_row(cls, row: dict[str, Any]) -> SlideEvidence:
        if row.get("kind") != "initial_pair_main_slide" or type(row.get("extra_call")) is not bool:
            raise ValueError("実main slide recordが必要です")
        value, caller = row["returned"], row["caller"]
        if value is None and row["exception"] is None:
            raise ValueError("返却欠測をquietへ補完しません")
        return cls(row["side"], base.Clock(row["frame_idx"], row["time_sec"]),
                   None if value is None else value["slide_motion"],
                   None if value is None else value["diff_score"],
                   None if value is None else value["threshold_used"],
                   row["after"]["cooldown"], not row["extra_call"], row["exception"],
                   caller["source"], caller["line"])


@dataclass(frozen=True)
class NextObservation:
    side: str
    clock: base.Clock
    active: bool
    next_pair: tuple[int, int] | None
    slide: SlideEvidence | None
    before_state: str
    after_state: str
    episode_id: str = base.EPISODE
    source_sha256: str = base.VIDEO_SHA
    run_sha256: str = base.RUN_SHA

    def __post_init__(self) -> None:
        if self.side not in base.SIDES or type(self.clock) is not base.Clock or type(self.active) is not bool:
            raise ValueError("観測のside/clock/activeが不正です")
        if self.next_pair is not None and (type(self.next_pair) is not tuple or len(self.next_pair) != 2
                                          or any(type(color) is not int for color in self.next_pair)):
            raise ValueError("NEXTは不変の整数pairが必要です")
        if self.slide is not None and type(self.slide) is not SlideEvidence:
            raise ValueError("slideの型が不正です")
        if type(self.before_state) is not str or type(self.after_state) is not str or (
            self.before_state not in STATES or self.after_state not in STATES
        ):
            raise ValueError("実state値が必要です")
        if any(type(value) is not str or not value for value in (
            self.episode_id, self.source_sha256, self.run_sha256
        )):
            raise ValueError("観測identityが必要です")


@dataclass(frozen=True)
class AccountingView:
    """privateの会計NEXT。実global _last_seen_nextと同じobjectを持たない。"""

    accounting: base.CpuAccountingSnapshot = field(default_factory=base.CpuAccountingSnapshot)
    accepted_next: tuple[int, int] | None = None
    landing_pending: tuple[int, tuple[int, int]] | None = None
    last_consumed_color: tuple[int, int] | None = None
    constraint_valid: bool = True


@dataclass(frozen=True)
class NextReceipt:
    observation: NextObservation
    before: AccountingView
    after: AccountingView
    reason: str
    enqueued: tuple[int, int] | None
    scope: str = field(default=SCOPE, init=False)
    input_authentication_scope: str = field(default=INPUT_SCOPE, init=False)
    live_permission: bool = field(default=False, init=False)
    physical_placement_verified: bool = field(default=False, init=False)
    same_color_placement_resolved: bool = field(default=False, init=False)
    accounting_basis_verified: bool = field(default=False, init=False)


@dataclass(frozen=True)
class _Bundle:
    view: AccountingView
    episode_id: str
    clock: base.Clock
    state: str
    last_receipt: NextReceipt | None = None


def _reason(observation: NextObservation) -> str:
    if not observation.active:
        return "inactive_clear"
    if observation.next_pair is None or any(color not in base.COLORS for color in observation.next_pair):
        return "next_unavailable_or_invalid"
    slide = observation.slide
    if slide is None or not slide.actual_call:
        return "slide_not_observed"
    if slide.exception is not None:
        return "slide_call_failed"
    if slide.pulse or slide.diff_score >= slide.threshold_used:
        return "moving_next_rejected"
    return "quiet_observation_accepted_not_placement_proof"


def _private_facade(view: AccountingView, side: str) -> SimpleNamespace:
    """原本blockへ渡す全mutable値を新規作成する。"""
    pipe = SimpleNamespace()
    for current in base.SIDES:
        active = view if current == side else AccountingView()
        state = active.accounting
        values = dict(tsumo_count=Counter(dict(state.counts)), pending_tsumo=deque(state.pending),
                      first_move_sec=state.first_move_sec, last_seen_next=active.accepted_next,
                      landing_pending=active.landing_pending, last_consumed_color=active.last_consumed_color,
                      constraint_valid=active.constraint_valid)
        for name, value in values.items():
            setattr(pipe, f"_{name}_{current.lower()}", value)
    return pipe


def _freeze_facade(pipe: Any, side: str, revision: int) -> AccountingView:
    suffix = side.lower()
    state = base.CpuAccountingSnapshot(
        counts=tuple(sorted(getattr(pipe, f"_tsumo_count_{suffix}").items())),
        pending=tuple(getattr(pipe, f"_pending_tsumo_{suffix}")),
        first_move_sec=getattr(pipe, f"_first_move_sec_{suffix}"), revision=revision)
    return AccountingView(state, getattr(pipe, f"_last_seen_next_{suffix}"),
                          getattr(pipe, f"_landing_pending_{suffix}"),
                          getattr(pipe, f"_last_consumed_color_{suffix}"),
                          getattr(pipe, f"_constraint_valid_{suffix}"))


def _evaluate(view: AccountingView, observation: NextObservation,
              blocks: base.FrozenAccountingBlocks) -> NextReceipt:
    reason, pipe = _reason(observation), _private_facade(view, observation.side)
    if not observation.active:
        blocks.replay("clear", {"self": pipe})
    pair = observation.next_pair if reason.startswith("quiet_") else None
    suffix = observation.side.lower()
    before_enqueue = tuple(getattr(pipe, f"_pending_tsumo_{suffix}"))
    blocks.replay("enqueue", {"self": pipe, "is_active": observation.active,
                              "frame_idx": observation.clock.frame_idx,
                              "next_pair_1p": pair if observation.side == "1P" else None,
                              "next_pair_2p": pair if observation.side == "2P" else None})
    after_enqueue = tuple(getattr(pipe, f"_pending_tsumo_{suffix}"))
    enqueued = after_enqueue[-1] if after_enqueue != before_enqueue else None
    blocks.replay("settle", {"self": pipe, "side": observation.side,
                             "prev_state": observation.before_state,
                             "ctx": SimpleNamespace(state=observation.after_state),
                             "signals": SimpleNamespace(time_sec=observation.clock.time_sec)})
    after = _freeze_facade(pipe, observation.side, view.accounting.revision + 1)
    return NextReceipt(observation, view, after, reason, enqueued)


class NextEnqueueOwner:
    """一つのside/runに属するCPU会計owner。観測一回につき状態束を一回swap。"""

    def __init__(self, side: str, blocks: base.FrozenAccountingBlocks) -> None:
        if side not in base.SIDES or type(blocks) is not base.FrozenAccountingBlocks:
            raise ValueError("sideと固定会計blockが必要です")
        for path, digest in REQUIRED_INPUT_SHA256.items():
            base._read_verified(base.Path(path), digest)
        self.side, self._blocks = side, blocks
        self._bundle = _Bundle(AccountingView(), base.EPISODE,
                               base.Clock(base.RESET_FRAME, base.RESET_FRAME / base.FPS), "menu")
        self._busy = False

    @property
    def snapshot(self) -> AccountingView:
        return self._bundle.view

    def _validate(self, observation: NextObservation) -> None:
        if self._busy or type(observation) is not NextObservation:
            raise ValueError("再入または不正な観測です")
        if (observation.side, observation.episode_id, observation.source_sha256, observation.run_sha256) != (
            self.side, self._bundle.episode_id, base.VIDEO_SHA, base.RUN_SHA
        ):
            raise ValueError("別side/source/run/resetの観測です")
        if observation.clock.frame_idx < self._bundle.clock.frame_idx:
            raise ValueError("観測clockの逆行を拒否します")
        slide = observation.slide
        if slide is not None and (slide.side != self.side or slide.clock != observation.clock):
            raise ValueError("slideのside/clockが一致しません")
        if slide is not None and slide.actual_call and (
            _canonical_path(slide.caller_source) != _canonical_path(str(base.PIPELINE))
            or slide.caller_line != CALL_LINES[self.side]
        ):
            raise ValueError("実frozen callerと一致しません")
        if not observation.active and observation.after_state != "menu":
            raise ValueError("inactiveと実stateの矛盾を拒否します")

    def process(self, observation: NextObservation) -> NextReceipt:
        self._validate(observation)
        before = self._bundle
        if observation.clock == before.clock:
            if before.last_receipt and observation == before.last_receipt.observation:
                return before.last_receipt
            raise ValueError("同frame異payloadを拒否します")
        if before.last_receipt is not None and observation.clock.frame_idx - before.clock.frame_idx != SAMPLING_STRIDE:
            raise ValueError("固定replayの欠落frameを跨いで受理しません")
        if observation.before_state != before.state:
            raise ValueError("前回実stateとの連続性がありません")
        self._busy = True
        try:
            receipt = _evaluate(before.view, observation, self._blocks)
            after = _Bundle(receipt.after, before.episode_id, observation.clock,
                            observation.after_state, receipt)
            if self._bundle is not before:
                raise ValueError("準備中の所有状態変更を拒否します")
            self._bundle = after
            return receipt
        finally:
            self._busy = False

    def reset(self, episode_id: str, clock: base.Clock) -> AccountingView:
        """明示的なCPU reset観測。新試合の自動認証やCounter保全ではない。"""
        if self._busy or type(clock) is not base.Clock or clock.frame_idx <= self._bundle.clock.frame_idx:
            raise ValueError("reset再入/clock不正を拒否します")
        if type(episode_id) is not str or not episode_id or episode_id == self._bundle.episode_id:
            raise ValueError("新resetの観測identityが必要です")
        before = self._bundle
        self._busy = True
        try:
            pipe = _private_facade(before.view, self.side)
            self._blocks.replay("clear", {"self": pipe})
            view = _freeze_facade(pipe, self.side, before.view.accounting.revision + 1)
            after = _Bundle(view, episode_id, clock, "menu")
            if self._bundle is not before:
                raise ValueError("reset準備中の所有状態変更を拒否します")
            self._bundle = after
            return view
        finally:
            self._busy = False
