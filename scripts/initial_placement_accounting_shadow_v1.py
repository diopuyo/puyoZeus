"""初手一回会計のCPU境界部品。live接続・自動物理認証・Counter補填はしない。

固定video38の独立検収をoracleにしたreplay専用。外部pipeline/Counterを受け取らず、
私有コピー上で凍結consume本体を実行した後、単独所有の不変状態を一回置換する。
この原子性はCPU owner内だけであり、実pipeline/collectorへの適用は未接続。
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from types import CodeType, MappingProxyType, SimpleNamespace
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / ".runtime_snapshots/event_first30_observed_context_v5_2026-08-30"
PIPELINE = FROZEN / "src/recognition_pipeline.py"
PIPELINE_SHA = "6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02"
RUN = ROOT / "data/verify/video38_initial_pair_observation_2026-09-07_v2"
RUN_SHA = "d2d012972fb61092b1ecae1e49e04ed1bf3b454110028bb71ac75358df005b67"
VIDEO_SHA = "b3728078cc2e8282065e5dd78ca1c13e6a443ffdf1dc4333e3da06f0852757d3"
EPISODE = "video38:reviewed-score-reset:frame32494"
SCOPE = "cpu_reviewed_replay_not_live_or_automatic_ground_truth"
FPS, RESET_FRAME, EMPTY_FRAME, LANDING_FRAME = 60, 32494, 32640, 32684
OBSERVATION_FRAMES = tuple(range(32496, 32644, 2))
SIDES, COLORS = ("1P", "2P"), frozenset(range(1, 6))
_BLOCK_LOAD_TOKEN = object()
LANDING_GRID_SHA = "403c5aa2ccbb798a8fc855652f5d31f63dbfaf9f7e0e47b3f640b02a95652233"
REQUIRED_INPUT_SHA256 = {
    str(PIPELINE): PIPELINE_SHA,
    str(FROZEN / "scripts/collect_boards_lean.py"):
        "672963055a9fffa66531411be0a51742ee57c35de481652a77d02158ce24bee8",
    str(RUN / "frames.jsonl"): RUN_SHA,
    str(RUN / "COMPLETE"): "c87ceadd5e07fd78284db101bb33c879165615e799cd991e8ef80b6bcef012e9",
    str(RUN / "INDEPENDENT_RUN_QA.md"): "6630a988da3dc632976405b58c72b5194f58078f8870a4d120a0208a12b68cff",
    str(ROOT / "data/verify/video38_accounting_start_review_2026-09-07_v1/PHYSICAL_QA.md"):
        "7f6fa93386267f786165488e17e9a3b7f95b09bc4ad0816bb6c8a1186d18d822",
}


class CpuState(StrEnum):
    """凍結会計blockの境界fixture。実SMのstateは変更しない。"""

    MENU = "menu"
    STABLE = "stable"
    TSUMO_FALL = "tsumo_fall"


@dataclass(frozen=True)
class Clock:
    frame_idx: int
    time_sec: float

    def __post_init__(self) -> None:
        if type(self.frame_idx) is not int or self.frame_idx < RESET_FRAME:
            raise ValueError("frameは承認reset以後の整数が必要です")
        if type(self.time_sec) not in (int, float) or not math.isfinite(self.time_sec):
            raise ValueError("時刻は有限数値が必要です")
        if self.time_sec != self.frame_idx / FPS:
            raise ValueError("固定動画のframe/timeが一致しません")


def _pair(value: tuple[int, int]) -> None:
    """mutable入力やUNKNOWNを初手色へ昇格させない。"""
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("pairは不変の2色tupleが必要です")
    if any(type(color) is not int or color not in COLORS for color in value):
        raise ValueError("未確認色を会計へ渡せません")


@dataclass(frozen=True)
class PairObservation:
    side: str
    clock: Clock
    next_pair: tuple[int, int]
    dnext_pair: tuple[int, int]
    episode_id: str = EPISODE
    source_sha256: str = VIDEO_SHA
    run_sha256: str = RUN_SHA

    def __post_init__(self) -> None:
        _pair(self.next_pair)
        _pair(self.dnext_pair)
        if self.side not in SIDES or type(self.clock) is not Clock:
            raise ValueError("side/clockが不正です")


@dataclass(frozen=True)
class LandingContext:
    side: str
    clock: Clock
    before_state: CpuState = CpuState.MENU
    after_state: CpuState = CpuState.STABLE
    action_revision: int | None = None
    episode_id: str = EPISODE
    reset_frame: int = RESET_FRAME
    active: bool = True
    execution_scope: str = SCOPE

    def __post_init__(self) -> None:
        if self.side not in SIDES or type(self.clock) is not Clock:
            raise ValueError("side/clockが不正です")
        if type(self.reset_frame) is not int or type(self.active) is not bool:
            raise ValueError("reset/activeの型が不正です")
        if self.action_revision is not None and (
            type(self.action_revision) is not int or self.action_revision < 0
        ):
            raise ValueError("actionは不変の観測値かNoneが必要です")


@dataclass(frozen=True)
class CpuLandingProof:
    """固定独立oracleの発行物。同値コピーや任意verified=Trueは受け取らない。"""

    side: str
    episode_id: str = EPISODE
    source_sha256: str = VIDEO_SHA
    run_sha256: str = RUN_SHA
    empty_frame: int = EMPTY_FRAME
    landing_frame: int = LANDING_FRAME
    landing_grid_sha256: str = LANDING_GRID_SHA
    scope: str = SCOPE


@dataclass(frozen=True)
class CpuAccountingSnapshot:
    """外部Counter参照を持たないCPU専用状態束。"""

    counts: tuple[tuple[int, int], ...] = ()
    pending: tuple[tuple[int, int], ...] = ()
    first_move_sec: float | None = None
    consumed_id: str | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        if type(self.counts) is not tuple or type(self.pending) is not tuple:
            raise ValueError("状態は不変tupleで必要です")
        for item in self.counts:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("Counter snapshotの型が不正です")
            color, count = item
            if type(color) is not int or color not in COLORS or type(count) is not int or count <= 0:
                raise ValueError("Counter snapshotの値が不正です")
        if tuple(sorted(dict(self.counts).items())) != self.counts:
            raise ValueError("Counterは重複なしの昇順が必要です")
        for pair in self.pending:
            _pair(pair)
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("会計版が不正です")
        if self.first_move_sec is not None and (
            type(self.first_move_sec) not in (int, float) or not math.isfinite(self.first_move_sec)
        ):
            raise ValueError("first_move時刻が不正です")
        if self.consumed_id is not None and type(self.consumed_id) is not str:
            raise ValueError("消費IDは不変文字列が必要です")


@dataclass(frozen=True)
class CpuConsumptionReceipt:
    before: CpuAccountingSnapshot
    after: CpuAccountingSnapshot
    context: LandingContext
    scope: str = field(default=SCOPE, init=False)
    live_permission: bool = field(default=False, init=False)
    accounting_basis_verified: bool = field(default=False, init=False)


def _read_verified(path: Path, digest: str) -> bytes:
    """承認済み入力だけを読取り、異版を混ぜない。"""
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != digest:
        raise ValueError(f"入力SHAが一致しません: {path}")
    return value


def _compile_lines(tree: ast.AST, first: int, last: int, *, inner: bool = False) -> CodeType:
    """固定SHAの実文だけを抽出し、類似式を再実装しない。"""
    statements = [node for node in ast.walk(tree) if isinstance(node, ast.stmt)
                  and first <= node.lineno and node.end_lineno <= last]
    body = [node for node in statements if not any(
        other is not node and other.lineno <= node.lineno and other.end_lineno >= node.end_lineno
        for other in statements)]
    body.sort(key=lambda node: node.lineno)
    if not body or body[0].lineno != first or body[-1].end_lineno != last:
        raise ValueError("凍結会計の抽出範囲が一致しません")
    if inner:
        if len(body) != 1 or not isinstance(body[0], ast.If):
            raise ValueError("consume入口が不正です")
        body = body[0].body
    return compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                   str(PIPELINE), "exec")


class FrozenAccountingBlocks:
    """実clear/enqueue/settleと共通consume本体をCPU fixtureだけで実行する。"""

    def __init__(self, token: object, programs: Mapping[str, CodeType]) -> None:
        if token is not _BLOCK_LOAD_TOKEN:
            raise ValueError("固定sourceのloadを経由してください")
        self._programs = MappingProxyType(dict(programs))

    @classmethod
    def load(cls) -> FrozenAccountingBlocks:
        tree = ast.parse(_read_verified(PIPELINE, PIPELINE_SHA))
        ranges = {"clear": (4500, 4516), "enqueue": (5117, 5148),
                  "settle": (7386, 7409), "consume": (7398, 7409)}
        return cls(_BLOCK_LOAD_TOKEN, {name: _compile_lines(tree, *bounds)
                                     for name, bounds in ranges.items()})

    def replay(self, name: str, namespace: dict[str, Any]) -> None:
        """私有fixture上のみ。モデル/実pipelineの呼出入口ではない。"""
        exec(self._programs[name], {"BoardState": CpuState, "VALID_PUYO_COLORS": COLORS}, namespace)

    def consume_copy(self, state: CpuAccountingSnapshot, side: str,
                     clock: Clock) -> CpuAccountingSnapshot:
        """外部stateを変えず、元pop・2色増分・first_move処理を評価する。"""
        pending, counts = deque(state.pending), Counter(dict(state.counts))
        pipe = SimpleNamespace(_first_move_sec_1p=state.first_move_sec,
                               _first_move_sec_2p=state.first_move_sec)
        self.replay("consume", {"pending": pending, "tsumo_count_target": counts,
                                "self": pipe, "side": side,
                                "signals": SimpleNamespace(time_sec=clock.time_sec)})
        return replace(state, counts=tuple(sorted(counts.items())), pending=tuple(pending),
                       first_move_sec=getattr(pipe, f"_first_move_sec_{side.lower()}"))


class CpuReviewOracle:
    """固定runの独立画像検収oracle。汎用liveや任意証跡には許可を出さない。"""

    def __init__(self, token: object, rows: list[dict[str, Any]]) -> None:
        if token is not _ORACLE_LOAD_TOKEN:
            raise ValueError("固定検収のloadを経由してください")
        self._observations = _reviewed_observations(rows)
        _validate_reviewed_landings(rows)
        self._proofs = {side: CpuLandingProof(side) for side in SIDES}
        self._owners: dict[str, object] = {}

    @classmethod
    def load(cls) -> CpuReviewOracle:
        payloads = {path: _read_verified(Path(path), digest)
                    for path, digest in REQUIRED_INPUT_SHA256.items()}
        rows = [json.loads(line) for line in payloads[str(RUN / "frames.jsonl")].splitlines()]
        return cls(_ORACLE_LOAD_TOKEN, rows)

    def proof(self, side: str) -> CpuLandingProof:
        return self._proofs[side]

    def observations(self, side: str) -> tuple[PairObservation, ...]:
        return self._observations[side]

    def bind(self, side: str, owner: object) -> None:
        if side not in SIDES or side in self._owners:
            raise ValueError("同一oracle/sideは一つのCPU ownerに限定します")
        self._owners[side] = owner

    def validate(self, owner: object, proof: CpuLandingProof, side: str,
                 observed: tuple[PairObservation, ...]) -> None:
        if self._owners.get(side) is not owner or self._proofs.get(side) is not proof:
            raise ValueError("別owner・別oracle・偽造proofを拒否します")
        if observed != self._observations[side]:
            raise ValueError("未認証・欠測・異なる初手観測を拒否します")


_ORACLE_LOAD_TOKEN = object()


def _reviewed_observations(rows: list[dict[str, Any]]) -> dict[str, tuple[PairObservation, ...]]:
    """固定windowは今回の検収scopeであり、新たな安定票閾値ではない。"""
    result: dict[str, list[PairObservation]] = {side: [] for side in SIDES}
    for row in rows:
        if row["kind"] != "initial_pair_inactive_next" or row["frame_idx"] not in OBSERVATION_FRAMES:
            continue
        for side in SIDES:
            pair = row["raw_pairs"][side]
            result[side].append(PairObservation(side, Clock(row["frame_idx"], row["time_sec"]),
                                                tuple(pair["next"]), tuple(pair["dnext"])))
    for values in result.values():
        if tuple(item.clock.frame_idx for item in values) != OBSERVATION_FRAMES:
            raise ValueError("初手観測windowの欠落/重複があります")
        if any(item.next_pair != (4, 4) or item.dnext_pair != (5, 5) for item in values):
            raise ValueError("独立検収した観測と一致しません")
    return {side: tuple(values) for side, values in result.items()}


def _validate_reviewed_landings(rows: list[dict[str, Any]]) -> None:
    """実保存stateと検収oracleの対応確認。CNN値から物理許可は作らない。"""
    reset = [row for row in rows if row["kind"] == "boundary_repair_mutation"
             and row.get("repair") == "start_epoch" and row["frame_idx"] == RESET_FRAME]
    if len(reset) != 1 or reset[0]["evidence"]["episode_id"] != EPISODE:
        raise ValueError("承認済みnew-game episodeがありません")
    for side in SIDES:
        matches = [row for row in rows if row["kind"] == "frame_side" and row["side"] == side
                   and row["frame_idx"] == LANDING_FRAME]
        if len(matches) != 1 or matches[0]["state"] != "stable":
            raise ValueError("初回着地の実stateが不一致です")
        if matches[0]["confirmed"]["sha256"] != LANDING_GRID_SHA:
            raise ValueError("初回盤面と独立検収の対応が不一致です")


class InitialPlacementOwner:
    """未消費観測のみinactiveを跨ぎ、CPU状態束へ一回だけ移管・消費する。"""

    def __init__(self, oracle: CpuReviewOracle, side: str) -> None:
        if type(oracle) is not CpuReviewOracle:
            raise ValueError("CPU独立oracleが必要です")
        oracle.bind(side, self)
        self._oracle, self.side = oracle, side
        self._state = CpuAccountingSnapshot()
        self._observations: tuple[PairObservation, ...] = ()
        self._clock = Clock(RESET_FRAME, RESET_FRAME / FPS)
        self._activity: tuple[Clock, bool] | None = None
        self._invalidated = self._busy = False

    @property
    def snapshot(self) -> CpuAccountingSnapshot:
        return self._state

    def _require_open(self) -> None:
        if self._busy or self._invalidated or self._state.consumed_id is not None:
            raise ValueError("再入・失効・消費済みownerです")

    def _check_clock(self, clock: Clock) -> None:
        if type(clock) is not Clock or clock.frame_idx < self._clock.frame_idx:
            raise ValueError("clockの逆行を拒否します")

    def observe(self, observation: PairObservation) -> None:
        self._require_open()
        if type(observation) is not PairObservation:
            raise ValueError("未認証入力の型が不正です")
        if (observation.side, observation.episode_id, observation.source_sha256,
            observation.run_sha256) != (self.side, EPISODE, VIDEO_SHA, RUN_SHA):
            raise ValueError("別side/source/run/resetの観測です")
        self._check_clock(observation.clock)
        if self._observations and observation.clock == self._observations[-1].clock:
            if observation != self._observations[-1]:
                raise ValueError("同frame異payloadを拒否します")
            return
        self._observations += (observation,)
        self._clock = observation.clock

    def observe_activity(self, clock: Clock, active: bool) -> None:
        self._require_open()
        self._check_clock(clock)
        if type(active) is not bool:
            raise ValueError("activeは観測boolが必要です")
        if self._activity and clock == self._activity[0] and active != self._activity[1]:
            raise ValueError("同frame異activeを拒否します")
        self._activity, self._clock = (clock, active), clock

    def invalidate_for_reset(self, clock: Clock) -> None:
        if self._busy:
            raise ValueError("消費準備中の再入resetを拒否します")
        self._check_clock(clock)
        self._observations = ()
        self._invalidated = True
        self._clock = clock

    def _validate_consumption(self, proof: CpuLandingProof, context: LandingContext) -> None:
        self._require_open()
        if type(context) is not LandingContext:
            raise ValueError("実境界contextが必要です")
        self._check_clock(context.clock)
        if (context.side, context.episode_id, context.reset_frame, context.execution_scope,
            context.clock.frame_idx, context.before_state, context.after_state,
            context.active, context.action_revision) != (
                self.side, EPISODE, RESET_FRAME, SCOPE, LANDING_FRAME,
                CpuState.MENU, CpuState.STABLE, True, None):
            raise ValueError("承認済みCPU初回着地の境界と一致しません")
        if self._activity != (context.clock, context.active):
            raise ValueError("同frameの実activity観測とcontextが一致しません")
        self._oracle.validate(self, proof, self.side, self._observations)
        if self._state != CpuAccountingSnapshot():
            raise ValueError("進行済みCounter/FIFO/first_move/版へ初手を補填しません")

    def consume(self, proof: CpuLandingProof, context: LandingContext,
                blocks: FrozenAccountingBlocks) -> CpuConsumptionReceipt:
        self._validate_consumption(proof, context)
        if type(blocks) is not FrozenAccountingBlocks:
            raise ValueError("固定会計blockが必要です")
        before = self._state
        self._busy = True
        try:
            staged = replace(before, pending=(self._observations[0].next_pair,))
            computed = blocks.consume_copy(staged, self.side, context.clock)
            if computed != replace(staged, counts=((4, 2),), pending=(),
                                   first_move_sec=context.clock.time_sec):
                raise ValueError("通常消費本体の結果が不一致です")
            if self._state is not before:
                raise ValueError("準備中のowner状態変更を拒否します")
            after = replace(computed, consumed_id=f"{EPISODE}:{self.side}:initial-placement",
                            revision=before.revision + 1)
            receipt = CpuConsumptionReceipt(before, after, context)
            self._state = after
            return receipt
        finally:
            self._busy = False
