"""連鎖本体・再捕捉episode・物理予測版を分離して保存する純粋台帳。

同一性の判定は既存 ``ChainIdResolver`` 等の責務であり、本台帳は外部から渡された
side/reset/action 世代と明示handleだけを検査する。終了、commit、公開解除、anchor承認は
提供しない。landing時のoriginは後続の再捕捉・再予測で上書きされない。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from src.board import Board
from src.chain import ChainResult
from src.chain_commit_candidate_v1 import BoardGrid
from src.chain_detector import CHAIN_MECHANISM_LANDING, ChainEvent
from src.scoring import calculate_chain_score


Side = Literal["1P", "2P"]
GenerationValue = int | None


class ChainPredictionLedgerError(RuntimeError):
    """予測台帳の基底例外。"""


class LedgerValidationError(ChainPredictionLedgerError, ValueError):
    """入力値が保存契約を満たさない。"""


class StaleChainHandleError(ChainPredictionLedgerError):
    """handle、side、reset、action、またはformula sessionが古い。"""


class ActiveChainExistsError(ChainPredictionLedgerError):
    """同じsideに未失効のprovisionalが既にある。"""


class ObservationOrderError(ChainPredictionLedgerError):
    """frameまたは時刻がside内で逆行した。"""


class PredictionScope(str, Enum):
    """予測得点の基準。callerではなく入力盤面SHAから台帳が決める。"""

    TOTAL_FROM_INSTANCE_ORIGIN = "total_from_instance_origin"
    REMAINING_FROM_REVISION_BOARD = "remaining_from_revision_board"


class ChainLedgerStatus(str, Enum):
    """provisional handleの保存状態。"""

    PROVISIONAL = "provisional"
    INVALIDATED = "invalidated"


@dataclass(frozen=True)
class ChainGeneration:
    """外部同一性解決器が渡すside/reset/action世代。NoneはUNKNOWNのまま。"""

    side: Side
    reset_epoch: GenerationValue
    action_revision: GenerationValue


@dataclass(frozen=True)
class ObservationPoint:
    """保存証拠の動画内位置。sequenceは同frame内順序を別途保持する。"""

    frame_idx: int
    time_sec: float


@dataclass(frozen=True)
class ProvisionalChainHandle:
    """一つの台帳instanceだけが発行できる明示handle。"""

    instance_id: int
    side: Side
    _owner_token: object


@dataclass(frozen=True)
class ChainEventEpisode:
    """start再捕捉1回の不変snapshot。物理連鎖IDではない。"""

    episode_revision: int
    ledger_sequence: int
    observed_at: ObservationPoint
    capture_source: str
    trigger_sec: float
    end_sec: float
    before_grid: BoardGrid
    before_sha256: str
    chain_count: int
    total_score: int
    mechanism: str | None
    score_estimated: bool


@dataclass(frozen=True)
class ChainPredictionRevision:
    """一回のsimulate結果。origin参照は最初のorigin入力版だけに固定する。"""

    prediction_revision: int
    ledger_sequence: int
    episode_revision: int
    available_at: ObservationPoint
    input_grid: BoardGrid
    input_sha256: str
    final_grid: BoardGrid
    final_sha256: str
    chain_count: int
    total_erased: int
    total_ojama: int
    calculated_total_score: int
    scope: PredictionScope
    is_origin_reference: bool


@dataclass(frozen=True)
class RawScoreEvidence:
    """anchorを決めずに保持するraw score / ScoreDelta証拠。"""

    ledger_sequence: int
    observed_at: ObservationPoint
    source: str
    raw_value: int | None
    prev_score: int | None
    cur_score: int | None
    delta: int | None
    is_valid: bool


@dataclass(frozen=True)
class FormulaEvidence:
    """初回session接続と各stepを、承認判定せず保存する。"""

    ledger_sequence: int
    observed_at: ObservationPoint
    source: str
    session_id: int
    step_index: int
    total_power: int
    step_product: int | None


@dataclass(frozen=True)
class FormulaSessionBinding:
    """provisionalへ最初に届いたformula sessionの不変な接続根拠。"""

    session_id: int
    ledger_sequence: int
    observed_at: ObservationPoint
    source: str


@dataclass(frozen=True)
class InvalidationReceipt:
    """旧handleを再開できなくする明示失効記録。"""

    ledger_sequence: int
    invalidated_at: ObservationPoint
    reason: str


@dataclass(frozen=True)
class ChainPredictionSnapshot:
    """外部へ返す全要素不変の保存snapshot。"""

    handle: ProvisionalChainHandle
    generation: ChainGeneration
    status: ChainLedgerStatus
    created_at: ObservationPoint
    origin_before_grid: BoardGrid
    origin_before_sha256: str
    origin_prediction_revision: int | None
    formula_session_binding: FormulaSessionBinding | None
    episodes: tuple[ChainEventEpisode, ...]
    predictions: tuple[ChainPredictionRevision, ...]
    raw_score_evidence: tuple[RawScoreEvidence, ...]
    formula_evidence: tuple[FormulaEvidence, ...]
    invalidation: InvalidationReceipt | None

    def copy_origin_board(self) -> Board:
        """台帳snapshotを変更しない独立Boardを返す。"""
        return _board_from_grid(self.origin_before_grid)


@dataclass
class _ChainRecord:
    """台帳内部だけで更新するcontainer。公開時は全てtupleへ凍結する。"""

    handle: ProvisionalChainHandle
    generation: ChainGeneration
    created_at: ObservationPoint
    origin_grid: BoardGrid
    origin_sha256: str
    episodes: list[ChainEventEpisode]
    predictions: list[ChainPredictionRevision]
    raw_evidence: list[RawScoreEvidence]
    formula_evidence: list[FormulaEvidence]
    last_point: ObservationPoint
    next_sequence: int = 2
    origin_prediction_revision: int | None = None
    formula_binding: FormulaSessionBinding | None = None
    invalidation: InvalidationReceipt | None = None


def _board_grid(board: Board) -> BoardGrid:
    """mutable Boardをtuple snapshotへ変換する。"""
    value = board.copy().to_dict()["grid"]
    return tuple(tuple(int(cell) for cell in row) for row in value)


def _board_from_grid(grid: BoardGrid) -> Board:
    """tuple snapshotから独立Boardを作る。"""
    return Board.from_list([list(row) for row in grid])


def _sha256(value: object) -> str:
    """JSON canonical表現のSHA-256を返す。"""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_generation(generation: ChainGeneration) -> None:
    """UNKNOWNを値へ推定せず、既知世代だけを厳密整数として検査する。"""
    if generation.side not in ("1P", "2P"):
        raise LedgerValidationError(f"不正なside: {generation.side}")
    for value in (generation.reset_epoch, generation.action_revision):
        if value is not None and (type(value) is not int or value < 0):
            raise LedgerValidationError("reset/actionは0以上の整数またはUNKNOWN(None)です")


def _point(frame_idx: int, time_sec: float) -> ObservationPoint:
    """boolや非有限値を混入させず観測位置を作る。"""
    if type(frame_idx) is not int or frame_idx < 0:
        raise LedgerValidationError("frame_idxは0以上の整数が必要です")
    if isinstance(time_sec, bool) or not isinstance(time_sec, (int, float)):
        raise LedgerValidationError("time_secは有限の数値が必要です")
    if not math.isfinite(float(time_sec)) or float(time_sec) < 0:
        raise LedgerValidationError("time_secは有限かつ0以上が必要です")
    return ObservationPoint(frame_idx, float(time_sec))


def _validate_source(source: str) -> str:
    """空のprovenanceを拒否する。"""
    if not isinstance(source, str) or not source.strip():
        raise LedgerValidationError("証拠sourceは空にできません")
    return source.strip()


def _optional_int(value: int | None, name: str) -> int | None:
    """欠測Noneを保持し、boolや負数を拒否する。"""
    if value is not None and (type(value) is not int or value < 0):
        raise LedgerValidationError(f"{name}は0以上の整数またはNoneが必要です")
    return value


def _optional_signed_int(value: int | None, name: str) -> int | None:
    """欠測Noneを保持し、符号付き整数ではboolだけを拒否する。"""
    if value is not None and type(value) is not int:
        raise LedgerValidationError(f"{name}は整数またはNoneが必要です")
    return value


def _event_episode(
    event: ChainEvent, revision: int, sequence: int,
    point: ObservationPoint, source: str,
) -> ChainEventEpisode:
    """mutable before_boardを含むeventを不変episodeへ変換する。"""
    times = (event.trigger_sec, event.end_sec)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in times):
        raise LedgerValidationError("event時刻は有限の数値が必要です")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in times):
        raise LedgerValidationError("event時刻は有限かつ0以上が必要です")
    if type(event.chain_count) is not int or event.chain_count <= 0:
        raise LedgerValidationError("event chain_countは正整数が必要です")
    if type(event.total_score) is not int or event.total_score < 0:
        raise LedgerValidationError("event total_scoreは0以上の整数が必要です")
    grid = _board_grid(event.before_board)
    return ChainEventEpisode(
        revision, sequence, point, _validate_source(source),
        float(event.trigger_sec), float(event.end_sec), grid, _sha256(grid),
        event.chain_count, event.total_score, event.mechanism, bool(event.score_estimated),
    )


def _validate_result(
    result: ChainResult, input_grid: BoardGrid, episode: ChainEventEpisode,
) -> None:
    """episode→input→simulate resultの非再計算provenanceを検査する。"""
    values = (result.chain_count, result.total_erased,
              result.total_ojama, result.participating_cells)
    if any(type(value) is not int or value < 0 for value in values):
        raise LedgerValidationError("ChainResultの集計値が不正です")
    if result.chain_count <= 0 or result.chain_count != len(result.steps):
        raise LedgerValidationError("1段以上でstepsと一致するChainResultが必要です")
    for expected_index, step in enumerate(result.steps, start=1):
        if type(step.chain_index) is not int or step.chain_index != expected_index:
            raise LedgerValidationError("step.chain_indexは1始まりの連番が必要です")
    for previous, current in zip(result.steps, result.steps[1:]):
        if _board_grid(previous.board_after) != _board_grid(current.board_before):
            raise LedgerValidationError("隣接stepのafter/before盤面が連続していません")
    first_grid = _board_grid(result.steps[0].board_before)
    final_grid = _board_grid(result.final_board)
    last_grid = _board_grid(result.steps[-1].board_after)
    if episode.before_grid != input_grid or first_grid != input_grid:
        raise LedgerValidationError("episode/input/result先頭盤面が一致しません")
    if last_grid != final_grid:
        raise LedgerValidationError("result最終stepとfinal_boardが一致しません")
    erased = sum(step.erased_count for step in result.steps)
    ojama = sum(step.erased_ojama for step in result.steps)
    if (result.total_erased != erased or result.total_ojama != ojama
            or result.participating_cells != result.total_erased):
        raise LedgerValidationError("ChainResultの集計値がstepsと一致しません")


class ChainPredictionLedger:
    """外部identityを受け、originを破壊せず証拠を追記する台帳。"""

    def __init__(self) -> None:
        self._owner_token = object()
        self._records: dict[int, _ChainRecord] = {}
        self._active_by_side: dict[Side, int] = {}
        self._used_generations: set[ChainGeneration] = set()
        self._last_side_point: dict[Side, ObservationPoint] = {}
        self._max_reset_by_side: dict[Side, int] = {}
        self._max_action_by_reset: dict[tuple[Side, int | None], int] = {}
        self._next_instance_id = 1

    def open_landing_provisional(
        self, *, generation: ChainGeneration, frame_idx: int, time_sec: float,
        origin_before_board: Board, landing_event: ChainEvent,
        capture_source: str = "landing",
    ) -> ProvisionalChainHandle:
        """landing originを一度だけ保存し、未承認handleを返す。"""
        _validate_generation(generation)
        point = _point(frame_idx, time_sec)
        self._check_side_order(generation.side, point)
        self._check_generation_order(generation)
        if generation.side in self._active_by_side:
            raise ActiveChainExistsError("同じsideのprovisionalを重複開始できません")
        if generation in self._used_generations:
            raise StaleChainHandleError("失効済み世代を再開できません")
        if landing_event.mechanism != CHAIN_MECHANISM_LANDING:
            raise LedgerValidationError("provisional originにはlanding eventが必要です")
        episode = _event_episode(landing_event, 1, 1, point, capture_source)
        origin_grid = _board_grid(origin_before_board)
        if episode.before_grid != origin_grid:
            raise LedgerValidationError("landing eventとorigin before盤面が一致しません")
        return self._store_new(generation, point, origin_grid, episode)

    def add_episode(
        self, handle: ProvisionalChainHandle, *, generation: ChainGeneration,
        frame_idx: int, time_sec: float, event: ChainEvent, capture_source: str,
    ) -> ChainEventEpisode:
        """start再捕捉をinstance変更なしでepisodeとして追記する。"""
        record, point = self._prepare_append(handle, generation, frame_idx, time_sec)
        episode = _event_episode(
            event, len(record.episodes) + 1, record.next_sequence, point, capture_source,
        )
        record.episodes.append(episode)
        self._finish_append(record, point)
        return episode

    def add_prediction(
        self, handle: ProvisionalChainHandle, *, generation: ChainGeneration,
        frame_idx: int, time_sec: float, episode_revision: int,
        input_board: Board, result: ChainResult,
    ) -> ChainPredictionRevision:
        """入力SHAからscopeを導出し、一回の物理予測を追記する。"""
        record, point = self._prepare_append(handle, generation, frame_idx, time_sec)
        if type(episode_revision) is not int or not 1 <= episode_revision <= len(record.episodes):
            raise LedgerValidationError("存在するepisode_revisionが必要です")
        input_grid = _board_grid(input_board)
        episode = record.episodes[episode_revision - 1]
        _validate_result(result, input_grid, episode)
        prediction = self._make_prediction(
            record, point, episode_revision, input_grid, result,
        )
        record.predictions.append(prediction)
        if prediction.is_origin_reference:
            record.origin_prediction_revision = prediction.prediction_revision
        self._finish_append(record, point)
        return prediction

    def record_raw_score(
        self, handle: ProvisionalChainHandle, *, generation: ChainGeneration,
        frame_idx: int, time_sec: float, source: str, raw_value: int | None,
        prev_score: int | None, cur_score: int | None, delta: int | None,
        is_valid: bool,
    ) -> RawScoreEvidence:
        """anchor選択をせず、生OCRとScoreDeltaを同じ順序で保存する。"""
        record, point = self._prepare_append(handle, generation, frame_idx, time_sec)
        if type(is_valid) is not bool:
            raise LedgerValidationError("is_validはboolが必要です")
        values = (
            _optional_int(raw_value, "raw_value"),
            _optional_int(prev_score, "prev_score"),
            _optional_int(cur_score, "cur_score"),
            _optional_signed_int(delta, "delta"),
        )
        evidence = RawScoreEvidence(
            record.next_sequence, point, _validate_source(source), *values, is_valid,
        )
        record.raw_evidence.append(evidence)
        self._finish_append(record, point)
        return evidence

    def record_formula(
        self, handle: ProvisionalChainHandle, *, generation: ChainGeneration,
        frame_idx: int, time_sec: float, source: str, session_id: int,
        step_index: int, total_power: int, step_product: int | None,
    ) -> FormulaEvidence:
        """初回sessionを一度だけ接続し、reset(step0)とstep証拠を保存する。"""
        record, point = self._prepare_append(handle, generation, frame_idx, time_sec)
        for value, name in ((session_id, "session_id"), (step_index, "step_index"),
                            (total_power, "total_power")):
            if type(value) is not int or value < 0:
                raise LedgerValidationError(f"{name}は0以上の整数が必要です")
        product = _optional_int(step_product, "step_product")
        clean_source = _validate_source(source)
        self._require_formula_session(record, session_id, point)
        evidence = FormulaEvidence(
            record.next_sequence, point, clean_source, session_id,
            step_index, total_power, product,
        )
        if record.formula_binding is None:
            record.formula_binding = FormulaSessionBinding(
                session_id, record.next_sequence, point, clean_source,
            )
        record.formula_evidence.append(evidence)
        self._finish_append(record, point)
        return evidence

    def invalidate(
        self, handle: ProvisionalChainHandle, *, generation: ChainGeneration,
        frame_idx: int, time_sec: float, reason: str,
    ) -> InvalidationReceipt:
        """明示的に失効させ、同じhandle・世代の再開を禁止する。"""
        record, point = self._prepare_append(handle, generation, frame_idx, time_sec)
        receipt = InvalidationReceipt(record.next_sequence, point, _validate_source(reason))
        record.invalidation = receipt
        self._finish_append(record, point)
        self._active_by_side.pop(record.generation.side, None)
        return receipt

    def snapshot(self, handle: ProvisionalChainHandle) -> ChainPredictionSnapshot:
        """失効後も監査できる不変snapshotを返す。"""
        record = self._record_for(handle, allow_invalidated=True)
        status = (ChainLedgerStatus.INVALIDATED if record.invalidation is not None
                  else ChainLedgerStatus.PROVISIONAL)
        return ChainPredictionSnapshot(
            record.handle, record.generation, status, record.created_at,
            record.origin_grid, record.origin_sha256,
            record.origin_prediction_revision, record.formula_binding,
            tuple(record.episodes), tuple(record.predictions),
            tuple(record.raw_evidence), tuple(record.formula_evidence),
            record.invalidation,
        )

    def _store_new(
        self, generation: ChainGeneration, point: ObservationPoint,
        origin_grid: BoardGrid, episode: ChainEventEpisode,
    ) -> ProvisionalChainHandle:
        """検査済みoriginを新規recordへ一度だけ登録する。"""
        instance_id = self._next_instance_id
        self._next_instance_id += 1
        handle = ProvisionalChainHandle(instance_id, generation.side, self._owner_token)
        record = _ChainRecord(
            handle, generation, point, origin_grid, _sha256(origin_grid),
            [episode], [], [], [], point,
        )
        self._records[instance_id] = record
        self._active_by_side[generation.side] = instance_id
        self._used_generations.add(generation)
        self._last_side_point[generation.side] = point
        if generation.reset_epoch is not None:
            self._max_reset_by_side[generation.side] = max(
                generation.reset_epoch,
                self._max_reset_by_side.get(generation.side, generation.reset_epoch),
            )
        if generation.action_revision is not None:
            key = (generation.side, generation.reset_epoch)
            self._max_action_by_reset[key] = max(
                generation.action_revision,
                self._max_action_by_reset.get(key, generation.action_revision),
            )
        return handle

    def _record_for(
        self, handle: ProvisionalChainHandle, *, allow_invalidated: bool = False,
    ) -> _ChainRecord:
        """別台帳のhandleや再構成handleを受理しない。"""
        if not isinstance(handle, ProvisionalChainHandle):
            raise StaleChainHandleError("provisional handleが必要です")
        record = self._records.get(handle.instance_id)
        if record is None or record.handle is not handle or handle._owner_token is not self._owner_token:
            raise StaleChainHandleError("この台帳が発行した同一handleではありません")
        if record.invalidation is not None and not allow_invalidated:
            raise StaleChainHandleError("失効済みhandleは再利用できません")
        return record

    def _prepare_append(
        self, handle: ProvisionalChainHandle, generation: ChainGeneration,
        frame_idx: int, time_sec: float,
    ) -> tuple[_ChainRecord, ObservationPoint]:
        """追記前にhandle・世代・単調位置をまとめて検査する。"""
        _validate_generation(generation)
        record = self._record_for(handle)
        if generation != record.generation or handle.side != generation.side:
            raise StaleChainHandleError("side/reset/action世代がprovisionalと一致しません")
        point = _point(frame_idx, time_sec)
        self._check_after(record.last_point, point)
        return record, point

    def _make_prediction(
        self, record: _ChainRecord, point: ObservationPoint,
        episode_revision: int, input_grid: BoardGrid, result: ChainResult,
    ) -> ChainPredictionRevision:
        """scopeとorigin参照をcaller入力なしで決める。"""
        final_grid = _board_grid(result.final_board)
        input_sha, final_sha = _sha256(input_grid), _sha256(final_grid)
        is_origin_input = input_sha == record.origin_sha256
        scope = (PredictionScope.TOTAL_FROM_INSTANCE_ORIGIN if is_origin_input
                 else PredictionScope.REMAINING_FROM_REVISION_BOARD)
        revision = len(record.predictions) + 1
        is_origin_reference = (
            revision == 1 and episode_revision == 1
            and point == record.created_at and is_origin_input
        )
        score = calculate_chain_score(result).total_score
        return ChainPredictionRevision(
            revision, record.next_sequence, episode_revision, point,
            input_grid, input_sha, final_grid, final_sha, result.chain_count,
            result.total_erased, result.total_ojama, score, scope, is_origin_reference,
        )

    def _finish_append(self, record: _ChainRecord, point: ObservationPoint) -> None:
        """成功した追記だけ位置とsequenceへ反映する。"""
        record.last_point = point
        record.next_sequence += 1
        self._last_side_point[record.generation.side] = point

    def _check_side_order(self, side: Side, point: ObservationPoint) -> None:
        """新規instanceでもside時系列を巻き戻さない。"""
        previous = self._last_side_point.get(side)
        if previous is not None:
            self._check_after(previous, point)

    def _check_generation_order(self, generation: ChainGeneration) -> None:
        """既知のreset/action番号だけは過去値への再開を拒否する。"""
        reset = generation.reset_epoch
        if reset is not None and reset < self._max_reset_by_side.get(generation.side, reset):
            raise StaleChainHandleError("reset_epochが既知の過去値へ逆行しています")
        action = generation.action_revision
        key = (generation.side, reset)
        if action is not None and action < self._max_action_by_reset.get(key, action):
            raise StaleChainHandleError("action_revisionが既知の過去値へ逆行しています")

    @staticmethod
    def _check_after(previous: ObservationPoint, current: ObservationPoint) -> None:
        """frame/timeを独立に単調検査し、同frame同時刻は許可する。"""
        if current.frame_idx < previous.frame_idx or current.time_sec < previous.time_sec:
            raise ObservationOrderError("frameまたは時刻が過去へ逆行しています")

    def _require_formula_session(
        self, record: _ChainRecord, session_id: int, point: ObservationPoint,
    ) -> None:
        """既bind sessionの変更を検出したら旧handleをfail-closedで失効する。"""
        binding = record.formula_binding
        if binding is not None and binding.session_id != session_id:
            reason = f"formula_session_changed:{binding.session_id}->{session_id}"
            record.invalidation = InvalidationReceipt(record.next_sequence, point, reason)
            self._finish_append(record, point)
            self._active_by_side.pop(record.generation.side, None)
            raise StaleChainHandleError("formula session変更により旧handleを失効しました")


__all__ = [
    "ActiveChainExistsError", "ChainEventEpisode", "ChainGeneration",
    "ChainLedgerStatus", "ChainPredictionLedger", "ChainPredictionLedgerError",
    "ChainPredictionRevision", "ChainPredictionSnapshot", "FormulaEvidence",
    "FormulaSessionBinding", "InvalidationReceipt", "LedgerValidationError",
    "ObservationOrderError", "ObservationPoint", "PredictionScope",
    "ProvisionalChainHandle", "RawScoreEvidence", "StaleChainHandleError",
]
