"""C-6 の物理 final を遅延適用するための純粋 transaction 契約。

このモジュールは候補の保持と一回性だけを担当する。検証証拠の生成、終了判定、
公開解除 policy、RecognitionPipeline への適用は意図的に実装しない。commit には
独立した検証 policy と公開解除 policy の両方が必要で、未接続なら fail-closed になる。

既存commitはeffect返却時に消費する互換API。外部適用はprepare/validate_prepared/
finalizeを使う。ownerは同一lock内で全適用データを事前構築し、現在contextとCounter
版をvalidateして単一参照swap後ただちにfinalizeする。callback部分書込のrollbackや
外部swapの原子性・永続化を本helperは保証しない。swap成功後のabortは禁止である。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Mapping, Protocol, cast

from src.board import Board, COLOR_BLUE, COLOR_GREEN, COLOR_PURPLE, COLOR_RED, COLOR_YELLOW
from src.chain_detector import ChainEvent


Side = Literal["1P", "2P"]
BoardGrid = tuple[tuple[int, ...], ...]
ColorCounts = tuple[tuple[int, int], ...]
PUYO_COLORS = frozenset({COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE})


class ChainCommitError(RuntimeError):
    """chain commit transaction の基底例外。"""


class CandidateValidationError(ChainCommitError, ValueError):
    """候補、context、または policy 応答が契約を満たさない。"""


class PolicyNotConnectedError(ChainCommitError):
    """検証または公開解除 policy が未接続である。"""


class StaleCandidateError(ChainCommitError):
    """候補と現在の side/reset/event/action 世代が一致しない。"""


class CounterUnderflowError(ChainCommitError):
    """現在の色 Counter から消去差分を安全に減算できない。"""


class TransactionConsumedError(ChainCommitError):
    """commit または discard 済みの transaction が再利用された。"""


class PreparedCommitError(ChainCommitError):
    """prepare所有権、版、適用前検査、または一回性の契約が不一致。"""


class ChainCommitState(str, Enum):
    """候補 transaction の一回性状態。"""

    PENDING = "pending"
    COMMITTED = "committed"
    DISCARDED = "discarded"


@dataclass(frozen=True)
class ChainEventSnapshot:
    """mutable な Board 参照を除去した ChainEvent の不変 snapshot。"""

    trigger_sec: float
    end_sec: float
    before_sha256: str
    chain_count: int
    total_erased: int
    total_score: int
    base_score: int
    all_clear_bonus_applied: int
    ojama_sent: int
    leftover_score: int
    is_all_clear: bool
    mechanism: str | None
    score_estimated: bool


@dataclass(frozen=True)
class ChainCommitIdentity:
    """候補を side と reset/event/action 世代へ結び付ける identity。"""

    side: Side
    reset_epoch: int
    event_sha256: str
    event_revision: int
    action_revision: int
    created_frame_idx: int
    created_time_sec: float


@dataclass(frozen=True)
class ChainCommitCandidate:
    """C-6 が計算した未検証 final と消去差分の不変候補。"""

    candidate_id: str
    identity: ChainCommitIdentity
    event: ChainEventSnapshot
    before_grid: BoardGrid
    final_grid: BoardGrid
    final_sha256: str
    erased_color_count: ColorCounts

    @property
    def event_sha256(self) -> str:
        """context 作成用の event identity を直接返す。"""
        return self.identity.event_sha256

    def copy_before_board(self) -> Board:
        """候補内 snapshot を変更しない独立 Board を返す。"""
        return _board_from_grid(self.before_grid)

    def copy_final_board(self) -> Board:
        """候補内 snapshot を変更しない独立 Board を返す。"""
        return _board_from_grid(self.final_grid)


@dataclass(frozen=True)
class ChainCommitContext:
    """commit 時点の現在世代。Counter の版とは独立して扱う。"""

    side: Side
    reset_epoch: int
    event_sha256: str
    event_revision: int
    action_revision: int
    current_frame_idx: int
    current_time_sec: float


class ChainCandidateVerificationPolicy(Protocol):
    """外部検証器が実装する fail-by-exception 契約。"""

    @property
    def policy_id(self) -> str: ...

    def require_verified(
        self, candidate: ChainCommitCandidate, context: ChainCommitContext,
    ) -> None: ...


class ChainCandidateReleasePolicy(Protocol):
    """外部の公開解除器が実装する fail-by-exception 契約。"""

    @property
    def policy_id(self) -> str: ...

    def require_release(
        self, candidate: ChainCommitCandidate, context: ChainCommitContext,
    ) -> None: ...


@dataclass(frozen=True)
class ChainCommitEffect:
    """pipeline が同一 frame で原子的に適用するための不変 effect。"""

    candidate_id: str
    final_grid: BoardGrid
    counter_after: ColorCounts
    committed_frame_idx: int
    committed_time_sec: float
    verification_policy_id: str
    release_policy_id: str

    def copy_final_board(self) -> Board:
        """適用先が所有する独立 Board を返す。"""
        return _board_from_grid(self.final_grid)

    def copy_counter(self) -> Counter[int]:
        """適用先が所有する現在 Counter の copy を返す。"""
        return Counter(dict(self.counter_after))


@dataclass(frozen=True)
class ChainDiscardReceipt:
    """未適用の候補破棄、またはprepared中断の理由receipt。"""

    candidate_id: str
    reason: str


@dataclass(frozen=True)
class PreparedChainCommit:
    """未消費effect。constructorやreplaceによる複製はtx所有物として受理しない。

    effect内のcommitted時刻は予定値であり、この型の生成はcommitを意味しない。
    counter_revisionはownerの単調な版で、内容が元に戻るABA更新でも進める。
    """

    effect: ChainCommitEffect
    context: ChainCommitContext
    counter_before: ColorCounts
    counter_revision: int


@dataclass(frozen=True)
class ChainCommitApplyTicket:
    """適用直前検査済みのtx専用ack ticket。外部適用済みの証明ではない。

    ownerはvalidateからswap/finalizeまでlockを離さず、任意callback、追加変換、
    yieldや世代更新を挟まない。適用前中断はabort、swap成功後は必ずfinalizeする。
    """

    prepared: PreparedChainCommit


def _board_grid(board: Board) -> BoardGrid:
    grid = board.copy().to_dict()["grid"]
    return tuple(tuple(int(cell) for cell in row) for row in grid)


def _board_from_grid(grid: BoardGrid) -> Board:
    return Board.from_list([list(row) for row in grid])


def _sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _event_snapshot(event: ChainEvent, before_sha256: str) -> ChainEventSnapshot:
    return ChainEventSnapshot(
        trigger_sec=float(event.trigger_sec), end_sec=float(event.end_sec),
        before_sha256=before_sha256, chain_count=int(event.chain_count),
        total_erased=int(event.total_erased), total_score=int(event.total_score),
        base_score=int(event.base_score),
        all_clear_bonus_applied=int(event.all_clear_bonus_applied),
        ojama_sent=int(event.ojama_sent), leftover_score=int(event.leftover_score),
        is_all_clear=bool(event.is_all_clear), mechanism=event.mechanism,
        score_estimated=bool(event.score_estimated),
    )


def _validate_creation(
    side: str, reset_epoch: int, event_revision: int, action_revision: int,
    created_frame_idx: int, created_time_sec: float, event: ChainEvent,
) -> None:
    if side not in ("1P", "2P"):
        raise CandidateValidationError(f"不正な side: {side}")
    integers = (reset_epoch, event_revision, action_revision, created_frame_idx)
    if any(type(value) is not int or value < 0 for value in integers):
        raise CandidateValidationError("epoch/revision/frame は 0 以上が必要です")
    if isinstance(created_time_sec, bool) or not math.isfinite(created_time_sec):
        raise CandidateValidationError("作成時刻は有限かつ 0 以上が必要です")
    if created_time_sec < 0:
        raise CandidateValidationError("作成時刻は有限かつ 0 以上が必要です")
    if type(event.chain_count) is not int or event.chain_count <= 0:
        raise CandidateValidationError("chain_count > 0 の event が必要です")
    event_times = (event.trigger_sec, event.end_sec)
    if any(isinstance(value, bool) or not math.isfinite(value) for value in event_times):
        raise CandidateValidationError("event 時刻は有限値が必要です")


def _normalize_erased_counts(counts: Mapping[int, int]) -> ColorCounts:
    normalized: list[tuple[int, int]] = []
    for color, count in counts.items():
        if type(color) is not int or color not in PUYO_COLORS:
            raise CandidateValidationError("消去差分の色が不正です")
        if type(count) is not int or count <= 0:
            raise CandidateValidationError("消去差分は通常色ごとの正整数が必要です")
        normalized.append((int(color), count))
    if not normalized:
        raise CandidateValidationError("空の消去差分は候補化できません")
    return tuple(sorted(normalized))


def _color_counts(grid: BoardGrid) -> dict[int, int]:
    return {
        color: sum(cell == color for row in grid for cell in row)
        for color in PUYO_COLORS
    }


def _validate_physical_delta(
    before_grid: BoardGrid, final_grid: BoardGrid, erased: ColorCounts,
) -> None:
    """full simulate の盤面差分だけを検査する。

    途中 formula event の ``total_erased`` / ``chain_count`` は full simulate の
    最終値ではないため、候補の色差分との一致 gate には使わない。
    """
    before, final = _color_counts(before_grid), _color_counts(final_grid)
    actual: dict[int, int] = {}
    for color in PUYO_COLORS:
        if final[color] > before[color]:
            raise CandidateValidationError("final に新しい色ぷよを増加できません")
        if before[color] > final[color]:
            actual[color] = before[color] - final[color]
    if tuple(sorted(actual.items())) != erased:
        raise CandidateValidationError("消去差分が before/final の物理差分と一致しません")


def create_chain_commit_candidate(
    *, side: Side, reset_epoch: int, event_revision: int, action_revision: int,
    created_frame_idx: int, created_time_sec: float, event: ChainEvent,
    final_board: Board, erased_color_count: Mapping[int, int],
) -> ChainCommitCandidate:
    """入力 Board を不変 snapshot 化し、未検証候補を作る。"""
    _validate_creation(
        side, reset_epoch, event_revision, action_revision,
        created_frame_idx, created_time_sec, event,
    )
    before_grid, final_grid = _board_grid(event.before_board), _board_grid(final_board)
    before_sha, final_sha = _sha256(before_grid), _sha256(final_grid)
    event_snapshot = _event_snapshot(event, before_sha)
    event_sha = _sha256(event_snapshot.__dict__)
    identity = ChainCommitIdentity(
        side, reset_epoch, event_sha, event_revision, action_revision,
        created_frame_idx, float(created_time_sec),
    )
    erased = _normalize_erased_counts(erased_color_count)
    _validate_physical_delta(before_grid, final_grid, erased)
    candidate_id = _sha256((identity.__dict__, final_sha, erased))
    return ChainCommitCandidate(
        candidate_id, identity, event_snapshot, before_grid, final_grid,
        final_sha, erased,
    )


def _validate_context(candidate: ChainCommitCandidate, context: ChainCommitContext) -> None:
    identity = candidate.identity
    revisions = (
        context.reset_epoch, context.event_revision,
        context.action_revision, context.current_frame_idx,
    )
    if any(type(value) is not int or value < 0 for value in revisions):
        raise CandidateValidationError("context の epoch/revision/frame が不正です")
    generations = (
        context.side == identity.side,
        context.reset_epoch == identity.reset_epoch,
        context.event_sha256 == identity.event_sha256,
        context.event_revision == identity.event_revision,
        context.action_revision == identity.action_revision,
    )
    if not all(generations):
        raise StaleCandidateError("現在の side/reset/event/action 世代と候補が不一致です")
    if context.current_frame_idx < identity.created_frame_idx:
        raise StaleCandidateError("候補作成前の frame へ commit できません")
    if isinstance(context.current_time_sec, bool) or not math.isfinite(context.current_time_sec):
        raise CandidateValidationError("commit 時刻は有限値が必要です")
    if context.current_time_sec < identity.created_time_sec:
        raise StaleCandidateError("候補作成前の時刻へ commit できません")


def _policy_id(policy: object, label: str) -> str:
    value = getattr(policy, "policy_id", None)
    if not isinstance(value, str) or not value.strip():
        raise CandidateValidationError(f"{label} policy_id がありません")
    return value


def _policy_method(
    policy: object, name: str,
) -> Callable[[ChainCommitCandidate, ChainCommitContext], object]:
    method = getattr(policy, name, None)
    if not callable(method):
        raise CandidateValidationError(f"policy に callable な {name} がありません")
    return cast(Callable[[ChainCommitCandidate, ChainCommitContext], object], method)


def _counter_snapshot(current: Mapping[int, int]) -> ColorCounts:
    """Counter入力を厳密検査して版比較可能な不変値へ変換する。"""
    copied: dict[int, int] = {}
    for color, count in current.items():
        if type(color) is not int or color not in PUYO_COLORS:
            raise CandidateValidationError("現在 Counter の色が不正です")
        if type(count) is not int or count < 0:
            raise CandidateValidationError("現在 Counter は通常色ごとの非負整数が必要です")
        copied[int(color)] = count
    return tuple(sorted(copied.items()))


def _counter_after(candidate: ChainCommitCandidate, current: Mapping[int, int]) -> ColorCounts:
    copied = dict(_counter_snapshot(current))
    for color, erased in candidate.erased_color_count:
        if copied.get(color, 0) < erased:
            raise CounterUnderflowError(f"色{color}の消去差分が現在 Counter を超えます")
        copied[color] -= erased
    return tuple(sorted(copied.items()))


def _effect_for(
    candidate: ChainCommitCandidate, current_counter: Mapping[int, int], context: ChainCommitContext,
    verification_policy: ChainCandidateVerificationPolicy | None,
    release_policy: ChainCandidateReleasePolicy | None,
) -> ChainCommitEffect:
    """互換commitとprepareで同じ検証・effect組立を使い、tx状態は変更しない。"""
    _validate_context(candidate, context)
    if verification_policy is None or release_policy is None:
        raise PolicyNotConnectedError("検証 policy と公開解除 policy の両方が必要です")
    counter_after = _counter_after(candidate, current_counter)
    verification_id = _policy_id(verification_policy, "verification")
    release_id = _policy_id(release_policy, "release")
    require_verified = _policy_method(verification_policy, "require_verified")
    require_release = _policy_method(release_policy, "require_release")
    if require_verified(candidate, context) is not None:
        raise CandidateValidationError("検証 policy は bool で承認できません")
    if require_release(candidate, context) is not None:
        raise CandidateValidationError("公開 policy は bool で承認できません")
    return ChainCommitEffect(candidate.candidate_id, candidate.final_grid, counter_after,
                             context.current_frame_idx, context.current_time_sec,
                             verification_id, release_id)


def _validate_counter_revision(revision: int) -> None:
    """内容比較だけでは検出できないABA更新をowner提供版へ結ぶ。"""
    if type(revision) is not int or revision < 0:
        raise CandidateValidationError("counter_revisionは非負整数が必要です")


class ChainCommitTransaction:
    """候補を一回だけcommitまたはdiscardする、単一owner専用の状態所有者。

    prepare後もstateはPENDINGだがcommit/別prepare/discardは拒否する。
    ticket発行後は再validateも拒否し、finalizeか適用前abortだけを認める。
    abortは両発行物を失効させ同じtxの再prepareを許す。排他lock自体は持たない。
    """

    def __init__(self, candidate: ChainCommitCandidate) -> None:
        self._candidate = candidate
        self._state = ChainCommitState.PENDING
        self._busy = False
        self._prepared: PreparedChainCommit | None = None
        self._apply_ticket: ChainCommitApplyTicket | None = None

    @property
    def candidate(self) -> ChainCommitCandidate:
        return self._candidate

    @property
    def state(self) -> ChainCommitState:
        return self._state

    def _require_pending(self) -> None:
        if self._busy:
            raise TransactionConsumedError("検証中 transaction への再入はできません")
        if self._state is not ChainCommitState.PENDING:
            raise TransactionConsumedError(f"transaction は {self._state.value} 済みです")

    def _require_no_prepared(self) -> None:
        """未完prepareを互換commitや別prepareで迂回させない。"""
        self._require_pending()
        if self._prepared is not None:
            raise PreparedCommitError("既存prepareをfinalizeまたは適用前abortしてください")

    def _require_prepared(self, prepared: PreparedChainCommit) -> None:
        """等値コピーでなく、このtxが発行した同一objectだけを認める。"""
        self._require_pending()
        if self._prepared is None or prepared is not self._prepared:
            raise PreparedCommitError("別tx・偽造・失効済みのprepareです")

    def commit(
        self, current_counter: Mapping[int, int], context: ChainCommitContext,
        verification_policy: ChainCandidateVerificationPolicy | None = None,
        release_policy: ChainCandidateReleasePolicy | None = None,
    ) -> ChainCommitEffect:
        """外部2 policy と世代が通った場合だけ、一回性 effect を返す。"""
        self._require_no_prepared()
        self._busy = True
        try:
            effect = _effect_for(self._candidate, current_counter, context,
                                 verification_policy, release_policy)
            self._state = ChainCommitState.COMMITTED
            return effect
        finally:
            self._busy = False

    def prepare(
        self, current_counter: Mapping[int, int], context: ChainCommitContext,
        verification_policy: ChainCandidateVerificationPolicy | None = None,
        release_policy: ChainCandidateReleasePolicy | None = None,
        *, counter_revision: int,
    ) -> PreparedChainCommit:
        """未消費effectを作る。例外時はPENDINGで、同じtxから再試行できる。"""
        self._require_no_prepared()
        self._busy = True
        try:
            _validate_counter_revision(counter_revision)
            before = _counter_snapshot(current_counter)
            effect = _effect_for(self._candidate, dict(before), context,
                                 verification_policy, release_policy)
            if _counter_snapshot(current_counter) != before:
                raise PreparedCommitError("prepare中に現在Counterが変更されました")
            prepared = PreparedChainCommit(effect, context, before, counter_revision)
            self._prepared = prepared
            return prepared
        finally:
            self._busy = False

    def validate_prepared(
        self, prepared: PreparedChainCommit, current_counter: Mapping[int, int],
        context: ChainCommitContext, *, counter_revision: int,
    ) -> ChainCommitApplyTicket:
        """ownerの同一lock内でswap直前に実現在版を照合し、ack専用ticketを発行。"""
        self._require_prepared(prepared)
        if self._apply_ticket is not None:
            raise PreparedCommitError("適用ticketは発行済みです")
        self._busy = True
        try:
            _validate_counter_revision(counter_revision)
            _validate_context(self._candidate, context)
            if context != prepared.context:
                raise PreparedCommitError("prepare後にcontext/frame/時刻が変更されました")
            if counter_revision != prepared.counter_revision:
                raise PreparedCommitError("prepare後にCounter版が変更されました")
            if _counter_snapshot(current_counter) != prepared.counter_before:
                raise PreparedCommitError("prepare後にCounter内容が変更されました")
            ticket = ChainCommitApplyTicket(prepared)
            self._apply_ticket = ticket
            return ticket
        finally:
            self._busy = False

    def finalize(self, ticket: ChainCommitApplyTicket) -> ChainCommitEffect:
        """ownerのswap成功直後のack。外部callback/変換/再検証は実行しない。

        正規ticketを事前取得しlockを保ったownerにのみ使用権がある。外部swapを
        観測せず、その成功を検証済みとは主張しない。既存commitの後戻しはしない。
        """
        self._require_pending()
        prepared = self._prepared
        if prepared is None or self._apply_ticket is None or ticket is not self._apply_ticket:
            raise PreparedCommitError("適用前検査なし・別tx・偽造・失効済みticketです")
        effect = prepared.effect
        self._state = ChainCommitState.COMMITTED
        self._prepared = None
        self._apply_ticket = None
        return effect

    def abort_prepared(self, prepared: PreparedChainCommit, reason: str) -> ChainDiscardReceipt:
        """外部適用前だけの中断。ticketを失効させ、同じtxをPENDINGで保持する。

        外部状態へ部分書込後のrollback APIではない。swap後は呼ばずfinalizeする。
        """
        self._require_prepared(prepared)
        self._busy = True
        try:
            if not isinstance(reason, str) or not reason.strip():
                raise CandidateValidationError("abort reasonは空にできません")
            receipt = ChainDiscardReceipt(self._candidate.candidate_id, reason.strip())
            self._prepared = None
            self._apply_ticket = None
            return receipt
        finally:
            self._busy = False

    def discard(self, reason: str) -> ChainDiscardReceipt:
        """候補を副作用なしで一回だけ破棄する。"""
        self._require_no_prepared()
        if not isinstance(reason, str) or not reason.strip():
            raise CandidateValidationError("discard reason は空にできません")
        receipt = ChainDiscardReceipt(self._candidate.candidate_id, reason.strip())
        self._state = ChainCommitState.DISCARDED
        return receipt


__all__ = [
    "CandidateValidationError", "ChainCandidateReleasePolicy",
    "ChainCandidateVerificationPolicy", "ChainCommitCandidate",
    "ChainCommitContext", "ChainCommitEffect", "ChainCommitError",
    "ChainCommitIdentity", "ChainCommitState", "ChainCommitTransaction", "ChainCommitApplyTicket",
    "ChainDiscardReceipt", "CounterUnderflowError", "PolicyNotConnectedError",
    "StaleCandidateError", "TransactionConsumedError",
    "PreparedChainCommit", "PreparedCommitError",
    "create_chain_commit_candidate",
]
