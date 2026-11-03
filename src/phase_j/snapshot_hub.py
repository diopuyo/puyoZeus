"""Phase J snapshotをlatest-winsで配信するthread-safe Hub。"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock

from .contracts import OverlaySnapshot
from .validator import (
    ValidationContext,
    ValidationReport,
    make_fail_closed_snapshot,
    validate_snapshot,
)


class SnapshotHubError(RuntimeError):
    """SnapshotHubがpublishを安全に完了できない場合の基底例外。"""


class InvalidInitialSnapshotError(SnapshotHubError):
    """初期snapshotが公開契約を満たさない。"""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("初期snapshotがPhase J公開契約を満たしません")


class SnapshotSessionMismatchError(SnapshotHubError):
    """別sessionのsnapshotが現在Hubへ届いた。"""


class SnapshotRevisionStaleError(SnapshotHubError):
    """reducer revisionが現在値より後退した。"""


class SnapshotIdentityConflictError(SnapshotHubError):
    """同一session/revision/stream_seqに異なる内容が届いた。"""


@dataclass(frozen=True, slots=True)
class PublishResult:
    """publish結果とfail-closed変換の有無。"""

    snapshot: OverlaySnapshot
    validation_report: ValidationReport
    fail_closed: bool
    published: bool = True


@dataclass(frozen=True, slots=True)
class SnapshotSubscription:
    """一subscriber専用の最大1件受信口。"""

    subscriber_id: int
    _queue: Queue[OverlaySnapshot]

    def get(self, timeout: float | None = None) -> OverlaySnapshot:
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> OverlaySnapshot:
        return self._queue.get_nowait()

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()


def _with_identity(
    snapshot: OverlaySnapshot,
    stream_seq: int,
    reducer_revision: int | None = None,
) -> OverlaySnapshot:
    """入力を変えず、Hub管理identityへ差し替えた複製を返す。"""
    payload = snapshot.to_mapping()
    payload["identity"]["stream_seq"] = stream_seq
    if reducer_revision is not None:
        payload["identity"]["reducer_revision"] = reducer_revision
    return OverlaySnapshot.from_mapping(payload)


def _with_stream_seq(snapshot: OverlaySnapshot, stream_seq: int) -> OverlaySnapshot:
    """candidateのstream_seqはplaceholderとして常にHub採番値で上書きする。"""
    return _with_identity(snapshot, stream_seq)


def _identity(snapshot: OverlaySnapshot) -> tuple[str, int, int]:
    identity = snapshot.identity
    session_id = identity.get("session_id")
    revision = identity.get("reducer_revision")
    stream_seq = identity.get("stream_seq")
    valid_ints = all(isinstance(value, int) and not isinstance(value, bool) for value in (revision, stream_seq))
    if not isinstance(session_id, str) or not session_id or not valid_ints:
        raise SnapshotHubError("snapshotの順序識別子が不正です")
    return session_id, revision, stream_seq


class SnapshotHub:
    """最新snapshot一件とsubscriber群を一つのlockで管理する。"""

    def __init__(
        self,
        initial_snapshot: OverlaySnapshot,
        validation_context: ValidationContext | None = None,
    ) -> None:
        self._lock = Lock()
        self._validation_context = validation_context or ValidationContext()
        initial = _with_stream_seq(initial_snapshot, 0)
        report = validate_snapshot(initial, self._validation_context)
        if not report.is_valid:
            raise InvalidInitialSnapshotError(report)
        session_id, revision, _ = _identity(initial)
        self._session_id = session_id
        self._latest_revision = revision
        self._next_stream_seq = 1
        self._latest = initial
        self._latest_digest = initial.content_digest()
        self._subscribers: dict[int, Queue[OverlaySnapshot]] = {}
        self._next_subscriber_id = 1

    @property
    def latest(self) -> OverlaySnapshot:
        with self._lock:
            return self._latest

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def subscribe(self) -> SnapshotSubscription:
        """登録と現在snapshot投入をpublishと同じlock内で行う。"""
        with self._lock:
            subscriber_id = self._next_subscriber_id
            self._next_subscriber_id += 1
            queue: Queue[OverlaySnapshot] = Queue(maxsize=1)
            self._subscribers[subscriber_id] = queue
            queue.put_nowait(self._latest)
            return SnapshotSubscription(subscriber_id, queue)

    def unsubscribe(self, subscription: SnapshotSubscription) -> None:
        """既に解除済みでも成功する。"""
        with self._lock:
            self._subscribers.pop(subscription.subscriber_id, None)

    def _idempotent_result(self, candidate: OverlaySnapshot) -> PublishResult | None:
        session_id, revision, claimed_seq = _identity(candidate)
        if session_id != self._session_id:
            raise SnapshotSessionMismatchError("現在Hubと異なるsessionです")
        if revision < self._latest_revision:
            raise SnapshotRevisionStaleError("reducer revisionが後退しています")
        same_identity = revision == self._latest_revision and claimed_seq == self._latest.identity["stream_seq"]
        if not same_identity:
            return None
        normalized = _with_stream_seq(candidate, claimed_seq)
        if normalized.content_digest() != self._latest_digest:
            raise SnapshotIdentityConflictError("同一snapshot identityに異なる内容があります")
        report = validate_snapshot(normalized, self._validation_context)
        return PublishResult(self._latest, report, fail_closed=False, published=False)

    def _validated_candidate(
        self,
        candidate: OverlaySnapshot,
        stream_seq: int,
    ) -> tuple[OverlaySnapshot, ValidationReport, bool]:
        assigned = _with_stream_seq(candidate, stream_seq)
        report = validate_snapshot(assigned, self._validation_context)
        if report.is_valid:
            return assigned, report, False
        fail_closed = make_fail_closed_snapshot(assigned, report)
        fail_closed = _with_identity(fail_closed, stream_seq, self._latest_revision)
        closed_report = validate_snapshot(fail_closed, self._validation_context)
        if not closed_report.is_valid:
            raise SnapshotHubError("fail-closed snapshotが公開契約を満たしません")
        return fail_closed, report, True

    @staticmethod
    def _put_latest(queue: Queue[OverlaySnapshot], snapshot: OverlaySnapshot) -> None:
        try:
            queue.put_nowait(snapshot)
            return
        except Full:
            pass
        try:
            queue.get_nowait()
        except Empty:
            pass
        queue.put_nowait(snapshot)

    def publish(self, candidate: OverlaySnapshot) -> PublishResult:
        """placeholderを採番し、検証・保存・通知を原子的に完了する。

        現在latestとidentity・内容が同一の再送だけは冪等とする。同じ内容でも
        placeholder identityによる明示publishは、新しいstream_seqを採番する。
        """
        with self._lock:
            # identity不正時は安全な順序を採番できないため現段階では例外とする。
            # J2/J3のSingleWriter runtimeが別のfail-closed snapshotをpublishする。
            idempotent = self._idempotent_result(candidate)
            if idempotent is not None:
                return idempotent
            stream_seq = self._next_stream_seq
            published, report, fail_closed = self._validated_candidate(candidate, stream_seq)
            session_id, revision, assigned_seq = _identity(published)
            if session_id != self._session_id or assigned_seq != stream_seq:
                raise SnapshotHubError("検証後snapshotの順序識別子が変化しました")
            if fail_closed:
                if revision != self._latest_revision:
                    raise SnapshotHubError("fail-closed snapshotがrevision水位を変更しました")
            else:
                self._latest_revision = revision
            self._next_stream_seq += 1
            self._latest = published
            self._latest_digest = published.content_digest()
            for queue in self._subscribers.values():
                self._put_latest(queue, published)
            return PublishResult(published, report, fail_closed=fail_closed)


__all__ = [
    "InvalidInitialSnapshotError",
    "PublishResult",
    "SnapshotHub",
    "SnapshotHubError",
    "SnapshotIdentityConflictError",
    "SnapshotRevisionStaleError",
    "SnapshotSessionMismatchError",
    "SnapshotSubscription",
]
