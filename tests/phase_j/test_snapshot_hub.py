"""Phase J SnapshotHubのS06〜S08契約試験。"""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty
from threading import Barrier
from typing import Any

import pytest

pytest.importorskip("jsonschema")

from src.phase_j.contracts import OverlaySnapshot
from src.phase_j.snapshot_hub import (
    InvalidInitialSnapshotError,
    SnapshotHub,
    SnapshotHubError,
    SnapshotIdentityConflictError,
    SnapshotRevisionStaleError,
    SnapshotSessionMismatchError,
)
from src.phase_j.validator import validate_snapshot

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs" / "schemas" / "examples"
CONCURRENT_PUBLISH_COUNT = 24
RACE_SUBSCRIBER_COUNT = 12
BURST_COUNT = 60


def _payload(name: str = "initial_hidden.json") -> dict[str, Any]:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def _snapshot(
    name: str = "initial_hidden.json",
    *,
    revision: int | None = None,
    stream_seq: int | None = None,
    session_id: str | None = None,
) -> OverlaySnapshot:
    payload = _payload(name)
    identity = payload["identity"]
    if revision is not None:
        identity["reducer_revision"] = revision
    if stream_seq is not None:
        identity["stream_seq"] = stream_seq
    if session_id is not None:
        identity["session_id"] = session_id
    return OverlaySnapshot.from_mapping(payload)


def _seq(snapshot: OverlaySnapshot) -> int:
    return int(snapshot.identity["stream_seq"])


def test_initial_snapshot_is_validated_assigned_zero_and_not_mutated() -> None:
    initial = _snapshot(stream_seq=99)
    before = initial.to_mapping()
    hub = SnapshotHub(initial)
    assert _seq(hub.latest) == 0
    assert initial.to_mapping() == before
    assert validate_snapshot(hub.latest).is_valid


def test_invalid_initial_snapshot_is_rejected() -> None:
    payload = _payload()
    payload["evaluations"]["practical"]["availability"] = "available"
    invalid = OverlaySnapshot.from_mapping(payload)
    with pytest.raises(InvalidInitialSnapshotError) as caught:
        SnapshotHub(invalid)
    assert not caught.value.report.is_valid


def test_publish_assigns_stream_sequence_without_mutating_candidate() -> None:
    hub = SnapshotHub(_snapshot())
    candidate = _snapshot("live_practical.json", revision=1, stream_seq=777)
    before = candidate.to_mapping()
    result = hub.publish(candidate)
    assert result.published and not result.fail_closed
    assert result.validation_report.is_valid
    assert _seq(result.snapshot) == 1
    assert hub.latest == result.snapshot
    assert candidate.to_mapping() == before


def test_explicit_same_content_republish_assigns_a_new_sequence() -> None:
    hub = SnapshotHub(_snapshot())
    candidate = _snapshot("live_practical.json", revision=1, stream_seq=777)
    first = hub.publish(candidate)
    second = hub.publish(candidate)
    assert first.published and second.published
    assert [_seq(first.snapshot), _seq(second.snapshot)] == [1, 2]
    assert second.snapshot.content_digest() == hub.latest.content_digest()


def test_invalid_publish_is_explicitly_converted_to_legal_fail_closed() -> None:
    hub = SnapshotHub(_snapshot())
    payload = _payload("live_practical.json")
    payload["identity"]["reducer_revision"] = 1
    payload["evaluations"]["practical"]["p2_win_probability"] = 0.5
    candidate = OverlaySnapshot.from_mapping(payload)
    before = candidate.to_mapping()
    result = hub.publish(candidate)
    published = result.snapshot.to_mapping()
    assert result.fail_closed and not result.validation_report.is_valid
    assert published["display"]["visibility"] == "hidden"
    assert published["display"]["status"] == "integrity_fault"
    assert published["evaluations"]["practical"]["availability"] == "unavailable"
    assert validate_snapshot(result.snapshot).is_valid
    assert candidate.to_mapping() == before


def test_invalid_future_revision_does_not_advance_revision_high_water() -> None:
    hub = SnapshotHub(_snapshot())
    payload = _payload("live_practical.json")
    payload["identity"]["reducer_revision"] = 100
    payload["identity"]["stream_seq"] = 777
    payload["evaluations"]["practical"]["p2_win_probability"] = 0.5
    invalid = hub.publish(OverlaySnapshot.from_mapping(payload))
    assert invalid.fail_closed
    assert invalid.snapshot.identity["reducer_revision"] == 0
    assert _seq(invalid.snapshot) == 1
    assert validate_snapshot(invalid.snapshot).is_valid

    legitimate = hub.publish(_snapshot("live_practical.json", revision=1, stream_seq=777))
    assert not legitimate.fail_closed
    assert legitimate.snapshot.identity["reducer_revision"] == 1
    assert _seq(legitimate.snapshot) == 2
    assert validate_snapshot(legitimate.snapshot).is_valid


def test_s06_full_subscriber_buffer_discards_only_old_snapshot() -> None:
    hub = SnapshotHub(_snapshot())
    subscription = hub.subscribe()
    for index in range(1, BURST_COUNT + 1):
        hub.publish(_snapshot("live_practical.json", revision=1, stream_seq=999))
    latest = subscription.get_nowait()
    assert _seq(latest) == BURST_COUNT
    assert latest.content_digest() == hub.latest.content_digest()
    with pytest.raises(Empty):
        subscription.get_nowait()


def test_s06_slow_consumers_converge_to_same_latest_content() -> None:
    hub = SnapshotHub(_snapshot())
    subscriptions = [hub.subscribe() for _ in range(5)]
    for _ in range(BURST_COUNT):
        hub.publish(_snapshot("live_practical.json", revision=1, stream_seq=999))
    snapshots = [subscription.get_nowait() for subscription in subscriptions]
    assert {_seq(snapshot) for snapshot in snapshots} == {BURST_COUNT}
    assert {snapshot.content_digest() for snapshot in snapshots} == {hub.latest.content_digest()}


def test_s07_subscribe_publish_race_never_loses_latest_update() -> None:
    hub = SnapshotHub(_snapshot())
    barrier = Barrier(RACE_SUBSCRIBER_COUNT + 1)

    def subscribe() -> Any:
        barrier.wait()
        return hub.subscribe()

    def publish() -> None:
        barrier.wait()
        hub.publish(_snapshot("live_practical.json", revision=1, stream_seq=999))

    with ThreadPoolExecutor(max_workers=RACE_SUBSCRIBER_COUNT + 1) as executor:
        subscribers = [executor.submit(subscribe) for _ in range(RACE_SUBSCRIBER_COUNT)]
        publisher = executor.submit(publish)
        subscriptions = [future.result() for future in subscribers]
        publisher.result()
    received = [subscription.get_nowait() for subscription in subscriptions]
    assert {_seq(snapshot) for snapshot in received} == {1}
    assert {snapshot.content_digest() for snapshot in received} == {hub.latest.content_digest()}


def test_s07_concurrent_publish_assigns_one_contiguous_sequence() -> None:
    hub = SnapshotHub(_snapshot())
    candidate = _snapshot("live_practical.json", revision=1, stream_seq=999)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: hub.publish(candidate), range(CONCURRENT_PUBLISH_COUNT)))
    assigned = sorted(_seq(result.snapshot) for result in results)
    assert assigned == list(range(1, CONCURRENT_PUBLISH_COUNT + 1))
    assert _seq(hub.latest) == CONCURRENT_PUBLISH_COUNT


def test_subscribe_immediately_receives_current_and_unsubscribe_is_idempotent() -> None:
    hub = SnapshotHub(_snapshot())
    hub.publish(_snapshot("live_practical.json", revision=1, stream_seq=999))
    subscription = hub.subscribe()
    assert _seq(subscription.get_nowait()) == 1
    assert hub.subscriber_count == 1
    hub.unsubscribe(subscription)
    hub.unsubscribe(subscription)
    assert hub.subscriber_count == 0


def test_same_identity_same_content_is_idempotent() -> None:
    initial = _snapshot()
    hub = SnapshotHub(initial)
    result = hub.publish(initial)
    assert not result.published
    assert not result.fail_closed
    assert _seq(hub.latest) == 0


def test_same_session_revision_and_sequence_with_different_content_is_rejected() -> None:
    hub = SnapshotHub(_snapshot())
    subscription = hub.subscribe()
    before = hub.latest.to_mapping()
    before_digest = hub.latest.content_digest()
    payload = _payload()
    payload["assets"]["app_build_id"] = "different-build"
    conflicting = OverlaySnapshot.from_mapping(payload)
    with pytest.raises(SnapshotIdentityConflictError):
        hub.publish(conflicting)
    received = subscription.get_nowait()
    assert _seq(hub.latest) == 0
    assert hub.latest.to_mapping() == before
    assert hub.latest.content_digest() == before_digest
    assert received.to_mapping() == before
    with pytest.raises(Empty):
        subscription.get_nowait()


def test_invalid_identity_is_rejected_without_changing_latest() -> None:
    hub = SnapshotHub(_snapshot())
    before = hub.latest.to_mapping()
    payload = _payload("live_practical.json")
    del payload["identity"]["reducer_revision"]
    with pytest.raises(SnapshotHubError, match="順序識別子"):
        hub.publish(OverlaySnapshot.from_mapping(payload))
    assert hub.latest.to_mapping() == before


def test_session_mismatch_and_revision_regression_are_rejected() -> None:
    hub = SnapshotHub(_snapshot())
    with pytest.raises(SnapshotSessionMismatchError):
        hub.publish(_snapshot("live_practical.json", revision=1, session_id="other-session"))
    hub.publish(_snapshot("live_practical.json", revision=2, stream_seq=999))
    with pytest.raises(SnapshotRevisionStaleError):
        hub.publish(_snapshot("live_practical.json", revision=1, stream_seq=999))
    assert _seq(hub.latest) == 1


def test_s08_30fps_style_burst_has_no_old_fifo_and_final_convergence() -> None:
    hub = SnapshotHub(_snapshot())
    fast = hub.subscribe()
    slow = hub.subscribe()
    fast_seen = []
    for index in range(1, BURST_COUNT + 1):
        result = hub.publish(_snapshot("live_practical.json", revision=1, stream_seq=999))
        fast_seen.append(_seq(fast.get_nowait()))
        assert _seq(result.snapshot) == index
    slow_latest = slow.get_nowait()
    assert fast_seen == list(range(1, BURST_COUNT + 1))
    assert _seq(slow_latest) == BURST_COUNT
    assert slow_latest.content_digest() == hub.latest.content_digest()


def test_candidate_mapping_remains_deeply_unchanged_after_publish() -> None:
    hub = SnapshotHub(_snapshot())
    candidate = _snapshot("both_lanes.json", revision=1, stream_seq=999)
    original = copy.deepcopy(candidate.to_mapping())
    hub.publish(candidate)
    assert candidate.to_mapping() == original
