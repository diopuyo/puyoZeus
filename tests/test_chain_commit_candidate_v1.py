"""未配線C-6候補helperの同定・承認境界・一回commitを独立CPU検収する。"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import collect_boards_lean as collector
from src import chain_commit_candidate_v1 as commit
from src.board import Board
from src.chain_detector import ChainEvent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/verify/video38_guard_control_lineage_2026-09-07_v1/frames.jsonl"
SOURCE_SHA = "86e45ee5bb9f89d34b4608b88a2367b2486894dffe659b77d1246283145b36c2"
FRAME, SOURCE_LINE = 34370, 7895
FPS, EPOCH, EVENT_REVISION, ACTION_REVISION = 60, 3, 7, 11
COLORS = tuple(range(1, 6))
EXTRA_COLOR, EXTRA_COUNT = 2, 2
COUNTER_REVISION = 19


@pytest.fixture(scope="module")
def measured() -> dict[str, Any]:
    """実46／旧final56の固定入力だけを読み、動画やモデルには触れない。"""
    content = SOURCE.read_bytes()
    assert hashlib.sha256(content).hexdigest() == SOURCE_SHA
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines()]
    found = [r for r in rows if r["kind"] == "control_c6_lineage"
             and r["frame_idx"] == FRAME and r["caller_line"] == SOURCE_LINE]
    assert len(found) == 1
    return found[0]


def _board(value: dict[str, Any]) -> Board:
    """各テストへ独立したmutable Boardを渡す。"""
    return Board.from_list(value["grid"])


def _counts(board: Board) -> Counter[int]:
    """現行色会計と同じ色ぷよだけを数える。"""
    return Counter({color: int(np.count_nonzero(board._grid == color)) for color in COLORS})


def _event(measured: dict[str, Any]) -> ChainEvent:
    """C-6が参照する実eventを再構成する。"""
    value, result = measured["effective_event"], measured["simulate_result"]
    return ChainEvent(trigger_sec=value["trigger_sec"], end_sec=value["end_sec"],
                      before_board=_board(value["before_board"]), chain_count=value["chain_count"],
                      total_erased=result["total_erased"], total_score=value["total_score"],
                      base_score=value["total_score"], all_clear_bonus_applied=0,
                      ojama_sent=0, leftover_score=0, is_all_clear=False,
                      mechanism=value["mechanism"], score_estimated=value["score_estimated"])


def _candidate(measured: dict[str, Any], **overrides: Any) -> Any:
    """保存inputから候補を作り、runtimeの盤面・Counterとは切り離す。"""
    event = _event(measured)
    final = _board(measured["simulate_result"]["final_board"])
    erased = _counts(event.before_board) - _counts(final)
    values = {"side": "1P", "reset_epoch": EPOCH, "event_revision": EVENT_REVISION,
              "action_revision": ACTION_REVISION, "created_frame_idx": FRAME,
              "created_time_sec": FRAME / FPS, "event": event, "final_board": final,
              "erased_color_count": erased}
    return commit.create_chain_commit_candidate(**(values | overrides))


def _context(candidate: Any, **overrides: Any) -> Any:
    """候補対象と検証完了時刻を結ぶ。将来の実validatorとは別のCPU fixture。"""
    values = {"side": "1P", "reset_epoch": EPOCH, "event_sha256": candidate.identity.event_sha256,
              "event_revision": EVENT_REVISION, "action_revision": ACTION_REVISION,
              "current_frame_idx": FRAME + 8, "current_time_sec": (FRAME + 8) / FPS}
    return commit.ChainCommitContext(**(values | overrides))


class _TestOnlyPolicy:
    """機械的なcommitテスト専用。実盤面や連鎖終端の検証済み証拠ではない。"""

    policy_id = "cpu-test-double-not-production-verification"

    def require_verified(self, candidate: Any, context: Any) -> None:
        """テスト条件でだけ明示的に成功するtrusted adapter代役。"""
        return None

    def require_release(self, candidate: Any, context: Any) -> None:
        """公開権判定の実装は未配線。代役の成功を実終了解除と呼ばない。"""
        return None


def _commit(transaction: Any, counts: Counter[int], context: Any) -> Any:
    """成功側に必要な両policyを必ず明示して渡す。"""
    return transaction.commit(counts, context, verification_policy=_TestOnlyPolicy(),
                              release_policy=_TestOnlyPolicy())


def test_measured_stale_final_creation_does_not_mutate_current_or_counter(
    measured: dict[str, Any],
) -> None:
    current = _board(measured["confirmed"])
    pending, t2 = current.copy(), current.copy()
    counts = _counts(_event(measured).before_board)
    originals = (current._grid.copy(), pending._grid.copy(), t2._grid.copy(), counts.copy())
    candidate = _candidate(measured)
    commit.ChainCommitTransaction(candidate)
    assert measured["confirmed"]["color"] == 46
    assert measured["simulate_result"]["final_board"]["color"] == 56
    for board, original in zip((current, pending, t2), originals[:3]):
        np.testing.assert_array_equal(board._grid, original)
    assert counts == originals[3]
    np.testing.assert_array_equal(candidate.copy_final_board()._grid,
                                  measured["simulate_result"]["final_board"]["grid"])


def test_candidate_snapshots_and_returned_copies_are_detached(measured: dict[str, Any]) -> None:
    event = _event(measured)
    final = _board(measured["simulate_result"]["final_board"])
    erased = _counts(event.before_board) - _counts(final)
    candidate = _candidate(measured, event=event, final_board=final, erased_color_count=erased)
    before_copy, final_copy = candidate.copy_before_board(), candidate.copy_final_board()
    event.before_board._grid.fill(0)
    final._grid.fill(0)
    erased.clear()
    candidate.copy_before_board()._grid.fill(0)
    candidate.copy_final_board()._grid.fill(0)
    np.testing.assert_array_equal(candidate.copy_before_board()._grid, before_copy._grid)
    np.testing.assert_array_equal(candidate.copy_final_board()._grid, final_copy._grid)


@pytest.mark.parametrize("verification,release", [(False, False), (True, False), (False, True)])
def test_both_real_policy_connections_are_required(measured: dict[str, Any],
                                                  verification: bool, release: bool) -> None:
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    counts = _counts(_event(measured).before_board)
    before = counts.copy()
    with pytest.raises(commit.PolicyNotConnectedError):
        tx.commit(counts, _context(candidate),
                  verification_policy=_TestOnlyPolicy() if verification else None,
                  release_policy=_TestOnlyPolicy() if release else None)
    assert counts == before
    # 失敗では消費されず、正しく両policyを接続した次の試行が可能。
    _commit(tx, counts, _context(candidate))


def test_raw_majority_and_elapsed_time_do_not_implicitly_authorize(measured: dict[str, Any]) -> None:
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    raw_votes = [candidate.copy_final_board() for _ in range(5)]
    assert all(np.array_equal(v._grid, raw_votes[0]._grid) for v in raw_votes)
    with pytest.raises(commit.PolicyNotConnectedError):
        tx.commit(_counts(_event(measured).before_board),
                  _context(candidate, current_frame_idx=FRAME * 2, current_time_sec=FRAME * 2 / FPS))
    with pytest.raises(TypeError):
        tx.commit(_counts(_event(measured).before_board), _context(candidate),
                  verified=True, timed_out=True)


@pytest.mark.parametrize("field,value", [("side", "2P"), ("reset_epoch", EPOCH + 1),
    ("event_revision", EVENT_REVISION + 1), ("action_revision", ACTION_REVISION + 1),
    ("event_sha256", "0" * 64), ("current_frame_idx", FRAME - 1),
    ("current_time_sec", FRAME / FPS - 1)])
def test_stale_context_rejected_before_commit(measured: dict[str, Any], field: str, value: Any) -> None:
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    counts = _counts(_event(measured).before_board)
    before = counts.copy()
    with pytest.raises(commit.StaleCandidateError):
        _commit(tx, counts, _context(candidate, **{field: value}))
    assert counts == before
    _commit(tx, counts, _context(candidate))


def test_same_before_with_different_event_is_not_same_candidate(measured: dict[str, Any]) -> None:
    first = _candidate(measured)
    later_event = replace(_event(measured), trigger_sec=FRAME / FPS, end_sec=(FRAME + 1) / FPS)
    second = _candidate(measured, event=later_event)
    np.testing.assert_array_equal(first.copy_before_board()._grid, second.copy_before_board()._grid)
    assert first.identity.event_sha256 != second.identity.event_sha256
    with pytest.raises(commit.StaleCandidateError):
        _commit(commit.ChainCommitTransaction(second), _counts(later_event.before_board), _context(first))


def test_commit_preserves_intervening_counter_increment_and_returns_detached_effect(
    measured: dict[str, Any],
) -> None:
    candidate = _candidate(measured)
    counts = _counts(_event(measured).before_board)
    counts[EXTRA_COLOR] += EXTRA_COUNT
    original = counts.copy()
    effect = _commit(commit.ChainCommitTransaction(candidate), counts, _context(candidate))
    expected = _counts(candidate.copy_final_board())
    expected[EXTRA_COLOR] += EXTRA_COUNT
    assert Counter(effect.copy_counter()) == expected
    assert counts == original
    np.testing.assert_array_equal(effect.copy_final_board()._grid, candidate.copy_final_board()._grid)
    effect.copy_counter().clear()
    effect.copy_final_board()._grid.fill(0)
    assert Counter(effect.copy_counter()) == expected
    np.testing.assert_array_equal(effect.copy_final_board()._grid, candidate.copy_final_board()._grid)


@pytest.mark.parametrize("first", ["commit", "discard"])
def test_consumed_transaction_rejects_second_commit_or_discard(measured: dict[str, Any], first: str) -> None:
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    counts, context = _counts(_event(measured).before_board), _context(candidate)
    if first == "commit":
        _commit(tx, counts, context)
    else:
        tx.discard("verified_current_disagrees_with_stale_final")
    original = counts.copy()
    with pytest.raises(commit.TransactionConsumedError):
        _commit(tx, counts, context)
    with pytest.raises(commit.TransactionConsumedError):
        tx.discard("duplicate")
    assert counts == original


def test_counter_underflow_rejects_without_clamp_or_partial_commit(measured: dict[str, Any]) -> None:
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    counts = Counter({color: 0 for color in COLORS})
    before = counts.copy()
    with pytest.raises(commit.CounterUnderflowError):
        _commit(tx, counts, _context(candidate))
    assert counts == before
    _commit(tx, _counts(_event(measured).before_board), _context(candidate))


@pytest.mark.parametrize("method", ["require_verified", "require_release"])
def test_policy_exception_leaves_pending_and_all_inputs_unchanged(
    measured: dict[str, Any], method: str,
) -> None:
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    counts, before_board = _counts(_event(measured).before_board), candidate.copy_before_board()
    original = counts.copy()
    policy = _TestOnlyPolicy()
    def fail(*args: Any) -> None:
        raise RuntimeError("injected policy failure")
    setattr(policy, method, fail)
    with pytest.raises(RuntimeError, match="injected policy failure"):
        tx.commit(counts, _context(candidate), verification_policy=policy, release_policy=policy)
    assert counts == original
    np.testing.assert_array_equal(candidate.copy_before_board()._grid, before_board._grid)
    _commit(tx, counts, _context(candidate))


class _DrainRecorder:
    """既存lean差分処理からの架空着地呼出を数えるだけの代役。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def on_tsumo_settled(self, side: str, time_sec: float) -> None:
        self.calls.append((side, time_sec))


def test_deferred_commit_and_discard_do_not_create_phantom_positive_tsumo_delta(
    measured: dict[str, Any],
) -> None:
    candidate = _candidate(measured)
    counts = _counts(_event(measured).before_board)
    tracker = _DrainRecorder()
    state = collector._SideState(ojama_prev_tsumo=sum(counts.values()) // 2)
    counts[EXTRA_COLOR] += EXTRA_COUNT
    collector._drain_ojama_by_tsumo_delta_lean(tracker, "p1", state, sum(counts.values()) // 2, 1.0)
    assert len(tracker.calls) == 1  # 実際に加えた2cellの着地分だけ。
    tracker.calls.clear()
    tx = commit.ChainCommitTransaction(candidate)
    effect = _commit(tx, counts, _context(candidate))
    after = effect.copy_counter()
    collector._drain_ojama_by_tsumo_delta_lean(tracker, "p1", state, sum(after.values()) // 2, 2.0)
    with pytest.raises(commit.TransactionConsumedError):
        _commit(tx, Counter(after), _context(candidate))
    commit.ChainCommitTransaction(candidate).discard("stale candidate without accounting commit")
    collector._drain_ojama_by_tsumo_delta_lean(tracker, "p1", state, sum(after.values()) // 2, 3.0)
    assert tracker.calls == []


@pytest.mark.parametrize("method", ["require_verified", "require_release"])
@pytest.mark.parametrize("response", [False, True, "verified", {"verified": True}])
def test_boolean_or_unvalidated_receipt_is_not_policy_success(
    measured: dict[str, Any], method: str, response: Any,
) -> None:
    """例外なしでもboolや任意receiptを承認Noneと取り違えない。"""
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    counts = _counts(_event(measured).before_board)
    policy = _TestOnlyPolicy()
    setattr(policy, method, lambda *args: response)
    with pytest.raises(commit.CandidateValidationError):
        tx.commit(counts, _context(candidate), verification_policy=policy, release_policy=policy)
    assert tx.state is commit.ChainCommitState.PENDING
    _commit(tx, counts, _context(candidate))


@pytest.mark.parametrize("broken", [True, False, object()])
def test_free_value_cannot_impersonate_connected_policy(measured: dict[str, Any], broken: Any) -> None:
    """未検証boolや型なしtokenを接続済み検証器として受理しない。"""
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    with pytest.raises(commit.CandidateValidationError):
        tx.commit(_counts(_event(measured).before_board), _context(candidate),
                  verification_policy=broken, release_policy=_TestOnlyPolicy())
    assert tx.state is commit.ChainCommitState.PENDING


@pytest.mark.parametrize("fault", ["blank_id", "missing_method"])
def test_policy_metadata_and_callable_contract_are_required(measured: dict[str, Any], fault: str) -> None:
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    policy = _TestOnlyPolicy()
    if fault == "blank_id":
        policy.policy_id = " "
    else:
        policy.require_verified = None
    with pytest.raises(commit.CandidateValidationError):
        tx.commit(_counts(_event(measured).before_board), _context(candidate),
                  verification_policy=policy, release_policy=_TestOnlyPolicy())
    assert tx.state is commit.ChainCommitState.PENDING


@pytest.mark.parametrize("nested", ["commit", "discard"])
def test_policy_reentry_cannot_consume_transaction_partially(measured: dict[str, Any], nested: str) -> None:
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    counts, context = _counts(_event(measured).before_board), _context(candidate)
    before = counts.copy()
    policy = _TestOnlyPolicy()
    def reenter(*args: Any) -> None:
        if nested == "commit":
            _commit(tx, counts, context)
        else:
            tx.discard("nested callback")
    policy.require_release = reenter
    with pytest.raises(commit.TransactionConsumedError):
        tx.commit(counts, context, verification_policy=policy, release_policy=policy)
    assert tx.state is commit.ChainCommitState.PENDING and counts == before
    _commit(tx, counts, context)


def test_effect_uses_verification_time_and_stale_discard_does_not_commit(measured: dict[str, Any]) -> None:
    candidate = _candidate(measured)
    context = _context(candidate)
    counts = _counts(_event(measured).before_board)
    effect = _commit(commit.ChainCommitTransaction(candidate), counts, context)
    assert effect.committed_frame_idx == context.current_frame_idx
    assert effect.committed_time_sec == context.current_time_sec
    assert effect.committed_frame_idx > candidate.identity.created_frame_idx
    tx = commit.ChainCommitTransaction(candidate)
    before = counts.copy()
    tx.discard("current46_disagrees_with_old56; no erasure commit")
    assert tx.state is commit.ChainCommitState.DISCARDED and counts == before


def test_new_functions_obey_fifty_line_rule() -> None:
    """新規helperと独立テストに50行規約を適用する。"""
    for path in (Path(commit.__file__), Path(__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno - node.lineno + 1 <= 50, (path.name, node.name)


@pytest.mark.parametrize("field,value", [("reset_epoch", True), ("event_revision", 7.0),
    ("action_revision", False), ("created_frame_idx", float(FRAME)), ("created_time_sec", True)])
def test_candidate_metadata_rejects_bool_and_noninteger_revisions(
    measured: dict[str, Any], field: str, value: Any,
) -> None:
    """False==0や7.0==7で別世代を同定済みにしない。"""
    with pytest.raises(commit.CandidateValidationError):
        _candidate(measured, **{field: value})


@pytest.mark.parametrize("field,value", [("reset_epoch", float(EPOCH)),
    ("event_revision", float(EVENT_REVISION)), ("current_frame_idx", float(FRAME + 8)),
    ("current_time_sec", float("nan"))])
def test_current_context_rejects_invalid_metadata_without_consumption(
    measured: dict[str, Any], field: str, value: Any,
) -> None:
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    with pytest.raises(commit.CandidateValidationError):
        _commit(tx, _counts(_event(measured).before_board), _context(candidate, **{field: value}))
    assert tx.state is commit.ChainCommitState.PENDING


@pytest.mark.parametrize("value", [True, -1, 1.5, float("nan")])
def test_invalid_counter_value_cannot_become_committed_effect(measured: dict[str, Any], value: Any) -> None:
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    counts = _counts(_event(measured).before_board)
    counts[EXTRA_COLOR] = value
    with pytest.raises(commit.CandidateValidationError):
        _commit(tx, counts, _context(candidate))
    assert tx.state is commit.ChainCommitState.PENDING
    assert counts[EXTRA_COLOR] is value


@pytest.mark.parametrize("fault", ["bool", "wrong_delta", "extra_color"])
def test_erasure_delta_must_be_valid_and_match_before_final(measured: dict[str, Any], fault: str) -> None:
    erased = _counts(_event(measured).before_board) - _counts(_board(measured["simulate_result"]["final_board"]))
    color = next(iter(erased))
    if fault == "bool":
        erased[color] = True
    elif fault == "wrong_delta":
        erased[color] += 1
    else:
        erased[9] = 1
    with pytest.raises(commit.CandidateValidationError):
        _candidate(measured, erased_color_count=erased)


def test_effect_construction_exception_does_not_commit_or_mutate(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    counts, context = _counts(_event(measured).before_board), _context(candidate)
    before = counts.copy()
    def fail_effect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected effect assembly failure")
    with monkeypatch.context() as patch:
        patch.setattr(commit, "ChainCommitEffect", fail_effect)
        with pytest.raises(RuntimeError, match="effect assembly failure"):
            _commit(tx, counts, context)
    assert tx.state is commit.ChainCommitState.PENDING and counts == before
    _commit(tx, counts, context)


def test_partial_event_metadata_does_not_discard_full_thirteen_chain_prediction(
    measured: dict[str, Any],
) -> None:
    """event途中count/消去未充填をfull物理差分の拒否条件にしない。"""
    path = ROOT / "data/verify/video38_confirmed_collapse_2026-09-07_v3_details/CPU_FIXTURE.json"
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == "0cd5d81736bb91131848506a3efd4674e8a4a17da4a5b30e66db3f36825bbf6c"
    fixture = json.loads(content)
    before, final = Board.from_list(fixture["landing"]), Board.from_list(fixture["physical_final"])
    event = replace(_event(measured), before_board=before, chain_count=2, total_erased=0)
    candidate = _candidate(measured, event=event, final_board=final,
                           erased_color_count=_counts(before) - _counts(final))
    assert candidate.event.chain_count == 2 and candidate.event.total_erased == 0
    np.testing.assert_array_equal(candidate.copy_final_board()._grid, fixture["physical_final"])
    with pytest.raises(commit.PolicyNotConnectedError):
        commit.ChainCommitTransaction(candidate).commit(_counts(before), _context(candidate))


@pytest.mark.parametrize("nested", ["commit", "discard"])
def test_policy_id_property_reentry_is_guarded_before_attribute_access(
    measured: dict[str, Any], nested: str,
) -> None:
    """Protocolのpropertyも外部実行なので、承認methodより前から再入を拒否する。"""
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    counts, context = _counts(_event(measured).before_board), _context(candidate)
    original = counts.copy()
    class ReentrantPolicy(_TestOnlyPolicy):
        @property
        def policy_id(self) -> str:
            if nested == "commit":
                _commit(tx, counts, context)
            else:
                tx.discard("property reentry")
            return "reentrant-policy"
    with pytest.raises(commit.TransactionConsumedError):
        tx.commit(counts, context, verification_policy=ReentrantPolicy(),
                  release_policy=_TestOnlyPolicy())
    assert tx.state is commit.ChainCommitState.PENDING and counts == original
    _commit(tx, counts, context)


def _prepare(
    tx: commit.ChainCommitTransaction, counts: Counter[int], context: commit.ChainCommitContext,
    revision: int = COUNTER_REVISION,
) -> commit.PreparedChainCommit:
    """実検証器ではない明示policyでprepareの機械契約だけを調べる。"""
    return tx.prepare(counts, context, _TestOnlyPolicy(), _TestOnlyPolicy(),
                      counter_revision=revision)


def _prepared_case(measured: dict[str, Any]) -> tuple[Any, ...]:
    """同一txでの中断・再試行を検査する独立入力を用意する。"""
    candidate = _candidate(measured)
    tx = commit.ChainCommitTransaction(candidate)
    counts, context = _counts(_event(measured).before_board), _context(candidate)
    counts[EXTRA_COLOR] += EXTRA_COUNT
    return tx, counts, context, _prepare(tx, counts, context)


def _validate(
    tx: commit.ChainCommitTransaction, prepared: commit.PreparedChainCommit,
    counts: Counter[int], context: commit.ChainCommitContext, revision: int = COUNTER_REVISION,
) -> commit.ChainCommitApplyTicket:
    """適用直前のowner入力を明示してticketを得る。"""
    return tx.validate_prepared(prepared, counts, context, counter_revision=revision)


def test_prepare_preserves_legacy_effect_without_consuming_or_external_mutation(
    measured: dict[str, Any],
) -> None:
    tx, counts, context, prepared = _prepared_case(measured)
    original = counts.copy()
    # 独立txは互換比較専用。失敗復旧には新txを作らない。
    expected = _commit(commit.ChainCommitTransaction(tx.candidate), counts, context)
    assert prepared.effect == expected
    assert prepared.counter_before == tuple(sorted(counts.items()))
    assert prepared.counter_revision == COUNTER_REVISION
    assert tx.state is commit.ChainCommitState.PENDING and counts == original
    prepared.effect.copy_counter().clear()
    prepared.effect.copy_final_board()._grid.fill(0)
    assert prepared.effect == expected
    with pytest.raises(commit.PreparedCommitError):
        tx.finalize(commit.ChainCommitApplyTicket(prepared))


def test_owner_prebuild_single_swap_then_ack_has_no_late_callbacks(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """owner側の単一参照swap例。helperによる外部原子性の証明ではない。"""
    tx, counts, context, prepared = _prepared_case(measured)
    owner_state = (tx.candidate.before_grid, tuple(sorted(counts.items())), COUNTER_REVISION)
    original = owner_state
    replacement = (prepared.effect.final_grid, prepared.effect.counter_after, COUNTER_REVISION + 1)
    ticket = _validate(tx, prepared, counts, context)
    assert owner_state is original and tx.state is commit.ChainCommitState.PENDING
    def unexpected_callback(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ack must not allocate effect or revalidate")
    with monkeypatch.context() as patch:
        patch.setattr(commit, "_counter_snapshot", unexpected_callback)
        patch.setattr(commit, "_validate_context", unexpected_callback)
        patch.setattr(commit, "ChainCommitEffect", unexpected_callback)
        owner_state = replacement  # 同一ownerの排他区間内で、構築済み状態を一回swap。
        result = tx.finalize(ticket)
    assert owner_state is replacement and result is prepared.effect
    assert tx.state is commit.ChainCommitState.COMMITTED
    with pytest.raises(commit.TransactionConsumedError):
        tx.finalize(ticket)
    with pytest.raises(commit.TransactionConsumedError):
        tx.abort_prepared(prepared, "too late")


@pytest.mark.parametrize("validated", [False, True])
def test_outstanding_prepare_blocks_other_consumption_and_abort_invalidates_all(
    measured: dict[str, Any], validated: bool,
) -> None:
    """未完prepare/ticketを残した旧API迂回と、validate後中断ticket再使用を拒否。"""
    tx, counts, context, prepared = _prepared_case(measured)
    ticket = _validate(tx, prepared, counts, context) if validated else None
    with pytest.raises(commit.PreparedCommitError):
        _commit(tx, counts, context)
    with pytest.raises(commit.PreparedCommitError):
        _prepare(tx, counts, context)
    with pytest.raises(commit.PreparedCommitError):
        tx.discard("bypass outstanding prepare")
    if validated:
        with pytest.raises(commit.PreparedCommitError):
            _validate(tx, prepared, counts, context)
    receipt = tx.abort_prepared(prepared, "owner cancelled before swap")
    assert receipt.candidate_id == tx.candidate.candidate_id
    assert tx.state is commit.ChainCommitState.PENDING
    with pytest.raises(commit.PreparedCommitError):
        tx.abort_prepared(prepared, "duplicate abort")
    with pytest.raises(commit.PreparedCommitError):
        _validate(tx, prepared, counts, context)
    replacement = _prepare(tx, counts, context)
    assert replacement == prepared and replacement is not prepared
    with pytest.raises(commit.PreparedCommitError):
        tx.finalize(ticket)
    with pytest.raises(commit.PreparedCommitError):
        _validate(tx, prepared, counts, context)
    new_ticket = _validate(tx, replacement, counts, context)
    with pytest.raises(commit.PreparedCommitError):
        tx.finalize(ticket)
    tx.finalize(new_ticket)


@pytest.mark.parametrize("operation", ["validate", "abort", "finalize"])
def test_identical_values_from_foreign_tx_or_frozen_copy_are_not_issued_objects(
    measured: dict[str, Any], operation: str,
) -> None:
    tx, counts, context, prepared = _prepared_case(measured)
    other, other_counts, other_context, foreign = _prepared_case(measured)
    assert prepared == foreign and prepared is not foreign
    if operation == "finalize":
        issued = _validate(tx, prepared, counts, context)
        alien = _validate(other, foreign, other_counts, other_context)
        assert issued == alien
        for fake in (alien, replace(issued), commit.ChainCommitApplyTicket(prepared)):
            with pytest.raises(commit.PreparedCommitError):
                tx.finalize(fake)
        tx.finalize(issued)
    else:
        for fake in (foreign, replace(prepared)):
            with pytest.raises(commit.PreparedCommitError):
                if operation == "validate":
                    _validate(tx, fake, counts, context)
                else:
                    tx.abort_prepared(fake, "foreign abort")
        tx.finalize(_validate(tx, prepared, counts, context))
    assert other.state is commit.ChainCommitState.PENDING


@pytest.mark.parametrize("field,value", [("side", "2P"), ("reset_epoch", EPOCH + 1),
    ("event_revision", EVENT_REVISION + 1), ("action_revision", ACTION_REVISION + 1),
    ("event_sha256", "0" * 64), ("current_frame_idx", FRAME + 9),
    ("current_time_sec", (FRAME + 9) / FPS)])
def test_prepare_binds_exact_generation_frame_and_time(
    measured: dict[str, Any], field: str, value: Any,
) -> None:
    tx, counts, context, prepared = _prepared_case(measured)
    original = counts.copy()
    with pytest.raises((commit.StaleCandidateError, commit.PreparedCommitError)):
        _validate(tx, prepared, counts, replace(context, **{field: value}))
    assert tx.state is commit.ChainCommitState.PENDING and counts == original
    with pytest.raises(commit.PreparedCommitError):
        tx.finalize(commit.ChainCommitApplyTicket(prepared))
    tx.abort_prepared(prepared, "stale before owner swap")


@pytest.mark.parametrize("fault", ["content", "revision_only", "aba"])
def test_counter_content_and_monotonic_revision_are_both_bound(
    measured: dict[str, Any], fault: str,
) -> None:
    tx, counts, context, prepared = _prepared_case(measured)
    revision = COUNTER_REVISION
    if fault in ("content", "aba"):
        counts[EXTRA_COLOR] += EXTRA_COUNT
    if fault == "aba":
        counts[EXTRA_COLOR] -= EXTRA_COUNT
    if fault != "content":
        revision += 1  # 同値へ戻った場合もownerの版は進める。
    original = counts.copy()
    with pytest.raises(commit.PreparedCommitError):
        _validate(tx, prepared, counts, context, revision)
    assert tx.state is commit.ChainCommitState.PENDING and counts == original
    tx.abort_prepared(prepared, "rebuild using actual current revision")
    new = _prepare(tx, counts, context, revision)
    tx.finalize(_validate(tx, new, counts, context, revision))
    expected = original.copy()
    expected.subtract(dict(tx.candidate.erased_color_count))
    assert new.effect.copy_counter() == expected


@pytest.mark.parametrize("revision", [True, -1, 1.5])
def test_invalid_counter_revision_rejected_at_prepare_and_validate(
    measured: dict[str, Any], revision: Any,
) -> None:
    tx, counts, context, prepared = _prepared_case(measured)
    with pytest.raises(commit.CandidateValidationError):
        _validate(tx, prepared, counts, context, revision)
    tx.abort_prepared(prepared, "restart after invalid revision")
    with pytest.raises(commit.CandidateValidationError):
        _prepare(tx, counts, context, revision)
    assert tx.state is commit.ChainCommitState.PENDING
    new = _prepare(tx, counts, context)
    tx.finalize(_validate(tx, new, counts, context))


@pytest.mark.parametrize("failure", ["policy", "effect", "prepared"])
def test_prepare_exception_does_not_consume_and_retries_same_transaction(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    tx, counts, context, previous = _prepared_case(measured)
    tx.abort_prepared(previous, "exercise same tx retry")
    original = counts.copy()
    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected prepare failure")
    with monkeypatch.context() as patch:
        if failure == "policy":
            patch.setattr(_TestOnlyPolicy, "require_release", fail)
        else:
            target = "ChainCommitEffect" if failure == "effect" else "PreparedChainCommit"
            patch.setattr(commit, target, fail)
        with pytest.raises(RuntimeError, match="injected prepare failure"):
            _prepare(tx, counts, context)
    assert tx.state is commit.ChainCommitState.PENDING and counts == original
    prepared = _prepare(tx, counts, context)
    tx.finalize(_validate(tx, prepared, counts, context))


def test_mutating_policy_is_detected_but_external_side_effect_is_not_rolled_back(
    measured: dict[str, Any],
) -> None:
    """任意callbackの部分書込をhelperが原子的に戻したという偽契約を作らない。"""
    tx, counts, context, previous = _prepared_case(measured)
    tx.abort_prepared(previous, "exercise dishonest policy")
    original = counts.copy()
    policy = _TestOnlyPolicy()
    def mutate(*args: Any) -> None:
        counts[EXTRA_COLOR] += EXTRA_COUNT
    policy.require_release = mutate
    with pytest.raises(commit.PreparedCommitError, match="prepare中"):
        tx.prepare(counts, context, policy, policy, counter_revision=COUNTER_REVISION)
    assert tx.state is commit.ChainCommitState.PENDING
    assert counts[EXTRA_COLOR] == original[EXTRA_COLOR] + EXTRA_COUNT
    prepared = _prepare(tx, counts, context, COUNTER_REVISION + 1)
    tx.finalize(_validate(tx, prepared, counts, context, COUNTER_REVISION + 1))


def _nested_operation(tx: Any, counts: Any, context: Any, operation: str) -> None:
    """検証callbackから全消費・中断入口への再入を試す。"""
    if operation == "commit":
        _commit(tx, counts, context)
    elif operation == "prepare":
        _prepare(tx, counts, context)
    elif operation == "discard":
        tx.discard("nested discard")
    elif operation == "abort":
        tx.abort_prepared(None, "nested abort")
    elif operation == "validate":
        _validate(tx, None, counts, context)
    else:
        tx.finalize(None)


@pytest.mark.parametrize("operation", ["commit", "prepare", "discard", "abort", "validate", "finalize"])
def test_prepare_reentry_is_rejected_before_any_nested_consumption(
    measured: dict[str, Any], operation: str,
) -> None:
    tx, counts, context, previous = _prepared_case(measured)
    tx.abort_prepared(previous, "restart for callback test")
    policy = _TestOnlyPolicy()
    def reenter(*args: Any) -> None:
        _nested_operation(tx, counts, context, operation)
    policy.require_verified = reenter
    with pytest.raises(commit.TransactionConsumedError):
        tx.prepare(counts, context, policy, policy, counter_revision=COUNTER_REVISION)
    assert tx.state is commit.ChainCommitState.PENDING
    prepared = _prepare(tx, counts, context)
    tx.finalize(_validate(tx, prepared, counts, context))


@pytest.mark.parametrize("failure", ["counter_read", "ticket", "reentry"])
def test_preflight_exception_keeps_prepared_and_does_not_issue_ack_ticket(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    tx, counts, context, prepared = _prepared_case(measured)
    def fail(*args: Any, **kwargs: Any) -> None:
        if failure == "reentry":
            tx.abort_prepared(prepared, "nested preflight abort")
        raise RuntimeError("injected preflight failure")
    with monkeypatch.context() as patch:
        target = "ChainCommitApplyTicket" if failure == "ticket" else "_counter_snapshot"
        patch.setattr(commit, target, fail)
        with pytest.raises((RuntimeError, commit.TransactionConsumedError)):
            _validate(tx, prepared, counts, context)
    assert tx.state is commit.ChainCommitState.PENDING
    with pytest.raises(commit.PreparedCommitError):
        tx.finalize(commit.ChainCommitApplyTicket(prepared))
    tx.finalize(_validate(tx, prepared, counts, context))


@pytest.mark.parametrize("validated", [False, True])
def test_owner_preapply_failure_aborts_without_partial_apply_or_new_transaction(
    measured: dict[str, Any], validated: bool,
) -> None:
    tx, counts, context, prepared = _prepared_case(measured)
    owner_state = (tx.candidate.before_grid, tuple(sorted(counts.items())))
    original = owner_state
    ticket = _validate(tx, prepared, counts, context) if validated else None
    with pytest.raises(RuntimeError, match="before swap"):
        try:
            raise RuntimeError("owner cancelled before swap")
        except RuntimeError:
            tx.abort_prepared(prepared, "owner failure before any external write")
            raise
    assert owner_state is original and tx.state is commit.ChainCommitState.PENDING
    with pytest.raises(commit.PreparedCommitError):
        tx.finalize(ticket)
    replacement = _prepare(tx, counts, context)
    ready = (replacement.effect.final_grid, replacement.effect.counter_after)
    ack = _validate(tx, replacement, counts, context)
    owner_state = ready
    tx.finalize(ack)
    assert owner_state is ready


def test_abort_exception_or_reentry_preserves_live_ticket_for_owner(
    measured: dict[str, Any],
) -> None:
    tx, counts, context, prepared = _prepared_case(measured)
    ticket = _validate(tx, prepared, counts, context)
    with pytest.raises(commit.CandidateValidationError):
        tx.abort_prepared(prepared, " ")
    class ReentrantReason(str):
        def strip(self, chars: str | None = None) -> str:
            tx.finalize(ticket)
            return "nested ack"
    with pytest.raises(commit.TransactionConsumedError):
        tx.abort_prepared(prepared, ReentrantReason("nested abort"))
    assert tx.state is commit.ChainCommitState.PENDING
    tx.finalize(ticket)
