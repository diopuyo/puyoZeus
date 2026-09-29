"""R1の一回制限・更新経路・読取り故障の検証。"""
from __future__ import annotations
from types import SimpleNamespace
import inspect
import numpy as np
from src.board import Board
from src.board_state_machine import BoardState
from src.placement_signal_runtime import PlacementSignalRuntime
from src.recognition_pipeline import RecognitionPipeline
from src.piece_persistence_guard import PiecePersistenceGuard
from tests.test_placement_signal_reconcile import observation


def setup_runtime() -> tuple:
    runtime = PlacementSignalRuntime(None, np.zeros((2, 2), np.uint8))
    ctx = SimpleNamespace(confirmed_board=Board(), state=BoardState.STABLE)
    def commit(side: str, board: Board, context: SimpleNamespace) -> None:
        context.confirmed_board = board.copy()
    pipe = SimpleNamespace(_sm_1p=SimpleNamespace(context=ctx), _set_deferred_confirmed=commit)
    runtime.history[0].extend([observation(8), observation(9), observation(10)])
    return runtime, pipe


def test_first_signal_only_even_when_rejected() -> None:
    runtime, pipe = setup_runtime()
    runtime.enabled_signals = (*runtime.enabled_signals, 'ojama')
    pipe._sm_1p.context.state = BoardState.CHAIN
    runtime.apply(pipe, 0, 'formula', observation(10))
    pipe._sm_1p.context.state = BoardState.STABLE
    runtime.apply(pipe, 0, 'ojama', observation(10))
    assert [r['reason'] for r in runtime.audit] == ['nonstable', 'already_consumed']
    assert pipe._sm_1p.context.confirmed_board.get(12, 0) == 0


def test_next_starts_new_cycle_and_logs_old_value() -> None:
    runtime, pipe = setup_runtime()
    runtime.apply(pipe, 0, 'next', observation(10))
    assert runtime.audit[0]['before'][12][0] == 0
    assert pipe._prev_confirmed_1p.get(12, 0) == 1
    assert not runtime.consumed[0]


def test_missing_classifier_fails_closed() -> None:
    runtime, pipe = setup_runtime()
    runtime.observe(pipe, 11, 11/60, np.zeros((1080, 1920, 3), np.uint8))
    assert runtime.audit[-1]['reason'] == 'reader_error'
    assert runtime.consumed == [True, True]
    assert pipe._sm_1p.context.confirmed_board.get(12, 0) == 0


def test_reset_drops_evidence_keeps_audit() -> None:
    runtime, pipe = setup_runtime()
    runtime.apply(pipe, 0, 'formula', observation(10))
    runtime.reset()
    assert not runtime.history[0] and runtime.audit


def test_native_gap_cannot_trigger_next_or_ojama() -> None:
    runtime, _ = setup_runtime()
    runtime.gray = np.zeros((1080, 1920), np.uint8)
    obs = observation(12)
    obs.cnn[1, 0] = obs.hsv[1, 0] = 9
    assert runtime.signals(0, obs, runtime.gray, 0) == []


def test_erasing_same_color_group() -> None:
    runtime, _ = setup_runtime()
    runtime.history[0][-1].cnn[12] = runtime.history[0][-1].hsv[12] = 1
    obs = observation(11)
    obs.cnn[:] = obs.hsv[:] = 0
    assert runtime.mark_erasure(0, obs).erasing


def test_commit_discards_only_corrected_cell_history() -> None:
    runtime, pipe = setup_runtime()
    pipe._pending_landing_vote_1p = [dict(cells=[(12, 0, 2), (12, 1, 3)])]
    pipe._stable_cnn_history_1p = {(12, 0): [2], (12, 1): [3]}
    pipe._landing_color_watch_1p = [((12, 0), 2.), ((12, 1), 2.)]
    runtime.apply(pipe, 0, 'next', observation(10))
    assert pipe._pending_landing_vote_1p[0]['cells'] == [(12, 1, 3)]
    assert pipe._stable_cnn_history_1p == {(12, 1): [3]}
    assert pipe._landing_color_watch_1p == [((12, 1), 2.)]


def test_pipeline_default_off_has_no_reader_side_effect() -> None:
    for function in (RecognitionPipeline.__init__, RecognitionPipeline.load_default):
        assert inspect.signature(function).parameters['enable_placement_signal_reconcile'].default is False
        assert inspect.signature(function).parameters['enable_placement_signal_ojama'].default is False
    pipe = RecognitionPipeline.__new__(RecognitionPipeline)
    pipe._placement_reconcile = None
    pipe.observe_placement_frame(1, 0., None)


def test_pipeline_native_feed_is_not_duplicated() -> None:
    runtime, pipe = setup_runtime()
    runtime.last_frame = 10
    runtime.observe(pipe, 10, 10/60, None)
    assert not runtime.audit


def test_other_guards_preserve_hidden_and_unmodified_cells() -> None:
    runtime, pipe = setup_runtime()
    pipe._piece_persistence_1p = PiecePersistenceGuard({(0, 0): 5, (12, 1): 3}, True)
    frozen = Board()
    frozen.set(0, 0, 5)
    frozen.set(12, 1, 3)
    pipe._glow_guard_1p = SimpleNamespace(frozen_board=frozen)
    runtime.apply(pipe, 0, 'next', observation(10))
    assert pipe._piece_persistence_1p._protected == {(0, 0): 5, (12, 1): 3, (12, 0): 1}
    assert pipe._glow_guard_1p.frozen_board.get(0, 0) == 5
    assert pipe._glow_guard_1p.frozen_board.get(12, 1) == 3
    assert pipe._glow_guard_1p.frozen_board.get(12, 0) == 1
    runtime.commit(pipe, 0, Board(), [dict(row=12, col=0)])
    assert pipe._piece_persistence_1p._protected == {(0, 0): 5, (12, 1): 3}


def test_disabled_ojama_does_not_consume_next_formula() -> None:
    runtime, pipe = setup_runtime()
    runtime.apply(pipe, 0, 'ojama', observation(10))
    assert not runtime.consumed[0] and not runtime.audit
    assert pipe._sm_1p.context.confirmed_board.get(12, 0) == 0
    runtime.apply(pipe, 0, 'formula', observation(10))
    assert runtime.audit[-1]['reason'] == 'corrected'


def test_signal_selection_preserves_formula_on_ojama_frame() -> None:
    runtime, _ = setup_runtime()
    obs = observation(11)
    obs.cnn[1, 0] = obs.hsv[1, 0] = 9
    assert runtime.signals(0, obs, np.zeros((2, 2), np.uint8), 1.) == ['formula']
    runtime = PlacementSignalRuntime(None, np.zeros((2, 2), np.uint8), enable_ojama=True)
    runtime.history[0].append(observation(10))
    assert runtime.signals(0, obs, np.zeros((2, 2), np.uint8), 1.) == ['formula', 'ojama']
    runtime.reset()
    assert runtime.enabled_signals == ('next', 'formula', 'ojama')
