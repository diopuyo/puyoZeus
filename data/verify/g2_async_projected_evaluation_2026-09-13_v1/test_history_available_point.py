"""既存台帳に履歴の取得時点を渡す小CPU検査。元rowのclock書換えではない。"""
from dataclasses import replace
import pytest
from src import chain_prediction_ledger_v1 as L
from scripts.chain_end_epoch_shadow_v1 import _origin_prediction
from test_diagnose_video38_prediction_ledger_shadow_v1 import _board, _event, _result

RECORDED_FRAME, AVAILABLE_FRAME, FPS = 34702, 35160, 60


def test_history_prediction_available_only_at_receiving_frame() -> None:
    ledger, generation = L.ChainPredictionLedger(), L.ChainGeneration('2P', 2, 44)
    board = _board()
    original = _event(board)
    point = dict(frame_idx=AVAILABLE_FRAME, time_sec=AVAILABLE_FRAME / FPS)
    handle = ledger.open_landing_provisional(generation=generation, **point,
        origin_before_board=board, landing_event=original,
        capture_source='same_run_history_received_later_not_live_early_compute')
    first = ledger.add_prediction(handle, generation=generation, **point,
        episode_revision=1, input_board=board, result=_result(board))
    later = replace(_event(_board(True), 'formula_read'), trigger_sec=RECORDED_FRAME / FPS + 1)
    episode = ledger.add_episode(handle, generation=generation, **point, event=later,
        capture_source='same_run_history_received_later_not_live_early_compute')
    ledger.add_prediction(handle, generation=generation, **point,
        episode_revision=episode.episode_revision, input_board=later.before_board,
        result=_result(later.before_board))
    assert _origin_prediction(ledger.snapshot(handle)) is first
    assert first.available_at.frame_idx == AVAILABLE_FRAME > RECORDED_FRAME
    assert ledger.snapshot(handle).episodes[0].trigger_sec == original.trigger_sec
    with pytest.raises(L.ObservationOrderError):
        ledger.add_episode(handle, generation=generation, frame_idx=RECORDED_FRAME,
            time_sec=RECORDED_FRAME / FPS, event=later, capture_source='rewound_clock')
