"""E34bの固定範囲・同時記録・旧新盤面差の定義を確認する。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from scripts import collect_e34b as collect
from scripts import report_e34b as report
from src.board import Board
from src.board_state_machine import BoardState
from src.exchange_event_record import ExchangeEventRecorder, read_records
from src.exchange_prefire_origin import recorded_match_gate
from src.match_range_gate import MatchRangeGate
from src.recognition_pipeline import RecognitionPipeline, SideResult, PipelineResult


def test_recorded_ranges_do_not_depend_on_external_file(tmp_path: Path) -> None:
    header = dict(video_id='absent', prefire_match_ranges=dict(ranges=[[1., 4.]], source='frozen.tsv'))
    gate = recorded_match_gate(header, tmp_path)
    assert gate.allows(1.) and gate.allows(4.)
    assert not gate.allows(.9) and not gate.allows(4.1)


def test_empty_recorded_ranges_are_not_a_success(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='保存済み試合範囲が空'):
        recorded_match_gate(dict(prefire_match_ranges=dict(ranges=[], source='empty.tsv')), tmp_path)


def test_capture_marks_the_same_frame_and_freezes_ranges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config: dict = {}
    def load(**kwargs: Any) -> Any:
        config.update(kwargs)
        return NS()
    monkeypatch.setattr(RecognitionPipeline, 'load_default', load)
    monkeypatch.setattr(ExchangeEventRecorder, 'update', ExchangeEventRecorder.update)
    monkeypatch.setattr(ExchangeEventRecorder, 'write', ExchangeEventRecorder.write)
    boundary = tmp_path/'matches.tsv'
    boundary.write_text('idx\tstart_sec\tend_sec\n1\t1\t4\n')
    collect.install_capture(MatchRangeGate(((1., 4.),), str(boundary)))
    RecognitionPipeline.load_default(stable_frame_count=3)
    assert config == dict(stable_frame_count=3, enable_landing_chain_record_hold=True,
                          enable_chain_active_record_hold=True)
    board = Board()
    side = SideResult('1P', BoardState.STABLE, board, None, board, None, 0, 0, None,
                      landing_chain_started=True)
    result = PipelineResult(30, 1., True, side, replace(side, side='2P', landing_chain_started=False))
    snap = NS(net_balance_capped=0, forecast_p1=0, total_dropped_to_p1=0, total_dropped_to_p2=0)
    final = NS(finalized_count_p1=0, finalized_count_p2=0, chain_total_score_p1=0, chain_total_score_p2=0)
    path = tmp_path/'inputs.gz'
    recorder = ExchangeEventRecorder(path, 'sample', True, tmp_path)
    recorder.update(result, snap, final, 1., 0, (None, None), (0, 0), (False, False))
    recorder.close()
    rows = list(read_records(path))
    assert rows[0]['prefire_match_ranges']['ranges'] == ((1., 4.),)
    assert rows[1]['args'][0].p1.prefire_origin_hold is True
    assert rows[1]['args'][0].p2.prefire_origin_hold is False
    assert result.p1.prefire_origin_hold is None


def test_input_difference_keeps_missing_boards_in_denominator(monkeypatch: pytest.MonkeyPatch) -> None:
    empty, filled = Board(), Board()
    filled._grid[-1, 0] = 1
    def row(t: float, a: Board | None, b: Board | None) -> tuple:
        return (NS(p1=NS(confirmed_board=a), p2=NS(confirmed_board=b)), None, None, t)
    old = [row(0., None, empty), row(1., empty, empty), row(2., filled, empty)]
    new = [row(0., None, empty), row(1., empty, filled), row(2., None, filled)]
    monkeypatch.setattr(report, 'updates', lambda path: iter(old if 'e31' in str(path) else new))
    value = report.input_difference('sample')
    assert value['rows'] == 3 and value['board_mismatch_rows'] == 2
    assert value['side_mismatch_rows'] == [1, 2]
    assert value['first_mismatch_sec'] == 1.


def test_input_difference_does_not_shrink_to_common_times(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = iter([iter([(None, None, None, 0.)]), iter([(None, None, None, 1.)])])
    monkeypatch.setattr(report, 'updates', lambda path: next(calls))
    with pytest.raises(AssertionError):
        report.input_difference('sample')


def test_snapshot_capture_reuses_e32_window_once_per_fire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts import e34b_snapshot_capture as capture
    calls = []
    monkeypatch.setattr(capture.cv2, 'VideoCapture', lambda _: NS(get=lambda _: 30., release=lambda: None))
    def window(*args: Any) -> dict:
        calls.append(args[3:])
        return dict(board=[[0]], raw_frames=[dict(t_sec=1.)], reason=None)
    monkeypatch.setattr(capture, 'capture_window', window)
    monkeypatch.setattr(capture.PrefireSnapshotReader, 'update', capture.PrefireSnapshotReader.update)
    observer = capture.SnapshotCapture(tmp_path/'video.mp4', tmp_path/'windows.gz')
    observer.install()
    reader, quiet = NS(), NS(chain_event=None)
    update = capture.PrefireSnapshotReader.update
    assert update(reader, None, (quiet, quiet), 1., 0) == (None, None)
    fire = NS(chain_event=NS(trigger_sec=2.))
    a = update(reader, None, (fire, quiet), 2., 0)
    assert a == update(reader, None, (fire, quiet), 2.1, 0)
    assert calls == [(2., 30., 1.)] and 'raw_frames' not in a[0]
    observer.close()


def test_d1_missing_origin_is_not_residual_improvement() -> None:
    from scripts.e34b_residuals import paired_counts
    base = dict(source='q', game=0, side='1P', trigger=1., cells=[dict(category='経路未確定：欠落')])
    other = dict(base, trigger=2.)
    missing = dict(base, cells=[], unmeasurable='missing_origin')
    paired = paired_counts(dict(off=[base, other], on=[missing, dict(other, cells=[])]))
    assert paired['fires'] == 1 and paired['total_fires'] == 2
    assert paired['residual_cells'] == dict(off=1, on=0)


def test_score_cohort_is_never_rebuilt_after_freezing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(report, 'OUT', tmp_path)
    (tmp_path/'COHORT.json').write_text('{"frozen": true}', encoding='utf-8')
    monkeypatch.setattr(report.score_trace, 'prepare_cohort', lambda: pytest.fail('固定済み行の再選択'))
    assert report.freeze_score_cohort() == dict(frozen=True)


def test_d1_changes_match_original_strictly_prefire_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import gzip
    from scripts import e34b_residuals as residuals
    monkeypatch.setattr(residuals, 'OUT', tmp_path)
    (tmp_path/'records').mkdir()
    with gzip.open(tmp_path/'records/sample.windows.json.gz', 'wt') as stream:
        stream.write('{"0:0:4.0": {}}')
    before, current = Board(), Board()
    before._grid[-1, 0], current._grid[-1, 0] = 1, 2
    rows = []
    for stamp, board in ((1., before), (2., None), (3., before), (4., current)):
        first = NS(confirmed_board=board, chain_event=NS(trigger_sec=4.) if stamp == 4 else None)
        second = NS(confirmed_board=None, chain_event=None)
        rows.append(dict(kind='update', args=(NS(p1=first, p2=second), None, None, stamp, 0)))
    monkeypatch.setattr(residuals, 'read_records', lambda _: iter(rows))
    result = residuals.observations('sample')['0:0:4.0']['changes']
    assert result == [before._grid.tolist(), before._grid.tolist()]


def test_score_failure_preserves_denominator_and_e33b_source_contract() -> None:
    import numpy as np
    from scripts.e34b_score_audit import correspondence, measured_summary
    target = dict(t_sec=1., game=0, side='1P', chain_id=2)
    row = dict(t_sec=1., game=0, source='unavoidable_death', value_sec=1.,
               scores=[dict(side='1P', chain_id=2, score=100.)])
    assert correspondence(target, row) == (None, 'nonprediction_display')
    assert correspondence(target, dict(row, source='S3_landing')) == (100., None)
    summary = measured_summary([100., np.nan], [100., 200.])
    assert summary['n'] == 2 and summary['measured_n'] == 1 and summary['unmeasurable_n'] == 1
    assert summary['mean'] is None and summary['complete'] is False
