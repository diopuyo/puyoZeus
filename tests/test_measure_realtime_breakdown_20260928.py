"""段別集計の二重計上・間引き・本体への計装互換を確認する。"""
from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.measure_realtime_breakdown_20260928 import (
    FrameMeter, PATH_FLAGS, STAGES, build_command, instrument_generate, summarize,
)


def test_nested_measurements_are_exclusive() -> None:
    """親10msのうち子4msは親へ再加算しない。"""
    meter = FrameMeter()
    timestamps = [0.0, 0.001, 0.003, 0.007, 0.011, 0.012]
    with patch('scripts.measure_realtime_breakdown_20260928.perf_counter', side_effect=timestamps):
        with meter.frame(0, 0.0):
            with meter.stage('exchange_other'):
                with meter.stage('features'):
                    pass
    assert meter.rows[0]['exchange_other'] == pytest.approx(6.0)
    assert meter.rows[0]['features'] == pytest.approx(4.0)
    assert meter.rows[0]['total_ms'] == pytest.approx(12.0)


def test_skipped_decode_is_counted_and_warmup_excluded() -> None:
    """処理fpsの分母に間引きデコードを含め、warmupを除く。"""
    meter = FrameMeter()
    for index in range(6):
        row = dict(frame=index, t_sec=float(index), processed=index % 2 == 0,
                   total_ms=10.0, **{name: 0.0 for name in STAGES})
        row['decode_1080p'] = 2.0
        meter.rows.append(row)
    result = summarize(meter, Namespace(start_sec=2.0, end_sec=6.0))
    assert result['frames'] == 2
    assert result['decoded_frames'] == 4
    assert result['processing_fps'] == pytest.approx(50.0)
    assert result['stages_ms']['decode_1080p']['mean'] == 4.0
    assert result['stages_ms']['features']['P99'] == 0.0


def test_existing_generate_boundaries_and_signature() -> None:
    """本体変更時に計装対象の取り違えを動画実行前に検出する。"""
    import inspect
    import scripts.visualize_advantage_overlay as overlay
    generated = instrument_generate(overlay, FrameMeter())
    assert inspect.signature(generated) == inspect.signature(overlay.generate)


def test_no_processed_frames_fails() -> None:
    """空区間を速度ゼロの成功結果にしない。"""
    with pytest.raises(ValueError, match='処理フレーム'):
        summarize(FrameMeter(), Namespace(start_sec=0.0, end_sec=1.0))


@pytest.mark.parametrize('no_render', [False, True])
def test_command_preserves_evaluation_flags(tmp_path: Path, no_render: bool) -> None:
    """評価設定を保ち、記録先を別runへ閉じ込める。"""
    command = ['python', '-m', 'worker', '--worker', '--exchange-event-update']
    for flag in (*PATH_FLAGS, '--video', '--start-sec', '--end-sec', '--warmup-sec',
                 '--exchange-event-model-dir'):
        command.extend([flag, 'original'])
    status = tmp_path / 'status.json'
    status.write_text(json.dumps(dict(command=command)), encoding='utf-8')
    options = Namespace(status=status, output=tmp_path, video=Path('video.mp4'),
                        start_sec=2600, end_sec=2610, warmup_sec=1, no_render=no_render)
    result = build_command(options)
    assert '--exchange-event-update' in result
    assert ('--no-render' in result) == no_render
    assert result[result.index('--exchange-event-model-dir') + 1] == 'models/exchange_event_v2'
    for flag, name in PATH_FLAGS.items():
        assert result[result.index(flag) + 1] == str(tmp_path / name)
