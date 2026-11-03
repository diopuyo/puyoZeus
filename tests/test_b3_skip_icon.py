"""配布eの予告省略・欠測・spawnへ渡す設定の最小回帰試験。"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src import production_config as config
from src.b3_column_spec import ICON_TYPES, Mask
from src.b3_features import icon_group, mode_group
from src.b3_state import SideState
from src.phase_j.live_speed import configure, observation_options, validate

FLAGS = ('b3_ledger_mode', 'b3_ledger_landing', 'b3_skip_icon_reading')


@pytest.mark.parametrize('parallel', [False, True])
def test_skip_never_constructs_or_calls_warning(monkeypatch: pytest.MonkeyPatch,
                                               parallel: bool) -> None:
    from src import b3_observation as observation
    from src import parallel_ojama_warning

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError('予告検出器を呼び出した')

    @dataclass
    class Snapshot:
        value: int = 0

    monkeypatch.setattr(observation, 'OjamaWarningDetector', forbidden)
    monkeypatch.setattr(parallel_ojama_warning, 'ParallelOjamaWarningDetector', forbidden)
    capture = observation.ObservationCapture(enable_warning_parallel=parallel, skip_icon_reading=True)
    monkeypatch.setattr(capture.ocr, 'read', lambda image: SimpleNamespace(score_1p=0, score_2p=0))
    monkeypatch.setattr(observation, '_drive_ojama_accounting_lean', lambda *args: Snapshot())
    monkeypatch.setattr(capture, 'side', lambda side, *args: dict(side=side.side))
    monkeypatch.setattr(capture.observer, 'update', lambda row: ())
    side = lambda name: SimpleNamespace(side=name, state=observation.BoardState.STABLE)
    result = SimpleNamespace(p1=side('1P'), p2=side('2P'), is_match_active=True, match_end_locked=False)
    row, events = capture.update(result, None, 0, 0.0, (0, 0))
    assert [s['icon'] for s in row['sides']] == [None, None]
    assert capture.warning is None and events == ()
    capture.close()
    capture.close()


def test_missing_icons_use_training_nan_path() -> None:
    gs = SimpleNamespace(sides=[SideState('1P'), SideState('2P')])
    for st in gs.sides:
        for t in range(10):
            st.icon.push(float(t), None, True)
    row: dict[str, Any] = {}
    for i, side in enumerate(('p1', 'p2')):
        assert np.isnan(icon_group(row, gs, i, side, 70, float('nan')))
        for stem in (*('icon_corr_'+kind for kind in ICON_TYPES), 'icon_corr_total',
                     'icon_corr_nf', 'icon_ub', 'icon_big_clipped', 'icon_omitted', 'av_icon'):
            assert np.isnan(row[f'{stem}_{side}'])
        assert row[f'mk_icon_{side}'] == Mask.NOT_YET_AVAILABLE
        # 旧学習と同じ。生の種類別列は参考列であり、補正特徴へ0を流さない。
        assert np.isnan(row[f'icon_raw_total_{side}'])


@pytest.mark.parametrize('key', FLAGS)
@pytest.mark.parametrize('bad', [1, 'true', None])
def test_reject_non_boolean_flags(key: str, bad: Any) -> None:
    with pytest.raises(ValueError, match=key):
        validate({key: bad})


def test_live_flags_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    for key in FLAGS:
        monkeypatch.delenv('PUYO_'+key.upper(), raising=False)
    parser = argparse.ArgumentParser()
    configure(argparse.Namespace(source='dshow', cpu_threads=1, **dict.fromkeys(FLAGS, True)), parser)
    assert observation_options() == {'skip_icon_reading': True}
    assert config.b3_ledger_mode_enabled() and config.b3_ledger_landing_enabled()
    gs = SimpleNamespace(sides=[SideState('1P'), SideState('2P')], sl_snap={'pending_p1': 30})
    row: dict[str, Any] = {}
    mode_group(row, gs, 1.0, [np.nan, np.nan])
    assert row['mode_exchange'] == 1 and row['mode_bits'] == 8
    configure(argparse.Namespace(source='dshow', cpu_threads=1), parser)
    assert observation_options() == {}
    assert not config.b3_ledger_mode_enabled() and not config.b3_ledger_landing_enabled()
    assert not config.B3_SKIP_ICON_READING_DEFAULT
    assert not config.B3_ICON_LINE_ALIAS_DEFAULT


def test_launcher_delivers_new_flags(tmp_path: Any) -> None:
    from src.phase_j.launcher import parse_user_config, build_pipeline_config
    user = parse_user_config(dict(device_name='試験', **dict.fromkeys(FLAGS, True)), tmp_path)
    actual = build_pipeline_config(user)
    assert all(actual[key] is True for key in FLAGS)


def test_ledger_landing_environment_disables_icon_landing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_B3_LEDGER_LANDING', '1')
    side = SideState('1P')
    side.board = np.zeros((13, 6), dtype=np.int8)
    side.state = 'OJAMA_FALL'
    side._update_landing(0.0, True)
    assert side.fall_start is None


@pytest.mark.parametrize('probability', [0.1, 0.9])
def test_b3_changes_cannot_override_confirmed_death(probability: float) -> None:
    from src.b3_live_display import B3Display
    display = B3Display()
    display.boundary(1)
    display.prediction(probability, 1, 0.0)
    display.update(0.0, 'confirmed_death', 1.0)
    assert display.source == 'confirmed_death' and display.raw == 1.0
