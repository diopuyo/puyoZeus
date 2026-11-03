"""長時間計測で母数・傾き・境界対応を取り違えないための回帰。"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.analyze_live_b15 import regression, rss_boundary, match_boundaries
from scripts.live_b15_plan import select_games, START, END, DURATION
from scripts.measure_live_b15 import command, run


@pytest.mark.parametrize('slope', [-38.0, 0.0, 38.0])
def test_rss_linear_regression_uses_hours_and_decimal_mb(slope: float) -> None:
    rows = [dict(at=5000+i*300, rss_bytes=(2082+slope*i/12)*1_000_000) for i in range(13)]
    result = regression(rows)
    assert result['slope_mb_per_hour'] == pytest.approx(slope)
    assert result['elapsed_hours'] == 1
    assert result['delta_mb'] == pytest.approx(slope)


def test_rss_boundary_requires_both_sides() -> None:
    before = [dict(at=90, rss_bytes=2100_000_000)]
    assert rss_boundary(before, 100)['decreased'] is None
    result = rss_boundary(before+[dict(at=110, rss_bytes=2080_000_000)], 100)
    assert result['delta_mb'] == -20 and result['decreased']
    assert regression([])['slope_mb_per_hour'] is None


def test_boundary_matching_reports_missing_extra_and_shift() -> None:
    expected = [dict(t_sec=t) for t in (100, 150, 200)]
    actual = [dict(t_sec=t) for t in (105, 202, 240)]
    result = match_boundaries(expected, actual)
    assert result['matched'] == 2
    assert result['missing'] == [dict(t_sec=150)]
    assert result['extra'] == [dict(t_sec=240)]
    assert [row['delta_sec'] for row in result['pairs']] == [5, 2]


def test_boundary_duplicate_is_not_counted_as_another_match() -> None:
    result = match_boundaries([dict(t_sec=100)], [dict(t_sec=100), dict(t_sec=101)])
    assert result['matched'] == 1 and len(result['extra']) == 1


def test_selection_counts_game_in_progress_and_excludes_end() -> None:
    rows = [dict(t_sec=t, local_game=i) for i, t in enumerate((90, 150, 200))]
    result = select_games(rows, 100, 200)
    assert [row['local_game'] for row in result] == [0, 1]
    assert result[0]['original_start'] == 90 and result[0]['left_censored']
    assert len(select_games(rows, 150, 200)) == 1


def test_command_preserves_b14_cpu_and_c_policy(tmp_path: Path) -> None:
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(cpu_threads=1, cnn_device='cpu')))
    with patch('scripts.diagnose_live_b8.case_command', return_value=['python', '--config', str(config)]) as make:
        args = command(tmp_path)
    assert make.call_args.args[0] == ('realtime', True, START, END, 1, 10)
    assert END-START == DURATION == 3600
    saved = json.loads(config.read_text())
    assert saved['runtime_audit'] and saved['event_priority'] and saved['coalesce_features']
    assert not saved['adaptive_evaluation'] and not saved['cpu_isolation']
    assert args[-1] == '--realtime'


def test_existing_launch_is_never_repeated(tmp_path: Path) -> None:
    (tmp_path/'launch.json').write_text('{}')
    with pytest.raises(ValueError, match='再実行'):
        run(tmp_path, 'test')
