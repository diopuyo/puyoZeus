"""評価セット採点器 (scripts/eval_set_score_20261001) の単体テスト。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import eval_set_score_20261001 as scorer

FPS = 30


def write_display(path: Path, t: np.ndarray, p1: np.ndarray, adv: np.ndarray | None = None) -> Path:
    """採点器が読む3列だけの display.npz を作る。"""
    adv = (p1 - .5) * 200 if adv is None else adv
    np.savez(path, t_sec=t, display_p1=p1, display_adv=adv)
    return path


def write_labels(path: Path, games: list[dict]) -> Path:
    path.write_text(json.dumps(dict(games=games)), encoding='utf-8')
    return path


def frames(start: float, end: float) -> np.ndarray:
    return np.arange(round(start*FPS), round(end*FPS)) / FPS


def test_time_mean_matches_hand_calculation(tmp_path: Path) -> None:
    t = frames(0, 10)
    p = np.where(t < 5, .8, .4)
    labels = write_labels(tmp_path/'l.json', [dict(game=1, winner='1P', start=0., end=10.)])
    result = scorer.score([write_display(tmp_path/'a.npz', t, p)], labels)
    expected = (-np.log(.8) - np.log(.4)) / 2
    assert result['summary']['m_time'] == pytest.approx(expected)
    assert result['summary']['frames'] == len(t)


def test_checkpoints_take_last_frame_at_or_before_time(tmp_path: Path) -> None:
    t = frames(0, 8)
    p = np.clip(t / 10, .01, .99)
    labels = write_labels(tmp_path/'l.json', [dict(game=1, winner='2P', start=0., end=8.)])
    row = scorer.score([write_display(tmp_path/'a.npz', t, p)], labels)['rows'][0]
    assert row['m_cp25'] == pytest.approx(-np.log(1 - .2))
    assert row['m_cp50'] == pytest.approx(-np.log(1 - .4))
    assert row['m_cp75'] == pytest.approx(-np.log(1 - .6))


def test_games_are_equally_weighted_and_frame_pool_is_separate(tmp_path: Path) -> None:
    t = frames(0, 30)
    p = np.where(t < 10, .9, .5)
    games = [dict(game=1, winner='1P', start=0., end=10.), dict(game=2, winner='1P', start=10., end=30.)]
    summary = scorer.score([write_display(tmp_path/'a.npz', t, p)], write_labels(tmp_path/'l.json', games))['summary']
    assert summary['m_time'] == pytest.approx((-np.log(.9) - np.log(.5)) / 2)
    assert summary['m_frame'] == pytest.approx((-np.log(.9) * 300 - np.log(.5) * 600) / 900)


def test_agreement_uses_last_third_and_zero_is_miss(tmp_path: Path) -> None:
    t = frames(0, 9)
    adv = np.where(t < 7.5, 10., 0.)
    labels = write_labels(tmp_path/'l.json', [dict(game=1, winner='1P', start=0., end=9.)])
    summary = scorer.score([write_display(tmp_path/'a.npz', t, np.full(len(t), .6), adv)], labels)['summary']
    assert summary['agree_frames'] == 90
    assert summary['agree_hits'] == 45


def test_parts_are_concatenated_and_overlap_rejected(tmp_path: Path) -> None:
    labels = write_labels(tmp_path/'l.json', [dict(game=1, winner='1P', start=0., end=10.)])
    a = write_display(tmp_path/'a.npz', frames(0, 5), np.full(150, .7))
    b = write_display(tmp_path/'b.npz', frames(5, 10), np.full(150, .7))
    assert scorer.score([b, a], labels)['summary']['frames'] == 300
    c = write_display(tmp_path/'c.npz', frames(4, 10), np.full(180, .7))
    with pytest.raises(ValueError):
        scorer.score([a, c], labels)


def test_epsilon_clip_and_nonfinite_rejected(tmp_path: Path) -> None:
    t = frames(0, 1)
    labels = write_labels(tmp_path/'l.json', [dict(game=1, winner='2P', start=0., end=1.)])
    row = scorer.score([write_display(tmp_path/'a.npz', t, np.ones(len(t)))], labels)['rows'][0]
    assert row['m_time'] == pytest.approx(-np.log(scorer.PROBABILITY_EPSILON))
    bad = np.full(len(t), .5)
    bad[3] = np.nan
    with pytest.raises(ValueError):
        scorer.score([write_display(tmp_path/'b.npz', t, bad)], labels)


def test_empty_game_window_is_an_error(tmp_path: Path) -> None:
    labels = write_labels(tmp_path/'l.json', [dict(game=1, winner='1P', start=20., end=30.)])
    with pytest.raises(ValueError):
        scorer.score([write_display(tmp_path/'a.npz', frames(0, 5), np.full(150, .5))], labels)
