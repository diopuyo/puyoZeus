"""発火前の最善手 Phase 4 (src/prefire_stable_queue.py・src/prefire_best_play_stable.py) の単体テスト (段1)。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src import prefire_best_play_stable as stable_module
from src.prefire_best_play_layer import NO_SIDE, PREFIRE_SOURCE, SideBest
from src.prefire_best_play_stable import DecisionHold, StableBestPlayLayer
from src.prefire_stable_queue import PROMOTION_SEC, STABLE_FRAMES, SideQueue

FRAME = 1 / 30
A, B = b'board-a', b'board-b'
P1, P2, P3, P4 = (1, 2), (3, 4), (2, 2), (4, 1)


def feed(queue: SideQueue, board: bytes, reading: tuple, start: float, frames: int) -> float:
    """同じ盤面・同じ読みを frames フレーム渡し、次の時刻を返す。"""
    for i in range(frames):
        queue.observe(start + i * FRAME, board, np.array(reading))
    return start + frames * FRAME


def test_reading_needs_consecutive_frames() -> None:
    queue = SideQueue()
    feed(queue, A, (*P1, *P2), 0.0, STABLE_FRAMES - 1)
    assert queue.reading() is None
    feed(queue, A, (*P1, *P2), 1.0, 1)
    assert tuple(queue.reading()) == (*P1, *P2)


def test_unread_sentinel_and_flicker_do_not_change_reading() -> None:
    queue = SideQueue()
    t = feed(queue, A, (*P1, *P2), 0.0, STABLE_FRAMES)
    t = feed(queue, A, (9, 9, *P2), t, STABLE_FRAMES + 2)      # 未読
    t = feed(queue, A, (*P3, *P2), t, 1)                         # 1フレームのちらつき
    feed(queue, A, (*P1, *P2), t, 1)
    assert tuple(queue.reading()) == (*P1, *P2)


def test_in_hand_comes_from_previous_interval_and_promotion() -> None:
    queue = SideQueue()
    t = feed(queue, A, (*P1, *P2), 0.0, STABLE_FRAMES)            # 盤面 A の区間 (最後の読みの next = P1)
    assert queue.known() == (0,) * 6                               # 1つ前の区間がないので手に持つ組は不明
    t = feed(queue, B, (*P1, *P2), t, STABLE_FRAMES)              # 盤面 B: 繰り上がり前 (next = 手に持つ組)
    assert queue.known() == (*P1, *P2, 0, 0)
    feed(queue, B, (*P2, *P3), t + PROMOTION_SEC, STABLE_FRAMES)  # 繰り上がり後
    assert queue.known() == (*P1, *P2, *P3)


def test_already_promoted_after_chain() -> None:
    queue = SideQueue()
    t = feed(queue, A, (*P1, *P2), 0.0, STABLE_FRAMES)
    feed(queue, B, (*P2, *P4), t, STABLE_FRAMES)                   # 先頭から next != 手に持つ組 → 繰り上がり済み
    assert queue.known() == (*P1, *P2, *P4)


def test_hold_delays_new_choice_then_shows_it() -> None:
    hold = DecisionHold(hold_sec=.3)
    assert hold.update(0.0, .5, (0, 1), .8) == (.5, True)         # まだ表示しない
    assert hold.update(0.2, .5, (0, 1), .8) == (.5, True)
    assert hold.update(0.31, .5, (0, 1), .8) == (.8, False)
    assert hold.update(0.4, .5, (0, 1), .82) == (.82, False)      # 同じ選択なら新しい値をすぐ出す


def test_hold_keeps_previous_correction_while_switching() -> None:
    hold = DecisionHold(hold_sec=.3)
    hold.update(0.0, .5, (0, 1), .8)
    hold.update(0.3, .5, (0, 1), .8)
    value, held = hold.update(0.4, .6, (NO_SIDE, 0), .6)          # 待つが最善に変わった直後
    expected = 1 / (1 + np.exp(-(np.log(.6 / .4) + np.log(.8 / .2))))
    assert held and value == pytest.approx(expected)
    assert hold.update(0.75, .6, (NO_SIDE, 0), .6) == (.6, False)


def test_short_flicker_never_reaches_display() -> None:
    hold = DecisionHold(hold_sec=.3)
    for i, decision in enumerate([(0, 1), (1, 1)] * 5):           # 1フレームごとに入れ替わる選択
        value, _ = hold.update(i * FRAME, .5, decision, .9 if decision[0] == 0 else .1)
        assert value == .5


def test_unavailable_result_keeps_state() -> None:
    hold = DecisionHold(hold_sec=0.0)
    hold.update(0.0, .5, (1, 2), .2)
    assert hold.update(0.1, .5, None, .5) == (pytest.approx(.2), True)


def test_lethal_flag_follows_displayed_choice() -> None:
    hold = DecisionHold(hold_sec=0.0)
    hold.update(0.0, .5, (0, 1), .98, lethal=True)
    assert hold.lethal
    hold.update(0.1, .5, (NO_SIDE, 0), .5)
    assert not hold.lethal


# ---- 予測層 ----

def history_item(t: float, grid: np.ndarray, queue: tuple) -> SimpleNamespace:
    return SimpleNamespace(t_sec=t, board=SimpleNamespace(_grid=grid), queue=np.array(queue))


def fake_overlay(source: str = 'G_fe', probability: float = .4) -> SimpleNamespace:
    empty, placed = np.zeros((13, 6), dtype=np.int8), np.zeros((13, 6), dtype=np.int8)
    placed[12, 0] = 1
    history = [history_item(i * FRAME, empty, (*P1, *P2)) for i in range(STABLE_FRAMES)]
    history += [history_item(1 + i * FRAME, placed, (*P2, *P3)) for i in range(STABLE_FRAMES)]
    tracker = SimpleNamespace(source=source, probability=probability, current=None, models=None)
    return SimpleNamespace(tracker=tracker, _start=0., _history=[list(history), list(history)],
                           _snapshots=[(1., None)], _game=0)


@pytest.fixture
def counted(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []
    monkeypatch.setattr(stable_module, 'build_context', lambda *a: SimpleNamespace(colors=()))
    monkeypatch.setattr(stable_module, 'side_best',
                        lambda ctx, a, ev: calls.append(a) or (SideBest(.95, 1, True, False) if a == 0 else None))
    return calls


def test_layer_waits_for_hold_then_jumps_and_restores(counted: list) -> None:
    overlay, layer = fake_overlay(), StableBestPlayLayer(hold_sec=.2)
    layer.apply(overlay, 2.0, 0)
    assert overlay.tracker.probability == .4 and overlay.tracker.source == 'G_fe'
    layer.apply(overlay, 2.25, 0)
    assert overlay.tracker.probability == .95 and overlay.tracker.source == PREFIRE_SOURCE
    layer.restore(overlay)
    assert overlay.tracker.probability == .4 and overlay.tracker.source == 'G_fe'
    assert len(counted) == 2   # 入力が変わらなければ再計算しない (片側ずつ1回)


def test_exchange_shows_evaluator_immediately_and_drops_hold(counted: list) -> None:
    overlay, layer = fake_overlay(), StableBestPlayLayer(hold_sec=0.0)
    layer.apply(overlay, 2.0, 0)
    layer.restore(overlay)
    overlay.tracker.source, overlay.tracker.probability = 'S3_provisional', .1
    layer.apply(overlay, 2.1, 0)
    assert overlay.tracker.probability == .1 and layer.hold.accepted == (NO_SIDE, 0)


def test_ignores_raw_queue_flicker_for_cache(counted: list) -> None:
    overlay, layer = fake_overlay(), StableBestPlayLayer(hold_sec=0.0)
    layer.apply(overlay, 2.0, 0)
    grid = overlay._history[0][-1].board._grid
    overlay._history[0].append(history_item(2.05, grid, (9, 9, *P3)))   # 未読の1フレーム
    layer.apply(overlay, 2.1, 0)
    assert len(counted) == 2


def test_known_uses_stable_queue(counted: list) -> None:
    overlay, layer = fake_overlay(), StableBestPlayLayer(hold_sec=0.0)
    layer.apply(overlay, 2.0, 0)
    assert layer.queues.sides[0].known() == (*P1, *P2, *P3)


def test_no_choice_leaves_value_bit_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stable_module, 'build_context', lambda *a: SimpleNamespace(colors=()))
    monkeypatch.setattr(stable_module, 'side_best', lambda *a: None)
    overlay, layer = fake_overlay(probability=.4123456789), StableBestPlayLayer(hold_sec=0.0)
    layer.apply(overlay, 2.0, 0)
    assert overlay.tracker.probability == .4123456789 and overlay.tracker.source == 'G_fe'
