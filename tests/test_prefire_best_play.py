"""発火前の最善手 (Phase 3) の単体テスト (段1)。答えが分かる組み立て盤面で探索・合成・予測層を確かめる。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src import prefire_best_play as search
from src import prefire_best_play_layer as layer_module
from src import prefire_exchange_sim as sim
from src.prefire_best_play_layer import BestPlayPrefireLayer, SideBest, combine

ROWS, COLS = 13, 6
RED, BLUE, GREEN, YELLOW, OJAMA = 1, 2, 3, 4, 9
FOUR = (RED, BLUE, GREEN, YELLOW)
ELAPSED = 10.0


def two_chain_grid() -> np.ndarray:
    """青を col1 の上に置けば 青4→赤5 の2連鎖になる盤面 (それ以外の組では撃てない)。"""
    grid = np.zeros((ROWS, COLS), dtype=np.int8)
    grid[12, 0], grid[9:12, 0], grid[7:9, 0] = GREEN, BLUE, RED
    grid[10:13, 1] = RED
    return grid


def buried_grid() -> np.ndarray:
    """おじゃまで 2段目まで埋まった盤面 (1段降れば窒息、色ぷよがなく返せない)。"""
    grid = np.zeros((ROWS, COLS), dtype=np.int8)
    grid[2:, :] = OJAMA
    return grid


def raw(grid: np.ndarray) -> bytes:
    return grid.astype(np.int8).tobytes()


def test_fire_now_found_with_pair_in_hand() -> None:
    lines, forced = search.fire_lines(raw(two_chain_grid()), (BLUE, YELLOW, GREEN, GREEN, GREEN, GREEN))
    assert [line.hand for line in lines][:1] == [1] and lines[0].chain_count == 2 and not forced


def test_no_fire_without_matching_colors() -> None:
    lines, _ = search.fire_lines(raw(two_chain_grid()), (YELLOW, YELLOW, GREEN, YELLOW, YELLOW, GREEN))
    assert all(line.chain_count < 2 for line in lines)


def test_extend_then_fire_with_next2_only() -> None:
    lines, _ = search.fire_lines(raw(two_chain_grid()), (YELLOW, YELLOW, YELLOW, GREEN, BLUE, BLUE))
    assert any(line.hand == 3 and line.chain_count == 2 for line in lines)


def test_unstable_board_has_no_options() -> None:
    grid = two_chain_grid()
    grid[9:12, 1] = 0
    grid[12, 2], grid[11, 2] = RED, RED   # 置く前から赤4が繋がる (誤読・連鎖途中) 盤面
    grid[12, 3] = RED
    assert search.fire_lines(raw(grid), (BLUE, BLUE, BLUE, BLUE, BLUE, BLUE)) == ((), False)


def test_unread_queue_has_no_options() -> None:
    assert search.fire_lines(raw(two_chain_grid()), (9, 9, BLUE, BLUE, 0, 0)) == ((), False)


def test_counter_is_exact_within_known_hands() -> None:
    known = (BLUE, YELLOW, GREEN, GREEN, GREEN, GREEN)
    values = search.counter_quantiles(raw(two_chain_grid()), known, 2, FOUR, ELAPSED)
    assert len(values) == 1 and values[0] >= sim.send_ojama(search.fire_lines(raw(two_chain_grid()), known)[0][0].score,
                                                            ELAPSED) - 1


def test_counter_beyond_known_uses_three_quantiles() -> None:
    values = search.counter_quantiles(raw(two_chain_grid()), (YELLOW, YELLOW, GREEN, GREEN, GREEN, GREEN), 6, FOUR, ELAPSED)
    assert len(values) == 3 and values[0] <= values[2]   # (p25, 平均, p75)。平均は偏りで p25 を下回りうる


def test_lethal_proof_on_buried_receiver() -> None:
    # E35 は上限による保守的な証明 (非連鎖手にも即時相殺を無償で与える)。証明できる量だけ死亡とする
    assert search.lethal(raw(buried_grid()), 7, 1, FOUR, ELAPSED)
    assert search.lethal(raw(buried_grid()), 60, 2, FOUR, ELAPSED)
    assert not search.lethal(raw(buried_grid()), 7, 2, FOUR, ELAPSED)   # 証明できない (楽観的な生存枝が残る)


def test_lethal_needs_four_known_colors_and_incoming() -> None:
    assert not search.lethal(raw(buried_grid()), 7, 1, (RED, BLUE, GREEN), ELAPSED)
    assert not search.lethal(raw(buried_grid()), 0, 1, FOUR, ELAPSED)
    assert not search.lethal(raw(np.zeros((ROWS, COLS), dtype=np.int8)), 7, 1, FOUR, ELAPSED)


# ---- 二人の合成 ----

def test_combine_waits_when_no_side_improves() -> None:
    assert combine(.4, (None, None)) == (.4, -1)
    assert combine(.4, (SideBest(.3, 1, False, False), SideBest(.6, 1, False, False))) == (.4, -1)


def test_combine_takes_only_improving_side() -> None:
    assert combine(.4, (SideBest(.9, 2, False, False), SideBest(.5, 1, False, False))) == (.9, 0)
    assert combine(.4, (None, SideBest(.1, 1, False, False))) == (.1, 1)


def test_combine_earlier_hand_moves_first_and_tie_is_logit_mean() -> None:
    assert combine(.5, (SideBest(.9, 2, False, False), SideBest(.2, 1, False, False))) == (.2, 1)
    value, chosen = combine(.5, (SideBest(.9, 1, False, False), SideBest(.1, 1, False, False)))
    assert chosen == 2 and abs(value - .5) < 1e-9


def test_combine_forced_side_cannot_wait() -> None:
    assert combine(.6, (SideBest(.3, 1, False, True), None)) == (.3, 0)


# ---- 1つの発火の値 (受け側の最善応手) ----

@pytest.fixture
def context() -> layer_module.Context:
    side = SimpleNamespace(board=SimpleNamespace(_grid=buried_grid()), queue=np.array([1, 2, 3, 4]))
    return layer_module.Context(None, (side, side), ((1, 2, 3, 4, 1, 2),) * 2, ELAPSED, FOUR, None, None, None)


LINE = search.FireLine(1, 2000, 3, raw(np.zeros((ROWS, COLS), dtype=np.int8)))


def test_full_counter_removes_the_jump(context: layer_module.Context, monkeypatch: pytest.MonkeyPatch) -> None:
    """受け側が送り量以上を返せるなら、受け側は返す方を選び、値は待つ値 (.5) のまま。"""
    sent = sim.send_ojama(LINE.score, ELAPSED)
    monkeypatch.setattr(search, 'lethal', lambda *a: False)
    monkeypatch.setattr(search, 'counter_quantiles', lambda *a: (sent,))
    monkeypatch.setattr(layer_module, 'exchange_value', lambda ctx, a, line, c, e: .5 if c >= sent else .9)
    value, certain = layer_module.line_value(context, 0, LINE)
    assert value == .5 and not certain
    assert combine(.5, (SideBest(value, 1, certain, False), None)) == (.5, -1)


def test_receiver_prefers_passive_when_counter_hurts(context, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search, 'lethal', lambda *a: False)
    monkeypatch.setattr(search, 'counter_quantiles', lambda *a: (1.0, 2.0, 3.0))
    monkeypatch.setattr(layer_module, 'exchange_value', lambda ctx, a, line, c, e: .6 if c == 0 else .8)
    assert layer_module.line_value(context, 0, LINE) == (.6, False)


def test_lethal_fire_gives_unavoidable_death_value(context, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search, 'lethal', lambda *a: True)
    assert layer_module.line_value(context, 0, LINE) == (.98, True)
    assert layer_module.line_value(context, 1, LINE) == (pytest.approx(.02), True)


# ---- 予測層 (状態を変えない・遅れ) ----

def fake_overlay(probability: float = .4, source: str = 'G_fe') -> SimpleNamespace:
    grid = two_chain_grid()
    before = SimpleNamespace(board=SimpleNamespace(_grid=np.zeros((ROWS, COLS), dtype=np.int8)),
                             queue=np.array([2, 4, 3, 3]), t_sec=.5)
    side = SimpleNamespace(board=SimpleNamespace(_grid=grid), queue=np.array([3, 3, 3, 3]), t_sec=1.)
    tracker = SimpleNamespace(source=source, probability=probability, current=None, models=None)
    return SimpleNamespace(tracker=tracker, _start=0., _history=[[before, side], [before, side]],
                           _snapshots=[(1., None)], _game=0)


@pytest.fixture
def fixed_sides(monkeypatch: pytest.MonkeyPatch):
    def install(first: SideBest | None, second: SideBest | None) -> None:
        monkeypatch.setattr(layer_module, 'build_context', lambda *a: None)
        monkeypatch.setattr(layer_module, 'side_best', lambda ctx, a, *rest: first if a == 0 else second)
    return install


def test_lethal_fire_now_jumps_and_restores(fixed_sides) -> None:
    fixed_sides(SideBest(.98, 1, True, False), None)
    overlay, layer = fake_overlay(), BestPlayPrefireLayer()
    layer.apply(overlay, 2., 0)
    assert overlay.tracker.probability == .98 and overlay.tracker.source == layer_module.PREFIRE_SOURCE
    layer.restore(overlay)
    assert overlay.tracker.probability == .4 and overlay.tracker.source == 'G_fe'


def test_no_viable_fire_leaves_value_bit_identical(fixed_sides) -> None:
    fixed_sides(None, None)
    overlay, layer = fake_overlay(.4123456789), BestPlayPrefireLayer()
    layer.apply(overlay, 2., 0)
    assert overlay.tracker.probability == .4123456789 and overlay.tracker.source == 'G_fe'   # 由来も書き換えない


def test_latency_withholds_result_until_ready(fixed_sides) -> None:
    fixed_sides(SideBest(.9, 1, False, False), None)
    overlay, layer = fake_overlay(), BestPlayPrefireLayer(latency_sec=.2)
    layer.apply(overlay, 2., 0)
    assert overlay.tracker.probability == .4
    layer.restore(overlay)
    layer.apply(overlay, 2.25, 0)
    assert overlay.tracker.probability == .9


def test_layer_skips_during_exchange(fixed_sides) -> None:
    fixed_sides(SideBest(.9, 1, False, False), None)
    overlay, layer = fake_overlay(source='S3'), BestPlayPrefireLayer()
    layer.apply(overlay, 2., 0)
    assert overlay.tracker.probability == .4 and not layer.trace


def test_known_pairs_put_pair_in_hand_first() -> None:
    """記録の NEXT は1つ先: 手に持つ組 = 直前の別盤面の NEXT (Phase 2 と同じ)。"""
    overlay = fake_overlay()
    assert layer_module.known_pairs(overlay._history[0]) == (2, 4, 3, 3, 3, 3)


# ---- 評価の構成 (full / fast / s3) ----

def test_unknown_evaluator_is_rejected() -> None:
    with pytest.raises(ValueError):
        BestPlayPrefireLayer(evaluator='hazard')


def _install_lines(monkeypatch: pytest.MonkeyPatch, lines: tuple) -> None:
    monkeypatch.setattr(search, 'fire_lines', lambda raw, known: (lines, False))
    monkeypatch.setattr(search, 'lethal', lambda *a: False)
    monkeypatch.setattr(search, 'counter_quantiles', lambda *a: (0.0,))


def test_s3_mode_picks_attacker_best_line(context, monkeypatch: pytest.MonkeyPatch) -> None:
    small, big = (search.FireLine(1, 100, 1, LINE.final), search.FireLine(2, 4000, 4, LINE.final))
    _install_lines(monkeypatch, (small, big))
    monkeypatch.setattr(layer_module, 'exchange_s3', lambda ctx, a, line, c, e: .55 if line.score < 1000 else .8)
    best = layer_module.side_best(context, 0, 's3')
    assert (best.value, best.hand) == (.8, 2)
    best2 = layer_module.side_best(context, 1, 's3')   # 2P は 1P 勝率が小さい方を選ぶ
    assert (best2.value, best2.hand) == (.55, 1)


def test_fast_mode_composes_only_the_chosen_line(context, monkeypatch: pytest.MonkeyPatch) -> None:
    small, big = (search.FireLine(1, 100, 1, LINE.final), search.FireLine(2, 4000, 4, LINE.final))
    _install_lines(monkeypatch, (small, big))
    monkeypatch.setattr(layer_module, 'exchange_s3', lambda ctx, a, line, c, e: .55 if line.score < 1000 else .8)
    calls = []
    monkeypatch.setattr(layer_module, 'exchange_inputs', lambda ctx, a, line, c, e: (None, None, None))
    monkeypatch.setattr(layer_module, 'landing_gfe', lambda ctx, b, i: calls.append(1) or .6)
    best = layer_module.side_best(context, 0, 'fast')
    assert len(calls) == 1 and best.value == pytest.approx(layer_module.logit_mean(.8, .6))


def test_gfe_mode_uses_only_landing_gfe(context, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_lines(monkeypatch, (LINE,))
    monkeypatch.setattr(layer_module, 'exchange_gfe', lambda ctx, a, line, c, e: .7)
    monkeypatch.setattr(layer_module, 'exchange_value', lambda *a: pytest.fail('S3 を使ってはいけない'))
    best = layer_module.side_best(context, 0, 'gfe')
    assert (best.value, best.hand, best.lethal) == (.7, 1, False)
