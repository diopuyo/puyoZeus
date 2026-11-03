"""row1 col2 幻ぷよ抑止 (src/row1_guard.py、既定OFF) の単体試験。"""
from __future__ import annotations

import os
from typing import Iterable

import pytest

from src.board import Board, COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN
from src.row1_guard import (
    EMPTY_EVIDENCE_SEC, SPAWN_HOLD_SEC, Row1Guard, floating_r1c2,
)

FRAME_SEC = 0.1


def board(cells: dict[tuple[int, int], int]) -> Board:
    b = Board()
    for (r, c), v in cells.items():
        b.set(r, c, v)
    return b


def run(guard: Row1Guard, seq: Iterable[tuple[Board, Board]], t0: float = 0.0) -> list[Board]:
    return [guard.step(b, raw, t0 + i * FRAME_SEC) for i, (b, raw) in enumerate(seq)]


STACK = {(2, 2): COLOR_BLUE, (3, 2): COLOR_BLUE}


def test_floating_r1_is_removed() -> None:
    """row2 が空の row1 は物理的にありえない。1フレーム目から外す。"""
    guard = Row1Guard()
    b = board({(1, 2): COLOR_RED})
    out = guard.step(b, b, 0.0)
    assert out.get(1, 2) == COLOR_EMPTY and b.get(1, 2) == COLOR_RED  # 元は不変
    assert floating_r1c2(b) and guard.counts["floating"] == 1


def test_real_stack_below_is_kept() -> None:
    """下が埋まった静止ぷよ (履歴上すでに確定) は残す。"""
    guard = Row1Guard()
    b = board({**STACK, (1, 2): COLOR_RED})
    outs = run(guard, [(b, b)] * 10)
    assert all(o.get(1, 2) == COLOR_RED for o in outs)


def test_spawn_shape_is_held_then_real_placement_accepted() -> None:
    """時間経過では戻さず、NEXT移動の同じコマで公開する。隠し段は変更しない。"""
    guard = Row1Guard()
    empty = board(STACK)
    spawn = board({**STACK, (0, 2): COLOR_GREEN, (1, 2): COLOR_RED})
    n = int(SPAWN_HOLD_SEC / FRAME_SEC) + 3
    outs = run(guard, [(empty, empty)] + [(spawn, spawn)] * n)
    held = [o for o in outs[1:] if o.get(1, 2) == COLOR_EMPTY]
    assert len(held) == n and all(o.get(0, 2) == COLOR_GREEN for o in held)
    assert guard.step(spawn, spawn, 10.0, next_moved=True) is spawn
    assert guard.counts["spawn_held"] == len(held)
    assert guard.counts["spawn_released"] == 1


def test_spawn_shape_horizontal_rotation_is_held() -> None:
    """r1 と隣列 r1 の同時出現 (回転中) も保留対象。隣列自体は触らない。"""
    guard = Row1Guard()
    empty = board(STACK)
    rot = board({**STACK, (1, 2): COLOR_RED, (1, 3): COLOR_GREEN})
    out = run(guard, [(empty, empty), (rot, rot)])[1]
    assert out.get(1, 2) == COLOR_EMPTY and out.get(1, 3) == COLOR_GREEN


def test_spawn_shape_with_moving_piece_never_confirmed() -> None:
    """ツモが動いて読取が空に戻れば確定しない (E1/E2 型の取り込み)。"""
    guard = Row1Guard()
    empty = board(STACK)
    spawn = board({**STACK, (1, 2): COLOR_RED, (0, 2): COLOR_GREEN})
    seq = [(empty, empty), (spawn, spawn), (spawn, spawn)] + [(spawn, empty)] * 15
    outs = run(guard, seq)
    assert all(o.get(1, 2) == COLOR_EMPTY for o in outs[1:])


def test_single_new_cell_without_companion_is_not_held() -> None:
    """同時に増えたのが row1 の 1 セルだけなら保留しない (本物の置き・影響最小)。"""
    guard = Row1Guard()
    empty = board(STACK)
    one = board({**STACK, (1, 2): COLOR_RED})
    outs = run(guard, [(empty, empty), (one, one)])
    assert outs[1].get(1, 2) == COLOR_RED


def test_x_mark_screen_empty_removes_phantom_after_evidence_window() -> None:
    """確定盤に居座る幻でも、読取が EMPTY_EVIDENCE_SEC 続けて空なら外す (×マスク後の読取)。"""
    guard = Row1Guard()
    phantom = board({**STACK, (1, 2): COLOR_RED})
    seen_empty = board(STACK)
    n = int(EMPTY_EVIDENCE_SEC / FRAME_SEC) + 2
    outs = run(guard, [(phantom, phantom)] + [(phantom, seen_empty)] * n)
    assert outs[1].get(1, 2) == COLOR_RED          # 証拠が足りない間は残す
    assert outs[-1].get(1, 2) == COLOR_EMPTY
    assert guard.counts["screen_empty"] >= 1


def test_empty_flicker_resets_evidence() -> None:
    """読取が一瞬だけ空でも、途切れたら証拠は積み直し。本物は残る。"""
    guard = Row1Guard()
    real = board({**STACK, (1, 2): COLOR_RED})
    gap = board(STACK)
    seq = [(real, real)] + [(real, gap), (real, gap), (real, real)] * 6
    assert all(o.get(1, 2) == COLOR_RED for o in run(guard, seq))


def test_frozen_confirmed_during_chain_is_released_by_screen() -> None:
    """連鎖凍結中 (候補盤が動かない) でも画面が空なら幻を外す (延命の停止)。"""
    guard = Row1Guard()
    frozen = board({**STACK, (1, 2): COLOR_RED, (4, 0): COLOR_GREEN})
    screen = board({(4, 0): COLOR_GREEN})
    outs = run(guard, [(frozen, frozen)] + [(frozen, screen)] * 30)
    assert outs[-1].get(1, 2) == COLOR_EMPTY
    assert outs[-1].get(4, 0) == COLOR_GREEN        # 他セルは触らない


def test_real_death_is_reported_within_hold_window() -> None:
    """8コマ目の合図で即公開し、固定時間の追加待ちを入れない。"""
    guard = Row1Guard()
    before = board(STACK)
    death = board({**STACK, (1, 2): COLOR_RED, (0, 2): COLOR_RED})
    guard.step(before, before, 0.)
    for index in range(1, 8):
        assert guard.step(death, death, index / 30).get(1, 2) == COLOR_EMPTY
    assert guard.step(death, death, 8 / 30, next_moved=True) is death


def test_none_raw_reading_does_not_clear() -> None:
    """読取が無い (None) ときは画面証拠を使わない。"""
    guard = Row1Guard()
    b = board({**STACK, (1, 2): COLOR_RED})
    assert all(g.get(1, 2) == COLOR_RED for g in [guard.step(b, None, i * 0.1) for i in range(20)])


def test_reset_clears_time_state_but_keeps_counts() -> None:
    guard = Row1Guard()
    b = board({(1, 2): COLOR_RED})
    guard.step(b, b, 0.0)
    guard.reset()
    assert guard.counts["floating"] == 1 and guard._held_since is None


# --- 既定OFF / 配線 ---

def test_pipeline_default_off_is_passthrough() -> None:
    from src.recognition_pipeline import RecognitionPipeline
    pipe = RecognitionPipeline.__new__(RecognitionPipeline)
    pipe._enable_row1_col2_guard = False
    pipe._row1_guards = {}
    b = board({(1, 2): COLOR_RED})
    assert pipe._apply_row1_col2_guard('1P', b, b, 0.0) is b
    assert pipe.row1_guard_counts() == {}


def test_pipeline_on_uses_guard_and_none_resets() -> None:
    from src.recognition_pipeline import RecognitionPipeline
    pipe = RecognitionPipeline.__new__(RecognitionPipeline)
    pipe._enable_row1_col2_guard = True
    pipe._row1_guards = {}
    b = board({(1, 2): COLOR_RED})
    assert pipe._apply_row1_col2_guard('2P', b, b, 0.0).get(1, 2) == COLOR_EMPTY
    assert pipe._apply_row1_col2_guard('2P', None, b, 0.1) is None
    assert pipe.row1_guard_counts()['2P']['floating'] == 1


def test_constructor_and_load_default_default_off() -> None:
    import inspect
    from src.recognition_pipeline import RecognitionPipeline
    for fn in (RecognitionPipeline.__init__, RecognitionPipeline.load_default):
        assert inspect.signature(fn).parameters['enable_row1_col2_guard'].default is False


def test_live_flag_default_on_single_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """2026-10-15 user承認で本番ON。既定値は production_config が単一情報源。"""
    from src import production_config
    from src.phase_j import live_speed, launcher
    assert production_config.LIVE_ROW1_GUARD_DEFAULT is True
    assert production_config.LIVE_ROW1_GUARD_DELIVERY[0].flag == 'live_row1_guard'
    assert live_speed.validate({})['live_row1_guard'] is True
    assert launcher.UserConfig.live_row1_guard is True
    assert live_speed.validate({'live_row1_guard': False})['live_row1_guard'] is False
    monkeypatch.delenv('PUYO_LIVE_ROW1_GUARD', raising=False)
    # 別試験が設定した式枠の製品ONを、このrow1単独の配送検査へ持ち込まない。
    monkeypatch.delenv('PUYO_ALIGN_SCORE_ROI', raising=False)
    cfg = {'a': 1}
    assert live_speed.recognition_config(cfg, live_input=True) is cfg
    monkeypatch.setenv('PUYO_LIVE_ROW1_GUARD', '1')
    assert live_speed.recognition_config(cfg, live_input=True)['enable_row1_col2_guard'] is True


def test_chain_start_releases_hold_for_real_placement() -> None:
    """置き終わりの合図=連鎖開始。本物の置き (直後に連鎖) は保留せず即公開する。"""
    guard = Row1Guard()
    empty = board(STACK)
    placed = board({**STACK, (1, 2): COLOR_RED, (1, 3): COLOR_RED})
    first = guard.step(empty, empty, 0.0)
    held = guard.step(placed, placed, 0.1)
    assert first is empty and held.get(1, 2) == COLOR_EMPTY
    out = guard.step(placed, placed, 0.2, in_chain=True)
    assert out.get(1, 2) == COLOR_RED and guard.counts["spawn_released_chain"] == 1
    # 以後は再保留しない
    assert guard.step(placed, placed, 0.3, in_chain=True).get(1, 2) == COLOR_RED


def test_screen_cleared_cell_does_not_flap_on_single_misread() -> None:
    """×の読みぶれや時間経過では戻さず、本物の置きの合図で戻す。"""
    guard = Row1Guard()
    phantom = board({**STACK, (1, 2): COLOR_RED})
    gone = board(STACK)
    n = int(EMPTY_EVIDENCE_SEC / FRAME_SEC) + 2
    seq = [(phantom, phantom)] + [(phantom, gone)] * n + [(phantom, phantom)] + [(phantom, gone)] * 3
    outs = run(guard, seq)
    assert all(o.get(1, 2) == COLOR_EMPTY for o in outs[n + 1:])
    # 読取が長く一致しても合図がなければ戻さない。
    outs2 = run(guard, [(phantom, phantom)] * 10, t0=10.0)
    assert all(o.get(1, 2) == COLOR_EMPTY for o in outs2)
    assert guard.step(phantom, phantom, 12., next_moved=True) is phantom
    assert guard.counts["screen_restored"] == 1


@pytest.mark.parametrize('signal', ['in_chain', 'next_moved', 'ojama_falling'])
@pytest.mark.parametrize('wait_frames', [0, 1, 7, 30])
def test_each_signal_releases_without_extra_frame(signal: str, wait_frames: int) -> None:
    """同時出現・短い待ち・長い待ちのいずれも合図のコマで確定する。"""
    guard, empty = Row1Guard(), board(STACK)
    real = board({**STACK, (1, 2): COLOR_RED, (0, 2): COLOR_GREEN})
    guard.step(empty, empty, 0.)
    for index in range(wait_frames):
        assert guard.step(real, real, (index + 1) / 30).get(1, 2) == COLOR_EMPTY
    assert guard.step(real, real, (wait_frames + 1) / 30, **{signal: True}) is real


@pytest.mark.parametrize('signal', ['in_chain', 'next_moved', 'ojama_falling'])
@pytest.mark.parametrize('raw_kind', ['empty', 'missing', 'different'])
def test_signal_requires_remaining_matching_piece(signal: str, raw_kind: str) -> None:
    """古い内部盤だけを根拠に、消えた組・読取欠落・色違いを公開しない。"""
    guard, empty = Row1Guard(), board(STACK)
    real = board({**STACK, (1, 2): COLOR_RED, (0, 2): COLOR_GREEN})
    raw = {'empty': empty, 'missing': None,
           'different': board({**STACK, (1, 2): COLOR_BLUE})}[raw_kind]
    guard.step(empty, empty, 0.)
    guard.step(real, real, .1)
    assert guard.step(real, raw, .2, **{signal: True}).get(1, 2) == COLOR_EMPTY


@pytest.mark.parametrize('signal', ['in_chain', 'next_moved', 'ojama_falling'])
def test_old_signal_does_not_confirm_new_spawn(signal: str) -> None:
    """継続している古い合図は新しい組の置き終わりに流用しない。"""
    guard, empty = Row1Guard(), board(STACK)
    real = board({**STACK, (1, 2): COLOR_RED, (0, 2): COLOR_GREEN})
    guard.step(empty, empty, 0., **{signal: True})
    assert guard.step(real, real, .1, **{signal: True}).get(1, 2) == COLOR_EMPTY


def test_disappearance_is_immediate_and_timer_never_restores() -> None:
    guard, empty = Row1Guard(spawn_hold_sec=0.), board(STACK)
    phantom = board({**STACK, (1, 2): COLOR_RED, (0, 2): COLOR_GREEN})
    guard.step(empty, empty, 0.)
    guard.step(phantom, phantom, .1)
    assert guard.step(phantom, empty, .2).get(1, 2) == COLOR_EMPTY
    assert guard.counts['spawn_removed'] == 1
    assert guard.step(phantom, phantom, 100.).get(1, 2) == COLOR_EMPTY


@pytest.mark.parametrize('signal', ['in_chain', 'next_moved', 'ojama_falling'])
def test_screen_suppression_restores_on_each_signal(signal: str) -> None:
    """画面空で外したセルの復帰にも、固定時間の追加待ちを入れない。"""
    guard, empty = Row1Guard(), board(STACK)
    real = board({**STACK, (1, 2): COLOR_RED})
    guard.step(real, empty, 0.)
    assert guard.step(real, empty, EMPTY_EVIDENCE_SEC).get(1, 2) == COLOR_EMPTY
    assert guard.step(real, real, EMPTY_EVIDENCE_SEC + FRAME_SEC, **{signal: True}) is real


@pytest.mark.parametrize('branch', ['spawn', 'floating', 'screen'])
def test_only_target_cell_changes_and_input_is_immutable(branch: str) -> None:
    """公開盤のコピーだけを変更し、内部盤として共有する入力と対象外セルを保つ。"""
    import numpy as np
    guard, empty = Row1Guard(), board(STACK)
    candidate = board({**STACK, (0, 2): COLOR_GREEN, (1, 2): COLOR_RED, (1, 3): COLOR_BLUE})
    if branch == 'floating':
        candidate.set(2, 2, COLOR_EMPTY)
    guard.step(empty if branch == 'spawn' else candidate, empty, 0.)
    original = candidate._grid.copy()
    raw = empty if branch == 'screen' else candidate
    output = guard.step(candidate, raw, EMPTY_EVIDENCE_SEC)
    assert np.array_equal(candidate._grid, original)
    assert np.argwhere(output._grid != original).tolist() == [[1, 2]]
