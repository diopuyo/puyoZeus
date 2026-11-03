"""コマ落ちに強い数え方・最初の型の段 1 試験 (DESIGN frame_drop_robust_20261023 §6 段 1)。

入れた箇所: R01・R04 (型 A 連続成立)、R02・R03 (型 C 窓と票)、B07 (型 H2 期限)。
見ること:
  1. 共通部品 (src/media_tick.py) の境界: m = max(2, ⌊N/4⌋)、成立条件、影、重みつき窓。
  2. 全コマの乱数列で、刻みモード ON と OFF の状態機械がビット単位で同じ (要件 E1)。
     各経路を実際に通った回数 (母数) と「刻みで早まった」件数 0 を確かめる (0 件を成功と読まない)。
  3. 間引き列で、成立が「全コマなら成立した刻み」より前にならず、観測 1 回では成立しない (E2・E3)。
  4. 配線: 既定 OFF、OFF では鍵も呼び出しも増えない、ON の値が届く (E4・E5)。
"""
from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import pytest

from src import media_tick as mt
from src.board import BOARD_COLS, BOARD_ROWS, Board
from src.board_state_machine import (
    BoardState, BoardStateMachine, DetectorSignals, StateContext,
    _vote_majority_board, _vote_majority_board_weighted,
)

# ---------------------------------------------------------------- 定数
SEED = 20261023
FULL_SEQUENCES = 250            # 全コマの乱数列の本数 (条件ごと)
FULL_LENGTH = 48                # 1 本あたりの観測数 (= 刻み数)
VIDEO_ORIGIN = 10500            # 動画のコマ番号の起点 (video_30 の区間の頭と同じ桁)
STABLE_N = 3                    # 配信の stable_frame_count
RECOVERY_N = 8                  # STABLE_RECOVERY_MIN_FRAMES
HISTORY_N = 5                   # DEFAULT_NON_STABLE_HISTORY_SIZE
EMPTY_TO_COLOR_VOTES = 3        # DEFAULT_EMPTY_TO_COLOR_MIN_VOTES
OBS21C_GAPS = (1, 2, 2, 2, 3, 3, 3, 4, 5, 6)   # 21c 実機の間隔の度数を粗くまねた列 (刻み)
# 配信 (L1_OFF の recognition_config) で状態機械に効く既定以外のフラグ。
LIVE_FLAGS = dict(enable_stable_recovery_gate=True, enable_initial_confirm_vote=True,
                  enable_recovery_counter_carryover=True, enable_cnn_flicker_hsv_fallback=True,
                  enable_match_start_full_clear=True, enable_gravity_filter_support=True,
                  merge_use_majority_value=True, enable_floating_gap_restore=True,
                  enable_ojama_column_stack_fix=True, enable_falling_puyo_integrity=True)
PLAIN_FLAGS = dict(enable_stable_recovery_gate=True)


# ---------------------------------------------------------------- 共通部品
@pytest.mark.parametrize('n, m', [(2, 2), (3, 2), (5, 2), (8, 2), (11, 2), (12, 3), (18, 4), (60, 15)])
def test_min_obs_values(n: int, m: int) -> None:
    assert mt.min_obs(n) == m


def test_tick_reached_boundary() -> None:
    # 回数 N で常に成立 (全コマの今の判定)
    assert mt.tick_reached(3, 3, 3)
    # 回数 2 (= m) と刻み 3 (= N) で成立、刻み 2 では不成立
    assert mt.tick_reached(2, 3, 3) and not mt.tick_reached(2, 2, 3)
    # 観測 1 回では刻みがいくら大きくても成立しない (E3)
    assert not mt.tick_reached(1, 100, 3)
    # N=2 は m=2=N なので回数どおり (刻みでは早まらない)
    assert not mt.tick_reached(1, 5, 2) and mt.tick_reached(2, 2, 2)
    # N=60 は m=15
    assert not mt.tick_reached(14, 60, 60) and mt.tick_reached(15, 60, 60)


def test_votes_reached_boundary() -> None:
    assert mt.votes_reached(3, 3, 3)                  # 全コマの今の判定
    assert mt.votes_reached(2, 4, 3)                  # 2 観測・重み 4
    assert not mt.votes_reached(2, 2, 3)              # 2 観測・重み 2
    assert not mt.votes_reached(1, 5, 3)              # 1 観測では通らない
    assert mt.votes_reached(1, 1, 1)                  # 裏付け 1 票は今のまま


def test_tick_stamp_and_clock() -> None:
    clock = mt.TickClock(2)
    stamps = [clock.advance(i) for i in (VIDEO_ORIGIN, VIDEO_ORIGIN+2, VIDEO_ORIGIN+6, VIDEO_ORIGIN+6)]
    assert [(s.now, s.prev_call) for s in stamps] == [(0, None), (1, 0), (3, 1), (3, 3)]
    assert [s.delta() for s in stamps] == [1, 1, 2, 1]   # 同じ刻みが 2 回来ても最低 1
    assert stamps[2].weight_after(1) == 2 and stamps[2].weight_after(0) == 1
    assert stamps[0].weight_after(None) == 1
    with pytest.raises(ValueError):
        mt.TickClock(0)


def test_tick_shadow_continuity_and_fallback() -> None:
    shadow = mt.TickShadow()
    assert shadow.observe(1, mt.TickStamp(0, None)) == 1
    assert shadow.observe(2, mt.TickStamp(2, 0)) == 3        # 前回も当たり → Δk=2
    assert shadow.observe(3, mt.TickStamp(5, 4)) == 4        # 前回 (4) は当たっていない → 1
    assert shadow.observe(1, mt.TickStamp(7, 5)) == 1        # 回数が戻った → 刻み = 回数
    assert shadow.observe(5, mt.TickStamp(8, 7)) == 5        # 飛び → 刻み = 回数 (安全側)
    assert shadow.observe(0, mt.TickStamp(9, 8)) == 0 and shadow.last_tick is None


def test_append_weighted_all_ones_equals_fifo() -> None:
    rng = random.Random(SEED)
    fifo: list[int] = []
    history: list[int] = []
    weights: list[int] = []
    for value in (rng.randrange(9) for _ in range(500)):
        fifo.append(value)
        if len(fifo) > HISTORY_N:
            fifo.pop(0)
        mt.append_weighted(history, weights, value, 1, HISTORY_N)
        assert history == fifo and weights == [1]*len(fifo)


def test_append_weighted_partial_trim() -> None:
    history: list[str] = []
    weights: list[int] = []
    for item, weight in (('a', 1), ('b', 2), ('c', 2), ('d', 2)):
        mt.append_weighted(history, weights, item, weight, HISTORY_N)
    assert history == ['b', 'c', 'd'] and weights == [1, 2, 2]   # 直近 5 刻み
    mt.append_weighted(history, weights, 'e', 9, HISTORY_N)       # 1 観測で窓を超える
    assert history == ['e'] and weights == [HISTORY_N]


# ---------------------------------------------------------------- 状態機械の台
class ScriptedDetector:
    """刻みごとに決めた遷移先を返す (コマ番号から刻みへ直して引く。間引きでも同じ台本)。"""

    def __init__(self, plan: dict[int, BoardState], step: int, origin: int) -> None:
        self.plan, self.step, self.origin = plan, step, origin

    def detect(self, ctx: StateContext, signals: DetectorSignals) -> BoardState | None:
        return self.plan.get((ctx.frame_idx-self.origin)//self.step)


def board_of(cells: dict[tuple[int, int], int]) -> Board:
    board = Board()
    for (r, c), v in cells.items():
        board.set(r, c, v)
    return board


def stacked_board(rng: random.Random, puyos: int) -> Board:
    """下から積んだ (重力に従う) 盤面。"""
    board = Board()
    for _ in range(puyos):
        col = rng.randrange(BOARD_COLS)
        for row in range(BOARD_ROWS-1, 0, -1):
            if board.get(row, col) == 0:
                board.set(row, col, rng.choice((1, 2, 3, 4)))
                break
    return board


def noisy(rng: random.Random, board: Board, rate: float) -> Board:
    out = board.copy()
    if rng.random() < rate:
        out.set(rng.randrange(1, BOARD_ROWS), rng.randrange(BOARD_COLS), rng.choice((0, 1, 2, 3, 4, 9)))
    return out


def random_script(rng: random.Random, length: int) -> tuple[list[dict[str, Any]], dict[int, BoardState]]:
    """1 本の観測列 (刻みごとの CNN・HSV・試合中か) と遷移の台本。"""
    truth = stacked_board(rng, rng.randrange(0, 20))
    sticky: tuple[tuple[int, int], int, int] | None = None   # (セル, 色, 残り) CNN と HSV が揃って続く誤り
    frames, plan = [], {}
    menu_until = rng.choice((0, 0, 3, 6))
    k = 0
    while k < length:
        if rng.random() < .2 and k >= menu_until:
            state = rng.choice((BoardState.TSUMO_FALL, BoardState.TSUMO_FALL, BoardState.CHAIN,
                                BoardState.OJAMA_FALL))
            plan[k] = state
            stay = rng.randrange(1, 7)
            plan[k+stay] = BoardState.STABLE
            truth = stacked_board(rng, rng.randrange(0, 30)) if state != BoardState.TSUMO_FALL \
                else _add_two(rng, truth)
        if sticky is None and rng.random() < .15:
            sticky = ((rng.randrange(1, BOARD_ROWS), rng.randrange(BOARD_COLS)), rng.choice((0, 1, 2, 9)),
                      rng.randrange(3, 14))
        cnn = noisy(rng, truth, .3)
        hsv = cnn.copy() if rng.random() < .8 else noisy(rng, truth, .5)
        if sticky is not None:
            (r, c), color, left = sticky
            cnn.set(r, c, color)
            hsv.set(r, c, color)
            sticky = None if left <= 1 else ((r, c), color, left-1)
        frames.append(dict(cnn=cnn, hsv=hsv, active=k >= menu_until or rng.random() < .3))
        k += 1
    return frames, plan


def _add_two(rng: random.Random, board: Board) -> Board:
    out = board.copy()
    for _ in range(2):
        col = rng.randrange(BOARD_COLS)
        for row in range(BOARD_ROWS-1, 0, -1):
            if out.get(row, col) == 0:
                out.set(row, col, rng.choice((1, 2, 3, 4)))
                break
    return out


def snapshot(ctx: StateContext) -> tuple:
    """比べる欄 (刻みモードで新しく足した欄は除く)。"""
    def grid(b: Board | None) -> bytes | None:
        return None if b is None else b.grid_bytes()
    return (ctx.state.value, grid(ctx.confirmed_board), grid(ctx.pending_board), ctx.pending_count,
            ctx.last_stable_idx, tuple(grid(b) for b in ctx.non_stable_cnn_history),
            tuple(sorted(ctx.stable_recovery_counters.items())), tuple(sorted(ctx.recovery_cells)),
            ctx.stable_warmup_remaining, ctx.chain_count, tuple(ctx.next_queue),
            tuple(sorted((k, tuple(v)) for k, v in ctx.cnn_flicker_history.items())))


def run_machine(frames: list[dict[str, Any]], plan: dict[int, BoardState], ticks: list[int],
                step: int, origin: int, ticked: bool, flags: dict[str, Any]) -> list[tuple]:
    """観測した刻み ticks だけを流し、各観測の後の snapshot を返す。"""
    sm = BoardStateMachine(detectors=[ScriptedDetector(plan, step, origin)], stable_frame_count=STABLE_N,
                           **flags)
    clock = mt.TickClock(step)
    out = []
    for k in ticks:
        index = origin+k*step
        stamp = clock.advance(index)
        frame = frames[k]
        signals = DetectorSignals(time_sec=index/60., cnn_board=frame['cnn'], is_match_active=frame['active'],
                                  hsv_board=frame['hsv'], media_tick=stamp if ticked else None)
        out.append(snapshot(sm.update(index, signals)))
    return out


# ---------------------------------------------------------------- E1 全コマで同一
@pytest.mark.parametrize('step, origin', [(2, VIDEO_ORIGIN), (1, VIDEO_ORIGIN), (1, 0)],
                         ids=['video_stride2_30Hz', 'video_stride1_60Hz', 'obs_index'])
@pytest.mark.parametrize('flags', [LIVE_FLAGS, PLAIN_FLAGS], ids=['live_flags', 'plain_flags'])
def test_full_frames_bit_identical(step: int, origin: int, flags: dict[str, Any]) -> None:
    """全コマ (毎回 Δk=1) では ON と OFF の状態が全観測で一致し、刻みで早まった件数が 0。"""
    rng = random.Random(SEED+step+origin)
    mt.audit_reset()
    observations = 0
    for _ in range(FULL_SEQUENCES):
        frames, plan = random_script(rng, FULL_LENGTH)
        ticks = list(range(FULL_LENGTH))
        off = run_machine(frames, plan, ticks, step, origin, False, flags)
        on = run_machine(frames, plan, ticks, step, origin, True, flags)
        assert on == off
        observations += len(ticks)
    audit = mt.audit_snapshot()
    # 母数: 各経路を実際に通ったこと (0 件を成功と読まない)
    for site in ('R01', 'R02', 'R04') + (('R03',) if flags.get('enable_initial_confirm_vote') else ()):
        assert audit.get(site+'.evaluated', 0) > 0, (site, audit)
    assert not {k: v for k, v in audit.items() if k.endswith('.early')}, audit
    assert observations == FULL_SEQUENCES*FULL_LENGTH


def test_off_path_never_touches_audit_or_new_fields() -> None:
    """OFF (media_tick=None) では診断の数えも新しい欄も書かれない (要件 E5)。"""
    rng = random.Random(SEED)
    mt.audit_reset()
    frames, plan = random_script(rng, FULL_LENGTH)
    sm = BoardStateMachine(detectors=[ScriptedDetector(plan, 2, 0)], stable_frame_count=STABLE_N, **LIVE_FLAGS)
    for k, frame in enumerate(frames):
        sm.update(2*k, DetectorSignals(time_sec=k/30., cnn_board=frame['cnn'], is_match_active=True,
                                       hsv_board=frame['hsv']))
        ctx = sm.context
        assert ctx.non_stable_cnn_weights == [] and ctx.recovery_tick_shadows == {}
        assert ctx.pending_tick == mt.TickShadow()
    assert mt.audit_snapshot() == {}


# ---------------------------------------------------------------- E2・E3 間引きで時間どおり
def first_reach(ticks: list[int], ticked: bool, step: int = 2) -> int | None:
    """MENU から同じ盤面を見続けたとき、最初に STABLE (初回確定) になった刻み。"""
    board = board_of({(12, 0): 1, (12, 1): 2})
    sm = BoardStateMachine(detectors=[], stable_frame_count=STABLE_N)
    clock = mt.TickClock(step)
    for k in ticks:
        stamp = clock.advance(VIDEO_ORIGIN+k*step)
        sm.update(VIDEO_ORIGIN+k*step, DetectorSignals(time_sec=k/30., cnn_board=board, is_match_active=True,
                                                       media_tick=stamp if ticked else None))
        if sm.context.state == BoardState.STABLE:
            return k
    return None


@pytest.mark.parametrize('ticks, on_tick, off_tick', [
    (list(range(10)), 2, 2),              # 全コマ: 3 回目 (刻み 2) で成立、ON/OFF 同じ
    (list(range(0, 20, 2)), 2, 4),        # 2 刻みおき: ON は刻み 2、OFF は 3 回目の刻み 4
    (list(range(0, 30, 3)), 3, 6),        # 3 刻みおき: ON は刻み 3 (遅れ 1 ≦ 間隔 3)
    ([0, 10, 11], 10, 11),                # 大きな空き: 2 回目の観測で成立 (1 回目では成立しない)
])
def test_r01_initial_confirm_on_time(ticks: list[int], on_tick: int, off_tick: int) -> None:
    assert first_reach(ticks, True) == on_tick
    assert first_reach(ticks, False) == off_tick


def test_r01_single_observation_never_confirms() -> None:
    assert first_reach([0], True) is None and first_reach([0], False) is None


def test_r01_random_thinning_property() -> None:
    """成立は全コマより前にならず、遅れは「成立をまたいだ観測の間隔」1 つ以内。OFF より遅くならない。"""
    rng = random.Random(SEED)
    full = first_reach(list(range(40)), True)
    assert full == STABLE_N-1
    checked = 0
    for _ in range(400):
        ticks, k = [], 0
        while k < 40:
            ticks.append(k)
            k += rng.choice(OBS21C_GAPS)
        on, off = first_reach(ticks, True), first_reach(ticks, False)
        assert on is not None and off is not None and full <= on <= off
        before = max(t for t in ticks if t < on)          # 成立の直前の観測
        assert before < full <= on or on == ticks[1]       # 遅れは成立をまたいだ間隔 1 つ
        checked += 1
    assert checked == 400


def history_vote(weights: list[int], seen_color: list[bool]) -> tuple[int, int]:
    """(1,1) セルだけが色 1 の観測を並べ、OFF と ON の多数決の (1,1) の値を返す。"""
    history = [board_of({(1, 1): 1}) if colored else Board() for colored in seen_color]
    off = _vote_majority_board(history, EMPTY_TO_COLOR_VOTES)
    on = _vote_majority_board_weighted(history, weights, EMPTY_TO_COLOR_VOTES)
    return off.get(1, 1), on.get(1, 1)


def test_r02_weighted_votes() -> None:
    assert history_vote([1, 1, 1, 1, 1], [False, False, True, True, True]) == (1, 1)   # 全コマ: 同じ
    assert history_vote([1, 1, 1, 1, 1], [False, False, False, True, True]) == (0, 0)  # 2 票: どちらも不採用
    assert history_vote([1, 2, 2], [False, True, True]) == (0, 1)                       # 2 観測・4 刻み: ON だけ採用
    assert history_vote([5], [True]) == (0, 0)                                          # 1 観測: ON でも不採用 (E3)


def test_r03_initial_confirm_weighted_with_fallback() -> None:
    """初回確定の多数決も重みで数え、観測不足のセルは単発の CNN 値 (fallback) を保つ。"""
    from src.board_state_machine import _build_initial_confirmed_board
    history = [Board(), board_of({(12, 1): 2}), board_of({(12, 1): 2})]
    new_cnn = board_of({(12, 1): 3, (12, 5): 3})            # 単発の CNN は (12,1) を別の色に読んだ
    off = _build_initial_confirmed_board(new_cnn, history, EMPTY_TO_COLOR_VOTES)
    on = _build_initial_confirmed_board(new_cnn, history, EMPTY_TO_COLOR_VOTES, [1, 2, 2])
    same = _build_initial_confirmed_board(new_cnn, history, EMPTY_TO_COLOR_VOTES, [1, 1, 1])
    assert off.get(12, 1) == 3 and on.get(12, 1) == 2       # OFF は 2 票で不足 → 単発値、ON は 2 観測・4 刻み
    assert off.get(12, 5) == 3 and on.get(12, 5) == 3       # 観測不足 → fallback
    assert same == off                                      # 重みがすべて 1 なら OFF と同じ


def recovery_reach(ticks: list[int], ticked: bool) -> int | None:
    """STABLE 中に CNN=HSV=赤 が confirmed=空 と食い違い続けたとき、復旧した刻み。"""
    stable = board_of({(12, 0): 2})
    seen = board_of({(12, 0): 2, (12, 1): 1})
    sm = BoardStateMachine(detectors=[], stable_frame_count=STABLE_N, enable_stable_recovery_gate=True)
    clock = mt.TickClock(1)
    for k in range(STABLE_N):    # 先に全コマで STABLE を確定させる
        clock.advance(k)
        sm.update(k, DetectorSignals(time_sec=k/30., cnn_board=stable, is_match_active=True, hsv_board=stable))
    assert sm.context.state == BoardState.STABLE
    for k in ticks:
        index = STABLE_N+k
        stamp = clock.advance(index)
        sm.update(index, DetectorSignals(time_sec=index/30., cnn_board=seen, is_match_active=True, hsv_board=seen,
                                         media_tick=stamp if ticked else None))
        if sm.context.confirmed_board.get(12, 1) == 1:
            return k
    return None


@pytest.mark.parametrize('ticks, on_tick, off_tick', [
    (list(range(20)), 7, 7),             # 全コマ: 8 回目で復旧、ON/OFF 同じ
    (list(range(0, 40, 2)), 8, 14),      # 2 刻みおき: ON 刻み 8 (遅れ 1)、OFF は 8 回目の刻み 14
    (list(range(0, 60, 3)), 9, 21),      # 3 刻みおき: ON 刻み 9 (遅れ 2 ≦ 間隔 3)
    ([0, 9, 10], 9, None),               # 2 回目の観測 (刻み 9) で成立。OFF は 3 回では未成立
])
def test_r04_recovery_on_time(ticks: list[int], on_tick: int, off_tick: int | None) -> None:
    assert recovery_reach(ticks, True) == on_tick
    assert recovery_reach(ticks, False) == off_tick


def test_r04_single_observation_never_recovers() -> None:
    assert recovery_reach([0], True) is None


# ---------------------------------------------------------------- B07 型 H2
def test_b07_move_window_frame() -> None:
    from src.b3_live_queues import LiveQueues
    queues = LiveQueues()
    mt.audit_reset()
    assert queues.move_window_frame(SimpleNamespace(), 100) is None               # OFF (刻みの幅なし)
    assert queues.prev_frame is None and mt.audit_snapshot() == {}
    pipe = SimpleNamespace(media_tick_step=2)
    assert [queues.move_window_frame(pipe, f) for f in (100, 102, 104)] == [100, 102, 104]   # 全コマ
    assert queues.move_window_frame(pipe, 110) == 106                             # 間引き: 前回 + 1 刻み
    assert queues.move_window_frame(pipe, 110) == 110                             # 同じコマが 2 回: 今のコマ
    assert mt.audit_snapshot() == {'B07.early': 1, 'B07.evaluated': 4}


def test_b07_live_queues_off_passes_no_new_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    """OFF では収集器への呼び出しが引数まで従来と同じ (古い収集器と組んでも壊れない)。"""
    from src import b3_live_queues as module
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(module.lean, '_process_side_lean', lambda *a, **k: seen.append(k))
    monkeypatch.setattr(module.lean, '_update_raw_pixel_stable', lambda *a, **k: True)
    queues = module.LiveQueues()
    side = SimpleNamespace(side='1P', confirmed_board=None, state=BoardState.STABLE, score=0, next_pair=None,
                           dnext_pair=None, chain_event=None, estimated_board=None, board_provenance=None,
                           cnn_board=None, raw_cnn_board=None)
    pipe = SimpleNamespace(_reader=None)
    queues._side(pipe, queues.states[0], side, None, 100, 1., 0)
    queues._side(pipe, queues.states[0], side, None, 106, 1., 0, 102)
    assert 'move_window_frame_idx' not in seen[0] and seen[1]['move_window_frame_idx'] == 102


def test_b07_process_side_lean_uses_window_frame() -> None:
    """締切 100 の窓に、コマ 110 で初めて新しい盤面が見えた。前回の観測 + 1 刻み (99) なら採用。"""
    from scripts import collect_boards_lean as lean
    rows: list[int] = []

    class Acc:
        def append(self, grid: Any, video: str, side: str, t: float, game: int, frame: int, **_: Any) -> None:
            rows.append(frame)

    def run(window: int | None) -> int:
        rows.clear()
        state = lean._SideState(move_window_deadline_fi=100, move_window_recorded=False,
                                prev_next_pair=(1, 2), prev_tsumo_count=3)
        kwargs = {} if window is None else dict(move_window_frame_idx=window)
        lean._process_side_lean(Acc(), state, '1P', board_of({(12, 0): 1}), BoardState.STABLE, 100, 'v', 1.0,
                                110, next_pair=(1, 2), tsumo_count=3, enable_move_segmented_recording=True,
                                **kwargs)
        return len(rows)
    assert run(None) == 0 and run(110) == 0     # 従来どおり (締切超え)
    assert run(99) == 1                         # 前回 + 1 刻みで見れば締切内


# ---------------------------------------------------------------- 配線と既定 OFF
def test_production_default_off_and_speed_key() -> None:
    from src.production_config import LIVE_FRAME_DROP_ROBUST_DEFAULT
    from src.phase_j.live_speed import SPEED_DEFAULTS
    assert LIVE_FRAME_DROP_ROBUST_DEFAULT is False
    assert SPEED_DEFAULTS['live_frame_drop_robust'] is False


def test_recognition_config_key_only_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_speed
    monkeypatch.delenv(live_speed.FRAME_DROP_ROBUST_ENV, raising=False)
    base = dict(stable_frame_count=3)
    assert live_speed.recognition_config(dict(base), False) == base
    monkeypatch.setenv(live_speed.FRAME_DROP_ROBUST_ENV, '0')
    assert live_speed.recognition_config(dict(base), False) == base
    monkeypatch.setenv(live_speed.FRAME_DROP_ROBUST_ENV, '1')
    assert live_speed.recognition_config(dict(base), False) == dict(base, enable_media_tick_counters=True)


def test_launcher_accepts_flag() -> None:
    from src.phase_j.launcher import UserConfig
    assert UserConfig.__dataclass_fields__['live_frame_drop_robust'].default is False


def bare_pipeline(enabled: bool) -> Any:
    from src.recognition_pipeline import RecognitionPipeline
    pipe = RecognitionPipeline.__new__(RecognitionPipeline)
    pipe._init_media_tick(enabled)
    return pipe


def test_pipeline_clock_on_and_off() -> None:
    off = bare_pipeline(False)
    off.set_media_tick_step(2)
    off._advance_media_tick(VIDEO_ORIGIN)
    assert off.media_tick_step is None and off._media_tick_stamp is None
    on = bare_pipeline(True)
    on._advance_media_tick(VIDEO_ORIGIN)                 # 幅が未設定: 回数どおり + 件数を残す
    assert on._media_tick_stamp is None and on._media_tick_unset_calls == 1
    on.set_media_tick_step(2)
    stamps = []
    for index in (VIDEO_ORIGIN, VIDEO_ORIGIN+2, VIDEO_ORIGIN+6):
        on._advance_media_tick(index)
        stamps.append((on._media_tick_stamp.now, on._media_tick_stamp.prev_call))
    assert on.media_tick_step == 2 and stamps == [(0, None), (1, 0), (3, 1)]


def r35_pipeline(pending: list[tuple[int, int]], stamp: mt.TickStamp | None) -> Any:
    """R35 の判定に要る欄だけを持つ認識器の器。"""
    from collections import Counter, deque
    pipe = bare_pipeline(stamp is not None)
    pipe._media_tick_stamp = stamp
    pipe._pending_tsumo_1p, pipe._pending_tsumo_2p = deque(pending), deque()
    pipe._tsumo_count_1p, pipe._tsumo_count_2p = Counter(), Counter()
    pipe._first_move_sec_1p = pipe._first_move_sec_2p = None
    return pipe


def test_r35_does_not_fire_on_full_frames() -> None:
    """前回の呼び出しから 1 刻み (全コマ) では、未着地の組があっても補わない (要件 E1)。"""
    mt.audit_reset()
    pipe = r35_pipeline([(1, 2)], mt.TickStamp(5, 4))
    assert pipe._r35_missed_placement('1P', 1.0, 'stable_plus2') is None
    assert pipe.tsumo_count('1P') == 0 and pipe._first_move_sec_1p is None and len(pipe._pending_tsumo_1p) == 1
    assert mt.audit_snapshot() == {'R35.stable_plus2.evaluated': 1}


def test_r35_fires_after_gap_with_pending_pair() -> None:
    from src.recognition_pipeline import R35_MIN_GAP_TICKS
    assert R35_MIN_GAP_TICKS == 2
    mt.audit_reset()
    pipe = r35_pipeline([(1, 2), (3, 4)], mt.TickStamp(6, 4))
    diag = pipe._r35_missed_placement('1P', 7.5, 'chain_start')
    assert diag == {'falling_pair_old': None, 'falling_pair_new': [1, 2], 'source': 'r35_chain_start'}
    assert pipe.tsumo_count('1P') == 1 and pipe._first_move_sec_1p == 7.5      # 最も古い組を 1 組だけ
    assert list(pipe._pending_tsumo_1p) == [(3, 4)]
    pipe._media_tick_stamp = mt.TickStamp(9, 6)
    pipe._r35_missed_placement('1P', 9.0, 'chain_start')
    assert pipe._first_move_sec_1p == 7.5                                      # 最初の手は上書きしない
    assert mt.audit_snapshot() == {'R35.chain_start.early': 2, 'R35.chain_start.evaluated': 2}


def test_r35_needs_next_moved() -> None:
    """NEXT が進んでいない (未着地の組が無い) なら補わない。"""
    pipe = r35_pipeline([], mt.TickStamp(6, 3))
    assert pipe._r35_missed_placement('2P', 1.0, 'stable_plus2') is None and pipe.tsumo_count('2P') == 0


def test_r35_off_is_noop() -> None:
    mt.audit_reset()
    pipe = r35_pipeline([(1, 2)], None)
    assert pipe._r35_missed_placement('1P', 1.0, 'stable_plus2') is None
    assert mt.audit_snapshot() == {} and len(pipe._pending_tsumo_1p) == 1


def test_live_process_configure_media_tick(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    from src.phase_j import live_process, live_speed
    calls: list[int] = []
    inner = SimpleNamespace(set_media_tick_step=calls.append, media_tick_step=None,
                            _enable_media_tick_counters=True)
    monkeypatch.setenv(live_speed.FRAME_DROP_ROBUST_ENV, '0')
    live_process.configure_media_tick(SimpleNamespace(pipe=inner), 2)
    assert calls == [] and capsys.readouterr().out == ''
    monkeypatch.setenv(live_speed.FRAME_DROP_ROBUST_ENV, '1')
    live_process.configure_media_tick(SimpleNamespace(pipe=inner), 2)
    assert calls == [2] and live_process.FRAME_DROP_ROBUST_MARK in capsys.readouterr().out


# ---------------------------------------------------------------- R06・R10 (型 A)・R15 (型 B)、2026-10-23 追加
ALL_STAGE2_SITES = frozenset({'R06', 'R10', 'R15'})


@pytest.fixture
def all_stage2_sites(monkeypatch: pytest.MonkeyPatch) -> None:
    """R06・R10 の性質試験は、採否 (production_config) に関係なく箇所を有効にして見る。"""
    import src.production_config as config
    monkeypatch.setattr(config, 'LIVE_FRAME_DROP_ROBUST_SITES', ALL_STAGE2_SITES)


def test_stage2_sites_default_is_empty() -> None:
    """23y (2026-10-23 夜) の採否: FDR は R15 だけを入れていたが、間引きの台の obs21c で足す前の版より +0.086pt (線 0.05) 悪く、
    R15 も外した。既定は空 = R06・R10・R15 は回数どおり (R01・R02/R03・R04・B07・R35 はフラグ ON で働く)。"""
    from src.production_config import LIVE_FRAME_DROP_ROBUST_SITES
    assert LIVE_FRAME_DROP_ROBUST_SITES == frozenset()
    assert not mt.site_enabled('R15') and not mt.site_enabled('R06') and not mt.site_enabled('R10')


@pytest.mark.parametrize('kind', ['R06', 'R10'])
def test_excluded_sites_count_like_off(kind: str) -> None:
    """既定の採否では、刻みモードでも R06・R10 は回数どおり (OFF と同じ成立の刻み)。"""
    counts = {k: 20 for k in range(120)}
    ticks = list(range(0, 60, 2))
    mt.audit_reset()
    assert settle_reach(kind, counts, ticks, True) == settle_reach(kind, counts, ticks, False) == 16
    assert kind+'.evaluated' not in mt.audit_snapshot()


def settle_detector(kind: str) -> Any:
    """R06 = 連鎖後の落下待ち (GravitySettleDetector)、R10 = おじゃま落下後の静止 (案B の退出)。"""
    if kind == 'R06':
        from src.state_detectors import GravitySettleDetector
        return GravitySettleDetector(), BoardState.GRAVITY_SETTLE
    from src.ojama_visual_detector import OjamaVisualDetector
    return OjamaVisualDetector(enable_ojama_fall_board_settle=True), BoardState.OJAMA_FALL


def puyo_board(count: int) -> Board:
    """下段から count 個の色ぷよを詰めた盤面 (ぷよ数だけが判定に効く)。"""
    board = Board()
    for i in range(count):
        board.set(BOARD_ROWS-1-i // BOARD_COLS, i % BOARD_COLS, 1)
    return board


def settle_reach(kind: str, counts: dict[int, int], ticks: list[int], ticked: bool) -> int | None:
    """刻み ticks だけ観測し、検出器が STABLE を返した刻み (返さなければ None)。"""
    detector, state = settle_detector(kind)
    clock = mt.TickClock(2)
    for k in ticks:
        stamp = clock.advance(VIDEO_ORIGIN+2*k)
        ctx = StateContext(state=state, frame_idx=VIDEO_ORIGIN+2*k)
        signals = DetectorSignals(time_sec=k/30., cnn_board=puyo_board(counts[k]), is_match_active=True,
                                  media_tick=stamp if ticked else None)
        if detector.detect(ctx, signals) == BoardState.STABLE:
            return k
    return None


@pytest.mark.parametrize('kind', ['R06', 'R10'])
def test_settle_full_frames_identical(kind: str, all_stage2_sites: None) -> None:
    """全コマの乱数列 (ぷよ数のゆれ) で、ON と OFF の成立の刻みが一致し、早まり 0 件。"""
    rng = random.Random(SEED)
    mt.audit_reset()
    reached = 0
    for _ in range(600):
        counts = {k: max(0, 20+rng.choice((0, 0, 0, 1, -1, 2, -2, 3))) for k in range(60)}
        ticks = list(range(60))
        off, on = settle_reach(kind, counts, ticks, False), settle_reach(kind, counts, ticks, True)
        assert on == off
        reached += off is not None
    audit = mt.audit_snapshot()
    assert audit.get(kind+'.evaluated', 0) > 0 and kind+'.early' not in audit, audit
    assert reached > 0


@pytest.mark.parametrize('kind', ['R06', 'R10'])
@pytest.mark.parametrize('ticks, on_tick, off_tick', [
    (list(range(40)), 9, 9),             # 全コマ: 最低待機 (刻み 1) の後 8 回連続で刻み 9
    (list(range(0, 60, 2)), 10, 16),     # 2 刻みおき: ON は刻み 10 (遅れ 1)、OFF は 8 回目の刻み 16
    (list(range(0, 90, 3)), 12, 24),     # 3 刻みおき: ON は刻み 12 (遅れ 3 ≦ 間隔 3)
])
def test_settle_thinned_on_time(kind: str, ticks: list[int], on_tick: int, off_tick: int,
                                all_stage2_sites: None) -> None:
    counts = {k: 20 for k in range(120)}   # ぷよ数が動かない (落下が収まっている)
    assert settle_reach(kind, counts, ticks, True) == on_tick
    assert settle_reach(kind, counts, ticks, False) == off_tick


@pytest.mark.parametrize('kind', ['R06', 'R10'])
def test_settle_single_stable_observation_does_not_settle(kind: str, all_stage2_sites: None) -> None:
    """最初の観測 + 静止 1 回だけでは、刻みがいくら空いても成立しない (E3)。"""
    counts = {k: 20 for k in range(60)}
    assert settle_reach(kind, counts, [0, 2, 30], True) == 30   # 2 回目の静止の観測で成立
    assert settle_reach(kind, counts, [0, 30], True) is None    # 静止の観測が 1 回だけ


def test_r15_warmup_step() -> None:
    from src.recognition_pipeline import OJAMA_TIER1_WARMUP_FRAMES, _update_ojama_tier1_warmup_counter as update
    stable, ojama = BoardState.STABLE, BoardState.OJAMA_FALL
    assert update(ojama, stable, 0) == OJAMA_TIER1_WARMUP_FRAMES == update(ojama, stable, 0, step=3)
    assert update(stable, stable, 8) == 7 == update(stable, stable, 8, step=1)   # 既定 = 従来
    assert update(stable, stable, 8, step=3) == 5 and update(stable, stable, 2, step=3) == 0   # 下限 0
    assert update(stable, ojama, 5, step=3) == 0


def test_r15_step_off_and_on(all_stage2_sites: None) -> None:   # R15 の性質は、採否 (既定は空) に関係なく箇所を有効にして見る
    mt.audit_reset()
    off = bare_pipeline(False)
    assert off._r15_warmup_step() == 1 and mt.audit_snapshot() == {}
    on = bare_pipeline(True)
    on._media_tick_stamp = mt.TickStamp(7, 4)
    assert on._r15_warmup_step() == 3
    on._media_tick_stamp = mt.TickStamp(8, 7)
    assert on._r15_warmup_step() == 1
    assert mt.audit_snapshot() == {'R15.early': 1, 'R15.evaluated': 2}
