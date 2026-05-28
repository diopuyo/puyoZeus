from __future__ import annotations
import sys
from pathlib import Path
import pytest
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from scripts.measure_stable_cell_acc import (
    VideoStats,
    _collect_col_metrics,
    _judge_i1_metrics,
    _build_i1_summary,
    PER_COL_UNKNOWN_WARNING,
    PER_COL_UNKNOWN_CRITICAL,
    NON_STABLE_CRITICAL_FRAMES,
    NON_STABLE_WARMUP_SEC,
    MIDGAME_START_SEC,
    MIDGAME_COL_EMPTY_CRITICAL,
    MIDGAME_COL_MIN_FRAMES,
    BOARD_ROWS,
    BOARD_COLS,
)
from src.board import (COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_UNKNOWN,)

# ==========================
# ヘルパー
# ==========================

class _FakeBoardI1:
    '''テスト用簡易盤面: col_overrides dict でセルを上書き可能。'''

    def __init__(self, fill_val: int = COLOR_RED, col_overrides: dict | None = None) -> None:
        self._fill = fill_val
        self._cell: dict[tuple[int, int], int] = {}
        if col_overrides:
            for col_idx, val in col_overrides.items():
                for row in range(BOARD_ROWS):
                    self._cell[(row, col_idx)] = val

    def set_cell(self, row: int, col: int, val: int) -> None:
        self._cell[(row, col)] = val

    def get(self, row: int, col: int) -> int:
        return self._cell.get((row, col), self._fill)


def _mk(vid: str = 'vtest_i1') -> VideoStats:
    '''テスト用 VideoStats を生成するファクトリ。'''
    return VideoStats(video_id=vid, is_holdout=False)


# ==========================
# _collect_col_metrics テスト
# ==========================

def test_i1_collect_no_unknown_stable_cells_incremented():
    '''全 cell が非 UNKNOWN の盤面では per_col_stable_cells が BOARD_ROWS 増えること。'''
    stats = _mk()
    board = _FakeBoardI1(fill_val=COLOR_RED)
    _collect_col_metrics(fi=0, t_sec=0.0, confirmed_board=board, stats=stats)
    for col in range(BOARD_COLS):
        assert stats.per_col_stable_cells[col] == BOARD_ROWS
    for col in range(BOARD_COLS):
        assert stats.per_col_unknown_cells[col] == 0


def test_i1_collect_all_unknown_increments_unknown_cells():
    '''v89_match01 27-30s 認識崩壊: 全 UNKNOWN 盤面で per_col_unknown_cells が BOARD_ROWS 増えること。'''
    stats = _mk()
    board = _FakeBoardI1(fill_val=COLOR_UNKNOWN)
    _collect_col_metrics(fi=0, t_sec=0.0, confirmed_board=board, stats=stats)
    for col in range(BOARD_COLS):
        assert stats.per_col_unknown_cells[col] == BOARD_ROWS, (
            f'col={col}: unknown={stats.per_col_unknown_cells[col]}'
        )


def test_i1_collect_partial_unknown_col0_only():
    '''col=0 のみ UNKNOWN の盤面では col=0 の unknown_cells のみ増えること。'''
    stats = _mk()
    board = _FakeBoardI1(fill_val=COLOR_RED, col_overrides={0: COLOR_UNKNOWN})
    _collect_col_metrics(fi=0, t_sec=0.0, confirmed_board=board, stats=stats)
    assert stats.per_col_unknown_cells[0] == BOARD_ROWS
    for col in range(1, BOARD_COLS):
        assert stats.per_col_unknown_cells[col] == 0, f'col={col} は 0 であるべき'


def test_i1_collect_midgame_empty_accumulated():
    '''v40_match01: t_sec >= MIDGAME_START_SEC で per_col_midgame_cells が加算されること。'''
    stats = _mk()
    board = _FakeBoardI1(fill_val=COLOR_EMPTY)
    t_mid = MIDGAME_START_SEC + 1.0
    _collect_col_metrics(fi=0, t_sec=t_mid, confirmed_board=board, stats=stats)
    for col in range(BOARD_COLS):
        assert stats.per_col_midgame_cells[col] == BOARD_ROWS
        assert stats.per_col_midgame_empty_cells[col] == BOARD_ROWS


def test_i1_collect_before_midgame_no_midgame_accum():
    '''t_sec < MIDGAME_START_SEC では midgame_cells は加算されないこと。'''
    stats = _mk()
    board = _FakeBoardI1(fill_val=COLOR_EMPTY)
    t_early = MIDGAME_START_SEC - 1.0
    _collect_col_metrics(fi=0, t_sec=t_early, confirmed_board=board, stats=stats)
    for col in range(BOARD_COLS):
        assert stats.per_col_midgame_cells.get(col, 0) == 0
        assert stats.per_col_midgame_empty_cells.get(col, 0) == 0


# ==========================
# _judge_i1_metrics テスト
# ==========================

def test_i1_judge_unknown_critical_fires():
    '''per_col_unknown_rate >= CRITICAL 閾値 で FAIL 文字列が返ること。

    必須テスト: v89_match01 27-30 秒 col=0,1 認識不能パターン。
    mismatch/replace が fail-silent でも本メトリクスは発火する。
    '''
    stats = _mk(vid='v89_match01')
    for row in range(BOARD_ROWS):
        stats.per_col_unknown_cells[0] += 1
        stats.per_col_stable_cells[0] += 1
        stats.per_col_unknown_cells[1] += 1
        stats.per_col_stable_cells[1] += 1
    for col in range(2, BOARD_COLS):
        stats.per_col_stable_cells[col] = BOARD_ROWS
    failures = _judge_i1_metrics([stats])
    assert len(failures) > 0, 'CRITICAL UNKNOWN 率で failures が空'
    col0_fails = [f for f in failures if 'col=0' in f]
    assert len(col0_fails) > 0, f'col=0 の FAIL なし: {failures}'
    col1_fails = [f for f in failures if 'col=1' in f]
    assert len(col1_fails) > 0, f'col=1 の FAIL なし: {failures}'
    assert any('CRITICAL' in f for f in failures)
    assert any('v89_match01' in f for f in failures)


def test_i1_judge_unknown_warning_fires():
    '''per_col_unknown_rate が WARNING 以上 CRITICAL 未満で WARNING が返ること。'''
    stats = _mk()
    total = 100
    stats.per_col_stable_cells[2] = total
    stats.per_col_unknown_cells[2] = 20
    for col in [0, 1, 3, 4, 5]:
        stats.per_col_stable_cells[col] = total
    failures = _judge_i1_metrics([stats])
    assert len(failures) > 0, 'WARNING UNKNOWN 率で failures が空'
    assert any('col=2' in f for f in failures)


def test_i1_judge_unknown_ok_no_failure():
    '''per_col_unknown_rate < WARNING 閾値 (5%) では UNKNOWN failures が空であること。'''
    stats = _mk()
    total = 200
    for col in range(BOARD_COLS):
        stats.per_col_stable_cells[col] = total
        stats.per_col_unknown_cells[col] = 5
    failures = _judge_i1_metrics([stats])
    unknown_fails = [f for f in failures if 'UNKNOWN' in f]
    assert len(unknown_fails) == 0, f'低 UNKNOWN 率で failures: {unknown_fails}'


def test_i1_judge_non_stable_critical_fires():
    '''non_stable_max_consecutive >= CRITICAL 閾値 で FAIL 文字列が返ること。'''
    stats = _mk()
    stats.non_stable_max_consecutive = NON_STABLE_CRITICAL_FRAMES
    failures = _judge_i1_metrics([stats])
    non_stable_fails = [f for f in failures if 'non_stable' in f]
    assert len(non_stable_fails) > 0, f'non_stable CRITICAL で failures なし: {failures}'


def test_i1_judge_non_stable_ok_no_failure():
    '''non_stable_max_consecutive < CRITICAL 閾値 では failures が空であること。'''
    stats = _mk()
    stats.non_stable_max_consecutive = NON_STABLE_CRITICAL_FRAMES - 1
    failures = _judge_i1_metrics([stats])
    non_stable_fails = [f for f in failures if 'non_stable' in f]
    assert len(non_stable_fails) == 0


def test_i1_judge_midgame_empty_critical_fires():
    '''per_col_midgame_empty_rate >= CRITICAL 閾値 で FAIL 文字列が返ること。

    必須テスト: v40_match01 col=1 全 EMPTY 誤判定パターン。
    '''
    stats = _mk(vid='v40_match01')
    n_cells = 60 * BOARD_ROWS
    stats.per_col_midgame_cells[1] = n_cells
    stats.per_col_midgame_empty_cells[1] = n_cells
    failures = _judge_i1_metrics([stats])
    assert len(failures) > 0, '全 EMPTY col=1 で failures が空'
    col1_fails = [f for f in failures if 'col=1' in f]
    assert len(col1_fails) > 0, f'col=1 の FAIL なし: {failures}'
    assert any('EMPTY' in f for f in failures)
    assert any('v40_match01' in f for f in failures)


def test_i1_judge_midgame_empty_below_min_frames_no_fire():
    '''per_col_midgame_cells < MIDGAME_COL_MIN_FRAMES では CRITICAL でも発火しないこと。'''
    stats = _mk()
    n_cells = max(1, MIDGAME_COL_MIN_FRAMES - 1)
    stats.per_col_midgame_cells[0] = n_cells
    stats.per_col_midgame_empty_cells[0] = n_cells
    failures = _judge_i1_metrics([stats])
    empty_fails = [f for f in failures if 'EMPTY' in f]
    assert len(empty_fails) == 0, f'frames 不足なのに EMPTY FAIL: {empty_fails}'


def test_i1_judge_midgame_empty_partial_ok():
    '''per_col_midgame_empty_rate < CRITICAL 閾値 (50%) では failures なし。'''
    stats = _mk()
    n_cells = 100
    stats.per_col_midgame_cells[3] = n_cells
    stats.per_col_midgame_empty_cells[3] = 50
    failures = _judge_i1_metrics([stats])
    col3_fails = [f for f in failures if ('EMPTY' in f and 'col=3' in f)]
    assert len(col3_fails) == 0


# ==========================
# _build_i1_summary テスト
# ==========================

def test_i1_summary_structure_has_required_keys():
    '''_build_i1_summary が必須キーを全て持つ dict を返すこと。'''
    stats = _mk()
    summary = _build_i1_summary([stats])
    assert 'per_col_unknown_worst' in summary
    assert 'non_stable_max_consecutive' in summary
    assert 'per_col_midgame_empty_worst' in summary
    assert 'thresholds' in summary
    t = summary['thresholds']
    assert 'per_col_unknown_warning' in t
    assert 'per_col_unknown_critical' in t
    assert 'non_stable_critical_frames' in t
    assert 'midgame_col_empty_critical' in t


def test_i1_summary_worst_video_selected_correctly():
    '''複数動画のうち worst-case 動画が正しく選ばれること。'''
    s1 = VideoStats(video_id='v_low', is_holdout=False)
    s2 = VideoStats(video_id='v_high', is_holdout=False)
    s1.per_col_stable_cells[0] = 100
    s1.per_col_unknown_cells[0] = 10
    s2.per_col_stable_cells[0] = 100
    s2.per_col_unknown_cells[0] = 40
    summary = _build_i1_summary([s1, s2])
    worst_col0 = summary['per_col_unknown_worst']['0']
    assert worst_col0['video'] == 'v_high', f'worst は v_high: {worst_col0}'
    assert abs(worst_col0['rate'] - 0.40) < 1e-9


def test_i1_summary_empty_stats_no_error():
    '''空の stats_list で _build_i1_summary がエラーなく動作すること。'''
    summary = _build_i1_summary([])
    assert 'per_col_unknown_worst' in summary
    assert summary['non_stable_max_consecutive']['max'] == 0


# ==========================
# VideoStats 後方互換テスト
# ==========================

def test_i1_videostats_backward_compat_defaults():
    '''VideoStats の既存フィールドが崩れておらず、新フィールドが正しく初期化されること。'''
    s = VideoStats(video_id='v_compat_test', is_holdout=False)
    assert s.total_cells == 0
    assert s.agreed_cells == 0
    assert s.stable_frame_count == 0
    assert s.physics_fix_count == 0
    assert s.all_three_agree_count == 0
    assert s.non_stable_max_consecutive == 0
    assert s.per_col_unknown_cells[0] == 0
    assert s.per_col_stable_cells[5] == 0
    assert s.per_col_midgame_cells[2] == 0
    assert s.per_col_midgame_empty_cells[3] == 0

