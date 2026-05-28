"""recognition_evaluator のユニットテスト (cycle 33+).

各メトリクスが既知パターンで正しく violation を flag できることを test。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_RED,
    Board,
)
from src.recognition_evaluator import (
    FrameEntry,
    RecognitionEvaluator,
    SEVERITY_CRITICAL,
)


def _make_empty_grid() -> list[list[int]]:
    return [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]


def _make_grid_with_color(color: int, count: int) -> list[list[int]]:
    """下段から count 個 cell を color で埋めた grid を返す."""
    grid = _make_empty_grid()
    placed = 0
    for r in range(BOARD_ROWS - 1, -1, -1):
        for c in range(BOARD_COLS):
            if placed >= count:
                return grid
            grid[r][c] = color
            placed += 1
    return grid


def _make_frame_entry(
    fi: int, p1_grid: list[list[int]] | None,
    p2_grid: list[list[int]] | None = None,
    p1_state: str = "stable", p2_state: str = "stable",
) -> FrameEntry:
    if p2_grid is None:
        p2_grid = _make_empty_grid()
    return FrameEntry(
        frame_idx=fi, t_sec=fi / 60.0,
        p1_state=p1_state, p2_state=p2_state,
        p1_confirmed=p1_grid, p2_confirmed=p2_grid,
    )


class TestPuyoCountConsistency:
    def test_normal_stable_to_stable_2_puyo_increase_no_violation(self):
        """STABLE 間で +2 (= ツモ着地) は正常."""
        evaluator = RecognitionEvaluator()
        evaluator.entries = [
            _make_frame_entry(0, _make_grid_with_color(COLOR_RED, 4)),
            _make_frame_entry(60, _make_grid_with_color(COLOR_RED, 6)),
        ]
        violations = evaluator.check_puyo_count_consistency("1P")
        assert len(violations) == 0

    def test_sudden_drop_critical(self):
        """STABLE→STABLE で 大幅減なら critical."""
        evaluator = RecognitionEvaluator()
        evaluator.entries = [
            _make_frame_entry(0, _make_grid_with_color(COLOR_RED, 30)),
            _make_frame_entry(60, _make_grid_with_color(COLOR_RED, 10)),
        ]
        violations = evaluator.check_puyo_count_consistency("1P")
        assert len(violations) == 1
        assert violations[0].metric == "puyo_count_drop"
        assert violations[0].severity == SEVERITY_CRITICAL


class TestRetrospectiveChain:
    def test_stable_to_chain_with_4_connect_no_violation(self):
        """STABLE → CHAIN 時に 4 連結あれば正常."""
        evaluator = RecognitionEvaluator()
        # 下段に red 4 個並べる
        grid = _make_empty_grid()
        grid[12][0] = COLOR_RED
        grid[12][1] = COLOR_RED
        grid[12][2] = COLOR_RED
        grid[12][3] = COLOR_RED
        evaluator.entries = [
            _make_frame_entry(0, grid, p1_state="stable"),
            _make_frame_entry(60, grid, p1_state="chain"),
        ]
        violations = evaluator.check_retrospective_chain("1P")
        assert len(violations) == 0

    def test_stable_to_chain_without_4_connect_violation(self):
        """STABLE → CHAIN 時に 4 連結なしなら critical."""
        evaluator = RecognitionEvaluator()
        # 4 連結なし
        grid = _make_empty_grid()
        grid[12][0] = COLOR_RED
        grid[12][1] = COLOR_BLUE
        evaluator.entries = [
            _make_frame_entry(0, grid, p1_state="stable"),
            _make_frame_entry(60, grid, p1_state="chain"),
        ]
        violations = evaluator.check_retrospective_chain("1P")
        assert len(violations) == 1
        assert violations[0].metric == "retrospective_chain_missing"


class TestAutoCorrection:
    def test_color_unchanged_no_violation(self):
        evaluator = RecognitionEvaluator()
        grid = _make_grid_with_color(COLOR_RED, 4)
        evaluator.entries = [
            _make_frame_entry(0, [row[:] for row in grid]),
            _make_frame_entry(60, [row[:] for row in grid]),
        ]
        violations = evaluator.check_auto_correction("1P")
        assert len(violations) == 0

    def test_red_to_blue_change_violation(self):
        """同位置 cell が RED → BLUE に変化 = 誤認 signal."""
        evaluator = RecognitionEvaluator()
        grid_a = _make_empty_grid()
        grid_a[12][0] = COLOR_RED
        grid_b = _make_empty_grid()
        grid_b[12][0] = COLOR_BLUE  # 色変化
        evaluator.entries = [
            _make_frame_entry(0, grid_a),
            _make_frame_entry(60, grid_b),
        ]
        violations = evaluator.check_auto_correction("1P")
        assert len(violations) == 1
        assert violations[0].metric == "auto_correction"


class TestFloatingPuyo:
    def test_no_floating_no_violation(self):
        """下段に詰まっていれば違反なし."""
        evaluator = RecognitionEvaluator()
        grid = _make_empty_grid()
        grid[12][0] = COLOR_RED
        grid[11][0] = COLOR_RED  # 上に積み、 浮きなし
        evaluator.entries = [_make_frame_entry(0, grid)]
        violations = evaluator.check_floating_puyo("1P")
        assert len(violations) == 0

    def test_floating_puyo_violation(self):
        """下段 EMPTY、 上段 puyo = 浮き = 違反."""
        evaluator = RecognitionEvaluator()
        grid = _make_empty_grid()
        # 下 (row=12) EMPTY、 row=10 だけ RED
        grid[10][0] = COLOR_RED
        evaluator.entries = [_make_frame_entry(0, grid)]
        violations = evaluator.check_floating_puyo("1P")
        assert any(v.metric == "floating_puyo" for v in violations)


class TestBgColorDominant:
    def test_normal_distribution_no_violation(self):
        """5 色がほぼ均等なら違反なし."""
        evaluator = RecognitionEvaluator()
        grid = _make_empty_grid()
        # 各色 3 cell ずつ配置 (= 全 cell の 5/78 = 6.4% < 35%)
        colors = [1, 2, 3, 4, 5]
        for i, col in enumerate(colors):
            for j in range(3):
                grid[12 - j][i] = col
        evaluator.entries = [_make_frame_entry(0, grid)]
        violations = evaluator.check_background_color_distribution("1P")
        assert len(violations) == 0

    def test_blue_dominant_violation(self):
        """全 cell の 35%+ が BLUE = 背景誤認 signal."""
        evaluator = RecognitionEvaluator()
        # 78 cell 中 30 cell (= 38%) を BLUE で埋める
        grid = _make_empty_grid()
        placed = 0
        for r in range(BOARD_ROWS - 1, -1, -1):
            for c in range(BOARD_COLS):
                if placed >= 30:
                    break
                grid[r][c] = COLOR_BLUE
                placed += 1
            if placed >= 30:
                break
        evaluator.entries = [_make_frame_entry(0, grid)]
        violations = evaluator.check_background_color_distribution("1P")
        assert any(v.metric == "bg_color_dominant" for v in violations)


class TestSparseColorPop:
    def test_no_pop_long_lived_puyo_no_violation(self):
        """puyo が長期間 (= 11+ frame) 持続なら散発的でない."""
        evaluator = RecognitionEvaluator()
        grid_with_puyo = _make_empty_grid()
        grid_with_puyo[12][0] = COLOR_RED
        # 11 frame 持続
        for i in range(12):
            evaluator.entries.append(_make_frame_entry(i * 60, [row[:] for row in grid_with_puyo]))
        # その後 empty に
        evaluator.entries.append(_make_frame_entry(720, _make_empty_grid()))
        violations = evaluator.check_sparse_color_pop("1P")
        assert len(violations) == 0  # 持続 > 10 frame なので散発的でない

    def test_short_lived_puyo_violation(self):
        """puyo が短期間 (= 5 frame) で消えれば散発的誤認."""
        evaluator = RecognitionEvaluator()
        # frame 0: empty
        evaluator.entries.append(_make_frame_entry(0, _make_empty_grid()))
        # frame 1-5: red
        grid_red = _make_empty_grid()
        grid_red[12][0] = COLOR_RED
        for i in range(1, 6):
            evaluator.entries.append(_make_frame_entry(i, [row[:] for row in grid_red]))
        # frame 6: empty (= 5 frame で消失)
        evaluator.entries.append(_make_frame_entry(6, _make_empty_grid()))
        violations = evaluator.check_sparse_color_pop("1P")
        assert len(violations) >= 1
        assert any(v.metric == "sparse_color_pop" for v in violations)

    def test_ojama_excluded_from_bg_color_dominant(self):
        """ojama 大量降下は bg_color_dominant を発火しない (= バグ修正確認)."""
        evaluator = RecognitionEvaluator()
        # 盤面の 50% を ojama で埋める
        grid_ojama = _make_empty_grid()
        from src.board import COLOR_OJAMA
        placed = 0
        for r in range(BOARD_ROWS - 1, -1, -1):
            for c in range(BOARD_COLS):
                if placed >= 40:
                    break
                grid_ojama[r][c] = COLOR_OJAMA
                placed += 1
            if placed >= 40:
                break
        evaluator.entries.append(_make_frame_entry(0, grid_ojama))
        violations = evaluator.check_background_color_distribution("1P")
        assert len(violations) == 0  # ojama 除外で発火しない


class TestEvaluateAll:
    def test_no_violations_clean_input(self):
        """正常な盤面のみなら違反ゼロ."""
        evaluator = RecognitionEvaluator()
        # 下段から積んだ整然とした盤面 (4 連結なし、 浮きなし)
        grid = _make_empty_grid()
        grid[12][0] = COLOR_RED
        grid[12][1] = COLOR_BLUE
        grid[12][2] = COLOR_RED
        grid[12][3] = COLOR_BLUE
        evaluator.entries = [
            _make_frame_entry(0, [row[:] for row in grid]),
            _make_frame_entry(60, [row[:] for row in grid]),
        ]
        report = evaluator.generate_report()
        # 違反ゼロまたは無視可能な数で ACCEPT
        assert report["verdict"] in ("ACCEPT", "REVIEW")

    def test_multiple_violations_reject(self):
        """大量の問題で REJECT."""
        evaluator = RecognitionEvaluator()
        # 浮き puyo × 多数 + bg_color_dominant
        for fi in range(0, 600, 60):
            grid = _make_empty_grid()
            # row=10 に BLUE 40 cell (浮き + dominant)
            for c in range(BOARD_COLS):
                grid[10][c] = COLOR_BLUE
                grid[8][c] = COLOR_BLUE
                grid[6][c] = COLOR_BLUE
                grid[4][c] = COLOR_BLUE
                grid[2][c] = COLOR_BLUE
            evaluator.entries.append(_make_frame_entry(fi, grid))
        report = evaluator.generate_report()
        # 浮き or bg_dominant violations が多数出る
        assert report["summary"]["critical"] > 0


# ============================
# KB (cycle 56, 2026-05-22): ojama 認識退行 catch
# ============================


def _make_grid_with_ojama(count: int) -> list[list[int]]:
    """下段から count 個 cell を OJAMA (= 9) で埋めた grid."""
    return _make_grid_with_color(9, count)


class TestOjamaDisappearance:
    """check_ojama_disappearance = STABLE 間で ojama → EMPTY 遷移 catch."""

    def test_ojama_to_empty_3_in_stable_critical(self) -> None:
        """STABLE → STABLE で ojama 3 個 → EMPTY = CRITICAL."""
        entries = [
            _make_frame_entry(0, _make_grid_with_ojama(5)),
            _make_frame_entry(60, _make_grid_with_empty := _make_empty_grid()),
        ]
        ev = RecognitionEvaluator()
        ev.entries = entries
        vs = ev.check_ojama_disappearance("1P")
        assert len(vs) >= 1
        assert vs[0].metric == "ojama_disappearance"
        assert vs[0].severity == SEVERITY_CRITICAL

    def test_ojama_to_empty_2_below_threshold_no_violation(self) -> None:
        """1 個消失は閾値 (= 3) 未満で発火しない."""
        entries = [
            _make_frame_entry(0, _make_grid_with_ojama(5)),
            _make_frame_entry(60, _make_grid_with_ojama(4)),
        ]
        ev = RecognitionEvaluator()
        ev.entries = entries
        vs = ev.check_ojama_disappearance("1P")
        assert len(vs) == 0

    def test_no_ojama_no_violation(self) -> None:
        """ojama 一切ないなら発火しない."""
        entries = [
            _make_frame_entry(0, _make_empty_grid()),
            _make_frame_entry(60, _make_empty_grid()),
        ]
        ev = RecognitionEvaluator()
        ev.entries = entries
        vs = ev.check_ojama_disappearance("1P")
        assert len(vs) == 0


class TestOjamaGlobalScarcity:
    """check_ojama_global_scarcity = 全 STABLE frame で ojama 認識率極小."""

    def test_zero_ojama_for_100_frames_critical(self) -> None:
        """100 STABLE frame で ojama 0 件 = CRITICAL."""
        entries = [
            _make_frame_entry(i, _make_empty_grid()) for i in range(150)
        ]
        ev = RecognitionEvaluator()
        ev.entries = entries
        vs = ev.check_ojama_global_scarcity("1P")
        assert len(vs) >= 1
        assert vs[0].metric == "ojama_global_scarcity"
        assert vs[0].severity == SEVERITY_CRITICAL

    def test_normal_ojama_5_percent_no_violation(self) -> None:
        """ojama 5% 認識なら発火しない (= 通常水準)."""
        # 全 cell 78 中 ojama 4 個 = 5.1%、 100 frame
        entries = [
            _make_frame_entry(i, _make_grid_with_ojama(4)) for i in range(150)
        ]
        ev = RecognitionEvaluator()
        ev.entries = entries
        vs = ev.check_ojama_global_scarcity("1P")
        assert len(vs) == 0

    def test_insufficient_frames_no_violation(self) -> None:
        """100 frame 未満なら発火しない (= 判定保留)."""
        entries = [
            _make_frame_entry(i, _make_empty_grid()) for i in range(50)
        ]
        ev = RecognitionEvaluator()
        ev.entries = entries
        vs = ev.check_ojama_global_scarcity("1P")
        assert len(vs) == 0


class TestErasureAlertsIntegration:
    """T4 PuyoErasureMonitor: board_log → FrameEntry → p_to_e_count 集計の統合テスト。"""

    def _make_frame_entry_with_alerts(
        self,
        fi: int,
        p1_alerts: list[list[int]],
        p2_alerts: list[list[int]],
    ) -> FrameEntry:
        """erasure_alerts 付きの FrameEntry を生成する。"""
        return FrameEntry(
            frame_idx=fi,
            t_sec=fi / 60.0,
            p1_state="stable",
            p2_state="stable",
            p1_confirmed=_make_empty_grid(),
            p2_confirmed=_make_empty_grid(),
            p1_erasure_alerts=p1_alerts,
            p2_erasure_alerts=p2_alerts,
        )

    def test_no_alerts_p_to_e_count_zero(self) -> None:
        """alert なしなら p_to_e_count = 0。"""
        entries = [
            self._make_frame_entry_with_alerts(i, [], []) for i in range(10)
        ]
        ev = RecognitionEvaluator()
        ev.entries = entries
        counts = ev.count_erasure_alerts()
        assert counts["total"] == 0
        assert counts["p1"] == 0
        assert counts["p2"] == 0

    def test_alerts_summed_correctly(self) -> None:
        """1P 2 件 + 2P 1 件 = total 3 件に集計される。"""
        entries = [
            self._make_frame_entry_with_alerts(0, [[5, 2], [6, 3]], []),
            self._make_frame_entry_with_alerts(1, [], [[4, 1]]),
        ]
        ev = RecognitionEvaluator()
        ev.entries = entries
        counts = ev.count_erasure_alerts()
        assert counts["p1"] == 2
        assert counts["p2"] == 1
        assert counts["total"] == 3

    def test_generate_report_contains_p_to_e_count(self) -> None:
        """generate_report() に p_to_e_count キーが含まれる。"""
        entries = [
            self._make_frame_entry_with_alerts(i, [[5, 2]], []) for i in range(5)
        ]
        ev = RecognitionEvaluator()
        ev.entries = entries
        report = ev.generate_report()
        assert "p_to_e_count" in report
        assert report["p_to_e_count"] == 5  # 5 frame × 1 alert

    def test_frame_entry_from_jsonable_with_erasure_alerts(self) -> None:
        """from_jsonable() が erasure_alerts を正しく読み込む。"""
        obj = {
            "frame_idx": 100,
            "t_sec": 1.67,
            "p1_state": "stable",
            "p2_state": "stable",
            "p1_confirmed": None,
            "p2_confirmed": None,
            "p1_erasure_alerts": [[5, 2], [6, 3]],
            "p2_erasure_alerts": [[4, 1]],
        }
        entry = FrameEntry.from_jsonable(obj)
        assert entry.p1_erasure_alerts == [[5, 2], [6, 3]]
        assert entry.p2_erasure_alerts == [[4, 1]]

    def test_frame_entry_from_jsonable_backward_compat_no_key(self) -> None:
        """古い board_log (erasure_alerts キーなし) でも空リストで処理継続。"""
        obj = {
            "frame_idx": 50,
            "t_sec": 0.83,
            "p1_state": "stable",
            "p2_state": "stable",
            "p1_confirmed": None,
            "p2_confirmed": None,
        }
        entry = FrameEntry.from_jsonable(obj)
        assert entry.p1_erasure_alerts == []
        assert entry.p2_erasure_alerts == []

    def test_generate_report_p_to_e_count_zero_on_old_board_log(self) -> None:
        """古い board_log (erasure_alerts キーなし) は p_to_e_count = 0。"""
        # erasure_alerts なしの FrameEntry (= backwards compat path)
        entries = [
            FrameEntry(
                frame_idx=i, t_sec=i / 60.0,
                p1_state="stable", p2_state="stable",
                p1_confirmed=_make_empty_grid(), p2_confirmed=_make_empty_grid(),
                # p1_erasure_alerts / p2_erasure_alerts はデフォルト []
            )
            for i in range(10)
        ]
        ev = RecognitionEvaluator()
        ev.entries = entries
        report = ev.generate_report()
        assert report["p_to_e_count"] == 0
