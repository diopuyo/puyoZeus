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
    AVG_PUYO_COUNT_CRITICAL_RATIO,
    FrameEntry,
    RecognitionEvaluator,
    SEVERITY_CRITICAL,
    compute_avg_puyo_count,
    judge_cycle,
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


# ============================
# C1: compute_avg_puyo_count
# ============================

class TestComputeAvgPuyoCount:
    def test_all_empty_boards(self) -> None:
        """全 STABLE フレームが空盤面 → avg = 0。"""
        entries = [
            {"frame_idx": i, "t_sec": i / 60.0,
             "p1_state": "stable", "p2_state": "stable",
             "p1_confirmed": _make_empty_grid(),
             "p2_confirmed": _make_empty_grid()}
            for i in range(5)
        ]
        result = compute_avg_puyo_count(entries)
        assert result["avg_puyo_count_per_stable_frame"] == 0.0
        assert result["n_stable_frames"] == 5

    def test_known_puyo_count(self) -> None:
        """1P に 10 個, 2P に 6 個 → 合算 avg = 16.0 / 1 frame = 16.0。"""
        grid_p1 = _make_grid_with_color(COLOR_RED, 10)
        grid_p2 = _make_grid_with_color(COLOR_BLUE, 6)
        entries = [
            {"frame_idx": 0, "t_sec": 0.0,
             "p1_state": "stable", "p2_state": "stable",
             "p1_confirmed": grid_p1, "p2_confirmed": grid_p2}
        ]
        result = compute_avg_puyo_count(entries)
        assert result["avg_puyo_count_per_stable_frame"] == 16.0
        assert result["n_stable_frames"] == 1

    def test_non_stable_frames_excluded(self) -> None:
        """non-stable フレームは集計から除外される。"""
        entries = [
            {"frame_idx": 0, "t_sec": 0.0,
             "p1_state": "chain", "p2_state": "chain",
             "p1_confirmed": None, "p2_confirmed": None},
            {"frame_idx": 1, "t_sec": 1.0,
             "p1_state": "stable", "p2_state": "stable",
             "p1_confirmed": _make_grid_with_color(COLOR_RED, 8),
             "p2_confirmed": _make_empty_grid()},
        ]
        result = compute_avg_puyo_count(entries)
        assert result["n_stable_frames"] == 1
        assert result["avg_puyo_count_per_stable_frame"] == 8.0

    def test_no_stable_frames_returns_zero(self) -> None:
        """STABLE フレームゼロ → avg=0, n=0。"""
        entries = [
            {"frame_idx": 0, "t_sec": 0.0,
             "p1_state": "menu", "p2_state": "menu",
             "p1_confirmed": None, "p2_confirmed": None}
        ]
        result = compute_avg_puyo_count(entries)
        assert result["avg_puyo_count_per_stable_frame"] == 0.0
        assert result["n_stable_frames"] == 0


# ============================
# C1: generate_report avg_puyo_count 統合
# ============================

class TestGenerateReportAvgPuyo:
    def test_report_includes_avg_puyo_count(self) -> None:
        """generate_report() の summary に avg_puyo_count_per_stable_frame が入る。"""
        ev = RecognitionEvaluator()
        ev.entries = [
            FrameEntry(
                frame_idx=i, t_sec=i / 60.0,
                p1_state="stable", p2_state="stable",
                p1_confirmed=_make_grid_with_color(COLOR_RED, 10),
                p2_confirmed=_make_grid_with_color(COLOR_BLUE, 5),
            )
            for i in range(3)
        ]
        report = ev.generate_report()
        summary = report["summary"]
        assert "avg_puyo_count_per_stable_frame" in summary
        assert summary["avg_puyo_count_per_stable_frame"] == 15.0
        assert summary["n_stable_frames"] == 3

    def test_report_reject_on_low_avg_puyo_ratio(self) -> None:
        """baseline_avg_puyo_count 指定時に ratio < 0.85 → verdict = REJECT。"""
        ev = RecognitionEvaluator()
        # 盤面に 5 個だけ → avg = 5.0
        ev.entries = [
            FrameEntry(
                frame_idx=i, t_sec=i / 60.0,
                p1_state="stable", p2_state="stable",
                p1_confirmed=_make_grid_with_color(COLOR_RED, 5),
                p2_confirmed=_make_empty_grid(),
            )
            for i in range(5)
        ]
        # baseline = 20 → ratio = 5/20 = 0.25 < 0.85
        report = ev.generate_report(baseline_avg_puyo_count=20.0)
        assert report["verdict"] == "REJECT"
        summary = report["summary"]
        assert summary["avg_puyo_count_ratio"] is not None
        assert summary["avg_puyo_count_ratio"] < AVG_PUYO_COUNT_CRITICAL_RATIO

    def test_report_no_reject_on_high_avg_puyo_ratio(self) -> None:
        """ratio >= 0.85 なら avg_puyo 起因の REJECT はなし。"""
        ev = RecognitionEvaluator()
        ev.entries = [
            FrameEntry(
                frame_idx=i, t_sec=i / 60.0,
                p1_state="stable", p2_state="stable",
                p1_confirmed=_make_grid_with_color(COLOR_RED, 18),
                p2_confirmed=_make_empty_grid(),
            )
            for i in range(3)
        ]
        # baseline = 20 → ratio = 18/20 = 0.90 >= 0.85
        report = ev.generate_report(baseline_avg_puyo_count=20.0)
        # critical が少なければ REJECT にならない
        assert report["verdict"] != "REJECT"

    def test_report_no_baseline_skips_ratio_check(self) -> None:
        """baseline_avg_puyo_count=None → ratio check skip、 avg_puyo_ratio=None。"""
        ev = RecognitionEvaluator()
        ev.entries = [
            FrameEntry(
                frame_idx=0, t_sec=0.0,
                p1_state="stable", p2_state="stable",
                p1_confirmed=_make_empty_grid(),
                p2_confirmed=_make_empty_grid(),
            )
        ]
        report = ev.generate_report()
        assert report["summary"]["avg_puyo_count_ratio"] is None


# ============================
# C3: judge_cycle
# ============================

class TestJudgeCycle:
    def _make_stats(
        self,
        critical: int = 0,
        p_to_e: int = 0,
        avg_puyo: float = 20.0,
    ) -> dict:
        return {
            "summary": {
                "critical": critical,
                "avg_puyo_count_per_stable_frame": avg_puyo,
            },
            "p_to_e_count": p_to_e,
        }

    def test_auto_accept_all_good(self) -> None:
        """全条件クリア → AUTO_ACCEPT_PROVISIONAL。"""
        base = self._make_stats(critical=5, p_to_e=10, avg_puyo=20.0)
        cand = self._make_stats(critical=5, p_to_e=9, avg_puyo=19.0)
        assert judge_cycle(base, cand) == "AUTO_ACCEPT_PROVISIONAL"

    def test_auto_reject_low_avg_puyo(self) -> None:
        """avg_puyo_ratio < 0.85 → AUTO_REJECT。"""
        base = self._make_stats(avg_puyo=20.0)
        cand = self._make_stats(avg_puyo=10.0)  # ratio = 0.5
        assert judge_cycle(base, cand) == "AUTO_REJECT"

    def test_auto_reject_critical_surge(self) -> None:
        """critical > baseline × 1.10 → AUTO_REJECT。"""
        base = self._make_stats(critical=100)
        cand = self._make_stats(critical=120, avg_puyo=20.0)  # +20% > +10%
        assert judge_cycle(base, cand) == "AUTO_REJECT"

    def test_auto_reject_p_to_e_surge(self) -> None:
        """p_to_e > baseline × 1.20 → AUTO_REJECT。"""
        base = self._make_stats(p_to_e=100, avg_puyo=20.0)
        cand = self._make_stats(p_to_e=130, avg_puyo=19.0)  # +30% > +20%
        assert judge_cycle(base, cand) == "AUTO_REJECT"

    def test_needs_review_critical_slightly_increased(self) -> None:
        """critical が baseline+3 (= ACCEPT_DELTA=2 超) かつ REJECT 閾値未満で NEEDS_REVIEW。

        REJECT 判定 = baseline_critical × 1.10 超。
        baseline=100 なら REJECT 閾値 = 110、 ACCEPT 上限 = 102。
        critical=105 は 102 超かつ 110 未満 → NEEDS_REVIEW。
        """
        base = self._make_stats(critical=100, avg_puyo=20.0)
        # +5 > ACCEPT_DELTA (2)、 かつ < REJECT (100 × 1.10 = 110)
        cand = self._make_stats(critical=105, avg_puyo=19.5)
        result = judge_cycle(base, cand)
        assert result == "NEEDS_REVIEW"


# ============================
# THREE_WAY_SUDDEN_DROP (2026-06-03): fail-silent 盲点炙り出し
# ============================


def _make_raw_grid(color: int, count: int) -> list[list[int]]:
    """下段から count 個 cell を color で埋めた raw grid (CNN / HSV 用)."""
    return _make_grid_with_color(color, count)


def _make_frame_entry_with_raw(
    fi: int,
    p1_confirmed: list[list[int]],
    p1_raw_cnn: list[list[int]] | None,
    p1_raw_hsv: list[list[int]] | None,
    p1_state: str = "stable",
) -> FrameEntry:
    """raw_cnn / raw_hsv 付きの FrameEntry を生成する (1P 専用ヘルパー)."""
    return FrameEntry(
        frame_idx=fi,
        t_sec=fi / 60.0,
        p1_state=p1_state,
        p2_state="stable",
        p1_confirmed=p1_confirmed,
        p2_confirmed=_make_empty_grid(),
        p1_raw_cnn_board=p1_raw_cnn,
        p1_raw_hsv_board=p1_raw_hsv,
    )


class TestThreeWaySuddenDrop:
    """check_three_way_sudden_drop のユニットテスト."""

    # --- 発火ケース ---

    def test_drop_gte_threshold_fires_critical(self) -> None:
        """3 者一致ぷよ数が -8 以上減少したとき CRITICAL を発火."""
        # frame 0: 3 者一致 20 個 (RED で統一)
        grid_20 = _make_raw_grid(COLOR_RED, 20)
        # frame 60: 3 者一致 10 個 (diff = -10)
        grid_10 = _make_raw_grid(COLOR_RED, 10)
        ev = RecognitionEvaluator()
        ev.entries = [
            _make_frame_entry_with_raw(0, grid_20, grid_20, grid_20),
            _make_frame_entry_with_raw(60, grid_10, grid_10, grid_10),
        ]
        vs = ev.check_three_way_sudden_drop("1P")
        assert len(vs) == 1
        assert vs[0].metric == "three_way_sudden_drop"
        assert vs[0].severity == SEVERITY_CRITICAL
        assert vs[0].extra["diff"] == -10
        assert vs[0].extra["chain_intervened"] is False

    def test_drop_exactly_threshold_fires(self) -> None:
        """diff == -THREE_WAY_DROP_THRESHOLD (= -8) ちょうどでも発火する."""
        from src.recognition_evaluator import THREE_WAY_DROP_THRESHOLD
        grid_a = _make_raw_grid(COLOR_RED, 20)
        grid_b = _make_raw_grid(COLOR_RED, 20 - THREE_WAY_DROP_THRESHOLD)
        ev = RecognitionEvaluator()
        ev.entries = [
            _make_frame_entry_with_raw(0, grid_a, grid_a, grid_a),
            _make_frame_entry_with_raw(60, grid_b, grid_b, grid_b),
        ]
        vs = ev.check_three_way_sudden_drop("1P")
        assert len(vs) == 1
        assert vs[0].extra["diff"] == -THREE_WAY_DROP_THRESHOLD

    # --- 非発火ケース: chain 介在 ---

    def test_chain_intervened_no_fire(self) -> None:
        """prev_stable → cur_stable の間に chain が介在する場合は発火しない."""
        grid_20 = _make_raw_grid(COLOR_RED, 20)
        grid_10 = _make_raw_grid(COLOR_RED, 10)
        ev = RecognitionEvaluator()
        ev.entries = [
            # frame 0: stable (20 個)
            _make_frame_entry_with_raw(0, grid_20, grid_20, grid_20),
            # frame 1: chain が介在
            FrameEntry(
                frame_idx=1, t_sec=1 / 60.0,
                p1_state="chain", p2_state="stable",
                p1_confirmed=grid_20, p2_confirmed=_make_empty_grid(),
                p1_raw_cnn_board=grid_20, p1_raw_hsv_board=grid_20,
            ),
            # frame 60: stable (10 個)
            _make_frame_entry_with_raw(60, grid_10, grid_10, grid_10),
        ]
        vs = ev.check_three_way_sudden_drop("1P")
        assert len(vs) == 0, "chain 介在時は発火しないはず"

    # --- 非発火ケース: ojama_fall 介在 ---

    def test_ojama_fall_intervened_no_fire(self) -> None:
        """prev_stable → cur_stable の間に ojama_fall が介在する場合は発火しない."""
        grid_20 = _make_raw_grid(COLOR_RED, 20)
        grid_10 = _make_raw_grid(COLOR_RED, 10)
        ev = RecognitionEvaluator()
        ev.entries = [
            _make_frame_entry_with_raw(0, grid_20, grid_20, grid_20),
            FrameEntry(
                frame_idx=1, t_sec=1 / 60.0,
                p1_state="ojama_fall", p2_state="stable",
                p1_confirmed=grid_20, p2_confirmed=_make_empty_grid(),
                p1_raw_cnn_board=grid_20, p1_raw_hsv_board=grid_20,
            ),
            _make_frame_entry_with_raw(60, grid_10, grid_10, grid_10),
        ]
        vs = ev.check_three_way_sudden_drop("1P")
        assert len(vs) == 0, "ojama_fall 介在時は発火しないはず"

    # --- 非発火ケース: tsumo_fall 介在 ---

    def test_tsumo_fall_intervened_no_fire(self) -> None:
        """prev_stable → cur_stable の間に tsumo_fall が介在する場合は発火しない."""
        grid_20 = _make_raw_grid(COLOR_RED, 20)
        grid_10 = _make_raw_grid(COLOR_RED, 10)
        ev = RecognitionEvaluator()
        ev.entries = [
            _make_frame_entry_with_raw(0, grid_20, grid_20, grid_20),
            FrameEntry(
                frame_idx=1, t_sec=1 / 60.0,
                p1_state="tsumo_fall", p2_state="stable",
                p1_confirmed=grid_20, p2_confirmed=_make_empty_grid(),
                p1_raw_cnn_board=grid_20, p1_raw_hsv_board=grid_20,
            ),
            _make_frame_entry_with_raw(60, grid_10, grid_10, grid_10),
        ]
        vs = ev.check_three_way_sudden_drop("1P")
        assert len(vs) == 0, "tsumo_fall 介在時は発火しないはず"

    # --- 非発火ケース: 3 者不一致 ---

    def test_three_way_mismatch_no_fire(self) -> None:
        """raw_cnn / raw_hsv / confirmed が一致しない cell が多い場合は発火しない.

        3 者一致数が少なければ diff も -8 未満になるため発火しない。
        """
        grid_red = _make_raw_grid(COLOR_RED, 20)
        grid_blue = _make_raw_grid(COLOR_BLUE, 20)  # CNN は BLUE
        # confirmed = RED 20 個、 CNN = BLUE (3 者不一致) → 3 者一致数 = 0
        ev = RecognitionEvaluator()
        ev.entries = [
            _make_frame_entry_with_raw(0, grid_red, grid_blue, grid_red),
            _make_frame_entry_with_raw(60, grid_red, grid_blue, grid_red),
        ]
        vs = ev.check_three_way_sudden_drop("1P")
        assert len(vs) == 0, "3 者不一致では 3 者一致数 = 0 で発火しないはず"

    # --- 非発火ケース: raw が None (古い board_log) ---

    def test_raw_none_no_fire(self) -> None:
        """raw_cnn / raw_hsv が None の場合 (古い board_log) は評価不能で発火しない."""
        grid = _make_raw_grid(COLOR_RED, 20)
        ev = RecognitionEvaluator()
        ev.entries = [
            # raw は None (= 古い board_log 形式)
            _make_frame_entry_with_raw(0, grid, None, None),
            _make_frame_entry_with_raw(60, _make_empty_grid(), None, None),
        ]
        vs = ev.check_three_way_sudden_drop("1P")
        assert len(vs) == 0, "raw = None のときは評価不能で発火しないはず"

    # --- 非発火ケース: 閾値未満の減少 ---

    def test_small_drop_below_threshold_no_fire(self) -> None:
        """diff = -7 (< THREE_WAY_DROP_THRESHOLD = -8) は発火しない."""
        from src.recognition_evaluator import THREE_WAY_DROP_THRESHOLD
        grid_a = _make_raw_grid(COLOR_RED, 20)
        grid_b = _make_raw_grid(COLOR_RED, 20 - (THREE_WAY_DROP_THRESHOLD - 1))
        ev = RecognitionEvaluator()
        ev.entries = [
            _make_frame_entry_with_raw(0, grid_a, grid_a, grid_a),
            _make_frame_entry_with_raw(60, grid_b, grid_b, grid_b),
        ]
        vs = ev.check_three_way_sudden_drop("1P")
        assert len(vs) == 0

    # --- backwards compat: from_jsonable で raw フィールドが読み込まれる ---

    def test_from_jsonable_reads_raw_boards(self) -> None:
        """from_jsonable() が p1_raw_cnn_board / p1_raw_hsv_board を正しく読む."""
        raw_grid = _make_raw_grid(COLOR_RED, 5)
        obj = {
            "frame_idx": 10,
            "t_sec": 0.17,
            "p1_state": "stable",
            "p2_state": "stable",
            "p1_confirmed": raw_grid,
            "p2_confirmed": None,
            "p1_raw_cnn_board": raw_grid,
            "p2_raw_cnn_board": None,
            "p1_raw_hsv_board": raw_grid,
            "p2_raw_hsv_board": None,
        }
        entry = FrameEntry.from_jsonable(obj)
        assert entry.p1_raw_cnn_board == raw_grid
        assert entry.p1_raw_hsv_board == raw_grid
        assert entry.p2_raw_cnn_board is None

    def test_from_jsonable_no_raw_fields_backward_compat(self) -> None:
        """古い board_log (raw フィールドなし) では None が入る (backwards compat)."""
        obj = {
            "frame_idx": 50,
            "t_sec": 0.83,
            "p1_state": "stable",
            "p2_state": "stable",
            "p1_confirmed": None,
            "p2_confirmed": None,
        }
        entry = FrameEntry.from_jsonable(obj)
        assert entry.p1_raw_cnn_board is None
        assert entry.p1_raw_hsv_board is None
        assert entry.p2_raw_cnn_board is None
        assert entry.p2_raw_hsv_board is None
