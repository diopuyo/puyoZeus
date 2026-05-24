"""
timeline_analyzer.py のテスト。

- 短い MP4 を生成して analyze_video が走ること
- ScorePoint が EVAL_INTERVAL_SEC ごとに並ぶこと
- chain_events が ChainEventSummary であること
- to_json / from_json round-trip
- boundaries_tsv 指定/未指定で動くこと
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.sampling_config import EVAL_INTERVAL_SEC
from src.timeline_analyzer import (
    ChainEventSummary,
    MatchSegment,
    ScorePoint,
    TimelineAnalyzer,
    TimelineResult,
    detect_match_boundaries_auto,
    from_json,
    parse_boundaries_tsv,
    to_json,
)

# ============================
# テスト用フィクスチャ
# ============================


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """1080p, 5fps, 8 秒の真っ黒な動画 (盤面=全空 として読まれる)。"""
    p = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 5.0
    duration_sec = 8.0
    n_frames = int(fps * duration_sec)
    writer = cv2.VideoWriter(str(p), fourcc, fps, (1920, 1080))
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(blank)
    writer.release()
    assert p.exists()
    return p


@pytest.fixture
def boundaries_tsv(tmp_path: Path) -> Path:
    """単一試合区間 (1.0s〜7.0s) を記述した matches.tsv。"""
    p = tmp_path / "matches.tsv"
    p.write_text(
        "idx\tstart_sec\tend_sec\tduration_sec\n"
        "1\t1.0\t7.0\t6.0\n",
        encoding="utf-8",
    )
    return p


# ============================
# TSV パース
# ============================


class TestParseBoundariesTsv:
    def test_parse_simple(self, boundaries_tsv: Path) -> None:
        rows = parse_boundaries_tsv(boundaries_tsv)
        assert len(rows) == 1
        idx, s, e = rows[0]
        assert idx == 1
        assert s == 1.0
        assert e == 7.0

    def test_parse_skips_header(self, tmp_path: Path) -> None:
        p = tmp_path / "m.tsv"
        p.write_text(
            "idx\tstart_sec\tend_sec\n"
            "1\t10.5\t20.5\n"
            "2\t30.0\t45.0\n",
            encoding="utf-8",
        )
        rows = parse_boundaries_tsv(p)
        assert rows == [(1, 10.5, 20.5), (2, 30.0, 45.0)]

    def test_parse_skips_blank_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "m.tsv"
        p.write_text(
            "idx\tstart_sec\tend_sec\n"
            "\n"
            "1\t1.0\t2.0\n",
            encoding="utf-8",
        )
        rows = parse_boundaries_tsv(p)
        assert rows == [(1, 1.0, 2.0)]


# ============================
# 自動境界検出 (失敗系)
# ============================


class TestDetectAutoBoundaries:
    def test_returns_empty_for_missing_video(self, tmp_path: Path) -> None:
        # MatchStateDetector は load_default に calibration が必要だが、
        # 不存在動画なら計算前に [] を返す。ダミーで dummy detector を渡す。
        class _DummyDetector:
            def detect(self, frame):  # noqa: D401
                from src.match_state import MatchDetectResult, MatchState
                return MatchDetectResult(
                    state=MatchState.NOT_IN_MATCH,
                    bg_value=0.0, bg_saturation=0.0, samples=0,
                )

        intervals = detect_match_boundaries_auto(
            tmp_path / "nope.mp4", _DummyDetector(),  # type: ignore[arg-type]
        )
        assert intervals == []


# ============================
# TimelineAnalyzer.analyze_video
# ============================


class TestAnalyzeVideoWithBoundaries:
    def test_runs_without_error(
        self, synthetic_video: Path, boundaries_tsv: Path,
    ) -> None:
        ana = TimelineAnalyzer()
        res = ana.analyze_video(synthetic_video, boundaries_tsv=boundaries_tsv)
        assert isinstance(res, TimelineResult)
        assert res.video_path == str(synthetic_video)
        assert res.duration_sec > 0

    def test_match_segment_created(
        self, synthetic_video: Path, boundaries_tsv: Path,
    ) -> None:
        ana = TimelineAnalyzer()
        res = ana.analyze_video(synthetic_video, boundaries_tsv=boundaries_tsv)
        assert len(res.match_segments) == 1
        seg = res.match_segments[0]
        assert isinstance(seg, MatchSegment)
        assert seg.match_idx == 1
        assert seg.start_sec == 1.0
        assert seg.end_sec == 7.0
        assert seg.duration_sec == pytest.approx(6.0)

    def test_score_points_at_eval_interval(
        self, synthetic_video: Path, boundaries_tsv: Path,
    ) -> None:
        ana = TimelineAnalyzer()
        res = ana.analyze_video(synthetic_video, boundaries_tsv=boundaries_tsv)
        seg = res.match_segments[0]
        assert len(seg.score_timeline) >= 2
        # ScorePoint の t_sec 増分が EVAL_INTERVAL_SEC に近いこと
        for prev, cur in zip(seg.score_timeline, seg.score_timeline[1:]):
            delta = cur.t_sec - prev.t_sec
            assert delta == pytest.approx(EVAL_INTERVAL_SEC, abs=1e-6)
        # 真っ黒盤面なので score=0 (両方空)
        for p in seg.score_timeline:
            assert isinstance(p, ScorePoint)
            assert p.score == 0.0

    def test_chain_events_are_summaries(
        self, synthetic_video: Path, boundaries_tsv: Path,
    ) -> None:
        ana = TimelineAnalyzer()
        res = ana.analyze_video(synthetic_video, boundaries_tsv=boundaries_tsv)
        seg = res.match_segments[0]
        for ev in seg.chain_events:
            assert isinstance(ev, ChainEventSummary)
            assert ev.side in {"1P", "2P"}
        # 真っ黒の動画は連鎖を発火させない
        assert len(seg.chain_events) == 0

    def test_short_segment_excluded(
        self, synthetic_video: Path, tmp_path: Path,
    ) -> None:
        """5 秒未満の区間は除外されること。"""
        tsv = tmp_path / "tiny.tsv"
        tsv.write_text(
            "idx\tstart_sec\tend_sec\n"
            "1\t1.0\t3.0\n",
            encoding="utf-8",
        )
        ana = TimelineAnalyzer()
        res = ana.analyze_video(synthetic_video, boundaries_tsv=tsv)
        assert res.match_segments == []


class TestAnalyzeVideoWithoutBoundaries:
    def test_runs_when_no_boundaries_no_calib(
        self, synthetic_video: Path,
    ) -> None:
        """boundaries 未指定 + calib も未指定なら空区間で正常終了する。"""
        ana = TimelineAnalyzer()
        res = ana.analyze_video(synthetic_video, boundaries_tsv=None)
        assert isinstance(res, TimelineResult)
        # match_state_detector が load 不能 → 区間は空
        assert res.match_segments == []


# ============================
# JSON シリアライザ
# ============================


class TestJsonRoundTrip:
    def test_round_trip_preserves_segments(
        self,
        synthetic_video: Path,
        boundaries_tsv: Path,
        tmp_path: Path,
    ) -> None:
        ana = TimelineAnalyzer()
        res = ana.analyze_video(synthetic_video, boundaries_tsv=boundaries_tsv)
        out = tmp_path / "tl.json"
        to_json(res, out)
        assert out.exists()

        loaded = from_json(out)
        assert loaded.video_path == res.video_path
        assert loaded.duration_sec == pytest.approx(res.duration_sec)
        assert loaded.fps == pytest.approx(res.fps)
        assert len(loaded.match_segments) == len(res.match_segments)

        orig_seg = res.match_segments[0]
        new_seg = loaded.match_segments[0]
        assert new_seg.match_idx == orig_seg.match_idx
        assert new_seg.start_sec == pytest.approx(orig_seg.start_sec)
        assert new_seg.end_sec == pytest.approx(orig_seg.end_sec)
        assert len(new_seg.score_timeline) == len(orig_seg.score_timeline)

    def test_round_trip_score_point_breakdown(
        self,
        synthetic_video: Path,
        boundaries_tsv: Path,
        tmp_path: Path,
    ) -> None:
        ana = TimelineAnalyzer()
        res = ana.analyze_video(synthetic_video, boundaries_tsv=boundaries_tsv)
        out = tmp_path / "tl.json"
        to_json(res, out)
        loaded = from_json(out)
        if not loaded.match_segments[0].score_timeline:
            pytest.skip("評価点が無いケースは round-trip 検証不要")
        orig_p = res.match_segments[0].score_timeline[0]
        new_p = loaded.match_segments[0].score_timeline[0]
        assert new_p.t_sec == pytest.approx(orig_p.t_sec)
        assert new_p.score == pytest.approx(orig_p.score)
        # breakdown のキーが保たれる
        assert set(new_p.breakdown.keys()) == set(orig_p.breakdown.keys())


# ============================
# meta 情報
# ============================


class TestMeta:
    def test_meta_contains_intervals(
        self, synthetic_video: Path, boundaries_tsv: Path,
    ) -> None:
        ana = TimelineAnalyzer()
        res = ana.analyze_video(synthetic_video, boundaries_tsv=boundaries_tsv)
        assert "board_interval_sec" in res.meta
        assert "eval_interval_sec" in res.meta
        # 数値形式
        assert float(res.meta["eval_interval_sec"]) == EVAL_INTERVAL_SEC
