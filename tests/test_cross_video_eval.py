"""
scripts/cross_video_eval.py のテスト。

入出力 (パス解決、評価ロジック、サマリ生成) を合成サンプルで検証する。
動画 I/O はテスト対象外 (重い & 環境依存)。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.old.cross_video_eval import (
    SAMPLE_MODE,
    VIDEO_IDS,
    build_summary,
    evaluate_on_videos,
    grid_search_per_video,
    video_paths,
)
from scripts.old.tune_weights import MatchSample
from src.old.indicators import ALL_INDICATOR_NAMES
from src.old.scorer import DEFAULT_WEIGHTS


def _empty_scores() -> dict[str, float]:
    """全指標 0 の辞書を返す。"""
    return {n: 0.0 for n in ALL_INDICATOR_NAMES}


def _make_sample(idx: int, winner: str, p1: dict[str, float], p2: dict[str, float]) -> MatchSample:
    return MatchSample(
        idx=idx, end_sec=float(idx * 60), winner=winner,
        p1_scores=p1, p2_scores=p2,
    )


# ============================
# video_paths
# ============================


class TestVideoPaths:
    def test_video_01_paths(self, tmp_path: Path):
        p = video_paths("video_01", root=tmp_path)
        assert p.video == tmp_path / "data/frames/video_01.mp4"
        assert p.boundaries_tsv == tmp_path / "data/verify/match_boundaries_v4/video_01/matches.tsv"
        assert p.winners_tsv == tmp_path / "data/verify/match_winners_v01.tsv"

    def test_video_02_paths(self, tmp_path: Path):
        p = video_paths("video_02", root=tmp_path)
        assert p.winners_tsv == tmp_path / "data/verify/match_winners_v02.tsv"

    def test_video_03_paths(self, tmp_path: Path):
        p = video_paths("video_03", root=tmp_path)
        assert p.winners_tsv == tmp_path / "data/verify/match_winners_v03.tsv"


# ============================
# evaluate_on_videos
# ============================


class TestEvaluateOnVideos:
    def test_three_videos_eval(self):
        """3 動画分のサンプルを評価し、各動画の matches / accuracy が返る。"""
        empty = _empty_scores()
        # 1P が main_chain で勝つサンプル
        s1 = {**empty, "main_chain_maturity": 0.9}
        s2 = {**empty, "main_chain_maturity": 0.1}
        all_samples = {
            "video_01": [_make_sample(1, "1P", s1, s2)],
            "video_02": [_make_sample(1, "1P", s1, s2), _make_sample(2, "1P", s1, s2)],
            "video_03": [_make_sample(1, "2P", s1, s2)],  # 不一致 (賭け 1P で 2P 勝)
        }
        weights = {**{n: 0.0 for n in ALL_INDICATOR_NAMES}, "main_chain_maturity": 1.0}
        out = evaluate_on_videos(weights, all_samples)
        assert out["video_01"]["matches"] == 1
        assert out["video_01"]["accuracy"] == pytest.approx(1.0)
        assert out["video_02"]["accuracy"] == pytest.approx(1.0)
        assert out["video_03"]["accuracy"] == pytest.approx(0.0)

    def test_empty_video(self):
        """サンプル 0 → matches=0, accuracy=0。"""
        out = evaluate_on_videos(DEFAULT_WEIGHTS, {"video_01": []})
        assert out["video_01"]["matches"] == 0
        assert out["video_01"]["accuracy"] == 0.0


# ============================
# grid_search_per_video
# ============================


class TestGridSearchPerVideo:
    def test_per_video_returns_dict_per_id(self):
        empty = _empty_scores()
        s1 = {**empty, "main_chain_maturity": 0.9}
        s2 = {**empty, "main_chain_maturity": 0.1}
        all_samples = {
            "video_01": [_make_sample(1, "1P", s1, s2)],
            "video_02": [_make_sample(1, "1P", s1, s2)],
            "video_03": [],  # 空でも DEFAULT_WEIGHTS が返る
        }
        out = grid_search_per_video(all_samples)
        assert set(out.keys()) == {"video_01", "video_02", "video_03"}
        # 空動画は DEFAULT_WEIGHTS と同じ
        assert out["video_03"] == DEFAULT_WEIGHTS


# ============================
# build_summary
# ============================


class TestBuildSummary:
    def test_summary_contains_all_videos(self):
        report = {
            "default_weights": {
                "video_01": {"matches": 30, "accuracy": 0.55},
                "video_02": {"matches": 50, "accuracy": 0.60},
                "video_03": {"matches": 40, "accuracy": 0.50},
            },
            "video_02_tuned_weights": {
                "video_01": {"matches": 30, "accuracy": 0.60},
                "video_02": {"matches": 50, "accuracy": 0.66},
                "video_03": {"matches": 40, "accuracy": 0.55},
            },
        }
        summary = build_summary(report)
        for vid in VIDEO_IDS:
            assert vid in summary
        assert "0.660" in summary
        assert "DEFAULT" in summary
        assert "v02 tuned" in summary


# ============================
# 整合性
# ============================


class TestConfig:
    def test_video_ids_three(self):
        assert len(VIDEO_IDS) == 3
        assert VIDEO_IDS == ("video_01", "video_02", "video_03")

    def test_sample_mode_midpoint(self):
        assert SAMPLE_MODE == "midpoint"
