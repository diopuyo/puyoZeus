"""
scripts/tune_weights.py の dry-run / ロジックテスト

入出力 (TSV ロード、per-indicator 評価、grid search) の正しさを検証する。
動画 I/O はテスト対象外 (重い & 環境依存) で、合成サンプルでロジックを検証。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.old.tune_weights import (
    GRID_SEARCH_TARGETS,
    WEIGHT_GRID,
    MatchSample,
    evaluate_weights,
    grid_search,
    load_winners,
    per_indicator_accuracy,
)
from src.old.indicators import ALL_INDICATOR_NAMES
from src.old.scorer import DEFAULT_WEIGHTS


# ============================
# load_winners
# ============================


class TestLoadWinners:
    def test_basic_parse(self, tmp_path: Path):
        tsv = tmp_path / "winners.tsv"
        tsv.write_text(
            "idx\tstart_sec\tend_sec\twinner\textra\n"
            "1\t10\t60\t1P\tx\n"
            "2\t70\t110\t2P\tx\n",
            encoding="utf-8",
        )
        out = load_winners(tsv)
        assert out == {1: (60.0, "1P"), 2: (110.0, "2P")}

    def test_skip_invalid_winner(self, tmp_path: Path):
        tsv = tmp_path / "winners.tsv"
        tsv.write_text(
            "idx\tstart\tend\twinner\n"
            "1\t10\t60\tDRAW\n"
            "2\t70\t110\t1P\n",
            encoding="utf-8",
        )
        out = load_winners(tsv)
        assert out == {2: (110.0, "1P")}


# ============================
# per_indicator_accuracy
# ============================


def _make_samples(records: list[tuple[str, dict[str, float], dict[str, float]]]) -> list[MatchSample]:
    """テスト用 MatchSample 列を生成する。"""
    return [
        MatchSample(idx=i, end_sec=float(i * 60), winner=w, p1_scores=p1, p2_scores=p2)
        for i, (w, p1, p2) in enumerate(records, start=1)
    ]


class TestPerIndicatorAccuracy:
    def test_perfect_indicator(self):
        """ある指標が常に勝者と一致 → 一致率 1.0。"""
        name = "main_chain_maturity"
        # 全指標を 0 で初期化、name のみ正しい方向に設定
        empty = {n: 0.0 for n in ALL_INDICATOR_NAMES}
        samples = _make_samples([
            ("1P", {**empty, name: 0.8}, {**empty, name: 0.2}),
            ("2P", {**empty, name: 0.1}, {**empty, name: 0.9}),
            ("1P", {**empty, name: 0.7}, {**empty, name: 0.3}),
        ])
        acc = per_indicator_accuracy(samples)
        assert acc[name] == pytest.approx(1.0)

    def test_death_risk_inverted(self):
        """death_risk は高い方が不利として扱う。"""
        empty = {n: 0.0 for n in ALL_INDICATOR_NAMES}
        # 1P の death_risk が高い → 2P が勝つはず
        samples = _make_samples([
            ("2P", {**empty, "death_risk": 0.9}, {**empty, "death_risk": 0.1}),
            ("1P", {**empty, "death_risk": 0.1}, {**empty, "death_risk": 0.8}),
        ])
        acc = per_indicator_accuracy(samples)
        assert acc["death_risk"] == pytest.approx(1.0)

    def test_empty_samples(self):
        assert per_indicator_accuracy([]) == {}


# ============================
# evaluate_weights / grid_search
# ============================


class TestEvaluateWeights:
    def test_default_weights_basic(self):
        """DEFAULT_WEIGHTS で 1P 圧勝サンプルが正しく分類される。"""
        empty = {n: 0.0 for n in ALL_INDICATOR_NAMES}
        # 1P は全指標 0.8、2P は 0.2
        s1 = {**empty}
        s2 = {**empty}
        for n in ALL_INDICATOR_NAMES:
            if n == "death_risk":
                s1[n] = 0.2  # 1P の死亡リスクは低い
                s2[n] = 0.8
            else:
                s1[n] = 0.8
                s2[n] = 0.2
        samples = _make_samples([("1P", s1, s2)])
        acc = evaluate_weights(samples, DEFAULT_WEIGHTS)
        assert acc == 1.0

    def test_zero_weights_zero_accuracy(self):
        """全重み 0 → diff=0 で全試合スキップ → 0/total = 0。"""
        zero_w = {n: 0.0 for n in ALL_INDICATOR_NAMES}
        empty = {n: 0.5 for n in ALL_INDICATOR_NAMES}
        samples = _make_samples([("1P", empty, empty)])
        acc = evaluate_weights(samples, zero_w)
        assert acc == 0.0


class TestGridSearch:
    def test_finds_better_than_zero(self):
        """grid search で 0 重みより良い結果を見つけられる。"""
        empty = {n: 0.0 for n in ALL_INDICATOR_NAMES}
        s1 = {**empty, "main_chain_maturity": 0.9}
        s2 = {**empty, "main_chain_maturity": 0.1}
        samples = _make_samples([
            ("1P", s1, s2),
            ("1P", s1, s2),
        ])
        zero_w = {n: 0.0 for n in ALL_INDICATOR_NAMES}
        best_w, best_acc = grid_search(samples, zero_w)
        # main_chain_maturity が GRID_SEARCH_TARGETS に含まれているか確認
        assert "main_chain_maturity" in GRID_SEARCH_TARGETS
        # 最良重みで >0 の精度が出ること
        assert best_acc > 0.0

    def test_grid_does_not_break_unsearched_weights(self):
        """grid search 対象外の指標重みは base_weights のまま保たれる。"""
        empty = {n: 0.0 for n in ALL_INDICATOR_NAMES}
        s1 = {**empty, "field_efficiency": 1.0}
        s2 = {**empty, "field_efficiency": 0.0}
        samples = _make_samples([("1P", s1, s2)])
        base = dict(DEFAULT_WEIGHTS)
        # field_efficiency は GRID_SEARCH_TARGETS に含まれない
        assert "field_efficiency" not in GRID_SEARCH_TARGETS
        best_w, _ = grid_search(samples, base)
        assert best_w["field_efficiency"] == base["field_efficiency"]


# ============================
# 整合性
# ============================


class TestConfig:
    def test_grid_targets_subset_of_indicators(self):
        for name in GRID_SEARCH_TARGETS:
            assert name in ALL_INDICATOR_NAMES

    def test_weight_grid_includes_zero(self):
        assert 0.0 in WEIGHT_GRID
