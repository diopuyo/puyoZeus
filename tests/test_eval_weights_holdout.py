"""
scripts/eval_weights_holdout.py のロジックテスト

合成 MatchSample を使い、データ分割の再現性 / disjoint 性 / 統合パイプライン
の挙動を検証する。動画 I/O は対象外。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.old.eval_weights_holdout import (
    DEFAULT_KFOLD_K,
    DEFAULT_TRAIN_RATIO,
    OVERFIT_GAP_THRESHOLD,
    HoldoutResult,
    HoldoutSplit,
    build_report,
    evaluate_grid_holdout,
    evaluate_grid_kfold,
    kfold_split,
    load_samples_from_cache,
    random_split,
)
from scripts.old.tune_weights import MatchSample, evaluate_weights, grid_search
from src.old.indicators import ALL_INDICATOR_NAMES
from src.old.scorer import DEFAULT_WEIGHTS


# ============================
# サンプル生成ユーティリティ
# ============================


def _make_samples(n: int = 50, separation: float = 0.6) -> list[MatchSample]:
    """合成 MatchSample 列を生成する。

    奇数番=1P 勝, 偶数番=2P 勝。勝者側の指標値が高くなるよう作る。
    """
    out: list[MatchSample] = []
    empty = {n_: 0.0 for n_ in ALL_INDICATOR_NAMES}
    for i in range(1, n + 1):
        is_p1 = (i % 2 == 1)
        winner = "1P" if is_p1 else "2P"
        # 勝者は high (separation), 敗者は low (1-separation)
        high, low = separation, 1.0 - separation
        # death_risk は高いほど不利なので逆向き
        s_p1 = {
            **empty,
            **{
                k: (high if is_p1 else low)
                for k in ALL_INDICATOR_NAMES if k != "death_risk"
            },
            "death_risk": (low if is_p1 else high),
        }
        s_p2 = {
            **empty,
            **{
                k: (low if is_p1 else high)
                for k in ALL_INDICATOR_NAMES if k != "death_risk"
            },
            "death_risk": (high if is_p1 else low),
        }
        out.append(MatchSample(
            idx=i,
            end_sec=float(i * 60),
            winner=winner,
            p1_scores=s_p1,
            p2_scores=s_p2,
        ))
    return out


# ============================
# random_split
# ============================


class TestRandomSplit:
    def test_reproducible_same_seed(self):
        """同 seed で同分割になる (再現性)。"""
        samples = _make_samples(n=20)
        split_a = random_split(samples, train_ratio=0.5, seed=42)
        split_b = random_split(samples, train_ratio=0.5, seed=42)
        a_idx = [s.idx for s in split_a.train]
        b_idx = [s.idx for s in split_b.train]
        assert a_idx == b_idx

    def test_different_seed_different_split(self):
        samples = _make_samples(n=20)
        split_a = random_split(samples, train_ratio=0.5, seed=1)
        split_b = random_split(samples, train_ratio=0.5, seed=2)
        a_idx = sorted(s.idx for s in split_a.train)
        b_idx = sorted(s.idx for s in split_b.train)
        # 完全一致する確率は極めて低い
        assert a_idx != b_idx

    def test_train_test_disjoint(self):
        samples = _make_samples(n=20)
        split = random_split(samples, train_ratio=0.5, seed=42)
        train_set = {s.idx for s in split.train}
        test_set = {s.idx for s in split.test}
        assert train_set & test_set == set()
        assert len(train_set) + len(test_set) == len(samples)

    def test_train_ratio_size(self):
        samples = _make_samples(n=50)
        split = random_split(samples, train_ratio=0.5, seed=42)
        assert len(split.train) == 25
        assert len(split.test) == 25

    def test_invalid_ratio(self):
        samples = _make_samples(n=10)
        with pytest.raises(ValueError):
            random_split(samples, train_ratio=0.0, seed=42)
        with pytest.raises(ValueError):
            random_split(samples, train_ratio=1.0, seed=42)


# ============================
# kfold_split
# ============================


class TestKFoldSplit:
    def test_k_folds(self):
        samples = _make_samples(n=20)
        folds = kfold_split(samples, k=5, seed=42)
        assert len(folds) == 5

    def test_test_sets_disjoint(self):
        """各 fold の test セットが互いに disjoint。"""
        samples = _make_samples(n=20)
        folds = kfold_split(samples, k=5, seed=42)
        all_test_indices: list[int] = []
        for _, test in folds:
            test_idx = [s.idx for s in test]
            # 重複なし
            assert len(test_idx) == len(set(test_idx))
            all_test_indices.extend(test_idx)
        # 全 fold 通して各 idx は 1 回だけ test 入り
        assert len(all_test_indices) == len(set(all_test_indices))
        assert sorted(all_test_indices) == sorted(s.idx for s in samples)

    def test_train_test_disjoint_per_fold(self):
        samples = _make_samples(n=20)
        folds = kfold_split(samples, k=4, seed=42)
        for train, test in folds:
            train_idx = {s.idx for s in train}
            test_idx = {s.idx for s in test}
            assert train_idx & test_idx == set()

    def test_reproducible(self):
        samples = _make_samples(n=20)
        folds_a = kfold_split(samples, k=5, seed=7)
        folds_b = kfold_split(samples, k=5, seed=7)
        for (ta, _), (tb, _) in zip(folds_a, folds_b):
            assert [s.idx for s in ta] == [s.idx for s in tb]

    def test_invalid_k(self):
        samples = _make_samples(n=10)
        with pytest.raises(ValueError):
            kfold_split(samples, k=1, seed=42)


# ============================
# evaluate_grid_holdout / kfold
# ============================


class TestEvaluateGridHoldout:
    def test_holdout_result_fields(self):
        samples = _make_samples(n=20)
        split = random_split(samples, train_ratio=0.5, seed=42)
        result = evaluate_grid_holdout(split)
        assert isinstance(result, HoldoutResult)
        assert 0.0 <= result.train_acc <= 1.0
        assert 0.0 <= result.test_acc <= 1.0
        assert isinstance(result.best_weights, dict)

    def test_separable_data_high_acc(self):
        """完全分離可能なデータで grid が高精度を出す。"""
        samples = _make_samples(n=40, separation=0.9)
        split = random_split(samples, train_ratio=0.5, seed=42)
        result = evaluate_grid_holdout(split)
        # train/test とも高精度 (合成データなので overfit は出にくい)
        assert result.train_acc >= 0.9
        assert result.test_acc >= 0.9


class TestEvaluateGridKFold:
    def test_kfold_result_fields(self):
        samples = _make_samples(n=20)
        result = evaluate_grid_kfold(samples, k=5, seed=42)
        assert result["k"] == 5
        assert len(result["fold_results"]) == 5
        assert 0.0 <= result["test_mean"] <= 1.0
        assert result["test_std"] >= 0.0

    def test_kfold_test_means_disjoint(self):
        """K-fold の各 fold が disjoint な test を使う (sanity)。"""
        samples = _make_samples(n=20)
        result = evaluate_grid_kfold(samples, k=5, seed=42)
        n_test_total = sum(f["n_test"] for f in result["fold_results"])
        assert n_test_total == len(samples)


# ============================
# build_report
# ============================


class TestBuildReport:
    def test_overfit_flag(self):
        """train>>test なら overfit_flag が True。"""
        samples = _make_samples(n=20)
        split = HoldoutSplit(
            train=samples[:10], test=samples[10:],
            seed=42, train_ratio=0.5,
        )
        # 人為的に train_acc=1.0, test_acc=0.5 のホールドアウト結果
        holdout_result = HoldoutResult(
            train_acc=1.0, test_acc=0.5,
            best_weights=dict(DEFAULT_WEIGHTS),
            n_train=10, n_test=10,
        )
        kfold_result = {
            "k": 5, "seed": 42, "fold_results": [],
            "test_mean": 0.5, "test_std": 0.0, "train_mean": 1.0,
        }
        report = build_report(
            samples=samples,
            split=split,
            holdout_result=holdout_result,
            kfold_result=kfold_result,
            grid_full_acc=0.95,
            default_full_acc=0.6,
            default_train_acc=0.6,
            default_test_acc=0.6,
        )
        assert report["overfit_flag"] is True
        assert report["generalization_gap"] >= OVERFIT_GAP_THRESHOLD
        assert "comparison_table" in report
        assert len(report["comparison_table"]) == 4

    def test_no_overfit_flag(self):
        samples = _make_samples(n=20)
        split = HoldoutSplit(
            train=samples[:10], test=samples[10:],
            seed=42, train_ratio=0.5,
        )
        holdout_result = HoldoutResult(
            train_acc=0.7, test_acc=0.68,
            best_weights=dict(DEFAULT_WEIGHTS),
            n_train=10, n_test=10,
        )
        kfold_result = {
            "k": 5, "seed": 42, "fold_results": [],
            "test_mean": 0.68, "test_std": 0.05, "train_mean": 0.7,
        }
        report = build_report(
            samples=samples, split=split,
            holdout_result=holdout_result,
            kfold_result=kfold_result,
            grid_full_acc=0.7,
            default_full_acc=0.6,
            default_train_acc=0.6, default_test_acc=0.6,
        )
        assert report["overfit_flag"] is False


# ============================
# load_samples_from_cache
# ============================


class TestLoadSamplesFromCache:
    def test_roundtrip(self, tmp_path: Path):
        samples = _make_samples(n=5)
        # tune_weights.py 形式で書き出し
        payload = {
            "per_match_features": [
                {
                    "idx": s.idx, "end_sec": s.end_sec,
                    "winner": s.winner,
                    "p1": s.p1_scores, "p2": s.p2_scores,
                }
                for s in samples
            ],
        }
        cache = tmp_path / "cache.json"
        cache.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8",
        )
        loaded = load_samples_from_cache(cache)
        assert len(loaded) == 5
        assert loaded[0].idx == 1
        assert loaded[0].winner == "1P"
        assert loaded[0].p1_scores["main_chain_maturity"] == pytest.approx(
            samples[0].p1_scores["main_chain_maturity"]
        )
