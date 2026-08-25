"""主因表示の除外リスト (ATTRIBUTION_EXCLUDED_INDICATORS) の回帰テスト。

## 背景 (2026-08-11 ロードマップ Phase1-3)
`scripts/visualize_advantage_overlay.py` の `_score_advantage()` が組み立てる
「主因」欄は **|差分| の大きい順** で上位3件を選ぶだけで、実際にモデルの予測
(p1/adv) へどれだけ寄与しているかは見ていない。 このため勝敗と無相関
(AUC 0.5000、2026-08-09/11 実測 73,416 ペア) な指標でも、たまたま差分値が
大きいと主因1位に選ばれてしまっていた (デモ実測「期待火力K1差 +0.64」が
主因1位表示された事例)。

本テストは以下を検証する:
  1. 除外リストに載る指標は既定で主因候補から弾かれる
  2. 判定値 (adv/p1) は除外の有無で **完全不変** (表示だけの修正である証明)
  3. デバッグフラグ (attribution_exclude=()) で除外前の全候補に戻せる
  4. 除外リストが実際の呼出経路 (HeavyAdvCache / _fresh_trackers / generate /
     CLI) に配線されていること (dead code 化防止)
"""
from __future__ import annotations

import inspect
import types
from pathlib import Path

import numpy as np
import pytest

from src.board import Board
from src.indicators_v2 import IndicatorV2Value
from src.production_config import ATTRIBUTION_EXCLUDED_INDICATORS

import scripts.visualize_advantage_overlay as vao


class _FakeModel:
    """`_score_advantage` が要求する最小 API (predict_proba + _puyo_feature_cols)
    だけを持つ偽モデル。何を渡されても同じ p1 を返す (主因の選定=|差分|順ソートは
    adv/p1 の実値と無関係にテストできるようにするため)。
    """

    _puyo_feature_cols = (
        "board_color_puyo_total", "expected_fire_k1", "expected_fire_k2",
        "saturated_chain_count", "current_max_chain",
    )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.array([[0.4, 0.6]])


def _fake_snap() -> types.SimpleNamespace:
    """`_score_advantage` が参照する snap の3属性だけを持つ最小スタブ。"""
    return types.SimpleNamespace(net_balance_capped=0, forecast_p1=0, forecast_p2=0)


@pytest.fixture()
def rigged_boards_and_model(monkeypatch):
    """expected_fire_k1/k2・saturated_chain_count の差分を人為的に極端に大きくする。

    実データでは AUC 0.5000 (無情報) だが、「差分が大きいだけで主因に選ばれる」
    現行ロジックの欠陥を決定論的に再現するための細工 (2026-08-09/11 実測を
    モデル呼び出しなしで高速に再現するため、実指標関数はモック化する)。
    """
    b1 = Board()
    b1.set(6, 0, 1)
    b1.set(6, 1, 1)  # board_color_puyo_total の diff を作る (excluded 指標より小さめ)
    b2 = Board()

    monkeypatch.setattr(
        vao.iv, "saturated_chain_count",
        lambda b: IndicatorV2Value(score=(1.0 if b is b1 else 0.0), raw=0.0),
    )

    def _fake_expected_fire(f1, f2, b1_, b2_, cols):
        if "expected_fire_k1" in cols:
            f1["expected_fire_k1"] = 1.0
            f2["expected_fire_k1"] = 0.0
        if "expected_fire_k2" in cols:
            f1["expected_fire_k2"] = 1.0
            f2["expected_fire_k2"] = 0.0

    monkeypatch.setattr(vao, "_fill_expected_fire_candidate", _fake_expected_fire)
    return b1, b2


class TestAttributionExclusionFiltersDrivers:
    """除外リストが実際に主因候補を弾くこと。"""

    def test_excluded_indicators_absent_from_default_drivers(
        self, rigged_boards_and_model,
    ) -> None:
        b1, b2 = rigged_boards_and_model
        _, _, drivers = vao._score_advantage(_FakeModel(), b1, b2, _fake_snap())
        names = {c for c, _ in drivers}
        hit = names & set(ATTRIBUTION_EXCLUDED_INDICATORS)
        assert not hit, f"除外対象のはずの指標が主因に出た: {hit}"

    def test_excluded_indicators_would_have_topped_without_exclusion(
        self, rigged_boards_and_model,
    ) -> None:
        """細工が効いている (=除外しなければ実際に主因1位に来る) ことの前提確認。

        これが成立しないと「除外して何も変わらなかった」だけの空テストになる。
        """
        b1, b2 = rigged_boards_and_model
        _, _, drivers = vao._score_advantage(
            _FakeModel(), b1, b2, _fake_snap(), attribution_exclude=())
        assert drivers, "主因候補が空 (細工が効いていない)"
        assert drivers[0][0] in ATTRIBUTION_EXCLUDED_INDICATORS

    def test_debug_flag_restores_excluded_candidates(
        self, rigged_boards_and_model,
    ) -> None:
        """attribution_exclude=() (デバッグ用途) で除外前候補が復活すること。"""
        b1, b2 = rigged_boards_and_model
        _, _, drivers_debug = vao._score_advantage(
            _FakeModel(), b1, b2, _fake_snap(), attribution_exclude=())
        names_debug = {c for c, _ in drivers_debug}
        assert names_debug & set(ATTRIBUTION_EXCLUDED_INDICATORS)


class TestAttributionExclusionPreservesJudgment:
    """判定値 (adv/p1) は除外の有無で完全不変 (表示だけの修正である証明)。"""

    def test_adv_and_p1_bit_identical_with_and_without_exclusion(
        self, rigged_boards_and_model,
    ) -> None:
        b1, b2 = rigged_boards_and_model
        adv_a, p1_a, _ = vao._score_advantage(_FakeModel(), b1, b2, _fake_snap())
        adv_b, p1_b, _ = vao._score_advantage(
            _FakeModel(), b1, b2, _fake_snap(), attribution_exclude=())
        assert adv_a == adv_b
        assert p1_a == p1_b

    def test_heavy_adv_cache_adv_unaffected_by_attribution_exclude(
        self, rigged_boards_and_model,
    ) -> None:
        """HeavyAdvCache 経由でも adv (self._adv) は除外リストの有無で不変。"""
        b1, b2 = rigged_boards_and_model
        snap = _fake_snap()
        sp1 = sp2 = types.SimpleNamespace(next_pair=None, dnext_pair=None)
        cache_default = vao.HeavyAdvCache(_FakeModel(), every=1)
        cache_debug = vao.HeavyAdvCache(_FakeModel(), every=1, attribution_exclude=())
        adv_a, *_ = cache_default.update(b1, b2, snap, sp1, sp2, 0.0)
        adv_b, *_ = cache_debug.update(b1, b2, snap, sp1, sp2, 0.0)
        assert adv_a == adv_b


class TestAttributionExclusionWiring:
    """除外リストが実際の呼出経路に配線されていること (dead code 化防止)。"""

    def test_score_advantage_default_param_is_production_config_list(self) -> None:
        sig = inspect.signature(vao._score_advantage)
        assert sig.parameters["attribution_exclude"].default == ATTRIBUTION_EXCLUDED_INDICATORS

    def test_heavy_adv_cache_accepts_and_defaults_attribution_exclude(self) -> None:
        sig = inspect.signature(vao.HeavyAdvCache.__init__)
        assert "attribution_exclude" in sig.parameters
        assert sig.parameters["attribution_exclude"].default == ATTRIBUTION_EXCLUDED_INDICATORS

    def test_fresh_trackers_accepts_attribution_exclude(self) -> None:
        sig = inspect.signature(vao._fresh_trackers)
        assert "attribution_exclude" in sig.parameters

    def test_generate_has_show_excluded_attribution_flag_default_false(self) -> None:
        sig = inspect.signature(vao.generate)
        assert "show_excluded_attribution" in sig.parameters
        assert sig.parameters["show_excluded_attribution"].default is False

    def test_cli_flag_wired_to_main(self) -> None:
        """--show-excluded-attribution が実スクリプトに定義されていること。

        (test_production_config.py の _script_text パターンを踏襲)
        """
        text = Path(vao.__file__).read_text(encoding="utf-8")
        assert "--show-excluded-attribution" in text
        assert "show_excluded_attribution=a.show_excluded_attribution" in text


class TestAttributionExcludedIndicatorsAreWellFormed:
    """除外リストの定義そのものが健全であること (typo・重複防止)。"""

    def test_non_empty_tuple_of_str(self) -> None:
        assert len(ATTRIBUTION_EXCLUDED_INDICATORS) >= 1
        assert all(isinstance(x, str) for x in ATTRIBUTION_EXCLUDED_INDICATORS)

    def test_no_duplicates(self) -> None:
        assert len(ATTRIBUTION_EXCLUDED_INDICATORS) == len(
            set(ATTRIBUTION_EXCLUDED_INDICATORS))

    def test_all_are_known_attribution_candidates(self) -> None:
        """除外対象は実際に主因候補になり得る指標名であること (typoでの空振り防止)。"""
        for name in ATTRIBUTION_EXCLUDED_INDICATORS:
            assert name in vao.JP_LABEL, f"{name} は JP_LABEL に無い (主因候補になり得ない)"

    def test_does_not_exclude_known_informative_indicators(self) -> None:
        """current_max_chain 等、有効性確定済みの指標は除外リストに含めない。"""
        informative = {"current_max_chain", "board_ojama_count", "board_color_puyo_total",
                       "ukeyasusa", "sub_chain_count"}
        assert informative.isdisjoint(ATTRIBUTION_EXCLUDED_INDICATORS)

    def test_excludes_diff_conn_pair_count_2026_08_22(self) -> None:
        """2026-08-22 修正④で追加。permutation_importance_full.csv
        (data/verify/retrain_model62_2026-08-21) 実測で52列中52位・負の寄与
        (importance_mean=-0.000376) と確認済み (production_config.py コメント
        参照)。生の整数カウント値のため未正規化な指標より絶対値が大きくなり
        やすく、無情報でも主因1位に出ていた (実測 t=6717.5s)。"""
        assert "diff_conn_pair_count" in ATTRIBUTION_EXCLUDED_INDICATORS
