"""#3 修正 (--counter-remaining-time) の意味論テスト。

docs/DEMO_REVIEW_2026-08-13.md #3: 打ち合い応手の時間予算 = 「観測済み連鎖数
×0.4秒」は (a) 経過時間を控除しない (b) 観測連鎖数を最終連鎖数と誤認する、
の二重にズレた意味論だった。正しい意味論 (残り時間 = anim(E[最終|N到達])
− 経過 + 着弾ラグ) を stateless なヘルパー関数単位で検証する。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.visualize_advantage_overlay import (
    CHAIN_LENGTH_CONDITIONAL_PATH,
    _ChainAttackObservation,
    _chain_remaining_time_budget_sec,
    _detect_chain_attacker,
    _expected_final_chain_count,
    _load_chain_length_conditional_table,
    _resolve_counter_time_budget,
)
from src.indicators_v2 import CHAIN_ANIM_PER_STEP_SEC, SEC_PER_HAND, estimate_chain_anim_duration_sec


# ============================
# _expected_final_chain_count
# ============================


def test_expected_final_empty_table_returns_observed_value() -> None:
    """テーブルが無い (ファイル不在等) 場合は保守的フォールバック
    (観測値=最終値とみなす、旧来近似と同じ) に後退する。"""
    assert _expected_final_chain_count(3, {}) == 3.0


def test_expected_final_uses_table_value_when_present() -> None:
    table = {1: 3.0, 2: 4.5, 3: 6.0}
    assert _expected_final_chain_count(2, table) == 4.5


def test_expected_final_clamps_at_max_observed_key() -> None:
    """実測上限を超える観測値は外挿せずクランプする (CLAUDE.md「シーン逆算
    禁止」= データ上限を超えて推測しない)。"""
    table = {1: 3.0, 2: 4.5, 16: 16.0}
    assert _expected_final_chain_count(20, table) == 16.0


def test_expected_final_zero_or_negative_returns_zero() -> None:
    assert _expected_final_chain_count(0, {1: 3.0}) == 0.0
    assert _expected_final_chain_count(-1, {1: 3.0}) == 0.0


# ============================
# _chain_remaining_time_budget_sec
# ============================


def test_remaining_time_subtracts_elapsed_and_adds_landing_lag() -> None:
    """残り時間 = anim(E[最終|N]) − 経過 + 着弾ラグ (SEC_PER_HAND)。"""
    table = {2: 5.0}  # N=2到達なら最終5連鎖と期待
    trigger_sec = 10.0
    t_sec = 11.0  # 発火から1秒経過
    budget = _chain_remaining_time_budget_sec(2, trigger_sec, t_sec, table)
    expected_anim = estimate_chain_anim_duration_sec(5.0)  # 0.4*5=2.0
    expected = expected_anim - 1.0 + SEC_PER_HAND
    assert budget == expected
    assert expected > 0.0


def test_remaining_time_never_negative() -> None:
    """経過時間がアニメ時間+着弾ラグを超えても 0 未満にはならない。"""
    table = {1: 1.0}
    budget = _chain_remaining_time_budget_sec(1, trigger_sec=0.0, t_sec=1000.0, table=table)
    assert budget == 0.0


def test_remaining_time_zero_chain_count_returns_zero() -> None:
    assert _chain_remaining_time_budget_sec(0, 0.0, 5.0, {}) == 0.0


# ============================
# _resolve_counter_time_budget (旧来経路 vs 新経路の切替)
# ============================


def test_resolve_time_budget_legacy_path_is_bit_identical() -> None:
    """enable_remaining_time=False は従来通り estimate_chain_anim_duration_sec
    (観測連鎖数) そのもの (backwards compat)。"""
    obs = _ChainAttackObservation(
        chain_count=3, trigger_sec=5.0, attacker_side="1P", attacker_event=None)
    budget = _resolve_counter_time_budget(obs, t_sec=100.0, enable_remaining_time=False,
                                          chain_len_table={99: 999.0})
    assert budget == CHAIN_ANIM_PER_STEP_SEC * 3


def test_resolve_time_budget_new_path_differs_from_legacy_due_to_elapsed() -> None:
    """enable_remaining_time=True は経過時間控除+着弾ラグ加算があるため、
    テーブルが空 (フォールバック=観測値そのまま) でも旧来値とは異なる。"""
    obs = _ChainAttackObservation(
        chain_count=2, trigger_sec=10.0, attacker_side="1P", attacker_event=None)
    legacy = _resolve_counter_time_budget(
        obs, t_sec=100.0, enable_remaining_time=False, chain_len_table={})
    new = _resolve_counter_time_budget(
        obs, t_sec=10.5, enable_remaining_time=True, chain_len_table={})
    assert legacy == CHAIN_ANIM_PER_STEP_SEC * 2
    assert new != legacy


def test_resolve_time_budget_no_attack_returns_zero_both_paths() -> None:
    obs = _ChainAttackObservation(
        chain_count=0, trigger_sec=0.0, attacker_side=None, attacker_event=None)
    assert _resolve_counter_time_budget(obs, 1.0, False, {}) == 0.0
    assert _resolve_counter_time_budget(obs, 1.0, True, {}) == 0.0


# ============================
# _detect_chain_attacker
# ============================


def _fake_side_result(chain_event) -> SimpleNamespace:
    return SimpleNamespace(chain_event=chain_event)


def test_detect_chain_attacker_picks_larger_chain_count() -> None:
    ev_small = SimpleNamespace(chain_count=2, trigger_sec=3.0)
    ev_big = SimpleNamespace(chain_count=5, trigger_sec=4.0)
    r_p1 = _fake_side_result(ev_small)
    r_p2 = _fake_side_result(ev_big)
    obs = _detect_chain_attacker(r_p1, r_p2, t_sec=10.0)
    assert obs.chain_count == 5
    assert obs.attacker_side == "2P"
    assert obs.trigger_sec == 4.0
    assert obs.attacker_event is ev_big


def test_detect_chain_attacker_no_events_returns_none() -> None:
    r_p1 = _fake_side_result(None)
    r_p2 = _fake_side_result(None)
    obs = _detect_chain_attacker(r_p1, r_p2, t_sec=10.0)
    assert obs.chain_count == 0
    assert obs.attacker_side is None
    assert obs.attacker_event is None


# ============================
# _load_chain_length_conditional_table
# ============================


def test_load_conditional_table_missing_file_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    assert _load_chain_length_conditional_table(missing) == {}


def test_load_conditional_table_real_file_has_int_keys_and_monotonic_values() -> None:
    """scripts/_build_chain_length_conditional_2026-08-13.py が生成した実ファイル
    (data/verify/chain_length_conditional_2026-08-13.json) を読み込めること。
    ファイルが (再生成待ち等で) 一時的に無い環境でも fail-safe に空dictへ
    後退するため、存在確認できた場合のみ内容を検証する。
    """
    if not CHAIN_LENGTH_CONDITIONAL_PATH.exists():
        return
    table = _load_chain_length_conditional_table()
    assert table  # 空でない
    assert all(isinstance(k, int) for k in table)
    assert all(isinstance(v, float) for v in table.values())
    # E[最終|N] は N について広義単調増加のはず (Nが大きいほど最終連鎖数も大きい)
    keys_sorted = sorted(table)
    values_sorted = [table[k] for k in keys_sorted]
    assert values_sorted == sorted(values_sorted)
    # E[最終|N] >= N (最終連鎖数は観測済みNより小さくなり得ない)
    for n, expected in table.items():
        assert expected >= float(n) - 1e-9
