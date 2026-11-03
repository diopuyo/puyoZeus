"""B3 層・段1門・段2門・封印門・副次CV/単調性の機械判定テスト。"""
import numpy as np
import pandas as pd
import pytest

from src.b3_eval_gates import (FAIL, HOLD, PASS, STOP, classify_monotonic, evaluate_stage1, check_leak_prechecks,
                               secondary_hold)
from src.b3_eval_stage2 import evaluate_sealed, evaluate_stage2
from src.b3_eval_strata import build_strata, honsen_states
from src.b3_train_config import OTHER_STRATA, PAIR_STRATA, REPORT_STRATA


def _mk_df(rows):
    base = {"video_id": "v", "game_idx": 1, "orientation": 0, "frame_idx": 0, "phase": 1, "mode_exchange": 0,
            "chain_active_p1": 0, "chain_active_p2": 0, "mk_board_p1": 0, "mk_board_p2": 0,
            "ep_closed_n_p1": 0, "ep_closed_n_p2": 0, "ep_honsen60_board_p1": np.nan, "ep_honsen60_board_p2": np.nan,
            "ep_honsen60_formula_p1": np.nan, "ep_honsen60_formula_p2": np.nan,
            "ep_chain7plus_p1": np.nan, "ep_chain7plus_p2": np.nan}
    return pd.DataFrame([{**base, "frame_idx": i, **r} for i, r in enumerate(rows)])


def test_honsen_after_persists_until_boundary_and_uses_primary_path():
    df = _mk_df([{}, {"ep_closed_n_p1": 1, "ep_honsen60_board_p1": 1.0}, {"ep_closed_n_p1": 1}, {"ep_closed_n_p1": 1}])
    honsen, unknown = honsen_states(df)
    assert honsen.tolist() == [False, True, True, True] and not unknown.any()
    # 次の試合では持ち越さない
    df2 = pd.concat([df, _mk_df([{}, {}]).assign(game_idx=2)], ignore_index=True)
    h2, _ = honsen_states(df2)
    assert h2.tolist() == [False, True, True, True, False, False]


def test_honsen_unknown_then_resolved_and_not_overriding_honsen():
    # 消費量欠測の発火 → 判定不能。次の発火が 0.6 未満と判明すると判定不能は終わる
    df = _mk_df([{}, {"ep_closed_n_p2": 1}, {"ep_closed_n_p2": 1},
                 {"ep_closed_n_p2": 2, "ep_honsen60_formula_p2": 0.0}, {"ep_closed_n_p2": 2}])
    honsen, unknown = honsen_states(df)
    assert unknown.tolist() == [False, True, True, False, False] and not honsen.any()
    # 既に本線後なら、その後の欠測発火でも本線後のまま (判定不能に入らない)
    df = _mk_df([{"ep_closed_n_p1": 1, "ep_honsen60_board_p1": 1.0}, {"ep_closed_n_p1": 1, "ep_closed_n_p2": 1}])
    h, u = honsen_states(df)
    assert h.tolist() == [True, True] and not u.any()


def test_honsen_secondary_path_used_when_primary_missing():
    df = _mk_df([{"ep_closed_n_p1": 1, "ep_honsen60_formula_p1": 1.0}])
    assert honsen_states(df)[0].tolist() == [True]


def test_honsen_is_causal_prefix_invariant():
    df = _mk_df([{}, {"ep_closed_n_p1": 1, "ep_honsen60_board_p1": 1.0}, {"ep_closed_n_p1": 1},
                 {"ep_closed_n_p2": 1}])
    full = honsen_states(df)[0]
    assert np.array_equal(honsen_states(df.iloc[:2])[0], full[:2])   # 未来の行を落としても過去の層は不変


def test_orientation_sequences_independent():
    a = _mk_df([{}, {"ep_closed_n_p1": 1, "ep_honsen60_board_p1": 1.0}])
    b = _mk_df([{}, {}]).assign(orientation=1)
    h, _ = honsen_states(pd.concat([a, b], ignore_index=True))
    assert h.tolist() == [False, True, False, False]


def test_build_strata_names_and_masks():
    df = _mk_df([{"phase": 0, "mode_exchange": 0}, {"phase": 2, "mode_exchange": 1, "chain_active_p1": 1, "chain_active_p2": 1},
                 {"phase": 1, "mode_exchange": 0, "mk_board_p1": 5}])
    s = build_strata(df)
    assert set(PAIR_STRATA + OTHER_STRATA + REPORT_STRATA + ("overall",)) <= set(s)
    assert s["phase_early"].tolist() == [True, False, False] and s["phase_late"].tolist() == [False, True, False]
    assert s["exchange"].tolist() == [False, True, False] and s["static"].tolist() == [True, False, True]
    assert s["both_chain"].tolist() == [False, True, False]
    assert s["both_build"].tolist() == [True, False, False]       # 3 行目は片側盤面が未観測


# ---- 段1 門 ----
def _stat(point, hi, games=30, groups=8, lo=None, brier=0.0, ece=0.0, ll_cand=0.6):
    return {"delta_ll": {"point": point, "hi": hi, "lo": point - 0.01 if lo is None else lo},
            "population": {"games": games, "groups": groups, "videos": groups, "rows": 1000, "held_sec": 1.0},
            "meets_min_population": games >= 20 and groups >= 5, "delta_brier": brier, "delta_ece": ece,
            "ll_cand": ll_cand, "ll_ref": ll_cand - point}


def _good_scores():
    st = {n: _stat(-0.003, 0.004) for n in PAIR_STRATA + OTHER_STRATA}
    st["overall"] = _stat(-0.004, 0.001, brier=-0.001, ece=0.0, ll_cand=0.60)
    return {"overall": st["overall"], "strata": st}


ALL_PRE = {k: 0 for k in ("future_suffix_prefix_hash", "train_replay_live_hash", "symmetry", "schema_mask",
                           "candidate_correspondence", "row_key_unique", "time_fire_id", "sealed_overlap",
                           "available_at", "group_separation")}


def test_stage1_pass_when_everything_met():
    r = evaluate_stage1(_good_scores(), oracle_ll=0.55, prechecks=ALL_PRE, time_gate_ok=True,
                        secondary={"d_mean": 0.005, "per_player": {"a": {"d": 0.01, "test_games": 30}}},
                        monotonic={"class": "none", "ok": True})
    assert r["verdict"] == PASS and not r["unmet"] and not r["not_evaluated"]


def test_stage1_missing_checks_are_hold_not_pass():
    r = evaluate_stage1(_good_scores(), oracle_ll=0.55)
    assert r["verdict"] == HOLD
    assert {"leak_prechecks", "time_gate", "secondary_player_cv", "monotonic_post_check"} <= set(r["not_evaluated"])


def test_stage1_overall_upper_gate_boundaries():
    s = _good_scores()
    s["overall"] = s["strata"]["overall"] = _stat(-0.001, 0.0021)
    assert evaluate_stage1(s, oracle_ll=0.5, prechecks=ALL_PRE)["verdict"] == FAIL
    s["overall"] = s["strata"]["overall"] = _stat(-0.001, 0.002)
    assert "overall_delta_ll" not in evaluate_stage1(s, oracle_ll=0.5, prechecks=ALL_PRE)["unmet"]
    s["overall"] = s["strata"]["overall"] = _stat(0.0, 0.001)         # 点推定 < 0 が必要 (0 は不可)
    assert "overall_delta_ll" in evaluate_stage1(s, oracle_ll=0.5)["unmet"]


def test_stage1_pair_strata_need_negative_point_and_others_have_tolerance():
    s = _good_scores()
    s["strata"]["exchange"] = _stat(0.0001, 0.003)
    s["strata"]["phase_late"] = _stat(0.005, 0.020)                   # ちょうど境界は許容
    r = evaluate_stage1(s, oracle_ll=0.5)
    assert "stratum_exchange" in r["unmet"] and "stratum_phase_late" not in r["unmet"]
    s["strata"]["phase_late"] = _stat(0.0051, 0.01)
    assert "stratum_phase_late" in evaluate_stage1(s, oracle_ll=0.5)["unmet"]
    s["strata"]["phase_late"] = _stat(0.0, 0.0201)
    assert "stratum_phase_late" in evaluate_stage1(s, oracle_ll=0.5)["unmet"]


def test_stage1_insufficient_population_is_hold_even_if_numbers_good():
    s = _good_scores()
    s["strata"]["honsen_after"] = _stat(-0.01, -0.005, games=19, groups=8)
    r = evaluate_stage1(s, oracle_ll=0.5, prechecks=ALL_PRE, time_gate_ok=True,
                        secondary={"d_mean": 0.0, "per_player": {}}, monotonic={"class": "none", "ok": True})
    assert r["verdict"] == HOLD and "stratum_honsen_after" in r["not_evaluated"]
    s["strata"]["honsen_after"] = _stat(-0.01, -0.005, games=30, groups=4)
    assert evaluate_stage1(s, oracle_ll=0.5)["not_evaluated"].count("stratum_honsen_after") == 1
    s["strata"]["both_chain"] = {"population": {"games": 0, "groups": 0}, "meets_min_population": False, "empty": True}
    assert "stratum_both_chain" in evaluate_stage1(s, oracle_ll=0.5)["not_evaluated"]


def test_stage1_brier_ece_gates():
    s = _good_scores()
    s["overall"]["delta_brier"] = 0.0021
    assert "overall_delta_brier" in evaluate_stage1(s, oracle_ll=0.5)["unmet"]
    s["overall"]["delta_brier"] = 0.002
    s["overall"]["delta_ece"] = 0.0101
    r = evaluate_stage1(s, oracle_ll=0.5)
    assert "overall_delta_ece" in r["unmet"] and "overall_delta_brier" not in r["unmet"]


def test_stage1_oracle_exceeded_stops_before_anything():
    s = _good_scores()                                                  # cand LL 0.60
    r = evaluate_stage1(s, oracle_ll=0.62, prechecks=ALL_PRE, time_gate_ok=True)
    assert r["verdict"] == STOP


def test_leak_prechecks_semantics():
    assert check_leak_prechecks(None)["ok"] is None
    part = dict(ALL_PRE)
    part["available_at"] = None
    assert check_leak_prechecks(part)["ok"] is None                     # 未実施は合格にしない
    part["available_at"] = 3
    assert check_leak_prechecks(part)["ok"] is False
    assert check_leak_prechecks(ALL_PRE)["ok"] is True


def test_secondary_hold_rules():
    assert secondary_hold(0.021, {})["hold"] is True
    assert secondary_hold(0.02, {})["hold"] is False
    assert secondary_hold(0.0, {"a": {"d": 0.06, "test_games": 20}})["hold"] is True
    assert secondary_hold(0.0, {"a": {"d": 0.06, "test_games": 19}})["hold"] is False   # test 20 試合未満は保留条件外
    assert secondary_hold(0.0, {"a": {"d": 0.05, "test_games": 40}})["hold"] is False


def test_monotonic_classification():
    assert classify_monotonic([], 1000.0)["class"] == "none"
    minor = classify_monotonic([{"name": "a", "held_sec": 40.0, "max_violation": 0.02, "games_over_005": 0}], 1000.0)
    assert minor["class"] == "minor" and minor["ok"] is True
    hold = classify_monotonic([{"name": "a", "held_sec": 60.0, "max_violation": 0.02}], 1000.0)
    assert hold["class"] == "hold" and hold["ok"] is False
    mid = classify_monotonic([{"name": "a", "held_sec": 40.0, "max_violation": 0.04, "games_over_005": 3}], 1000.0)
    assert mid["class"] == "intermediate" and mid["ok"] is None
    many = classify_monotonic([{"name": "a", "held_sec": 10.0, "max_violation": 0.06, "games_over_005": 20}], 1000.0)
    assert many["class"] == "hold"


# ---- 段2 / 封印 ----
def _stage2_ok():
    return {"false_confirms": {"records18": 0, "short5": 0, "game14": 0, "all115": 0}, "first_confirm_sec_3min": 2765.0,
            "confirms": {"lost": 0, "delay_p95_diff": 0.2, "new_delay_over_5s": 0},
            "m_time": {"point": 0.001, "lo": -0.01, "hi": 0.019}, "m_agree": {"point": -0.004, "lo": -0.029, "hi": 0.01},
            "dll": {"point": -0.001, "hi": 0.01}, "brier_diff": 0.002, "review_windows_done": True,
            "time_gate_ok": True, "forecast72": {"sole_confirms": 3, "false_confirms": 0}}


def test_stage2_pass_and_boundaries():
    assert evaluate_stage2(_stage2_ok())["verdict"] == PASS
    x = _stage2_ok(); x["m_time"]["hi"] = 0.020
    assert "m_time_vs_R" in evaluate_stage2(x)["unmet"]                  # 上端 < 0.020 は厳密不等号
    x = _stage2_ok(); x["m_agree"]["lo"] = -0.030
    assert "m_agree_vs_R" in evaluate_stage2(x)["unmet"]
    x = _stage2_ok(); x["first_confirm_sec_3min"] = 2766.1
    assert "first_confirm_3min" in evaluate_stage2(x)["unmet"]
    x = _stage2_ok(); x["false_confirms"]["all115"] = 1
    assert evaluate_stage2(x)["verdict"] == FAIL
    x = _stage2_ok(); x["forecast72"]["false_confirms"] = 1
    assert "forecast72_false_confirms" in evaluate_stage2(x)["unmet"]
    x = _stage2_ok(); del x["review_windows_done"]
    assert evaluate_stage2(x)["verdict"] == HOLD


def _sealed_sets(cc=6):
    s = {"games": 49, "false_confirms": 0, "undetermined_confirms": 0, "correct_confirms_R": cc,
         "confirms": {"lost": 0, "delay_p95_diff": 0.1, "new_delay_over_5s": 0},
         "dll": {"point": 0.0}, "brier_diff": 0.0}
    return {"v43": dict(s), "sealed01": dict(s, games=54), "sealed02": dict(s, games=55)}


def _sealed_total():
    return {"dll": {"point": -0.001, "hi": 0.005}, "brier_diff": 0.001, "ece_diff": 0.0, "games_exchange": 40,
            "games_build": 60}


def test_sealed_pass_and_hold_when_prerequisites_not_met():
    r = evaluate_sealed(_sealed_sets(), _sealed_total(), time_gate_ok=True, forecast72={"false_confirms": 0})
    assert r["verdict"] == PASS
    r = evaluate_sealed(_sealed_sets(cc=4), _sealed_total(), time_gate_ok=True, forecast72={"false_confirms": 0})
    assert r["verdict"] == HOLD                                         # 5 件未満 → 閾値を緩めず保留
    t = _sealed_total(); t["games_exchange"] = 19
    assert evaluate_sealed(_sealed_sets(), t, time_gate_ok=True, forecast72={"false_confirms": 0})["verdict"] == HOLD
    r = evaluate_sealed(_sealed_sets(), _sealed_total(), time_gate_ok=True, forecast72={"false_confirms": 0},
                        opening_complete=False)
    assert r["verdict"] == HOLD


def test_sealed_any_false_confirm_fails_and_per_set_gates():
    s = _sealed_sets(); s["sealed01"]["false_confirms"] = 1
    assert evaluate_sealed(s, _sealed_total(), time_gate_ok=True, forecast72={"false_confirms": 0})["verdict"] == FAIL
    s = _sealed_sets(); s["v43"]["undetermined_confirms"] = 1
    assert "v43:undetermined_confirms" in evaluate_sealed(s, _sealed_total(), time_gate_ok=True)["unmet"]
    s = _sealed_sets(); s["v43"]["dll"] = {"point": 0.0101}
    assert "v43:delta_ll" in evaluate_sealed(s, _sealed_total(), time_gate_ok=True)["unmet"]
    t = _sealed_total(); t["dll"] = {"point": 0.0, "hi": 0.0101}
    assert "total_delta_ll" in evaluate_sealed(_sealed_sets(), t, time_gate_ok=True)["unmet"]
