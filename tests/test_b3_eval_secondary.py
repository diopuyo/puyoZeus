"""副次 CV (対戦者単位) の分割・抽出テスト。"""
import json
from pathlib import Path

import pytest

from src.b3_eval_secondary import (PlayerExtractionError, control_split, d_mean, d_player, eligible_players,
                                   extract_players, leave_player_split, video_players)
from src.b3_train_folds import normalize_player, video_to_group

TARGETS = Path("D:/puyo_analyzer/wt_b2/data/verify/recollect_2026-10-03/targets.json")
TARGETS_LINUX = Path("/mnt/d/puyo_analyzer/wt_b2/data/verify/recollect_2026-10-03/targets.json")


def test_extract_plain_and_quoted_and_failure():
    assert extract_players("【マスター・3ブロック】light vs SAKI 30先(解説：こんけあ)【第2回】") == [("light", "SAKI")]
    q = extract_players("【S級リーグ】最終戦「やまだ vs のらすけ」「ともくん vs delta」30先【新おいうリーグ】")
    assert q == [("やまだ", "のらすけ"), ("ともくん", "delta")]
    with pytest.raises(PlayerExtractionError):
        extract_players("対戦者の書かれていないタイトル")


def test_normalized_alias_pepepeman_same_person():
    ps = video_players({"a": "【A級・1ブロック】にくじゃが vs ペペペマン 30先", "b": "【A級・1ブロック】いつき vs ぺぺぺマン 30先"})
    assert ps["a"] & ps["b"] == {normalize_player("ぺぺぺマン")}


def _toy():
    titles = {f"v{i}": f"【x】P{i % 3} vs Q{i} 30先" for i in range(9)}
    players = video_players(titles)
    gmap = {f"v{i}": f"g{i}" for i in range(9)}
    return players, gmap


def test_eligible_and_leave_split_no_group_cross():
    players, gmap = _toy()
    el = eligible_players(players, 3)
    assert set(el) == {"p0", "p1", "p2"} and all(len(v) == 3 for v in el.values())
    sp = leave_player_split("p0", players, gmap)
    assert sp["test"] == ["v0", "v3", "v6"] and not (set(sp["train"]) & set(sp["test"]))
    # 同一 group の別動画 (分割・再投稿) は test 側へ引き込む
    gmap2 = dict(gmap, v1="g0")
    sp2 = leave_player_split("p0", players, gmap2)
    assert "v1" in sp2["test"] and "v1" not in sp2["train"]


def test_control_degenerates_when_all_player_videos_are_test():
    players, gmap = _toy()
    sp = leave_player_split("p0", players, gmap)
    c = control_split("p0", sp, players, gmap)
    assert c["control"] is None and "縮退" in c["reason"]


def test_d_metrics():
    assert d_player(0.70, 0.66) == pytest.approx(0.04)
    assert d_mean({"a": {"d": 0.1, "test_games": 10}, "b": {"d": 0.0, "test_games": 30}}) == pytest.approx(0.025)


@pytest.mark.skipif(not (TARGETS.exists() or TARGETS_LINUX.exists()), reason="B2 targets.json が無い環境")
def test_real_targets_extract_all_and_fourteen_eligible_players():
    p = TARGETS if TARGETS.exists() else TARGETS_LINUX
    data = json.loads(p.read_text(encoding="utf-8"))
    use = [t for t in data if t["video_id"] not in ("video_c139", "video_c140")]
    ps = video_players({t["video_id"]: t["title"] for t in use})
    assert len(ps) == 58                                   # 全動画で抽出できる (抽出不能なら停止する規則)
    el = eligible_players(ps)
    assert len(el) == 14                                   # 事前登録 §3.3 の 14 名
    assert {normalize_player(n) for n in ("のらすけ", "ともくん", "delta", "ゆうき")} <= set(el)
    assert len(el[normalize_player("のらすけ")]) == 8
    gm = video_to_group({t["video_id"]: t["youtube_id"] for t in use}, [("iZZyNGtJ3G8", "zB8d1FuCStU")])
    assert len(set(gm.values())) == 57                      # 事前登録 §2.2 の 58 本全体で 57 group


# ---- 単調性の事後検査 ----
def test_monotonic_check_detects_violation_layers():
    import numpy as np
    import pandas as pd
    from src.b3_eval_gates import classify_monotonic
    from src.b3_eval_monotonic import check_monotonic, make_count_perturb, static_total_sec

    n = 40
    df = pd.DataFrame({"video_id": "v", "game_idx": np.repeat([1, 2], n // 2), "mode_exchange": 0, "phase": 1,
                       "held_sec": 1.0, "own_oj": 0.0, "opp_oj": 0.0, "own_fc": 0.0, "opp_fc": 0.0,
                       "cells_color_p1": np.linspace(10, 30, n)})
    df.loc[:19, "own_oj"] = 12.0                                       # 層を 2 つに分ける

    def predict(d):                                                    # 層 A (own_oj=12) だけ非単調 (逆向き)
        x = d["cells_color_p1"].to_numpy(dtype=float)
        sign = np.where(d["own_oj"].to_numpy() == 12.0, -1.0, 1.0)
        return 1 / (1 + np.exp(-sign * (x - 20) / 10))

    from src.b3_eval_monotonic import stage_layers
    keys = stage_layers(df, "own_oj", "opp_oj", "own_fc", "opp_fc")
    layers = check_monotonic(df, predict, make_count_perturb({"cells_color_p1": 1.0}), keys)
    bad = [l for l in layers if l["max_violation"] > 0]
    assert len(layers) == 2 and len(bad) == 1 and bad[0]["name"].startswith("2|")
    assert bad[0]["held_sec"] == 20.0 and static_total_sec(df) == 40.0
    assert classify_monotonic(layers, static_total_sec(df))["class"] == "hold"      # 保持秒 50% > 5%
    ok = check_monotonic(df, lambda d: 1 / (1 + np.exp(-(d["cells_color_p1"].to_numpy(dtype=float) - 20) / 10)),
                         make_count_perturb({"cells_color_p1": 1.0}), keys)
    assert all(l["max_violation"] == 0 for l in ok)
