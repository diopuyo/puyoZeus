"""B3 fold 割当・group 規則のテスト (事前登録 §2.2〜§2.3)。"""
import hashlib

import pytest

from src.b3_train_folds import (FoldLeakError, assert_no_group_leak, assign_folds, check_group_minimum, control_drop_order,
                                fold_summary, group_key, inner_assignment, normalize_player, outer_assignment,
                                video_to_group, write_assignment_tsv)


def _games(n=23):
    return {f"g{i:02d}": 3 + (i * 7) % 11 for i in range(n)}


def test_group_key_sorted_and_joined():
    assert group_key(["zB8d1FuCStU", "iZZyNGtJ3G8"]) == "iZZyNGtJ3G8+zB8d1FuCStU"
    assert group_key(["abc"]) == "abc"
    with pytest.raises(ValueError):
        group_key([])


def test_video_to_group_merges_repost_and_split_same_youtube():
    vy = {"video_c110": "zB8d1FuCStU", "video_c112": "iZZyNGtJ3G8", "video_c96s1": "KmRmSHe7DPg",
          "video_c96s2": "KmRmSHe7DPg", "video_x": "other"}
    m = video_to_group(vy, merges=[("iZZyNGtJ3G8", "zB8d1FuCStU")])
    assert m["video_c110"] == m["video_c112"] == "iZZyNGtJ3G8+zB8d1FuCStU"
    assert m["video_c96s1"] == m["video_c96s2"] == "KmRmSHe7DPg"
    assert m["video_x"] == "other"


def test_assign_deterministic_and_input_order_free():
    g = _games()
    a1, a2 = outer_assignment(g), outer_assignment(dict(reversed(list(g.items()))))
    assert a1 == a2
    assert set(a1.values()) == {0, 1, 2, 3, 4}


def test_assign_matches_manual_procedure():
    g = {"a": 5, "b": 3, "c": 4}
    salt = "S|"
    order = sorted(g, key=lambda k: hashlib.sha256((salt + k).encode()).hexdigest())
    totals = [0, 0]
    expect = {}
    for k in order:
        f = min((0, 1), key=lambda i: (totals[i], i))
        expect[k] = f
        totals[f] += g[k]
    assert assign_folds(g, 2, salt) == expect


def test_assign_balanced_games():
    g = _games(40)
    a = outer_assignment(g)
    tot = [sum(g[k] for k in a if a[k] == f) for f in range(5)]
    assert max(tot) - min(tot) <= max(g.values())


def test_inner_assignment_only_train_groups_and_different_salt():
    g = _games()
    outer = outer_assignment(g)
    inner = inner_assignment(g, 2, outer)
    assert set(inner) == {k for k in g if outer[k] != 2}
    assert set(inner.values()) == {0, 1, 2}
    assert inner_assignment(g, 0, outer) != inner


def test_no_group_leak_detected():
    assert_no_group_leak({"a", "b"}, {"c"})
    with pytest.raises(FoldLeakError):
        assert_no_group_leak({"a", "b"}, {"b"})


def test_too_few_groups():
    with pytest.raises(ValueError):
        assign_folds({"a": 1, "b": 1}, 5, "x")
    with pytest.raises(ValueError):
        check_group_minimum(49, 50)
    check_group_minimum(50, 50)


def test_tsv_sha_changes_with_assignment(tmp_path):
    g = _games(10)
    a = outer_assignment(g)
    s1 = write_assignment_tsv(tmp_path / "a.tsv", g, a, {})
    b = dict(a)
    k = next(iter(b))
    b[k] = (b[k] + 1) % 5
    s2 = write_assignment_tsv(tmp_path / "b.tsv", g, b, {})
    assert s1 != s2 and len(s1) == 64
    assert fold_summary(g, a)["0"]["groups"] >= 1


def test_normalize_player_katakana_and_width():
    assert normalize_player("ペペペマン") == normalize_player("ぺぺぺマン")
    assert normalize_player(" Ｓ Ａ Ｋ Ｉ ") == "saki"


def test_control_drop_order_deterministic():
    vids = ["v1", "v2", "v3", "v4"]
    assert control_drop_order("のらすけ", vids) == control_drop_order("のらすけ", list(reversed(vids)))
