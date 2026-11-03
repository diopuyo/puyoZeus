"""案A (受ける側から見た正味の予告) の試験: 式・列仕様・側交換・配信と学習の一致・既定OFFの不変・導出台本。

重い検証 (学習・比較) は含まない。実データの再現は C:/puyo1_work/ledger_chain_diag_20261025 がある環境だけ。
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import b3_column_spec, b3_rows
from src.b3_column_spec import (NET_LEDGER_COLUMN_NAMES, NET_LEDGER_FEATURE_NAMES, NET_TA_STEMS, NET_TA_VARIANTS, SIDES, Kind, Mask,
                                build_column_specs, swap_sides)
from src.b3_features import build_features
from src.b3_net_ledger import (FRAME_DTYPES, SOURCE_COLUMNS, derive_frame, derive_net_columns, derive_net_row,
                               margin_columns, settle_net, settle_net_sequential)
from src.b3_source import load_lean
from src.b3_state import GameState
from src.b3_timeaxis import UNCANCELLABLE_OJAMA, _fill_margins
from src.b3_timing import TimingTable
from tests.b3_synth import build_record

ROOT = Path(__file__).resolve().parents[1]
NAN = float("nan")
SMALL_MAX = 12            # 全組の網羅の上限 (0..12 の 4 変数 = 28,561 組)
RANDOM_ROWS = 4000
BIG_COUNT_MAX = 700       # 72 個以上 (死亡確定の根拠) を超える量も含める
DIAG_DIR = Path("C:/puyo1_work/ledger_chain_diag_20261025" if os.name == "nt"
                else "/mnt/c/puyo1_work/ledger_chain_diag_20261025")


# ----------------------------------------------------------------------------- 式

def test_closed_form_equals_the_game_rule_in_both_orders_exhaustively() -> None:
    """net=max(0,x,x-y) は、ゲーム規則を順に適用した結果と (どちらの側を先にしても) 全組で一致する。"""
    grid = np.array(list(itertools.product(range(SMALL_MAX + 1), repeat=4)), dtype=np.float64)
    n1, n2 = settle_net(grid[:, 0], grid[:, 1], grid[:, 2], grid[:, 3])
    for row, a, b in zip(grid.astype(int), n1, n2):
        for first in (0, 1):
            assert settle_net_sequential(*row, first=first) == (a, b), (row, first)


def test_closed_form_equals_the_game_rule_on_large_random_values() -> None:
    rng = np.random.default_rng(1)
    grid = rng.integers(0, BIG_COUNT_MAX, size=(RANDOM_ROWS, 4))
    n1, n2 = settle_net(*grid.T.astype(np.float64))
    for row, a, b in zip(grid, n1, n2):
        assert settle_net_sequential(*map(int, row), first=0) == (a, b) == settle_net_sequential(*map(int, row), first=1)


@pytest.mark.parametrize("newest_first", [False, True])
def test_closed_form_matches_the_real_ledger_settlement(newest_first: bool) -> None:
    """台帳の実コード (_convert → _offset_own → 相手の予告へ積む) で両側の未反映量を決済した後の pending と一致する。

    決済の順序 (P1 先/P2 先) と、相殺の消費順 (古い塊から/新しい塊から=割り込み) のどれでも総量は同じ。
    """
    from src.b3_score_ledger import MODE_OBSERVED, ScoreLedgerV2, StepRecord, _Batch
    rng = np.random.default_rng(6)
    cases = list(itertools.product(range(6), repeat=4)) + [tuple(int(v) for v in rng.integers(0, 150, 4)) for _ in range(400)]
    for p1, p2, g1, g2 in cases:
        want = tuple(float(v) for v in settle_net(p1, p2, g1, g2))
        for order in ((0, 1), (1, 0)):
            ledger = ScoreLedgerV2(mode=MODE_OBSERVED, newest_first_offset=newest_first, enable_inflight=False,
                                   observed_landing_rule=False)
            for side, pending in enumerate((p1, p2)):             # 事前の予告は 2 塊に分ける (消費順を効かせる)
                head = pending // 2
                ledger.batches[side] = [_Batch(n=n, src_side=1 - side, chain_id=k) for k, n in enumerate((head, pending - head)) if n > 0]
            for side, generated in ((order[0], (g1, g2)[order[0]]), (order[1], (g1, g2)[order[1]])):
                ledger.st[side].acc = generated * ledger.rate_base      # 換算後ちょうど generated 個
                ledger._convert(side, StepRecord(t_formula=0.0, product=0), 0.0)
            assert (float(ledger._pending(0)), float(ledger._pending(1))) == want, (p1, p2, g1, g2, order, newest_first)


def test_net_identities() -> None:
    """両側 inflight=0 なら pending に一致。df は x-y。net は 0 以上。最終の p1-p2 は決済順に依らない。"""
    rng = np.random.default_rng(2)
    p1, p2, g1, g2 = rng.integers(0, BIG_COUNT_MAX, size=(4, RANDOM_ROWS)).astype(np.float64)
    n1, n2 = settle_net(p1, p2, g1, g2)
    assert (n1 >= 0).all() and (n2 >= 0).all()
    assert np.array_equal(n1 - n2, (p1 - g1) - (p2 - g2))                  # = dpend + (g2 - g1) (受ける側向き)
    z1, z2 = settle_net(p1, p2, np.zeros_like(g1), np.zeros_like(g2))
    assert np.array_equal(z1, p1) and np.array_equal(z2, p2)               # inflight 無しは確定値のまま


@pytest.mark.parametrize("pending,inflight,expected", [
    ((0, 39), (134, 0), (0, 173)),     # RESULT.md S1: 単発大連鎖 (naive +95 は逆符号、net は 2P が 173 受ける)
    ((7, 0), (3, 13), (17, 0)),        # S2: 両側同時 (naive -3、net は 1P が 17 受ける)
    ((5, 5), (3, 3), (2, 2)),          # 両側が互いに相殺しても残る
    ((0, 5), (10, 10), (0, 5)),
    ((5, 0), (10, 3), (0, 2)),
    ((72, 0), (0, 0), (72, 0)),
])
def test_known_cases(pending: tuple, inflight: tuple, expected: tuple) -> None:
    n1, n2 = settle_net(*pending, *inflight)
    assert (float(n1), float(n2)) == expected


def test_out_of_domain_inputs_are_missing_not_propagated() -> None:
    """NaN・無限・負の個数は両側とも NaN (他の行は影響を受けない)。"""
    p1 = np.array([1.0, NAN, 1.0, 1.0, np.inf, 2.0])
    p2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    g1 = np.array([1.0, 1.0, NAN, -1.0, 1.0, 1.0])
    g2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    n1, n2 = settle_net(p1, p2, g1, g2)
    assert np.isnan(n1[1:5]).all() and np.isnan(n2[1:5]).all()
    assert (n1[0], n2[0]) == (0.0, 0.0) and (n1[5], n2[5]) == (1.0, 1.0)


def test_margin_columns_match_time_axis_fill_margins() -> None:
    """余地 M=Q-I の派生式は b3_timeaxis._fill_margins (既存の sl 版) と同じ。"""
    rng = np.random.default_rng(3)
    qs = [*rng.uniform(0, 300, 200), NAN, 0.0]
    for q, i in itertools.product(qs[:60] + [NAN], [*rng.integers(0, 300, 30).astype(float), NAN, 0.0,
                                                    float(UNCANCELLABLE_OJAMA)]):
        out: dict = {}
        _fill_margins(out, q, q, {"pred": NAN, "sl": i, "icon": NAN})
        mine = margin_columns(q, i)
        for stem in NET_TA_STEMS:
            want = out[f"{stem}_sl_med"]
            got = float(mine[stem])
            assert (np.isnan(want) and np.isnan(got)) or want == got, (stem, q, i, want, got)
    edge = UNCANCELLABLE_OJAMA                                                    # 残受け (I-Q) が 72 ちょうどで 1
    assert float(margin_columns(100.0, 100.0 + edge)["ta_unc72"]) == 1.0
    assert float(margin_columns(100.0, 100.0 + edge - 1)["ta_unc72"]) == 0.0


# ----------------------------------------------------------------------------- 列仕様

def test_off_spec_is_unchanged_and_on_only_appends() -> None:
    off = build_column_specs(enable_inflight=False, enable_net_ledger=False)
    m3 = build_column_specs(enable_inflight=True, enable_net_ledger=False)
    on = build_column_specs(enable_inflight=True, enable_net_ledger=True)
    assert [c.name for c in off] == b3_column_spec.COLUMN_NAMES == [c.name for c in build_column_specs(False)]
    assert not set(NET_LEDGER_COLUMN_NAMES) & set(b3_column_spec.COLUMN_NAMES)     # 既定のモジュール列に案Aは無い
    assert m3[:len(off)] == off and on[:len(m3)] == m3                          # 順序保持: 末尾追加だけ
    assert tuple(c.name for c in on[len(m3):]) == NET_LEDGER_COLUMN_NAMES and len(NET_LEDGER_COLUMN_NAMES) == 32
    assert len(off) == 459 and len(m3) == 469 and len(on) == 501


def test_net_columns_require_inflight_columns() -> None:
    with pytest.raises(ValueError, match="sl_inflight"):
        build_column_specs(enable_inflight=False, enable_net_ledger=True)


def test_net_column_kinds_are_explicit_per_column() -> None:
    """側の量は SIDE_ABS (入れ替え)、差は DIFF_RAW (符号反転)、mask は audit。既存規約の自動検査も通す。"""
    on = build_column_specs(enable_inflight=True, enable_net_ledger=True)
    by_name = {c.name: c for c in on}
    new = [by_name[n] for n in NET_LEDGER_COLUMN_NAMES]
    assert by_name["df_sl_net"].kind is Kind.DIFF_RAW and by_name["df_sl_net"].counterpart == ""
    assert by_name["mk_df_sl_net"].kind is Kind.COMMON
    for c in new:
        if c.name in ("df_sl_net", "mk_df_sl_net"):
            continue
        assert c.kind is Kind.SIDE_ABS and c.name.endswith(("_p1", "_p2")), c.name
        partner = by_name[c.counterpart]
        assert partner.counterpart == c.name and partner.dtype == c.dtype
        assert c.role == ("audit" if c.name.startswith("mk_") else "feature")
        if c.mask:
            assert c.mask in by_name and by_name[c.mask].dtype == "i1"
    normalized = [c for c in new if "_nf_" in c.name or "_mg_" in c.name]          # 正規化の規約: 比率列は 0-1
    assert len(normalized) == 2 + 2 * len(NET_TA_VARIANTS) and all(c.unit == "0-1" for c in normalized)
    assert len({c.name for c in on}) == len(on)


def test_fast_variants_follow_the_existing_exclusion_rule() -> None:
    """_fast 列は保存だけ (EXCLUDED_FAST_COLUMNS の規則: group=time_axis かつ名前に _fast)。net 版も同じ規則に乗る。"""
    on = build_column_specs(enable_inflight=True, enable_net_ledger=True)
    fast = [c.name for c in on if c.group == "time_axis" and "_fast" in c.name and c.name in NET_LEDGER_COLUMN_NAMES]
    assert len(fast) == 2 * len(NET_TA_STEMS)                                      # 5 語幹 x 2 側
    assert all("slnet_fast" in n for n in fast) and "slnet_med" not in " ".join(fast)


def test_render_markdown_default_stays_off_and_lists_net_when_asked() -> None:
    base = b3_column_spec.render_markdown()
    assert "net_pending" not in base and b3_column_spec.render_markdown(False, False) == base
    text = b3_column_spec.render_markdown(enable_inflight=True, enable_net_ledger=True)
    assert all(f"| {n} |" in text for n in NET_LEDGER_COLUMN_NAMES) and "max(0,x,x-y)" in text


def test_frame_dtypes_match_to_frame() -> None:
    assert all(FRAME_DTYPES[k] == b3_rows._DTYPES[k] for k in FRAME_DTYPES)


# ----------------------------------------------------------------------------- 側交換

def _random_full_row(specs, rng: np.random.Generator) -> dict:
    """列仕様どおり全列を持つ乱数の行。導出元は『ありそうな値』に上書きする (個数・欠測・mask)。"""
    row: dict = {}
    for c in specs:
        row[c.name] = ("x|g1|f2|o0" if c.dtype == "str" else float(rng.uniform(0, 1)) if c.dtype.startswith("f")
                       else int(rng.integers(0, 4)))
    row["orientation"], row["row_key"] = 0, "v|g1|f2|o0"
    pend = rng.integers(0, 120, 2).astype(float)
    infl = rng.integers(0, 200, 2).astype(float) * (rng.random(2) < 0.6)
    nan_row = rng.random() < 0.15
    for k, side in enumerate(SIDES):
        other = SIDES[1 - k]
        row[f"sl_pending_{side}"] = NAN if nan_row else pend[k]
        row[f"sl_inflight_{side}"] = NAN if nan_row else infl[k]
        row[f"mk_sl_{side}"] = int(Mask.HISTORY_INCOMPLETE) if nan_row else 0
        row[f"mk_sl_inflight_{side}"] = int(Mask.HISTORY_INCOMPLETE) if nan_row else 0
        ta_on = rng.random() < 0.7
        row[f"ta_i_sl_{side}"] = row[f"sl_pending_{other}"] if ta_on and not nan_row else NAN
        row[f"ta_q_med_{side}"] = float(rng.integers(0, 300)) if ta_on else NAN
        row[f"ta_q_fast_{side}"] = float(rng.integers(0, 300)) if ta_on else NAN
        row[f"mk_ta_{side}"] = int(rng.choice([0, 3, 6])) if ta_on else int(Mask.NOT_APPLICABLE)
    return row


def _same(a: object, b: object) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        a, b = float(a), float(b)
        return (np.isnan(a) and np.isnan(b)) or a == b
    return a == b


def test_swap_kinds_and_derivation_commute() -> None:
    """側を入れ替えた行の net 列 = 導出した net 列を台帳規則で入れ替えたもの (全列・欠測込み)。"""
    on = build_column_specs(enable_inflight=True, enable_net_ledger=True)
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(600):
        row = _random_full_row(on, rng)
        row.update(derive_net_row(row))
        swapped = swap_sides(row, specs=on)
        assert swap_sides(swapped, specs=on).keys() == row.keys()
        recomputed = derive_net_row(swapped)                     # 入れ替えた入力から導出し直す
        for name in NET_LEDGER_COLUMN_NAMES:
            assert _same(recomputed[name], swapped[name]), (name, recomputed[name], swapped[name])
            checked += 1
        for k in (1, 2):                                          # 種類の直接確認: 側の量は入れ替わり、差は符号反転
            assert _same(swapped[f"net_pending_p{k}"], row[f"net_pending_p{3 - k}"])
        assert _same(swapped["df_sl_net"], -row["df_sl_net"])
    assert checked == 600 * len(NET_LEDGER_COLUMN_NAMES)


# ----------------------------------------------------------------------------- 欠測 mask

def _sources(**over: object) -> dict:
    src: dict = {k: 0.0 for k in SOURCE_COLUMNS}
    src.update({f"ta_q_med_{s}": 50.0 for s in SIDES} | {f"ta_q_fast_{s}": 40.0 for s in SIDES}
               | {f"ta_i_sl_{s}": 0.0 for s in SIDES} | {f"mk_ta_{s}": 0 for s in SIDES})
    src.update(over)
    return src


def _one(cols: dict, name: str) -> float:
    return float(cols[name][0])


def test_masks_nan_always_has_a_reason_and_upstream_reasons_pass_through() -> None:
    cols = derive_net_columns(_sources(sl_pending_p1=NAN, mk_sl_p1=int(Mask.HISTORY_INCOMPLETE)))
    assert np.isnan(_one(cols, "net_pending_p1")) and _one(cols, "mk_sl_net_p1") == Mask.HISTORY_INCOMPLETE
    assert _one(cols, "mk_df_sl_net") == Mask.HISTORY_INCOMPLETE
    # 上流の mask が 0 なのに値が欠測 → 黙って mask=0 にせず UNOBSERVABLE
    cols = derive_net_columns(_sources(sl_inflight_p2=NAN))
    assert np.isnan(_one(cols, "net_pending_p2")) and _one(cols, "mk_sl_net_p2") == Mask.UNOBSERVABLE
    assert np.isnan(_one(cols, "ta_i_slnet_p1")) and _one(cols, "mk_ta_slnet_p1") == Mask.UNOBSERVABLE
    # 時間軸側の理由 (mk_ta) は有限でも引き継ぐ (bound 区間は mk_ta=UNPREDICTABLE のまま値は有限)
    cols = derive_net_columns(_sources(mk_ta_p1=int(Mask.UNPREDICTABLE), ta_i_sl_p1=3.0))
    assert np.isfinite(_one(cols, "ta_m_slnet_med_p1")) and _one(cols, "mk_ta_slnet_p1") == Mask.UNPREDICTABLE
    # 健全な行は 0。時間軸が非該当 (q 欠測) の側は mk_ta の理由を引き継ぐ
    ok = derive_net_columns(_sources(ta_i_sl_p1=3.0, sl_pending_p2=3.0))
    assert _one(ok, "mk_ta_slnet_p1") == 0 and _one(ok, "mk_sl_net_p1") == 0
    off = derive_net_columns(_sources(ta_q_med_p2=NAN, ta_q_fast_p2=NAN, ta_i_sl_p2=NAN, mk_ta_p2=int(Mask.NOT_APPLICABLE)))
    assert np.isnan(_one(off, "ta_i_slnet_p2")) and _one(off, "mk_ta_slnet_p2") == Mask.NOT_APPLICABLE


def test_ta_net_uses_the_receiver_side_and_gates_on_ta_i_sl() -> None:
    """攻撃側 p1 の I は受け側 p2 の net。ta_i_sl が欠測の行は net 版も欠測 (存在する場所が同じ)。"""
    cols = derive_net_columns(_sources(sl_pending_p1=0.0, sl_pending_p2=39.0, sl_inflight_p1=134.0, sl_inflight_p2=0.0,
                                       ta_i_sl_p1=39.0, ta_i_sl_p2=0.0, ta_q_med_p1=200.0, ta_q_fast_p1=150.0))
    assert _one(cols, "net_pending_p2") == 173.0 and _one(cols, "ta_i_slnet_p1") == 173.0   # 受け側 p2 の net
    assert _one(cols, "ta_m_slnet_med_p1") == 27.0 and _one(cols, "ta_recv_slnet_med_p1") == 0.0
    assert _one(cols, "ta_send_slnet_fast_p1") == 0.0 and _one(cols, "ta_recv_slnet_fast_p1") == 23.0
    assert _one(cols, "ta_i_slnet_p2") == 0.0 and _one(cols, "ta_unc72_slnet_med_p1") == 0.0
    gated = derive_net_columns(_sources(sl_inflight_p1=5.0, ta_i_sl_p1=NAN))
    assert np.isnan(_one(gated, "ta_i_slnet_p1")) and np.isfinite(_one(gated, "net_pending_p2"))


def test_derive_frame_handles_pandas_missing_and_empty_frames() -> None:
    """pandas の <NA> (Int8 の mask・欠測の実数) と 0 行でも落ちず、欠測は mask 付きの NaN になる。"""
    src = pd.DataFrame({k: pd.Series([0.0, 5.0, NAN], dtype="float64") for k in SOURCE_COLUMNS})
    for k in SOURCE_COLUMNS:
        if k.startswith("mk_"):
            src[k] = pd.array([0, 0, pd.NA], dtype="Int8")
    src["sl_pending_p1"], src["sl_inflight_p1"] = [0.0, 5.0, 1.0], [0.0, 0.0, 0.0]
    out = derive_frame(src)
    assert list(out.columns) == list(NET_LEDGER_COLUMN_NAMES)
    assert out["net_pending_p1"].tolist()[:2] == [0.0, 5.0] and out["mk_sl_net_p1"].tolist() == [0, 0, int(Mask.UNOBSERVABLE)]
    assert pd.isna(out["net_pending_p1"][2]) and str(out["mk_sl_net_p1"].dtype) == "Int8"
    empty = derive_frame(src.iloc[:0])
    assert len(empty) == 0 and list(empty.columns) == list(NET_LEDGER_COLUMN_NAMES)


def test_missing_source_columns_fail_loudly() -> None:
    src = _sources()
    del src["sl_inflight_p1"]
    with pytest.raises(KeyError, match="sl_inflight_p1"):
        derive_net_columns(src)


def test_scalar_and_array_paths_are_bit_identical() -> None:
    """配信 (1 行ずつ) と学習 (配列) が同じ値を返す。log1p など配列版の SIMD 経路でも 1 ビットも違わない。"""
    rng = np.random.default_rng(5)
    n = 3000
    src = _sources()
    arrays = {k: np.full(n, v, dtype=np.float64) for k, v in src.items()}
    arrays["sl_pending_p1"], arrays["sl_pending_p2"] = (rng.integers(0, 150, n).astype(float) for _ in range(2))
    arrays["sl_inflight_p1"], arrays["sl_inflight_p2"] = (rng.integers(0, 300, n).astype(float) for _ in range(2))
    arrays["sl_inflight_p1"][::17] = NAN
    arrays["ta_q_med_p1"] = rng.uniform(0, 400, n)
    arrays["ta_i_sl_p1"] = np.where(rng.random(n) < 0.8, arrays["sl_pending_p2"], NAN)
    vec = derive_net_columns(arrays)
    for r in range(0, n, 7):
        one = derive_net_columns({k: v[r] for k, v in arrays.items()})
        for name in NET_LEDGER_COLUMN_NAMES:
            assert np.array_equal(one[name], vec[name][r:r + 1], equal_nan=True), (name, r)


# ----------------------------------------------------------------------------- 配信と学習の一致 (build_features 経由)

def _snap(pending: tuple, inflight: tuple, **flags: bool) -> dict:
    snap: dict = {"available_at": 1.0, "rate": 70.0, "history_incomplete": False, "history_waiting": False, **flags}
    for i in (1, 2):
        for key in ("generated", "offset", "dropped", "carry", "observed_landing", "board_increase", "board_excess"):
            snap[f"{key}_p{i}"] = 0.0
        snap[f"pending_p{i}"], snap[f"inflight_p{i}"] = pending[i - 1], inflight[i - 1]
        snap[f"inflight_send_p{i}"] = max(0.0, inflight[i - 1] - pending[i - 1]) if np.isfinite(inflight[i - 1]) else NAN
    return snap


def _game_state(tmp_path: Path, snap: dict, attacker: int) -> GameState:
    """攻撃側 attacker が連鎖予測を持ち、時間軸の列が有限になる状態 (台帳 snapshot は直接与える)。"""
    gs = GameState(load_lean(build_record(tmp_path)))
    gs.t = gs.av_frame = 1.0
    for st in gs.sides:
        st.board, st.board_t = np.zeros((13, 6), dtype=np.int8), 1.0
        for _ in range(6):
            st.icon.push(1.0, {"icons": [], "total_count": 0}, True)     # アイコン補正を有効 (mk_ta=OK にする)
    a = gs.sides[attacker]
    a.on_fire_start({"fire_id": "a", "trigger_sec": 1.0, "observed_at_sec": 1.0, "chain_before": a.board.tolist()})
    a.episode.pred_l, a.episode.pred_score = 1, 40
    gs.sl_snap = snap
    return gs


CASES = [((0, 0), (0, 0), {}), ((0, 39), (134, 0), {}), ((7, 0), (3, 13), {}), ((5, 5), (3, 3), {}), ((0, 5), (10, 10), {}),
         ((72, 0), (0, 0), {}), ((12, 162), (150, 0), {}), ((3, 4), (0, 6), {}),
         ((NAN, NAN), (NAN, NAN), {"history_incomplete": True}), ((NAN, NAN), (NAN, NAN), {"history_waiting": True})]


def _use_specs(monkeypatch: pytest.MonkeyPatch, enable_net: bool) -> list:
    """読込時に決まるモジュール変数を、M3 (+案A) の列構成へ一時的に差し替える。"""
    specs = build_column_specs(enable_inflight=True, enable_net_ledger=enable_net)
    for module in (b3_column_spec, b3_rows):
        monkeypatch.setattr(module, "COLUMN_SPECS", specs)
        monkeypatch.setattr(module, "COLUMN_NAMES", [c.name for c in specs])
        monkeypatch.setattr(module, "SPEC_BY_NAME", {c.name: c for c in specs})
    return specs


def _rows(tmp_path: Path) -> list[dict]:
    rows = []
    for k, ((pending, inflight, flags), attacker) in enumerate(itertools.product(CASES, (0, 1))):
        gs = _game_state(tmp_path / f"c{k}", _snap(pending, inflight, **flags), attacker)
        gs.game_idx, gs.frame, gs.t = 1, k + 1, 1.0 + 0.5 * k
        rows.append(b3_rows._make_row(gs, TimingTable(), "v", 1, 1))
    return rows


def _frame(rows: list[dict]) -> pd.DataFrame:
    rows = [dict(r) for r in rows]
    b3_rows.finalize_weights(rows)
    return b3_rows.to_frame(b3_rows.with_mirrors(rows))


def test_off_rows_are_unchanged_and_on_rows_only_add_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_specs(monkeypatch, enable_net=False)
    off = _rows(tmp_path / "off")
    _use_specs(monkeypatch, enable_net=True)
    on = _rows(tmp_path / "on")
    assert len(off) == len(on) == 2 * len(CASES)
    for a, b in zip(off, on):
        assert not set(NET_LEDGER_COLUMN_NAMES) & set(a)                          # OFF の行に案A の列は無い
        assert set(b) - set(a) == set(NET_LEDGER_COLUMN_NAMES)                    # ON はそれだけを足す
        assert all(_same(a[k], b[k]) for k in a), [k for k in a if not _same(a[k], b[k])]   # 既存の値は 1 ビットも変わらない
    assert sum(np.isfinite(r["ta_m_slnet_med_p1"]) or np.isfinite(r["ta_m_slnet_med_p2"]) for r in on) >= len(CASES)


def test_live_rows_equal_frame_derivation_including_mirrored_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """配信の行 (build_features) の案A 列 = 同じ行の既存列から parquet 用の関数で導出した列。原向きも複製も。"""
    specs = _use_specs(monkeypatch, enable_net=True)
    full = _frame(_rows(tmp_path))
    names = list(NET_LEDGER_COLUMN_NAMES)
    assert list(full.columns[-len(names):]) == names and len(full.columns) == len(specs)
    derived = derive_frame(full.drop(columns=names))
    pd.testing.assert_frame_equal(derived, full[names], check_exact=True)       # 値も dtype も完全一致
    assert (full.orientation == 1).any() and (full.orientation == 0).any()      # 側交換した行を含む
    nan_cases = sum(1 for _p, _i, flags in CASES if flags)                       # 履歴不完全・待ちの行は欠測 (mask で理由)
    assert full["net_pending_p1"].notna().sum() == len(full) - nan_cases * 2 * 2   # 2 攻撃側 x 原向き/複製
    for c in specs:                                                               # NaN には必ず非0の理由 (既存規約)
        if c.mask and c.name in NET_LEDGER_COLUMN_NAMES:
            isna = full[c.name].isna().to_numpy()
            assert not (isna & (full[c.mask].astype(float).to_numpy() == 0)).any(), c.name


def test_live_rows_have_finite_ta_net_columns_where_ta_sl_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_specs(monkeypatch, enable_net=True)
    for row in _rows(tmp_path):
        for side in SIDES:
            assert np.isfinite(row[f"ta_i_sl_{side}"]) == np.isfinite(row[f"ta_i_slnet_{side}"])
            for v in NET_TA_VARIANTS:
                sl = v.replace("slnet", "sl")
                assert np.isfinite(row[f"ta_m_{sl}_{side}"]) == np.isfinite(row[f"ta_m_{v}_{side}"])
    # 受け側 p2 が 39 受けて p1 が 134 生成中: ta_i_sl は 39、net は 173。ここが修正点。
    gs = _game_state(tmp_path / "s1", _snap((0, 39), (134, 0)), attacker=0)
    gs.game_idx, gs.frame = 1, 1
    row = b3_rows._make_row(gs, TimingTable(), "v", 1, 1)
    assert (row["ta_i_sl_p1"], row["ta_i_slnet_p1"], row["net_pending_p2"], row["df_sl_net"]) == (39.0, 173.0, 173.0, -173.0)


# ----------------------------------------------------------------------------- 導出台本

def _write_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, pd.DataFrame]:
    _use_specs(monkeypatch, enable_net=True)
    full = _frame(_rows(tmp_path / "rows"))
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    full.drop(columns=list(NET_LEDGER_COLUMN_NAMES)).to_parquet(src_dir / "video_x.parquet", engine="pyarrow",
                                                                compression="zstd", index=False)
    (src_dir / "video_x.manifest.json").write_text(json.dumps({"attempt": "a1", "inputs": {"lean_sha256": "z"}}),
                                                   encoding="utf-8")
    return src_dir, full


def test_derive_script_appends_columns_without_touching_the_rest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import derive_b3_net_ledger_cols as script
    src_dir, full = _write_inputs(tmp_path, monkeypatch)
    out_dir = tmp_path / "out"
    assert script.main(["--input", str(src_dir), "--out-dir", str(out_dir), "--dry-run"]) == 0 and not out_dir.exists()
    assert script.main(["--input", str(src_dir), "--out-dir", str(out_dir)]) == 0
    before, after = pd.read_parquet(src_dir / "video_x.parquet"), pd.read_parquet(out_dir / "video_x.parquet")
    names = list(NET_LEDGER_COLUMN_NAMES)
    assert list(after.columns) == list(before.columns) + names                  # 末尾に足しただけ・順序保持
    pd.testing.assert_frame_equal(after[list(before.columns)], before, check_exact=True)       # 元の列は不変
    pd.testing.assert_frame_equal(after[names], full[names], check_exact=True)  # 配信の関数と同じ値
    manifest = json.loads((out_dir / "video_x.manifest.json").read_text(encoding="utf-8"))
    assert manifest["attempt"] == "a1" and manifest["inputs"] == {"lean_sha256": "z"}   # 元のキーは不変
    note = manifest[script.MANIFEST_KEY]
    assert note["added_columns"] == names and note["report"]["rows"] == len(full)
    assert note["report"]["finite_counts"]["net_pending_p1"] > 0 and (out_dir / "video_x.done").exists()
    assert note["source_sha256"] == script.sha256_file(src_dir / "video_x.parquet")


def test_derive_script_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import derive_b3_net_ledger_cols as script
    src_dir, full = _write_inputs(tmp_path, monkeypatch)
    out_dir = tmp_path / "out"
    script.main(["--input", str(src_dir / "video_x.parquet"), "--out-dir", str(out_dir)])
    with pytest.raises(FileExistsError):                                         # 既存出力は --force が要る
        script.main(["--input", str(src_dir), "--out-dir", str(out_dir)])
    assert script.main(["--input", str(src_dir), "--out-dir", str(out_dir), "--force"]) == 0
    with pytest.raises(ValueError, match="同じ"):                                 # 入力の上書きはしない
        script.main(["--input", str(src_dir), "--out-dir", str(src_dir), "--force"])
    with pytest.raises(ValueError, match="既にある"):                              # 二重に足さない
        script.main(["--input", str(out_dir), "--out-dir", str(tmp_path / "out2")])
    no_inflight = tmp_path / "noinfl"
    no_inflight.mkdir()
    full.drop(columns=[c for c in full.columns if c.startswith(("sl_inflight", "mk_sl_inflight"))
                       or c in NET_LEDGER_COLUMN_NAMES]).to_parquet(no_inflight / "v.parquet")
    with pytest.raises(KeyError, match="sl_inflight"):                           # 導出元が無い行は黙って埋めない
        script.main(["--input", str(no_inflight), "--out-dir", str(tmp_path / "out3")])
    with pytest.raises(FileNotFoundError):
        script.main(["--input", str(tmp_path / "none"), "--out-dir", str(tmp_path / "out4")])


# ----------------------------------------------------------------------------- フラグ配線 (読込時に決まる)

def _import_state(env: dict[str, str]) -> subprocess.CompletedProcess:
    code = ("import src.b3_column_spec as c;"
            "print(len(c.COLUMN_SPECS), c.NET_LEDGER_COLUMN_NAMES[0] in c.SPEC_BY_NAME, 'sl_inflight_p1' in c.SPEC_BY_NAME)")
    clean = {k: v for k, v in os.environ.items() if not k.startswith("PUYO_B3_")}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**clean, "PYTHONPATH": str(ROOT), **env},
                          capture_output=True, text=True, timeout=300)


def test_flag_wiring_default_off_and_env_on() -> None:
    off = _import_state({})
    assert off.stdout.split() == ["459", "False", "False"], off.stderr[-400:]
    m3 = _import_state({"PUYO_B3_M3": "1"})
    assert m3.stdout.split() == ["469", "False", "True"], m3.stderr[-400:]
    both = _import_state({"PUYO_B3_M3": "1", "PUYO_B3_NET_LEDGER_COLS": "1"})
    assert both.stdout.split() == ["501", "True", "True"], both.stderr[-400:]
    alone = _import_state({"PUYO_B3_NET_LEDGER_COLS": "1"})        # 前提の inflight 列が無いなら、黙らず落ちる
    assert alone.returncode != 0 and "sl_inflight" in alone.stderr


def test_training_feature_selection_picks_up_net_columns_and_skips_fast_ones() -> None:
    """案A ON の読込で、学習側の feature_columns が net の特徴 17 列だけを拾い、_fast 10 列と mask は拾わない。"""
    code = ("from src.b3_column_spec import COLUMN_NAMES, NET_LEDGER_COLUMN_NAMES, NET_LEDGER_FEATURE_NAMES;"
            "from src.b3_train_model import EXCLUDED_FAST_COLUMNS, feature_columns;"
            "net=[c for c in feature_columns(COLUMN_NAMES) if c in NET_LEDGER_COLUMN_NAMES];"
            "print(len(net), net==list(NET_LEDGER_FEATURE_NAMES), sum(c in EXCLUDED_FAST_COLUMNS for c in NET_LEDGER_COLUMN_NAMES))")
    clean = {k: v for k, v in os.environ.items() if not k.startswith("PUYO_B3_")}
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env={**clean, "PYTHONPATH": str(ROOT), "PUYO_B3_M3": "1", "PUYO_B3_NET_LEDGER_COLS": "1"})
    assert r.stdout.split() == ["17", "True", "10"], r.stderr[-400:]
    assert len(NET_LEDGER_FEATURE_NAMES) == 17 and not any("_fast" in n or n.startswith("mk_") for n in NET_LEDGER_FEATURE_NAMES)


def test_production_flag_default_is_off() -> None:
    from src import production_config
    assert production_config.B3_NET_LEDGER_COLS_DEFAULT is False and not production_config.b3_net_ledger_cols_enabled()


# ----------------------------------------------------------------------------- 実データの再現 (計装の評価行・5 場面)

@pytest.mark.skipif(not (DIAG_DIR / "scenes.json").exists(), reason="計装の評価行 (ledger_chain_diag_20261025) が無い")
def test_reproduces_the_diagnosis_numbers_on_real_evaluation_rows() -> None:
    from scripts import verify_b3_net_ledger_diag as diag
    result = diag.run(DIAG_DIR)
    ok = diag.reproduced(result)
    assert all(ok.values()), ok
    s = result["summary"]
    assert tuple(s["T1_net"]) == (98.2, 5336) and tuple(s["T1_icon_now"]) == (85.1, 4185)
    assert s["exact_rows"] == s["exact_net_hit"] == 4316 and s["both_sided"] == s["both_net_hit"] == 431
    assert all(abs(x["max_abs_diff_vs_dsum_recv"]) == 0.0 for x in result["scenes"])
