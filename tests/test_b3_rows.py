"""B3 行生成器の前提検査: (a) リーク監査 (b) prefix 不変 (c) 側交換の可換性 (d) schema/mask (e) 行キー一意。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.b3_column_spec import COLUMN_NAMES, COLUMN_SPECS, Kind, swap_sides
from src.b3_rows import AV_EPS, finalize_weights, generate_rows, mirror_row, to_frame, with_mirrors
from src.b3_timing import TimingTable
from tests.b3_synth import FIRE_FRAME, N_FRAMES, build_record

FUTURE_DEPENDENT = {"held_sec", "weight", "won_p1"}
B2 = Path("/mnt/d/puyo_analyzer/wt_b2/data/verify/recollect_2026-10-03") if os.name != "nt" else Path("D:/puyo_analyzer/wt_b2/data/verify/recollect_2026-10-03")
REAL_FRAMES = 12000


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    rec = build_record(root / "a")
    rec_swap = build_record(root / "b", swap=True)
    table = TimingTable()
    rows, info = generate_rows(rec, "synth", table)
    rows_swap, _ = generate_rows(rec_swap, "synth", table)
    return rec, rec_swap, table, rows, rows_swap, info


def _feature_cols():
    return [c.name for c in COLUMN_SPECS if c.role not in ("label",) and c.name not in FUTURE_DEPENDENT and c.name != "row_key"]


def _same(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if a is None or b is None:
        return a is b
    if np.isnan(a) and np.isnan(b):
        return True
    return a == pytest.approx(b, rel=1e-9, abs=1e-9)


def test_rows_exist_and_cover_triggers(synth):
    _, _, _, rows, _, info = synth
    assert len(rows) > 50 and info["frames_in_match"] + info["excluded_outside_match_frames"] == N_FRAMES
    assert info["leak_violations"] == 0
    masks = {r["trigger_mask"] for r in rows}
    assert any(m & 1 for m in masks) and any(m & 4 for m in masks) and any(m & 64 for m in masks)


def test_a_leak_audit_available_at_not_after_t(synth):
    _, _, _, rows, _, _ = synth
    assert all(r["av_max"] <= r["t_sec"] + AV_EPS for r in rows)          # 母数 = 全行
    assert all(r["av_frame"] <= r["t_sec"] + AV_EPS for r in rows)
    assert sum(np.isfinite(r["av_max"]) for r in rows) == len(rows)


def test_b_prefix_invariance(synth):
    rec, _, table, rows, _, _ = synth
    cut = FIRE_FRAME + 120
    cut_rows, _ = generate_rows(rec, "synth", table, max_frame=cut)
    full_prefix = [r for r in rows if r["frame_idx"] <= cut]
    assert len(cut_rows) == len(full_prefix) > 20
    for a, b in zip(cut_rows, full_prefix):
        assert a["row_key"] == b["row_key"]
        for c in _feature_cols():
            assert _same(a[c], b[c]), (c, a[c], b[c])


def test_c_side_swap_commutes_with_generation(synth):
    """1P/2P を入れ替えた記録から作った行 = 元の行を台帳規則で交換した行。"""
    _, _, _, rows, rows_swap, _ = synth
    assert len(rows) == len(rows_swap)
    for a, b in zip(rows, rows_swap):
        sw = swap_sides(a)
        assert a["frame_idx"] == b["frame_idx"]
        for c in _feature_cols():
            if c in ("orientation",):
                continue
            assert _same(sw[c], b[c]), (c, sw[c], b[c])


def test_c2_mirror_twice_is_identity_and_weights(synth):
    _, _, _, rows, _, _ = synth
    rows = [dict(r) for r in rows]
    finalize_weights(rows)
    full = with_mirrors(rows)
    assert len({r["row_key"] for r in full}) == len(full)                      # (e) 行キー一意
    back = mirror_row(mirror_row(rows[5]))
    assert all(_same(back[c], rows[5][c]) for c in COLUMN_NAMES if c != "won_p1" or True)
    total = sum(r["weight"] for r in full)
    assert total == pytest.approx(1.0)                                         # 試合内重み合計1 (1試合のみ)
    orig = sum(r["weight"] for r in full if r["orientation"] == 0)
    assert orig == pytest.approx(0.5)


def test_d_schema_and_mask_consistency(synth):
    _, _, _, rows, _, _ = synth
    rows = [dict(r) for r in rows]
    finalize_weights(rows)
    df = to_frame(with_mirrors(rows))
    assert list(df.columns) == COLUMN_NAMES
    bad = 0
    checked = 0
    for c in COLUMN_SPECS:
        if c.mask and c.name in df.columns:
            isna = df[c.name].isna().to_numpy()
            mk = df[c.mask].astype("float").to_numpy()
            checked += int(isna.sum())
            bad += int(((isna) & (mk == 0)).sum())                              # NaN には必ず非0の理由
    assert checked > 0 and bad == 0, (checked, bad)
    assert df["won_p1"].notna().all()


def test_ledger_not_in_feature_role(synth):
    _, _, _, rows, _, _ = synth
    assert all(c.role == "R_only" for c in COLUMN_SPECS if c.name.startswith("r_ledger"))


def test_a2b_second_leg_is_deferred(synth):
    _, _, _, rows, _, _ = synth
    assert all(r["mk_ta2_p1"] == 9 and np.isnan(r["ta2_q_med_p1"]) for r in rows)


def test_user_phase_and_dead_board_not_zeroed(synth):
    _, _, _, rows, _, _ = synth
    assert all(r["phase"] in (0, 1, 2) for r in rows)
    assert all(r["cells_total_p1"] <= 72 for r in rows if np.isfinite(r["cells_total_p1"]))


@pytest.mark.skipif(not (B2 / "receipts" / "video_51.json").exists(), reason="B2 実データ無し")
def test_real_video51_prefix_invariance_and_leak():
    from src.b3_source import resolve_attempt
    p = resolve_attempt("video_51")
    t = TimingTable()
    rows, info = generate_rows(p.record, "video_51", t, max_frame=REAL_FRAMES, require_winner=False)
    cut = REAL_FRAMES - 1500
    rows_cut, _ = generate_rows(p.record, "video_51", t, max_frame=cut, require_winner=False)
    pref = [r for r in rows if r["frame_idx"] <= cut]
    assert info["leak_violations"] == 0 and len(rows) > 200
    assert sum(r["mk_sl_p1"] == 0 for r in rows) > 50       # 台帳v2 の出力が行に届いている (口の配線)
    assert len(pref) == len(rows_cut)
    for a, b in zip(rows_cut, pref):
        for c in _feature_cols():
            assert _same(a[c], b[c]), (c, a["frame_idx"])


@pytest.mark.skipif(not (B2 / "receipts" / "video_51.json").exists(), reason="B2 実データ無し")
def test_real_video51_nan_always_has_mask_reason():
    """実データ区間で『NaN なのに mask=0』が無い (合成では両側盤面が揃うため漏れた欠陥の再発防止)。"""
    from src.b3_source import resolve_attempt
    p = resolve_attempt("video_51")
    rows, _ = generate_rows(p.record, "video_51", TimingTable(), max_frame=REAL_FRAMES, require_winner=False)
    for r in rows:
        r["held_sec"], r["weight"] = 0.1, 0.1
    df = to_frame(with_mirrors(rows))
    cells, bad = 0, {}
    for c in COLUMN_SPECS:
        if c.mask:
            isna = df[c.name].isna().to_numpy()
            mk = df[c.mask].astype(float).to_numpy()
            cells += int(isna.sum())
            n = int((isna & (mk == 0)).sum())
            if n:
                bad[c.name] = n
    assert cells > 0 and not bad, bad
