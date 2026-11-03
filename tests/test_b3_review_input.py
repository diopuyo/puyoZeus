"""B3 レビュー入力形式 (src/b3_review_input.py) のテスト。"""
import math

import pytest

from src import b3_review_input as bri
from src.b3_review_synth import synth_rows


def test_roundtrip_csv_preserves_none_and_zero(tmp_path):
    rows = synth_rows()
    path = tmp_path / "in.csv"
    bri.save_review_input(path, rows)
    back = bri.load_review_input(path)
    assert len(back) == len(rows)
    # 0 と欠測 (None) を混同しない
    r = next(x for x in back if x["t_sec"] == 10.0)
    assert r["p1_sl_pending"] == 0.0 and r["p2_sl_generated"] is None
    assert next(x for x in back if x["t_sec"] == 3.5)["p_display"] is None


def test_validate_rejects_bad_rows():
    rows = synth_rows()
    assert bri.validate_rows(rows) == []
    rows[2]["p_display"] = 1.5
    rows[3]["mode"] = "xx"
    rows[4]["t_sec"] = 0.0
    msgs = " ".join(bri.validate_rows(rows))
    assert "[0,1]" in msgs and "mode" in msgs and "降順" in msgs


def test_from_b3_rows_maps_columns_and_does_not_invent():
    row = {"t_sec": 5.0, "frame_idx": 300, "video_id": "v", "game_idx": 2, "phase": 1,
           "ta_H_med_p1": 3.0, "ta_q_med_p1": 12.0, "sl_pending_p2": 6.0, "icon_corr_total_p2": 4.0,
           "ta_m_pred_med_p1": -2.0, "mk_ta_p1": 0, "mk_ta_p2": 8}
    out = bri.from_b3_rows(row, {"p_static": 0.4, "mode": "static", "p2_death_state": "suspect"})
    assert out["p1_H_med"] == 3.0 and out["p1_Q_med"] == 12.0 and out["p2_sl_pending"] == 6.0
    assert out["p1_M"] == -2.0            # sl 版が無ければ pred 版へ落ちる
    assert out["p1_I_sl"] is None and out["p2_death_state"] == "suspect"
    assert out["p1_death_state"] == "none" and out["p_static"] == 0.4
    assert set(out) == set(bri.INPUT_FIELDS)


def test_from_b3_rows_prefers_sl_margin():
    out = bri.from_b3_rows({"ta_m_sl_med_p1": 1.0, "ta_m_pred_med_p1": 9.0}, {})
    assert out["p1_M"] == 1.0


def test_from_b3_rows_iter_length_mismatch_raises():
    with pytest.raises(ValueError):
        bri.from_b3_rows_iter([{}], [])


def test_ledger_icon_gap_rules():
    assert bri.ledger_icon_gap({"p1_sl_pending": 0, "p1_icon_corr": 0}, "p1") is None
    assert bri.ledger_icon_gap({"p1_sl_pending": 6, "p1_icon_corr": None}, "p1") is None
    assert bri.ledger_icon_gap({"p1_sl_pending": 6, "p1_icon_corr": 4}, "p1") == 2


def test_derive_diff_and_missing_names():
    d = bri.derive({"p_display": 0.6, "p_ref_display": 0.5, "p1_mk_ta": 8, "missing_code": 99}, 7)
    assert math.isclose(d["p_diff_vs_ref"], 0.1) and d["frame_out"] == 7
    assert d["p1_missing_name"] == "該当なし" and d["missing_name"] == "不明(99)"
    assert d["p2_missing_name"] == "--"
    assert set(d) == set(bri.CSV_FIELDS)


def test_row_at_before_first_is_none_and_picks_last_le():
    rows = [{"t_sec": 1.0}, {"t_sec": 2.0}, {"t_sec": 3.0}]
    times = [1.0, 2.0, 3.0]
    assert bri.row_at(rows, times, 0.5) is None
    assert bri.row_at(rows, times, 2.0)["t_sec"] == 2.0
    assert bri.row_at(rows, times, 2.9)["t_sec"] == 2.0
    assert bri.row_at(rows, times, 99)["t_sec"] == 3.0


def test_missing_codes_match_column_spec():
    """欠測コードの語彙が B3_COLUMN_SPEC の Mask と一致 (測定器の健全性)。"""
    from src.b3_column_spec import Mask
    assert {int(m) for m in Mask} == set(bri.MISSING_NAMES)


def test_row_mapping_stems_exist_in_column_spec():
    """変換の語幹がすべて実在の b3_rows 列 (側接尾辞つき) であること。名前の取り違えを防ぐ。"""
    from src.b3_column_spec import COLUMN_NAMES
    names = set(COLUMN_NAMES)
    for stem in bri.ROWS_TO_SIDE:
        assert f"{stem}_p1" in names and f"{stem}_p2" in names, stem
    for key in bri.ROWS_TO_COMMON:
        assert key in names, key
