"""B3 レビュー帯の描画テスト (src/b3_review_panel.py)。"""
import numpy as np

from src import b3_review_panel as pan
from src.b3_review_synth import synth_rows


def _frame(w=1280, h=720):
    return np.full((h, w, 3), 40, np.uint8)


def test_panel_adds_band_and_keeps_original_pixels():
    rows = synth_rows()
    series = pan.GraphSeries.from_rows(rows)
    f = _frame()
    out = pan.draw_b3_panel(f, rows[20], series, 10.0)
    assert out.shape[1] == 1280 and out.shape[0] == pan.panel_size(1280, 720)[1]
    assert np.array_equal(out[:720], f)              # 元画像に触れない
    assert out[720:].std() > 5                       # 帯に何か描かれている


def test_panel_scales_with_width_and_handles_none_row():
    series = pan.GraphSeries.from_rows(synth_rows())
    out = pan.draw_b3_panel(_frame(640, 360), None, series, -1.0)
    assert out.shape[:2] == (pan.panel_size(640, 360)[1], 640)


def test_lines_show_missing_as_dashes_not_zero():
    r = synth_rows()[0]
    text = " ".join(pan.common_lines(r) + pan.side_lines(r, "p1"))
    assert "F_exchange --" in text and "H 中央値/速い --/--" in text
    r2 = synth_rows()[20]
    assert "予告 0 " in " ".join(pan.side_lines(r2, "p1"))   # 0 は '--' にならない


def test_side_lines_count_matches_constant():
    r = synth_rows()[20]
    assert len(pan.side_lines(r, "p1")) == pan.SIDE_LINES
    assert len(pan.common_lines(r)) == pan.COMMON_LINES


def test_death_basis_and_ref_diff_are_displayed():
    rows = synth_rows()
    dead = next(r for r in rows if r["t_sec"] == 18.0)
    text = " ".join(pan.side_lines(dead, "p2"))
    assert "証明済み死亡" in text and "相殺不能の予告72個以上" in text
    mid = next(r for r in rows if r["t_sec"] == 10.0)
    assert "差(B3表示−R)" in pan.common_lines(mid)[2] and "pt" in pan.common_lines(mid)[2]


def test_graph_gap_is_not_interpolated():
    rows = synth_rows()
    series = pan.GraphSeries.from_rows(rows)
    i = [r["t_sec"] for r in rows].index(3.5)
    assert np.isnan(series.b3[i])                    # 欠測は NaN のまま (補間しない)
    img = pan.draw_graph(1280, series, 3.5, pan.default_font)
    assert img.size == (1280, pan.GRAPH_HEIGHT)
