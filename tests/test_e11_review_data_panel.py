"""E11の表示専用データと既存描画からの独立性を検証する。"""
from __future__ import annotations

import csv
import inspect
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts import review_data_panel as panel
from scripts import visualize_advantage_overlay as vao
from src.board import Board
from src.chain_id_resolver import ChainIdResolver
from src.chain_id_resolver import ChainObservation, ObservationKind
from src.exchange_event_features import D_COLUMNS, SIDE_COLUMNS
from src.exchange_event_tracker import ExchangeChainRecord, ExchangeRecord
from tests.test_exchange_event_overlay import stub


def fixture_overlay() -> NS:
    """左右非対称の評価記録と確定盤面を用意する。"""
    chain = ExchangeChainRecord("1P", 1, 1.0, 1.0, formula_total=700,
                                score_ready_sec=2.0, score_delta=700)
    record = ExchangeRecord(1, 0, 1.0, chains=[chain], values=[
        dict(source="S1", p1=.6), dict(source="S3_provisional", p1=.7), dict(source="S3", p1=.8)])
    projection = dict(source="unavoidable_death", t_sec=2.0, p1=.98, base_p1=.8,
                      gfe_p1=.9, incoming=[0, 10], hands=[1, 2], near_future_send=[None, 3],
                      required_cancel=[0, 8], dead_sides=["2P"])
    tracker = NS(current=record, firing=NS(prefire_sides=np.full((2, len(SIDE_COLUMNS)), .3)),
                 source="unavoidable_death", probability=.98, _static_probability=.55,
                 _score_elapsed=0.0, resolver=ChainIdResolver())
    landing = NS(last=projection, death=projection, death_record=record, identity=(0, 1))
    histories = [[NS(t_sec=1.0, board=Board())], [NS(t_sec=1.5, board=Board())]]
    return NS(tracker=tracker, _landing_projection=landing, _history=histories)


def review_row(overlay: NS | None = None) -> dict:
    """重い特徴計算を避け、行の配線だけを検証する。"""
    result = NS(p1=NS(state=NS(name="CHAIN")), p2=NS(state=NS(name="STABLE")))
    snap = NS(forecast_p1=2, forecast_p2=8, net_balance_capped=6)
    return panel.build_review_row(overlay or fixture_overlay(), result, snap, 90, 3, 0, .85, 70,
        lambda *args: {}, lambda own, opp, net, forecast: {name: net for name in D_COLUMNS})


def test_shared_row_probabilities_and_missing() -> None:
    row = review_row()
    assert set(row) == set(panel.CSV_FIELDS)
    assert row["p1_G_fe"] == .55 and row["p1_S3"] == .8
    assert row["p1_selected"] == .98 and row["p1_display"] == .85
    assert .8 < row["p1_combined"] < .9
    assert row["1P_near_future_send"] is None
    assert row["2P_near_future_send"] == 3
    assert row["2P_death_reason"] == "RESPONSE_SHORTFALL"
    assert row["2P_confirmed_incoming"] == 10
    assert row["2P_incoming_rows"] == 10 / 6
    assert row["1P_formula_total"] == 700 and row["1P_formula_send"] == 10
    assert row["1P_d_ukeyasusa"] == 6 and row["2P_d_ukeyasusa"] == -6


def test_closed_exchange_does_not_leak_projection() -> None:
    overlay = fixture_overlay()
    overlay.tracker.current = overlay._landing_projection.death_record = None
    overlay._landing_projection.death = None
    row = review_row(overlay)
    assert row["exchange_id"] is None and row["p1_S3"] is None
    assert row["p1_landing_gfe"] is None and row["2P_incoming"] is None
    assert row["2P_k1"] is None


def test_chain_step_and_unconfirmed_amount_are_distinct() -> None:
    overlay = fixture_overlay()
    overlay.tracker.resolver.push(ChainObservation("1P", 1.0, ObservationKind.FORMULA_STEP,
                                                   chain_count=4, total_score=700))
    chain = overlay.tracker.current.chains[0]
    chain.chain_id = overlay.tracker.resolver.active()[0].chain_id
    chain.score_ready_sec = None
    row = review_row(overlay)
    assert row["1P_chain_count"] == 4
    assert row["1P_formula_send"] == 10
    assert row["1P_confirmed_send"] is None
    assert row["2P_confirmed_send"] == 0
    assert row["2P_confirmed_incoming"] is None


@pytest.mark.parametrize("layout", ["overlay", "panel"])
def test_generate_review_writer_size(monkeypatch: pytest.MonkeyPatch,
                                     tmp_path: Path, layout: str) -> None:
    stub(monkeypatch)
    sizes, frames = [], []

    def writer(path: str, codec: int, fps: float, size: tuple) -> NS:
        sizes.append(size)
        return NS(write=lambda frame: frames.append(frame.shape[:2][::-1]), release=lambda: None)

    monkeypatch.setattr(vao.cv2, "VideoWriter", writer)
    vao.generate(Path("short.mp4"), tmp_path / "review.mp4", .1, 0,
                 layout=layout, review_data_panel=True, enable_exchange_event_update=True,
                 exchange_event_m0_predictor=lambda b, q: .5)
    original = (vao.OUT_W, vao.CANVAS_H) if layout == "overlay" else (vao.PANEL_CANVAS_W, vao.PANEL_CANVAS_H)
    graph_height = (vao.GRAPH_H if layout == "overlay" else vao.panel_layout_regions()["graph"][3])
    assert sizes == [panel.panel_size(original[0], original[1] + graph_height)]
    assert frames == sizes * 3


def test_csv_round_trip(tmp_path: Path) -> None:
    path, row = tmp_path / "nested/review.csv", review_row()
    writer = panel.ReviewCsv(path)
    writer.write(row)
    writer.close()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 1 and float(saved[0]["p1_display"]) == row["p1_display"]
    assert saved[0]["1P_near_future_send"] == ""


@pytest.mark.parametrize("width,height", [(1280, 1110), (1920, 1080)])
def test_panel_keeps_original_pixels_and_readable_font(width: int, height: int) -> None:
    original = np.full((height, width, 3), 71, dtype=np.uint8)
    drawn = panel.draw_review_panel(original, review_row(), vao.JP_LABEL, vao._font)
    assert drawn.shape[:2][::-1] == panel.panel_size(width, height)
    np.testing.assert_array_equal(drawn[:height], original)
    assert panel.FONT_SIZE >= 12
    common, columns = panel.panel_lines(review_row(), vao.JP_LABEL)
    font = vao._font(panel.FONT_SIZE)
    assert max(font.getlength(line) for line in common) <= 1280 - 2 * panel.PADDING
    assert max(font.getlength(line) for lines in columns for line in lines) <= 640 - 2 * panel.PADDING
    assert panel.PADDING + (len(common) + len(columns[0])) * panel.LINE_HEIGHT <= panel.PANEL_HEIGHT


@pytest.mark.parametrize("flag", ["review_data_panel", "review_data_csv"])
def test_review_requires_new_evaluator(flag: str, tmp_path: Path) -> None:
    value = True if flag == "review_data_panel" else tmp_path / "review.csv"
    with pytest.raises(ValueError, match="exchange-event-update"):
        vao.generate(Path("missing.mp4"), tmp_path / "out.mp4", 1, 0, **{flag: value})


def test_defaults_off() -> None:
    parameters = inspect.signature(vao.generate).parameters
    assert parameters["review_data_panel"].default is False
    assert parameters["review_data_csv"].default is None


def test_generate_csv_excludes_warmup_and_preserves_display(monkeypatch: pytest.MonkeyPatch,
                                                           tmp_path: Path) -> None:
    stub(monkeypatch)
    options = dict(start_sec=1, warmup_sec=1, render=False, enable_exchange_event_update=True,
                   exchange_event_m0_predictor=lambda b, q: .5)
    for suffix in ("plain", "review"):
        extra = {} if suffix == "plain" else {"review_data_csv": tmp_path / "review.csv"}
        vao.generate(Path("short.mp4"), tmp_path / f"{suffix}.mp4", 2, 0,
                     dump_display_timeline_path=tmp_path / f"{suffix}.npz", **options, **extra)
    with np.load(tmp_path / "plain.npz") as plain, np.load(tmp_path / "review.npz") as review:
        for name in plain.files:
            np.testing.assert_array_equal(plain[name], review[name])
    with (tmp_path / "review.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 60 and float(rows[0]["t_sec"]) == 1
    assert int(rows[-1]["frame_index"]) == 89
