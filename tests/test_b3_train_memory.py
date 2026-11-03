"""省メモリ経路の予測不変性と列・行の共有（合成データのみ）。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.b3_column_spec import COLUMN_NAMES, COLUMN_SPECS, EXCLUDED_FAST_COLUMNS
from src.b3_eval_cv import TrainContext
from src.b3_train_data import load_training_rows
from src.b3_train_matrix import MATRIX_ATTR, available_columns, prepare_frame
from src.b3_train_model import DEFAULT_GRID, feature_columns, fit_hgb, fit_two_state, predict_hgb, to_matrix
from src.b3_train_rcv import rcv_trainer
from src.b3_train_synth import make_rows


def _legacy_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], cfg: object) -> np.ndarray:
    """変更前の状態別float32変換・全NaN除外の手順を独立に再現する。"""
    out = np.full(len(test), np.nan)
    for mode in (0, 1):
        tr, sel = train[train.mode_exchange == mode], test.mode_exchange.to_numpy() == mode
        X = np.empty((len(tr), len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            X[:, j] = tr[c].to_numpy(dtype=np.float32, na_value=np.nan)
        keep = ~np.all(np.isnan(X), axis=0)
        model = fit_hgb(X[:, keep], tr.won_p1.to_numpy(dtype=np.float64),
                        tr.weight.to_numpy(dtype=np.float64), cfg, 1003)
        tx = test.loc[sel, np.array(cols)[keep]].to_numpy(dtype=np.float32, na_value=np.nan)
        out[sel] = predict_hgb(model, tx)
    return out


@pytest.mark.parametrize("cfg", DEFAULT_GRID, ids=lambda c: c.name)
def test_memory_predictions_bitwise(cfg: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B3_HGB_THREADS", "8")
    df, _ = make_rows(n_videos=10, games_per_video=2, rows_per_game=45)
    # 列除外によるモデル変更と、省メモリ化の数値不変性を分ける。速い列も含めて検証。
    cols = [s.name for s in COLUMN_SPECS if s.role == "feature" and s.name != "mode_exchange"]
    tr, te = df[df.video_id < "video_s07"], df[df.video_id >= "video_s07"]
    expected = _legacy_predict(tr, te, cols, cfg)
    shared = prepare_frame(df, cols)
    actual = fit_two_state(shared.loc[tr.index], cols, cfg, 1003).predict(shared.loc[te.index])
    assert actual.tobytes() == expected.tobytes()


def test_excluded_fast_columns_are_complete() -> None:
    dependent = {f"{stem}_{s}" for stem in ("ta_req_h", "ta_done_h", "ta_trunc") for s in ("p1", "p2")}
    expected = [c for c in COLUMN_NAMES if c.startswith("ta_") and ("_fast" in c or c in dependent)]
    assert list(EXCLUDED_FAST_COLUMNS) == expected
    assert len(expected) == 44
    assert not set(expected) & set(feature_columns(COLUMN_NAMES, include_mask_codes=True))
    assert "ta_H_med_p1" in feature_columns(COLUMN_NAMES)
    path = Path(__file__).resolve().parents[1] / "docs/b3_prereg_fill/excluded_fast_columns.txt"
    assert path.read_text(encoding="utf-8").splitlines() == expected


def test_projected_loader_shares_matrix_and_rcv(tmp_path: Path) -> None:
    df, gmap = make_rows(n_videos=10, games_per_video=2, rows_per_game=24)
    paths = {}
    for vid, sub in df.groupby("video_id", sort=True):
        paths[vid] = tmp_path / f"{vid}.parquet"
        sub.to_parquet(paths[vid], index=False)
    cols = feature_columns(df.columns)
    shared = load_training_rows(paths, cols)
    original = pd.concat([df[df.video_id == v] for v in sorted(paths)], ignore_index=True)
    # parquetはNaNの符号を正規化するため、入力の欠測は意味で比較する。
    assert np.array_equal(to_matrix(shared, cols), to_matrix(original, cols), equal_nan=True)
    sub = shared[shared.video_id < "video_s07"]
    assert sub.attrs[MATRIX_ATTR] is shared.attrs[MATRIX_ATTR]
    assert not sub.attrs[MATRIX_ATTR].values.flags.writeable
    assert not set(EXCLUDED_FAST_COLUMNS) & set(available_columns(shared))
    ctx = TrainContext(0, 1003, gmap, {}, cols)
    legacy, info_old = rcv_trainer()(original[original.video_id < "video_s07"], ctx)
    optimized, info_new = rcv_trainer()(sub, ctx)
    test = shared.video_id >= "video_s07"
    assert info_old == info_new
    assert legacy(original[test]).tobytes() == optimized(shared[test]).tobytes()


def test_probe_stops_before_any_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.b3_train_memory_probe import measure
    from src import b3_train_model
    df, gmap = make_rows(n_videos=10, games_per_video=2, rows_per_game=24)
    cols = feature_columns(df.columns)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("メモリ計測で予測を実行してはならない")

    monkeypatch.setattr(b3_train_model, "predict_hgb", forbidden)
    report = measure(prepare_frame(df, cols), gmap, cols)
    assert report["stopped_before_first_tree"]
    assert report["fit_input"]["dtype"] == "float64"
