"""M0の参照復元・fold隔離・静止側接続の契約テスト。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import b3_m0_inputs as inputs


def input_fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[pd.DataFrame, np.ndarray]:
    raw = np.zeros((13, 6), dtype=np.int8)
    raw[0, 0], raw[12, 5] = 9, 10
    other = np.ones((13, 6), dtype=np.int8)
    frame = {"game_idx": 7, "frame": 30, "t_sec": 1.0,
             "sides": [{"confirmed_board": raw}, {"confirmed_board": other}]}
    monkeypatch.setattr(inputs, "iter_frames", lambda *a: iter([frame]))
    lean = SimpleNamespace(frame=np.array([29, 29]), side=np.array(["1P", "2P"]),
                           t_sec=np.array([0.9, 0.9]), queue=np.array([[1, 2, 9, 10], [3, 4, 5, 0]]))
    monkeypatch.setattr(inputs, "load_lean", lambda *a: lean)
    rows = []
    for orient in (0, 1):
        row = dict(game_idx=7, frame_idx=30, t_sec=1.0, orientation=orient)
        for side, name in enumerate(inputs.SIDES):
            original = side ^ orient
            row.update({f"av_board_{name}": 1.0, f"av_queue_{name}": 0.9,
                        f"lean_row_{name}": original, f"cells_total_{name}": (1, 78)[original]})
        rows.append(row)
    return pd.DataFrame(rows), raw


def test_recovery_preserves_hidden_row_and_swaps(monkeypatch: pytest.MonkeyPatch) -> None:
    df, raw = input_fixture(monkeypatch)
    boards, queues = inputs.recover_inputs(df, Path("unused"))
    np.testing.assert_array_equal(boards[0, 0], inputs.board_categories_from_raw(raw))
    np.testing.assert_array_equal(boards[0], boards[1, ::-1])
    np.testing.assert_array_equal(queues[0], queues[1, ::-1])
    assert queues[0, 0].tolist() == [1, 2, 0, 0]


def test_queue_millisecond_rounding_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    df, _ = input_fixture(monkeypatch)
    df["av_queue_p1"] = 0.9 - 1 / 3000
    inputs.recover_inputs(df, Path("unused"))


@pytest.mark.parametrize("column,value", [("av_board_p1", 2.0), ("av_board_p1", np.nan),
    ("av_board_p1", 0.8), ("cells_total_p1", 2), ("lean_row_p1", 1), ("av_queue_p1", 1.1)])
def test_invalid_references_fail(monkeypatch: pytest.MonkeyPatch, column: str, value: float) -> None:
    df, _ = input_fixture(monkeypatch)
    df.loc[0, column] = value
    with pytest.raises(ValueError):
        inputs.recover_inputs(df, Path("unused"))


def test_registered_fold_hash(tmp_path: Path) -> None:
    groups, folds = inputs.fold_maps()
    assert groups["video_c110"] == groups["video_c112"]
    assert folds["video_29"] == 3
    fake = tmp_path / "folds.tsv"
    fake.write_bytes(inputs.DEFAULT_FOLDS.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        inputs.fold_maps(fake)


def test_nested_oof_never_uses_outer_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src import b3_m0_training as training
    import torch
    n = 12
    df = pd.DataFrame({"row_key": [f"r{i}" for i in range(n)], "video_id": [f"v{i // 2}" for i in range(n)],
                       "game_idx": [0] * n, "won_p1": [0, 1] * (n // 2)})
    groups = {v: v for v in df.video_id}
    samples = training.canonical(df, np.zeros((n, 2, 13, 6), np.int8), np.zeros((n, 2, 4), np.int8), groups)
    folds = np.array([0, 0] + [1] * (n - 2))
    calls = []

    def fake_fit(train, test, *args) -> tuple[np.ndarray, dict]:
        assert "v0" not in train.source_groups
        assert not set(train.source_groups) & set(test.source_groups)
        calls.append((set(train.source_groups), set(test.source_groups)))
        return np.ones(len(test.labels)), {}

    monkeypatch.setattr(training, "fit_predict", fake_fit)
    values, records = training.nested_fold(samples, folds, 0, tmp_path, SimpleNamespace(), torch.device("cpu"))
    assert len(calls) == 4 and len(records) == 4
    assert np.isfinite(values).all()
    assert calls[-1][1] == {"v0"}
    assert sum(len(pred) for _, pred in calls[:-1]) == 5


def test_production_logit_and_game_weights() -> None:
    from src.b3_m0_training import canonical, production_logits
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(production_logits(p, 0.7), 0.7 * np.log(p / (1 - p)), atol=1e-14)
    df = pd.DataFrame(dict(row_key=["a", "b", "c"], video_id=["v"] * 3, game_idx=[0, 0, 1], won_p1=[0, 1, 1]))
    samples = canonical(df, np.zeros((3, 2, 13, 6), np.int8), np.zeros((3, 2, 4), np.int8), {"v": "g"})
    np.testing.assert_array_equal(samples.weights, [0.5, 0.5, 1])


def test_rcv_static_only_external_column(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import b3_train_rcv as rcv
    from src.b3_train_synth import make_rows
    from src.b3_eval_cv import TrainContext
    df, _ = make_rows(n_videos=2, games_per_video=2, rows_per_game=10)
    df = df.copy()
    df["video_id"] = df.video_id.map({"video_s00": "video_29", "video_s01": "video_30"})
    df["m0_outer_fold"] = 0
    df["m0_role"] = np.where(df.video_id == "video_30", "outer_test", "inner_train_oof")
    df["external"] = np.linspace(-2, 2, len(df))
    captured = []
    original = rcv.fit_two_state

    def fit(frame, columns, *args, **kwargs):
        captured.append(list(columns))
        return original(frame, columns, *args, **kwargs)

    monkeypatch.setattr(rcv, "fit_two_state", fit)
    train, test = df[df.video_id == "video_29"], df[df.video_id == "video_30"]
    predict, info = rcv.rcv_trainer(m0_column="external")(train, TrainContext(0, 1003, {}, {}, []))
    assert np.isfinite(predict(test)).all()
    assert set(rcv.M0_COLUMNS).issubset(captured[0])
    assert not set(rcv.M0_COLUMNS) & set(captured[1])
    assert "m0_logit" not in info["coverage_unmapped"]
    with pytest.raises(ValueError, match="役割"):
        predict(test.assign(m0_role="inner_train_oof"))
    with pytest.raises(ValueError, match="非有限"):
        predict(test.assign(external=np.nan))


def test_m0_join_is_keyed_and_keeps_attributes() -> None:
    from src.b3_train_rcv import _m0_attach
    df = pd.DataFrame({"row_key": ["a", "b"]})
    df.attrs["marker"] = "preserved"
    table = pd.DataFrame({"row_key": ["b", "a"], "m0_logit": [2, 1], "m0_outer_fold": [0, 0],
                          "m0_role": ["outer_test"] * 2}).set_index("row_key")
    joined = _m0_attach(df, table)
    assert joined.m0_logit.tolist() == [1, 2] and joined.attrs == df.attrs
    with pytest.raises(ValueError, match="重複"):
        _m0_attach(df, pd.concat([table, table]))


def test_smoke_artifact_cannot_enter_stage1(tmp_path: Path) -> None:
    import json
    from src.b3_train_rcv import _load_m0_table
    payload = {"smoke": True, "fold_sha256": inputs.FOLD_SHA256}
    (tmp_path / "fold0.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="試走"):
        _load_m0_table(tmp_path, 0)


def test_artifact_hash_and_keyed_load(tmp_path: Path) -> None:
    import json
    from src.b3_train_rcv import _load_m0_table
    path = tmp_path / "fold0.parquet"
    pd.DataFrame({"row_key": ["r"], "m0_logit": [1]}).to_parquet(path)
    payload = {"smoke": False, "fold_sha256": inputs.FOLD_SHA256, "logits_sha256": inputs.sha256_file(path)}
    (tmp_path / "fold0.json").write_text(json.dumps(payload), encoding="utf-8")
    assert _load_m0_table(tmp_path, 0).loc["r", "m0_logit"] == 1
    pd.DataFrame({"row_key": ["r"], "m0_logit": [2]}).to_parquet(path)
    with pytest.raises(ValueError, match="ハッシュ"):
        _load_m0_table(tmp_path, 0)


def test_tune_split_requires_disjoint_games_for_smoke() -> None:
    from src.b3_m0_training import canonical, tune_split
    df = pd.DataFrame(dict(row_key=["a", "b", "c", "d"], video_id=["v"] * 4,
                           game_idx=[0, 0, 1, 1], won_p1=[0, 1, 1, 0]))
    samples = canonical(df, np.zeros((4, 2, 13, 6), np.int8), np.zeros((4, 2, 4), np.int8), {"v": "g"})
    with pytest.raises(ValueError, match="不足"):
        tune_split(samples)
    fit, tune, policy = tune_split(samples, smoke=True)
    assert not set(samples.game_keys[fit]) & set(samples.game_keys[tune])
    assert policy == "smoke_only_game"


def test_stage1_rejects_different_source_rows(tmp_path: Path) -> None:
    import json
    from src.b3_train_rcv import validate_m0_sources
    source = tmp_path / "video.parquet"
    source.write_bytes(b"changed source")
    manifest = {"sources": {"video": {"rows_sha256": "old"}}}
    (tmp_path / "fold0.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="学習行ハッシュ"):
        validate_m0_sources(tmp_path, {"video": source}, [0])


def test_cli_preserves_production_recipe() -> None:
    from scripts.b3_m0_per_fold import parse_args
    args = parse_args(["--rows-dir", "rows", "--out", "out"])
    assert args.device == "cuda" and args.seed == 0 and args.epochs == 12 and args.batch_size == 512
    with pytest.raises(SystemExit):
        parse_args(["--rows-dir", "rows", "--out", "out", "--epochs", "1"])
    with pytest.raises(SystemExit):
        parse_args(["--rows-dir", "rows", "--out", "out", "--videos", "video_29"])
