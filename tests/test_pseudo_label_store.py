"""LabelStore + PseudoLabelSample のテスト."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import (
    COMPONENT_NEXT,
    COMPONENT_SCORE,
    PseudoLabelSample,
)


def _make_sample(
    component: str = COMPONENT_SCORE,
    timestamp: float = 1.0,
    label: int = 5,
    confidence: float = 0.95,
) -> PseudoLabelSample:
    patch = np.zeros((50, 40, 3), dtype=np.uint8)
    return PseudoLabelSample(
        component=component,
        timestamp=timestamp,
        input_data={"patch": patch},
        label=label,
        confidence=confidence,
        metadata={"frame_idx": 100, "side": "1P"},
    )


def test_pseudo_label_to_jsonable_roundtrip():
    """to_jsonable / from_jsonable が情報保存して復元できること."""
    s = _make_sample()
    obj = s.to_jsonable()
    s2 = PseudoLabelSample.from_jsonable(obj)
    assert s2.component == s.component
    assert s2.timestamp == pytest.approx(s.timestamp)
    assert s2.label == s.label
    assert s2.confidence == pytest.approx(s.confidence)
    # ndarray も復元される
    assert isinstance(s2.input_data["patch"], np.ndarray)
    assert s2.input_data["patch"].shape == (50, 40, 3)
    assert s2.metadata["frame_idx"] == 100


def test_pseudo_label_with_nested_data():
    """tuple / dict / numpy scalar の混在も serialize できる."""
    s = PseudoLabelSample(
        component=COMPONENT_NEXT,
        timestamp=2.5,
        input_data={
            "patch_top": np.ones((10, 10, 3), dtype=np.uint8),
            "side": "2P",
        },
        label={"top_color": 1, "bot_color": 2},
        confidence=0.99,
        metadata={"delta_counts": {1: 1, 2: 1}},
    )
    obj = s.to_jsonable()
    j = json.dumps(obj)  # 完全 JSON 化できること
    assert "1" in j
    s2 = PseudoLabelSample.from_jsonable(json.loads(j))
    assert s2.label["top_color"] == 1


def test_label_store_append_and_load(tmp_path: Path):
    """append / load の往復確認."""
    store = LabelStore(video_id="video_test01", root=tmp_path)
    samples = [_make_sample(label=i) for i in range(3)]
    store.append(samples)
    loaded = list(store.load(COMPONENT_SCORE))
    assert len(loaded) == 3
    labels = sorted(s.label for s in loaded)
    assert labels == [0, 1, 2]


def test_label_store_per_component_separation(tmp_path: Path):
    """component 別に別ファイルに保存されること."""
    store = LabelStore(video_id="vidB", root=tmp_path)
    samples = [
        _make_sample(component=COMPONENT_SCORE, label=1),
        _make_sample(component=COMPONENT_NEXT, label={"top_color": 1, "bot_color": 2}),
    ]
    store.append(samples)
    score_file = tmp_path / "vidB" / "score.jsonl"
    next_file = tmp_path / "vidB" / "next.jsonl"
    assert score_file.is_file()
    assert next_file.is_file()
    assert score_file.read_text(encoding="utf-8").count("\n") == 1
    assert next_file.read_text(encoding="utf-8").count("\n") == 1


def test_label_store_stats(tmp_path: Path):
    """stats が件数を正しく返すこと."""
    store = LabelStore(video_id="vidC", root=tmp_path)
    store.append([_make_sample(label=i) for i in range(5)])
    store.append([_make_sample(component=COMPONENT_NEXT,
                                label={"top_color": 1, "bot_color": 2})])
    stats = store.stats()
    assert stats["score"] == 5
    assert stats["next"] == 1


def test_label_store_load_missing_component_returns_empty(tmp_path: Path):
    """存在しない component を load すると空."""
    store = LabelStore(video_id="vidD", root=tmp_path)
    out = list(store.load(COMPONENT_SCORE))
    assert out == []


def test_label_store_skip_corrupt_lines(tmp_path: Path):
    """破損 JSON 行は silent skip される."""
    store = LabelStore(video_id="vidE", root=tmp_path)
    store.append([_make_sample(label=1)])
    # 破損行を追加
    path = tmp_path / "vidE" / "score.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write("not a json line\n")
        f.write('{"incomplete": true\n')
    loaded = list(store.load(COMPONENT_SCORE))
    assert len(loaded) == 1


def test_label_store_list_videos(tmp_path: Path):
    """list_videos で video_id 一覧."""
    LabelStore("vid_a", root=tmp_path)
    LabelStore("vid_b", root=tmp_path)
    videos = LabelStore.list_videos(root=tmp_path)
    assert set(videos) == {"vid_a", "vid_b"}


def test_label_store_aggregate_stats(tmp_path: Path):
    """aggregate_stats: 全 video の component 別件数."""
    s1 = LabelStore("v1", root=tmp_path)
    s1.append([_make_sample(label=1), _make_sample(label=2)])
    s2 = LabelStore("v2", root=tmp_path)
    s2.append([_make_sample(component=COMPONENT_NEXT,
                             label={"top_color": 1, "bot_color": 2})])
    agg = LabelStore.aggregate_stats(root=tmp_path)
    assert agg["v1"]["score"] == 2
    assert agg["v2"]["next"] == 1


def test_label_store_video_id_required():
    """video_id 空は ValueError."""
    with pytest.raises(ValueError):
        LabelStore(video_id="")
