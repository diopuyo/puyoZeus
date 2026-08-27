"""scripts/_ab_stage_compare_2026-08-18.py の軽量単体テスト。

このスクリプトの主目的 (CSVビルド・学習) は重い処理のため、ここでは
純粋関数 (選定ロジック・旧CSV抽出・npzスナップショット) のみを、実物の
npz/CSVを使わない小さな合成データで検証する (memory
`feedback_viz_eval_required.md` の対象外、数値ロジックのユニットテスト)。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def mod():
    """ハイフン入りファイル名のためモジュールとして直接ロードする
    (tests/test_build_chain_anim_duration_table_2026-08-14.py と同じ方式)。"""
    path = Path(__file__).resolve().parent.parent / "scripts" / (
        "_ab_stage_compare_2026-08-18.py"
    )
    spec = importlib.util.spec_from_file_location("_ab_stage_compare_for_test", path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["_ab_stage_compare_for_test"] = m
    spec.loader.exec_module(m)
    return m


def _make_ok_row(target_id: str, finished_at: str) -> dict:
    return {
        "target_id": target_id, "video_id": f"vid_{target_id}", "tier": "マスター",
        "collect_status": "OK", "rows": "100", "finished_at": finished_at,
    }


def test_select_paired_target_ids_excludes_broken_and_missing_old(mod):
    ok_rows = [
        _make_ok_row("1", "2026-08-18 07:00:00"),
        _make_ok_row("2", "2026-08-18 08:00:00"),
        _make_ok_row("c26", "2026-08-18 09:00:00"),  # BROKEN_VIDEOS
        _make_ok_row("3", "2026-08-18 10:00:00"),
        _make_ok_row("99", "2026-08-18 11:00:00"),  # 旧CSVに無い (新規動画)
    ]
    known_old_video_ids = {"video_1", "video_2", "video_3", "video_c26"}
    broken_videos = ("c26", "c30", "c58", "c69")

    def _npz_exists(npz_dir, tid):
        return True

    class DummyPath:
        def __init__(self, tid):
            self.tid = tid

        def exists(self):
            return True

    class DummyDir:
        def __truediv__(self, name):
            return DummyPath(name)

    selected = mod.select_paired_target_ids(
        ok_rows, DummyDir(), known_old_video_ids, broken_videos, min_ok=3,
    )
    assert selected == ["1", "2", "3"]

    selected_too_many = mod.select_paired_target_ids(
        ok_rows, DummyDir(), known_old_video_ids, broken_videos, min_ok=4,
    )
    assert selected_too_many is None


def test_count_paired_candidates(mod):
    ok_rows = [
        _make_ok_row("1", "2026-08-18 07:00:00"),
        _make_ok_row("c30", "2026-08-18 08:00:00"),
    ]
    known_old_video_ids = {"video_1", "video_c30"}
    broken_videos = ("c26", "c30", "c58", "c69")

    class DummyPath:
        def exists(self):
            return True

    class DummyDir:
        def __truediv__(self, name):
            return DummyPath()

    n = mod.count_paired_candidates(ok_rows, DummyDir(), known_old_video_ids, broken_videos)
    assert n == 1  # c30 は BROKEN_VIDEOS のため除外


def test_extract_old_csv_subset(mod, tmp_path):
    old_csv = tmp_path / "old.csv"
    df = pd.DataFrame({
        "video_id": ["video_1", "video_1", "video_2", "video_3"],
        "won": [1, 0, 1, 0],
        "some_col": [0.1, 0.2, 0.3, 0.4],
    })
    df.to_csv(old_csv, index=False)
    out_csv = tmp_path / "subset.csv"

    n_rows = mod.extract_old_csv_subset(old_csv, ["1", "2"], out_csv)

    assert n_rows == 3
    out_df = pd.read_csv(out_csv)
    assert set(out_df["video_id"].unique()) == {"video_1", "video_2"}
    assert len(out_df) == 3


def test_extract_old_csv_subset_raises_on_missing_video(mod, tmp_path):
    old_csv = tmp_path / "old.csv"
    pd.DataFrame({"video_id": ["video_1"], "won": [1]}).to_csv(old_csv, index=False)
    out_csv = tmp_path / "subset.csv"

    with pytest.raises(RuntimeError, match="video_99"):
        mod.extract_old_csv_subset(old_csv, ["1", "99"], out_csv)


def test_snapshot_npz_subset_hardlinks_or_copies(mod, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    dst_dir = tmp_path / "dst"
    for tid in ("1", "2", "3"):
        (src_dir / f"{tid}.npz").write_bytes(b"dummy-npz-content")

    mod.snapshot_npz_subset(src_dir, dst_dir, ["1", "2"])

    assert (dst_dir / "1.npz").exists()
    assert (dst_dir / "2.npz").exists()
    assert not (dst_dir / "3.npz").exists()
    assert (dst_dir / "1.npz").read_bytes() == b"dummy-npz-content"


def test_stage_out_dir_and_stage_is_done(mod, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = mod.stage_out_dir("data/verify/retrain_stage_N", 30)
    assert str(d) == "data/verify/retrain_stage_N30_2026-08-18"

    assert mod.stage_is_done(d, "both") is False

    (d / "old").mkdir(parents=True)
    (d / "old" / "summary.json").write_text("{}", encoding="utf-8")
    (d / "new_no_lock").mkdir(parents=True)
    (d / "new_no_lock" / "summary.json").write_text("{}", encoding="utf-8")
    (d / "new_with_lock").mkdir(parents=True)
    (d / "new_with_lock" / "summary.json").write_text("{}", encoding="utf-8")

    assert mod.stage_is_done(d, "both") is True
    assert mod.stage_is_done(d, "no_lock") is True


def test_build_trend_report_handles_no_completed_stages(mod, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trend_dir = tmp_path / "trend"
    mod.build_trend_report((30, 60), "data/verify/retrain_stage_N", trend_dir)
    report = (trend_dir / "trend_report.md").read_text(encoding="utf-8")
    assert "まだ完了した断面がありません" in report
