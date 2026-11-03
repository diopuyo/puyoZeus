"""B3 レビュー動画の生成・検収・スマホ分割 (src/b3_review_video.py)。"""
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import b3_review_video as vid
from src.b3_review_synth import synth_rows


def test_mobile_bitrate_keeps_segment_under_limit():
    kbps = vid.mobile_video_kbps()
    worst = (kbps + vid.MOBILE_AUDIO_KBPS) * 1000 / 8 * vid.SEGMENT_SEC
    assert worst <= vid.LIMIT_BYTES                  # 上限 30MiB に収まる計算
    assert kbps > 300                                # 極端に潰れていない


def test_mobile_commands_follow_user_rules():
    shrink, split = vid.mobile_commands(Path("a.mp4"), Path("d"), "s", "ff")
    s = " ".join(shrink)
    assert "scale=1280:-2" in s and "-crf 28" in s and "aac" in s and "96k" in s
    assert "+faststart" in s and "-maxrate" in s
    assert "-segment_time 270" in " ".join(split) and "-c copy" in " ".join(split)


def test_selfcheck_counts_coverage_with_denominators():
    rows = synth_rows()
    sc = vid.selfcheck(rows, 0.0, 20.0)
    assert sc["rows_in_window"] == len(rows)
    assert sc["prob_observed"]["p_display"] == len(rows) - 2   # 3.0, 3.5 の 2 行が欠測
    assert sc["modes"]["exchange"] == 12 and sc["death"]["p2_proven_dead"] == 7
    assert vid.selfcheck(rows, 5.0, 20.0)["uncovered_head_sec"] == 0.0
    none = vid.selfcheck([], 0.0, 10.0)
    assert none["rows_in_window"] == 0 and none["uncovered_head_sec"] == 10.0


def _write_video(path: Path, seconds=2, fps=10, size=(320, 180)) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        w.write(np.full((size[1], size[0], 3), 20 + i, np.uint8))
    w.release()


def _need_ffmpeg() -> None:
    try:
        vid.ffmpeg_exe()
    except RuntimeError:
        pytest.skip("ffmpeg なし")


def test_render_review_end_to_end_csv_matches_frames(tmp_path):
    _need_ffmpeg()
    src = tmp_path / "src.mp4"
    _write_video(src)
    rows = synth_rows(duration_sec=2.0, step_sec=0.5)
    res = vid.render_review(src, rows, tmp_path / "o.mp4", tmp_path / "o.csv", with_audio=False)
    assert res["frames"] == 20 and res["encoder_returncode"] == 0
    cap = cv2.VideoCapture(str(tmp_path / "o.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == res["size"][1]
    with (tmp_path / "o.csv").open(encoding="utf-8-sig") as f:
        out = list(csv.DictReader(f))
    assert len(out) == 20 and out[0]["t_sec"] == "0.0" and out[-1]["frame_out"] == "19"
    assert (tmp_path / "o.selfcheck.json").exists()


def test_make_mobile_splits_and_checks_size(tmp_path):
    _need_ffmpeg()
    src = tmp_path / "src.mp4"
    _write_video(src, seconds=3)
    parts = vid.make_mobile(src, tmp_path / "m", "t")
    assert len(parts) >= 1 and all(p["ok"] for p in parts)
    assert not (tmp_path / "m" / "t_mobile_full.mp4").exists()
