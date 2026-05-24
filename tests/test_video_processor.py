"""
video_processor.py のテスト

OpenCV VideoWriter で合成動画を生成し、フレーム抽出・盤面変換・保存をテストする。
実ネットワークには接続しない (download_video はモックで検証)。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_RED,
    Board,
)
from src.image_reader import BoardRegion, ImageReader
from src.storage import STATUS_DOWNLOADED, STATUS_PROCESSED, StorageManager
from src.video_processor import (
    DEFAULT_FRAME_INTERVAL_SEC,
    FrameResult,
    VideoAnalysis,
    VideoProcessor,
)

# ============================
# テスト用合成動画ヘルパー
# ============================

SYNTH_FPS: int = 10
SYNTH_FRAME_W: int = 320
SYNTH_FRAME_H: int = 240
SYNTH_DURATION_SEC: int = 3  # 秒


def make_synthetic_video(output_path: Path, fps: int = SYNTH_FPS) -> Path:
    """
    単色フレームの合成動画を生成する。

    Args:
        output_path: 出力先 .mp4 パス。
        fps: フレームレート。

    Returns:
        Path: 生成した動画ファイルのパス。
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (SYNTH_FRAME_W, SYNTH_FRAME_H)
    )
    total = fps * SYNTH_DURATION_SEC
    for i in range(total):
        # フレームごとに輝度を変化させる (動画が空でないことを確認)
        val = int(255 * i / total)
        frame = np.full((SYNTH_FRAME_H, SYNTH_FRAME_W, 3), val, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return output_path


# ============================
# フィクスチャ
# ============================

@pytest.fixture
def tmp_video(tmp_path: Path) -> Path:
    """一時合成動画を生成する。"""
    return make_synthetic_video(tmp_path / "test.mp4")


@pytest.fixture
def tmp_processor(tmp_path: Path) -> VideoProcessor:
    """一時ディレクトリを使うVideoProcessorを生成する。"""
    storage = StorageManager(history_path=tmp_path / "history.json")
    return VideoProcessor(
        storage=storage,
        frame_interval_sec=1.0,
        frames_dir=tmp_path / "frames",
        boards_dir=tmp_path / "boards",
    )


# ============================
# FrameResult テスト
# ============================

class TestFrameResult:
    def test_to_dict_has_required_keys(self):
        fr = FrameResult(
            frame_index=0,
            timestamp_sec=0.0,
            board_1p=Board(),
            board_2p=Board(),
        )
        d = fr.to_dict()
        assert "frame_index" in d
        assert "timestamp_sec" in d
        assert "board_1p" in d
        assert "board_2p" in d

    def test_round_trip(self):
        fr = FrameResult(
            frame_index=5,
            timestamp_sec=2.5,
            board_1p=Board(),
            board_2p=Board(),
        )
        restored = FrameResult.from_dict(fr.to_dict())
        assert restored.frame_index == 5
        assert restored.timestamp_sec == pytest.approx(2.5)
        assert isinstance(restored.board_1p, Board)
        assert isinstance(restored.board_2p, Board)


# ============================
# VideoAnalysis テスト
# ============================

class TestVideoAnalysis:
    def test_to_dict_keys(self):
        va = VideoAnalysis(
            url="https://example.com",
            video_path="/tmp/test.mp4",
            fps=30.0,
            total_frames=900,
            duration_sec=30.0,
        )
        d = va.to_dict()
        assert "url" in d
        assert "fps" in d
        assert "frame_results" in d
        assert isinstance(d["frame_results"], list)


# ============================
# get_video_info テスト
# ============================

class TestGetVideoInfo:
    def test_get_video_info_returns_dict(
        self, tmp_processor: VideoProcessor, tmp_video: Path
    ):
        info = tmp_processor.get_video_info(tmp_video)
        assert "fps" in info
        assert "total_frames" in info
        assert "duration_sec" in info

    def test_fps_is_positive(
        self, tmp_processor: VideoProcessor, tmp_video: Path
    ):
        info = tmp_processor.get_video_info(tmp_video)
        assert info["fps"] > 0

    def test_total_frames_matches_synth(
        self, tmp_processor: VideoProcessor, tmp_video: Path
    ):
        info = tmp_processor.get_video_info(tmp_video)
        expected = SYNTH_FPS * SYNTH_DURATION_SEC
        # 数フレームの誤差は許容
        assert abs(info["total_frames"] - expected) <= 5

    def test_duration_close_to_synth(
        self, tmp_processor: VideoProcessor, tmp_video: Path
    ):
        info = tmp_processor.get_video_info(tmp_video)
        assert abs(info["duration_sec"] - SYNTH_DURATION_SEC) < 1.0

    def test_file_not_found(self, tmp_processor: VideoProcessor):
        with pytest.raises(FileNotFoundError):
            tmp_processor.get_video_info("/nonexistent/video.mp4")


# ============================
# extract_frames テスト
# ============================

class TestExtractFrames:
    def test_extract_returns_list(
        self, tmp_processor: VideoProcessor, tmp_video: Path
    ):
        frames = tmp_processor.extract_frames(tmp_video, interval_sec=1.0)
        assert isinstance(frames, list)
        assert len(frames) > 0

    def test_extract_tuple_structure(
        self, tmp_processor: VideoProcessor, tmp_video: Path
    ):
        frames = tmp_processor.extract_frames(tmp_video, interval_sec=1.0)
        idx, ts, frame = frames[0]
        assert isinstance(idx, int)
        assert isinstance(ts, float)
        assert frame.ndim == 3

    def test_extract_frame_count(
        self, tmp_processor: VideoProcessor, tmp_video: Path
    ):
        # 3秒動画を1秒間隔で抽出 → 約3フレーム
        frames = tmp_processor.extract_frames(tmp_video, interval_sec=1.0)
        assert 2 <= len(frames) <= 4

    def test_extract_shorter_interval_more_frames(
        self, tmp_processor: VideoProcessor, tmp_video: Path
    ):
        frames_1s = tmp_processor.extract_frames(tmp_video, interval_sec=1.0)
        frames_05s = tmp_processor.extract_frames(tmp_video, interval_sec=0.5)
        assert len(frames_05s) >= len(frames_1s)

    def test_timestamps_are_increasing(
        self, tmp_processor: VideoProcessor, tmp_video: Path
    ):
        frames = tmp_processor.extract_frames(tmp_video, interval_sec=0.5)
        timestamps = [ts for _, ts, _ in frames]
        assert timestamps == sorted(timestamps)

    def test_file_not_found(self, tmp_processor: VideoProcessor):
        with pytest.raises(FileNotFoundError):
            tmp_processor.extract_frames("/nonexistent/video.mp4")

    def test_default_interval_used(
        self, tmp_processor: VideoProcessor, tmp_video: Path
    ):
        """interval_sec=None でデフォルト値が使われる。"""
        frames = tmp_processor.extract_frames(tmp_video, interval_sec=None)
        assert len(frames) > 0


# ============================
# process_frames テスト
# ============================

class TestProcessFrames:
    def test_process_frames_returns_list(self, tmp_processor: VideoProcessor):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frames_input = [(0, 0.0, frame), (10, 1.0, frame)]
        results = tmp_processor.process_frames(frames_input)
        assert len(results) == 2
        assert all(isinstance(r, FrameResult) for r in results)

    def test_frame_index_preserved(self, tmp_processor: VideoProcessor):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        results = tmp_processor.process_frames([(42, 4.2, frame)])
        assert results[0].frame_index == 42
        assert results[0].timestamp_sec == pytest.approx(4.2)

    def test_boards_are_board_instances(self, tmp_processor: VideoProcessor):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        results = tmp_processor.process_frames([(0, 0.0, frame)])
        assert isinstance(results[0].board_1p, Board)
        assert isinstance(results[0].board_2p, Board)

    def test_empty_input(self, tmp_processor: VideoProcessor):
        assert tmp_processor.process_frames([]) == []


# ============================
# process_video_file テスト
# ============================

class TestProcessVideoFile:
    def test_returns_video_analysis(
        self, tmp_processor: VideoProcessor, tmp_video: Path
    ):
        analysis = tmp_processor.process_video_file(tmp_video)
        assert isinstance(analysis, VideoAnalysis)

    def test_frame_results_not_empty(
        self, tmp_processor: VideoProcessor, tmp_video: Path
    ):
        analysis = tmp_processor.process_video_file(tmp_video, interval_sec=1.0)
        assert len(analysis.frame_results) > 0

    def test_url_stored(
        self, tmp_processor: VideoProcessor, tmp_video: Path
    ):
        url = "https://example.com/test"
        analysis = tmp_processor.process_video_file(tmp_video, url=url)
        assert analysis.url == url

    def test_file_not_found(self, tmp_processor: VideoProcessor):
        with pytest.raises(FileNotFoundError):
            tmp_processor.process_video_file("/nonexistent.mp4")


# ============================
# save_analysis / load_analysis テスト
# ============================

class TestSaveLoadAnalysis:
    def _make_analysis(self) -> VideoAnalysis:
        return VideoAnalysis(
            url="https://example.com",
            video_path="/tmp/test.mp4",
            fps=30.0,
            total_frames=90,
            duration_sec=3.0,
            frame_results=[
                FrameResult(
                    frame_index=0,
                    timestamp_sec=0.0,
                    board_1p=Board(),
                    board_2p=Board(),
                )
            ],
        )

    def test_save_creates_json(
        self, tmp_processor: VideoProcessor, tmp_path: Path
    ):
        analysis = self._make_analysis()
        out = tmp_processor.save_analysis(analysis, tmp_path / "out.json")
        assert out.exists()

    def test_json_is_valid(
        self, tmp_processor: VideoProcessor, tmp_path: Path
    ):
        analysis = self._make_analysis()
        out = tmp_processor.save_analysis(analysis, tmp_path / "out.json")
        with out.open() as f:
            data = json.load(f)
        assert "frame_results" in data

    def test_load_restores_analysis(
        self, tmp_processor: VideoProcessor, tmp_path: Path
    ):
        analysis = self._make_analysis()
        out = tmp_processor.save_analysis(analysis, tmp_path / "out.json")
        restored = tmp_processor.load_analysis(out)
        assert restored.url == analysis.url
        assert restored.fps == pytest.approx(30.0)
        assert len(restored.frame_results) == 1

    def test_auto_output_path(
        self, tmp_processor: VideoProcessor
    ):
        """output_path=None で自動的にboards_dirへ保存される。"""
        analysis = self._make_analysis()
        out = tmp_processor.save_analysis(analysis)
        assert out.exists()
        assert out.suffix == ".json"


# ============================
# download_video テスト (モック)
# ============================

class TestDownloadVideo:
    def test_download_records_url(
        self, tmp_processor: VideoProcessor, tmp_path: Path
    ):
        """ダウンロード成功時にURLが記録される。"""
        fake_video = tmp_path / "fake_video.mp4"
        fake_video.write_bytes(b"fake")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=str(fake_video) + "\n",
                stderr="",
            )
            url = "https://www.youtube.com/watch?v=test_id"
            result_path = tmp_processor.download_video(url, output_dir=tmp_path)

        assert result_path == fake_video
        record = tmp_processor._storage.get_record(url)
        assert record is not None
        assert record.status == STATUS_DOWNLOADED

    def test_download_raises_on_failure(
        self, tmp_processor: VideoProcessor, tmp_path: Path
    ):
        """yt-dlp がエラーを返すと RuntimeError が発生する。"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="エラー: 動画が見つかりません",
            )
            with pytest.raises(RuntimeError, match="yt-dlpダウンロード失敗"):
                tmp_processor.download_video(
                    "https://www.youtube.com/watch?v=invalid",
                    output_dir=tmp_path,
                )
