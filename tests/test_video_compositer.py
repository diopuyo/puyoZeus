"""
video_compositer.py のテスト

実 mp4 を合成書き出しまでテストする。ffmpeg 音声結合は
利用可能性に応じて条件分岐する。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video_compositer import (
    DEFAULT_SAMPLING_INTERVAL_SEC,
    CompositeOptions,
    CompositeResult,
    VideoCompositor,
)


def _has_ffmpeg() -> bool:
    return VideoCompositor._ffmpeg_available()


def _ffmpeg_bin() -> str | None:
    return VideoCompositor._resolve_ffmpeg_bin()


def _probe_has_audio(video_path: Path) -> bool:
    """ffmpeg で動画に音声トラックが含まれているか判定する。"""
    bin_path = _ffmpeg_bin()
    if bin_path is None:
        return False
    # ffprobe は別バイナリなので ffmpeg -i + stderr で代替
    result = subprocess.run(
        [bin_path, "-i", str(video_path), "-hide_banner"],
        capture_output=True, text=True,
    )
    return "Audio:" in result.stderr


# ============================
# ヘルパー
# ============================


def _make_synthetic_video(path: Path, frames: int, fps: float = 10.0) -> None:
    """黒色の短い合成動画を生成する (テスト用)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 180))
    try:
        for _ in range(frames):
            frame = np.zeros((180, 320, 3), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


@pytest.fixture
def input_video(tmp_path: Path) -> Path:
    p = tmp_path / "input.mp4"
    _make_synthetic_video(p, frames=20, fps=10.0)
    return p


@pytest.fixture
def output_path(tmp_path: Path) -> Path:
    return tmp_path / "output.mp4"


# ============================
# CompositeOptions / Result
# ============================


class TestCompositeOptions:
    def test_default_sampling_interval(self):
        opts = CompositeOptions()
        assert opts.sampling_interval_sec == DEFAULT_SAMPLING_INTERVAL_SEC
        assert opts.mux_audio is True


# ============================
# VideoCompositor - 基本
# ============================


class TestComposite:
    def test_file_not_found_raises(self, tmp_path: Path):
        comp = VideoCompositor()
        with pytest.raises(FileNotFoundError):
            comp.composite(
                input_path=tmp_path / "missing.mp4",
                output_path=tmp_path / "out.mp4",
            )

    def test_output_file_created(self, input_video, output_path):
        comp = VideoCompositor()
        # 音声結合は合成動画に音声がないのでスキップ
        result = comp.composite(
            input_path=input_video,
            output_path=output_path,
            options=CompositeOptions(mux_audio=False),
        )
        assert isinstance(result, CompositeResult)
        assert output_path.exists()
        assert result.total_frames == 20

    def test_result_metadata(self, input_video, output_path):
        comp = VideoCompositor()
        result = comp.composite(
            input_video, output_path,
            options=CompositeOptions(mux_audio=False),
        )
        assert result.fps > 0
        assert result.width == 320
        assert result.height == 180
        assert result.analyzed_frames >= 1

    def test_analyzed_frames_less_than_total(self, input_video, output_path):
        # 1秒間隔サンプリング=10fpsなら10フレーム毎=解析2-3回
        comp = VideoCompositor()
        result = comp.composite(
            input_video, output_path,
            options=CompositeOptions(
                sampling_interval_sec=1.0,
                mux_audio=False,
            ),
        )
        assert result.analyzed_frames <= result.total_frames

    def test_short_sampling_analyzes_every_frame(
        self, input_video, output_path,
    ):
        # sampling=0.01秒 → 全フレーム解析相当
        comp = VideoCompositor()
        result = comp.composite(
            input_video, output_path,
            options=CompositeOptions(
                sampling_interval_sec=0.01,
                mux_audio=False,
            ),
        )
        assert result.analyzed_frames == result.total_frames


# ============================
# VideoCompositor - 進捗コールバック
# ============================


class TestProgressCallback:
    def test_callback_invoked_per_frame(self, input_video, output_path):
        calls: list[dict] = []

        def cb(payload: dict) -> None:
            calls.append(payload)

        comp = VideoCompositor()
        comp.composite(
            input_video, output_path,
            options=CompositeOptions(
                mux_audio=False,
                progress_callback=cb,
            ),
        )
        assert len(calls) == 20
        assert "current_frame" in calls[0]
        assert "total_frames" in calls[0]


# ============================
# VideoCompositor - 音声結合
# ============================


class TestAudioMux:
    def test_audio_mux_skipped_when_disabled(self, input_video, output_path):
        comp = VideoCompositor()
        result = comp.composite(
            input_video, output_path,
            options=CompositeOptions(mux_audio=False),
        )
        assert result.audio_muxed is False

    @pytest.mark.skipif(
        not _has_ffmpeg(),
        reason="ffmpeg が未インストール",
    )
    def test_audio_mux_attempts_when_available(
        self, input_video, output_path,
    ):
        # 音声無しの合成動画に音声結合を試みる (成功/失敗どちらでも可)
        comp = VideoCompositor()
        result = comp.composite(
            input_video, output_path,
            options=CompositeOptions(mux_audio=True),
        )
        # 音声無し動画が入力なので音声結合は失敗するかもしれないが、
        # 出力ファイル自体は存在すること
        assert output_path.exists()


@pytest.mark.skipif(
    not _has_ffmpeg(),
    reason="ffmpeg が未インストール",
)
class TestAudioMuxEndToEnd:
    """ffmpeg で音声付き動画を生成し、合成後も音声が保持されることを検証する。"""

    @pytest.fixture
    def audio_video(self, tmp_path: Path) -> Path:
        """音声トラック付きの合成動画 (2秒、10fps) を ffmpeg で生成する。"""
        out = tmp_path / "with_audio.mp4"
        bin_path = _ffmpeg_bin()
        cmd = [
            bin_path, "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x180:r=10:d=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(out),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        assert _probe_has_audio(out), "生成動画に音声が無い"
        return out

    def test_composite_preserves_audio(self, audio_video: Path, tmp_path: Path):
        out = tmp_path / "composed.mp4"
        comp = VideoCompositor()
        result = comp.composite(
            audio_video, out,
            options=CompositeOptions(mux_audio=True),
        )
        assert out.exists()
        assert result.audio_muxed is True
        assert _probe_has_audio(out), "合成後に音声が消えている"

    def test_composite_without_audio_flag_has_no_audio(
        self, audio_video: Path, tmp_path: Path,
    ):
        out = tmp_path / "composed_silent.mp4"
        comp = VideoCompositor()
        result = comp.composite(
            audio_video, out,
            options=CompositeOptions(mux_audio=False),
        )
        assert out.exists()
        assert result.audio_muxed is False
        assert not _probe_has_audio(out), "音声無効化したのに音声が残っている"
