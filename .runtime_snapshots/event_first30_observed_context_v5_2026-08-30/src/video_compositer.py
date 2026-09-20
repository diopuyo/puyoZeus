"""
動画合成出力モジュール

入力動画の各フレームに分析結果のオーバーレイを合成し、
新しい動画ファイルとして書き出す。

解析は SAMPLING_INTERVAL_SEC 毎に行い、間のフレームは直近の結果を再利用する。
音声トラックは ffmpeg で元動画から結合する (ffmpeg 利用不可ならスキップ)。
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from src.analyzer import AnalysisResult, Analyzer
from src.overlay import OverlayRenderer

# ============================
# 定数定義
# ============================

# デフォルト解析サンプル間隔 (秒)
DEFAULT_SAMPLING_INTERVAL_SEC: float = 0.5

# OpenCV VideoWriter の fourcc (MP4 互換)
DEFAULT_FOURCC: str = "mp4v"

# 出力 fps がソースから取れない場合のフォールバック
FALLBACK_FPS: float = 30.0

# ffmpeg バイナリ名 (システム PATH もしくは imageio-ffmpeg 同梱)
FFMPEG_BIN: str = "ffmpeg"
FFMPEG_AUDIO_MUX_ARGS: tuple[str, ...] = (
    "-y",
    "-i", "{video}",
    "-i", "{audio_src}",
    "-c:v", "copy",
    "-c:a", "aac",
    "-map", "0:v:0",
    "-map", "1:a:0?",
    "-shortest",
    "{output}",
)

# 進捗コールバックに渡す引数名
PROGRESS_CURRENT: str = "current_frame"
PROGRESS_TOTAL: str = "total_frames"


# ============================
# データクラス
# ============================


@dataclass
class CompositeOptions:
    """
    動画合成のオプション。

    Attributes:
        sampling_interval_sec: 解析サンプリング間隔 (秒)。
        mux_audio: True なら ffmpeg で元動画の音声を合成する。
        fourcc: VideoWriter の fourcc 文字列。
        progress_callback: 進捗コールバック (任意)。
    """
    sampling_interval_sec: float = DEFAULT_SAMPLING_INTERVAL_SEC
    mux_audio: bool = True
    fourcc: str = DEFAULT_FOURCC
    progress_callback: Callable[[dict[str, Any]], None] | None = None


@dataclass(frozen=True)
class CompositeResult:
    """
    動画合成の結果メタデータ。

    Attributes:
        output_path: 出力動画パス。
        total_frames: 処理した総フレーム数。
        analyzed_frames: 実際に解析を行ったフレーム数。
        audio_muxed: 音声結合が成功したか。
        fps: 出力 fps。
        width: 出力幅。
        height: 出力高さ。
    """
    output_path: Path
    total_frames: int
    analyzed_frames: int
    audio_muxed: bool
    fps: float
    width: int
    height: int


# ============================
# VideoCompositor
# ============================


class VideoCompositor:
    """
    動画合成を行うクラス。

    Usage:
        comp = VideoCompositor()
        result = comp.composite(
            input_path="in.mp4",
            output_path="out.mp4",
        )
    """

    def __init__(
        self,
        analyzer: Analyzer | None = None,
        renderer: OverlayRenderer | None = None,
    ) -> None:
        """
        Args:
            analyzer: 分析エンジン (None ならデフォルト)。
            renderer: 描画エンジン (None ならデフォルト)。
        """
        self._analyzer = analyzer or Analyzer()
        self._renderer = renderer or OverlayRenderer()

    # ============================
    # 公開メソッド
    # ============================

    def composite(
        self,
        input_path: str | Path,
        output_path: str | Path,
        options: CompositeOptions | None = None,
    ) -> CompositeResult:
        """
        入力動画を合成し、出力パスに書き出す。

        Args:
            input_path: 入力動画パス。
            output_path: 出力動画パス。
            options: 合成オプション (None ならデフォルト)。

        Returns:
            CompositeResult: 合成結果メタデータ。

        Raises:
            FileNotFoundError: 入力ファイルが存在しない場合。
            RuntimeError: VideoWriter が開けない場合。
        """
        opts = options or CompositeOptions()
        input_path = Path(input_path)
        output_path = Path(output_path)
        if not input_path.exists():
            raise FileNotFoundError(f"入力動画が存在しません: {input_path}")

        info = self._open_video(input_path)
        try:
            temp_output = self._temp_video_path(output_path)
            writer = self._open_writer(
                temp_output, info["fps"], info["width"], info["height"],
                opts.fourcc,
            )
            try:
                total, analyzed = self._process_frames(
                    info["cap"], writer, info, opts,
                )
            finally:
                writer.release()

            audio_muxed = False
            if opts.mux_audio and self._ffmpeg_available():
                audio_muxed = self._mux_audio(
                    temp_output, input_path, output_path,
                )
                if audio_muxed:
                    temp_output.unlink(missing_ok=True)

            if not audio_muxed:
                shutil.move(str(temp_output), str(output_path))

            return CompositeResult(
                output_path=output_path,
                total_frames=total,
                analyzed_frames=analyzed,
                audio_muxed=audio_muxed,
                fps=info["fps"],
                width=info["width"],
                height=info["height"],
            )
        finally:
            info["cap"].release()

    # ============================
    # 内部メソッド: 動画 IO
    # ============================

    @staticmethod
    def _open_video(path: Path) -> dict[str, Any]:
        """入力動画を開いてメタ情報を返す。"""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "cap": cap,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total,
        }

    @staticmethod
    def _open_writer(
        path: Path, fps: float, width: int, height: int, fourcc: str,
    ) -> cv2.VideoWriter:
        """VideoWriter を開く。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        writer = cv2.VideoWriter(
            str(path), fourcc_code, fps, (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter を開けません: {path}")
        return writer

    @staticmethod
    def _temp_video_path(output_path: Path) -> Path:
        """音声結合前の一時動画パスを返す。"""
        return output_path.with_suffix(".tmp.mp4")

    # ============================
    # 内部メソッド: フレーム処理
    # ============================

    def _process_frames(
        self,
        cap: cv2.VideoCapture,
        writer: cv2.VideoWriter,
        info: dict[str, Any],
        opts: CompositeOptions,
    ) -> tuple[int, int]:
        """全フレームを処理し (総数, 解析回数) を返す。"""
        fps = info["fps"]
        frames_per_sample = max(1, int(round(fps * opts.sampling_interval_sec)))

        cached_result: AnalysisResult | None = None
        total = 0
        analyzed = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 必要なタイミングで再解析
            if total % frames_per_sample == 0 or cached_result is None:
                timestamp = total / fps
                cached_result = self._analyzer.analyze_frame(
                    frame, timestamp=timestamp,
                )
                analyzed += 1

            composed = self._renderer.render(frame, cached_result)
            writer.write(composed)

            if opts.progress_callback is not None:
                opts.progress_callback({
                    PROGRESS_CURRENT: total,
                    PROGRESS_TOTAL: info["total_frames"],
                })

            total += 1

        return total, analyzed

    # ============================
    # 内部メソッド: 音声結合
    # ============================

    @staticmethod
    def _resolve_ffmpeg_bin() -> str | None:
        """
        利用可能な ffmpeg バイナリパスを返す。

        優先順:
            1. システム PATH の ffmpeg
            2. imageio-ffmpeg 同梱バイナリ (pip でインストール済みの場合)
        """
        system_bin = shutil.which(FFMPEG_BIN)
        if system_bin:
            return system_bin
        try:
            import imageio_ffmpeg  # type: ignore[import-not-found]

            return imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, Exception):
            return None

    @classmethod
    def _ffmpeg_available(cls) -> bool:
        """ffmpeg が実行可能か確認する。"""
        return cls._resolve_ffmpeg_bin() is not None

    @classmethod
    def _mux_audio(
        cls,
        video_path: Path, audio_src: Path, output_path: Path,
    ) -> bool:
        """元動画の音声を出力動画に結合する。成功で True。"""
        ffmpeg_bin = cls._resolve_ffmpeg_bin()
        if ffmpeg_bin is None:
            return False
        cmd = [ffmpeg_bin] + [
            arg.format(
                video=str(video_path),
                audio_src=str(audio_src),
                output=str(output_path),
            )
            for arg in FFMPEG_AUDIO_MUX_ARGS
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            return output_path.exists()
        except subprocess.CalledProcessError:
            return False
