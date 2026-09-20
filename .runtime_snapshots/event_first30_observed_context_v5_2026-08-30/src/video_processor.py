"""
YouTube動画→盤面変換モジュール

yt-dlpで動画をダウンロードし、OpenCVでフレームを抽出して
ImageReaderで盤面データに変換。結果をJSONで保存する。
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.board import Board
from src.image_reader import BoardRegion, ImageReader
from src.storage import STATUS_CLEANED, STATUS_DOWNLOADED, STATUS_ERROR, STATUS_PROCESSED, StorageManager

# ============================
# 定数定義
# ============================

# フレームサンプリング間隔 (秒)
DEFAULT_FRAME_INTERVAL_SEC: float = 1.0

# デフォルト出力ディレクトリ
DEFAULT_FRAMES_DIR: Path = Path("data/frames")
DEFAULT_BOARDS_DIR: Path = Path("data/boards")

# yt-dlp デフォルトオプション
YTDLP_FORMAT: str = "bestvideo[height<=1080][ext=mp4]+bestaudio/best[height<=1080]"
YTDLP_MERGE_FORMAT: str = "mp4"


# ============================
# データクラス
# ============================


@dataclass
class FrameResult:
    """
    1フレームの解析結果。

    Attributes:
        frame_index: 動画内のフレーム番号。
        timestamp_sec: 動画内の時刻 (秒)。
        board_1p: 1P側の盤面データ。
        board_2p: 2P側の盤面データ。
    """
    frame_index: int
    timestamp_sec: float
    board_1p: Board
    board_2p: Board

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する (JSON保存用)。"""
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "board_1p": self.board_1p.to_dict(),
            "board_2p": self.board_2p.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameResult":
        """辞書からFrameResultを復元する。"""
        return cls(
            frame_index=data["frame_index"],
            timestamp_sec=data["timestamp_sec"],
            board_1p=Board.from_dict(data["board_1p"]),
            board_2p=Board.from_dict(data["board_2p"]),
        )


@dataclass
class VideoAnalysis:
    """
    1動画の解析結果まとめ。

    Attributes:
        url: 元動画のURL。
        video_path: ダウンロードした動画ファイルのパス。
        fps: 動画のフレームレート。
        total_frames: 動画の総フレーム数。
        duration_sec: 動画の長さ (秒)。
        frame_results: フレームごとの解析結果リスト。
    """
    url: str
    video_path: str
    fps: float
    total_frames: int
    duration_sec: float
    frame_results: list[FrameResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する (JSON保存用)。"""
        return {
            "url": self.url,
            "video_path": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_sec": self.duration_sec,
            "frame_results": [r.to_dict() for r in self.frame_results],
        }


# ============================
# VideoProcessor
# ============================


class VideoProcessor:
    """
    YouTube動画をダウンロードして盤面データに変換するクラス。

    Usage:
        processor = VideoProcessor()
        analysis = processor.process_url("https://youtube.com/watch?v=xxx")
        processor.save_analysis(analysis, "data/boards/match_001.json")
    """

    def __init__(
        self,
        image_reader: ImageReader | None = None,
        storage: StorageManager | None = None,
        frame_interval_sec: float = DEFAULT_FRAME_INTERVAL_SEC,
        frames_dir: Path = DEFAULT_FRAMES_DIR,
        boards_dir: Path = DEFAULT_BOARDS_DIR,
    ) -> None:
        """
        Args:
            image_reader: 盤面読み取り器。Noneの場合はデフォルトを使用。
            storage: ストレージ管理器。Noneの場合はデフォルトを使用。
            frame_interval_sec: フレーム抽出間隔 (秒)。
            frames_dir: フレーム画像の一時保存ディレクトリ。
            boards_dir: 盤面JSONの保存ディレクトリ。
        """
        self._reader: ImageReader = image_reader or ImageReader()
        self._storage: StorageManager = storage or StorageManager()
        self._frame_interval_sec: float = frame_interval_sec
        self._frames_dir: Path = frames_dir
        self._boards_dir: Path = boards_dir

        self._boards_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    # ============================
    # ダウンロード
    # ============================

    def download_video(self, url: str, output_dir: Path | None = None) -> Path:
        """
        yt-dlpで動画をダウンロードする。

        Args:
            url: YouTube動画のURL。
            output_dir: 保存先ディレクトリ。Noneの場合はframes_dirを使用。

        Returns:
            Path: ダウンロードした動画ファイルのパス。

        Raises:
            RuntimeError: ダウンロード失敗時。
        """
        save_dir = output_dir or self._frames_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        output_template = str(save_dir / "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--format", YTDLP_FORMAT,
            "--merge-output-format", YTDLP_MERGE_FORMAT,
            "--output", output_template,
            "--print", "after_move:filepath",
            "--no-playlist",
            url,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"yt-dlpダウンロード失敗: {result.stderr.strip()}"
            )

        video_path = Path(result.stdout.strip().splitlines()[-1])
        if not video_path.exists():
            raise RuntimeError(
                f"ダウンロードファイルが見つかりません: {video_path}"
            )

        self._storage.record_video(url, metadata={"video_path": str(video_path)})
        self._storage.update_status(url, STATUS_DOWNLOADED)
        return video_path

    # ============================
    # フレーム抽出
    # ============================

    def extract_frames(
        self,
        video_path: str | Path,
        interval_sec: float | None = None,
    ) -> list[tuple[int, float, np.ndarray]]:
        """
        動画ファイルからフレームを一定間隔で抽出する。

        Args:
            video_path: 動画ファイルのパス。
            interval_sec: 抽出間隔 (秒)。Noneの場合はデフォルト値を使用。

        Returns:
            list of (frame_index, timestamp_sec, frame_bgr): 抽出したフレームリスト。

        Raises:
            FileNotFoundError: 動画ファイルが存在しない場合。
            RuntimeError: 動画を開けない場合。
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # フォールバック

        interval = interval_sec if interval_sec is not None else self._frame_interval_sec
        frame_step = max(1, int(fps * interval))

        frames: list[tuple[int, float, np.ndarray]] = []
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_step == 0:
                timestamp = frame_index / fps
                frames.append((frame_index, timestamp, frame))
            frame_index += 1

        cap.release()
        return frames

    def get_video_info(self, video_path: str | Path) -> dict[str, Any]:
        """
        動画のメタ情報を返す。

        Args:
            video_path: 動画ファイルのパス。

        Returns:
            dict: fps, total_frames, duration_sec を含む辞書。

        Raises:
            FileNotFoundError: ファイルが存在しない場合。
            RuntimeError: 動画を開けない場合。
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        fps = fps if fps > 0 else 30.0
        duration_sec = total_frames / fps if fps > 0 else 0.0

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration_sec,
        }

    # ============================
    # 盤面変換
    # ============================

    def process_frames(
        self,
        frames: list[tuple[int, float, np.ndarray]],
    ) -> list[FrameResult]:
        """
        フレームリストから盤面データリストを生成する。

        Args:
            frames: extract_frames() の返り値。

        Returns:
            list[FrameResult]: フレームごとの解析結果。
        """
        results: list[FrameResult] = []
        for frame_index, timestamp_sec, frame_bgr in frames:
            board_1p, board_2p = self._reader.read_both_boards(frame_bgr)
            results.append(FrameResult(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                board_1p=board_1p,
                board_2p=board_2p,
            ))
        return results

    def process_video_file(
        self,
        video_path: str | Path,
        url: str = "",
        interval_sec: float | None = None,
    ) -> VideoAnalysis:
        """
        動画ファイルを処理して VideoAnalysis を返す。

        Args:
            video_path: 動画ファイルのパス。
            url: 元動画のURL (記録用、省略可)。
            interval_sec: フレーム抽出間隔 (秒)。

        Returns:
            VideoAnalysis: 解析結果。
        """
        info = self.get_video_info(video_path)
        frames = self.extract_frames(video_path, interval_sec)
        frame_results = self.process_frames(frames)

        return VideoAnalysis(
            url=url,
            video_path=str(video_path),
            fps=info["fps"],
            total_frames=info["total_frames"],
            duration_sec=info["duration_sec"],
            frame_results=frame_results,
        )

    def process_url(
        self,
        url: str,
        interval_sec: float | None = None,
        delete_after: bool = True,
    ) -> VideoAnalysis:
        """
        URLから動画をダウンロードして盤面データに変換する。

        Args:
            url: YouTube動画のURL。
            interval_sec: フレーム抽出間隔 (秒)。
            delete_after: 処理後に動画ファイルを削除するか。

        Returns:
            VideoAnalysis: 解析結果。
        """
        video_path = self.download_video(url)

        try:
            analysis = self.process_video_file(str(video_path), url=url, interval_sec=interval_sec)
            self._storage.update_status(
                url, STATUS_PROCESSED, frame_count=len(analysis.frame_results)
            )
        except Exception as e:
            self._storage.update_status(url, STATUS_ERROR)
            raise RuntimeError(f"動画処理エラー: {e}") from e
        finally:
            if delete_after:
                self._storage.cleanup(str(video_path))
                self._storage.update_status(url, STATUS_CLEANED)

        return analysis

    # ============================
    # 保存
    # ============================

    def save_analysis(
        self, analysis: VideoAnalysis, output_path: str | Path | None = None
    ) -> Path:
        """
        VideoAnalysis をJSONファイルに保存する。

        Args:
            analysis: 保存する解析結果。
            output_path: 出力ファイルパス。Noneの場合は自動生成。

        Returns:
            Path: 保存したJSONファイルのパス。
        """
        if output_path is None:
            video_stem = Path(analysis.video_path).stem if analysis.video_path else "unknown"
            output_path = self._boards_dir / f"{video_stem}.json"

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(analysis.to_dict(), f, ensure_ascii=False, indent=2)

        return out_path

    def load_analysis(self, json_path: str | Path) -> VideoAnalysis:
        """
        JSONファイルからVideoAnalysisを復元する。

        Args:
            json_path: JSONファイルのパス。

        Returns:
            VideoAnalysis: 復元した解析結果。
        """
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        return VideoAnalysis(
            url=data.get("url", ""),
            video_path=data.get("video_path", ""),
            fps=data.get("fps", 30.0),
            total_frames=data.get("total_frames", 0),
            duration_sec=data.get("duration_sec", 0.0),
            frame_results=[
                FrameResult.from_dict(r) for r in data.get("frame_results", [])
            ],
        )
