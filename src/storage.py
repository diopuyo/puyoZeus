"""
ストレージ管理モジュール

処理済み動画URLを記録し、ダウンロード済み動画ファイルを削除してストレージを節約する。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ============================
# 定数定義
# ============================
DEFAULT_HISTORY_PATH: Path = Path("data/video_history.json")
DATA_DIR: Path = Path("data")

# ステータス定数
STATUS_PENDING: str = "pending"
STATUS_DOWNLOADED: str = "downloaded"
STATUS_PROCESSED: str = "processed"
STATUS_CLEANED: str = "cleaned"
STATUS_ERROR: str = "error"

VALID_STATUSES: frozenset[str] = frozenset(
    {STATUS_PENDING, STATUS_DOWNLOADED, STATUS_PROCESSED, STATUS_CLEANED, STATUS_ERROR}
)


# ============================
# データクラス
# ============================

@dataclass
class VideoRecord:
    """
    処理済み動画の記録。

    Attributes:
        url: 動画のURL。
        recorded_at: 記録日時 (ISO 8601形式)。
        frame_count: 抽出したフレーム数。
        status: 処理状態 (pending/downloaded/processed/cleaned/error)。
        metadata: 追加メタデータ (タイトル・長さ等)。
    """
    url: str
    recorded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    frame_count: int = 0
    status: str = STATUS_PENDING
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する (JSON保存用)。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoRecord":
        """辞書からVideoRecordを復元する。"""
        return cls(
            url=data["url"],
            recorded_at=data.get("recorded_at", datetime.now().isoformat()),
            frame_count=data.get("frame_count", 0),
            status=data.get("status", STATUS_PENDING),
            metadata=data.get("metadata", {}),
        )


# ============================
# StorageManager
# ============================

class StorageManager:
    """
    ストレージ管理クラス。

    動画URLの記録・履歴管理・動画ファイル削除・容量レポートを提供する。
    """

    def __init__(self, history_path: Path = DEFAULT_HISTORY_PATH) -> None:
        """
        初期化。

        Args:
            history_path: 履歴JSONファイルのパス。
        """
        self._history_path: Path = history_path
        self._history_path.parent.mkdir(parents=True, exist_ok=True)

    # ============================
    # URL記録
    # ============================

    def record_video(self, url: str, metadata: dict[str, Any] | None = None) -> VideoRecord:
        """
        URLとメタデータを履歴JSONに記録する。
        同一URLが既に存在する場合はメタデータを更新する。

        Args:
            url: 動画のURL。
            metadata: 追加メタデータ (タイトル・長さ等)。省略可。

        Returns:
            VideoRecord: 追加または更新されたレコード。
        """
        records = self._load_records()
        existing = next((r for r in records if r.url == url), None)

        if existing is not None:
            if metadata:
                existing.metadata.update(metadata)
            record = existing
        else:
            record = VideoRecord(
                url=url,
                metadata=metadata or {},
            )
            records.append(record)

        self._save_records(records)
        return record

    def update_status(
        self, url: str, status: str, frame_count: int | None = None
    ) -> VideoRecord:
        """
        指定URLのステータスを更新する。

        Args:
            url: 対象動画のURL。
            status: 新しいステータス。
            frame_count: フレーム数 (更新する場合)。

        Returns:
            VideoRecord: 更新されたレコード。

        Raises:
            ValueError: URLが見つからない場合、またはステータスが無効な場合。
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"無効なステータス: {status}")

        records = self._load_records()
        record = next((r for r in records if r.url == url), None)
        if record is None:
            raise ValueError(f"URLが履歴に存在しません: {url}")

        record.status = status
        if frame_count is not None:
            record.frame_count = frame_count

        self._save_records(records)
        return record

    # ============================
    # 履歴取得
    # ============================

    def get_history(self) -> list[VideoRecord]:
        """
        処理済み動画の履歴一覧を返す。

        Returns:
            list[VideoRecord]: 全レコードのリスト。
        """
        return self._load_records()

    def get_record(self, url: str) -> VideoRecord | None:
        """
        指定URLのレコードを返す。存在しない場合はNone。

        Args:
            url: 検索対象のURL。

        Returns:
            VideoRecord | None: 見つかったレコード、またはNone。
        """
        return next((r for r in self._load_records() if r.url == url), None)

    # ============================
    # ファイル削除
    # ============================

    def cleanup(self, video_path: str) -> bool:
        """
        動画ファイルを削除する。URL記録は保持する。

        Args:
            video_path: 削除する動画ファイルのパス。

        Returns:
            bool: 削除成功ならTrue、ファイルが存在しない場合はFalse。
        """
        path = Path(video_path)
        if not path.exists():
            return False
        path.unlink()
        return True

    def cleanup_directory(self, dir_path: str) -> int:
        """
        ディレクトリ内の動画ファイルを全削除する。

        Args:
            dir_path: 削除対象ディレクトリのパス。

        Returns:
            int: 削除したファイル数。
        """
        video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
        target_dir = Path(dir_path)
        if not target_dir.is_dir():
            return 0

        deleted_count = 0
        for file_path in target_dir.iterdir():
            if file_path.suffix.lower() in video_extensions:
                file_path.unlink()
                deleted_count += 1
        return deleted_count

    # ============================
    # ストレージレポート
    # ============================

    def get_storage_usage(self) -> dict[str, Any]:
        """
        data/ ディレクトリ以下の容量レポートを返す。

        Returns:
            dict: 各ディレクトリの容量情報を含む辞書。
                  - total_bytes: data/ 以下の合計バイト数
                  - dirs: サブディレクトリ別の容量 (bytes)
                  - file_counts: サブディレクトリ別のファイル数
        """
        data_dir = DATA_DIR

        def dir_size(path: Path) -> tuple[int, int]:
            """(バイト数, ファイル数) を返す。"""
            total_bytes = 0
            total_files = 0
            if not path.is_dir():
                return 0, 0
            for entry in path.rglob("*"):
                if entry.is_file():
                    total_bytes += entry.stat().st_size
                    total_files += 1
            return total_bytes, total_files

        total_bytes, total_files = dir_size(data_dir)

        dirs: dict[str, int] = {}
        file_counts: dict[str, int] = {}
        if data_dir.is_dir():
            for subdir in data_dir.iterdir():
                if subdir.is_dir():
                    size, count = dir_size(subdir)
                    dirs[subdir.name] = size
                    file_counts[subdir.name] = count

        return {
            "total_bytes": total_bytes,
            "total_files": total_files,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "dirs": dirs,
            "file_counts": file_counts,
        }

    # ============================
    # 内部ユーティリティ
    # ============================

    def _load_records(self) -> list[VideoRecord]:
        """履歴JSONを読み込んでVideoRecordのリストを返す。"""
        if not self._history_path.exists():
            return []
        with self._history_path.open("r", encoding="utf-8") as f:
            raw: list[dict[str, Any]] = json.load(f)
        return [VideoRecord.from_dict(item) for item in raw]

    def _save_records(self, records: list[VideoRecord]) -> None:
        """VideoRecordのリストを履歴JSONに保存する。"""
        with self._history_path.open("w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in records],
                f,
                ensure_ascii=False,
                indent=2,
            )
