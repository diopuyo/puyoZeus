"""
storage.py のテスト
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.storage import (
    STATUS_CLEANED,
    STATUS_DOWNLOADED,
    STATUS_ERROR,
    STATUS_PENDING,
    STATUS_PROCESSED,
    StorageManager,
    VideoRecord,
)

SAMPLE_URL = "https://www.youtube.com/watch?v=test_video_001"
SAMPLE_URL_2 = "https://www.youtube.com/watch?v=test_video_002"


# ============================
# フィクスチャ
# ============================

@pytest.fixture
def tmp_manager(tmp_path: Path) -> StorageManager:
    """一時ディレクトリを使うStorageManagerを生成する。"""
    history_file = tmp_path / "video_history.json"
    return StorageManager(history_path=history_file)


# ============================
# VideoRecord テスト
# ============================

class TestVideoRecord:
    def test_default_status_is_pending(self):
        record = VideoRecord(url=SAMPLE_URL)
        assert record.status == STATUS_PENDING

    def test_to_dict_contains_url(self):
        record = VideoRecord(url=SAMPLE_URL)
        d = record.to_dict()
        assert d["url"] == SAMPLE_URL

    def test_round_trip(self):
        record = VideoRecord(
            url=SAMPLE_URL,
            frame_count=100,
            status=STATUS_PROCESSED,
            metadata={"title": "テスト動画"},
        )
        restored = VideoRecord.from_dict(record.to_dict())
        assert restored.url == record.url
        assert restored.frame_count == record.frame_count
        assert restored.status == record.status
        assert restored.metadata == record.metadata

    def test_recorded_at_is_set_automatically(self):
        record = VideoRecord(url=SAMPLE_URL)
        assert record.recorded_at != ""


# ============================
# record_video テスト
# ============================

class TestRecordVideo:
    def test_record_new_url(self, tmp_manager: StorageManager):
        record = tmp_manager.record_video(SAMPLE_URL)
        assert record.url == SAMPLE_URL
        assert record.status == STATUS_PENDING

    def test_record_with_metadata(self, tmp_manager: StorageManager):
        meta = {"title": "ぷよぷよ対戦", "duration": 300}
        record = tmp_manager.record_video(SAMPLE_URL, metadata=meta)
        assert record.metadata["title"] == "ぷよぷよ対戦"
        assert record.metadata["duration"] == 300

    def test_duplicate_url_updates_metadata(self, tmp_manager: StorageManager):
        tmp_manager.record_video(SAMPLE_URL, metadata={"title": "初回"})
        record = tmp_manager.record_video(SAMPLE_URL, metadata={"title": "更新"})
        # URLが重複していても件数は1件
        history = tmp_manager.get_history()
        assert len(history) == 1
        assert record.metadata["title"] == "更新"

    def test_multiple_urls_recorded(self, tmp_manager: StorageManager):
        tmp_manager.record_video(SAMPLE_URL)
        tmp_manager.record_video(SAMPLE_URL_2)
        history = tmp_manager.get_history()
        assert len(history) == 2


# ============================
# get_history テスト
# ============================

class TestGetHistory:
    def test_empty_history(self, tmp_manager: StorageManager):
        assert tmp_manager.get_history() == []

    def test_history_persists_to_json(self, tmp_path: Path):
        history_file = tmp_path / "history.json"
        mgr = StorageManager(history_path=history_file)
        mgr.record_video(SAMPLE_URL)

        # 別インスタンスで読み直す
        mgr2 = StorageManager(history_path=history_file)
        history = mgr2.get_history()
        assert len(history) == 1
        assert history[0].url == SAMPLE_URL

    def test_get_record_existing(self, tmp_manager: StorageManager):
        tmp_manager.record_video(SAMPLE_URL)
        record = tmp_manager.get_record(SAMPLE_URL)
        assert record is not None
        assert record.url == SAMPLE_URL

    def test_get_record_not_found(self, tmp_manager: StorageManager):
        result = tmp_manager.get_record("https://example.com/nonexistent")
        assert result is None


# ============================
# update_status テスト
# ============================

class TestUpdateStatus:
    def test_update_status_to_processed(self, tmp_manager: StorageManager):
        tmp_manager.record_video(SAMPLE_URL)
        record = tmp_manager.update_status(SAMPLE_URL, STATUS_PROCESSED, frame_count=120)
        assert record.status == STATUS_PROCESSED
        assert record.frame_count == 120

    def test_update_status_persists(self, tmp_path: Path):
        history_file = tmp_path / "history.json"
        mgr = StorageManager(history_path=history_file)
        mgr.record_video(SAMPLE_URL)
        mgr.update_status(SAMPLE_URL, STATUS_DOWNLOADED)

        mgr2 = StorageManager(history_path=history_file)
        record = mgr2.get_record(SAMPLE_URL)
        assert record is not None
        assert record.status == STATUS_DOWNLOADED

    def test_update_invalid_status(self, tmp_manager: StorageManager):
        tmp_manager.record_video(SAMPLE_URL)
        with pytest.raises(ValueError, match="無効なステータス"):
            tmp_manager.update_status(SAMPLE_URL, "invalid_status")

    def test_update_nonexistent_url(self, tmp_manager: StorageManager):
        with pytest.raises(ValueError, match="URLが履歴に存在しません"):
            tmp_manager.update_status("https://nonexistent.com", STATUS_PROCESSED)


# ============================
# cleanup テスト
# ============================

class TestCleanup:
    def test_cleanup_existing_file(self, tmp_manager: StorageManager, tmp_path: Path):
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video data")

        result = tmp_manager.cleanup(str(video_file))
        assert result is True
        assert not video_file.exists()

    def test_cleanup_nonexistent_file(self, tmp_manager: StorageManager, tmp_path: Path):
        result = tmp_manager.cleanup(str(tmp_path / "nonexistent.mp4"))
        assert result is False

    def test_cleanup_url_record_preserved(self, tmp_manager: StorageManager, tmp_path: Path):
        """動画ファイルを削除してもURLレコードは保持される。"""
        tmp_manager.record_video(SAMPLE_URL)
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"data")

        tmp_manager.cleanup(str(video_file))
        history = tmp_manager.get_history()
        assert len(history) == 1
        assert history[0].url == SAMPLE_URL

    def test_cleanup_directory(self, tmp_manager: StorageManager, tmp_path: Path):
        # 動画ファイルを複数作成
        (tmp_path / "a.mp4").write_bytes(b"data")
        (tmp_path / "b.webm").write_bytes(b"data")
        (tmp_path / "keep.json").write_bytes(b"{}")  # JSON は削除されない

        count = tmp_manager.cleanup_directory(str(tmp_path))
        assert count == 2
        assert not (tmp_path / "a.mp4").exists()
        assert not (tmp_path / "b.webm").exists()
        assert (tmp_path / "keep.json").exists()  # JSON は残る

    def test_cleanup_nonexistent_directory(self, tmp_manager: StorageManager):
        count = tmp_manager.cleanup_directory("/nonexistent/path")
        assert count == 0


# ============================
# get_storage_usage テスト
# ============================

class TestGetStorageUsage:
    def test_usage_returns_dict_with_required_keys(self, tmp_manager: StorageManager):
        usage = tmp_manager.get_storage_usage()
        assert "total_bytes" in usage
        assert "total_files" in usage
        assert "total_mb" in usage
        assert "dirs" in usage
        assert "file_counts" in usage

    def test_usage_total_mb_is_float(self, tmp_manager: StorageManager):
        usage = tmp_manager.get_storage_usage()
        assert isinstance(usage["total_mb"], float)
