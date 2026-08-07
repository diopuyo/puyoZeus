"""#24 打ち合い計測器 Step5 (scripts/extract_exchange_event_frames.py) の単体テスト。

実動画・実DL・実cv2デコードは行わない軽量テスト (tmp_path + monkeypatch のみ)。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.extract_exchange_event_frames as target
from scripts.extract_exchange_event_frames import (
    FRAME_TAGS,
    _format_event_section,
    _video_id_to_stem,
    compute_event_timestamps,
    download_video_via_ytdlp,
    ensure_video_available,
    grab_frame,
    render_review_sheet,
    resolve_cached_video_path,
    to_windows_path,
)


# =============================================================================
# _video_id_to_stem
# =============================================================================

def test_video_id_to_stem_strips_prefix() -> None:
    assert _video_id_to_stem("video_c27") == "c27"


def test_video_id_to_stem_passthrough_without_prefix() -> None:
    assert _video_id_to_stem("c27") == "c27"


# =============================================================================
# compute_event_timestamps
# =============================================================================

class TestComputeEventTimestamps:
    def test_pre_fire_is_one_second_before(self) -> None:
        ts, _clamped = compute_event_timestamps(t_sec=100.0, sim_k_hands=2.0, approx_fire_chains=4.0)
        assert ts["pre_fire"] == pytest.approx(99.0)
        assert ts["fire"] == pytest.approx(100.0)

    def test_landing_uses_sec_per_hand_and_buffer(self) -> None:
        from src.indicators_v2 import SEC_PER_HAND
        ts, _clamped = compute_event_timestamps(t_sec=100.0, sim_k_hands=3.0, approx_fire_chains=4.0)
        expected_landing = 100.0 + 3.0 * SEC_PER_HAND + target.LANDING_BUFFER_SEC
        assert ts["landing"] == pytest.approx(expected_landing)

    def test_pre_fire_clamped_to_zero_near_video_start(self) -> None:
        ts, _clamped = compute_event_timestamps(t_sec=0.3, sim_k_hands=1.0, approx_fire_chains=4.0)
        assert ts["pre_fire"] == 0.0

    def test_pre_chain_start_uses_chain_duration_and_margin(self) -> None:
        # t_sec=100, approx_fire_chains=4 -> 100 - (4*1.8 + 1.0) = 100 - 8.2 = 91.8
        ts, clamped = compute_event_timestamps(t_sec=100.0, sim_k_hands=2.0, approx_fire_chains=4.0)
        expected = 100.0 - (4.0 * target.CHAIN_DURATION_SEC_PER_CHAIN + target.PRE_CHAIN_START_MARGIN_SEC)
        assert ts["pre_chain_start"] == pytest.approx(expected)
        assert clamped is False

    def test_pre_chain_start_clamped_to_zero_near_video_start(self) -> None:
        # t_sec が小さく、連鎖数が大きい -> 動画先頭 (0秒) にクランプされるはず
        ts, clamped = compute_event_timestamps(t_sec=2.0, sim_k_hands=1.0, approx_fire_chains=10.0)
        assert ts["pre_chain_start"] == 0.0
        assert clamped is True

    def test_pre_chain_start_not_clamped_when_comfortably_positive(self) -> None:
        ts, clamped = compute_event_timestamps(t_sec=500.0, sim_k_hands=1.0, approx_fire_chains=2.0)
        assert ts["pre_chain_start"] > 0.0
        assert clamped is False


# =============================================================================
# resolve_cached_video_path / ensure_video_available (ローカルキャッシュ優先)
# =============================================================================

class TestResolveCachedVideoPath:
    def test_finds_video_in_wsl_native_dir(self, tmp_path, monkeypatch) -> None:
        native_dir = tmp_path / "native"
        native_dir.mkdir()
        (native_dir / "video_c99.mp4").write_bytes(b"dummy")
        monkeypatch.setattr(target, "WSL_NATIVE_FRAMES_DIR", native_dir)
        monkeypatch.setattr(target, "REPO_FRAMES_DIR", tmp_path / "does_not_exist")
        found = resolve_cached_video_path("video_c99")
        assert found == native_dir / "video_c99.mp4"

    def test_falls_back_to_repo_frames_dir(self, tmp_path, monkeypatch) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / "video_c99.mp4").write_bytes(b"dummy")
        monkeypatch.setattr(target, "WSL_NATIVE_FRAMES_DIR", tmp_path / "no_native")
        monkeypatch.setattr(target, "REPO_FRAMES_DIR", repo_dir)
        found = resolve_cached_video_path("video_c99")
        assert found == repo_dir / "video_c99.mp4"

    def test_returns_none_when_not_cached_anywhere(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(target, "WSL_NATIVE_FRAMES_DIR", tmp_path / "a")
        monkeypatch.setattr(target, "REPO_FRAMES_DIR", tmp_path / "b")
        assert resolve_cached_video_path("video_c99") is None


class TestEnsureVideoAvailable:
    def test_cache_hit_reports_not_downloaded(self, tmp_path, monkeypatch) -> None:
        native_dir = tmp_path / "native"
        native_dir.mkdir()
        (native_dir / "video_c99.mp4").write_bytes(b"dummy")
        monkeypatch.setattr(target, "WSL_NATIVE_FRAMES_DIR", native_dir)
        monkeypatch.setattr(target, "REPO_FRAMES_DIR", tmp_path / "no_repo")
        path, was_downloaded = ensure_video_available("video_c99", tmp_path / "dl", url_map=None)
        assert path == native_dir / "video_c99.mp4"
        assert was_downloaded is False

    def test_missing_video_and_no_url_map_returns_none(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(target, "WSL_NATIVE_FRAMES_DIR", tmp_path / "a")
        monkeypatch.setattr(target, "REPO_FRAMES_DIR", tmp_path / "b")
        path, was_downloaded = ensure_video_available("video_unknown", tmp_path / "dl", url_map=None)
        assert path is None
        assert was_downloaded is False


class TestDownloadVideoViaYtdlp:
    def test_returns_none_when_video_id_absent_from_url_map(self, tmp_path) -> None:
        assert download_video_via_ytdlp("video_unknown", tmp_path, url_map={}) is None

    def test_returns_none_when_url_map_is_none(self, tmp_path) -> None:
        assert download_video_via_ytdlp("video_unknown", tmp_path, url_map=None) is None


# =============================================================================
# grab_frame (存在しない動画は None を返す、実デコードはしない)
# =============================================================================

def test_grab_frame_returns_none_for_missing_video(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.mp4"
    assert grab_frame(missing, t_sec=1.0) is None


# =============================================================================
# to_windows_path
# =============================================================================

class TestToWindowsPath:
    def test_converts_mnt_c_prefix(self) -> None:
        result = to_windows_path(Path("/mnt/c/Users/test/frame.png"))
        assert result == "C:\\Users\\test\\frame.png"

    def test_non_mnt_c_path_returns_resolved_str(self, tmp_path) -> None:
        p = tmp_path / "frame.png"
        result = to_windows_path(p)
        assert result == str(p.resolve())


# =============================================================================
# review_sheet.md 生成
# =============================================================================

def _make_event_row() -> pd.Series:
    return pd.Series({
        "video_id": "video_c27", "game_idx": 12, "t_sec": 1271.3, "fire_side": "1P",
        "phase": "序", "approx_fire_chains": 9.0,
        "selection_reason": "序_乖離度1位", "selection_series": "主系列",
        "net_ojama_after_oof_pred": 42.0, "sim_damage_score": 0.7,
        "sim_k_hands": 2.0, "sim_expected_counter_ojama": 10.0,
        "stack_pred_net_ojama_after": 38.0, "net_ojama_after": 40.0,
        "taiou_success": 0, "survived": 1, "closer_to_actual_rank_based": "案D",
    })


class TestFormatEventSection:
    def test_contains_key_fields(self) -> None:
        row = _make_event_row()
        frame_paths = {tag: Path(f"/mnt/c/dummy/{tag}.png") for tag in FRAME_TAGS}
        section = _format_event_section(row, event_no=1, frame_paths=frame_paths, pre_chain_start_clamped=False)
        assert "イベント01" in section
        assert "video_c27" in section
        assert "案D" in section
        assert "失敗" in section  # taiou_success=0
        assert "生存" in section  # survived=1
        assert "approx_fire_chains=9" in section

    def test_missing_frame_reported_as_failure(self) -> None:
        row = _make_event_row()
        frame_paths = {tag: None for tag in FRAME_TAGS}
        section = _format_event_section(row, event_no=1, frame_paths=frame_paths, pre_chain_start_clamped=False)
        assert "フレーム取得失敗" in section

    def test_pre_chain_start_has_estimate_note(self) -> None:
        row = _make_event_row()
        frame_paths = {tag: Path(f"/mnt/c/dummy/{tag}.png") for tag in FRAME_TAGS}
        section = _format_event_section(row, event_no=1, frame_paths=frame_paths, pre_chain_start_clamped=False)
        assert target.PRE_CHAIN_START_ESTIMATE_NOTE in section
        assert target.PRE_CHAIN_START_CLAMPED_NOTE not in section

    def test_pre_chain_start_has_clamped_note_when_clamped(self) -> None:
        row = _make_event_row()
        frame_paths = {tag: Path(f"/mnt/c/dummy/{tag}.png") for tag in FRAME_TAGS}
        section = _format_event_section(row, event_no=1, frame_paths=frame_paths, pre_chain_start_clamped=True)
        assert target.PRE_CHAIN_START_CLAMPED_NOTE in section


class TestRenderReviewSheet:
    def test_writes_file_with_all_events(self, tmp_path) -> None:
        df = pd.DataFrame([_make_event_row(), _make_event_row()])
        frame_paths_by_event = {
            1: {tag: Path(f"/mnt/c/dummy/1_{tag}.png") for tag in FRAME_TAGS},
            2: {tag: None for tag in FRAME_TAGS},
        }
        clamped_by_event = {1: False, 2: True}
        out_path = render_review_sheet(df, frame_paths_by_event, tmp_path, clamped_by_event)
        assert out_path.exists()
        text = out_path.read_text(encoding="utf-8")
        assert "イベント01" in text and "イベント02" in text
        assert "レビューシート" in text

    def test_default_clamped_map_is_all_false(self, tmp_path) -> None:
        """pre_chain_start_clamped_by_event 省略時 (後方互換) は全てFalse扱い。"""
        df = pd.DataFrame([_make_event_row()])
        frame_paths_by_event = {1: {tag: Path(f"/mnt/c/dummy/{tag}.png") for tag in FRAME_TAGS}}
        out_path = render_review_sheet(df, frame_paths_by_event, tmp_path)
        text = out_path.read_text(encoding="utf-8")
        assert target.PRE_CHAIN_START_CLAMPED_NOTE not in text
