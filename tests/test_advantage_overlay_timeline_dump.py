"""タイムラインdump工事 (2026-08-11) のテスト。

- TimelineDumpRow の npz 保存/読み込みの往復整合 (純粋なデータ層、動画不要)
- generate() の render=False / dump_timeline_path 配線
  (RecognitionPipeline.load_default・cv2.VideoCapture/VideoWriter・
  _train_model を全て軽量スタブに差し替え、実動画・実モデル無しで検証する)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402


# ============================
# save_timeline_dump / load_timeline_dump の往復整合
# ============================

def _sample_rows() -> list[vao.TimelineDumpRow]:
    return [
        vao.TimelineDumpRow(
            t_sec=1.5, game_idx=0, adv_raw=12.3, adv_ema=10.1, p1=0.6,
            pending_p1=5, pending_p2=0, room1=60, room2=72,
            is_dead1=False, is_dead2=False,
            drivers_top1_name="board_color_puyo_total", drivers_top1_val=0.4,
            drivers_top3_names=("board_color_puyo_total", "max_column_height", ""),
            drivers_top3_vals=(0.4, -0.2, 0.0),
            score1=1200, score2=800, b1_hash=111, b2_hash=222,
            state1="STABLE", state2="STABLE",
        ),
        vao.TimelineDumpRow(
            t_sec=3.25, game_idx=1, adv_raw=-40.0, adv_ema=-38.2, p1=0.1,
            pending_p1=0, pending_p2=80, room1=72, room2=4,
            is_dead1=False, is_dead2=True,
            drivers_top1_name="", drivers_top1_val=0.0,
            drivers_top3_names=("", "", ""), drivers_top3_vals=(0.0, 0.0, 0.0),
            score1=vao.TIMELINE_DUMP_SCORE_NONE_SENTINEL, score2=5000,
            b1_hash=333, b2_hash=444, state1="CHAIN", state2="STABLE",
        ),
    ]


class TestTimelineDumpRoundTrip:
    def test_save_load_preserves_all_fields(self, tmp_path: Path) -> None:
        rows = _sample_rows()
        path = tmp_path / "v_test.npz"
        vao.save_timeline_dump(path, "v_test", rows)
        video_id, loaded = vao.load_timeline_dump(path)
        assert video_id == "v_test"
        assert len(loaded) == len(rows)
        for original, restored in zip(rows, loaded):
            assert restored == original

    def test_empty_rows_round_trip(self, tmp_path: Path) -> None:
        """0レコード (settled が一度も起きなかった動画) でも例外にならない。"""
        path = tmp_path / "v_empty.npz"
        vao.save_timeline_dump(path, "v_empty", [])
        video_id, loaded = vao.load_timeline_dump(path)
        assert video_id == "v_empty"
        assert loaded == []

    def test_none_score_sentinel_round_trips(self, tmp_path: Path) -> None:
        """score=None (OCR失敗) は TIMELINE_DUMP_SCORE_NONE_SENTINEL に変換される
        (npz は int 列のため None をそのまま格納できない)。
        """
        board = _empty_board()
        row = vao._build_timeline_dump_row(
            t_sec=0.0, game_idx=0, adv_raw=0.0, adv_ema=0.0, p1=0.5,
            pending_p1=0, pending_p2=0, room1=72, room2=72,
            b1=board, b2=board,
            drivers=[], score1=None, score2=None, state1="MENU", state2="MENU",
        )
        assert row.score1 == vao.TIMELINE_DUMP_SCORE_NONE_SENTINEL
        assert row.score2 == vao.TIMELINE_DUMP_SCORE_NONE_SENTINEL


def _empty_board() -> Any:
    from src.board import Board
    return Board.from_list(np.zeros((13, 6), dtype=np.int8).tolist())


# ============================
# generate() render / dump 配線 (実動画・実モデル不要の軽量スタブ)
# ============================

class _FakeSide:
    def __init__(self) -> None:
        self.state = BoardState.MENU
        self.confirmed_board = None
        self.score: int | None = None
        self.chain_event = None
        self.next_pair = None


class _FakeResult:
    def __init__(self) -> None:
        self.p1 = _FakeSide()
        self.p2 = _FakeSide()


class _FakePipeline:
    """RecognitionPipeline.load_default の軽量スタブ。常に MENU 状態を返す
    (b1/b2 が STABLE にならないため有利不利の再計算経路には入らないが、
    render/dump の配線=描画有無・npz保存有無だけを見るには十分)。"""

    def update(self, fi: int, t: float, frame: object) -> _FakeResult:
        return _FakeResult()

    def tsumo_count(self, side: str) -> int:
        return 0


class _FakeCapture:
    """cv2.VideoCapture の軽量スタブ (n_frames 枚のダミーフレームを返す)。"""

    def __init__(self, n_frames: int = 4) -> None:
        self._n = n_frames
        self._i = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        import cv2
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._n)
        return 0.0

    def set(self, *_a: object) -> bool:
        return True

    def read(self) -> tuple[bool, "np.ndarray | None"]:
        if self._i >= self._n:
            return False, None
        self._i += 1
        return True, np.zeros((vao.OUT_H, vao.OUT_W, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


class _SpyVideoWriter:
    """cv2.VideoWriter の軽量スタブ。生成された回数を記録する。"""

    instances: list["_SpyVideoWriter"] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        _SpyVideoWriter.instances.append(self)
        self.write_count = 0

    def write(self, _frame: object) -> None:
        self.write_count += 1

    def release(self) -> None:
        pass


@pytest.fixture()
def _stub_heavy_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """generate() から実動画・実CNN・実モデル学習を排除する共通スタブ。"""
    monkeypatch.setattr(vao, "_train_model", lambda *_a, **_k: object())
    monkeypatch.setattr(
        vao.RecognitionPipeline, "load_default",
        staticmethod(lambda **_k: _FakePipeline()),
    )
    monkeypatch.setattr(vao.cv2, "VideoCapture", lambda *_a: _FakeCapture())
    _SpyVideoWriter.instances = []
    monkeypatch.setattr(vao.cv2, "VideoWriter", _SpyVideoWriter)


class TestGenerateRenderAndDumpWiring:
    def test_default_behavior_unchanged_creates_video_writer(
        self, tmp_path: Path, _stub_heavy_pipeline: None,
    ) -> None:
        """render/dump_timeline_path 省略時 (既存呼出元と同じ呼び方) は従来通り
        VideoWriter が生成される (backwards compat)。"""
        written = vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=0.1, sample_interval=0.15,
        )
        assert len(_SpyVideoWriter.instances) == 1
        assert _SpyVideoWriter.instances[0].write_count == written
        assert written > 0

    def test_no_render_skips_video_writer(
        self, tmp_path: Path, _stub_heavy_pipeline: None,
    ) -> None:
        """render=False では VideoWriter が一切生成されない (動画が作られない)。"""
        written = vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=0.1, sample_interval=0.15, render=False,
        )
        assert _SpyVideoWriter.instances == []
        assert written > 0  # 計算自体は最後まで走る

    def test_dump_timeline_path_writes_npz(
        self, tmp_path: Path, _stub_heavy_pipeline: None,
    ) -> None:
        """dump_timeline_path 指定時は npz が書き出される (中身は0件でも良い、
        本テストは MENU 状態のみのスタブで settled に到達しないため空)。"""
        dump_path = tmp_path / "dump.npz"
        vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=0.1, sample_interval=0.15, render=False,
            dump_timeline_path=dump_path,
        )
        assert dump_path.exists()
        video_id, rows = vao.load_timeline_dump(dump_path)
        assert video_id == "dummy_never_opened"
        assert rows == []  # スタブは MENU 固定 = settled 更新が一度も起きない

    def test_no_dump_path_means_no_file_written(
        self, tmp_path: Path, _stub_heavy_pipeline: None,
    ) -> None:
        """dump_timeline_path=None (既定) では npz を一切書かない (backwards compat)。"""
        out_dir_before = set(tmp_path.iterdir())
        vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=0.1, sample_interval=0.15, render=False,
        )
        new_files = set(tmp_path.iterdir()) - out_dir_before
        assert not any(p.suffix == ".npz" for p in new_files)
