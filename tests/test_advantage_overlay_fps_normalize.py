"""visualize_advantage_overlay.generate() の 60fps→30fps 正規化 (2026-08-12) テスト。

背景: collect_boards_lean.py (収集) は src.fps_normalize.resolve_normalize_fps_30_stride
により 2026-07-30 から既定で 60fps→stride2 処理している。visualize_advantage_overlay.py
(デモ/レビュー動画生成) だけが全フレーム処理のままだと、認識状態機械のフレーム数定数
(30fps 前提でコメント済み) が実時間半分で発火し、収集・学習データと異なる認識意味論で
動いてしまう。本テストは generate() が同じ正規化を適用することを検証する。

実動画・実CNN・実モデル学習は使わず、tests/test_advantage_overlay_timeline_dump.py と
同じ方式で cv2.VideoCapture/VideoWriter/RecognitionPipeline.load_default/_train_model
を軽量スタブに差し替える。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.fps_normalize import resolve_normalize_fps_30_stride  # noqa: E402


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


class _SpyPipeline:
    """RecognitionPipeline.load_default の軽量スタブ。呼び出し毎の (fi, t) を記録する。

    常に MENU を返す (b1/b2 が STABLE にならないため有利不利の再計算経路には
    入らないが、stride 間引きが pipe.update() を実際に何回・どの (fi,t) で
    呼んでいるかだけを見るには十分)。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def update(self, fi: int, t: float, frame: object) -> _FakeResult:
        self.calls.append((fi, t))
        return _FakeResult()

    def tsumo_count(self, side: str) -> int:
        return 0


class _FakeCapture:
    """cv2.VideoCapture の軽量スタブ (fps/n_frames を自由に設定できる版)。"""

    def __init__(self, fps: float, n_frames: int) -> None:
        self._fps = fps
        self._n = n_frames
        self._i = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        import cv2
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
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
    """cv2.VideoWriter の軽量スタブ。構築時の引数 (fps 等) と write() 回数を記録する。"""

    instances: list["_SpyVideoWriter"] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        _SpyVideoWriter.instances.append(self)
        self.write_count = 0

    def write(self, _frame: object) -> None:
        self.write_count += 1

    def release(self) -> None:
        pass


def _stub(monkeypatch: pytest.MonkeyPatch, fps: float, n_frames: int) -> _SpyPipeline:
    """generate() から実動画・実CNN・実モデル学習を排除する共通スタブを配線する。

    Returns:
        _SpyPipeline インスタンス (呼び出し記録の検査用)。
    """
    monkeypatch.setattr(vao, "_train_model", lambda *_a, **_k: object())
    spy_pipeline = _SpyPipeline()
    monkeypatch.setattr(
        vao.RecognitionPipeline, "load_default",
        staticmethod(lambda **_k: spy_pipeline),
    )
    monkeypatch.setattr(vao.cv2, "VideoCapture", lambda *_a: _FakeCapture(fps, n_frames))
    _SpyVideoWriter.instances = []
    monkeypatch.setattr(vao.cv2, "VideoWriter", _SpyVideoWriter)
    return spy_pipeline


class TestNormalizeFps30Stride:
    """60fps→stride2 (実効30fps) が選ばれ、pipe.update() 呼び出しが間引かれる。"""

    def test_60fps_selects_stride_2_and_halves_pipeline_updates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        n_frames = 20
        spy = _stub(monkeypatch, fps=60.0, n_frames=n_frames)
        written = vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15,
        )
        # resolve_normalize_fps_30_stride(60.0) == 2 (既存テストで確認済みの純関数)
        assert resolve_normalize_fps_30_stride(60.0) == 2
        # stride 対象フレーム (fi=0,2,4,...,18) のみ pipe.update() が呼ばれる
        assert [fi for fi, _t in spy.calls] == list(range(0, n_frames, 2))
        assert written == n_frames // 2

    def test_60fps_writer_fps_is_effective_30(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub(monkeypatch, fps=60.0, n_frames=20)
        vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15,
        )
        assert len(_SpyVideoWriter.instances) == 1
        # cv2.VideoWriter(path, fourcc, fps, size) の第3引数 (位置引数) が fps
        written_fps = _SpyVideoWriter.instances[0].args[2]
        assert written_fps == pytest.approx(30.0)

    def test_t_sec_is_absolute_wall_clock_time_not_compressed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """stride で間引いても t (pipe.update に渡る時刻) は fi/fps の絶対時刻の
        まま (収集側 collect_boards_lean.py:831 と同じ方式)。間引き後の連番
        (0,1,2,...) を fps で割った"圧縮された"時刻になっていないことを保証する。
        """
        spy = _stub(monkeypatch, fps=60.0, n_frames=20)
        vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15,
        )
        expected_ts = [fi / 60.0 for fi in range(0, 20, 2)]
        actual_ts = [t for _fi, t in spy.calls]
        assert actual_ts == pytest.approx(expected_ts)


class TestNormalizeFps30BackwardsCompat:
    """30fps 入力・normalize_fps_30=False は stride=1 で従来挙動と完全一致する。"""

    def test_30fps_input_is_unaffected_stride_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        n_frames = 20
        spy = _stub(monkeypatch, fps=30.0, n_frames=n_frames)
        written = vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15,
        )
        assert resolve_normalize_fps_30_stride(30.0) == 1
        # 全フレームで pipe.update() が呼ばれる (間引きなし、従来挙動)
        assert [fi for fi, _t in spy.calls] == list(range(n_frames))
        assert written == n_frames
        written_fps = _SpyVideoWriter.instances[0].args[2]
        assert written_fps == pytest.approx(30.0)

    def test_normalize_fps_30_false_disables_stride_even_at_60fps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """normalize_fps_30=False を明示すると 60fps でも stride=1 (全フレーム処理、
        従来 (2026-08-12 以前) の挙動を完全再現する)。"""
        n_frames = 20
        spy = _stub(monkeypatch, fps=60.0, n_frames=n_frames)
        written = vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15, normalize_fps_30=False,
        )
        assert [fi for fi, _t in spy.calls] == list(range(n_frames))
        assert written == n_frames
        written_fps = _SpyVideoWriter.instances[0].args[2]
        assert written_fps == pytest.approx(60.0)

    def test_default_matches_production_config_flag(self) -> None:
        """generate() の既定値は production_config の単一情報源と一致する
        (CLAUDE.md「採用フラグは production_config.py が単一情報源」規約)。
        """
        import inspect
        from src.production_config import OVERLAY_NORMALIZE_FPS_30_ENABLED_BY_DEFAULT
        sig = inspect.signature(vao.generate)
        assert (
            sig.parameters["normalize_fps_30"].default
            == OVERLAY_NORMALIZE_FPS_30_ENABLED_BY_DEFAULT
        )
