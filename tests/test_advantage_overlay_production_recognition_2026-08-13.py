"""visualize_advantage_overlay.generate() の本番構成配線是正 (2026-08-13) テスト。

根因調査 (2026-08-13) の副次発見:
  1. RECOGNITION_ADOPTED (本番採用の認識フラグ群、effect-gate/burst-guard-v2 等)
     が RecognitionPipeline.load_default() へ一切転送されておらず、デモ/レビュー
     動画が本番より劣化した認識で生成されていた (2026-08-08 の
     --early-fire-reaction 付け忘れ事故と同型)。
  2. 認識入力が 1920x1080 に正規化されず、表示キャンバス用サイズ
     (OUT_W/OUT_H=1280x720) のまま RecognitionPipeline.update() に渡っていた
     (BoardRegion の絶対px座標較正は 1920x1080 前提のため座標系が不整合。
     720p 入力 + burst-guard 有効でクラッシュすることを診断で実証済み)。

本テストは tests/test_advantage_overlay_fps_normalize.py と同じ方式で
cv2.VideoCapture/VideoWriter/RecognitionPipeline.load_default/_train_model を
軽量スタブに差し替え、実動画・実CNN・実モデル学習を使わずに配線のみを検証する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.production_config import (  # noqa: E402
    OVERLAY_PRODUCTION_RECOGNITION_ENABLED_BY_DEFAULT,
    OVERLAY_RESIZE_1080P_ENABLED_BY_DEFAULT,
    RECOGNITION_ADOPTED,
    recognition_load_default_kwargs,
)


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
    """RecognitionPipeline.load_default の軽量スタブ。

    update() 呼び出し毎の frame.shape を記録する (1080p正規化の検証用)。
    """

    def __init__(self) -> None:
        self.frame_shapes: list[tuple[int, ...]] = []

    def update(self, fi: int, t: float, frame: object) -> _FakeResult:
        self.frame_shapes.append(frame.shape)  # type: ignore[union-attr]
        return _FakeResult()

    def tsumo_count(self, side: str) -> int:
        return 0


class _FakeCapture:
    """cv2.VideoCapture の軽量スタブ (任意の解像度・フレーム数を指定できる版)。"""

    def __init__(self, fps: float, n_frames: int, w: int, h: int) -> None:
        self._fps = fps
        self._n = n_frames
        self._w = w
        self._h = h
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
        return True, np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


class _SpyVideoWriter:
    """cv2.VideoWriter の軽量スタブ。構築引数と write() 回数を記録する。"""

    instances: list["_SpyVideoWriter"] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        _SpyVideoWriter.instances.append(self)
        self.write_count = 0

    def write(self, _frame: object) -> None:
        self.write_count += 1

    def release(self) -> None:
        pass


def _stub(
    monkeypatch: pytest.MonkeyPatch, *, w: int = vao.OUT_W, h: int = vao.OUT_H,
    fps: float = 30.0, n_frames: int = 10,
) -> tuple[_SpyPipeline, list[dict]]:
    """generate() から実動画・実CNN・実モデル学習を排除する共通スタブを配線する。

    Returns:
        (spy_pipeline, load_default_calls) — 後者は load_default() へ渡された
        kwargs の記録 (フラグ自動適用の検証用)。
    """
    monkeypatch.setattr(vao, "_train_model", lambda *_a, **_k: object())
    spy_pipeline = _SpyPipeline()
    load_default_calls: list[dict] = []

    def _fake_load_default(**kwargs: object) -> _SpyPipeline:
        load_default_calls.append(kwargs)
        return spy_pipeline

    monkeypatch.setattr(
        vao.RecognitionPipeline, "load_default", staticmethod(_fake_load_default),
    )
    monkeypatch.setattr(
        vao.cv2, "VideoCapture", lambda *_a: _FakeCapture(fps, n_frames, w, h),
    )
    _SpyVideoWriter.instances = []
    monkeypatch.setattr(vao.cv2, "VideoWriter", _SpyVideoWriter)
    return spy_pipeline, load_default_calls


class TestProductionRecognitionAutoApply:
    """既定 (use_production_recognition=True) で RECOGNITION_ADOPTED が
    load_default() へ自動転送されること (項目1の是正確認)。"""

    def test_default_applies_all_recognition_adopted_kwargs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _spy, calls = _stub(monkeypatch)
        vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15,
        )
        assert len(calls) == 1
        expected = recognition_load_default_kwargs()
        assert expected, "RECOGNITION_ADOPTED が空では検証にならない"
        for key, value in expected.items():
            assert calls[0].get(key) == value, f"{key} が転送されていない"

    def test_no_production_recognition_skips_adopted_kwargs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--no-production-recognition 相当 (use_production_recognition=False) では
        RECOGNITION_ADOPTED のキーが一切渡らない (A/B比較・後方互換確認)。"""
        _spy, calls = _stub(monkeypatch)
        vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15,
            use_production_recognition=False,
        )
        assert len(calls) == 1
        for key in recognition_load_default_kwargs():
            assert key not in calls[0], f"{key} が無効化時にも渡っている"

    def test_default_matches_production_config_flag(self) -> None:
        """generate() の既定値が production_config の単一情報源と一致すること
        (CLAUDE.md「採用フラグは production_config.py が単一情報源」規約)。"""
        import inspect
        sig = inspect.signature(vao.generate)
        assert (
            sig.parameters["use_production_recognition"].default
            == OVERLAY_PRODUCTION_RECOGNITION_ENABLED_BY_DEFAULT
        )

    def test_recognition_adopted_not_empty(self) -> None:
        """本テストの前提 (RECOGNITION_ADOPTED に検証対象がある) を保証する。"""
        assert len(RECOGNITION_ADOPTED) >= 1


class TestResize1080pNormalization:
    """認識入力を 1920x1080 に正規化してから pipe.update() へ渡すこと
    (項目2の是正確認、720p入力でのクラッシュ回避)。"""

    def test_720p_input_is_upscaled_to_1080p_for_recognition(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """720p (1280x720) 入力でも resize_1080p=True (既定) なら
        RecognitionPipeline.update() に渡る frame は 1920x1080 になる。"""
        spy, _calls = _stub(monkeypatch, w=1280, h=720, n_frames=5)
        vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15,
        )
        assert spy.frame_shapes, "pipe.update() が一度も呼ばれていない"
        for shape in spy.frame_shapes:
            assert shape[:2] == (vao.NATIVE_H, vao.NATIVE_W)

    def test_native_1080p_input_is_not_redundantly_resized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """既に 1920x1080 の入力ではリサイズ判定が no-op で通ること
        (最頻ケースでの余計な劣化が無いことの確認)。"""
        spy, _calls = _stub(monkeypatch, w=1920, h=1080, n_frames=5)
        vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15,
        )
        for shape in spy.frame_shapes:
            assert shape[:2] == (vao.NATIVE_H, vao.NATIVE_W)

    def test_resize_1080p_false_reproduces_legacy_behavior(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """resize_1080p=False (--no-resize-1080p 相当) では従来通り
        OUT_W/OUT_H (1280x720) のフレームがそのまま認識に渡る (backwards compat)。"""
        spy, _calls = _stub(monkeypatch, w=1920, h=1080, n_frames=5)
        vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15, resize_1080p=False,
        )
        for shape in spy.frame_shapes:
            assert shape[:2] == (vao.OUT_H, vao.OUT_W)

    def test_default_matches_production_config_flag(self) -> None:
        """generate() の既定値が production_config の単一情報源と一致すること。"""
        import inspect
        sig = inspect.signature(vao.generate)
        assert (
            sig.parameters["resize_1080p"].default
            == OVERLAY_RESIZE_1080P_ENABLED_BY_DEFAULT
        )


class TestRenderingUnaffectedByRecognitionResize:
    """1080p正規化は描画・出力キャンバスサイズに影響しないこと (720p入力でも
    描画が壊れない/クラッシュしないことの確認)。"""

    def test_720p_input_still_writes_expected_canvas_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub(monkeypatch, w=1280, h=720, n_frames=5)
        written = vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15,
        )
        assert written > 0
        assert len(_SpyVideoWriter.instances) == 1
        # VideoWriter(path, fourcc, fps, (canvas_w, canvas_h)) の canvas size は
        # 認識用リサイズと独立 (overlay レイアウト既定の OUT_W/CANVAS_H)。
        canvas_size = _SpyVideoWriter.instances[0].args[3]
        assert canvas_size == (vao.OUT_W, vao.CANVAS_H)
        assert _SpyVideoWriter.instances[0].write_count == written

    def test_panel_layout_also_unaffected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """layout='panel' (1920x1080キャンバス) でも720p入力の認識正規化と
        独立に描画できること。"""
        _stub(monkeypatch, w=1280, h=720, n_frames=5)
        written = vao.generate(
            Path("dummy_never_opened.mp4"), tmp_path / "out.mp4",
            max_sec=1.0, sample_interval=0.15, layout="panel",
        )
        assert written > 0
        canvas_size = _SpyVideoWriter.instances[0].args[3]
        assert canvas_size == (vao.PANEL_CANVAS_W, vao.PANEL_CANVAS_H)
