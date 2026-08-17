"""scripts/visualize_recognition.py の本番構成配線是正 (2026-08-13) テスト。

横展開監査 (docs/CROSS_CUTTING_AUDIT_2026-08-13.md P1) の発見:
scripts/visualize_advantage_overlay.py (eacb1f3) と同型の配線漏れが
scripts/visualize_recognition.py にも存在した。本ファイルは
RECOGNITION_ADOPTED (バーストガード等6フラグ) の CLI 既定値が全て False の
ままで、明示的にフラグを渡さない限りレビュー動画が本番より劣化した認識で
生成されていた。VISUALIZATION_ADOPTED (連鎖数実測化/連鎖表示ホールド) も
同様。

本テストは2層構成:
  1. `resolve_production_config_overrides` の純関数テスト (全分岐)。
  2. `main()` レベルの転送値検証 (`RecognitionPipeline.load_default` /
     `cv2.VideoCapture` / `cv2.VideoWriter` を軽量スタブに差し替え、実動画・
     実CNNを使わずに配線のみを検証する。
     tests/test_advantage_overlay_production_recognition_2026-08-13.py と
     同じ「文字列があるかでなく実際に渡された値を検証する」パターン)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_recognition as vr  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.production_config import (  # noqa: E402
    RECOGNITION_ADOPTED,
    VISUALIZATION_ADOPTED,
    recognition_load_default_kwargs,
)


class _NS:
    """argparse.Namespace の軽量代替 (テスト専用、属性を自由に生やせる)。"""


def _bare_args() -> _NS:
    """RECOGNITION_ADOPTED 6キー・VISUALIZATION_ADOPTED 2キーの生 CLI 値を
    「一切明示指定していない」状態で持つ args を作る。"""
    a = _NS()
    a.enable_effect_gate = False
    a.enable_burst_guard_v2 = False
    a.enable_transition_merge_guard = False
    a.enable_hidden_row_burst_guard = False
    a.enable_match_transition_debounce = False
    a.burst_gate_open_threshold = None
    a.enable_chain_formula_simulate_verify = None  # BooleanOptionalAction 既定 None
    a.overlay_chain_hold_until_end = False
    a.no_overlay_chain_hold_until_end = False
    return a


class TestResolveProductionConfigOverridesUnit:
    """resolve_production_config_overrides の純関数テスト (全分岐)。"""

    def test_production_on_applies_recognition_and_visualization_adopted(self) -> None:
        resolved = vr.resolve_production_config_overrides(_bare_args(), True, True)
        expected_recognition = recognition_load_default_kwargs()
        assert expected_recognition, "RECOGNITION_ADOPTED が空では検証にならない"
        for key, value in expected_recognition.items():
            assert resolved[key] == value, f"{key} が本番値に解決されていない"
        assert resolved["enable_chain_formula_simulate_verify"] is True
        assert resolved["overlay_chain_hold_until_end"] is True

    def test_production_off_reproduces_legacy_all_disabled(self) -> None:
        resolved = vr.resolve_production_config_overrides(_bare_args(), False, False)
        assert resolved["enable_effect_gate"] is False
        assert resolved["enable_burst_guard_v2"] is False
        assert resolved["enable_transition_merge_guard"] is False
        assert resolved["enable_hidden_row_burst_guard"] is False
        assert resolved["enable_match_transition_debounce"] is False
        assert resolved["burst_gate_open_threshold"] is None
        assert resolved["enable_chain_formula_simulate_verify"] is False
        assert resolved["overlay_chain_hold_until_end"] is False

    def test_explicit_recognition_flag_wins_even_when_production_off(self) -> None:
        args = _bare_args()
        args.enable_effect_gate = True
        resolved = vr.resolve_production_config_overrides(args, False, False)
        assert resolved["enable_effect_gate"] is True
        assert resolved["enable_burst_guard_v2"] is False

    def test_explicit_no_chain_formula_simulate_verify_wins_even_when_production_on(
        self,
    ) -> None:
        """--no-chain-formula-simulate-verify (明示 False) は production ON でも維持。"""
        args = _bare_args()
        args.enable_chain_formula_simulate_verify = False
        resolved = vr.resolve_production_config_overrides(args, True, True)
        assert resolved["enable_chain_formula_simulate_verify"] is False

    def test_explicit_no_overlay_chain_hold_until_end_wins_even_when_production_on(
        self,
    ) -> None:
        args = _bare_args()
        args.no_overlay_chain_hold_until_end = True
        resolved = vr.resolve_production_config_overrides(args, True, True)
        assert resolved["overlay_chain_hold_until_end"] is False

    def test_recognition_adopted_and_visualization_adopted_not_empty(self) -> None:
        """本テストの前提 (検証対象がある) を保証する。"""
        assert len(RECOGNITION_ADOPTED) >= 1
        assert len(VISUALIZATION_ADOPTED) >= 1


# ============================
# main() レベルの転送値検証 (cv2/RecognitionPipeline スタブ)
# ============================


class _FakeSide:
    def __init__(self) -> None:
        self.state = BoardState.MENU
        self.confirmed_board = None
        self.score: "int | None" = None
        self.next_pair = None
        self.dnext_pair = None
        self.prob_board = None
        self.erasure_alerts = None
        self.chain_event = None


class _FakeResult:
    def __init__(self) -> None:
        self.p1 = _FakeSide()
        self.p2 = _FakeSide()


class _SpyPipeline:
    """RecognitionPipeline.load_default の軽量スタブ (_reader/_online_hsv は
    None にして main() 側の hasattr/is not None ガードを自然に通す)。"""

    def __init__(self) -> None:
        self._reader = None
        self._online_hsv = None

    def update(self, fi: int, t: float, frame: object) -> _FakeResult:
        return _FakeResult()


class _FakeCapture:
    """cv2.VideoCapture の軽量スタブ。"""

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
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        return 0.0

    def read(self) -> "tuple[bool, np.ndarray | None]":
        if self._i >= self._n:
            return False, None
        self._i += 1
        return True, np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


class _SpyVideoWriter:
    """cv2.VideoWriter の軽量スタブ。"""

    instances: list["_SpyVideoWriter"] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        _SpyVideoWriter.instances.append(self)

    def write(self, _frame: object) -> None:
        pass

    def release(self) -> None:
        pass


def _stub_main_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict]:
    """main() から実動画・実CNNを排除する共通スタブを配線する。

    Returns:
        load_default() へ渡された kwargs のリスト (呼び出し1回分)。
    """
    load_default_calls: list[dict] = []

    def _fake_load_default(**kwargs: object) -> _SpyPipeline:
        load_default_calls.append(kwargs)
        return _SpyPipeline()

    monkeypatch.setattr(
        vr.RecognitionPipeline, "load_default", staticmethod(_fake_load_default),
    )
    monkeypatch.setattr(
        vr.cv2, "VideoCapture",
        lambda *_a: _FakeCapture(fps=30.0, n_frames=3, w=1920, h=1080),
    )
    _SpyVideoWriter.instances = []
    monkeypatch.setattr(vr.cv2, "VideoWriter", _SpyVideoWriter)
    return load_default_calls


class TestMainForwardsResolvedFlagsToLoadDefault:
    """main() が実際に RecognitionPipeline.load_default へ渡す kwargs を検証する。"""

    def test_default_applies_recognition_adopted_kwargs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        calls = _stub_main_dependencies(monkeypatch)
        monkeypatch.setattr(sys, "argv", [
            "visualize_recognition.py",
            "--video", "dummy_never_opened.mp4",
            "--output", str(tmp_path / "out.mp4"),
            "--max-sec", "0.1",
        ])
        rc = vr.main()
        assert rc is None or rc == 0
        assert len(calls) == 1
        expected = recognition_load_default_kwargs()
        for key, value in expected.items():
            assert calls[0].get(key) == value, f"{key} が既定で本番値に解決されていない"
        assert calls[0].get("enable_chain_formula_simulate_verify") is True

    def test_no_production_recognition_skips_recognition_adopted_kwargs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        calls = _stub_main_dependencies(monkeypatch)
        monkeypatch.setattr(sys, "argv", [
            "visualize_recognition.py",
            "--video", "dummy_never_opened.mp4",
            "--output", str(tmp_path / "out.mp4"),
            "--max-sec", "0.1",
            "--no-production-recognition",
        ])
        vr.main()
        assert len(calls) == 1
        for key in recognition_load_default_kwargs():
            assert calls[0].get(key) in (False, None), (
                f"{key} が --no-production-recognition 指定時にも本番値のまま"
            )

    def test_no_production_visualization_skips_chain_formula_simulate_verify(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        calls = _stub_main_dependencies(monkeypatch)
        monkeypatch.setattr(sys, "argv", [
            "visualize_recognition.py",
            "--video", "dummy_never_opened.mp4",
            "--output", str(tmp_path / "out.mp4"),
            "--max-sec", "0.1",
            "--no-production-visualization",
        ])
        vr.main()
        assert len(calls) == 1
        assert calls[0].get("enable_chain_formula_simulate_verify") is False
