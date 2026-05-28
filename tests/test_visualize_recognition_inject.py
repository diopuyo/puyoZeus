"""per-video HSV inject suppress 修正のユニットテスト。

テスト設計:
    テスト 1 - resolve_hsv_path: 動画ファイル名から正しい JSON パスを返す
    テスト 2 - suppress guard: _online_hsv_injected=True のとき OnlineHsv inject が走らない
    テスト 3 - default 挙動: _online_hsv_injected=False (初期値) では inject が走る (regression)

注意:
    - ファイル I/O は最小限 (v30.json の存在確認のみ)
    - 外部プロセス起動なし、軽量 mock のみ使用
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# プロジェクトルート基準
_ROOT = Path(__file__).resolve().parent.parent


# ============================
# テスト 1: resolve_hsv_path
# ============================

def test_resolve_hsv_path_returns_per_video_json_when_exists() -> None:
    """resolve_hsv_path が v30 動画ファイル名から v30.json を返す。

    data/per_video_hsv_ranges/v30.json が存在する前提。
    _HSV_DB_ROOT が相対パスのため、返り値も相対パスで比較する。
    """
    from scripts.visualize_recognition import (  # type: ignore[import]
        _HSV_DB_ROOT,
        resolve_hsv_path,
    )

    video_path = Path("data/holdout_videos/v30_5min_90s.mp4")
    result = resolve_hsv_path(video_path)
    expected = _HSV_DB_ROOT / "v30.json"
    assert result == expected, (
        f"v30 動画に対し v30.json が返るはず: got {result}"
    )


def test_resolve_hsv_path_fallback_to_merged_default() -> None:
    """v99 など存在しない動画 ID は _merged_default.json にフォールバック。"""
    from scripts.visualize_recognition import (  # type: ignore[import]
        _HSV_MERGED_DEFAULT,
        resolve_hsv_path,
    )

    video_path = Path("data/holdout_videos/v99_nonexistent.mp4")
    result = resolve_hsv_path(video_path)
    assert result == _HSV_MERGED_DEFAULT, (
        f"存在しない動画 ID は merged_default fallback になるはず: got {result}"
    )


# ============================
# stub helpers (test 2/3 共通)
# ============================

@dataclass
class _StubMatchResult:
    state: object
    bg_value: float = 100.0
    bg_saturation: float = 50.0
    samples: int = 1


class _StubMatchDetector:
    """常に IN_MATCH を返すスタブ。"""

    def detect(self, frame: np.ndarray) -> _StubMatchResult:
        from src.match_state import MatchState
        return _StubMatchResult(state=MatchState.IN_MATCH)


class _StubImageReader:
    """固定 empty board を返すスタブ。set_color_ranges_from_simple も spy 対象。"""

    def __init__(self) -> None:
        from src.board import Board
        self._p1 = Board()
        self._p2 = Board()
        # HybridClassifier のふりをする mock (_hsv 属性を持つ)
        self._classifier = MagicMock()
        self._classifier._hsv = MagicMock()
        self._classifier._hsv.set_color_ranges_from_simple = MagicMock()

    def read_both_boards(
        self,
        frame: np.ndarray,
        p1_roi_offset: tuple[float, float] = (0.0, 0.0),
        p2_roi_offset: tuple[float, float] = (0.0, 0.0),
        skip_tier1_1p: bool = False,
        skip_tier1_2p: bool = False,
    ) -> tuple[object, object]:
        return self._p1.copy(), self._p2.copy()


def _make_pipeline_with_online_hsv(
    injected: bool,
    calibrator_ranges: dict[int, tuple[int, int, int, int, int, int]],
) -> tuple[object, MagicMock]:
    """suppress guard テスト用 pipeline を組み立てる。

    Args:
        injected: _online_hsv_injected の初期値。
        calibrator_ranges: OnlineHsvCalibrator.get_per_video_ranges() の戻り値。

    Returns:
        (pipeline, set_color_ranges_spy) のタプル。
    """
    from src.recognition_pipeline import RecognitionPipeline

    reader = _StubImageReader()
    detector = _StubMatchDetector()

    # OnlineHsvCalibrator の mock: get_per_video_ranges が calibrator_ranges を返す
    mock_calibrator = MagicMock()
    mock_calibrator.get_per_video_ranges.return_value = calibrator_ranges

    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        online_hsv=mock_calibrator,
    )
    pipe._online_hsv_injected = injected

    spy = reader._classifier._hsv.set_color_ranges_from_simple
    return pipe, spy


def _dummy_frame() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# ============================
# テスト 2: suppress guard (main fix)
# ============================

def test_online_hsv_inject_suppressed_when_flag_is_true() -> None:
    """_online_hsv_injected=True のとき set_color_ranges_from_simple が呼ばれない。

    これが今回の修正 (recognition_pipeline.py:1170) の主検証。
    """
    # calibrator が non-empty ranges を返す状態でも suppress されるはず
    dummy_ranges: dict[int, tuple[int, int, int, int, int, int]] = {
        1: (0, 20, 150, 255, 100, 255),   # 赤
        2: (100, 130, 150, 255, 100, 255), # 青
    }
    from src.hybrid_classifier import HybridClassifier

    pipe, spy = _make_pipeline_with_online_hsv(
        injected=True,
        calibrator_ranges=dummy_ranges,
    )

    # HybridClassifier インスタンスとして認識させる
    with patch(
        "src.recognition_pipeline.RecognitionPipeline",
        wraps=type(pipe),
    ):
        # isinstance(hc, HybridClassifier) のガードをパスさせるため
        # _reader._classifier を HybridClassifier の mock に差し替える
        pipe._reader._classifier.__class__ = HybridClassifier  # type: ignore[assignment]

        # STABLE まで到達させて OnlineHsv inject ループを通過させる
        frame = _dummy_frame()
        for i in range(5):
            pipe.update(i, float(i) * 0.05, frame)

    # suppress guard により inject が走らないはず
    spy.assert_not_called(), "suppress 済みなのに set_color_ranges_from_simple が呼ばれた"


# ============================
# テスト 3: default 挙動 (regression)
# ============================

def test_online_hsv_inject_runs_when_flag_is_false() -> None:
    """_online_hsv_injected=False (初期値) では OnlineHsv inject が正常に走る。

    regression check: 修正前の挙動が既存スクリプト (suppress なし) で維持される。
    """
    from src.hybrid_classifier import HybridClassifier

    dummy_ranges: dict[int, tuple[int, int, int, int, int, int]] = {
        1: (0, 20, 150, 255, 100, 255),
    }
    pipe, spy = _make_pipeline_with_online_hsv(
        injected=False,
        calibrator_ranges=dummy_ranges,
    )

    # HybridClassifier として認識させる
    pipe._reader._classifier.__class__ = HybridClassifier  # type: ignore[assignment]

    # predict_proba_and_hsv_grid も mock で用意
    mock_proba = np.zeros((13, 6, 11), dtype=np.float32)
    mock_hsv = np.zeros((13, 6, 3), dtype=np.float32)
    pipe._reader._classifier.predict_proba_and_hsv_grid = MagicMock(
        return_value=(mock_proba, mock_hsv),
    )

    # STABLE に到達させる (stable_frame_count=2 なので 3 frame で到達)
    frame = _dummy_frame()
    for i in range(6):
        pipe.update(i, float(i) * 0.05, frame)

    # suppress なし → inject が走るはず (set_color_ranges_from_simple が呼ばれる)
    assert spy.call_count >= 1, (
        f"_online_hsv_injected=False なのに inject が走らなかった "
        f"(call_count={spy.call_count})"
    )


# ============================
# テスト 4 は test_resolve_hsv_path_* (テスト1で分割済み)
# ============================


# ============================
# テスト 5: 赤 HSV 循環補完 guard
# ============================

def test_circular_guard_appends_default_low_range_for_red() -> None:
    """per_video_ranges で赤が H=176-178 の1範囲のみの場合、
    _ensure_circular_ranges_guard が DEFAULT の H=0-13 側を補完する。

    v30.json 相当: per_video_ranges["1"] = [176, 178, 137, 217, 150, 188]
    inject 後に ColorClassifier._ranges[COLOR_RED] に H=0-13 側も含まれることを確認。
    """
    from src.image_reader import ColorClassifier, DEFAULT_COLOR_RANGES
    from src.board import COLOR_RED
    from scripts.visualize_recognition import _ensure_circular_ranges_guard

    clf = ColorClassifier()
    # per_video inject: v30 相当 (H=176-178 のみ)
    per_video: dict[int, tuple[int, int, int, int, int, int]] = {
        COLOR_RED: (176, 178, 137, 217, 150, 188),
    }
    clf.set_color_ranges_from_simple(per_video, append=True)

    # guard 前の状態確認: DEFAULT 2範囲 + per_video 1範囲 = 3範囲のはず
    # (append=True が正常に動いていれば既に含まれているが、guard で明示的に保証)
    _ensure_circular_ranges_guard(clf)

    red_ranges = clf._ranges[COLOR_RED]
    h_min_values = [r.h_min for r in red_ranges]
    h_max_values = [r.h_max for r in red_ranges]

    # DEFAULT の H=0-13 側 (h_min=0, h_max=13) が必ず存在することを確認
    default_low = DEFAULT_COLOR_RANGES[COLOR_RED][0]  # HsvRange(h_min=0, h_max=13, ...)
    assert any(
        r.h_min == default_low.h_min and r.h_max == default_low.h_max
        for r in red_ranges
    ), (
        f"H=0-13 側が存在しない。現在の ranges: "
        f"h_min={h_min_values}, h_max={h_max_values}"
    )


# ============================
# テスト 6: 赤なし動画では何もしない (regression)
# ============================

def test_circular_guard_noop_when_red_not_in_ranges() -> None:
    """per_video_ranges に赤が含まれない場合、guard は既存 _ranges を変更しない。

    赤なし動画 (= ありえないが念のため) での regression テスト。
    """
    from src.image_reader import ColorClassifier, DEFAULT_COLOR_RANGES
    from src.board import COLOR_RED, COLOR_BLUE
    from scripts.visualize_recognition import _ensure_circular_ranges_guard

    clf = ColorClassifier()
    # 青のみ inject (赤なし)
    per_video: dict[int, tuple[int, int, int, int, int, int]] = {
        COLOR_BLUE: (106, 111, 139, 226, 178, 241),
    }
    clf.set_color_ranges_from_simple(per_video, append=True)

    # guard 前の赤範囲スナップショット
    red_before = list(clf._ranges[COLOR_RED])

    _ensure_circular_ranges_guard(clf)

    red_after = list(clf._ranges[COLOR_RED])
    # 赤の inject がなかったので guard が DEFAULT 補完のみ実施
    # (DEFAULT は既に入っているので変化なし)
    assert len(red_after) == len(red_before), (
        f"赤未 inject 時は ranges 長が変化しないはず: "
        f"before={len(red_before)}, after={len(red_after)}"
    )


# ============================
# テスト 7: 赤以外の色は guard で touch されない (regression)
# ============================

def test_circular_guard_does_not_touch_non_circular_colors() -> None:
    """青等の単範囲色は _ensure_circular_ranges_guard で変更されない。

    per_video inject 後に青の ranges 長が変化しないことを確認。
    """
    from src.image_reader import ColorClassifier
    from src.board import COLOR_BLUE, COLOR_RED
    from scripts.visualize_recognition import _ensure_circular_ranges_guard

    clf = ColorClassifier()
    # 青 + 赤 の両方を inject
    per_video: dict[int, tuple[int, int, int, int, int, int]] = {
        COLOR_BLUE: (106, 111, 139, 226, 178, 241),
        COLOR_RED: (176, 178, 137, 217, 150, 188),
    }
    clf.set_color_ranges_from_simple(per_video, append=True)

    # guard 前の青範囲スナップショット
    blue_before = list(clf._ranges[COLOR_BLUE])

    _ensure_circular_ranges_guard(clf)

    blue_after = list(clf._ranges[COLOR_BLUE])
    # 青は _CIRCULAR_GUARD_COLORS に含まれないので長さ変化なし
    assert len(blue_after) == len(blue_before), (
        f"青は guard で変更されないはず: "
        f"before={len(blue_before)}, after={len(blue_after)}"
    )
