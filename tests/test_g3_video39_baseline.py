"""video39の実CLI設定と、FPS注入前の拒否/復元を検査する。"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any

import pytest
from scripts import g3_video39_baseline as V
from tests.test_g3_admission import collector


def test_arguments_only_declared_differences(collector: Any) -> None:
    c = collector
    plan = V.G.read(V.REPAIR / 'VIDEO39_BASELINE_PLAN.json')
    old = V.G.read(Path(plan['reference_prepare']))
    before = c.collect_lean
    with V.loaded(V.BASE, V.BASE_SHA) as base:
        expected = base.collection_arguments(c, {'collection_tokens': old['original_collection_tokens']})
    actual, receipt = V.arguments(c, plan['tokens'], Path(plan['video_path']))
    expected.update(start_sec=0.0, max_sec=V.END / V.FPS, precise_seek=False,
                    sample_interval_frames=1, score_region_calibration_path=V.CALIBRATION)
    assert actual == expected and c.collect_lean is before
    assert Path(receipt['video']).resolve() == Path(plan['video_path']).resolve()
    assert int(actual['max_sec'] * V.FPS) == V.END


@pytest.mark.parametrize('fps,valid', [(30.00000953592166, True), (30.0, False), (29.97, False)])
def test_native_fps_before_override(fps: float, valid: bool) -> None:
    source = V.source_contract()
    class Capture:
        closed = False
        def isOpened(self) -> bool:
            return not self.closed
        def get(self, key: int) -> float:
            return {1: fps, 2: source['frame_count'], 3: source['width'], 4: source['height']}[key]
        def release(self) -> None:
            self.closed = True
    native = Capture()
    factory = lambda *args, **kwargs: native
    cv = N(VideoCapture=factory, CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2,
           CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4)
    evidence = V.M.document(V.CLOCK_EVIDENCE, V.CLOCK_SHA)
    cv.__version__ = evidence['opencv']['version']
    cv.getBuildInformation = lambda: evidence['opencv']['build']
    c, state = N(cv2=cv), dict(decoded_frames=0)
    original_source = V.G.SOURCE
    with ExitStack() as stack:
        V.bind(stack)
        with V.checked_capture(c, source, state):
            if valid:
                cap = c.cv2.VideoCapture(V.G.SOURCE)
                assert cap.get(cv.CAP_PROP_FPS) == 30
            else:
                with pytest.raises(ValueError, match='baseline_actual_container'):
                    c.cv2.VideoCapture(V.G.SOURCE)
    assert native.closed and cv.VideoCapture is factory and V.G.SOURCE is original_source
    assert state['actual_container']['fps'] == fps


def test_changed_build_rejected_before_capture() -> None:
    """別backendへ無条件に平均fpsの許容を持ち越さない。"""
    cv = N(__version__='different', getBuildInformation=lambda: 'different')
    with pytest.raises(ValueError, match='native_clock_build'):
        V.native_clock(cv, V.source_contract(), {})
