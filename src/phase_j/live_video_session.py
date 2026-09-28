"""動画も実入力と同じ入力確認・無補正ウォームアップを経由する。"""
from __future__ import annotations

from typing import Any, Iterator
from pathlib import Path

from .live_device import PuyoScreenVerifier, VERIFY_PERIOD_SEC
from .live_device_session import DeviceSession
from .live_source import CapturedFrame
from .live_input_guard import FrameContinuity


class VerifiedVideo:
    def __init__(self, source: Any, session: DeviceSession, verifier: Any = None,
                 continuous: bool = False) -> None:
        self.source, self.session = source, session
        self.verifier = verifier or PuyoScreenVerifier()
        self.gated_frames = 0
        self.continuous = continuous
        self.continuity = FrameContinuity(VERIFY_PERIOD_SEC)

    def transport_ready(self, frame: CapturedFrame) -> bool:
        """形式契約の逸脱と映像停止を、盤面内容より優先して遮断する。"""
        size = getattr(frame, 'source_size', None) or (frame.image.shape[1], frame.image.shape[0])
        if not self.continuity.ready(frame.image, size, frame.media_sec):
            self.session.hold('verifying')
            self.gated_frames += 1
            return False
        return True

    def fault_gate(self, frame: CapturedFrame) -> bool:
        """故障ラベルは参照せず、入力サイズ・画素・経過時刻で再確認する。"""
        if not self.transport_ready(frame):
            return False
        if not self.verifier(frame.image):
            self.session.hold('no_puyo_screen')
            self.gated_frames += 1
            return False
        return True

    @property
    def dropped(self) -> int:
        return self.source.dropped

    @property
    def dropped_times(self) -> list[float]:
        return self.source.dropped_times

    def __iter__(self) -> Iterator[CapturedFrame]:
        verified, last_check = False, float('-inf')
        for frame in self.source:
            if self.continuous:
                if self.fault_gate(frame):
                    yield frame
                continue
            if not self.transport_ready(frame):
                verified, last_check = False, float('-inf')
                continue
            # 固定ファイルは入口だけ確認する。対戦演出で履歴を切らず、試合内外は認識器へ渡す。
            if not verified and frame.media_sec-last_check >= VERIFY_PERIOD_SEC:
                verified, last_check = self.verifier(frame.image), frame.media_sec
            if not verified:
                self.session.hold('no_puyo_screen')
                self.gated_frames += 1
                continue
            yield frame


def video_session(pipe: Any, source: Any, queue: Any, config: Any,
                  continuous: bool = False) -> tuple:
    session = DeviceSession(pipe, Path('.'), 0, queue)
    session.switch(config)
    source.on_hold = session.hold
    session.source = VerifiedVideo(source, session, continuous=continuous)
    return session, session.source
