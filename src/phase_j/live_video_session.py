"""動画も実入力と同じ入力確認・無補正ウォームアップを経由する。"""
from __future__ import annotations

from typing import Any, Iterator
from pathlib import Path

from .live_device import PuyoScreenVerifier, VERIFY_PERIOD_SEC
from .live_device_session import DeviceSession
from .live_source import CapturedFrame


class VerifiedVideo:
    def __init__(self, source: Any, session: DeviceSession, verifier: Any = None) -> None:
        self.source, self.session = source, session
        self.verifier = verifier or PuyoScreenVerifier()
        self.gated_frames = 0

    @property
    def dropped(self) -> int:
        return self.source.dropped

    @property
    def dropped_times(self) -> list[float]:
        return self.source.dropped_times

    def __iter__(self) -> Iterator[CapturedFrame]:
        verified, last_check = False, float('-inf')
        for frame in self.source:
            # 固定ファイルは入口だけ確認する。対戦演出で履歴を切らず、試合内外は認識器へ渡す。
            if not verified and frame.media_sec-last_check >= VERIFY_PERIOD_SEC:
                verified, last_check = self.verifier(frame.image), frame.media_sec
            if not verified:
                self.session.hold('no_puyo_screen')
                self.gated_frames += 1
                continue
            yield frame


def video_session(pipe: Any, source: Any, queue: Any, config: Any) -> tuple:
    session = DeviceSession(pipe, Path('.'), 0, queue)
    session.switch(config)
    session.source = VerifiedVideo(source, session)
    return session, session.source
