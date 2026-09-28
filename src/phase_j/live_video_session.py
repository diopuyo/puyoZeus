"""動画も実入力と同じ入力確認・無補正ウォームアップを経由する。"""
from __future__ import annotations

from typing import Any, Iterator
from pathlib import Path
import hashlib

from .live_device import PuyoScreenVerifier, VERIFY_PERIOD_SEC
from .live_device_session import DeviceSession
from .live_source import CapturedFrame

FINGERPRINT_STRIDE = 8


class VerifiedVideo:
    def __init__(self, source: Any, session: DeviceSession, verifier: Any = None,
                 continuous: bool = False) -> None:
        self.source, self.session = source, session
        self.verifier = verifier or PuyoScreenVerifier()
        self.gated_frames = 0
        self.continuous = continuous
        self.previous_size: tuple | None = None
        self.digest: bytes | None = None
        self.changed_at = 0.

    def fault_gate(self, frame: CapturedFrame) -> bool:
        """故障ラベルは参照せず、入力サイズ・画素・経過時刻で再確認する。"""
        size_changed = self.previous_size is not None and frame.source_size != self.previous_size
        self.previous_size = frame.source_size
        digest = hashlib.blake2b(frame.image[::FINGERPRINT_STRIDE, ::FINGERPRINT_STRIDE].tobytes()).digest()
        if digest != self.digest:
            self.digest, self.changed_at = digest, frame.media_sec
        repeated = frame.media_sec-self.changed_at >= VERIFY_PERIOD_SEC
        valid = self.verifier(frame.image)
        if size_changed or repeated or not valid:
            self.session.hold('no_puyo_screen' if not valid else 'verifying')
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
