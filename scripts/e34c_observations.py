"""旧収集の評価動作を変えず、E16/E26/E31と同じ原映像観測を記録する。"""
from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path
from typing import Any

import cv2

from scripts.enrich_e31_snapshots import capture_window, fire_keys
from src.exchange_event_record import ExchangeEventRecorder
from src.exchange_event_terminal import ObservedDeathDetector
from src.midchain_board_reader import MidchainBoardReader
from src.prefire_snapshot_reader import PrefireSnapshotReader

FRAME_SIZE = (1920, 1080)


class Observations:
    """評価器へは戻さず、記録直前の複製へ画像由来の補助列を付ける。"""

    def __init__(self, video: Path, dest: Path, source: str) -> None:
        self.cap, self.window_cap = cv2.VideoCapture(str(video)), cv2.VideoCapture(str(video))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        assert self.fps > 0
        self.dest, self.required = dest, fire_keys(source)
        self.death, self.midchain = ObservedDeathDetector(), MidchainBoardReader()
        self.reader = PrefireSnapshotReader()
        self.windows: dict[str, dict] = {}
        self.game, self.start = None, 0.

    def frame(self, stamp: float) -> Any:
        """E16/E26と同じround(t×原fps)の原画像を取る。"""
        index = round(stamp*self.fps)
        position = round(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if position > index or position == 0 and index > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        while round(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) < index:
            assert self.cap.grab()
        success, frame = self.cap.read()
        assert success
        return frame

    def observed(self, result: Any, stamp: float, game: int) -> Any:
        """旧補完と同じリサイズ・読取器・画像窓を適用する。"""
        if game != self.game:
            self.game, self.start = game, stamp
        frame = self.frame(stamp)
        native = (frame.shape[1], frame.shape[0]) == FRAME_SIZE
        terminal = frame if native else cv2.resize(frame, FRAME_SIZE)
        midframe = frame if native else cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        boards = self.midchain.read(midframe, (result.p1, result.p2))
        sides = []
        for idx, (side, board) in enumerate(zip((result.p1, result.p2), boards)):
            window, event = None, side.chain_event
            key = (game, idx, event.trigger_sec) if event is not None else None
            if event is not None:
                name = f'{game}:{idx}:{event.trigger_sec}'
                if name not in self.windows:
                    self.windows[name] = capture_window(self.reader, self.window_cap, idx,
                        event.trigger_sec, self.fps, self.start)
                if key in self.required:
                    window = {k: v for k, v in self.windows[name].items() if k != 'raw_frames'}
            sides.append(replace(side, midchain_board=board, prefire_snapshot=window))
        return replace(result, p1=sides[0], p2=sides[1],
                       confirmed_dead_sides=self.death.update(terminal), terminal_evidence_available=True)

    def install(self) -> None:
        """新観測は原票だけに追加し、旧評価器の呼出条件を保つ。"""
        original = ExchangeEventRecorder.update
        def update(recorder: Any, result: Any, *args: Any) -> None:
            original(recorder, self.observed(result, args[2], args[3]), *args)
        ExchangeEventRecorder.update = update

    def close(self) -> None:
        """D1残差の根拠となる画像窓を保存する。"""
        self.cap.release()
        self.window_cap.release()
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.dest, 'wt', encoding='utf-8') as stream:
            json.dump(self.windows, stream, ensure_ascii=False, separators=(',', ':'))
