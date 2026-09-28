"""kill時にも残るライブキュー・開始境界の周期記録。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Event, Thread
import time
from typing import Any

SAMPLE_SEC = 1.


class RuntimeTelemetry:
    def __init__(self, bridge: Any, sink: Any, output: Path | None) -> None:
        self.bridge, self.sink, self.output = bridge, sink, output
        self.stop = Event()
        self.thread = Thread(target=self.run, daemon=True, name='live-telemetry')
        self.error: Exception | None = None

    def __enter__(self) -> RuntimeTelemetry:
        if self.output:
            self.thread.start()
        return self

    def snapshot(self) -> dict:
        return dict(at=time.perf_counter(), evaluation_pid=os.getpid(),
            progress=getattr(self.bridge, 'progress', {}),
            queue_depth=self.bridge.queue.qsize()+self.bridge.batch_remaining,
            game_starts=list(getattr(self.bridge, 'game_starts', [])),
            input_status=self.sink.publisher.input_calibration)

    def run(self) -> None:
        try:
            with (self.output/'runtime.jsonl').open('a', encoding='utf-8') as stream:
                while True:
                    stream.write(json.dumps(self.snapshot())+'\n')
                    stream.flush()
                    if self.stop.wait(SAMPLE_SEC):
                        return
        except Exception as error:
            self.error = error

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        if self.output:
            self.thread.join(SAMPLE_SEC*2)
            if not self.error:
                with (self.output/'runtime.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(self.snapshot())+'\n')
        if self.error:
            raise RuntimeError('ライブ計装に失敗しました') from self.error
