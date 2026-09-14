"""原whole driverの終了後に実状態を保存する。旧collectorの票は偽装しない。"""
from __future__ import annotations

import json
from typing import Any

FILENAME = 'WHOLE_M1_DRIVER_STATUS.json'


def save(driver: Any, original: Any, body: Any) -> None:
    bridge, session = driver.bridge, driver.session
    packet = dict(start_frame=driver.start_frame, last_frame=driver.last, closed=driver.closed,
        stopped=driver.stopped, installed=session is not None,
        consumer_forbidden=bridge is not None and bridge.consumer is original.forbidden_collect,
        session_restored=session is not None and session.restored,
        error=None if driver.error is None else repr(driver.error),
        body_error=None if body is None else repr(body), quality_gate_clear=False)
    with (driver.state['output'] / FILENAME).open('x', encoding='utf-8') as stream:
        json.dump(packet, stream, ensure_ascii=False, allow_nan=False)


def driver_class(original: Any) -> type:
    class Driver(original.Driver):
        def close(self, kind: Any, body: Any, trace: Any) -> bool:
            was_closed, failure = self.closed, None
            try:
                return super().close(kind, body, trace)
            except BaseException as error:
                failure = error
                raise
            finally:
                if not was_closed:
                    try: save(self, original, body)
                    except BaseException as error:
                        self.status_save_error = repr(error)
                        if body is None and failure is None: raise
    return Driver
