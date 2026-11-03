"""read_board の Rust 経路 (live_native_read_board) が実際に有効か、認識子プロセスで記録する。

要求値 (live_trial_flags.json の flags) だけでは、子プロセスで拡張が読め、前提の一括経路・
ネイティブ HSV が有効だったかが分からない (fail-silent の防止)。live_board_runtime.py と同じ形で、
認識器を調べた実効値を runtime へ追記する。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .live_native_hsv import RECORD_PATH_ENV

REASON_ENABLED = 'enabled'
REASON_DISABLED = 'disabled'
REASON_ACTIVATION_FAILED = 'activation_failed'
REASON_NEEDS_VECTORIZED = 'requires_live_board_vectorized'
REASON_NEEDS_NATIVE_HSV = 'requires_native_hsv'


def runtime_status(pipe: Any) -> dict[str, Any]:
    """要求値でなく認識器の状態を調べる。働かない場合は理由を返す。"""
    from .live_speed import NATIVE_READ_BOARD_KEY
    if os.environ.get('PUYO_'+NATIVE_READ_BOARD_KEY.upper()) != '1':
        return dict(active=False, reason=REASON_DISABLED)
    reader = getattr(pipe, '_reader', None)
    if reader is None or not getattr(reader, '_live_native_read_board', False):
        return dict(active=False, reason=REASON_ACTIVATION_FAILED)
    if not getattr(reader, '_live_board_vectorized', False):
        return dict(active=False, reason=REASON_NEEDS_VECTORIZED)
    hsv = reader._get_hsv_classifier() if hasattr(reader, '_get_hsv_classifier') else None
    if not bool(getattr(hsv, '_enable_native_hsv', False)):
        return dict(active=False, reason=REASON_NEEDS_NATIVE_HSV)
    return dict(active=True, reason=REASON_ENABLED)


def record_runtime(pipe: Any) -> dict[str, Any]:
    """親が残した要求値を保持し、子で確認した結果を追記する。"""
    from .live_speed import NATIVE_READ_BOARD_KEY
    status = runtime_status(pipe)
    print('LIVE_NATIVE_READ_BOARD '+json.dumps(status, ensure_ascii=False), flush=True)
    destination = os.environ.get(RECORD_PATH_ENV)
    if destination:
        path = Path(destination)
        record = json.loads(path.read_text(encoding='utf-8'))
        record.setdefault('runtime', {})[NATIVE_READ_BOARD_KEY] = status
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    return status
