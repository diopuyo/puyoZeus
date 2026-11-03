"""盤面読取の一括経路 (live_board_vectorized) が実際に有効か、認識子プロセスで記録する。

要求値 (live_trial_flags.json の flags) だけでは、子プロセスで本当に一括経路が使われたかが
分からない (fail-silent の防止)。ネイティブ HSV の記録 (live_native_hsv.py) と同じ形で、
認識器を調べた実効値を runtime へ追記する。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .live_native_hsv import RECORD_PATH_ENV

# 一括経路のうち、ネイティブ HSV が無いと働かない部分があることを記録に残す。
REASON_ENABLED = 'enabled'
REASON_ENABLED_WITHOUT_NATIVE_HSV = 'enabled_without_native_hsv'


def runtime_status(pipe: Any) -> dict[str, Any]:
    """要求値でなく認識器の状態を調べる。無効時は理由を返す。"""
    from .live_speed import BOARD_VECTORIZED_KEY
    if os.environ.get('PUYO_'+BOARD_VECTORIZED_KEY.upper()) != '1':
        return dict(active=False, reason='disabled')
    reader = getattr(pipe, '_reader', None)
    if reader is None or not getattr(reader, '_live_board_vectorized', False):
        return dict(active=False, reason='activation_failed')
    hsv = reader._get_hsv_classifier() if hasattr(reader, '_get_hsv_classifier') else None
    native = bool(getattr(hsv, '_enable_native_hsv', False))
    return dict(active=True, native_hsv=native,
                reason=REASON_ENABLED if native else REASON_ENABLED_WITHOUT_NATIVE_HSV)


def record_runtime(pipe: Any) -> dict[str, Any]:
    """親が残した要求値を保持し、子で確認した結果を追記する。"""
    from .live_speed import BOARD_VECTORIZED_KEY
    status = runtime_status(pipe)
    print('LIVE_BOARD_VECTORIZED '+json.dumps(status, ensure_ascii=False), flush=True)
    destination = os.environ.get(RECORD_PATH_ENV)
    if destination:
        path = Path(destination)
        record = json.loads(path.read_text(encoding='utf-8'))
        record.setdefault('runtime', {})[BOARD_VECTORIZED_KEY] = status
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    return status
