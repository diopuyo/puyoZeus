"""配信本体の Rust HSV が実際に有効か、認識子プロセスで記録する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

RECORD_PATH_ENV = 'PUYO_LIVE_TRIAL_FLAGS_PATH'


def configure_record_path(output: Path | None) -> None:
    """起動ごとに記録先を配送し、前回起動の環境変数を残さない。"""
    from .live_speed import TRIAL_FLAGS_FILE
    os.environ.pop(RECORD_PATH_ENV, None)
    if isinstance(output, Path) and output.is_dir():
        os.environ[RECORD_PATH_ENV] = str((output/TRIAL_FLAGS_FILE).resolve())


def runtime_status(pipe: Any) -> dict[str, Any]:
    """要求値でなく本体の分類器を調べ、無効時には理由を返す。"""
    from .live_speed import NATIVE_HSV_MAIN_KEY
    if os.environ.get('PUYO_'+NATIVE_HSV_MAIN_KEY.upper()) != '1':
        return dict(active=False, reason='disabled')
    hsv = pipe._reader._get_hsv_classifier()
    if hsv is None:
        return dict(active=False, reason='hsv_classifier_unavailable')
    if getattr(hsv, '_vote_mode', False):
        return dict(active=False, reason='vote_mode_unsupported')
    if getattr(hsv, '_enable_native_hsv', False):
        return dict(active=True, reason='enabled')
    try:
        import puyo_core
        reason = ('activation_failed' if hasattr(puyo_core, 'classify_cells_hsv')
                  else 'classify_cells_hsv_unavailable')
    except (ImportError, OSError) as error:
        reason = f'native_extension_unavailable: {type(error).__name__}: {error}'
    return dict(active=False, reason=reason)


def record_runtime(pipe: Any) -> dict[str, Any]:
    """親が残した要求値を保持し、子で確認した結果を追記する。"""
    from .live_speed import NATIVE_HSV_MAIN_KEY
    status = runtime_status(pipe)
    print('LIVE_NATIVE_HSV_MAIN '+json.dumps(status, ensure_ascii=False), flush=True)
    destination = os.environ.get(RECORD_PATH_ENV)
    if destination:
        path = Path(destination)
        record = json.loads(path.read_text(encoding='utf-8'))
        record.setdefault('runtime', {})[NATIVE_HSV_MAIN_KEY] = status
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    return status
