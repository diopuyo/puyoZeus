"""配布入口の既定を補い、利用者・開発試験の明示値を保つ。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.production_config import (
    RELEASE_LIVE_DEFAULTS, RELEASE_LIVE_MODEL_PATHS, RELEASE_LIVE_ENV_DEFAULTS,
    RELEASE_OBS_DEVICE_NAME, RELEASE_OBS_COLOR_CORRECTION,
)


def release_environment(environ: dict[str, str]) -> dict[str, str]:
    """配布子プロセスの環境を作る。明示値（0や空文字も含む）は保持する。"""
    return {**RELEASE_LIVE_ENV_DEFAULTS, **environ}


def release_user_config(data: dict[str, Any], app_root: Path) -> dict[str, Any]:
    """資産はアプリ内、明示した相対パスは既存launcherの設定場所基準とする。"""
    models = {key: str((app_root / relative).resolve())
              for key, relative in RELEASE_LIVE_MODEL_PATHS.items()}
    color = ({'dshow_color_correction': RELEASE_OBS_COLOR_CORRECTION}
             if data.get('device_name', RELEASE_OBS_DEVICE_NAME) == RELEASE_OBS_DEVICE_NAME
             else {})
    return {**RELEASE_LIVE_DEFAULTS, **models, **color, **data}


def distribution_user_config(settings: dict[str, Any], root: Path,
                             port: int) -> dict[str, Any]:
    """機器・内部ポート・出力先・表示だけを常駐設定から重ねる。"""
    return release_user_config(dict(
        source='dshow', device_name=settings['device_name'],
        device_index=settings['device_index'], port=port,
        output_dir=str(root / 'output'), overlay=dict(settings['overlay']),
    ), root / 'app')
