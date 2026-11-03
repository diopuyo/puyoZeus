"""配布用設定。保存先は各展開先に閉じ、既存ランチャ設定と分離する。"""
from __future__ import annotations
from src.phase_j.live_color_roi import full_image

import json
import locale
import os
from pathlib import Path
import socket
from typing import Any
from urllib.parse import urlsplit

from .distribution_logs import RELAY_PLACEHOLDER
from .overlay_config import validate_overlay
from src.production_config import DISTRIBUTION_APP_WINDOW

DEFAULT_PORT = 8765
MAX_PORT_ATTEMPTS = 100
PREVIEW_WIDTH = 640
DEFAULTS: dict[str, Any] = dict(completed=False, language='auto', device_index=0,
    device_name='OBS Virtual Camera', port=DEFAULT_PORT, check_updates=False,
    relay_url=RELAY_PLACEHOLDER, releases_url='https://puyolive.example.invalid/releases.json',
    auto_input=True, obs_enabled=False, obs_port=4455, obs_password='', desktop_shortcut=False, start_menu_shortcut=True, autostart=False, overlay={},
    distribution_app_window=DISTRIBUTION_APP_WINDOW)


def os_language() -> str:
    """ブラウザ言語よりOSの既定UI言語を優先する。"""
    language = locale.getlocale()[0] or 'en'
    if os.name == 'nt':
        import ctypes
        language = locale.windows_locale.get(ctypes.windll.kernel32.GetUserDefaultUILanguage(), language)
    return 'ja' if language.lower().startswith('ja') else 'en'


def load_settings(root: Path) -> dict[str, Any]:
    path = root / 'distribution.json'
    return validate_settings(json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {})


def validate_settings(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) - DEFAULTS.keys():
        raise ValueError('不明な設定 / Unknown settings')
    value = {**DEFAULTS, **data}
    for key in ('auto_input', 'obs_enabled', 'completed', 'check_updates', 'desktop_shortcut', 'start_menu_shortcut', 'autostart', 'distribution_app_window'):
        if type(value[key]) is not bool:
            raise ValueError(f'{key}: true / false')
    for key, low, high in (('obs_port', 1024, 65535), ('port', 1024, 65535), ('device_index', 0, 64)):
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f'{key}: {low}–{high}')
    if value['language'] not in ('auto', 'ja', 'en'):
        raise ValueError('language: auto / ja / en')
    if not isinstance(value['device_name'], str) or not value['device_name'].strip():
        raise ValueError('入力名が必要 / Input name required')
    for key in ('relay_url', 'releases_url'):
        url = urlsplit(value[key])
        if url.scheme != 'https' or not url.hostname or url.username or url.password:
            raise ValueError(f'{key}: HTTPS URL required')
    if not isinstance(value['obs_password'], str):
        raise ValueError('OBS password must be text')
    value['overlay'] = validate_overlay(value['overlay'])
    return value


def save_settings(root: Path, settings: dict[str, Any]) -> None:
    value = validate_settings(settings)
    temporary = root / 'distribution.json.tmp'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(root / 'distribution.json')


def available_port(preferred: int) -> int:
    for port in range(preferred, min(65536, preferred + MAX_PORT_ATTEMPTS)):
        with socket.socket() as candidate:
            try:
                candidate.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise OSError('空きポートがありません / No free port')


def devices() -> list[dict[str, Any]]:
    """名前とDirectShow indexを列挙するだけで、カメラは開かない。"""
    import cv2
    from cv2_enumerate_cameras import enumerate_cameras
    return [dict(index=item.index, name=item.name) for item in enumerate_cameras(cv2.CAP_DSHOW)]


def camera_preview(index: int, diagnostics: dict[str, Any] | None = None) -> bytes:
    """利用者が選んだ入力だけを短時間開き、画像は保存しない。"""
    import cv2
    from .live_source import NATIVE_SIZE, RECOGNITION_HZ
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE_SIZE[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_SIZE[1])
        capture.set(cv2.CAP_PROP_FPS, RECOGNITION_HZ)
        ok, frame = capture.read()
        if not ok:
            raise ValueError('入力を読めません / Cannot read input')
        height, width = frame.shape[:2]
        if diagnostics is not None:
            fps = capture.get(cv2.CAP_PROP_FPS)
            diagnostics.update(input_resolution=[width, height], fps=fps if fps > 0 else None,
                               measurement_source='setup_input_test')
        image = cv2.resize(full_image(frame), (PREVIEW_WIDTH, max(1, round(height * PREVIEW_WIDTH / width))))
        ok, buffer = cv2.imencode('.jpg', image)
        if not ok:
            raise ValueError('画像を変換できません / Cannot encode frame')
        return buffer.tobytes()
    finally:
        capture.release()
