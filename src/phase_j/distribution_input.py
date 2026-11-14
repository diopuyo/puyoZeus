"""入力探索を機器ごとの子プロセスへ隔離し、停止したドライバを待ち続けない。"""
from __future__ import annotations
from src.phase_j.live_color_roi import full_image

import json
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import cv2
import numpy as np

PROBE_TIMEOUT = 10
PROBE_FRAMES = 12
MARKER_COLORS = ((225, 55, 197), (41, 220, 179), (248, 196, 51), (54, 79, 230))
MARKER_CELL = 6
MARKER_TOLERANCE = 0.92
MARKER_COLOR_MARGIN = 25
# 半分の縮尺では色差間引き後の単色部分が2×2画素（輪郭面積1）になる。
MARKER_MIN_AREA, MARKER_MAX_AREA = 1, 120
MARKER_SEARCH_WIDTH, MARKER_SEARCH_HEIGHT = MARKER_CELL * 4, MARKER_CELL * 3
FEEDBACK_HELP = '表示が映り込んでいます。OBS の仮想カメラの出力を「ソース」→ ゲームの映像にしてください。'
STOPPED_SIZE = (96, 54)
STOPPED_MEAN_ERROR = 8
STOPPED_PIXEL_ERROR = 24
STOPPED_MATCH_FRACTION = .95
STOPPED_DETAIL_ERROR = 12
STOPPED_LOGO_ROWS, STOPPED_LOGO_COLUMNS = slice(6, 32), slice(35, 61)
STOPPED_CAMERA_ROWS, STOPPED_CAMERA_COLUMNS = slice(38, 48), slice(43, 53)
MESSAGES = {
    'waiting': '試合が映るのを待っています / Waiting for a match to appear',
    'obs_stopped': 'OBS の仮想カメラが止まっています。OBS の「仮想カメラ開始」を押してください / OBS Virtual Camera has stopped. Click Start Virtual Camera in OBS.',
    'ready': 'ゲーム画面を確認しました / Game input verified',
    'no_input': '入力が見つかりません / No input found',
    'unavailable': 'OBS の仮想カメラが開始されていないか、入力を読み取れません / Start OBS Virtual Camera or check the input',
    'wrong_size': '入力を1920×1080に設定してください / Set input to 1920×1080',
    'not_game': 'ゲーム画面ではありません。試合を映して再検索してください / Not a game screen; show a match and scan again',
    'feedback': FEEDBACK_HELP + ' / The overlay appears in the input. Set OBS Virtual Camera output to Source and select the game video.',
    'timeout': '入力の応答がありません / Input timed out',
}


def marker_image() -> np.ndarray:
    """全テーマ共通の小さな色・形の目印。CSS側と同じ並びを使う。"""
    row = np.array(MARKER_COLORS, dtype=np.uint8)[:, ::-1]
    pattern = np.stack((row, row[::-1], np.roll(row, 1, axis=0)))
    return cv2.resize(pattern, (MARKER_CELL * 4, MARKER_CELL * 3), interpolation=cv2.INTER_NEAREST)


def has_feedback(frame: np.ndarray) -> bool:
    """移動・縮小された表示でも目印を照合し、単色領域では反応しない。"""
    if frame is None or not frame.size or frame.ndim != 3 or frame.shape[2] != 3:
        return False
    frame = full_image(frame)
    mask = _marker_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [cv2.boundingRect(contour) for contour in contours
                  if MARKER_MIN_AREA <= cv2.contourArea(contour) <= MARKER_MAX_AREA]
    for x, y, width, height in candidates:
        # 候補の色セルを含む目印の全体が入る、参照の最大幅・高さまでを探す。
        crop = frame[max(0, y-MARKER_SEARCH_HEIGHT):y+height+MARKER_SEARCH_HEIGHT,
                     max(0, x-MARKER_SEARCH_WIDTH):x+width+MARKER_SEARCH_WIDTH]
        if _matches_marker(crop):
            return True
    return False


def _matches_marker(crop: np.ndarray) -> bool:
    """理想参照との相関が低すぎる窓では、合格不可能な変換参照を省く。"""
    from .feedback_templates import reference_groups
    from .live_verify_reasons import feedback_score
    for original, variants in reference_groups(MARKER_TOLERANCE):
        if crop.shape[0] < original.shape[0] or crop.shape[1] < original.shape[1]:
            continue
        correlation = cv2.minMaxLoc(cv2.matchTemplate(crop, original, cv2.TM_CCOEFF_NORMED))[1]
        feedback_score(correlation)
        if correlation >= MARKER_TOLERANCE:
            return True
        for template, lower in variants:
            if correlation < lower:
                continue
            score = cv2.minMaxLoc(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED))[1]
            feedback_score(score)
            if score >= MARKER_TOLERANCE:
                return True
    return False


def _marker_mask(frame: np.ndarray) -> np.ndarray:
    """色の不一致で候補探索から落ちないよう、601復号後の色も探す。"""
    from .live_color_roi import simulate_dshow
    color = np.array(MARKER_COLORS[0][::-1], dtype=np.uint8).reshape(1, 1, 3)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for candidate in (color, simulate_dshow(color)):
        center = candidate.reshape(3).astype(int)
        mask |= cv2.inRange(frame, np.maximum(0, center - MARKER_COLOR_MARGIN),
                            np.minimum(255, center + MARKER_COLOR_MARGIN))
    return mask


@lru_cache(maxsize=1)
def stopped_reference() -> np.ndarray | None:
    """提供された停止画面を縮小した照合専用の参照画像。"""
    data = Path(__file__).with_name('obs_camera_stopped.png').read_bytes()
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def is_obs_stopped(frame: np.ndarray) -> bool:
    """背景だけでなく中央のOBSロゴと下部の停止記号も一致を要求する。"""
    if frame is None or not frame.size or frame.ndim != 3 or frame.shape[2] != 3:
        return False
    reference = stopped_reference()
    if reference is None:
        return False
    small = cv2.resize(full_image(frame), STOPPED_SIZE, interpolation=cv2.INTER_AREA)
    error = np.abs(small.astype(np.float32) - reference.astype(np.float32))
    # 参照は96×54。中央上部のロゴと下部の斜線入りカメラを別に照合する。
    logo = error[STOPPED_LOGO_ROWS, STOPPED_LOGO_COLUMNS]
    camera = error[STOPPED_CAMERA_ROWS, STOPPED_CAMERA_COLUMNS]
    return bool(error.mean() <= STOPPED_MEAN_ERROR
        and np.mean(np.max(error, axis=2) <= STOPPED_PIXEL_ERROR) >= STOPPED_MATCH_FRACTION
        and logo.mean() <= STOPPED_DETAIL_ERROR and camera.mean() <= STOPPED_DETAIL_ERROR)


def inspect_frame(frame: np.ndarray, verifier: Callable | None = None) -> str:
    from .live_device import PuyoScreenVerifier
    if is_obs_stopped(frame):
        return 'obs_stopped'
    if has_feedback(frame):
        return 'feedback'
    if frame.shape[:2] != (1080, 1920):
        return 'wrong_size'
    return 'ready' if (verifier or PuyoScreenVerifier())(frame) else 'not_game'


def probe(index: int) -> dict[str, Any]:
    import base64
    from .live_preview import thumbnail
    from .live_device import PuyoScreenVerifier
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    result: dict[str, Any] = dict(reason='unavailable')
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        capture.set(cv2.CAP_PROP_FPS, 30)
        verifier = PuyoScreenVerifier()
        for _ in range(PROBE_FRAMES):
            ok, frame = capture.read()
            if not ok:
                continue
            result = dict(reason=inspect_frame(frame, verifier), input_resolution=list(frame.shape[1::-1]),
                          fps=capture.get(cv2.CAP_PROP_FPS))
            result['preview'] = base64.b64encode(thumbnail(frame)).decode('ascii')
            if result['reason'] in ('ready', 'feedback'):
                break
        return result
    finally:
        capture.release()


def isolated_probe(index: int) -> dict[str, Any]:
    from .release_config import release_environment
    executable = str(Path(sys.executable).with_name('python.exe')) if os.name == 'nt' else sys.executable
    try:
        # 解析の子 process と同じく配布の旗を付ける (11/14: 0.6.6 は切り抜き画像を同梱しないので、
        # 画像なし方式 PUYO_IMAGE_FREE_ALL が無いと得点が読めず、ゲーム画面も「ゲーム画面ではない」になった)
        run = subprocess.run([executable, '-s', '-m', __name__, str(index)], capture_output=True,
            timeout=PROBE_TIMEOUT, encoding='utf-8', errors='replace', cwd=Path(__file__).resolve().parents[2],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            env=release_environment(dict(os.environ)))
        return json.loads(run.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
        return dict(reason='timeout')


def discover(items: list[dict], probe_fn: Callable = isolated_probe) -> dict[str, Any]:
    """全候補を試し、合格したOBSを優先する。失敗候補は選択しない。"""
    results = [{**item, **probe_fn(item['index'])} for item in items]
    ordered = sorted(results, key=lambda item: 'obs virtual camera' not in item['name'].lower())
    selected = next((item for item in ordered if item['reason'] == 'ready'), None)
    reason = 'ready' if selected else next((item['reason'] for item in ordered
        if item['reason'] == 'feedback'), ordered[0]['reason'] if ordered else 'no_input')
    observed = selected or (ordered[0] if ordered else {})
    message = MESSAGES[reason]
    name = observed.get('name', '')
    if reason == 'wrong_size' and 'obs virtual camera' not in name.lower():
        message = (f'この入力 ({name}) はゲームの映像ではないようです。OBS の仮想カメラ (OBS Virtual Camera) を選んでください'
                   f' / This input ({name}) does not appear to be game video. Select OBS Virtual Camera.')
    return dict(devices=results, selected=selected, reason=reason, message=message,
                input_resolution=observed.get('input_resolution'), fps=observed.get('fps'))


if __name__ == '__main__':
    print(json.dumps(probe(int(sys.argv[1]))))
