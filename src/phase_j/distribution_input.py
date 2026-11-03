"""入力探索を機器ごとの子プロセスへ隔離し、停止したドライバを待ち続けない。"""
from __future__ import annotations
from src.phase_j.live_color_roi import full_image

import json
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
FEEDBACK_HELP = 'OBS の仮想カメラの出力を「ソース」にして、ゲームの映像だけを選んでください。'
MESSAGES = {
    'ready': 'ゲーム画面を確認しました / Game input verified',
    'no_input': '入力が見つかりません / No input found',
    'unavailable': 'OBS の仮想カメラが開始されていないか、入力を読み取れません / Start OBS Virtual Camera or check the input',
    'wrong_size': '入力を1920×1080に設定してください / Set input to 1920×1080',
    'not_game': 'ゲーム画面ではありません。試合を映して再検索してください / Not a game screen; show a match and scan again',
    'feedback': FEEDBACK_HELP + ' / Set Virtual Camera output to Source and select only the game capture.',
    'timeout': '入力の応答がありません / Input timed out',
}


def marker_image() -> np.ndarray:
    """全テーマ共通の小さな色・形の目印。CSS側と同じ並びを使う。"""
    row = np.array(MARKER_COLORS, dtype=np.uint8)[:, ::-1]
    pattern = np.stack((row, row[::-1], np.roll(row, 1, axis=0)))
    return cv2.resize(pattern, (MARKER_CELL * 4, MARKER_CELL * 3), interpolation=cv2.INTER_NEAREST)


def has_feedback(frame: np.ndarray) -> bool:
    """移動・縮小された表示でも目印を照合し、単色領域では反応しない。"""
    from .live_verify_reasons import feedback_score
    color = np.array(MARKER_COLORS[0][::-1])
    mask = cv2.inRange(full_image(frame), np.maximum(0, color - 25), np.minimum(255, color + 25))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [cv2.boundingRect(contour) for contour in contours if 2 <= cv2.contourArea(contour) <= 120]
    for scale in (1.0, .8, .7, .5):
        template = cv2.resize(marker_image(), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for x, y, width, height in candidates:
            margin = MARKER_CELL * 5
            crop = frame[max(0, y-margin):y+height+margin, max(0, x-margin):x+width+margin]
            if min(crop.shape[:2]) < max(template.shape[:2]):
                continue
            correlation = cv2.minMaxLoc(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED))[1]
            feedback_score(correlation)
            if correlation >= MARKER_TOLERANCE:
                return True
    return False


def inspect_frame(frame: np.ndarray, verifier: Callable | None = None) -> str:
    from .live_device import PuyoScreenVerifier
    if has_feedback(frame):
        return 'feedback'
    if frame.shape[:2] != (1080, 1920):
        return 'wrong_size'
    return 'ready' if (verifier or PuyoScreenVerifier())(frame) else 'not_game'


def probe(index: int) -> dict[str, Any]:
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
            if result['reason'] in ('ready', 'feedback'):
                break
        return result
    finally:
        capture.release()


def isolated_probe(index: int) -> dict[str, Any]:
    executable = str(Path(sys.executable).with_name('python.exe')) if os.name == 'nt' else sys.executable
    try:
        run = subprocess.run([executable, '-m', __name__, str(index)], capture_output=True,
            timeout=PROBE_TIMEOUT, encoding='utf-8', errors='replace', cwd=Path(__file__).resolve().parents[2],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
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
    return dict(devices=results, selected=selected, reason=reason, message=MESSAGES[reason],
                input_resolution=observed.get('input_resolution'), fps=observed.get('fps'))


if __name__ == '__main__':
    print(json.dumps(probe(int(sys.argv[1]))))
