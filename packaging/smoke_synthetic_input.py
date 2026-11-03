"""実インストール先のコードへ動く図形入力を渡す。実カメラ・OBS設定は変更しない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import cv2

from src.phase_j.distribution_http import make_server
from src.phase_j.distribution_runtime import DistributionRuntime
from src.phase_j.live_device import DeviceConfig, DirectShowSource
from src.phase_j.live_publish import LivePublisher

TRIAL_PARENT = Path('D:/puyo_analyzer/b3_data/codex_tasks/dist_app_release')
DURATION_SECONDS = 60
FRAME_SHAPE = (1080, 1920, 3)
COLOR_LEVELS = 200
ASSETS = {name: 'synthetic-test' for name in ('app_build_id', 'recognition_model_hash',
    'recognition_config_hash', 'prediction_model_hash', 'calibration_hash')}


class SyntheticCapture:
    """テスト専用の変化する無地映像。盤面判定は本物の検査器を通す。"""

    def __init__(self, *args: Any) -> None:
        self.frame = 0

    def isOpened(self) -> bool:
        return True

    def set(self, key: int, value: float) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        self.frame += 1
        image = np.full(FRAME_SHAPE, self.frame % COLOR_LEVELS, dtype=np.uint8)
        cv2.rectangle(image, (200, 200), (500, 500), (80, 100, 220), -1)
        if self.frame > COLOR_LEVELS:
            from src.phase_j.distribution_input import marker_image
            marker = marker_image()
            image[700:700+marker.shape[0], 800:800+marker.shape[1]] = marker
        return True, image

    def release(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18785)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.parent != TRIAL_PARENT.resolve() or not root.name.startswith('installed_trial'):
        raise ValueError('試験専用インストール先だけで実行できます')
    runtime = DistributionRuntime(root)
    server = make_server(runtime, args.port)
    publisher = LivePublisher(ASSETS, '127.0.0.1', args.port + 1)
    runtime.inner_port = args.port + 1
    publisher.start()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    statuses: list[str] = []
    def status(phase: str) -> None:
        if not statuses or statuses[-1] != phase:
            statuses.append(phase)
            publisher.input_pending(dict(phase=phase, progress=0, at=time.perf_counter()))
    source = DirectShowSource(DeviceConfig('Synthetic test only', 0), DURATION_SECONDS,
        lambda: status('verifying'), capture_factory=SyntheticCapture, on_status=status)
    print(f'図形入力テストのみ: http://127.0.0.1:{args.port}/', flush=True)
    try:
        for frame in source:
            raise AssertionError('無地のテスト映像をゲーム盤面と判定しました')
    finally:
        runtime.stop()
        server.shutdown()
        server.server_close()
        publisher.close()
    (TRIAL_PARENT / 'synthetic-input-smoke.json').write_text(json.dumps(
        dict(input='synthetic changing shapes; no physical camera', statuses=statuses,
             code=str(Path(__import__('src.phase_j.live_device', fromlist=['__file__']).__file__))),
        ensure_ascii=False, indent=2), encoding='utf-8')
    assert 'no_puyo_screen' in statuses
    assert 'overlay_feedback' in statuses


if __name__ == '__main__':
    main()
