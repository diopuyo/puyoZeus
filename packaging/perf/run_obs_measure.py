"""OBS 仮想カメラ入力で配布版を一定時間走らせる (性能計測用)。配布物の python で `cwd = 配布物の app/` から実行する。

利用者と同じ経路 (launcher.prepare → 配布既定のフラグ・ONNX 選択・較正保存先) を通し、
pipeline 設定に duration_sec だけを足して終了させる (利用者用の設定には無い計測専用項目)。
使い方: python run_obs_measure.py <出力 dir> <機器 index> <秒数>
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

APP_ROOT = Path.cwd()
sys.path.insert(0, str(APP_ROOT))
from src.phase_j import launcher  # noqa: E402

BYPASS_ENV = 'PERF_BYPASS_VERIFIER'  # 診断専用: 入力の「ぷよ画面確認」を常に合格にする (spawn 子にも top-level で適用される)
if os.environ.get(BYPASS_ENV) == '1':
    from src.phase_j import live_device
    live_device.PuyoScreenVerifier.__call__ = lambda self, image: True

PORT = 8791  # 既存の利用者設定 (8765) と衝突させない
COLOR_CORRECTION = '601to709'


def main() -> None:
    out, index, seconds = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    os.environ[launcher.CALIBRATION_DIR_ENV] = str(out / 'calibration')  # 配布物 (MANIFEST 対象) を汚さない
    user_config = out / 'puyo_live.json'
    user_config.write_text(json.dumps(dict(source='dshow', device_name='OBS Virtual Camera', device_index=index,
                                           port=PORT, output_dir=str(out / 'output'),
                                           dshow_color_correction=COLOR_CORRECTION), ensure_ascii=False), encoding='utf-8')
    options = launcher.parse_launcher_args(['--config', str(user_config), '--require-manifest'])
    user, argv = launcher.prepare(options)
    pipeline_config = Path(argv[argv.index('--config')+1])
    data = json.loads(pipeline_config.read_text(encoding='utf-8'))
    data['duration_sec'] = seconds
    data.update(recognition_audit=True, runtime_audit=True)  # 認識 1 frame の時間と SSE 受信時刻の記録 (計測専用)
    pipeline_config.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[measure] pipeline config:', json.dumps(data, ensure_ascii=False), flush=True)
    print('[measure] CNN backend env:', os.environ.get('PUYO_CNN_BACKEND'), flush=True)
    os.chdir(APP_ROOT)
    sys.argv = [launcher.PIPELINE_MODULE, *argv]
    from scripts.run_live_pipeline_20260928 import main as pipeline_main
    pipeline_main()


if __name__ == '__main__':
    main()
