"""同梱Pythonの終了コード、関連単体試験、補正全画像の一致を保存する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'logs/live_parallel'
TESTS = ['test_live_color_buffer.py', 'test_live_speed.py', 'test_launcher.py',
    'test_launcher_env.py', 'test_launcher_manifest.py', 'test_cnn_parallel_runtime.py',
    'test_cnn_onnx.py', 'phase_j/test_live_device.py', 'phase_j/test_dshow_color.py']


def run(name: str, command: list[str]) -> None:
    from live_parallel_probe import guard
    guard()
    with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log,
            creationflags=0x4000, env=dict(os.environ, PYTHONUTF8='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'), check=False)
    (OUT/f'{name}.exit.json').write_text(json.dumps(dict(returncode=completed.returncode)), encoding='utf-8')
    assert completed.returncode == 0, name


def pixels() -> None:
    from live_parallel_probe import FRAMES, guard
    import cv2
    from src.phase_j.live_color_buffer import ColorBuffer
    from src.phase_j.dshow_color import correct_601_as_709
    buffer = ColorBuffer()
    expected, actual = hashlib.sha256(), hashlib.sha256()
    count = 0
    for path in sorted(Path(FRAMES).glob('frame_*.png'))[:210]:
        guard()
        frame = cv2.imread(str(path))
        before = correct_601_as_709(frame).tobytes()
        after = buffer.correct(frame).tobytes()
        assert before == after, path
        expected.update(before)
        actual.update(after)
        count += 1
    assert count == 210
    (OUT/'color_pixels.json').write_text(json.dumps(dict(frames=count, bytes_per_frame=frame.nbytes,
        before_sha256=expected.hexdigest(), after_sha256=actual.hexdigest(),
        reused=buffer.reused, allocated=buffer.allocated), indent=2), encoding='utf-8')


def main() -> None:
    sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    for mode in ('off', 'on'):
        run('verified_smoke_'+mode, [str(OUT/f'soak_{mode}'/'PuyoLive/python/python.exe'),
            '-B', str(ROOT/'scripts/smoke_live_parallel.py'), mode, 'verified_smoke'])
    command = ("import sys,os; sys.path[:0]="+repr([str(ROOT), _public_dev_text('<work>/assets/devsite')])+
        _public_dev_text("; os.chdir('<work>/assets/baseline_b/app'); import pytest; raise SystemExit(pytest.main(")+repr([str(ROOT/'tests'/name) for name in TESTS]+
            ['-q', '--junitxml='+str(OUT/'tests.xml')])+"))")
    run('tests_final', [sys.executable, '-c', command])
    pixels()


if __name__ == '__main__':
    main()
