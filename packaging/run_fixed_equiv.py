"""固定BGR入力で既存ライブパイプラインを実行。spawn子でも入力差替を維持する。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

# 埋込Pythonの_pthで除外される検査ツールのディレクトリを明示する。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from equiv_frames import CachedCapture, META_NAME

PREFIX = 'PUYO_EQUIV_'
PACKAGES = ('numpy', 'torch', 'onnxruntime', 'scikit-learn', 'opencv-python-headless', 'opencv-python')


def install_capture() -> None:
    """本体・配布物は変更せず、この検査processとspawn子だけで差し替える。"""
    import cv2
    cache = Path(os.environ[PREFIX + 'CACHE'])
    output = Path(os.environ[PREFIX + 'OUTPUT'])
    source = json.loads((cache / META_NAME).read_text(encoding='utf-8'))['source']
    counter = 0

    def capture(video: Any, *args: Any, **kwargs: Any) -> CachedCapture:
        nonlocal counter
        if Path(str(video)).name != source:
            raise ValueError(f'比較対象外の入力要求: {video}')
        counter += 1
        return CachedCapture(cache, output / f'pixels-{os.getpid()}-{counter}.json')

    cv2.VideoCapture = capture


def run(args: argparse.Namespace) -> None:
    """配布の認識設定を使い、比較時だけ前回と同じ同期・全フレーム処理にする。"""
    app, cache, output = args.app.resolve(), args.cache.resolve(), args.output.resolve()
    code_root = getattr(args, 'code_root', None)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update({PREFIX + 'CACHE': str(cache), PREFIX + 'OUTPUT': str(output),
                       PREFIX + 'APP': str(app), 'CUDA_VISIBLE_DEVICES': '',
                       'PYTHONDONTWRITEBYTECODE': '1'})
    if code_root:
        os.environ[PREFIX + 'CODE_ROOT'] = str(code_root.resolve())
    if args.backend == 'onnx':
        os.environ['PUYO_CNN_BACKEND'] = 'onnx'
    else:
        os.environ.pop('PUYO_CNN_BACKEND', None)
    os.chdir(app)
    sys.path.insert(0, str(app))
    if code_root:
        sys.path.insert(0, str(code_root.resolve()))
    install_capture()
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    from scripts import run_live_pipeline_20260928 as live
    from scripts.run_e3_exchange_eval_20260926 import SEED
    from src import production_config
    sys.argv = ['live', '--video', str(args.video), '--start-sec', str(args.start),
                '--end-sec', str(args.end), '--warmup-sec', '1', '--output', str(output),
                '--no-split-evaluation', '--no-async-counter', '--port', str(args.port),
                '--score-zero-from-ocr', '--fast-telop', '--fast-terminal', '--recognition-audit']
    if getattr(args, 'b3', False):
        sys.argv.remove('--no-split-evaluation')
        sys.argv += ['--b3-judgment', '--b3-model', str(app / 'models/b3/model.joblib'),
                     '--b3-timing-table', str(app / 'models/b3/timing_table.json')]
    metadata = dict(backend=args.backend, packages=versions, argv=sys.argv,
                    seed=SEED, production_flags=production_config.exchange_event_flags(),
                    manifest_sha256=hashlib.sha256((app / 'MANIFEST.json').read_bytes()).hexdigest(),
                    environment={key: value for key, value in os.environ.items()
                                 if key.startswith('PUYO_')})
    (output / 'execution.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    live.main()
    (output / 'DONE').write_text('正常終了\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('app', 'cache', 'output', 'video'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--start', type=float, default=226.)
    parser.add_argument('--end', type=float, default=340.)
    parser.add_argument('--backend', choices=('torch', 'onnx'), default='torch')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--b3', action='store_true', help='配布対象のB3も固定画素から照合する')
    parser.add_argument('--code-root', type=Path, help='資産を共用して照合するソースツリー')
    run(parser.parse_args())


if __name__ == '__mp_main__':
    sys.path.insert(0, os.environ[PREFIX + 'APP'])
    if PREFIX + 'CODE_ROOT' in os.environ:
        sys.path.insert(0, os.environ[PREFIX + 'CODE_ROOT'])
    install_capture()
elif __name__ == '__main__':
    main()
