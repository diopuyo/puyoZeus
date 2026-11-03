"""保存された認識フレーム列を同じ動画から復号し、全出力を比較用に保存する。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from enum import Enum
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VIDEO = Path('C:/puyo1_tools/soak/media/video_d04.mp4')
BASE = Path('D:/puyo_analyzer/packaging/build_20261021b_stage/PuyoLive/app')


def canonical(value: Any) -> Any:
    import numpy as np
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, set):
        return sorted(canonical(item) for item in value)
    if hasattr(value, '__dict__'):
        return canonical(vars(value))
    return value


class Messages:
    def put(self, value: Any) -> None:
        pass


def prepare(args: argparse.Namespace) -> Any:
    sys.path.insert(0, str(ROOT if args.candidate else BASE))
    import src.phase_j
    src.phase_j.__path__.append(str(ROOT/'src/phase_j'))
    config = json.loads((args.source/'pipeline_config.json').read_text())
    for key, value in config.items():
        if isinstance(value, bool) or key in ('cpu_threads', 'evaluation_nice', 'b3_warning_workers'):
            os.environ['PUYO_'+key.upper()] = str(int(value))
    os.environ.update(PUYO_OBS_FAST_AUDIT='1', PUYO_OBS_EXACT_SEED='1',
        PUYO_LIVE_STAGE_TIMING='1', PUYO_CNN_BACKEND='onnx',
        PUYO_CALIBRATION_DIR=str(args.out/'calibration'))
    os.chdir(BASE)
    import numpy as np
    import torch
    from scripts.run_e3_exchange_eval_20260926 import SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    from src.phase_j.live_cpu import apply_runtime
    apply_runtime('recognition')
    from src.recognition_pipeline import RecognitionPipeline
    pipe = RecognitionPipeline.load_default(**json.loads((args.source/'recognition_config.json').read_text()))
    if args.candidate:
        from src.phase_j.live_obs_prepare import prepare_recognition
        prepare_recognition(pipe)
    if args.candidate:
        from src.phase_j.live_obs_recognition_timing import install, install_import_timing
        install(pipe)
        install_import_timing()
    from src.phase_j.live_device_session import DeviceSession
    from src.phase_j.live_config import input_identity
    session = DeviceSession(pipe, Path('.'), 0, Messages())
    session.switch(input_identity(str(args.source/'pipeline_config.json'), str(VIDEO)))
    from src.phase_j.live_b3 import prepare_capture
    prepare_capture(session)
    return session


def replay(args: argparse.Namespace, session: Any, indices: list[dict]) -> int:
    import cv2
    from src.phase_j.live_source import CapturedFrame
    from src.phase_j.live_bridge import recognize
    from src.phase_j import live_stage_timing as timing
    count, cursor = 0, 0
    capture = cv2.VideoCapture(str(VIDEO))
    assert capture.isOpened()
    with ExitStack() as stack:
        stack.callback(capture.release)
        timing.install(stack, args.out, 'recognition')
        stream = stack.enter_context(gzip.open(args.out/'outputs.jsonl.gz', 'wt', encoding='utf-8'))
        for item in indices:
            target = item['frame']
            while cursor < target:
                assert capture.grab(), cursor
                cursor += 1
            ok, image = capture.read()
            assert ok, target
            cursor += 1
            if image.shape[:2] != (1080, 1920):
                image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
            frame = CapturedFrame(target, item['t_sec'], 0., 0., image)
            notice = recognize(session, frame)
            # 認識の全フィールドとB3入力を保存。輸送時計だけを除外する。
            value = dict(frame=target, ready=session.publishing_ready,
                result=canonical(notice.result()), pipeline=canonical(notice.pipeline),
                b3=canonical(notice.b3))
            stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False)+'\n')
            timing.finish()
            count += 1
            if count % 100 == 0:
                print(dict(rows=count, t_sec=item['t_sec']), flush=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--candidate', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.source/'spool/frame_timing_recognition.jsonl'
    indices = [dict(frame=r['frame'], t_sec=r['t_sec']) for line in path.read_text().splitlines()
        if (r := json.loads(line)).get('frame') is not None and r.get('t_sec', 1200) < 1200]
    if args.limit:
        indices = indices[:args.limit]
    (args.out/'inputs.json').write_text(json.dumps(indices), encoding='utf-8')
    session = prepare(args)
    started = time.perf_counter()
    try:
        count = replay(args, session, indices)
    finally:
        from src.phase_j.live_b3 import close_capture
        close_capture(session)
    (args.out/'receipt.json').write_text(json.dumps(dict(rows=count,
        elapsed_sec=time.perf_counter()-started, video=str(VIDEO),
        input_sha256=sha256((args.out/'inputs.json').read_bytes()).hexdigest())), encoding='utf-8')


if __name__ == '__main__':
    main()
