"""配布版 python (Windows) / WSL で、認識 1 frame の費用と決定を同一 PNG 連番に対して測る。

使い方 (cwd = 配布物の app/。._pth が cwd を見ないので cwd は models/ 等の相対パス解決用):
    python bench_recognition.py --frames DIR --n 900 --out result.json [--digests d.json] [--profile p.txt]
環境変数: PUYO_FAST_TERMINAL / PUYO_CNN_BACKEND / PUYO_TORCH_THREADS などは呼び出し側で設定する。
割り算の単位は「認識が実際に処理した frame」(先頭 WARMUP_FRAMES を暖機として除外)。PNG 読込は計時外。
出力: P50/P95/P99/mean と、frame 毎の決定 digest (盤面・状態など result の全フィールドを正準化したハッシュ)。
"""
from __future__ import annotations

import argparse
import cProfile
import dataclasses
import hashlib
import io
import json
import os
import pickle
import pstats
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # 配布版と同じ CPU のみ
BENCH_SRC_ENV = 'PUYO_BENCH_SRC'  # 比較用: 別コミットの src を先頭へ差し込む (基準側の測定用)
if os.environ.get(BENCH_SRC_ENV):
    sys.path.insert(0, os.environ[BENCH_SRC_ENV])
from src.phase_j.live_cpu import configure_environment  # noqa: E402

CPU_THREADS, EVALUATION_NICE = 1, 10  # launcher.PIPELINE_FIXED と同値
configure_environment(CPU_THREADS, EVALUATION_NICE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

WARMUP_FRAMES = 60
PERCENTILES = (50, 95, 99)
MILLISECONDS = 1000.0
SOURCE_FPS = 60.0
GC_RELAXED_THRESHOLD, GC_RELAXED_GEN1, GC_RELAXED_GEN2 = 50000, 20, 100
# 段別計時の対象 (module, class, method)。包含時間 (入れ子は二重に数える) で、原因の当たりを付けるための目安
STAGE_TARGETS = (
    ('src.telop_detector', 'TelopDetector', 'detect'), ('src.match_end_detector', 'MatchEndDetector', 'detect'),
    ('src.match_end_detector', 'MatchEndDetector', 'update'), ('src.score_zero', 'ScoreZeroDetector', 'detect'),
    ('src.next_detector', 'NextDetector', 'detect_both'), ('src.image_reader', 'ImageReader', 'read_both_boards'),
    ('src.match_state', 'MatchStateDetector', 'detect'), ('src.next_slide_detector', 'NextSlideDetector', 'update'),
    ('src.score_ocr', 'ScoreTracker', 'update'), ('src.score_ocr', 'FormulaStepAccumulator', 'update'),
    ('src.exchange_event_terminal', 'ObservedDeathDetector', 'update'),
)
STAGE_TOTAL = 'recognize_total'
TIMING_KEY_PARTS = ('cost', 'sec', 'elapsed', 'time', 'latency')  # 壁時計依存の値は digest から除く


def canonical(value: Any, sink: 'hashlib._Hash', key: str = '') -> None:
    """result を型ごとに正準なバイト列へ落として hash へ流す (壁時計由来の key は除外)。"""
    if any(part in key.lower() for part in TIMING_KEY_PARTS):
        return
    if isinstance(value, np.ndarray):
        sink.update(str(value.dtype).encode() + str(value.shape).encode() + np.ascontiguousarray(value).tobytes())
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            canonical(getattr(value, field.name), sink, field.name)
    elif isinstance(value, dict):
        for name in sorted(value, key=repr):
            canonical(value[name], sink, str(name))
    elif isinstance(value, (list, tuple)):
        sink.update(b'[')
        for item in value:
            canonical(item, sink, key)
        sink.update(b']')
    elif hasattr(value, '__dict__') and not callable(value):
        for name in sorted(vars(value)):
            canonical(getattr(value, name), sink, name)
    else:
        sink.update(repr(value).encode())


def png_frames(directory: Path):
    """stride2 抽出済みの PNG 連番から CapturedFrame を作る (読込は generator 内=計時外)。"""
    from src.phase_j.live_source import CapturedFrame
    for path in sorted(directory.glob('frame_*.png')):
        index = int(path.stem.split('_')[1])
        now = time.perf_counter()
        yield CapturedFrame(index, index/SOURCE_FPS, now, now, cv2.imread(str(path)))


def load_pipe(config_path: Path):
    from src.phase_j.live_cpu import apply_runtime
    from src.recognition_pipeline import RecognitionPipeline
    apply_runtime('recognition')
    return RecognitionPipeline.load_default(**json.loads(config_path.read_text(encoding='utf-8')))


def install_stage_timers(store: dict[str, float]) -> None:
    """対象メソッドを計時付きに差し替える (この process 内だけ。決定は変えない)。"""
    import functools
    import importlib
    for module_name, class_name, method in STAGE_TARGETS:
        owner = getattr(importlib.import_module(module_name), class_name, None)
        original = getattr(owner, method, None)
        if original is None:
            print('stage target missing:', module_name, class_name, method, file=sys.stderr)
            continue
        label = f'{class_name}.{method}'

        def make(function, name):
            @functools.wraps(function)
            def timed(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    store[name] = store.get(name, 0.0)+(time.perf_counter()-start)*MILLISECONDS
            return timed
        setattr(owner, method, make(original, label))


def run(pipe: Any, directory: Path, limit: int, profiler: cProfile.Profile | None,
        stages: list[dict[str, float]] | None = None, store: dict[str, float] | None = None) -> tuple[list[float], list[str]]:
    from src.phase_j.live_bridge import recognize
    times: list[float] = []
    digests: list[str] = []
    for count, frame in enumerate(png_frames(directory)):
        if count >= limit:
            break
        if profiler is not None and count >= WARMUP_FRAMES:
            profiler.enable()
        if store is not None:
            store.clear()
        start = time.perf_counter()
        notice = recognize(pipe, frame)
        times.append((time.perf_counter()-start)*MILLISECONDS)
        if stages is not None and store is not None:
            stages.append(dict(store))
        if profiler is not None and count >= WARMUP_FRAMES:
            profiler.disable()
        sink = hashlib.sha256()
        canonical(pickle.loads(notice.result_bytes), sink)
        canonical(notice.pipeline, sink)
        digests.append(sink.hexdigest()[:16])
    return times, digests


def summarize(times: list[float]) -> dict[str, float]:
    steady = np.array(times[WARMUP_FRAMES:])
    stats = {f'p{p}': float(np.percentile(steady, p)) for p in PERCENTILES}
    return dict(n=int(steady.size), mean=float(steady.mean()), **stats)


def set_affinity(mask: int) -> None:
    """Windows: 論理 CPU の集合を固定する (P コア / E コアの切り分け実験用。本番設定ではない)。"""
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
        raise OSError('SetProcessAffinityMask 失敗')


def apply_experiments(options: argparse.Namespace) -> None:
    """実験用の実行時設定 (決定は変えない前提。決定同一は digest で毎回確かめる)。"""
    import gc
    if options.cv2_threads:
        cv2.setNumThreads(options.cv2_threads)
    if options.torch_threads:
        import torch
        torch.set_num_threads(options.torch_threads)
    if options.gc == 'freeze':
        gc.collect()
        gc.freeze()
        gc.set_threshold(GC_RELAXED_THRESHOLD, GC_RELAXED_GEN1, GC_RELAXED_GEN2)
    elif options.gc == 'off':
        gc.disable()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv2-threads', type=int, default=0)
    parser.add_argument('--torch-threads', type=int, default=0)
    parser.add_argument('--stages', type=Path, help='段別計時 (frame 毎) の出力先 json')
    parser.add_argument('--gc', choices=('default', 'freeze', 'off'), default='default')
    parser.add_argument('--affinity', type=lambda text: int(text, 16), default=0, help='16進マスク (Windows のみ、0=変更なし)')
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--n', type=int, default=900+WARMUP_FRAMES)
    parser.add_argument('--config', type=Path, default=Path('D:/puyo_analyzer/packaging/perf/recognition_config.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--digests', type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--tag', default='')
    options = parser.parse_args()
    if options.affinity:
        set_affinity(options.affinity)
    pipe = load_pipe(options.config)
    apply_experiments(options)
    profiler = cProfile.Profile() if options.profile else None
    stages: list[dict[str, float]] | None = [] if options.stages else None
    store: dict[str, float] = {}
    if options.stages:
        install_stage_timers(store)
    times, digests = run(pipe, options.frames, options.n, profiler, stages, store)
    if options.stages:
        options.stages.write_text(json.dumps(stages), encoding='utf-8')
    import src
    result = dict(tag=options.tag, src=str(Path(src.__file__).parent), frames_total=len(times), python=sys.version.split()[0], platform=sys.platform,
                  env={k: v for k, v in os.environ.items() if k.startswith('PUYO_')}, **summarize(times))
    options.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    if options.digests:
        options.digests.write_text(json.dumps(digests), encoding='utf-8')
        options.digests.with_suffix('.times.json').write_text(json.dumps([round(t, 3) for t in times]), encoding='utf-8')
    if profiler is not None:
        buffer = io.StringIO()
        pstats.Stats(profiler, stream=buffer).sort_stats('tottime').print_stats(45)
        profiler.dump_stats(str(options.profile.with_suffix('.prof')))  # 呼び出し元の照会用
        options.profile.write_text(buffer.getvalue(), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
