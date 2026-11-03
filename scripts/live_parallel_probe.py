"""保存PNGで並列候補を交互比較する診断台。製品の既定値は変更しない。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import threading
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(_public_dev_text('<work>/assets/baseline_b/app'))
OUT = ROOT / 'logs/live_parallel'
FRAMES = (_public_dev_text('<work>/scratchpad/speed/framesA'))
MIN_FREE_BYTES = 6 * 1024**3
BELOW_NORMAL = 0x4000
LOGITS: list[tuple[str, Any]] = []
LOCK = threading.Lock()


def free_bytes() -> int:
    """依存パッケージを追加せずWindowsの空き物理メモリを確認する。"""
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ('total', 'available', 'page_total', 'page_available', 'virtual',
             'virtual_available', 'extended')]
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError()
    return int(memory.available)


def guard() -> None:
    """開始前および各コマ境界で利用者の負荷制限を守る。"""
    if Path(_public_dev_text('<work>/HOLD_FOR_OBS')).exists():
        raise RuntimeError('HOLD_FOR_OBSが存在するため試験を停止')
    if free_bytes() < MIN_FREE_BYTES:
        raise RuntimeError('空きメモリが6GiB未満のため試験を停止')


def install_ort(intra: int, inter: int, parallel: bool, no_spin: bool, product: bool) -> None:
    """既存のモデル・入力・演算を保ちセッションの実行設定だけ変える。"""
    import onnxruntime as ort
    from src import cnn_onnx
    original = ort.InferenceSession

    def session(path: str, options: Any, **kwargs: Any) -> Any:
        if product:
            return original(path, options, **kwargs)
        options.intra_op_num_threads = intra
        options.inter_op_num_threads = inter
        options.execution_mode = (ort.ExecutionMode.ORT_PARALLEL if parallel
                                  else ort.ExecutionMode.ORT_SEQUENTIAL)
        if no_spin:
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        return original(path, options, **kwargs)
    ort.InferenceSession = session
    run = cnn_onnx.run_logits

    def logits(session: Any, batch: Any) -> Any:
        output = run(session, batch)
        key = Path(session._model_path).name + hashlib.sha256(batch.tobytes()).hexdigest()
        with LOCK:
            LOGITS.append((key, output.copy()))
        return output
    cnn_onnx.run_logits = logits


def install_sides() -> Any:
    """診断限定: 共有較正を使わない設定で左右の読取だけを並行する。"""
    from concurrent.futures import ThreadPoolExecutor
    from src import image_reader
    pool = ThreadPoolExecutor(max_workers=1)
    source = textwrap.dedent(inspect.getsource(image_reader.ImageReader.read_both_boards))
    before = 'board_1p = self.read_board(frame, p1_region, hsv_full=hsv_full, skip_tier1=skip_tier1_1p)'
    after = ('assert not self._enable_side_sat_calibration, "側別較正の共有状態は並行不可"\n'
             '    pending = pool.submit(self.read_board, frame, p1_region, hsv_full=hsv_full, skip_tier1=skip_tier1_1p)')
    assert source.count(before) == 1
    source = source.replace(before, after).replace('return board_1p, board_2p',
                                                   'return pending.result(), board_2p')
    namespace = dict(vars(image_reader), pool=pool)
    exec(compile(source, '<parallel-probe>', 'exec'), namespace)
    image_reader.ImageReader.read_both_boards = namespace['read_both_boards']
    return pool


def configure(bench: Any, args: Any) -> None:
    """モデル資産と設定を固定し、実行直前に候補を適用する。"""
    build = bench.build_pipe

    def build_pipe(*values: Any, **kwargs: Any) -> Any:
        import cv2
        os.chdir(ASSETS)
        cv2.setNumThreads(args.cv)
        if args.idle:
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
        if args.cpu_sets:
            from ctypes import wintypes
            from src.phase_j import live_win_cores
            ids = live_win_cores.performance_cpu_set_ids(live_win_cores.read_cpu_sets())[:args.cpu_sets]
            kernel = live_win_cores._kernel32()
            if not kernel.SetProcessDefaultCpuSets(kernel.GetCurrentProcess(),
                    (wintypes.ULONG * len(ids))(*ids), len(ids)):
                raise ctypes.WinError()
        install_ort(args.intra, args.inter, args.parallel, args.no_spin, args.product)
        return build(*values, **kwargs)
    bench.build_pipe = build_pipe
    original = bench.install_recognize_wrapper

    def wrap(stages: bool) -> None:
        original(stages)
        from src.phase_j import live_process
        recognize = live_process.recognize

        def checked(*values: Any, **kwargs: Any) -> Any:
            guard()
            return recognize(*values, **kwargs)
        live_process.recognize = checked
    bench.install_recognize_wrapper = wrap


def install_prepare(bench: Any, mode: str) -> None:
    """固定画像・固定映像時刻で取り込み相当のコピーと色補正を重ねる。"""
    from concurrent.futures import ThreadPoolExecutor
    from src.phase_j.dshow_color import correct_601_as_709
    from src.phase_j.live_color_buffer import ColorBuffer
    buffer = ColorBuffer()
    correct = buffer.correct if mode == 'reuse' else correct_601_as_709

    def run(frames: list, indexes: list[int], pipe: Any, stages: bool) -> None:
        import time
        from src.phase_j.live_source import CapturedFrame
        from src.phase_j import live_process

        def prepare(index: int) -> Any:
            return correct(frames[index].copy())

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(prepare, 0) if mode == 'overlap' else None
            start = time.perf_counter()
            for position, index in enumerate(indexes):
                guard()
                image = pending.result() if pending is not None else prepare(position)
                if mode == 'overlap' and position + 1 < len(indexes):
                    pending = pool.submit(prepare, position + 1)
                now = time.perf_counter()
                live_process.recognize(pipe, CapturedFrame(index, index/30.0, now, now, image))
            (OUT / args_tag(bench) / 'prepare.json').write_text(json.dumps(dict(
                mode=mode, frames=len(indexes), wall_sec=time.perf_counter()-start,
                reused=buffer.reused, allocated=buffer.allocated)), encoding='utf-8')
    bench.run_core = run


def args_tag(bench: Any) -> str:
    """準備段試験の保存先をベンチの明示属性から取得する。"""
    return bench.parallel_probe_tag


def parse_args() -> Any:
    """診断条件を明示引数で受け取る。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--intra', type=int, default=1)
    parser.add_argument('--inter', type=int, default=1)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--no-spin', action='store_true')
    parser.add_argument('--idle', action='store_true')
    parser.add_argument('--cpu-sets', type=int, default=0)
    parser.add_argument('--sides', action='store_true')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--prepare', choices=('serial', 'overlap', 'reuse'))
    parser.add_argument('--video-start', type=float)
    parser.add_argument('--frames', type=int, default=900)
    parser.add_argument('--product', action='store_true')
    parser.add_argument('--source-root', type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    guard()
    sys.path[:0] = [str(args.source_root), str(ROOT/'scripts'), _public_dev_text('<work>/speed2')]
    import bench_live2 as bench
    configure(bench, args)
    bench.parallel_probe_tag = args.tag
    if args.prepare:
        install_prepare(bench, args.prepare)
    pool = install_sides() if args.sides else None
    video = None
    if args.video_start is not None:
        from live_parallel_video import VideoFrames, FPS
        video = VideoFrames(args.video_start, args.frames)
        first = round(args.video_start * FPS)
        bench.load_frames = lambda *unused: (video, list(range(first, first+args.frames)))
    output = OUT / args.tag
    output.mkdir(parents=True, exist_ok=True)
    sys.argv = ['bench', '--app', str(args.source_root), '--frames', FRAMES, '--n', '150',
        '--warmup', '60', '--layer', 'core', '--rec-config',
        'C:/puyo1_tools/soak/run_k/output/recognition_config.json', '--work', str(output),
        '--out', str(output/'summary.json'), '--digests', str(output/'digests.json'),
        '--calibration-dir', str(output/'calibration')]
    flags = dict(PUYO_CNN_BACKEND='onnx', PUYO_LIVE_CPU_THREADS=1,
        PUYO_LIVE_ONNX_PARALLEL=int(args.product),
        PUYO_PERFORMANCE_CORES=1, PUYO_B3_SCORE_MATMUL=1,
        PUYO_B3_WARNING_PARALLEL=1, PUYO_B3_WARNING_WORKERS=4,
        PUYO_B3_SKIP_ICON_READING=1, PUYO_B3_LEDGER_MODE=1,
        PUYO_B3_LEDGER_LANDING=1, PUYO_B3_SHARE_SCORE_READ=1)
    for key, value in flags.items():
        sys.argv += ['--env', f'{key}={value}']
    try:
        bench.main()
        import numpy as np
        ordered = sorted(LOGITS, key=lambda row: row[0])
        np.savez(output/'logits.npz', **{str(i): row[1] for i, row in enumerate(ordered)})
        (output/'keys.json').write_text(json.dumps([row[0] for row in ordered]), encoding='utf-8')
        (output/'candidate.json').write_text(json.dumps(vars(args), default=str), encoding='utf-8')
    finally:
        if video is not None:
            video.close()
        if pool is not None:
            pool.shutdown(wait=True)


if __name__ == '__main__':
    main()
