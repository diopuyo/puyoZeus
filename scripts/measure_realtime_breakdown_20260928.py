"""既存overlayをperf_counterで段別計装する。既定は短区間のスモーク用。"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any, Callable, Iterator
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path('/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer')
DEFAULT_STATUS = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/'
                      'review_zenchi_g41_43_e14/status.json')
DEFAULT_VIDEO = ASSETS / 'data/frames/video_zenchi_c0BQoMJwwQU.mp4'
START_SEC, END_SEC, WARMUP_SEC = 2600.0, 2610.0, 1.0
MILLISECONDS = 1000.0
PERCENTILES = (50, 95, 99)
STAGES = ('decode_1080p', 'recognition', 'exchange_other', 'features',
          'M0_G_fe', 'S1_S3', 'landing', 'render', 'write')
EXCHANGE_STAGES = ('exchange_other', 'features', 'M0_G_fe', 'S1_S3', 'landing')
PATH_FLAGS = {'--out': 'overlay.mp4', '--dump-timeline': 'settled.npz',
              '--dump-display-timeline': 'display.npz',
              '--exchange-event-record': 'inputs.jsonl.gz',
              '--dump-exchange-events': 'events.jsonl', '--review-data-csv': 'review_data.csv'}


class FrameMeter:
    """各原動画フレームを記録し、子計装の時間を親から差し引く。"""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.stack: list[list[Any]] = []

    @contextmanager
    def frame(self, index: int, timestamp: float) -> Iterator[None]:
        self.current = dict(frame=index, t_sec=timestamp, processed=False,
                            **{name: 0.0 for name in STAGES})
        start = perf_counter()
        try:
            yield
        finally:
            self.current['total_ms'] = (perf_counter() - start) * MILLISECONDS
            self.rows.append(self.current)
            self.current = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.current is None:
            yield
            return
        entry = [name, perf_counter(), 0.0]
        self.stack.append(entry)
        if name == 'recognition':
            self.current['processed'] = True
        try:
            yield
        finally:
            elapsed = perf_counter() - entry[1]
            self.stack.pop()
            self.current[name] += (elapsed - entry[2]) * MILLISECONDS
            if self.stack:
                self.stack[-1][2] += elapsed

    def wrap(self, function: Callable[..., Any], name: str | Callable[..., str]) -> Callable[..., Any]:
        """旧診断器のperf_counterラップ方式を、復元可能な形で再用する。"""
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            label = name(*args, **kwargs) if callable(name) else name
            with self.stage(label):
                return function(*args, **kwargs)
        return wrapped


def stage_node(name: str, body: list[ast.stmt]) -> ast.With:
    """指定した文列だけを計装し、評価文そのものは維持する。"""
    context = ast.parse(f"with _realtime_meter.stage('{name}'):\n    pass").body[0]
    context.body = body
    return context


def instrument_generate(module: Any, meter: FrameMeter) -> Callable[..., Any]:
    """本体ファイルを変更せず、既存generateのループへ計装を挿入する。"""
    tree = ast.parse(inspect.getsource(module.generate))
    function = tree.body[0]
    loops = [n for n in function.body if isinstance(n, ast.For)
             and ast.unparse(n.target) == 'fi']
    if len(loops) != 1:
        raise ValueError('generateのフレームループが変更されています')
    loop = loops[0]
    body, drawing = [], []
    rendering = False
    found: set[str] = set()
    for node in loop.body:
        code = ast.unparse(node)
        if code.startswith('waiting ='):
            rendering = True
        if code == 'writer.write(frame_out)':
            body.extend([stage_node('render', drawing), stage_node('write', [node])])
            rendering = False
            found.add('write')
        elif rendering:
            drawing.append(node)
        else:
            body.extend(instrument_statement(node, code, found))
    if found != {'decode', 'resize', 'recognition', 'exchange', 'write', 'display_resize'}:
        raise ValueError(f'計装対象の欠落: {found}')
    context = ast.parse('with _realtime_meter.frame(fi, fi / fps):\n    pass').body[0]
    context.body = body
    loop.body = [context]
    namespace = dict(vars(module), _realtime_meter=meter)
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(module), 'exec'), namespace)
    return namespace['generate']


def instrument_statement(node: ast.stmt, code: str, found: set[str]) -> list[ast.stmt]:
    """デコード・認識・撃ち合いの既存境界を厳密に選ぶ。"""
    targets = (('ok, frame = cap.read()', 'decode', 'decode_1080p'),
               ('if resize_1080p:', 'resize', 'decode_1080p'),
               ('r = pipe.update(', 'recognition', 'recognition'),
               ('if event_overlay is not None:', 'exchange', 'exchange_other'))
    for prefix, marker, name in targets:
        if code.startswith(prefix):
            found.add(marker)
            return [stage_node(name, [node])]
    if code.startswith('if frame.shape[:2] != (OUT_H, OUT_W):'):
        found.add('display_resize')
        # no-render時には表示専用の縮小も行わない。
        guard = ast.parse('if render:\n    pass').body[0]
        guard.body = [stage_node('render', [node])]
        return [guard]
    return [node]


def install_substages(stack: ExitStack, meter: FrameMeter, vao: Any) -> None:
    """特徴生成・モデル・仮想着弾を排他的に計測する。"""
    import src.exchange_event_evaluator as evaluator
    import src.exchange_event_overlay as overlay
    from src.exchange_event_landing import ExchangeLandingProjection
    from src.exchange_event_m0 import FileM0Predictor
    targets = [(vao, '_exchange_static_input', 'features'),
               (overlay, 'prefire_side_features', 'features'),
               (evaluator, 'build_features', 'features'),
               (evaluator, 'count_sides', 'features'),
               (FileM0Predictor, '__call__', 'M0_G_fe'),
               (ExchangeLandingProjection, 'update', 'landing'),
               (evaluator.FileExchangeModels, 'predict_source_probability', model_stage)]
    for owner, name, label in targets:
        stack.enter_context(patch.object(owner, name, meter.wrap(getattr(owner, name), label)))


def model_stage(instance: Any, model_name: str, features: np.ndarray) -> str:
    """G_feとS1/S3の推論時間を区別する。"""
    return 'M0_G_fe' if model_name == 'G_fe' else 'S1_S3'


def build_command(options: argparse.Namespace) -> list[str]:
    """指定statusのフラグを維持し、入出力と計測区間だけ置換する。"""
    saved = json.loads(options.status.read_text(encoding='utf-8'))['command']
    command = saved[saved.index('--worker') + 1:]
    replacements = dict(PATH_FLAGS)
    replacements = {key: str(options.output / value) for key, value in replacements.items()}
    replacements.update({'--video': str(options.video), '--start-sec': str(options.start_sec),
                         '--end-sec': str(options.end_sec), '--warmup-sec': str(options.warmup_sec),
                         '--exchange-event-model-dir': 'models/exchange_event_v2'})
    for flag, value in replacements.items():
        if flag not in command:
            raise ValueError(f'statusの必須フラグがありません: {flag}')
        command[command.index(flag) + 1] = value
    if options.no_render:
        command.append('--no-render')
    return command


def statistics(values: list[float]) -> dict[str, float]:
    """未呼出フレームのゼロも含めて同一母集団で集計する。"""
    return dict(mean=float(np.mean(values)),
                **{f'P{p}': float(np.percentile(values, p)) for p in PERCENTILES})


def summarize(meter: FrameMeter, options: argparse.Namespace) -> dict[str, Any]:
    """間引きフレームのデコード費用を直前の処理フレームへ加算する。"""
    raw = [r for r in meter.rows if options.start_sec <= r['t_sec'] < options.end_sec]
    grouped: list[dict[str, Any]] = []
    for row in raw:
        if row['processed']:
            grouped.append(dict(row))
        elif grouped:
            for name in (*STAGES, 'total_ms'):
                grouped[-1][name] += row[name]
    if not grouped:
        raise ValueError('計測対象の処理フレームがありません')
    for row in grouped:
        row['exchange_total'] = sum(row[key] for key in EXCHANGE_STAGES)
        row['other'] = row['total_ms'] - sum(row[key] for key in STAGES)
    names = (*STAGES, 'exchange_total', 'other', 'total_ms')
    duration = sum(r['total_ms'] for r in grouped) / MILLISECONDS
    return dict(frames=len(grouped), decoded_frames=len(raw), processing_seconds=duration,
                processing_fps=len(grouped) / duration,
                stages_ms={key: statistics([r[key] for r in grouped]) for key in names},
                frames_ms=grouped, raw_frames_ms=raw)


def parse_args() -> argparse.Namespace:
    """本測定の区間・並列負荷は明示指定で変更できる。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=DEFAULT_VIDEO)
    parser.add_argument('--status', type=Path, default=DEFAULT_STATUS)
    parser.add_argument('--output', type=Path, default=Path('logs/realtime_breakdown_20260928'))
    parser.add_argument('--start-sec', type=float, default=START_SEC)
    parser.add_argument('--end-sec', type=float, default=END_SEC)
    parser.add_argument('--warmup-sec', type=float, default=WARMUP_SEC)
    parser.add_argument('--parallel-count', type=int, default=1,
                        help='同時に動かす計測プロセス数の申告値。外部負荷は含めない')
    parser.add_argument('--no-render', action='store_true')
    options = parser.parse_args()
    if not 0 <= options.start_sec < options.end_sec or options.warmup_sec < 0:
        parser.error('計測区間とwarmupが不正です')
    if options.parallel_count < 1:
        parser.error('並列数は1以上です')
    return options


def run(options: argparse.Namespace) -> dict[str, Any]:
    """本体の評価設定・乱数初期化を再現し、終了時に計装を復元する。"""
    import cv2
    import torch
    import scripts.visualize_advantage_overlay as vao
    from scripts.run_e3_exchange_eval_20260926 import SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    command = build_command(options)
    meter = FrameMeter()
    start = perf_counter()
    with ExitStack() as stack:
        install_substages(stack, meter, vao)
        stack.enter_context(patch.object(vao, 'generate', instrument_generate(vao, meter)))
        stack.enter_context(patch.object(sys, 'argv', ['measure_realtime', *command]))
        vao.main()
    result = summarize(meter, options)
    result.update(command=command, wall_seconds_including_setup=perf_counter() - start,
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  parallel_count=options.parallel_count, torch_threads=torch.get_num_threads(),
                  opencv_threads=cv2.getNumThreads(), nice=os.nice(0),
                  status_sha256=hashlib.sha256(options.status.read_bytes()).hexdigest(),
                  status_source=str(options.status), warmup_sec=options.warmup_sec,
                  no_render=options.no_render,
                  timing='perf_counter壁時計。子段は親から除外。exchange_totalのみ包含値。')
    return result


def main() -> None:
    """フレーム明細と集計を同じJSONに保存する。"""
    options = parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    result = run(options)
    destination = options.output / 'breakdown.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('frames_ms', 'raw_frames_ms')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
