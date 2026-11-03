"""本番描画CLIの初期化・会計・R判定をそのまま使う連続比較用アダプター。"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import numpy as np


def compile_runtime(module: Any) -> tuple:
    """本番と同じCLI引数解決を使い、描画より前の認識・会計だけを抽出する。"""
    tree = ast.parse(inspect.getsource(module.generate))
    function = tree.body[0]
    index = next(i for i, n in enumerate(function.body)
                 if isinstance(n, ast.For) and ast.unparse(n.target) == 'fi')
    loop = function.body[index]
    codes = [ast.unparse(n) for n in loop.body]
    first = next(i for i, c in enumerate(codes) if c.startswith('r = pipe.update('))
    last = next(i for i, c in enumerate(codes) if c.startswith('ps1, ps2 ='))
    body = loop.body[first:last+1]
    body.insert(1, ast.parse('r = _enrich(pipe, recog_frame, r, t)').body[0])
    single = ast.For(target=ast.Name(id='_once', ctx=ast.Store()),
                     iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()), body=body, orelse=[])
    step = compile(ast.fix_missing_locations(ast.Module(body=[single], type_ignores=[])),
                   inspect.getfile(module), 'exec')
    function.body = function.body[:index] + ast.parse('return locals()').body
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(module), 'exec'), namespace)
    return namespace['generate'], step


def initialize(video: Path, output: Path) -> tuple:
    from scripts import visualize_advantage_overlay as module
    init, code = compile_runtime(module)
    result = []

    def collect(*args: Any, **kwargs: Any) -> int:
        kwargs.update(render=False, normalize_fps_30=False, start_sec=0., end_sec=0.)
        result.append(init(*args, **kwargs))
        return 0

    argv = ['image_free_runtime', '--video', str(video), '--out', str(output / 'unused.mp4'),
            '--production-exchange-event', '--no-normalize-fps-30', '--no-render']
    with patch.object(sys, 'argv', argv), patch.object(module, 'generate', collect):
        module.main()
    state = result[0]
    state['cap'].release()
    state.pop('cap')
    # ライブ同様に、既存CNN観測から途中盤面・発火前窓・勝敗を一度だけ付加する。
    for name in ('terminal_detector', 'midchain_reader', 'snapshot_reader'):
        state[name] = None
    return state, code


class Runtime:
    def __init__(self, video: Path, output: Path, config: dict) -> None:
        from scripts import visualize_advantage_overlay as module
        from src.phase_j.live_snapshot import enrich_recognition
        from src.b3_train_model import load_model
        from src.b3_timing import TimingTable
        from src.b3_realtime import RealtimeRows
        from src.b3_live_display import B3Display
        self.state, self.code = initialize(video, output)
        self.globals = dict(vars(module), _enrich=enrich_recognition)
        self.model = load_model(Path(config['b3_model']))
        self.rows = RealtimeRows(TimingTable.from_json(Path(config['b3_table']).read_text()),
                                 measure_timing=False)
        self.display = B3Display()
        self.physical: set[float] = set()
        self.epoch: Any = None
        self.adopted_game: int | None = None
        self.config = config
        from scripts.train_image_free_all import teachers
        self.probes = teachers()

    def step(self, frame: np.ndarray, index: int, fps: float) -> dict:
        from src.phase_j.live_snapshot import prepare_recognition
        from src.phase_j.live_source import CapturedFrame
        from src.phase_j.live_b3 import capture
        state, stamp = self.state, index / fps
        pipe = state['pipe']
        prepare_recognition(pipe)
        pipe.observe_placement_frame(index, stamp, frame)
        state.update(fi=index, t=stamp, recog_frame=frame, frame=frame)
        namespace = dict(self.globals, **state)
        exec(self.code, namespace)
        state.update({k: namespace[k] for k in set(state) | (set(namespace) - set(self.globals))})
        result = state['r']
        counts = tuple(pipe.tsumo_count(s) for s in ('1P', '2P'))
        packet = capture(pipe, result, CapturedFrame(index, stamp, 0., 0., frame), counts)
        row = self.evaluate(packet, state['event_overlay'])
        from scripts.image_free_runtime_observe import observe
        return observe(pipe, result, packet, row, self.display, frame, self.probes)

    def evaluate(self, packet: tuple, overlay: Any) -> dict | None:
        from src.b3_realtime import RealtimeRows
        from src.b3_live_display import B3Display
        from src.phase_j.live_b3 import reference_state
        from scripts.visualize_advantage_overlay import _WINPROB_K
        frame, events, queues, epoch = packet
        if epoch != self.epoch:
            self.rows = RealtimeRows(self.rows.table, measure_timing=False)
            self.display, self.epoch = B3Display(), epoch
            self.physical.clear()
            self.adopted_game = None
        adopted = frame.get('adopted_game')
        if adopted is not None and adopted != self.adopted_game:
            self.adopted_game, self.display = adopted, B3Display()
        previous_game = self.display.game
        self.display.boundary(frame['game_idx'])
        if previous_game != self.display.game:
            self.physical.clear()
        self.display.chain.observe(frame)
        row = self.rows.update(frame, events, queues)
        if row is not None:
            self.display.prediction(self.model.predict_row(row), int(row['mode_exchange']), frame['t_sec'])
        reference = reference_state(overlay.tracker)
        self.physical.update(float(t) for t in reference['physical'] if t <= frame['t_sec'])
        self.display.update(frame['t_sec'], reference['source'], reference['probability'],
                            len(self.physical), _WINPROB_K, bool(frame['match_active']))
        return row

    def __getstate__(self) -> dict:
        """コードとモジュール名前空間は再構成し、時系列状態だけを保存する。"""
        values = dict(self.__dict__)
        values.pop('code')
        values.pop('globals')
        values['state'] = {k: v for k, v in self.state.items()
                           if k not in ('frame', 'recog_frame', 'raw_native')}
        return values

    def __setstate__(self, values: dict) -> None:
        from scripts import visualize_advantage_overlay as module
        from src.phase_j.live_snapshot import enrich_recognition
        self.__dict__.update(values)
        _, self.code = compile_runtime(module)
        self.globals = dict(vars(module), _enrich=enrich_recognition)
