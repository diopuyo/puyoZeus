"""video39のcold連続NEXTを元実装の私有namespaceへ接続する。全G3移植ではない。"""
from __future__ import annotations
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator
from scripts import g3_model_process as P

SOURCE = 'video_39'
NEXT = P.M.ROOT / 'scripts/next_enqueue_live_shadow_v1.py'
NEXT_SHA = 'e5ebff6827119c616319b4598fce1428c98643b621ff02e985736b43783d9237'
END, FPS, STRIDE = 11878, 30, 1


def validate(item: dict) -> None:
    """既定source/時計/範囲だけを受け入れる。GO認定と盤面GTは別の条件。"""
    P.M.require(item.get('source') == SOURCE, 'next_source39_only')
    P.M.source_contract(SOURCE, item['source_id'].removeprefix('sha256:'))
    P.M.require(item['time_base'] == [1, FPS] and item['stride'] == STRIDE
                and item['end_frame_exclusive'] == END, 'next_source39_clock_bounds')


@contextmanager
def namespace(item: dict) -> Iterator[Any]:
    """元BASE/globalを変更せず、元pipeline SHA/行番号guardをそのまま使う。"""
    validate(item)
    source_id = item['source_id']
    with P.loaded(NEXT, NEXT_SHA) as module:
        original = module.BASE
        P.M.require(original.FPS == 60 and module.FIRST_FRAME == original.RESET_FRAME
                    and module.STRIDE == 2, 'next_original_configuration')
        module.BASE = SimpleNamespace(**vars(original))
        module.BASE.FPS = FPS
        module.FIRST_FRAME, module.LAST_FRAME, module.STRIDE = 0, END - 1, STRIDE
        native_clock = module._clock
        def clock(frame: int, time_sec: float) -> None:
            P.request_gate(dict(frame=frame, row=dict(source_id=source_id,
                           frame_idx=frame, time_sec=time_sec)),
                           dict(source_id=source_id, end_frame_exclusive=END,
                                stride=STRIDE, time_base=[1, FPS]))
            native_clock(frame, time_sec)
        module._clock = clock
        yield module


class Recorder:
    """既存writerを再用し、source/run来歴を追加する。判定器は増やさない。"""
    def __init__(self, original: Any, source_id: str, run_id: str) -> None:
        self.original, self.source_id, self.run_id = original, source_id, run_id

    def emit(self, row: dict) -> None:
        P.M.require('source_id' not in row and 'run_id' not in row, 'next_foreign_row_identity')
        self.original.emit(row | dict(source_id=self.source_id, run_id=self.run_id))


def install(stack: Any, collector: Any, rec: Any, item: dict, run_id: str) -> Any:
    """既存wrapperより前の元cold pipeline専用。元installが二重併存を拒否する。"""
    P.M.require(type(run_id) is str and bool(run_id), 'next_run_identity')
    module = stack.enter_context(namespace(item))
    controller = module.install(stack, collector, Recorder(rec, item['source_id'], run_id))
    controller.g3_source_contract = dict(source=SOURCE, source_id=item['source_id'],
        run_id=run_id, first=0, end_exclusive=END, stride=STRIDE,
        time_base=[1, FPS], software_reset_is_game_proof=False,
        physical_placement_verified=False, quality_gate_clear=False)
    return controller
