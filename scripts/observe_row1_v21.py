"""PC2短区間用の読み取り専用計装。entryからinstallを一度呼ぶ。"""
from __future__ import annotations

import dataclasses
import enum
import json
from pathlib import Path
from typing import Any

import numpy as np


def pack(value: Any) -> Any:
    """オブジェクトのreprやアドレスを比較票に混入させない。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, enum.Enum):
        return value.name
    if hasattr(value, '_grid'):
        return pack(value._grid)
    if dataclasses.is_dataclass(value):
        return {f.name: pack(getattr(value, f.name)) for f in dataclasses.fields(value)
                if f.name != 'prob_board'}
    if isinstance(value, dict):
        return {str(k): pack(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [pack(v) for v in value]
    if type(value).__name__ == 'DiskFIFO':
        return pack_fifo(value)
    if type(value).__name__ == 'deque':
        return [pack(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((pack(v) for v in value), key=repr)
    raise TypeError(type(value).__name__)


def pack_fifo(fifo: Any) -> dict:
    """DiskFIFOを消費せずに未読分を別ハンドルで読む (計装は読み取り専用)。"""
    import pickle
    fifo.writer.flush()
    with open(fifo.writer.name, 'rb') as stream:
        stream.seek(fifo.reader.tell())
        items = [pickle.load(stream) for _ in range(fifo.count)]
    return {'count': fifo.count, 'items': pack(items)}


def write(path: Path, value: Any) -> None:
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(pack(value), ensure_ascii=False)+'\n')


def install(cls: type, work: Path) -> None:
    """elのobserve_internal.installの置換。新しい出力先だけに追記する。"""
    original, update = cls._apply_row1_col2_guard, cls.update

    def guard(self: Any, side: str, published: Any, raw_cnn: Any, time_sec: float,
              in_chain: bool = False, next_moved: bool = False,
              ojama_falling: bool = False, settled: bool = False) -> Any:
        result = original(self, side, published, raw_cnn, time_sec,
                          in_chain, next_moved, ojama_falling, settled)
        self._v21_observation[side] = dict(before=pack(published), after=pack(result),
            raw_cnn_board=pack(raw_cnn), signals=dict(formula=in_chain,
                next_moved=next_moved, ojama_falling=ojama_falling, settled=settled))
        return result

    def observed(self: Any, frame_idx: int, time_sec: float, frame: Any) -> Any:
        self._v21_observation = {}
        result = update(self, frame_idx, time_sec, frame)
        sides = {}
        for side, suffix in (('1P', '1p'), ('2P', '2p')):
            ctx = getattr(self, '_sm_'+suffix).context
            sides[side] = dict(guard=self._v21_observation.get(side),
                confirmed_board=ctx.confirmed_board,
                prev_confirmed=getattr(self, '_prev_confirmed_'+suffix),
                prev_stable_confirmed=getattr(self, '_prev_stable_confirmed_'+suffix),
                tsumo_count=self.tsumo_count(side),
                pending_tsumo=getattr(self, '_pending_tsumo_'+suffix))
        write(work/'internal_history.jsonl', dict(frame=frame_idx, time_sec=time_sec,
                                                sides=sides, result=result))
        return result

    cls._apply_row1_col2_guard, cls.update = guard, observed
    install_b3(work)


def install_b3(work: Path) -> None:
    """観測・イベント・NEXT行・全特徴列をモデル推論と同じ入口で採取する。"""
    from src.phase_j import live_b3
    capture, predict = live_b3.capture, live_b3.LiveB3._predict

    def observed_capture(pipe: Any, result: Any, frame: Any, counts: tuple) -> Any:
        value = capture(pipe, result, frame, counts)
        if value is not None:
            write(work/'b3_inputs.jsonl', dict(frame=frame.index, counts=counts, input=value))
        return value

    def observed_predict(self: Any, row: dict, notice: Any) -> None:
        write(work/'b3_features.jsonl', dict(frame=notice.frame, features=row))
        predict(self, row, notice)

    live_b3.capture, live_b3.LiveB3._predict = observed_capture, observed_predict
