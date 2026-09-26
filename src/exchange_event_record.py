"""撃ち合い評価境界をgzip JSONLで記録する。認識画像は保存しない。"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.board import Board
from src.board_state_machine import BoardState
from src.exchange_event_evaluator import StaticInput

RECORD_VERSION = 1
COMPRESSION_LEVEL = 6
SIDE_FIELDS = ("state", "confirmed_board", "score", "next_pair", "dnext_pair",
               "next_slide_motion")
CHAIN_FIELDS = ("trigger_sec", "mechanism", "chain_count", "total_score")
SNAPSHOT_FIELDS = ("net_balance_capped", "forecast_p1", "total_dropped_to_p1",
                   "total_dropped_to_p2")
FINALIZATION_FIELDS = ("finalized_count_p1", "finalized_count_p2",
                       "chain_total_score_p1", "chain_total_score_p2")


def encode(value: Any) -> Any:
    """配列の精度・欠測・列挙型を損失なく保存する。"""
    if isinstance(value, Board):
        return {"board": encode(value._grid)}
    if isinstance(value, np.ndarray):
        return {"array": value.tolist(), "dtype": value.dtype.str}
    if isinstance(value, BoardState):
        return {"state": value.name}
    if isinstance(value, SimpleNamespace):
        return {"namespace": encode(vars(value))}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"tuple": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


def decode(value: Any) -> Any:
    """記録した型を復元し、呼出側の可変盤面とは分離する。"""
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "board" in value:
        board = Board()
        board._grid = decode(value["board"])
        return board
    if "array" in value:
        return np.asarray(value["array"], dtype=value["dtype"])
    if "state" in value and len(value) == 1:
        return BoardState[value["state"]]
    if "namespace" in value:
        return SimpleNamespace(**decode(value["namespace"]))
    if "tuple" in value:
        return tuple(decode(v) for v in value["tuple"])
    return {k: decode(v) for k, v in value.items()}


def fields(value: Any, names: tuple[str, ...]) -> SimpleNamespace:
    """参照対象を明示し、将来の入力不足は属性エラーで検出する。"""
    return SimpleNamespace(**{name: getattr(value, name) for name in names})


def static_key(boards: tuple[Board, Board], snapshot: Any) -> str:
    """経過秒・M0と独立したD入力のキーを作る。"""
    raw = json.dumps(encode((tuple(b._grid for b in boards),
        snapshot.net_balance_capped, snapshot.forecast_p1)), separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def read_records(path: Path) -> Iterator[dict]:
    """一度に全動画を展開せず、完了マーカーまで順に読む。"""
    complete = False
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = decode(json.loads(next(stream)))
        if header.get("version") != RECORD_VERSION:
            raise ValueError("非対応の撃ち合い記録版")
        yield header
        for line in stream:
            row = decode(json.loads(line))
            complete = row["kind"] == "complete"
            yield row
    if not complete:
        raise ValueError("撃ち合い記録が未完了")


class ExchangeEventRecorder:
    """update入力と静止特徴を観測するだけの記録器。"""

    def __init__(self, path: Path, video_id: str, per_side_settled: bool,
                 model_dir: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = gzip.open(path, "wt", encoding="utf-8", compresslevel=COMPRESSION_LEVEL)
        self.write(dict(kind="header", version=RECORD_VERSION, video_id=video_id,
                        per_side_settled=per_side_settled, model_dir=str(model_dir)))
        self.static_keys: set[str] = set()
        self.frames = 0

    def write(self, row: dict) -> None:
        """浮動小数点はJSONの往復精度で維持する。NaNは欠測として保存する。"""
        self.stream.write(json.dumps(encode(row), ensure_ascii=False, separators=(",", ":")) + "\n")

    def update(self, result: Any, snapshot: Any, finalization: Any, t_sec: float,
               game_idx: int, formula_totals: tuple, displayed_scores: tuple,
               formula_visible: tuple) -> None:
        """overlay呼出直前に入力を複製し、後続の認識器更新と切り離す。"""
        sides = []
        for side in (result.p1, result.p2):
            saved = fields(side, SIDE_FIELDS)
            saved.chain_event = (fields(side.chain_event, CHAIN_FIELDS)
                                 if side.chain_event is not None else None)
            sides.append(saved)
        self.write(dict(kind="update", args=(SimpleNamespace(p1=sides[0], p2=sides[1]),
            fields(snapshot, SNAPSHOT_FIELDS), fields(finalization, FINALIZATION_FIELDS),
            t_sec, game_idx, formula_totals, displayed_scores, formula_visible)))
        self.frames += 1

    def wrap_static(self, builder: Any) -> Any:
        """実際に計算したD・静止入力を保存し、戻り値をそのまま返す。"""
        def build(boards: tuple, snapshot: Any, elapsed: float, m0: float) -> StaticInput:
            result = builder(boards, snapshot, elapsed, m0)
            key = static_key(boards, snapshot)
            if key not in self.static_keys:
                self.write(dict(kind="static", key=key, input=asdict(result)))
                self.static_keys.add(key)
            self.write(dict(kind="static_call", key=key, elapsed_sec=elapsed,
                            m0_probability_1p=m0, source_side=result.source_side))
            return result
        return build

    def close(self) -> None:
        """完了マーカーとgzip末尾を確定する。"""
        self.write(dict(kind="complete", frames=self.frames))
        self.stream.close()
