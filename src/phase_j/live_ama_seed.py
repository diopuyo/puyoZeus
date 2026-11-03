"""候補の順を保つ完全一致照合。色の除外と各組の不一致除外を前倒しする。"""
from __future__ import annotations
from typing import Any
import numpy as np
from src.ama_tsumo_seed_v5 import identify, excluded_mask
from src.tsumo_seed_identifier import Match, abs_pair_code
from src.tsumo_seed_generator import PAIR_COUNT


class ExactTable:
    def __init__(self, table: Any, colors: list) -> None:
        self.table = table
        self.colors = colors
        self.rows: np.ndarray | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.table, name)

    def identify_tape_view(self, observed: list, start: int = 0, max_mismatch: int = 0) -> Match:
        n = min(len(observed), PAIR_COUNT-start)
        if max_mismatch != 0 or n <= 0:
            return self.table.identify_tape_view(observed, start, max_mismatch)
        if self.rows is None:
            self.rows = np.flatnonzero(excluded_mask(self.table, self.colors))
        values = np.asarray(observed[:n], dtype=np.int64)
        codes = abs_pair_code(values[:, 0], values[:, 1])
        rows, tape = self.rows, self.table.tape_codes
        for offset, code in enumerate(codes):
            rows = rows[tape[rows, start+offset] == code]
            if not len(rows):
                break
        # 元実装の誤一致0件は、開始位置→種番号の昇順。順序を変えない。
        return Match(rows, np.zeros(len(rows), dtype=np.int64))


def identify_exact(table: Any, history: dict, colors: list) -> Any:
    # 特定・曖昧・向き不一致・範囲外の判定は元の関数をそのまま使う。
    return identify(ExactTable(table, colors), history, colors)
