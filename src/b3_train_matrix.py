"""候補と R_cv が共有する float32 行列と、軽量な行索引。"""
from __future__ import annotations

from dataclasses import dataclass, field
import mmap
import tempfile
from typing import Sequence

import numpy as np
import pandas as pd

MATRIX_ATTR = "b3_matrix"
ROW_POSITION = "_b3_row_position"
RETAIN_COLUMNS = (
    "row_key", "video_id", "game_idx", "frame_idx", "t_sec", "orientation", "held_sec", "weight", "won_p1",
    "av_max", "mode_exchange", "phase", "elapsed_first_move", "cells_total_p1", "cells_total_p2",
    "mk_board_p1", "mk_board_p2",
    *(f"{stem}_{side}" for stem in ("chain_active", "ep_closed_n", "ep_honsen60_board",
                                    "ep_honsen60_formula", "ep_chain7plus") for side in ("p1", "p2")))


@dataclass(eq=False)
class SharedMatrix:
    values: np.ndarray
    columns: tuple[str, ...]
    aliases: dict[str, str] = field(default_factory=dict)

    def position(self, name: str) -> int:
        return self.columns.index(self.aliases.get(name, name))

    def __deepcopy__(self, memo: dict) -> SharedMatrix:
        """pandas の行抽出時も読み取り専用行列は複製しない。"""
        return self

    def release_pages(self) -> None:
        """読取済み一時ファイルのページを解放する（値はファイルに残す）。"""
        if isinstance(self.values, np.memmap) and hasattr(self.values._mmap, "madvise"):
            self.values._mmap.madvise(mmap.MADV_DONTNEED)


def allocate_matrix(rows: int, columns: int) -> np.memmap:
    """一時ファイルを裏に持つ列連続行列。マッピングの寿命終了でファイルも消える。"""
    with tempfile.TemporaryFile() as backing:
        return np.memmap(backing, dtype=np.float32, mode="w+", shape=(rows, columns), order="F")


def available_columns(df: pd.DataFrame) -> list[str]:
    store = df.attrs.get(MATRIX_ATTR)
    return list(dict.fromkeys([*df.columns, *(store.columns if store else ()),
                              *(store.aliases if store else ())]))


def column_values(df: pd.DataFrame, name: str) -> np.ndarray:
    """メタデータは元精度、共有特徴は従来と同じ float32 丸めで取得する。"""
    if name in df.columns:
        return df[name].to_numpy(dtype=np.float64, na_value=np.nan)
    store = df.attrs[MATRIX_ATTR]
    values = store.values[df[ROW_POSITION].to_numpy(), store.position(name)]
    store.release_pages()
    return values


def matrix_values(df: pd.DataFrame, columns: Sequence[str], dtype: type = np.float32) -> np.ndarray:
    """選択した行・列だけを1回確保する。float64化前に従来のfloat32丸めを保つ。"""
    out = np.empty((len(df), len(columns)), dtype=dtype)
    store = df.attrs.get(MATRIX_ATTR)
    rows = df[ROW_POSITION].to_numpy() if store else None
    for j, name in enumerate(columns):
        if store and (name in store.columns or name in store.aliases):
            out[:, j] = store.values[rows, store.position(name)]
            store.release_pages()
        else:
            out[:, j] = df[name].to_numpy(dtype=np.float32, na_value=np.nan)
    return out


def attach_matrix(df: pd.DataFrame, values: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
    """新規の軽量フレームに元行位置と読み取り専用行列を結び付ける。"""
    values.flags.writeable = False
    df[ROW_POSITION] = np.arange(len(df))
    df.attrs[MATRIX_ATTR] = SharedMatrix(values, tuple(columns))
    if isinstance(values, np.memmap):
        values.flush()
        df.attrs[MATRIX_ATTR].release_pages()
    return df


def prepare_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """通常の公開 API でも内側CVの全候補で1つの行列を共有する。"""
    if MATRIX_ATTR in df.attrs:
        return df
    values = matrix_values(df, columns)
    retained = [c for c in df.columns if c not in columns or c in RETAIN_COLUMNS]
    return attach_matrix(df[retained].copy(), values, columns)
