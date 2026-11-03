"""B3行の参照列から本番M0入力を復元する（特徴行の再生成はしない）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.advantage_m0_current_cnn_v1 import (
    BOARD_ROWS, BOARD_COLS, QUEUE_SLOT_COUNT, board_categories_from_raw, queue_categories_from_raw,
)
from src.b3_source import iter_frames, load_lean, resolve_attempt, sha256_file

FOLD_SHA256 = "22a058395eea9a315c27f39d8b7d7cf781d201c24fa74549eb240cf065a59213"
DEFAULT_FOLDS = Path(__file__).resolve().parents[1] / "docs/b3_prereg_fill/outer_folds_rows_v2.tsv"
SIDES = ("p1", "p2")
INPUT_COLUMNS = ("row_key", "video_id", "game_idx", "frame_idx", "t_sec", "orientation", "won_p1",
                 *(f"{stem}_{side}" for stem in ("av_board", "av_queue", "lean_row", "cells_total") for side in SIDES))
TIME_TOL = 1e-4
# leanの秒は小数3桁丸め。取得フレーム時刻で頭打ちした差とfloat32丸めを許容する。
QUEUE_TIME_TOL = 1e-3 + TIME_TOL


def fold_maps(path: Path = DEFAULT_FOLDS) -> tuple[dict[str, str], dict[str, int]]:
    """事前登録正本をハッシュ検査し、動画groupと外側foldを返す。"""
    if sha256_file(path) != FOLD_SHA256:
        raise ValueError("外側fold正本のSHA256が不一致")
    table = pd.read_csv(path, sep="\t")
    groups, folds = {}, {}
    for row in table.itertuples(index=False):
        for video in row.videos.split(","):
            if video in groups:
                raise ValueError("外側foldの動画重複")
            groups[video], folds[video] = row.group, int(row.fold)
    return groups, folds


def checked_record(path: Path, b2_root: Path) -> Path:
    """行生成時のattemptと元記録を固定し、別の取り直しとの混在を拒む。"""
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    source = resolve_attempt(path.stem, b2_root)
    if source.attempt != manifest["attempt"]:
        raise ValueError("B3行とB2採用attemptが不一致")
    for name, key in (("observations.jsonl.gz", "observations_sha256"), ("lean.npz", "lean_sha256")):
        if sha256_file(source.record / name) != manifest["inputs"][key]:
            raise ValueError(f"B3入力ハッシュが不一致: {name}")
    return source.record


def board_lookup(df: pd.DataFrame, record: Path) -> dict[tuple[int, float, int], np.ndarray]:
    """取得時刻と試合を完全一致させる。左右交換行は元の側へ戻して参照する。"""
    wanted = set()
    for row in df.itertuples(index=False):
        if row.orientation not in (0, 1):
            raise ValueError("orientationが0/1でない")
        for side, name in enumerate(SIDES):
            at = float(getattr(row, f"av_board_{name}"))
            if not np.isfinite(at) or at > row.t_sec + TIME_TOL:
                raise ValueError("M0盤面が欠測または未来参照")
            wanted.add((row.game_idx, at, side ^ int(row.orientation)))
    found = {}
    for frame in iter_frames(record, int(df.frame_idx.max())):
        for side, raw in enumerate(frame["sides"]):
            key = (frame["game_idx"], frame["t_sec"], side)
            if key in wanted and raw.get("confirmed_board") is not None:
                found[key] = board_categories_from_raw(np.asarray(raw["confirmed_board"]))
    if wanted - found.keys():
        raise ValueError("B3の盤面取得時刻に一致するconfirmed_boardが無い")
    return found


def recover_inputs(df: pd.DataFrame, record: Path) -> tuple[np.ndarray, np.ndarray]:
    """13×6の上下左右を保持し、9→6/10→7、queue欠測→0の本番変換を使う。"""
    found, lean = board_lookup(df, record), load_lean(record)
    boards = np.empty((len(df), len(SIDES), BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    queues = np.zeros((len(df), len(SIDES), QUEUE_SLOT_COUNT), dtype=np.int8)
    for index, row in enumerate(df.itertuples(index=False)):
        for side, name in enumerate(SIDES):
            original = side ^ int(row.orientation)
            at = float(getattr(row, f"av_board_{name}"))
            board = found[(row.game_idx, at, original)]
            if np.count_nonzero((board >= 1) & (board <= 6)) != getattr(row, f"cells_total_{name}"):
                raise ValueError("復元盤面とB3盤面総量が不一致")
            boards[index, side] = board
            pointer = int(getattr(row, f"lean_row_{name}"))
            if pointer == -1:
                continue
            if pointer < 0 or pointer >= len(lean.frame):
                raise ValueError("lean_rowが範囲外")
            if lean.frame[pointer] > row.frame_idx or lean.side[pointer] != f"{original + 1}P":
                raise ValueError("queueが未来参照または側不一致")
            queues[index, side] = queue_categories_from_raw(lean.queue[pointer])
            av = float(getattr(row, f"av_queue_{name}"))
            if queues[index, side].any() and (not np.isfinite(av) or av > row.t_sec + TIME_TOL
                    or abs(av - min(lean.t_sec[pointer], row.t_sec)) > QUEUE_TIME_TOL):
                raise ValueError(f"queue取得時刻が不一致: row={index} side={name} av={av} "
                                 f"lean_t={lean.t_sec[pointer]} t={row.t_sec}")
    return boards, queues
