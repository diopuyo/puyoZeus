"""B3 学習行 (b3_rows_v1 の parquet) の読込と、学習前の機械検査 (事前登録 §3.1 のうち行だけで検査できるもの)。

読込は必要列だけを pyarrow から float32/Int で取る (全 58 動画は大きいので列を絞れるようにする)。
`<video>.done` マーカーが無い動画は読まない (未完ファイルを採用しない)。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.b3_train_config import MIN_GROUPS_TOTAL

AV_TOLERANCE = 1e-4          # b3_rows.AV_EPS と同じ (lean.npz の float32 丸め)
WEIGHT_SUM_TOL = 1e-6
KEY_COLUMNS = ("row_key", "video_id", "game_idx", "frame_idx", "t_sec", "orientation", "held_sec", "weight", "won_p1",
               "av_max")


def list_complete_parquets(rows_dir: Path, videos: Optional[Iterable[str]] = None) -> dict[str, Path]:
    """完了マーカー (<video>.done) のある parquet だけ。videos 指定時は欠落を例外にする。"""
    out = {}
    for done in sorted(rows_dir.glob("*.done")):
        vid = done.stem
        pq = rows_dir / f"{vid}.parquet"
        if pq.exists():
            out[vid] = pq
    if videos is not None:
        missing = [v for v in videos if v not in out]
        if missing:
            raise FileNotFoundError(f"完了済み parquet が無い動画: {missing}")
        out = {v: out[v] for v in videos}
    return out


def load_rows(paths: Mapping[str, Path], columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """動画ごとの parquet を連結する。columns を渡せばその列 + キー列だけ読む。"""
    cols = None if columns is None else sorted(set(columns) | set(KEY_COLUMNS))
    parts = [pd.read_parquet(p, columns=cols) for _, p in sorted(paths.items())]
    if not parts:
        raise ValueError("読む parquet が無い")
    return pd.concat(parts, ignore_index=True)


def load_training_rows(paths: Mapping[str, Path], columns: Sequence[str]) -> pd.DataFrame:
    """必要列だけを動画単位で読み、全体の特徴行列は float32 で1回だけ確保する。"""
    import pyarrow.parquet as pq
    from src.b3_train_matrix import RETAIN_COLUMNS, allocate_matrix, attach_matrix
    from src.b3_train_rcv import d_column_map
    if not paths:
        raise ValueError("読む parquet が無い")
    files = [pq.ParquetFile(p) for _, p in sorted(paths.items())]
    names = list(dict.fromkeys([*columns, *(c for c in d_column_map().values() if c is not None)]))
    retained = list(RETAIN_COLUMNS)
    requested = sorted(set(names) | set(retained))
    values = allocate_matrix(sum(f.metadata.num_rows for f in files), len(names))
    parts, offset = [], 0
    for file in files:
        # バッチ上限で圧縮率や動画長に依存した読込ピークを防ぐ。
        for batch in file.iter_batches(batch_size=READ_BATCH_ROWS, columns=requested, use_threads=False):
            frame = batch.to_pandas()
            end = offset + len(frame)
            for j, name in enumerate(names):
                values[offset:end, j] = frame[name].to_numpy(dtype=np.float32, na_value=np.nan)
            parts.append(frame[retained].copy())
            offset = end
            del frame, batch
    return attach_matrix(pd.concat(parts, ignore_index=True), values, names)


READ_BATCH_ROWS = 16_384


def precheck_rows(df: pd.DataFrame, group_of_video: Mapping[str, str], min_groups: int = MIN_GROUPS_TOTAL,
                  enforce_min_groups: bool = True) -> dict:
    """§3.1 の行単体検査。返す値は違反件数 (0 が合格)。None は行だけでは検査できず未実施。

    available_at: av_max <= t_sec (+許容)。weight: 試合ごと合計 1。複製: 原向き/複製が揃い won_p1 が補になる。
    """
    n = len(df)
    dup = int(df["row_key"].duplicated().sum())
    av = df["av_max"].to_numpy(dtype=np.float64, na_value=np.nan)
    t = df["t_sec"].to_numpy(dtype=np.float64)
    av_viol = int(np.count_nonzero(np.isfinite(av) & (av > t + AV_TOLERANCE)))
    av_missing = int(np.count_nonzero(~np.isfinite(av)))
    wsum = df.groupby(["video_id", "game_idx"])["weight"].sum()
    w_bad = int((np.abs(wsum - 1.0) > WEIGHT_SUM_TOL).sum())
    y_nan = int(df["won_p1"].isna().sum())
    unmapped = sorted(set(df["video_id"].unique()) - set(group_of_video))
    groups = {group_of_video[v] for v in df["video_id"].unique() if v in group_of_video}
    orient = df.groupby(["video_id", "game_idx"])["orientation"].nunique()
    mirror_missing = int((orient < 2).sum())
    res = {"rows": n, "games": int(len(wsum)), "videos": int(df["video_id"].nunique()), "groups": len(groups),
           "row_key_unique": dup, "available_at": av_viol, "available_at_not_recorded": av_missing,
           "weight_sum_games_bad": w_bad, "label_nan": y_nan, "videos_without_group": len(unmapped),
           "games_missing_mirror": mirror_missing,
           "mirror_label_complement": _mirror_label_violations(df)}
    if enforce_min_groups and len(groups) < min_groups:
        res["group_count_below_minimum"] = len(groups)
    res["violations_total"] = int(sum(v for k, v in res.items() if k in (
        "row_key_unique", "available_at", "available_at_not_recorded", "weight_sum_games_bad", "label_nan",
        "videos_without_group", "games_missing_mirror", "mirror_label_complement", "group_count_below_minimum")))
    return res


def _mirror_label_violations(df: pd.DataFrame) -> int:
    """試合内で原向きの won_p1 と複製の won_p1 の和が 1 になっているか (側交換でラベルが補になる)。"""
    g = df.groupby(["video_id", "game_idx", "orientation"])["won_p1"].agg(["min", "max"])
    bad = 0
    for (v, gm), sub in g.groupby(level=[0, 1]):
        o = sub.droplevel([0, 1])
        if set(o.index) != {0, 1}:
            continue
        if o.loc[0, "min"] != o.loc[0, "max"] or o.loc[1, "min"] != o.loc[1, "max"] or \
                abs(float(o.loc[0, "min"]) + float(o.loc[1, "min"]) - 1.0) > 1e-9:
            bad += 1
    return bad


def load_group_map(targets_json: Path, merges: Sequence[Sequence[str]] = ()) -> dict[str, str]:
    """B2 targets.json (video_id, youtube_id) から group 対応を作る。"""
    from src.b3_train_folds import video_to_group
    data = json.loads(targets_json.read_text(encoding="utf-8"))
    return video_to_group({t["video_id"]: t["youtube_id"] for t in data}, merges)
