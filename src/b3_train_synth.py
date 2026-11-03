"""テスト・試走用の合成学習行 (台帳 b3_rows_v1 の全列を持つ)。

信号は「自側の色ぷよ数 - 相手の色ぷよ数」を勝敗と弱く相関させる。実データではない。
原向き (orientation=0) と 1P/2P 交換複製 (orientation=1) を持ち、試合内の重み合計は 1 (各 0.5)。
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from src.b3_column_spec import COLUMN_SPECS, swap_sides

DEFAULT_SEED = 7


def make_rows(n_videos: int = 10, games_per_video: int = 3, rows_per_game: int = 40, seed: int = DEFAULT_SEED,
              signal: float = 1.2, noise_cols_nan_rate: float = 0.2, groups: Optional[Mapping[str, str]] = None
              ) -> tuple[pd.DataFrame, dict[str, str]]:
    """合成 DataFrame と video->group を返す。groups 指定が無ければ 1 動画 1 group。"""
    rng = np.random.default_rng(seed)
    out_rows: list[dict] = []
    gmap: dict[str, str] = {}
    for v in range(n_videos):
        vid = f"video_s{v:02d}"
        gmap[vid] = (groups or {}).get(vid, f"yt{v:02d}")
        for g in range(1, games_per_video + 1):
            winner = int(rng.integers(0, 2))
            t = np.sort(rng.uniform(0, 100, rows_per_game))
            for i in range(rows_per_game):
                row = _one_row(rng, vid, g, i, float(t[i]), winner, signal, noise_cols_nan_rate)
                out_rows.append(row)
    rows = out_rows
    # 重み: 試合内 held 比、原向き 0.5
    df = pd.DataFrame(rows)
    for (vid, g), idx in df.groupby(["video_id", "game_idx"]).groups.items():
        sub = df.loc[idx].sort_values("frame_idx")
        held = np.diff(sub["t_sec"].to_numpy(), append=sub["t_sec"].iloc[-1] + 0.03)
        w = held / held.sum() * 0.5
        df.loc[sub.index, "held_sec"] = held
        df.loc[sub.index, "weight"] = w
    mirrored = [swap_sides(r) for r in df.to_dict("records")]
    for r in mirrored:
        r["orientation"] = 1
        r["row_key"] = r["row_key"].rsplit("|o", 1)[0] + "|o1"
    full = pd.concat([df, pd.DataFrame(mirrored)], ignore_index=True)
    return _to_ledger_frame(full), gmap


def _one_row(rng: np.random.Generator, vid: str, g: int, i: int, t: float, winner: int, signal: float,
             nan_rate: float) -> dict:
    row: dict = {}
    c1 = float(rng.normal(25, 3)) + (4.0 if winner == 1 else -4.0) * signal
    c2 = float(rng.normal(25, 3))
    for s in COLUMN_SPECS:
        n = s.name
        if s.role == "audit" and n.startswith("mk_"):
            row[n] = 0
        elif s.role in ("audit",):
            row[n] = np.nan if n.startswith("av_") and n.endswith(("p1", "p2")) else 0.0
        elif s.dtype == "str":
            row[n] = ""
        elif s.dtype.startswith("i"):
            row[n] = 0
        else:
            row[n] = float(rng.normal()) if rng.random() > nan_rate else np.nan
    row.update(row_key=f"{vid}|g{g}|f{i}|o0", video_id=vid, game_idx=g, frame_idx=i, t_sec=t, orientation=0,
               trigger_mask=1, held_sec=0.03, weight=0.0, won_p1=float(winner), av_max=max(0.0, t - 0.01),
               av_frame=t, phase=int(rng.integers(0, 3)), mode_exchange=int(rng.random() < 0.4), mode_bits=0,
               cells_color_p1=c1, cells_color_p2=c2, cells_total_p1=c1 + 5, cells_total_p2=c2 + 5,
               elapsed_first_move=t, chain_active_p1=int(rng.random() < 0.1), chain_active_p2=int(rng.random() < 0.1),
               lean_row_p1=i, lean_row_p2=i)
    row["df_cells_color"] = c1 - c2
    return row


def _to_ledger_frame(df: pd.DataFrame) -> pd.DataFrame:
    """台帳の列順・整数列は pandas nullable (本番 to_frame に倣う)。"""
    out = df[[s.name for s in COLUMN_SPECS]].copy()
    for s in COLUMN_SPECS:
        if s.dtype in ("i4", "i1"):
            out[s.name] = pd.array(out[s.name], dtype="Int32" if s.dtype == "i4" else "Int8")
        elif s.dtype == "f4":
            out[s.name] = out[s.name].astype("float32")
    return out
