"""B3 時間軸の物理量テーブル (1手時間=高さ別、連鎖の段時間・終端)。

設計 RETRAIN_DESIGN §2.2: 1手時間は置く列の高さ・盤面最大高さで層別し、中央値版と下位10%版を持つ。
推定器 (fit_*) と適用 (TimingTable) を分離し、学習 fold 内の推定値を差し込める。
今回の既定テーブルは全データ推定 (fit_scope=all_data_not_fold_safe)。外側 test を含むため
学習評価には使わず、fold 内で fit_hand_table を呼び直して TimingTable を差し替える。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Iterable, Optional

import numpy as np

SEC_PER_HAND_DEFAULT = 0.733          # src/indicators_v2.py SEC_PER_HAND (中央値, 初期参考)
HAND_FAST_RATIO_DEFAULT = 0.6         # 下位10%表が無いときの暫定比 (テーブル未fit用・fallback記録対象)
STAGE_SEC_DEFAULT = 1.17              # src/exchange_prefire_stage_timeout.py STAGE_INTERVAL_SEC
TAIL_SEC_DEFAULT = 1.17               # 最終段後の操作復帰までの暫定値 (B2 較正で置換)
HAND_FAST_QUANTILE = 0.10
HAND_SAMPLE_MAX_SEC = 8.0             # これ以上の間隔は思考停止/連鎖とみなし除外
MIN_BUCKET_N = 30
MAX_HEIGHT_EDGES: tuple[int, ...] = (6, 9, 11)   # <6, 6-8, 9-10, >=11
COL_HEIGHT_EDGES: tuple[int, ...] = (4, 8)       # <4, 4-7, >=8
BOARD_ROWS = 13
PLACED_CELLS = 2


def column_heights(grid: np.ndarray) -> np.ndarray:
    """各列の高さ (最上段の非空セルまでの段数)。grid は 13x6。"""
    occupied = grid != 0
    top = np.where(occupied.any(axis=0), occupied.argmax(axis=0), BOARD_ROWS)
    return (BOARD_ROWS - top).astype(np.int64)


def height_bucket(value: float, edges: tuple[int, ...]) -> int:
    return int(np.searchsorted(np.asarray(edges), value, side="right"))


@dataclass(frozen=True)
class TimingTable:
    """適用側テーブル。キーは 'm{b}' (最大高さ) / 'm{b}c{c}' (+置く列高さ) / 'all'。"""

    hand_median: dict[str, float] = field(default_factory=dict)
    hand_fast: dict[str, float] = field(default_factory=dict)
    hand_n: dict[str, int] = field(default_factory=dict)
    stage_sec: float = STAGE_SEC_DEFAULT
    tail_sec: float = TAIL_SEC_DEFAULT
    fit_scope: str = "default_constants_not_fitted"
    meta: dict = field(default_factory=dict)

    def hand_sec(self, max_height: Optional[float], col_height: Optional[float] = None
                 ) -> tuple[float, float, str]:
        """(中央値版, 下位10%版, 使ったキー)。未fit/欠測は既定定数 (キーに default)。"""
        keys = []
        if max_height is not None and np.isfinite(max_height):
            mb = height_bucket(max_height, MAX_HEIGHT_EDGES)
            if col_height is not None and np.isfinite(col_height):
                keys.append(f"m{mb}c{height_bucket(col_height, COL_HEIGHT_EDGES)}")
            keys.append(f"m{mb}")
        keys.append("all")
        for key in keys:
            if key in self.hand_median and self.hand_n.get(key, 0) >= MIN_BUCKET_N:
                return self.hand_median[key], self.hand_fast[key], key
        return SEC_PER_HAND_DEFAULT, SEC_PER_HAND_DEFAULT * HAND_FAST_RATIO_DEFAULT, "default"

    def to_json(self) -> str:
        return json.dumps({"hand_median": self.hand_median, "hand_fast": self.hand_fast, "hand_n": self.hand_n,
                           "stage_sec": self.stage_sec, "tail_sec": self.tail_sec,
                           "fit_scope": self.fit_scope, "meta": self.meta}, ensure_ascii=False, indent=1)

    @staticmethod
    def from_json(text: str) -> "TimingTable":
        d = json.loads(text)
        return TimingTable(d["hand_median"], d["hand_fast"], {k: int(v) for k, v in d["hand_n"].items()},
                           float(d["stage_sec"]), float(d["tail_sec"]), d["fit_scope"], d.get("meta", {}))


def collect_hand_samples(grids: np.ndarray, sides: np.ndarray, games: np.ndarray,
                         t_sec: np.ndarray) -> np.ndarray:
    """lean 行 (側別に時刻順) から 1手間隔サンプル (N,3)=[間隔秒, 前盤面最大高さ, 置いた列の高さ] を作る。

    盤面の非空セルがちょうど2個増えた連続行だけを使う (消去・おじゃま着弾を含む間隔を除外)。
    """
    out: list[tuple[float, float, float]] = []
    order = np.lexsort((t_sec, games, sides))
    prev = None
    for i in order:
        if prev is not None and sides[prev] == sides[i] and games[prev] == games[i]:
            dt = float(t_sec[i] - t_sec[prev])
            before, after = grids[prev], grids[i]
            gain = int((after != 0).sum() - (before != 0).sum())
            if 0 < dt < HAND_SAMPLE_MAX_SEC and gain == PLACED_CELLS:
                h0, h1 = column_heights(before), column_heights(after)
                changed = h1 > h0
                if changed.any():
                    out.append((dt, float(h0.max()), float(h0[changed].max())))
        prev = i
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def fit_timing_table(hand_samples: np.ndarray, stage_gaps: Iterable[float] = (),
                     tails: Iterable[float] = (), fit_scope: str = "all_data_not_fold_safe",
                     meta: Optional[dict] = None) -> TimingTable:
    """1手時間 (高さ別 中央値/下位10%) と段時間・終端を推定する。fold 内サンプルだけを渡せば fold 内推定。"""
    med: dict[str, float] = {}
    fast: dict[str, float] = {}
    cnt: dict[str, int] = {}
    s = np.asarray(hand_samples, dtype=np.float64).reshape(-1, 3)

    def put(key: str, mask: np.ndarray) -> None:
        if mask.sum() > 0:
            v = s[mask, 0]
            med[key], fast[key], cnt[key] = float(np.median(v)), float(np.quantile(v, HAND_FAST_QUANTILE)), int(mask.sum())
    if len(s):
        mb = np.searchsorted(np.asarray(MAX_HEIGHT_EDGES), s[:, 1], side="right")
        cb = np.searchsorted(np.asarray(COL_HEIGHT_EDGES), s[:, 2], side="right")
        put("all", np.ones(len(s), dtype=bool))
        for b in np.unique(mb):
            put(f"m{b}", mb == b)
            for c in np.unique(cb[mb == b]):
                put(f"m{b}c{c}", (mb == b) & (cb == c))
    gaps, tl = np.asarray(list(stage_gaps), dtype=float), np.asarray(list(tails), dtype=float)
    m = dict(meta or {})
    m.update(n_hand=int(len(s)), n_stage_gaps=int(len(gaps)), n_tails=int(len(tl)))
    return TimingTable(med, fast, cnt, float(np.median(gaps)) if len(gaps) else STAGE_SEC_DEFAULT,
                       float(np.median(tl)) if len(tl) else TAIL_SEC_DEFAULT, fit_scope, m)
