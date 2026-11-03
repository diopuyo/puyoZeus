"""撃った盤を画面の読みから直接取る (scene_68 の直し、PC1・user 2026-10-28 17:00 の決まり)。学習と配信で同じファイルを使う。

user の確認: 置いた組は撃つ直前の NEXT、連鎖は置いた時だけ始まる、掛け算式は 1 段目で必ず出る (数字は合計済み)。置き場所の推測はしない。
決まり:
- 側ごとに、毎コマの画面の読みの盤 raw_cnn_board (13×6) を直近 KEEP コマ持つ。一番上の段 (隠し段) の 10 以上は 0、
  それ以外で 0〜5・9 でない値は不明としておじゃま (9) に置く (計算器の決まり)。
- 連鎖の掛け算式が最初に valid になったコマの時刻 t_formula と、その式の product (1 段目の得点) で、t_formula より前の直近 BACK コマ
  (t−1 … t−6) の盤を新しい順に計算し、1 段目の得点が product とぴったり一致した最初の盤を「撃った盤」とする。無ければ None。
- 計算は呼ぶ側が渡す scorer(board 13×6 int) -> 段ごとの得点の列 (B3 は ChainSimulator、EX/retR は native.simulate_chain_trace_py)。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

KEEP = 12
BACK = 6
HIDDEN_ROW_MARK_MIN = 10
OJAMA = 9


def clean(raw: Any) -> Optional[np.ndarray]:
    """raw_cnn_board → 13×6 の int8。読めなければ None。"""
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get('grid', raw.get('array', raw.get('board')))
        if isinstance(raw, dict):
            raw = raw.get('array')
    try:
        a = np.asarray(raw, dtype=np.int64)
    except (TypeError, ValueError):
        return None
    if a.size != 78:
        return None
    a = a.reshape(13, 6).copy()
    a[0][a[0] >= HIDDEN_ROW_MARK_MIN] = 0
    a[(a != 0) & ~((a >= 1) & (a <= 5)) & (a != OJAMA)] = OJAMA
    return a.astype(np.int8)


@dataclass(frozen=True)
class FireBoard:
    t_formula: float
    product1: int
    board: np.ndarray          # 13×6 int8
    steps: tuple[int, ...]     # 計算の段ごとの得点
    back: int                  # t_formula から何コマ前の盤か (1〜BACK)


class RawHistory:
    """1 側の直近の画面の読みの盤。毎コマ push する。"""

    def __init__(self) -> None:
        self.frames: deque = deque(maxlen=KEEP)

    def push(self, t: float, raw: Any) -> None:
        self.frames.append((float(t), clean(raw)))

    def clear(self) -> None:
        self.frames.clear()

    def find(self, t_formula: float, product1: int, scorer: Callable[[np.ndarray], list[int]]) -> Optional[FireBoard]:
        cands = [(t, b) for t, b in self.frames if t < t_formula - 1e-9][-BACK:]
        for k, (_, board) in enumerate(reversed(cands), 1):
            if board is None:
                continue
            steps = scorer(board)
            if steps and int(steps[0]) == int(product1):
                return FireBoard(float(t_formula), int(product1), board, tuple(int(x) for x in steps), k)
        return None


class FireBoardTracker:
    """両側の毎コマの観測から、連鎖の掛け算式が最初に valid になったコマを見つけて撃った盤を取る (EX/retR 用)。
    1 つの連鎖では最初の valid だけを使う。式が REARM_SEC 以上 valid でなければ次の連鎖とみなす。"""
    REARM_SEC = 1.5

    def __init__(self, scorer: Callable[[np.ndarray], list[int]]) -> None:
        self.scorer = scorer
        self.hist = [RawHistory(), RawHistory()]
        self.last_valid = [None, None]
        self.found: list[Optional[FireBoard]] = [None, None]
        self.game = None

    def observe(self, t: float, game: int, raw: list[dict]) -> None:
        if game != self.game:
            self.game = game
            for h in self.hist:
                h.clear()
            self.last_valid, self.found = [None, None], [None, None]
        for i, side in enumerate(raw[:2]):
            form = side.get('formula') or {}
            valid = bool(form.get('valid')) and form.get('product') is not None
            if valid:
                fresh = self.last_valid[i] is None or t - self.last_valid[i] >= self.REARM_SEC
                if fresh:
                    fb = self.hist[i].find(t, int(form['product']), self.scorer)
                    self.found[i] = fb if fb is not None else FireBoard(t, int(form['product']), None, (), 0)
                self.last_valid[i] = t
            self.hist[i].push(t, side.get('raw_cnn_board'))

    def for_chain(self, side: int, started: float, now: float) -> Optional[FireBoard]:
        """その側の今の連鎖 (始まり started) の撃った盤。式がまだ出ていない・取れなかった時は None。"""
        fb = self.found[side]
        if fb is None or fb.board is None or fb.t_formula < started - 0.5 or fb.t_formula > now + 1e-9:
            return None
        return fb


# --- 依頼 A (PC1・user 2026-10-28): 側ごとの連鎖の予測を 1 つにする。B3 と EX/retR が同じクラスを同じ入力で使う ---
# 決まり:
# - 連鎖の識別子 = 連鎖の最初の式 (式が REARM_SEC 以上 valid でなかった後の最初の valid) の時刻。段が進んでも変えない。
# - 段: 連鎖の中の valid の塊 (間が BLOCK_GAP_SEC を超えたら次の塊) の最初の product。前の段と同じ値の塊は、表示が切れて同じ段が
#   2 回数えられたものとして 1 段にまとめる。
# - 1 段目: 式の直前 BACK コマから、計算の 1 段目の得点が式と合う盤 = 撃った盤 (FireBoardTracker と同じ)。
# - A2 取り直し: 段 k の式が予測の段 k と合わない (予測が先に終わった時を含む) なら、段 k の式の直前 BACK コマから、
#   k 段目の得点表で計算した 1 段目が式 k と合う盤を取り、残りの段を計算し直す。
# - A3 救済: 取り直せなければ捨てずに「観測済みの段 + それまでの予測の残りの段」。
# - 計算は native.simulate_chain_trace_py (puyo_core)。得点表は通 (連鎖・色数・連結のボーナス)。off=0 なら native の得点と同じ。

CHAIN_BONUS = (0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512)
COLOR_BONUS = (0, 0, 3, 6, 12, 24)
GROUP_BONUS = {4: 0, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7}
BLOCK_GAP_SEC = 0.3


class NativeTrace:
    """盤 (13×6) → (段ごとの [(色, 個数), ...], 連鎖の後の盤)。native を渡さなければ puyo_core を読む。
    pickle できる (配信の B3 は状態を pickle して別の process に送る。2026-10-28 PC1: 中の関数では送れず B3 が止まった)。
    pickle ではモジュールとキャッシュを送らず、受け側で最初に呼ばれた時に puyo_core を読み直す。計算は前の関数と同じ。"""
    CACHE_MAX = 4096

    def __init__(self, native: Any = None) -> None:
        self._native, self._cache = native, {}

    def __getstate__(self) -> dict:
        return {}

    def __setstate__(self, state: dict) -> None:
        self._native, self._cache = None, {}

    def __call__(self, board: np.ndarray):
        if self._native is None:
            import puyo_core as native   # type: ignore[import-not-found]
            self._native = native
        key = board.tobytes()
        if key not in self._cache:
            tr = self._native.simulate_chain_trace_py(list(map(int, board.ravel())), True)
            groups = [[(int(g[0]), len(g[1])) for g in s[2]] for s in tr]
            final = np.asarray(tr[-1][1], dtype=np.int8).reshape(13, 6) if tr else board.copy()
            if len(self._cache) > self.CACHE_MAX:
                self._cache.clear()
            self._cache[key] = (groups, final)
        return self._cache[key]


def native_trace(native: Any = None) -> NativeTrace:
    """盤 (13×6) → (段ごとの [(色, 個数), ...], 連鎖の後の盤) の計算 (pickle できる NativeTrace)。"""
    return NativeTrace(native)


def step_scores(groups: list[list[tuple[int, int]]], off: int = 0) -> list[int]:
    """段ごとの得点。off は何段目から始まった連鎖か (0 = 1 段目から)。"""
    out = []
    for i, gs in enumerate(groups):
        n = sum(x for _, x in gs)
        bonus = CHAIN_BONUS[min(i+off, len(CHAIN_BONUS)-1)] + COLOR_BONUS[len({c for c, _ in gs})] + sum(GROUP_BONUS.get(x, 10) for _, x in gs)
        out.append(10*n*max(1, bonus))
    return out


@dataclass
class Forecast:
    chain_id: float                    # 連鎖の最初の式の時刻 (段が進んでも変わらない)
    observed: tuple[int, ...]          # 観測済みの段の得点 (式)
    steps: tuple[int, ...]             # 観測済み + 予測の残りの段の得点。予測が無ければ observed と同じ
    source: str                        # 'fire_board' / 'refind' / 'rescue' / 'none'
    final_board: Optional[np.ndarray]  # 計算の最終盤 (連鎖の後の盤)。予測が無ければ None
    last_valid_t: float
    refinds: int = 0
    rescues: int = 0

    @property
    def total(self) -> int:
        return int(sum(self.steps))

    @property
    def remaining_steps(self) -> int:
        return max(0, len(self.steps)-len(self.observed))

    @property
    def has_prediction(self) -> bool:
        return self.source != 'none'


def _dump(name: str, t: float, fc: Forecast) -> None:
    """確かめ用 (PUYO_FORECAST_DUMP=<ファイル> の時だけ): 予測が変わるたびに 1 行。B3 と EX/retR の一致の確かめに使う。"""
    import json
    import os
    path = os.environ.get('PUYO_FORECAST_DUMP')
    if path:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(side=name, t=round(t, 4), id=fc.chain_id, obs=list(fc.observed), steps=list(fc.steps), src=fc.source)) + '\n')


SPLIT_ENV = 'PUYO_CHAIN_SPLIT_PLACE'
# 次の回 2 (PC1 10/30): 自分の連鎖中は操作中のツモが無い (user の決まり)。式が出ている間 (最後の有効な式から 0.5 秒未満) の
# tsumo_count の増えは置きと数えない。60 本で空振り 148 → 4 / 8,747、録画で確かめた種類 473 → 423 / 605 (f4_split_why.py の gap05)。
PLACE_GAP_SEC = 0.5


def split_enabled() -> bool:
    """依頼 §4 (PC1 10/30、RESULT_UNDERPRED 追記 2): 最後の式の後にその側の置き (観測の tsumo_count の増え) があれば、
    1.5 秒以内の式でも別の連鎖にする (b3_score_ledger._splits_chain と同じ考え)。既定 OFF、'1' で ON。"""
    import os
    return os.environ.get(SPLIT_ENV, '0') == '1'


class ChainForecast:
    """1 側の連鎖の予測。毎コマ observe(t, formula, raw_cnn_board) を入れ、current() で今の連鎖の予測を読む。"""
    REARM_SEC = 1.5

    def __init__(self, trace: Optional[Callable] = None, name: str = '') -> None:
        from src.chain_continuity import end_enabled, recheck_enabled
        self.end_formula, self.recheck = end_enabled(), recheck_enabled()
        self.trace = trace if trace is not None else native_trace()
        self.name = name
        self.hist = RawHistory()
        self.split = split_enabled()   # 依頼 §4 (旗 ON の時だけ)
        self.reset()

    def reset(self) -> None:
        from src.chain_continuity import ChainBoundary
        self.boundary = ChainBoundary() if self.end_formula else None
        self.hist.clear()
        self.last_valid: Optional[float] = None
        self.block_last: Optional[float] = None
        self.fc: Optional[Forecast] = None
        self._pred: Optional[list[int]] = None
        self.tsumo: Optional[int] = None        # 依頼 §4: 前のコマの tsumo_count
        self.placed_t: Optional[float] = None   # 依頼 §4: 最後に置いた (tsumo_count が増えた) 時刻

    def _find(self, t: float, product: int, off: int) -> Optional[np.ndarray]:
        cands = [(tt, b) for tt, b in self.hist.frames if tt < t-1e-9][-BACK:]
        for _, b in reversed(cands):
            if b is None:
                continue
            groups, _ = self.trace(b)
            s = step_scores(groups, off)
            if s and s[0] == product:
                return b
        return None

    def observe(self, t: float, formula: Optional[dict], raw_board: Any, side: Optional[dict] = None) -> None:
        if self.boundary is not None:
            self.boundary.observe(t, dict(side or {}, formula=formula, raw_cnn_board=raw_board))
        if self.split and side:
            tc = side.get('tsumo_count')
            if tc is not None:
                if self.tsumo is not None and int(tc) > self.tsumo and (self.last_valid is None or t-self.last_valid >= PLACE_GAP_SEC):
                    self.placed_t = float(t)   # 式が出ている間 (最後の有効な式から PLACE_GAP_SEC 未満) の増えは連鎖の途中の読みの揺れ
                self.tsumo = int(tc)
        form = formula or {}
        valid = bool(form.get('valid')) and form.get('product') is not None
        if valid:
            p = int(form['product'])
            expired = not self.end_formula and self.last_valid is not None and t-self.last_valid >= self.REARM_SEC
            separated = self.boundary is not None and self.fc is not None and self.boundary.t > self.fc.chain_id
            if self.last_valid is None or expired or separated or self._placed_since_last():
                self._new_chain(t, p)
                _dump(self.name, t, self.fc)
            elif t-self.block_last > BLOCK_GAP_SEC and p != self.fc.observed[-1]:
                self._new_step(t, p)
                _dump(self.name, t, self.fc)
            self.last_valid = self.block_last = t
            self.fc.last_valid_t = t
        if self.recheck:
            self._recheck(t, raw_board, side)
        self.hist.push(t, raw_board)

    def _recheck(self, t: float, raw: Any, side: Optional[dict]) -> None:
        from src.chain_continuity import recheck_board
        board = recheck_board(self, t, raw, side)
        if board is None:
            return
        groups, final = self.trace(board)
        scores = step_scores(groups)
        self._pred = scores
        self.fc.steps, self.fc.final_board = tuple(scores), final
        self.fc.source = 'fire_recheck'

    def _new_chain(self, t: float, p: int) -> None:
        b = self._find(t, p, 0)
        if b is None:
            self._pred, src, final = None, 'none', None
        else:
            groups, final = self.trace(b)
            self._pred, src = step_scores(groups, 0), 'fire_board'
        self.fc = Forecast(round(t, 6), (p,), tuple(self._pred) if self._pred else (p,), src, final, t)

    def _new_step(self, t: float, p: int) -> None:
        fc = self.fc
        obs = list(fc.observed)+[p]
        k = len(obs)
        pred, src, final, refinds, rescues = self._pred, fc.source, fc.final_board, fc.refinds, fc.rescues
        if pred is None or len(pred) < k or pred[k-1] != p:
            b = self._find(t, p, k-1)
            if b is not None:   # A2 取り直し
                groups, final = self.trace(b)
                pred, src, refinds = obs[:-1]+step_scores(groups, k-1), 'refind', refinds+1
            elif pred is not None:   # A3 救済: 観測済み + それまでの予測の残り
                pred, src, rescues = obs+list(pred[k:]), 'rescue', rescues+1
        self._pred = pred
        steps = tuple(obs+list(pred[k:])) if pred is not None else tuple(obs)
        self.fc = Forecast(fc.chain_id, tuple(obs), steps, src if pred is not None else 'none', final, t, refinds, rescues)

    def _placed_since_last(self) -> bool:
        """依頼 §4 (旗 ON の時だけ): 最後の式より後にこの側が置いたか (= 前の連鎖は終わっている)。"""
        return not self.end_formula and self.split and self.placed_t is not None and self.last_valid is not None and self.placed_t > self.last_valid

    def current(self, now: Optional[float] = None) -> Optional[Forecast]:
        """今の連鎖の予測。最後の式から REARM_SEC 以上たったら (連鎖が終わった後) None。now を渡さなければ時刻を見ない。
        依頼 §4 の旗 ON では、最後の式の後にこの側が置いていても None (次の式は別の連鎖)。"""
        if self.fc is None:
            return None
        if self.boundary is not None and self.boundary.t > self.fc.chain_id:
            return None
        if not self.end_formula and now is not None and now-self.fc.last_valid_t >= self.REARM_SEC:
            return None
        if self._placed_since_last():
            return None
        return self.fc


class PairForecast:
    """両側の ChainForecast (EX/retR 用)。毎コマ observe(t, game, raw[2]) を入れる。試合が変わったら両側を捨てる。"""

    def __init__(self, trace: Optional[Callable] = None) -> None:
        trace = trace if trace is not None else native_trace()
        self.sides = [ChainForecast(trace, '1P'), ChainForecast(trace, '2P')]
        self.game = None

    def observe(self, t: float, game: int, raw: list[dict]) -> None:
        if game != self.game:
            self.game = game
            for f in self.sides:
                f.reset()
        for f, side in zip(self.sides, raw[:2]):
            f.observe(t, side.get('formula'), side.get('raw_cnn_board'), side)

    def current(self, side: int, now: float) -> Optional[Forecast]:
        return self.sides[side].current(now)
