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
    hidden: tuple = ()                 # 依頼 F4: 撃った盤の隠し段に推定から入れた (列, 色)。旗 OFF・入れなければ空
    # 依頼 F4 の確率版 (旗 = 2 の時だけ。それ以外は None・0): 隠し段の確率で重みをつけた総得点の期待値と上振れ幅 (上位 10% − 一番ありそう)
    exp_total: Optional[float] = None
    upside: Optional[int] = None
    n_comb: int = 0
    combos: tuple = ()                 # (重み, 段ごとの得点, 最終盤) の組合せ。連鎖が始まった後の絞りに使う
    comb_mass: Optional[float] = None  # 撃った時に残した組合せの確率の和 (上限で切った分は 1 より小さい)
    comb_ms: Optional[float] = None    # 撃った時の組合せの計算時間 (ms)
    lower_bound: bool = False

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


HIDDEN_ENV = 'PUYO_FORECAST_HIDDEN'


def hidden_enabled() -> bool:
    """依頼 F4: 撃った盤の隠し段 (13 段目) に推定のぷよを入れるか。既定 OFF、'1' で ON (確実な所だけ)。"""
    import os
    return os.environ.get(HIDDEN_ENV, '0') == '1'


def hidden_prob_enabled() -> bool:
    """依頼 F4 の確率版: '2' で確実な所は '1' と同じに入れ、不確実な所は確率で持つ。"""
    import os
    return os.environ.get(HIDDEN_ENV, '0') == '2'


def _pair(value: Any) -> Optional[tuple[int, int]]:
    """NEXT の値 (dict(tuple=...)・[a, b]・None) → 1〜5 の 2 色の組。読めなければ None。"""
    if isinstance(value, dict):
        value = value.get('tuple', value.get('pair'))
    try:
        a, b = (int(x) for x in value)
    except (TypeError, ValueError):
        return None
    return (a, b) if 1 <= a <= 5 and 1 <= b <= 5 else None


def _state(side: dict) -> Optional[str]:
    s = side.get('state')
    return s.get('state') if isinstance(s, dict) else s


class HiddenRow:
    """依頼 F4 (user の決まり 9/28・10/6・10/17): 1 側の隠し段 (行 0) の推定。確実な場合だけ持ち、不確実なら空 (今と同じ)。
    - 置いた組 = NEXT が 1 つ繰り上がった時 (新しい NEXT = 前の NEXT2) の前の NEXT。繰り上がりが読めない時は組の色なし。
    - 確定盤 (STABLE) が変わった時、可視段 (行 1〜12) に増えたぷよが 1 個だけで、その色が置いた組の片方なら、もう片方は隠し段にある。
      入る列の候補 = 縦置き (増えた 1 個が行 1 = 12 段目、同じ列) と横置き (隣の列で、置く前に満杯 = 行 1 が埋まっていた列)。
      候補がちょうど 1 列で、その列の推定が空の時だけ置く。それ以外 (候補 0・2 以上、2 個とも見えない、3 個以上・おじゃま・消えた) は置かない。
    - 推定のある列の行 1 が空いた (下が消えて落ちた) 確定盤では、その列の一番上の色と比べて当たり・外れを数え、推定を消す。
    - 隠し段のぷよは接着しない・消えない (12 段目以内に落ちてから連結の対象) ので、推定は落ちるまで持つ。"""

    def __init__(self) -> None:
        self.stats = dict(placements=0, case_both_visible=0, case_one_visible=0, no_pair=0, mismatch=0, ambiguous=0, no_candidate=0,
                          placed=0, fell=0, hit=0, miss=0, recognized_conflict=0, other=0)
        self.reset()

    def reset(self) -> None:
        self.est: list[Optional[int]] = [None]*6
        self.prev: Optional[np.ndarray] = None
        self.queue: Optional[tuple] = None
        self.active: Optional[tuple[int, int]] = None
        self.active_known = False

    def update(self, side: Optional[dict]) -> None:
        if not side:
            return
        nxt, dnx = _pair(side.get('next_pair')), _pair(side.get('dnext_pair'))
        if nxt is not None and (self.queue is None or nxt != self.queue[0]):
            if self.queue is not None and self.queue[1] is not None and nxt == self.queue[1]:
                self.active, self.active_known = self.queue[0], True   # 1 つ繰り上がった: 前の NEXT が今置いている組
            else:
                self.active, self.active_known = None, False
        if nxt is not None:
            self.queue = (nxt, dnx)
        if _state(side) != 'STABLE':
            return
        board = clean(side.get('confirmed_board'))
        if board is None:
            return
        prev, self.prev = self.prev, board
        if prev is None or np.array_equal(prev[1:], board[1:]):
            return
        vis_p, vis_b = prev[1:] != 0, board[1:] != 0
        new = np.argwhere(~vis_p & vis_b)
        gone = np.argwhere(vis_p & ~vis_b)
        # 推定のある列で行 1 (12 段目) が空いた → 隠し段のぷよは落ちた。一番上の色と比べる
        for j in range(6):
            if self.est[j] is not None and prev[1][j] != 0 and board[1][j] == 0:
                col = board[1:, j]
                top = col[col != 0]
                self.stats['fell'] += 1
                if len(top) and int(top[0]) == self.est[j]:
                    self.stats['hit'] += 1
                else:
                    self.stats['miss'] += 1
                self.est[j] = None
        if len(gone) or (len(new) and (board[1:][~vis_p & vis_b] == OJAMA).any()):
            self.stats['other'] += 1   # 消えた (連鎖)・おじゃまが降った: 置いた組の推定はしない
            return
        if len(new) == 0 or len(new) > 2:
            if len(new) > 2:
                self.stats['other'] += 1
            return
        self.stats['placements'] += 1
        pair, self.active = self.active, None
        if pair is None:
            self.stats['no_pair'] += 1
            return
        if len(new) == 2:
            self.stats['case_both_visible'] += 1
            return
        self.stats['case_one_visible'] += 1
        (r, c), color = new[0], int(board[1:][tuple(new[0])])
        if color not in pair:
            self.stats['mismatch'] += 1
            return
        other = pair[1] if pair[0] == color else pair[0]
        cand = []
        if r == 0:   # 行 1 (12 段目) に入った: 縦置きで上の 1 個は隠し段
            cand.append(c)
        for j in (c-1, c+1):
            if 0 <= j < 6 and prev[1][j] != 0:   # 隣の列が置く前に満杯: 横置きの片方は隠し段
                cand.append(j)
        cand = [j for j in cand if self.est[j] is None]
        if len(cand) != 1:
            self.stats['ambiguous' if cand else 'no_candidate'] += 1
            return
        self.est[cand[0]] = other
        self.stats['placed'] += 1

    def apply(self, board: Optional[np.ndarray]) -> tuple[Optional[np.ndarray], tuple]:
        """盤の隠し段に推定を入れた写しと、入れた (列, 色)。認識した盤に既に行 0 の値がある列はそのまま (違えば数える)。"""
        if board is None or not any(v is not None for v in self.est):
            return board, ()
        out, ins = board.copy(), []
        for j, v in enumerate(self.est):
            if v is None:
                continue
            if out[0][j] != 0:
                if int(out[0][j]) != v:
                    self.stats['recognized_conflict'] += 1
                continue
            out[0][j] = v
            ins.append((j, int(v)))
        return out, tuple(ins)


# --- 依頼 F4 の確率版 (PC1・user 10/29「隠しは不明瞭な部分は確率で持たせるので仕方ない部分はある」、9/28 の決定) ---
# 決まり (旗 PUYO_FORECAST_HIDDEN=2 の時だけ。'1' の確実な所は同じに入れる):
# - 不確実な置きは空にせず、列ごとの {0 (空): p, 色: p} で持つ。
#   - 組の色が取れない時: 1 個だけ見えて候補の列があれば、隠し段の色は試合の色の割合 (この試合で可視段に増えたぷよ + 今の可視段の色の数)。
#   - 候補の列が 2 つ以上: 入っている確率を列で等分する。
#   - 入っている確率は 1 − DROPOUT_PRIOR (「1 個だけ見えた」が盤の取りこぼしの割合、F4 追記 1 の 60 本で約 16%)。
#   - 環境 PUYO_FORECAST_HIDDEN_GUARD=1 なら、確実な所も 1 − DROPOUT_PRIOR の確率にする (取りこぼしの守り、確かめ用)。
# - 撃った盤: 確率の列の組合せを列挙し (上位を累積 RETAINED_MASS まで、最大 MAX_COMBINATIONS)、計算する。
#   Forecast.steps は一番ありそうな組合せの連鎖。exp_total は重みつきの総得点の期待値、upside は上位 10% の総得点 − 一番ありそうな総得点。
# - 連鎖が始まった後は、観測した段の得点と合わない組合せを外して絞る。全部外れたら今と同じ取り直し・救済。
# - 落ちた時の確かめ: 確率の列の 12 段目が空いた確定盤で、その列の一番上の色の確率を足す (期待の当たり数) と、一番ありそうな色の当たりを数える。
# - 満杯の列の事前 (追記 4、PC1 10/29 23:3x、user 9/28「隠し段は空 + 試合の 4 色で確率で持つ」):
#   環境 PUYO_FORECAST_HIDDEN_FULLPRIOR=<表の json> なら、確実な推定・置きの確率が無い満杯の列 (行 1 が埋まり行 0 が空) すべてに、
#   入っている確率 = 表 (撃った盤の満杯の列の数 × 満杯になってからの置きの数、学習の 60 本で数えた割合)、色 = 試合の色に一様 を置く。
#   組合せの上限は PUYO_FORECAST_HIDDEN_MAXCOMB (既定 MAX_COMBINATIONS)。
DROPOUT_PRIOR = 0.16
MAX_COMBINATIONS = 4096
RETAINED_MASS = 0.99
UPSIDE_QUANTILE = 0.9
_TABLES: dict = {}


def _guard_enabled() -> bool:
    import os
    return os.environ.get('PUYO_FORECAST_HIDDEN_GUARD', '0') == '1'


def _full_prior_table() -> Optional[dict]:
    import json
    import os
    path = os.environ.get('PUYO_FORECAST_HIDDEN_FULLPRIOR')
    if not path:
        return None
    if path not in _TABLES:
        with open(path, encoding='utf-8') as f:
            _TABLES[path] = json.load(f)
    return _TABLES[path]


def _max_combinations() -> int:
    import os
    return int(os.environ.get('PUYO_FORECAST_HIDDEN_MAXCOMB', MAX_COMBINATIONS))


def _bucket(x: int, bounds: list) -> int:
    return next((i for i, (lo, hi) in enumerate(bounds) if lo <= x <= hi), len(bounds)-1)


class HiddenBelief(HiddenRow):
    """HiddenRow の確実な推定 (est) はそのまま、不確実な置きを列ごとの確率 (dist) で持つ。"""

    def __init__(self) -> None:
        super().__init__()
        self.stats.update(prob_placed_nopair=0, prob_placed_ambiguous=0, guard_placed=0, prob_fell=0, prob_hit=0, prob_expected_hits=0.)
        self.guard = _guard_enabled()
        self.table = _full_prior_table()

    def reset(self) -> None:
        super().reset()
        self.dist: list[Optional[dict]] = [None]*6
        self.color_counts: dict[int, int] = {}
        self.kp: list[Optional[int]] = [None]*6   # 満杯になってからの置きの数 (満杯でなければ None)

    def _share(self, board: np.ndarray) -> dict[int, float]:
        """試合の色の割合 = この試合で可視段に増えたぷよの色の数 + 今の可視段の色の数 (試合の始めでも偏らないように)。"""
        counts = dict(self.color_counts)
        for c, n in zip(*np.unique(board[1:], return_counts=True)):
            if 1 <= c <= 5:
                counts[int(c)] = counts.get(int(c), 0)+int(n)
        if not counts:
            counts = {c: 1 for c in range(1, 6)}
        total = sum(counts.values())
        return {c: n/total for c, n in counts.items()}

    def _set_dist(self, cols: list[int], colors: dict[int, float]) -> None:
        occ = (1-DROPOUT_PRIOR)/len(cols)
        for j in cols:
            self.dist[j] = {0: 1-occ, **{c: occ*p for c, p in colors.items()}}

    def update(self, side: Optional[dict]) -> None:
        if not side:
            return
        nxt, dnx = _pair(side.get('next_pair')), _pair(side.get('dnext_pair'))
        if nxt is not None and (self.queue is None or nxt != self.queue[0]):
            if self.queue is not None and self.queue[1] is not None and nxt == self.queue[1]:
                self.active, self.active_known = self.queue[0], True
            else:
                self.active, self.active_known = None, False
        if nxt is not None:
            self.queue = (nxt, dnx)
        if _state(side) != 'STABLE':
            return
        board = clean(side.get('confirmed_board'))
        if board is None:
            return
        prev, self.prev = self.prev, board
        for j in range(6):
            self.kp[j] = None if board[1][j] == 0 else 0 if self.kp[j] is None else self.kp[j]
        if prev is None or np.array_equal(prev[1:], board[1:]):
            return
        vis_p, vis_b = prev[1:] != 0, board[1:] != 0
        new = np.argwhere(~vis_p & vis_b)
        gone = np.argwhere(vis_p & ~vis_b)
        for j in range(6):
            if prev[1][j] != 0 and board[1][j] == 0:
                col = board[1:, j]
                top = col[col != 0]
                if self.est[j] is not None:
                    self.stats['fell'] += 1
                    if len(top) and int(top[0]) == self.est[j]:
                        self.stats['hit'] += 1
                    else:
                        self.stats['miss'] += 1
                    self.est[j] = None
                if self.dist[j] is not None:
                    d = self.dist[j]
                    self.stats['prob_fell'] += 1
                    if len(top):
                        self.stats['prob_expected_hits'] += float(d.get(int(top[0]), 0.))
                        best = max((c for c in d if c != 0), key=lambda c: d[c], default=None)
                        self.stats['prob_hit'] += int(best is not None and best == int(top[0]))
                    self.dist[j] = None
            if board[1][j] == 0:
                self.dist[j] = None
        if len(gone) or (len(new) and (board[1:][~vis_p & vis_b] == OJAMA).any()):
            self.stats['other'] += 1
            return
        if len(new) == 0 or len(new) > 2:
            if len(new) > 2:
                self.stats['other'] += 1
            return
        for c in board[1:][~vis_p & vis_b]:
            self.color_counts[int(c)] = self.color_counts.get(int(c), 0)+1
        for j in range(6):
            if self.kp[j] is not None and prev[1][j] != 0:
                self.kp[j] += 1
        self.stats['placements'] += 1
        pair, self.active = self.active, None
        one = len(new) == 1
        (r, c), color = (new[0], int(board[1:][tuple(new[0])])) if one else ((None, None), None)
        cand = []
        if one:
            if r == 0:
                cand.append(c)
            for j in (c-1, c+1):
                if 0 <= j < 6 and prev[1][j] != 0:
                    cand.append(j)
        if pair is None:
            self.stats['no_pair'] += 1
            free = [j for j in cand if self.est[j] is None and self.dist[j] is None]
            if one and free:
                self._set_dist(free, self._share(board))
                self.stats['prob_placed_nopair'] += 1
            return
        if not one:
            self.stats['case_both_visible'] += 1
            return
        self.stats['case_one_visible'] += 1
        if color not in pair:
            self.stats['mismatch'] += 1
            return
        other = pair[1] if pair[0] == color else pair[0]
        cand = [j for j in cand if self.est[j] is None]
        if len(cand) != 1:
            self.stats['ambiguous' if cand else 'no_candidate'] += 1
            free = [j for j in cand if self.dist[j] is None]
            if free:
                self._set_dist(free, {other: 1.})
                self.stats['prob_placed_ambiguous'] += 1
            return
        if self.guard:
            self._set_dist(cand, {other: 1.})
            self.stats['guard_placed'] += 1
            return
        self.est[cand[0]] = other
        self.dist[cand[0]] = None
        self.stats['placed'] += 1

    def belief(self, board: Optional[np.ndarray]) -> tuple:
        """撃った盤に入れる確率の列: ((列, ((色, 確率), ...)), ...)。認識した盤に行 0 の値がある列・確実な推定の列は外す。"""
        if board is None:
            return ()
        out = [(j, tuple(sorted(d.items()))) for j, d in enumerate(self.dist)
               if d is not None and self.est[j] is None and board[0][j] == 0]
        if self.table is not None:   # 満杯の列の事前 (表の割合、試合の色に一様)
            have = {j for j, _ in out}
            colors = sorted(self._share(board))
            nf = _bucket(int(np.sum(board[1] != 0)), self.table['buckets']['n_full'])
            for j in range(6):
                if j in have or self.est[j] is not None or board[1][j] == 0 or board[0][j] != 0:
                    continue
                kb = _bucket(self.kp[j] or 0, self.table['buckets']['k_place'])
                q = float(self.table['rate'].get(f'{nf}|{kb}', 0.))
                if q > 0 and colors:
                    out.append((j, tuple(sorted({0: 1-q, **{c: q/len(colors) for c in colors}}.items()))))
            out.sort()
        return tuple(out)


def _combinations(belief: tuple) -> tuple[list[tuple[tuple, float]], float]:
    """列ごとの確率 → 上位の組合せ ((列, 色), ...) と重み (累積 RETAINED_MASS まで、最大 PUYO_FORECAST_HIDDEN_MAXCOMB、足して 1) と、残した確率の和。
    確率の高い順に最良優先で取り出す (全部の積を並べない)。"""
    import heapq
    cols = [j for j, _ in belief]
    opts = [sorted(((c, p) for c, p in d if p > 0), key=lambda x: (-x[1], x[0])) for _, d in belief]
    limit = _max_combinations()

    def weight(idx):
        w = 1.
        for o, i in zip(opts, idx):
            w *= o[i][1]
        return w
    start = (0,)*len(opts)
    heap, seen, out, mass = [(-weight(start), start)], {start}, [], 0.
    while heap and len(out) < limit:
        nw, idx = heapq.heappop(heap)
        out.append((tuple(zip(cols, (o[i][0] for o, i in zip(opts, idx)))), -nw))
        mass -= nw
        if mass >= RETAINED_MASS:
            break
        for k in range(len(idx)):
            if idx[k]+1 < len(opts[k]):
                nxt = idx[:k]+(idx[k]+1,)+idx[k+1:]
                if nxt not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, (-weight(nxt), nxt))
    return [(combo, w/mass) for combo, w in out], mass


def _summary(combos: tuple) -> tuple[int, float, int]:
    """組合せ (重み, 段, 最終盤) → (一番ありそうな位置, 総得点の期待値, 上振れ幅)。"""
    best = max(range(len(combos)), key=lambda i: combos[i][0])
    totals = sorted((sum(s), w) for w, s, _ in combos)
    exp = float(sum(t*w for t, w in totals))
    acc, q = 0., totals[-1][0]
    for t, w in totals:
        acc += w
        if acc >= UPSIDE_QUANTILE-1e-12:
            q = t
            break
    return best, exp, int(q-sum(combos[best][1]))


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
        self.hidden = HiddenBelief() if hidden_prob_enabled() else HiddenRow() if hidden_enabled() else None   # 依頼 F4 (旗 ON の時だけ)
        self.split = split_enabled()   # 依頼 §4 (旗 ON の時だけ)
        self._ins: deque = deque(maxlen=KEEP)
        self.reset()

    def reset(self) -> None:
        self.hist.clear()
        if hasattr(self, '_light_physics'):
            del self._light_physics
        from src.chain_continuity import ChainBoundary
        self.boundary = ChainBoundary() if self.end_formula else None
        self.last_valid: Optional[float] = None
        self.block_last: Optional[float] = None
        self.fc: Optional[Forecast] = None
        self._pred: Optional[list[int]] = None
        self.tsumo: Optional[int] = None        # 依頼 §4: 前のコマの tsumo_count
        self.placed_t: Optional[float] = None   # 依頼 §4: 最後に置いた (tsumo_count が増えた) 時刻
        if self.hidden is not None:
            self.hidden.reset()
            self._ins.clear()

    def _find_t(self, t: float, product: int, off: int) -> tuple[Optional[float], Optional[np.ndarray]]:
        cands = [(tt, b) for tt, b in self.hist.frames if tt < t-1e-9][-BACK:]
        for tt, b in reversed(cands):
            if b is None:
                continue
            groups, _ = self.trace(b)
            s = step_scores(groups, off)
            if s and s[0] == product:
                return tt, b
        return None, None

    def _find(self, t: float, product: int, off: int) -> Optional[np.ndarray]:
        return self._find_t(t, product, off)[1]

    def _inserted(self, tt: Optional[float], k: int = 1) -> tuple:
        """撃った盤の時刻 tt に入れた (列, 色) (k = 1)・確率の列 (k = 2、確率版だけ)。"""
        return next((x[k] for x in self._ins if x[0] == tt and len(x) > k), ()) if self.hidden is not None and tt is not None else ()

    def _prob(self, tt: Optional[float], board: np.ndarray, groups: list, final: np.ndarray) -> dict:
        """確率版: 撃った盤 (確実な所は入れた盤) に確率の列の組合せを入れて計算する。組合せが無ければ今の盤だけ (重み 1)。"""
        import time
        started = time.perf_counter()
        belief = self._inserted(tt, 2)
        combos = []
        cands, mass = _combinations(belief) if belief else ([((), 1.)], 1.)
        for combo, w in cands:
            if combo:
                b = board.copy()
                for j, c in combo:
                    b[0][j] = c
                g, f = self.trace(b)
            else:
                g, f = groups, final
            combos.append((w, tuple(step_scores(g, 0)), f))
        best, exp, up = _summary(tuple(combos))
        return dict(combos=tuple(combos), best=best, exp=exp, up=up, mass=mass, ms=1000*(time.perf_counter()-started))

    def observe(self, t: float, formula: Optional[dict], raw_board: Any, side: Optional[dict] = None) -> None:
        """side は配信入力の 1 側 (state・confirmed_board・next_pair・dnext_pair)。旗 ON (依頼 F4) の時だけ隠し段の推定に使う。"""
        if self.boundary is not None:
            self.boundary.observe(t, dict(side or {}, formula=formula, raw_cnn_board=raw_board))
        if self.hidden is not None:
            self.hidden.update(side)
            raw_board, ins = self.hidden.apply(clean(raw_board))
            if isinstance(self.hidden, HiddenBelief):
                self._ins.append((float(t), ins, self.hidden.belief(raw_board)))
            else:
                self._ins.append((float(t), ins))
        if self.split and side:
            tc = side.get('tsumo_count')
            if tc is not None:
                if self.tsumo is not None and int(tc) > self.tsumo and (self.last_valid is None or t-self.last_valid >= PLACE_GAP_SEC):
                    self.placed_t = float(t)   # 式が出ている間 (最後の有効な式から PLACE_GAP_SEC 未満) の増えは連鎖の途中の読みの揺れ
                self.tsumo = int(tc)
        form = formula or {}
        valid = bool(form.get('valid')) and form.get('product') is not None
        if not valid and (side or {}).get('light_next_consumed'):
            self._light_hidden_fire = False
        if valid:
            p = int(form['product'])
            expired = not self.end_formula and self.last_valid is not None and t-self.last_valid >= self.REARM_SEC
            separated = self.boundary is not None and self.fc is not None and self.boundary.t > self.fc.chain_id
            if self.last_valid is None or expired or separated or self._placed_since_last():
                from src.light_consumption import enabled as light_enabled
                unknown = light_enabled() and bool((side or {}).get('light_hidden_inflow_unknown'))
                self._new_chain(t, p, uncertain_hidden=unknown)
                _dump(self.name, t, self.fc)
            elif t-self.block_last > BLOCK_GAP_SEC and p != self.fc.observed[-1]:
                self._new_step(t, p)
                _dump(self.name, t, self.fc)
            self.last_valid = self.block_last = t
            self.fc.last_valid_t = t
        if self.recheck:
            self._recheck(t, raw_board, side)
        self.hist.push(t, raw_board)
        from src.light_refind import update_physical_refind
        update_physical_refind(self, t, raw_board, side)

    def _recheck(self, t: float, raw: Any, side: Optional[dict]) -> None:
        from src.chain_continuity import recheck_board
        board = recheck_board(self, t, raw, side)
        if board is None:
            return
        groups, final = self.trace(board)
        self._pred = step_scores(groups)
        fc = self.fc
        self.fc = Forecast(fc.chain_id, fc.observed, tuple(self._pred), 'fire_recheck', final,
                           fc.last_valid_t, fc.refinds, fc.rescues)

    def _new_chain(self, t: float, p: int, uncertain_hidden: bool = False) -> None:
        tt, b = self._find_t(t, p, 0)
        self._light_hidden_fire = uncertain_hidden
        prob = {}
        if b is None:
            self._pred, src, final = None, 'none', None
        else:
            groups, final = self.trace(b)
            self._pred, src = step_scores(groups, 0), 'fire_board'
            if isinstance(self.hidden, HiddenBelief):
                r = self._prob(tt, b, groups, final)
                w, steps, final = r['combos'][r['best']]
                self._pred = list(steps)
                prob = dict(exp_total=r['exp'], upside=r['up'], n_comb=len(r['combos']), combos=r['combos'], comb_mass=r['mass'], comb_ms=r['ms'])
        self.fc = Forecast(round(t, 6), (p,), tuple(self._pred) if self._pred else (p,), src, final, t, hidden=self._inserted(tt), **prob)

    def _new_step(self, t: float, p: int) -> None:
        fc = self.fc
        obs = list(fc.observed)+[p]
        k = len(obs)
        pred, src, final, refinds, rescues, hidden = self._pred, fc.source, fc.final_board, fc.refinds, fc.rescues, fc.hidden
        prob = {}
        if fc.combos:   # 確率版: 観測した段と合う組合せだけに絞る。全部外れたら今と同じ取り直し・救済
            keep = tuple((w, s, f) for w, s, f in fc.combos if len(s) >= k and s[k-1] == p)
            if keep:
                mass = sum(w for w, _, _ in keep)
                keep = tuple((w/mass, s, f) for w, s, f in keep)
                best, exp, up = _summary(keep)
                pred, final = list(keep[best][1]), keep[best][2]
                prob = dict(exp_total=exp, upside=up, n_comb=len(keep), combos=keep)
        if pred is None or len(pred) < k or pred[k-1] != p:
            tt, b = self._find_t(t, p, k-1)
            if b is not None:   # A2 取り直し
                groups, final = self.trace(b)
                pred, src, refinds, hidden = obs[:-1]+step_scores(groups, k-1), 'refind', refinds+1, self._inserted(tt)
            elif pred is not None:   # A3 救済: 観測済み + それまでの予測の残り
                pred, src, rescues = obs+list(pred[k:]), 'rescue', rescues+1
        self._pred = pred
        steps = tuple(obs+list(pred[k:])) if pred is not None else tuple(obs)
        self.fc = Forecast(fc.chain_id, tuple(obs), steps, src if pred is not None else 'none', final, t, refinds, rescues, hidden, **prob)

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
        from src.light_consumption import guarded_forecast
        from src.light_refind import physical_forecast
        return guarded_forecast(physical_forecast(self, self.fc), getattr(self, '_light_hidden_fire', False))


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
