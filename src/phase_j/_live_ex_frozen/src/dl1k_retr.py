"""retR: 受け側が着弾の締め切りまでに返せる最大の送り量 C を、本物のツモ (種の位置) と Rust の ama の return_profile で求める列 (dl1k 研究専用・既定 OFF)。

PC1 2026-10-28 15:40/15:45 の定め。学習 (dl1k_extract と同じ再生) と配信 (live_ex_engine) は、この関数を同じ ExEngine の状態で呼ぶ。
定め (受け側 s・攻め側 o = 1-s。両 side を受け側として計算する):
- I = 射影の予告 value['pending'][s] (EX の射影: 既に発火した両側の連鎖の送りを、予測の段の得点まで台帳に足して相殺した後に s に残る数)。
  両者連鎖中も同じ式 (retF2 の「両方の連鎖を最後まで進めた後に受け側に残る予告」)。攻め側の発火盤の予測 (chains[o]['best']) で
  I が埋まる (retF3 の「I 不明の埋め」に当たる)。
- 締め切り: 攻め側が連鎖中 (engine._resolving(o)) で予測が有効なら end_o = max(今, 始まり + duration(段数), 各段の得点の時刻の最大)
  (prefire_context_v410.chain_events と同じ)。受け側の打ち始め start = 受け側が連鎖中なら end_s、それ以外は今。
  h = floor((end_o + SEC_PER_HAND − start + 1e-9) / SEC_PER_HAND) (EX の探索の手数と同じ。0 未満は 0、上限 40)。
  攻め側が連鎖中でなく I > 0 (受け側の連鎖の後に予告が降る) なら h = 1。どちらでもなければ NaN (攻撃なし)。
  PUYO_RETR_H_FORECAST=1 (依頼 F3、既定 OFF) なら end_o・end_s を共通の連鎖の予測 (engine.retr_fire.current) から作る (forecast_end)。
  予測が無い側は今の chain_end のまま。retr_diag に h_old (今の作り方の h) と end_src ('forecast' / 'ex') を残す。
- 盤: 受け側が連鎖中でなければ今の確定盤 (state.sides[s].board)、連鎖中なら予測の最終盤 value['boards'][s]。
  連鎖中で予測が無い間は、受け側が連鎖中でない最後の計算の値の組 (C, R, h, 致死) を持ち越す (retF2 の (A)、時間の上限なし。同じ試合の中だけ)。
- C: h = 0 なら 0。受け側が連鎖中でなく、今の盤がもう連鎖を含むなら、その連鎖の得点 // レート。
  それ以外は、種と位置が分かっている時の本物のツモ 16 組 (受け側が次に置く組から。game_seed・seed_pairs) と盤を
  ama の return_profile (幅 250・深さ 16) に渡し、C = cummax[min(h,16)−1] // レート。種・位置が分からない・16 組に足りない・盤が 4 色に収まらない時は NaN。
  種: テープの観測列の先頭から 4 組ずつずらした 8 組の窓で最初に一意な種 (各側)、集まった種が 1 つなら試合の種として固定。
  位置: テープの own の末尾 6・5・4 組を種のテープ視点の列で照合し、abs_next に近い所 = 今の NEXT の絶対手番 p。連鎖中は p から、それ以外は p−1 (操作中の組) から。
- R = I − C。致死 = R > 0 かつ 盤の 3 列目 (列番号 2) の高さ + min(R,30)//6 ≥ 12。
列: retR_{c,r,h,lethal}_{p1,p2} (p1/p2 = 受け側の物理 side)。向き 1 の行は p1/p2 を入れ替える。
撃った側のセカンド (PC1 10/30 REQUEST_second_col、旗 PUYO_RETR_SECOND=1、既定 OFF で今とビット一致): 受け側 s ごとに
retR_second_s = 撃った側 o のセカンド (おじゃま個数)、retR_net2_s = (retR_c − I) − retR_second。定めは second_of・return_end・hands_in。
PUYO_RETR_OWN_CHAIN=1 は未反映の既発火送りをC/Cvisへ含め、その分をIで二重相殺しない。
PUYO_RETR_LONG_ATTACK_H=1 は現在のEX・共通予測・観測時刻から締切を取り直す。両旗とも既定OFF。
PUYO_RETR_RECEIVER_FORECAST=1 は共通予測がEXより長い受け側の送りも射影へ反映する（既定OFF）。
"""
from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
import subprocess
import time
from typing import Any

import numpy as np

from src.phase_j._live_ex_frozen.src.exchange_count_ledger import FORMULA_SCORE_DELAY_SEC, STEP_SEC
from src.phase_j._live_ex_frozen.src.indicators_v2 import SEC_PER_HAND
from src.phase_j._live_ex_frozen.src.prefire_context_v410 import duration

SIDES = (0, 1)
NAMES = ('p1', 'p2')
STEMS = ('retR_c', 'retR_r', 'retR_h', 'retR_lethal')
COLUMNS = tuple(f'{s}_{p}' for s in STEMS for p in NAMES)
DEPTH, WIDTH, MAX_H = 16, 250, 40
DEATH_COL, DEATH_H, ROWS, COLS = 2, 12, 13, 6
DEFAULT_RATE = 70
CACHE_MAX = 4096
VISIBLE_ENV = 'PUYO_RETR_VISIBLE'
LANDING_ENV = 'PUYO_RETR_LANDING'
DROP_CAP = 30
VISIBLE_STEMS = ('retR_cvis', 'retR_rvis', 'retR_unseen', 'retR_vis_frac',
                 'retR_rvis_room', 'retR_r_room')
VISIBLE_COLUMNS = tuple(f'{s}_{p}' for s in VISIBLE_STEMS for p in NAMES)
VISIBLE_HANDS, BUSY_VISIBLE_HANDS, ROOM_RATIO_MAX = 3, 2, 10.


def visible_enabled() -> bool:
    """見えている手の列を追加するか。既定 OFF。"""
    from src.production_config import RETR_VISIBLE_DEFAULT
    return os.environ.get(VISIBLE_ENV, str(int(RETR_VISIBLE_DEFAULT))) == '1'


def landing_enabled() -> bool:
    """確定済み予告の落下を探索に入れるか。既定 OFF。"""
    from src.production_config import RETR_LANDING_DEFAULT
    return os.environ.get(LANDING_ENV, str(int(RETR_LANDING_DEFAULT))) == '1'


def confirmed_pending(engine: Any, side: int) -> int:
    """連鎖終了で確定し、相殺・落下後に台帳に残る数。射影の予測は含めない。"""
    return int(engine.state.sl.core._pending(side, confirmed_only=True))


def visible_values(C: float, I: int, h: int, busy: bool, board: np.ndarray,
                   cummax: list | None, rate: int) -> tuple[float, ...]:
    """cvis は見える3手（連鎖中2手）の最大、rvis は I から引いた残り。

    unseen は全手の最大 C との差、vis_frac は cvis/C（C=0なら1）。
    rvis_room と r_room は正の残りを3列目の空き個数で割り、上限10。
    空き0で残り正なら10、残り0以下なら0。既に盤上にある連鎖は C 全量が見える。
    """
    v = BUSY_VISIBLE_HANDS if busy else VISIBLE_HANDS
    cvis = C if cummax is None else float(int(cummax[min(h, v)-1]) // rate)
    if h == 0:
        cvis = 0.
    room = COLS * max(DEATH_H - int(heights(board)[DEATH_COL]), 0)

    def ratio(remaining: float) -> float:
        if remaining <= 0:
            return 0.
        return min(remaining / room, ROOM_RATIO_MAX) if room else ROOM_RATIO_MAX

    return cvis, I-cvis, C-cvis, cvis/C if C else 1., ratio(I-cvis), ratio(I-C)
EXE_ENV = 'PUYO_AMA_REFERENCE_EXE'   # 学習 (Linux) は ama_reference_linux。配信の pyd は ama_reference_json_py を持つ

# 撃った側のセカンド (PC1 10/30 REQUEST_second_col、user の決定)。旗 PUYO_RETR_SECOND=1 の時だけ列を足す (既定 OFF = 今のままの 8 列)。
# 列 (受け側 s の物理 side の名前で): retR_second = 撃った側 o のセカンド (おじゃま個数)、retR_net2 = 返した後の差 − そのセカンド
# = (retR_c − I) − retR_second = −retR_r − retR_second (おじゃま個数)、retR_secn = o が置ける手数 N、retR_sect = o が置ける時間 (秒、0 未満は 0)。
# N と時間はモデルがセカンドの効きを手数で弱められるように入れる (PC1 10/30 18 時台の判断、user の 10/22 の決まり)。
SECOND_ENV = 'PUYO_RETR_SECOND'
SECOND_STEMS = ('retR_second', 'retR_net2', 'retR_secn', 'retR_sect')
SECOND_COLUMNS = tuple(f'{s}_{p}' for s in SECOND_STEMS for p in NAMES)
# 1 手にかかる時間 (秒): 着地の段 (0 = 隠し最上段 → 12 = 最下段) ごとの最速側 30% の平均 (user の定め 10/30、ちぎりは無視)。
# 出どころ: share/b3_c2_ex_20261026/move_time_table/move_time_table.json の tables["0.3"] (PC1、8/3 の測りの元データ 83 本)。
MOVE_TIME_SEC = (0.154, 0.214, 0.293, 0.316, 0.343, 0.373, 0.406, 0.453, 0.482, 0.535, 0.542, 0.472, 0.474)
# 返しの連鎖の段数 (まだ撃っていない受け側で、盤に連鎖が無く ama の得点しか無い時): ama は最善の手順の盤を返さないので、
# 学習の 60 本の観測の連鎖 (f4t/split2、0.5 秒の直しの後の分け方、18,003 本) の「その得点の連鎖の段数の中央」(得点の区間ごと、単調に整えた)。
# 区間の下端 STEP_SCORE_EDGES[i] 以上・次の下端未満なら STEP_MEDIAN[i] 段 (PC1 10/30 の判断 (b))。
# PC1 11/1: 両旗とも既定 OFF。段周期は duration と共通の STEP_SEC を使う。
MOVE_TIME_TOP50_ENV = 'PUYO_RETR_MOVE_TIME_TOP50'
DEADLINE_EXTEND_ENV = 'PUYO_RETR_DEADLINE_EXTEND'
MOVE_TIME_SEC_TOP50 = (0.242, 0.330, 0.402, 0.424, 0.449, 0.479, 0.509, 0.565, 0.616, 0.689, 0.685, 0.599, 0.602)


def extended_end(end: float | None, predicted_end: float, observed: int,
                 predicted: int, last_step: float | None, now: float, busy: bool) -> float | None:
    """予測を超えて継続中の連鎖だけ、最後の観測から1段先へ締め切りを延ばす。"""
    if os.environ.get(DEADLINE_EXTEND_ENV, '0') != '1' or not busy or end is None:
        return end
    if last_step is not None and (observed > predicted or now >= predicted_end):
        return max(end, last_step+STEP_SEC)
    return end


def extend_side_end(engine: Any, value: dict, side: int, now: float,
                    end: float | None, fc: Any = None) -> float | None:
    """観測は当該連鎖の現在までの証跡のみ。返し終了も同じ ends を使う。"""
    if os.environ.get(DEADLINE_EXTEND_ENV, '0') != '1' or end is None:
        return end
    chain = engine.chains[side]
    if chain is None or not engine._resolving(side):
        return end
    obs = chain.get('observed', [])
    n_obs = max((int(x['count']) for x in obs), default=0)
    n_pred = len(chain['best']['scores'])
    last = max((float(x['t']) for x in obs), default=None)
    pred_end = float(chain['started'])+duration(n_pred)
    if fc is not None:
        n_pred = len(fc.steps)
        n_obs = len(fc.observed)
        last = float(fc.last_valid_t)
        pred_end = float(fc.chain_id)-FORMULA_SCORE_DELAY_SEC+duration(n_pred)
    return extended_end(end, pred_end, n_obs, n_pred, last, now, True)


STEP_SCORE_EDGES = (0, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 4000, 5500, 7000, 9000, 12000, 15000, 20000, 25000, 32000,
                    40000, 50000, 65000, 80000, 100000)
STEP_MEDIAN = (1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 12, 13, 15)


def second_enabled() -> bool:
    """撃った側のセカンドの列を足すか。既定 OFF、'1' で ON。"""
    return os.environ.get(SECOND_ENV, '0') == '1'


def blank() -> dict:
    out = {c: math.nan for c in COLUMNS}
    if second_enabled():
        out.update({c: math.nan for c in SECOND_COLUMNS})
    if visible_enabled():
        out.update({c: math.nan for c in VISIBLE_COLUMNS})
    return out


def swap(extra: dict) -> dict:
    out = dict(extra)
    for s in STEMS+SECOND_STEMS+VISIBLE_STEMS:
        if f'{s}_p1' in extra:
            out[f'{s}_p1'], out[f'{s}_p2'] = extra[f'{s}_p2'], extra[f'{s}_p1']
    return out


class Ama:
    """return_profile の呼び出し。pyd に ama_reference_json_py があればそれ、無ければ外部の実行ファイル (同じ JSON・同じ値)。"""

    def __init__(self, native: Any) -> None:
        self.force_external = os.environ.get('PUYO_AMA_FORCE_EXTERNAL', '0') == '1'
        self.fn = getattr(native, 'ama_reference_json_py', None)
        if self.force_external:
            self.fn = None
        self.proc = None
        self.cache: OrderedDict = OrderedDict()
        self.calls = self.hits = 0
        if self.force_external:
            from multiprocessing.util import Finalize
            from threading import Lock
            self._request_lock, self._cache_lock = Lock(), Lock()
            # 複製はこのAmaを参照するだけ。作成元の終了時に一度だけ閉じる。
            self._finalizer = Finalize(self, self.close, exitpriority=10)

    def _ask(self, request: dict) -> dict:
        if self.force_external:
            with self._request_lock:
                return self._ask_unlocked(request)
        return self._ask_unlocked(request)

    def _ask_unlocked(self, request: dict) -> dict:
        line = json.dumps(request)
        if self.fn is not None:   # return_profile は 1 枝なので threads は効かない。配信と同じく 1 を渡す
            return json.loads(self.fn(line, 1))
        if self.proc is None:
            exe = os.environ.get(EXE_ENV)
            if not exe:
                raise RuntimeError(f'ama_reference_json_py も {EXE_ENV} も無い')
            command, env = [exe], {'AMA_THREADS': '1'}
            if self.force_external:
                env = dict(os.environ, AMA_THREADS='1')
                if os.name == 'nt' and exe.lower().endswith('.cmd'):
                    command = [os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/c', exe]
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                         env=env)
        self.proc.stdin.write(line+'\n')
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def profile(self, board: list[int], pairs: list[list[int]], pending: int = 0,
                rate: int = DEFAULT_RATE, drop_cap: int = DROP_CAP) -> list[int]:
        if self.force_external:
            with self._cache_lock:
                return self._profile(board, pairs, pending, rate, drop_cap)
        return self._profile(board, pairs, pending, rate, drop_cap)

    def _profile(self, board: list[int], pairs: list[list[int]], pending: int,
                 rate: int, drop_cap: int) -> list[int]:
        key = (tuple(board), tuple(map(tuple, pairs)))
        if pending:
            key += (pending, rate, drop_cap)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.calls += 1
        request = dict(board=board, visible=pairs, op='return_profile', width=WIDTH, depth=DEPTH)
        if pending:
            request.update(pending=pending, rate=rate, drop_cap=drop_cap)
        out = list(self._ask(request)['cummax'])
        self.cache[key] = out
        if len(self.cache) > CACHE_MAX:
            self.cache.popitem(last=False)
        return out

    def close(self) -> None:
        if self.force_external:
            with self._request_lock:
                self._close()
        else:
            self._close()

    def _close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
            self.proc = None


def heights(board: np.ndarray) -> np.ndarray:
    occ = board != 0
    top = np.where(occ.any(0), occ.argmax(0), ROWS)
    return ROWS-top


def chain_score(native: Any, board: np.ndarray) -> tuple[int, int]:
    cells = [c if 0 <= c <= 5 or c == 9 else 9 for c in map(int, board.ravel())]   # 不明のセルはおじゃまに (計算器の決まり)
    trace = native.simulate_chain_trace_py(cells, True)
    return len(trace), int(sum(int(step[4]) for step in trace))


def chain_end(engine: Any, value: dict, side: int, now: float) -> float | None:
    """prefire_context_v410.committed / chain_events と同じ連鎖の終わり。予測が無効なら None。"""
    chain, rec = engine.chains[side], value['reconciliation'][side]
    if not value['board_prediction_active'][side] or chain is None or rec is None:
        return None
    trigger, scores = chain['started'], chain['best']['scores']
    stamps = rec['times'] or [trigger+FORMULA_SCORE_DELAY_SEC+i*STEP_SEC for i in range(len(scores))]
    return max(now, trigger+duration(len(scores)), max(stamps, default=now))


H_FORECAST_ENV = 'PUYO_RETR_H_FORECAST'


def h_forecast_enabled() -> bool:
    """依頼 F3: 連鎖の終わり (h の締め切りと受け側の打ち始め) を共通の連鎖の予測から作るか。既定 OFF、'1' で ON。"""
    return os.environ.get(H_FORECAST_ENV, '0') == '1'


def forecast_end(fc: Any, now: float) -> float:
    """共通の連鎖の予測 (src/fire_board.Forecast) からの連鎖の終わり。chain_end と同じ物差し (duration・FORMULA_SCORE_DELAY_SEC・STEP_SEC)。
    段数 = len(fc.steps)。起点 = fc.chain_id (最初の式の時刻) − FORMULA_SCORE_DELAY_SEC (chain_end の trigger と同じ基準)。
    観測済みの段は最後の式の時刻 fc.last_valid_t が分かるので、そこから残りの段 × STEP_SEC 後も終わりの候補に入れる (段が遅れた時の分)。"""
    n = len(fc.steps)
    trigger = float(fc.chain_id)-FORMULA_SCORE_DELAY_SEC
    stamps = [trigger+FORMULA_SCORE_DELAY_SEC+i*STEP_SEC for i in range(n)]
    stamps.append(float(fc.last_valid_t)+max(0, n-len(fc.observed))*STEP_SEC)
    return max(now, trigger+duration(n), max(stamps))


def side_end(engine: Any, value: dict, side: int, now: float, tracker: Any) -> tuple[float | None, str]:
    """側 side の連鎖の終わりと出どころ。旗 ON で共通の予測があればそれ ('forecast')、無ければ EX の chain_end ('ex')。"""
    if h_forecast_enabled() and tracker is not None and hasattr(tracker, 'current'):
        fc = tracker.current(side, now)
        if fc is not None and fc.has_prediction:
            return extend_side_end(engine, value, side, now, forecast_end(fc, now), fc), 'forecast'
    return extend_side_end(engine, value, side, now, chain_end(engine, value, side, now)), 'ex'


def encode(board: np.ndarray, pairs: list) -> tuple[list[int], list[list[int]]] | None:
    """盤と組の色を ama の 1..4 に詰める (retcolF.rust_profiles と同じ)。5 色・不明の値なら None。"""
    cells = [int(c) for c in board.ravel()]
    used = sorted({c for c in cells if 1 <= c <= 5} | {int(c) for p in pairs for c in p})
    if len(used) > 4 or any(c not in (0, 9) and not 1 <= c <= 5 for c in cells) or any(not 1 <= int(c) <= 5 for p in pairs for c in p):
        return None
    used += [c for c in range(1, 6) if c not in used][:4-len(used)]
    code = {c: i+1 for i, c in enumerate(used)}
    return ([0 if c == 0 else 9 if c == 9 else code[c] for c in cells],
            [[code[int(a)], code[int(b)]] for a, b in pairs])


def steps_from_score(score: int) -> int:
    """その得点の連鎖の段数の中央 (STEP_MEDIAN、学習の観測の連鎖から)。0 点は 0 段。"""
    if score <= 0:
        return 0
    return STEP_MEDIAN[sum(1 for v in STEP_SCORE_EDGES if v <= score)-1]


def hands_in(board: np.ndarray, budget: float) -> int:
    """budget 秒に置ける手数。1 手の時間は MOVE_TIME_SEC[着地の段]。ama は置き場所を返さないので、着地の段は
    盤の列の平均の高さ (隠し段を含む) で近似し、1 手ごとに 2 個 / 6 列 だけ高くする (ちぎりは無視、user の定め)。"""
    if not budget > 0:
        return 0
    level = float(np.count_nonzero(np.asarray(board) != 0))/COLS
    n, used = 0, 0.
    while n < MAX_H:
        row = min(ROWS-1, max(0, ROWS-1-int(level)))
        step = (MOVE_TIME_SEC_TOP50 if os.environ.get(MOVE_TIME_TOP50_ENV, '0') == '1' else MOVE_TIME_SEC)[row]
        if used+step > budget+1e-9:
            break
        used += step
        n += 1
        level += 2/COLS
    return n


def return_end(ret: dict, start: float, ends_s: float | None) -> tuple[float | None, int | None]:
    """受け側の返しの連鎖の終わり (と見積もった段数)。ret は compute_retr が C を出した道:
    - 'busy': 受け側はもう撃っている → その連鎖の終わり ends_s (段数は None)。
    - 'board_chain': 盤にもう連鎖がある → 1 手 (SEC_PER_HAND) 後に発火、段数は計算の段数。
    - 'ama': return_profile の cummax が h 手の値に最初に届いた手 k (0 始まり) → 打ち始め + (k+1) × SEC_PER_HAND に発火、
      段数は steps_from_score(その得点)。h と同じ物差し (SEC_PER_HAND)。
    終わり = 発火 + duration(段数) (EX の連鎖の時間軸と同じ)。返しが無い ('none') なら (None, 0)。"""
    kind = ret['kind']
    if kind == 'after_small':
        return (return_end(ret['after'], ret['start'], None) if ret['after']['kind'] != 'none'
                else (ret['end'], None))
    if kind == 'busy':
        return ends_s, None
    if kind == 'none':
        return None, 0
    if kind == 'board_chain':
        fire, steps = start+SEC_PER_HAND, int(ret['steps'])
    else:
        cm, h = ret['cummax'], ret['h']
        goal = cm[min(h, DEPTH)-1]
        k = next(i for i, v in enumerate(cm) if v >= goal)
        fire, steps = start+(k+1)*SEC_PER_HAND, steps_from_score(int(cm[k]))
    return fire+duration(steps), steps


def second_of(engine: Any, value: dict, s: int, now: float, ends: list, resolving: list, ret: dict,
              rate: int, tracker: Any) -> tuple[float, dict, str]:
    """受け側 s の返しの間に撃った側 o = 1−s が置ける手数での、o のセカンド (おじゃま個数) と確かめ用の値・理由。
    置ける時間 = 返しの終わり − o が操作に戻る時刻 (o が連鎖中なら max(今, o の連鎖の終わり)、それ以外は今。EX の Z の start と同じ)。
    盤 = o が連鎖中なら連鎖の予測の最後の盤 (共通の予測の final_board、無ければ EX の予測の最終盤)、それ以外は今の確定盤。
    ツモ = o の本物のツモ 16 組 (seed_pairs、o が連鎖中なら今の NEXT から)。値 = cummax[min(N,16)−1] // レート (N 手、N = 0 なら 0)。"""
    o = 1-s
    d: dict = dict(ret=ret['kind'])
    end, steps = return_end(ret, now, ends[s])
    d.update(ret_end=end, ret_steps=steps)
    if ret['kind'] == 'none':
        d.update(T=0., N=0)
        return 0., d, 'no_return'
    if end is None:
        return math.nan, d, 'no_return_end'
    back = max(now, ends[o]) if resolving[o] and ends[o] is not None else now
    T = end-back
    d.update(back=back, T=T)
    board = None
    if resolving[o]:
        fc = tracker.current(o, now) if tracker is not None and hasattr(tracker, 'current') else None
        if fc is not None and fc.has_prediction and fc.final_board is not None:
            board, d['board'] = np.asarray(fc.final_board, dtype=np.int64).reshape(ROWS, COLS), 'forecast'
        elif value['board_prediction_active'][o]:
            board, d['board'] = np.asarray(value['boards'][o], dtype=np.int64).reshape(ROWS, COLS), 'ex'
    else:
        raw = engine.state.sides[o].board
        if raw is not None:
            board, d['board'] = np.asarray(raw, dtype=np.int64).reshape(ROWS, COLS), 'now'
    if board is None:
        return math.nan, d, 'no_board_o'
    N = hands_in(board, T)
    d['N'] = N
    if N == 0:
        return 0., d, 'n0'
    pairs, pwhy = seed_pairs(engine, o, resolving[o], getattr(engine, 'retr_seed_matcher', None))
    if pairs is None:
        return math.nan, d, pwhy
    if len(pairs) < DEPTH:
        return math.nan, d, 'short_seed_pairs'
    enc = encode(board, pairs)
    if enc is None:
        return math.nan, d, 'five_colors_or_unknown'
    cm = engine.retr_ama.profile(*enc)
    return float(int(cm[min(N, DEPTH)-1])//rate), d, 'ama'


FORECAST_ENV = 'PUYO_RETR_FORECAST'


def forecast_enabled() -> bool:
    """依頼 A の連鎖の予測 (src/fire_board.ChainForecast) を使うか。学習側の既定は ON、'0' で前の FireBoardTracker に戻す。"""
    return os.environ.get(FORECAST_ENV, '1') != '0'


def fire_tracker(native: Any) -> Any:
    """撃った盤の追跡 (src/fire_board.py)。段ごとの得点は native.simulate_chain_trace_py (EX の連鎖の計算と同じ)。
    既定は両側の ChainForecast (連鎖の識別子で選ぶ・段ごとの取り直し・救済)。PUYO_RETR_FORECAST=0 で前の FireBoardTracker。"""
    from src.phase_j._live_ex_frozen.src.fire_board import FireBoardTracker, PairForecast, native_trace
    if forecast_enabled():
        return PairForecast(native_trace(native))

    def scorer(board: np.ndarray) -> list[int]:
        return [int(step[4]) for step in native.simulate_chain_trace_py(list(map(int, board.ravel())), True)]
    return FireBoardTracker(scorer)


def pending_with_fire(engine: Any, side: int, steps: tuple[int, ...],
                      exclude_side: int | None = None) -> list[int]:
    """PostV47.projection と同じ射影を、側 side の連鎖の段の得点だけ撃った盤の計算 (steps) に替えて行い、両側の予告を返す。"""
    from src.phase_j._live_ex_frozen.src.prefire_pending_v47 import project
    from src.phase_j._live_ex_frozen.src.prefire_post_v47 import remaining_send
    from src.phase_j._live_ex_frozen.src.dl1k_retr_fix import receiver_forecast
    snap, now = engine.state.sl_snap, engine.state.t
    first = snap['margin_origin_sec']
    first = engine.state.first_move_t() if first is None else first
    first = engine.state.game_start_t if first is None else first
    amounts, triggers, ends = [], [], []
    for i, chain in enumerate(engine.chains):
        active = chain is not None and not chain['finished']
        if i == side and active:
            chain = dict(chain, best=dict(chain['best'], scores=list(steps)))
        elif active and i != exclude_side:
            fc = receiver_forecast(engine, i)
            if fc is not None:
                chain = dict(chain, best=dict(chain['best'], scores=list(fc.steps)))
        scores = chain['best']['scores'] if active else []
        times = [chain['started']+FORMULA_SCORE_DELAY_SEC+k*STEP_SEC for k in range(len(scores))]
        amounts.append(0 if i == exclude_side else remaining_send(chain, snap[f'generated_p{i+1}'], first))
        triggers.append(chain['started'] if active else None)
        ends.append(max(now, times[-1]) if times else None)
    return list(project(engine.state.sl.core, amounts, now, triggers, ends)['pending'])


def game_seed(engine: Any) -> int | None:
    """試合の種 (returnable_b.Seeds.game_seed と同じ): 各側のテープの観測列 own を先頭から 4 組ずつずらした 8 組の窓で照合し、
    最初に 1 つの種に一意に当たった窓の種を集める。集まった種がちょうど 1 つなら、その試合の種として固定する (以後は照合しない)。
    観測列の後半は誤読・作り直しが混じるので、全体の照合 (EX の snapshot) より先頭の窓のほうが決まりやすい。"""
    game = engine.state.game_idx
    cached = engine.retr_seed.get(game)
    if cached is not None:
        return cached
    found = set()
    for st in engine.knowledge.tape.sides:
        own = list(st.own)
        for i in range(0, max(0, len(own)-7), 4):
            hit = engine.knowledge.lookup._match(own[i:i+8])
            if len({x for x, _ in hit}) == 1:
                found.add(hit[0][0])
                break
    if len(found) == 1:
        engine.retr_seed[game] = found.pop()
        return engine.retr_seed[game]
    return None


def seed_pairs(engine: Any, side: int, busy: bool, matcher: Any = None) -> tuple[list | None, str]:
    """受け側が次に置く組から 16 組の本物のツモ。位置は returnable_b.Seeds.future と同じ: テープの own の末尾 m 組 (6・5・4) を種の
    テープ視点の列で探し、見つかった中で abs_next に近い位置の「今の NEXT の絶対手番」p。連鎖中 (busy) なら p から、
    そうでなければ操作中の組 p−1 から (EX の knowledge.snapshot と同じ向き)。"""
    from src.phase_j._live_ex_frozen.src.tsumo_seed_identifier import abs_pair_code
    lookup = engine.knowledge.lookup
    if lookup is None:
        return None, 'no_lookup'
    seed = game_seed(engine)
    if seed is None:
        return None, 'no_seed'
    from .dl1k_retr_history import recover
    recovered = recover(engine, side, busy, seed, DEPTH)
    if recovered is not None:
        return recovered, 'ok'
    st = engine.knowledge.tape.sides[side]
    if st.queue is None or len(st.own) < 4:
        return None, 'short_history'
    table = lookup.table
    codes_all = table.tape_codes[seed]
    for m in (6, 5, 4):
        if len(st.own) < m:
            continue
        arr = np.asarray(st.own[-m:])
        codes = abs_pair_code(arr[:, 0], arr[:, 1])
        cand = ([q for q in range(0, len(codes_all)-m+1) if (codes_all[q:q+m] == codes).all()]
                if matcher is None else matcher(codes_all, codes))
        if cand:
            exp = st.abs_next()-(m-2)
            q = min(cand, key=lambda c: abs(c-exp))
            p = int(table.tape_to_absolute_index(seed, q+m-2))
            first = p if busy else p-1
            if first < 0:
                return None, 'pos_before_start'
            from .dl1k_retr_tsumo import synchronize
            first = synchronize(engine, side, busy, seed, first, DEPTH)
            return [tuple(map(int, x)) for x in table.predict_absolute(seed, first, DEPTH).tolist()], 'ok'
    return None, 'pos_nomatch'


class _RetrRun:
    """1回の計算の局所変数。試合をまたぐ状態は従来どおりengineだけが持つ。"""

    def __init__(self, engine: Any, value: dict) -> None:
        from src.phase_j._live_ex_frozen.src import dl1k_retr_fix as fix
        self.fix, self.engine, self.value = fix, engine, value
        self.extra, self.why = blank(), ['', '']
        if not hasattr(engine, 'retr_ama'):
            engine.retr_ama, engine.retr_pre, engine.retr_seed = Ama(engine.native), {}, {}
        self.now, self.game = engine.state.t, engine.state.game_idx
        rate = value.get('rate')
        self.rate = int(rate) if rate and rate > 0 else DEFAULT_RATE
        self.ends_old = [chain_end(engine, value, i, self.now) for i in SIDES]
        self.ends = [extend_side_end(engine, value, i, self.now, self.ends_old[i]) for i in SIDES]
        self.end_src, self.h_fc = ['ex', 'ex'], h_forecast_enabled()
        if self.h_fc:
            for i in SIDES:
                self.ends[i], self.end_src[i] = side_end(engine, value, i, self.now, getattr(engine, 'retr_fire', None))
        self.before_long = list(self.ends)
        self.ends = [fix.updated_end(engine, value, i, self.ends[i]) for i in SIDES]
        self.resolving = [bool(engine._resolving(i)) for i in SIDES]
        engine.retr_diag = self.diag = [None, None]
        self.sec_on, self.vis_on, self.landing_on = second_enabled(), visible_enabled(), landing_enabled()
        if self.vis_on and not hasattr(engine, 'retr_visible_pre'):
            engine.retr_visible_pre = {}

    def second(self, s: int, ret: dict, remainder: float) -> None:
        """撃った側のセカンド。計測値も従来どおり診断だけに保存する。"""
        ts = time.perf_counter()
        sec, d2, w2 = second_of(self.engine, self.value, s, self.now, self.ends, self.resolving,
                               ret, self.rate, getattr(self.engine, 'retr_fire', None))
        self.extra[f'retR_second_{NAMES[s]}'] = sec
        self.extra[f'retR_net2_{NAMES[s]}'] = -remainder-sec if math.isfinite(remainder) and math.isfinite(sec) else math.nan
        ok = math.isfinite(sec) and d2.get('N') is not None
        self.extra[f'retR_secn_{NAMES[s]}'] = float(d2['N']) if ok else math.nan
        self.extra[f'retR_sect_{NAMES[s]}'] = max(0., float(d2['T'])) if ok and d2.get('T') is not None else math.nan
        self.diag[s].update({f'sec_{k}': v for k, v in d2.items()}, sec_why=w2, sec_rate=self.rate,
                            sec_ms=round((time.perf_counter()-ts)*1000., 2))

    def incoming(self, s: int) -> int:
        """旧射影をそのまま計算する。OWN_CHAINの二重相殺回避は別経路で行う。"""
        engine, o, now = self.engine, 1-s, self.now
        incoming = int(self.value['pending'][s])
        chain = engine.chains[o]
        original, fire_back, fc_id, fc_tot = incoming, None, None, [None, None]
        tracker = getattr(engine, 'retr_fire', None)
        if tracker is not None and hasattr(tracker, 'current'):
            for i in SIDES:
                fc = tracker.current(i, now)
                fc_tot[i] = None if fc is None or not fc.has_prediction else fc.total
        if self.resolving[o] and chain is not None and tracker is not None:
            if hasattr(tracker, 'current'):
                fc = tracker.current(o, now)
                if fc is not None and fc.has_prediction:
                    incoming, fire_back, fc_id = int(pending_with_fire(engine, o, fc.steps)[s]), fc.source, fc.chain_id
            else:
                fb = tracker.for_chain(o, float(chain['started']), now)
                if fb is not None:
                    incoming, fire_back = int(pending_with_fire(engine, o, fb.steps)[s]), fb.back
        self.diag[s] = dict(I=incoming, I_pred=original, fire_back=fire_back, fc_id=fc_id,
            fc_total_o=fc_tot[o], fc_total_s=fc_tot[s], res_s=self.resolving[s], res_o=self.resolving[o],
            end_o=self.ends[o], t0_o=None if chain is None else float(chain['started']),
            n_obs_o=None if chain is None else max((ob['count'] for ob in chain.get('observed', [])), default=0),
            n_pred_o=None if chain is None else len(chain['best']['scores']))
        if self.fix.enabled(self.fix.RECEIVER_FORECAST_ENV):
            receiver = self.fix.receiver_forecast(engine, s)
            self.diag[s]['receiver_forecast_steps'] = None if receiver is None else len(receiver.steps)
        if self.fix.enabled('PUYO_RETR_FIRE_ORIGIN_HOLD'):
            self.diag[s].update(origin_s=(engine.chains[s] or {}).get('retr_origin'),
                                origin_o=(engine.chains[o] or {}).get('retr_origin'))
        return incoming

    def hands(self, s: int, incoming: int) -> int | None:
        """残り手数と既存予測との差を計算する。未知の締切は旧経路では欠測。"""
        o, engine = 1-s, self.engine
        tracker = getattr(engine, 'retr_fire', None)
        if self.h_fc:
            fc = tracker.current(o, self.now) if tracker is not None and hasattr(tracker, 'current') else None
            self.diag[s].update(end_src=self.end_src[o], end_src_s=self.end_src[s],
                end_o_old=self.ends_old[o], h_old=None, n_fc_o=len(fc.steps) if fc is not None and fc.has_prediction else None)
            if self.resolving[o] and self.ends_old[o] is not None:
                start = self.ends_old[s] if self.resolving[s] and self.ends_old[s] is not None else self.now
                self.diag[s]['h_old'] = min(MAX_H, max(0, int((self.ends_old[o]+SEC_PER_HAND-start+self.fix.HAND_EPSILON)/SEC_PER_HAND)))
        if self.resolving[o] and self.ends[o] is not None:
            start = self.ends[s] if self.resolving[s] and self.ends[s] is not None else self.now
            h = min(MAX_H, max(0, int((self.ends[o]+SEC_PER_HAND-start+self.fix.HAND_EPSILON)/SEC_PER_HAND)))
            self.diag[s]['h'] = h
            return h
        if self.resolving[o]:
            self.why[s] = 'attacker_no_prediction'
        elif incoming > 0:
            return 1
        else:
            self.why[s] = 'no_attack'
        return None

    def board(self, s: int) -> np.ndarray | None:
        """旧経路の盤選択と持ち越し。新しいOWN_CHAINはこの経路を通さない。"""
        engine = self.engine
        if self.resolving[s]:
            if self.value['board_prediction_active'][s]:
                return np.asarray(self.value['boards'][s], dtype=np.int64).reshape(ROWS, COLS)
            pre = engine.retr_pre.get(s)
            if pre is not None and pre[0] == self.game:
                for stem, val in zip(STEMS, pre[1]):
                    self.extra[f'{stem}_{NAMES[s]}'] = val
                self.why[s] = 'carry'
                if self.vis_on:
                    for stem, val in zip(VISIBLE_STEMS, engine.retr_visible_pre.get(s, (math.nan,)*len(VISIBLE_STEMS))):
                        self.extra[f'{stem}_{NAMES[s]}'] = val
                if self.sec_on:
                    self.second(s, dict(kind='busy'), float(pre[1][1]))
            else:
                self.why[s] = 'no_final_no_pre'
            return None
        raw = engine.state.sides[s].board
        if raw is None:
            self.why[s] = 'no_board'
            return None
        return np.asarray(raw, dtype=np.int64).reshape(ROWS, COLS)

    def capacity(self, s: int, board: np.ndarray, h: int, pending: int) -> tuple:
        """旧探索の盤・種・幅・深さ・換算順序を維持する。"""
        engine, capacity, ret, cummax = self.engine, None, None, None
        if h == 0:
            capacity, self.why[s], ret = 0, 'h0', dict(kind='none')
        elif not self.resolving[s]:
            count, score = chain_score(engine.native, board)
            if count > 0:
                capacity, self.why[s], ret = score//self.rate, 'board_chain', dict(kind='board_chain', steps=count)
        if capacity is None:
            pairs, pwhy = seed_pairs(engine, s, self.resolving[s], getattr(engine, 'retr_seed_matcher', None))
            if pairs is None:
                self.why[s] = pwhy
            elif len(pairs) < DEPTH:
                self.why[s] = 'short_seed_pairs'
            else:
                enc = encode(board, pairs)
                if enc is None:
                    self.why[s] = 'five_colors_or_unknown'
                else:
                    cummax = (engine.retr_ama.profile(*enc, pending=pending, rate=self.rate)
                              if self.landing_on else engine.retr_ama.profile(*enc))
                    capacity, self.why[s] = int(cummax[min(h, DEPTH)-1])//self.rate, 'ama'
                    ret = dict(kind='ama', cummax=list(cummax), h=h) if cummax[min(h, DEPTH)-1] > 0 else dict(kind='none')
        return capacity, ret, cummax

    def store(self, s: int, board: np.ndarray, incoming: int, h: int,
              capacity: float, ret: dict, cummax: list | None) -> None:
        """旧列の値・順序と次回持ち越しを保存する。"""
        remainder = incoming-capacity
        lethal = float(remainder > 0 and heights(board)[DEATH_COL]+min(max(remainder, 0), DROP_CAP)//COLS >= DEATH_H)
        vals = (float(capacity), float(remainder), float(h), lethal)
        for stem, val in zip(STEMS, vals):
            self.extra[f'{stem}_{NAMES[s]}'] = val
        if self.vis_on:
            visible = visible_values(float(capacity), incoming, h, self.resolving[s], board, cummax, self.rate)
            for stem, val in zip(VISIBLE_STEMS, visible):
                self.extra[f'{stem}_{NAMES[s]}'] = val
            if not self.resolving[s]:
                self.engine.retr_visible_pre[s] = visible
        if self.sec_on:
            self.second(s, dict(kind='busy') if self.resolving[s] else ret, float(remainder))
        if not self.resolving[s]:
            self.engine.retr_pre[s] = (self.game, vals)

    def side(self, s: int) -> None:
        """受け側1つを計算する。旗OFFは旧分岐順、ONは既発火分を別会計にする。"""
        engine, fix = self.engine, self.fix
        pending = confirmed_pending(engine, s) if self.landing_on else 0
        if not self.resolving[s]:
            engine.retr_pre[s] = (self.game, (math.nan,)*4)
            if self.vis_on:
                engine.retr_visible_pre[s] = (math.nan,)*len(VISIBLE_STEMS)
        incoming = self.incoming(s)
        if self.landing_on:
            self.diag[s]['pending'] = pending
        if fix.enabled(fix.LONG_ATTACK_ENV):
            self.diag[s].update(end_before_long=self.before_long[1-s], end_s=self.ends[s], long_attack_h=True)
        if fix.enabled(fix.OWN_CHAIN_ENV) and self.resolving[s]:
            self.why[s], remainder = fix.apply_own(engine, self.value, s, self.ends, self.extra, self.diag[s], self.rate)
            if self.sec_on and self.why[s] != 'no_attack':
                self.second(s, dict(kind='busy'), remainder)
            return
        h = self.hands(s, incoming)
        if h is None:
            return
        board = self.board(s)
        if board is None:
            return
        from .dl1k_retr_fall import apply as apply_fall
        if apply_fall(self, s, board, incoming, h):
            return
        from .dl1k_retr_continue import apply
        if apply(self, s, board, incoming, h):
            return
        capacity, ret, cummax = self.capacity(s, board, h, pending)
        if capacity is not None:
            self.store(s, board, incoming, h, capacity, ret, cummax)


def compute_retr(engine: Any, value: dict) -> tuple[dict, list[str]]:
    """射影の直後・死亡証明の前の状態から、返しの列と側ごとの理由を返す。"""
    run = _RetrRun(engine, value)
    from .dl1k_retr_tsumo import enabled
    if enabled():
        engine.retr_sync_diag = {}
    for side in SIDES:
        run.side(side)
        if enabled() and run.diag[side] is not None:
            run.diag[side]['tsumo_sync'] = engine.retr_sync_diag.get(side)
    return run.extra, run.why
