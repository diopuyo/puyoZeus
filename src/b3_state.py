"""B3 因果的な状態機械: B2 の観測フレームとイベントを時刻順に畳み込む。

不変条件: 状態は「処理済みフレーム/イベントの available_at <= 現在時刻」の情報だけで更新される。
未来のフレーム・イベント・lean 行は参照しない (prefix 不変テストで検証)。
各状態量には取得時刻 (available_at) を併記し、行の av_* 列へ出す。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Deque, Optional

import numpy as np

from src import puyo_core_bridge as native
from src import b3_trial_config as production_config
from src.board import Board
from src.b3_column_spec import ICON_TYPES
from src.b3_source import LeanTable
from src.b3_chain_prediction import EpisodePrediction, fire_board_option, fire_board_steps, origin_candidates
from src.fire_board import ChainForecast, RawHistory
from src.phase_j.live_fire_board_config import enabled as fire_board_enabled
from src.b3_score_ledger_port import new_ledger, validate_snapshot
from src.production_config import GHOST_CHAIN_RULE_ENABLED

SIDE_NAMES: tuple[str, str] = ("1P", "2P")
ICON_WINDOW = 15            # 補正用の時間窓 (フレーム数, 約0.5秒@30fps)
ICON_MIN_SAMPLES = 5
ICON_TYPE_ALIAS = {"crown": "crown", "big_crown": "crown", "moon": "moon", "star": "star",
                   "rock": "rock", "large": "large", "small": "small"}
EPISODE_CLOSE_SEC = 1.0     # CHAIN/GRAVITY_SETTLE が無い時間でエピソードを閉じる (統括: 発火後1秒)
STABLE_RUN_FRAMES = 2
LEAN_START_SLACK_SEC = 0.5
MATCH_TIME_EPS = 1e-4       # lean の float32 時刻丸めを吸収する
NEVER_CHAINED_CLOSE_SEC = 3.0   # CHAIN フレームが現れない発火断片を閉じる
LAND_CLOSE_SEC = 1.0        # OJAMA_FALL が途切れてから着弾区間を確定するまで
STEP_BACKFILL_SEC = 0.05    # 式の段が fire_start より先に届く場合の許容
PLACE_SPLIT_MARGIN_SEC = 0.3   # 依頼 §4: 段の無いエピソードで、発火からこれより後の置きだけを次の連鎖の置きとみなす
# 次の回 2 (PC1 10/30): 自分の連鎖中は操作中のツモが無い (user の決まり)。最後の有効な式から 0.5 秒未満の tsumo_count の増えは
# 置きと数えない (wt_r src/fire_board.PLACE_GAP_SEC と同じ。60 本で空振り 148 → 4 / 8,747)。
PLACE_GAP_SEC = 0.5
RECENT_STEP_KEEP_SEC = 5.0
FORECAST_MATCH_SEC = 1.0    # 依頼 A: 連鎖の予測の識別子 (最初の式の時刻) がエピソードの最初の発火 − この秒数以後なら同じ連鎖


def forecast_enabled() -> bool:
    """依頼 A の連鎖の予測 (src/fire_board.ChainForecast) を使うか。学習側の既定は ON、PUYO_B3_FORECAST=0 で前の撃った盤の扱いに戻す。
    配信では撃った盤の旗 (live_b3_fire_board) が ON の時だけ使う (OFF なら撃った盤そのものを使わない)。"""
    import os
    return os.environ.get("PUYO_B3_FORECAST", "1") != "0"
UB_WINDOW_SEC = 30.0
CHAINISH = frozenset({"CHAIN", "GRAVITY_SETTLE"})
QUIET = frozenset({"STABLE", "TSUMO_FALL", "OJAMA_FALL"})
COLOR_CELLS = (1, 2, 3, 4, 5)
MIN_COLOR_COUNT = 3


@lru_cache(maxsize=4096)
def chain_prediction(raw: bytes) -> tuple[int, float, bytes]:
    """起点盤面の連鎖シミュレーション (連鎖数, 完走得点, 完走後盤面)。幽霊連鎖規則込み。"""
    grid = np.frombuffer(raw, dtype=np.int8).reshape(13, 6)
    r = native.simulate_chain(Board.from_list(grid.astype(int).tolist()),
                              exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    final = np.asarray(r.final_board._grid, dtype=np.int8)
    return int(r.chain_count), float(r.exact_score), final.tobytes()


def color_count(grid: Optional[np.ndarray]) -> float:
    if grid is None:
        return float("nan")
    return float(np.isin(grid, COLOR_CELLS).sum())


def color_conflict(grid: Optional[np.ndarray]) -> bool:
    """4色の試合で5色が同時に見える盤面は認識矛盾。"""
    from src.match_color_evidence import GAME_COLOR_COUNT
    return grid is not None and len(set(np.unique(grid)) & set(COLOR_CELLS)) > GAME_COLOR_COUNT


@dataclass
class Fragment:
    fire_id: str
    start_t: float
    chain_before: Optional[np.ndarray]
    pre: Optional[dict]
    complete_t: Optional[float] = None
    consumed_board: Optional[float] = None
    consumed_formula: Optional[float] = None


@dataclass
class Episode:
    """連続する fire_start/fire_complete 断片を1エピソードにまとめる (断片化対策)。"""
    first_trigger: float
    start_t: float
    frags: list[Fragment] = field(default_factory=list)
    steps: list[tuple[float, float, float]] = field(default_factory=list)  # (t, erased, product)
    seen_chain: bool = False
    stable_run: int = 0
    last_chainish_t: float = float("nan")
    sig_stable_t: Optional[float] = None
    sig_next_t: Optional[float] = None
    sig_ojama_t: Optional[float] = None
    closed_t: Optional[float] = None
    censored: bool = False
    pred_l: float = float("nan")
    pred_score: float = float("nan")
    pred_final: Optional[bytes] = None
    denominator: float = float("nan")
    prediction: EpisodePrediction = field(default_factory=EpisodePrediction)

    def update_prediction(self) -> None:
        """公開値は常に同じ候補の組にする。"""
        self.pred_l = self.prediction.length
        self.pred_score = self.prediction.score
        self.pred_final = self.prediction.final

    @property
    def ended_t(self) -> Optional[float]:
        ts = [x for x in (self.sig_stable_t, self.sig_next_t, self.sig_ojama_t) if x is not None]
        return min(ts) if ts else None

    @property
    def active(self) -> bool:
        return self.closed_t is None and self.ended_t is None and not self.censored

    def board_consumed(self) -> Optional[float]:
        vals = [f.consumed_board for f in self.frags]
        return None if (not vals or any(v is None for v in vals)) else float(sum(vals))

    def steps_complete(self) -> bool:
        return bool(self.steps) and np.isfinite(self.pred_l) and len(self.steps) == int(self.pred_l)


class IconCorrector:
    """画面アイコン読みの補正: 試合中ゲート + 時間窓の種類別中央値 (誤りは過大側のため)。"""

    def __init__(self) -> None:
        self.window: Deque[dict[str, float]] = deque(maxlen=ICON_WINDOW)
        self.totals: Deque[float] = deque(maxlen=ICON_WINDOW * 2)
        self.raw: dict[str, float] = {t: 0.0 for t in ICON_TYPES}
        self.raw_total = float("nan")
        self.last_icons: list = []
        self.t = float("nan")

    def push(self, t: float, icon: Optional[dict], gate: bool) -> None:
        if icon is None:
            return
        types = {k: 0.0 for k in ICON_TYPES}
        for it in icon.get("icons", []):
            key = ICON_TYPE_ALIAS.get(it.get("icon_type"))
            if production_config.B3_ICON_LINE_ALIAS_DEFAULT and it.get("icon_type") == "line":
                key = "large"
            if key:
                types[key] += float(it.get("count_value", 0))
        self.raw, self.raw_total, self.t = types, float(icon.get("total_count", 0)), t
        self.last_icons = list(icon.get("icons", []))
        if gate:
            self.window.append(types)
            self.totals.append(self.corrected_total())

    def corrected(self) -> Optional[dict[str, float]]:
        if len(self.window) < ICON_MIN_SAMPLES:
            return None
        return {k: float(np.median([w[k] for w in self.window])) for k in ICON_TYPES}

    def corrected_total(self) -> float:
        c = self.corrected()
        if c is None:
            return float("nan")
        return float(sum(c[k] for k in ICON_TYPES))  # count_value は個数換算済み

    def reset(self) -> None:
        self.window.clear()
        self.totals.clear()


@dataclass
class Landing:
    finalize_t: float
    icon_drop: float
    board_net: float


class SideState:
    """片側の因果状態。"""

    def __init__(self, name: str, enable_fire_board: bool | None = None) -> None:
        from src.chain_continuity import ChainBoundary, end_enabled
        self.end_formula = end_enabled()
        self.boundary = ChainBoundary() if self.end_formula else None
        self.continuity_side: dict = {}
        self.pending_fire: Optional[dict] = None
        self.name = name
        self.enable_fire_board = fire_board_enabled() if enable_fire_board is None else enable_fire_board
        self.state = ""
        self.board: Optional[np.ndarray] = None
        self.board_t = float("nan")
        self.score: Optional[float] = None
        self.score_conflict = False
        self.score_max: Optional[float] = None
        self.score_hist: Deque[tuple[float, float]] = deque()
        self.slide = False
        self.icon = IconCorrector()
        self.queue: tuple[int, ...] = (0, 0, 0, 0)
        self.queue_t = float("nan")
        self.lean_ptr = -1
        self.lean_row = -1
        self.first_place_t: Optional[float] = None
        self.recent_steps: Deque[tuple[float, float, float]] = deque()
        self.episode: Optional[Episode] = None
        self.closed: list[Episode] = []
        self.fire_t = float("nan")      # 最後に使った発火系イベントの available_at
        self.formula_t = float("nan")
        self.fall_start: Optional[float] = None
        self.fall_last_t = float("nan")
        self.fall_ojama_before = float("nan")
        self.fall_icon_before = float("nan")
        self.landing: Optional[Landing] = None
        self.land_count = 0
        self.board_changed = False
        self.prediction_colors: tuple[int, ...] = ()
        self.raw_hist = RawHistory()     # scene_68: 毎コマの画面の読みの盤 (撃った盤を取るため)
        # 依頼 A: EX/retR と同じ連鎖の予測。配信の旗が OFF なら作らない (今と同じ)
        self.forecast = ChainForecast(name=name) if self.enable_fire_board and forecast_enabled() else None
        # 依頼 §4 (PC1 10/30、既定 OFF: PUYO_CHAIN_SPLIT_PLACE=1): エピソードの最後の段 (無ければ発火) の後にこの側の置き
        # (観測の tsumo_count の増え) があれば、次の発火・式は別のエピソード (b3_score_ledger._splits_chain と同じ考え)。
        import os
        self.split_on_place = os.environ.get("PUYO_CHAIN_SPLIT_PLACE", "0") == "1"
        self.tsumo: Optional[int] = None
        self.placed_t: Optional[float] = None
        self.formula_valid_t: Optional[float] = None   # 次の回 2: 最後に有効な式が見えた時刻

    # --- フレーム更新 ---
    def update_frame(self, t: float, side: dict, gate: bool) -> None:
        previous_score = self.score_max if self.end_formula else self.score
        if self.boundary is not None:
            self.boundary.observe(t, side)
            self.continuity_side = side
        self.state = side["state"]
        if self.enable_fire_board:
            self.raw_hist.push(t, side.get("raw_cnn_board"))
        if self.forecast is not None:
            self.forecast.observe(t, side.get("formula"), side.get("raw_cnn_board"), side=side)
        self.slide = bool(side.get("next_slide_motion"))
        if self.split_on_place:
            tc = side.get("tsumo_count")
            if tc is not None:
                if self.tsumo is not None and int(tc) > self.tsumo and (
                        self.formula_valid_t is None or t - self.formula_valid_t >= PLACE_GAP_SEC):
                    self.placed_t = t   # 次の回 2: 式が出ている間 (連鎖の途中) の増えは読みの揺れとして数えない
                self.tsumo = int(tc)
            form = side.get("formula") or {}
            if form.get("valid") and form.get("product") is not None:
                self.formula_valid_t = t
        sc = side.get("score")
        self.score_conflict = sc is not None and self.score_max is not None and float(sc) < self.score_max
        self.score = self.score_max if self.score_conflict else None if sc is None else float(sc)
        if self.score is not None:
            self.score_max = self.score
        if self.score is not None and gate:
            self.score_hist.append((t, self.score))
            while self.score_hist and t - self.score_hist[0][0] > UB_WINDOW_SEC:
                self.score_hist.popleft()
        self.icon.push(t, side.get("icon"), gate)
        board = side.get("confirmed_board")
        self.board_changed = False
        if board is not None:
            arr = np.asarray(board, dtype=np.int8)
            if self.board is None or not np.array_equal(arr, self.board):
                self.board_changed = True
            self.board = arr
            self.board_t = t
        self._update_episode(t)
        self._update_landing(t, gate)
        if self.end_formula:
            self._continue_score(t, previous_score, gate)
            self._continue_raw_formula(t, side, gate)

    def _update_episode(self, t: float) -> None:
        if self.end_formula:
            self._formula_episode_end(t)
            return
        ep = self.episode
        if ep is None or ep.closed_t is not None:
            return
        if self.state in CHAINISH:
            ep.last_chainish_t = t
        if self.state == "CHAIN":
            ep.seen_chain, ep.stable_run = True, 0
            ep.sig_stable_t = ep.sig_next_t = ep.sig_ojama_t = None
        elif ep.seen_chain and self.state in QUIET:
            ep.stable_run += 1
            if ep.stable_run >= STABLE_RUN_FRAMES and ep.sig_stable_t is None:
                ep.sig_stable_t = t
            if self.state == "OJAMA_FALL" and ep.sig_ojama_t is None:
                ep.sig_ojama_t = t
            if self.slide and self.state in ("STABLE", "TSUMO_FALL") and ep.sig_next_t is None:
                ep.sig_next_t = t
        never_chained = not ep.seen_chain and t - ep.start_t >= NEVER_CHAINED_CLOSE_SEC
        if never_chained or (ep.seen_chain and np.isfinite(ep.last_chainish_t)
                             and t - ep.last_chainish_t >= EPISODE_CLOSE_SEC):
            ep.closed_t = t
            self.closed.append(ep)

    def _formula_episode_end(self, t: float) -> None:
        ep = self.episode
        if ep is None or ep.closed_t is not None:
            return
        if self.boundary.t > ep.first_trigger:
            if self.boundary.kind == 'next':
                ep.sig_next_t = self.boundary.t
            elif self.boundary.kind == 'ojama':
                ep.sig_ojama_t = self.boundary.t
            ep.closed_t = t
            self.closed.append(ep)
        elif self.state in CHAINISH:
            ep.seen_chain, ep.last_chainish_t = True, t
        if self.forecast is not None:
            self._apply_forecast(t)
            ep.update_prediction()

    def _continue_raw_formula(self, t: float, side: dict, gate: bool) -> None:
        form = side.get('formula') or {}
        if gate and form.get('valid') and form.get('product') is not None:
            self._continue_formula(t, t)

    def close_match(self, t: float) -> None:
        """確定済みの試合終了は、NEXT が来なくても活動を残さない。"""
        if not self.end_formula:
            return
        self.boundary.t, self.boundary.kind = t, 'match_end'
        ep = self.episode
        if ep is not None and ep.closed_t is None:
            ep.closed_t = t
            self.closed.append(ep)

    def _continue_score(self, t: float, previous: Optional[float], gate: bool) -> None:
        from src.chain_continuity import MIN_CHAIN_SCORE
        if not gate or previous is None or self.score is None or self.score - previous < MIN_CHAIN_SCORE:
            return
        ep = self.episode
        if ep is not None and self.boundary.t > ep.first_trigger:
            if self.pending_fire is None or float(self.pending_fire['trigger_sec']) <= self.boundary.t:
                return  # 終了後の遅れた得点だけで前の連鎖を開き直さない。
        self._continue_formula(t, t)

    def _continue_formula(self, started: float, available: float) -> None:
        ep = self.episode
        if ep is not None and not ep.censored and self.boundary.t <= ep.first_trigger:
            ep.sig_stable_t, ep.closed_t = None, None
            self.closed = [old for old in self.closed if old is not ep]
            return
        from src.chain_continuity import confirmed_fire
        before = confirmed_fire(self.continuity_side, started, available)
        self.on_fire_start(dict(fire_id=f'{self.name}:formula:{started}', trigger_sec=started,
                               observed_at_sec=available, chain_before=before, pre=None,
                               continuity_evidence=True))

    def _update_landing(self, t: float, gate: bool) -> None:
        if production_config.b3_ledger_landing_enabled():
            return
        if gate and self.state == "OJAMA_FALL":
            if self.fall_start is None:
                self.fall_start = t
                self.fall_ojama_before = _ojama_cells(self.board)
                tot = [x for x in self.icon.totals if np.isfinite(x)]
                self.fall_icon_before = max(tot) if tot else float("nan")
            self.fall_last_t = t
        elif self.fall_start is not None and (not gate or t - self.fall_last_t >= LAND_CLOSE_SEC):
            after = self.icon.corrected_total()
            net = _ojama_cells(self.board) - self.fall_ojama_before
            # 落下状態の誤検出では履歴を更新せず、受け側盤面の純増を必須にする。
            if net > 0:
                self.landing = Landing(t, self.fall_icon_before - after, net)
                self.land_count += 1
            self.fall_start = None

    # --- イベント更新 ---
    def _placed_after(self, ep: Episode) -> bool:
        """依頼 §4 (旗 ON の時だけ): エピソードの最後の段 (無ければ最初の発火) より後にこの側が置いたか。"""
        # 段が無い時は、連鎖を始めた組の置きと区別するため発火 + PLACE_SPLIT_MARGIN_SEC より後だけ
        last = ep.steps[-1][0] if ep.steps else ep.first_trigger + PLACE_SPLIT_MARGIN_SEC
        return self.split_on_place and self.placed_t is not None and self.placed_t > last

    def _close_on_place(self, t: float) -> None:
        """依頼 §4: 置きの後の発火・式が来たら、開いているエピソードを閉じる (次は別のエピソード)。"""
        if self.end_formula:
            return  # 新方式は NEXT と盤面の裏付けを _formula_episode_end で確認する。
        ep = self.episode
        if ep is not None and ep.closed_t is None and not ep.censored and self._placed_after(ep):
            ep.closed_t = t
            self.closed.append(ep)

    def on_formula(self, e: dict) -> None:
        step = (float(e["t_sec"]), float(e["erased"]), float(e["product"]))
        if self.end_formula:
            if self.boundary.kind == 'match_end' and step[0] <= self.boundary.t:
                return
            self._continue_formula(step[0], float(e['observed_at_sec']))
            if not self.episode.steps:
                self.episode.first_trigger = min(self.episode.first_trigger, step[0])
        self.recent_steps.append(step)
        while self.recent_steps and step[0] - self.recent_steps[0][0] > RECENT_STEP_KEEP_SEC:
            self.recent_steps.popleft()
        self._close_on_place(float(e["observed_at_sec"]))
        if self.episode is not None and self.episode.closed_t is None and self.forecast is not None:
            ep = self.episode   # 依頼 A: 置き場所の補完の候補 (pred_origin) は今までどおり段で絞り、予測は連鎖の予測があればそれを使う
            ep.steps.append(step)
            origin = getattr(ep, "pred_origin", None) or ep.prediction
            origin.observe(step[2], float(e["observed_at_sec"]))
            ep.pred_origin = origin
            self._apply_forecast(float(e["observed_at_sec"]))
            ep.update_prediction()
        elif self.episode is not None and self.episode.closed_t is None:
            self.episode.steps.append(step)
            self.episode.prediction.observe(step[2], float(e["observed_at_sec"]))
            backup = getattr(self.episode, "pred_backup", None)
            if backup is not None:   # 撃った盤に替えた後も、元の候補を同じ段で絞り続ける (撃った盤が後の段と食い違った時に戻すため)
                backup.observe(step[2], float(e["observed_at_sec"]))
                if not self.episode.prediction.options and backup.options:
                    self.episode.prediction, self.episode.pred_backup = backup, None
                    self.episode.fire_board_conflict = True
            self.episode.update_prediction()
            self._fire_board(float(e["observed_at_sec"]))
        self.formula_t = float(e["observed_at_sec"])

    def on_fire_start(self, e: dict) -> None:
        if self.end_formula and float(e['trigger_sec']) <= self.boundary.t:
            return  # 境界より古い baseline の再通知は新しい連鎖ではない。
        if self.end_formula and not self._observed_start(e):
            self.pending_fire = dict(e)
            return
        before = None if e.get("chain_before") is None else np.asarray(e["chain_before"], dtype=np.int8)
        frag = Fragment(e["fire_id"], float(e["observed_at_sec"]), before, e.get("pre"))
        self._close_on_place(float(e["observed_at_sec"]))
        ep = self.episode
        if self.end_formula and ep is not None and not ep.censored and self.boundary.t <= ep.first_trigger:
            ep.closed_t = ep.sig_stable_t = None
            self.closed = [old for old in self.closed if old is not ep]
        if ep is None or ep.closed_t is not None or ep.censored:
            ep = Episode(float(e["trigger_sec"]), float(e["observed_at_sec"]))
            ep.steps = [s for s in self.recent_steps if s[0] >= ep.first_trigger - STEP_BACKFILL_SEC
                        and (not self.end_formula or s[0] > self.boundary.t)]
            options, reason = origin_candidates(before, self.prediction_colors, self.queue)
            ep.prediction = EpisodePrediction(options=options, reason=reason)
            ep.prediction.publish(float(e["observed_at_sec"]))
            for step in ep.steps:
                ep.prediction.observe(step[2], float(e["observed_at_sec"]))
            ep.update_prediction()
            if before is not None:
                ep.denominator = color_count(before)
            self.episode = ep
        ep.frags.append(frag)
        if self.forecast is not None:
            if getattr(ep, "pred_origin", None) is None:
                ep.pred_origin = ep.prediction
            self._apply_forecast(float(e["observed_at_sec"]))
            ep.update_prediction()
        elif ep.steps:
            self._fire_board(float(e["observed_at_sec"]))
        self.fire_t = float(e["observed_at_sec"])

    def _observed_start(self, e: dict) -> bool:
        """状態・baseline 由来の候補だけでは新しい活動区間を作らない。"""
        if self.episode is not None and self.episode.active:
            return True
        if e.get('continuity_evidence'):
            self.pending_fire = None
            return True
        formula = self.continuity_side.get('formula') or {}
        if formula.get('valid') and formula.get('product') is not None:
            return True
        return any(s[0] > self.boundary.t and s[0] >= float(e['trigger_sec']) - STEP_BACKFILL_SEC
                   for s in self.recent_steps)

    def _apply_forecast(self, available_at: float) -> None:
        """依頼 A: 今の連鎖の予測 (ChainForecast、EX/retR と同じ) がこのエピソードの連鎖なら、その段ごとの得点・最終盤を予測の候補 1 つにする。
        段の食い違いは ChainForecast が取り直し・救済で直すので、ここでは式で絞らない (観測済みの段の得点は ep.steps の合計)。
        予測が無い時は置き場所の補完の候補 (pred_origin) に戻す。"""
        ep = self.episode
        if ep is None or ep.closed_t is not None or ep.censored:
            return
        origin = getattr(ep, "pred_origin", None) or ep.prediction
        fc = self.forecast.current()
        if fc is None or not fc.has_prediction or fc.final_board is None or fc.chain_id < ep.first_trigger - FORECAST_MATCH_SEC:
            ep.prediction, ep.forecast = origin, None
            return
        prefix = tuple(int(x) for x in np.cumsum(fc.steps))
        option = dict(prefix=prefix, colors=(), score=int(prefix[-1]), board=np.asarray(fc.final_board).astype(int).tolist(), weight=1)
        prediction = EpisodePrediction(options=[option], reason="")
        prediction.observed_count = len(ep.steps)
        prediction.observed_score = float(sum(s[2] for s in ep.steps))
        prediction.publish(available_at)
        ep.prediction, ep.forecast = prediction, fc

    def _fire_board(self, available_at: float) -> None:
        """scene_68 の直し: 連鎖の 1 段目の式が出たら、そのコマの 1〜6 コマ前の画面の読みの盤から、計算の 1 段目の得点が式の product と
        一致する最初の盤を撃った盤とし、連鎖の予測をその盤の完走候補 1 つに替える (置き場所の推測はしない)。1 つの連鎖で 1 回だけ試す。"""
        ep = self.episode
        if not self.enable_fire_board:
            return
        if ep is None or ep.closed_t is not None or ep.censored or getattr(ep, "fire_board_tried", False) or not ep.steps:
            return
        ep.fire_board_tried = True
        t_formula, _, product1 = ep.steps[0]
        fb = self.raw_hist.find(t_formula, int(product1), fire_board_steps)
        option = None if fb is None else fire_board_option(fb.board)
        if option is None:
            return
        prediction = EpisodePrediction(options=[option], reason="")
        prediction.publish(available_at)
        for step in ep.steps:
            prediction.observe(step[2], available_at)
        if not prediction.options:   # 観測済みの段と食い違う撃った盤は使わない (元の候補のまま)
            return
        ep.fire_board, ep.pred_backup, ep.prediction = fb, ep.prediction, prediction
        ep.update_prediction()

    def on_fire_complete(self, e: dict) -> None:
        ep = self.episode
        if ep is not None:
            for f in ep.frags:
                if f.fire_id == e["fire_id"]:
                    f.complete_t = float(e["observed_at_sec"])
                    f.consumed_board = e.get("consumed_board_delta")
                    f.consumed_formula = e.get("consumed_formula")
        self.fire_t = float(e["observed_at_sec"])

    def on_fire_censored(self, e: dict) -> None:
        if self.episode is not None and self.episode.closed_t is None:
            self.episode.censored = True
        self.fire_t = float(e["observed_at_sec"])


def _predict(before: np.ndarray) -> tuple[float, float, bytes]:
    n, score, final = chain_prediction(np.ascontiguousarray(before, dtype=np.int8).tobytes())
    return float(n), score, final


def _ojama_cells(grid: Optional[np.ndarray]) -> float:
    return float("nan") if grid is None else float((grid == 9).sum())


class ColorTracker:
    """試合内で因果的に観測した色の頻度 (盤面・NEXT)。4色が出揃うまでは5色探索。"""

    def __init__(self) -> None:
        self.counts = np.zeros(6, dtype=np.int64)

    def add_board(self, grid: np.ndarray) -> None:
        for c in COLOR_CELLS:
            self.counts[c] += int((grid == c).sum())

    def add_queue(self, queue: tuple[int, ...]) -> None:
        for c in queue:
            if 1 <= c <= 5:
                self.counts[c] += 1

    def known(self) -> tuple[int, tuple[int, ...]]:
        """(観測色数, 探索に使う色集合)。"""
        seen = [c for c in COLOR_CELLS if self.counts[c] >= MIN_COLOR_COUNT]
        if len(seen) < 4:
            return len(seen), COLOR_CELLS
        top = sorted(sorted(seen, key=lambda c: (-self.counts[c], c))[:4])
        return 4, tuple(top)


class GameState:
    """両側+試合共通の因果状態。"""

    def __init__(self, lean: LeanTable, match_mapping: Optional[dict[int, dict]] = None) -> None:
        self.lean = lean
        self.match_mapping = match_mapping
        self._match_key: Optional[tuple[int, float]] = None
        self._accept_match_observations = False
        self.lean_rows = [lean.rows_of(s) for s in SIDE_NAMES]
        self.sides = [SideState(n) for n in SIDE_NAMES]
        self.colors = ColorTracker()
        self.game_idx = -1
        self.game_start_t = 0.0
        self.t = float("nan")
        self.frame = -1
        self.av_frame = float("nan")
        self.in_match = False
        self.ledger: dict = {}
        self.sl = new_ledger()
        self.sl_snap: Optional[dict] = None

    def reset_game(self, game_idx: int) -> None:
        from src.b3_features import clear_static_caches
        clear_static_caches()
        self.game_idx = game_idx
        self.game_start_t = self.t if np.isfinite(self.t) else 0.0
        self.sides = [SideState(n) for n in SIDE_NAMES]
        self.colors = ColorTracker()
        self.sl = new_ledger()
        self.sl_snap = None

    def update_frame(self, d: dict, placed_sides: tuple[str, ...] = ()) -> bool:
        """フレームを適用。placed_sides = 同フレームの置き終わり信号の側名 (台帳v2へ渡す)。試合が切り替わったら True。"""
        self.ledger = d.get("ledger") or {}
        self.t, self.frame, self.av_frame = float(d["t_sec"]), int(d["frame"]), float(d["available_at_sec"])
        new_game = d["game_idx"] != self.game_idx
        if new_game:
            self.reset_game(int(d["game_idx"]))
            self.game_start_t = self.t
        s1, s2 = d["sides"]
        if d.get('match_end_locked'):
            for st in self.sides:
                st.close_match(self.t)
        active = bool(d.get("match_active")) and s1["state"] != "MENU" and s2["state"] != "MENU" and (
            s1.get("score") is not None or s2.get("score") is not None)
        inside, started = self._match_boundary(active)
        self._accept_match_observations = inside
        if not inside:
            self.in_match = False
            return new_game
        self.in_match = active
        for st, raw in zip(self.sides, (s1, s2)):
            st.update_frame(self.t, raw, self.in_match)
            if st.board_changed and st.board is not None:
                self.colors.add_board(st.board)
        self._advance_lean()
        if self.sl is not None:
            snap = self.sl.update(d, placed_sides)
            self.sl_snap = None if snap is None else validate_snapshot(snap)
            if production_config.b3_ledger_landing_enabled():
                self._observe_ledger_landings()
        return new_game or started

    def _observe_ledger_landings(self) -> None:
        """得点台帳が観測した控除量を、そのフレームの着弾特徴・更新契機へ渡す。"""
        snap = self.sl_snap
        if not self.in_match or not snap or snap.get("history_incomplete") or snap.get("history_waiting"):
            return
        for event in snap.get("events", ()):
            if event.kind != "observed_landing" or int(event.get("n", 0)) <= 0:
                continue
            st = self.sides[event.side]
            st.landing = Landing(event.available_at_sec, float(event.get("n")),
                                 float(event.get("board_increase")))
            st.land_count += 1

    def _match_boundary(self, active: bool) -> tuple[bool, bool]:
        """lean の開始で一度だけ全状態を初期化し、試合外の観測を遮断する。"""
        if self.match_mapping is None:
            return True, False
        span = self.match_mapping.get(self.game_idx, {})
        start, end = span.get("start_sec"), span.get("end_sec")
        if start is None or end is None or not start - MATCH_TIME_EPS <= self.t <= end + MATCH_TIME_EPS:
            return False, False
        key = (self.game_idx, float(start))
        if key == self._match_key:
            return True, False
        if not active:
            return False, False
        self.reset_game(self.game_idx)
        self.game_start_t = self.t  # lean 範囲内でも得点未観測の開始前画面は取り込まない。
        self._match_key = key
        return True, True

    def _advance_lean(self) -> None:
        """frame_idx<=現在 の最新 lean 行へ進め、NEXT 既知情報を更新する (同一試合のみ)。"""
        for i, st in enumerate(self.sides):
            rows = self.lean_rows[i]
            while st.lean_ptr + 1 < len(rows) and self.lean.frame[rows[st.lean_ptr + 1]] <= self.frame:
                st.lean_ptr += 1
                r = rows[st.lean_ptr]
                slack = LEAN_START_SLACK_SEC if self.match_mapping is None else MATCH_TIME_EPS
                if self.lean.t_sec[r] >= self.game_start_t - slack:   # 試合番号でなく時刻で対応 (番号は1ずれる)
                    st.lean_row = int(r)
                    q = tuple(int(v) if 1 <= int(v) <= 5 else 0 for v in self.lean.queue[r])
                    st.queue, st.queue_t = q, min(float(self.lean.t_sec[r]), self.t)   # lean の t_sec は ms 丸め。frame<=現在なので現在時刻で頭打ち
                    self.colors.add_queue(q)

    def apply_event(self, e: dict) -> int:
        """イベントを適用し、トリガビット (b3_rows.TRIGGER_BITS 値) を返す。"""
        if self.match_mapping is not None and (not self._accept_match_observations or
                                               float(e["observed_at_sec"]) < self.game_start_t - MATCH_TIME_EPS):
            return 0
        if e["game_idx"] != self.game_idx or e.get("side") not in SIDE_NAMES:
            return 0
        st = self.sides[SIDE_NAMES.index(e["side"])]
        k = e["kind"]
        if k == "placement_signal":
            if st.first_place_t is None:
                st.first_place_t = float(e["observed_at_sec"])
            return 1
        if k == "formula_step":
            st.on_formula(e)
            return 2
        if k == "fire_start":
            _n, st.prediction_colors = self.colors.known()
            st.on_fire_start(e)
            return 4
        if k == "fire_complete":
            st.on_fire_complete(e)
            return 4
        if k == "fire_censored":
            st.on_fire_censored(e)
            return 4
        return 0

    def first_move_t(self) -> Optional[float]:
        ts = [s.first_place_t for s in self.sides if s.first_place_t is not None]
        return min(ts) if ts else None
