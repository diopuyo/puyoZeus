"""評価器の切替で表示値が飛ばないようにする、撃ち合い評価の表示平滑 (既定OFF・表示のみ)。

既存の `_ExchangeDisplayEMA` (scripts/visualize_advantage_overlay.py) には 3 つの欠点があった。
  (a) 評価器が値を持たない間 (waiting_confirmed) は、旧評価器の生値を EMA を通さず直接表示する。
  (b) 新評価器へ戻ると、waiting 前の古い EMA 状態から再開し、1 フレームだけ大きく飛ぶ。
  (c) 物理イベントのない由来切替 (S3_landing→G_fe 等) に緩和がなく、値が段差になる。
本クラスは (a)(b)(c) を直す。評価器の値・確率・死亡判定・保持は一切変えず、表示値だけを作る。
確定死亡 (confirmed_death) は従来どおり遅延させず即時表示し、EMA 状態もその値へ合わせる。
状態を持つのでラッパー側 (呼出元) に置く。観測指標ではない。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 既存 EMA と同じ時定数 (scripts/visualize_advantage_overlay.py の EMA_ALPHA と同値。循環 import 回避で再掲し、
# 一致は tests/test_exchange_display_smoothing.py で固定する)。
EMA_ALPHA = 0.25
# 物理イベントのない由来切替に掛ける線形ブレンドの長さ (秒)。0.5 秒 = 約 15 フレーム (30fps)。
# 学習した値ではなく「1 手の設置間隔 (約 1.3 秒) より十分短く、EMA の時定数 (約 0.13 秒) より長い」物理量で決めた。
BLEND_SEC = 0.5
# 切替の前後この秒数以内に物理イベント (発火・連鎖終了・着地) の記録があれば正当な変化とみなしブレンドしない。
EVENT_GRACE_SEC = 0.3
# 遅延させず即時表示する由来 (EMA にも状態を合わせる)。
IMMEDIATE_SOURCES = ("confirmed_death",)
# 切替の前後どちらかがこれらなら、ブレンドしない (死亡は物理イベント)。
DEATH_SOURCES = ("confirmed_death", "unavoidable_death")

# export_state の先頭に置く識別子 (旧形式の (adv, 確率, last_sec) と取り違えない)。
STATE_MARKER = "switch_aware_v1"

Pair = tuple[float, float]


def event_stamp(tracker: Any) -> tuple:
    """発火・連鎖終了信号・着地の記録数。増えたフレームが物理イベントの観測フレーム。"""
    current = getattr(tracker, "current", None)
    records = getattr(tracker, "records", ())
    if current is None:
        return (len(records), None, 0, 0, 0)
    ended = sum(1 for chain in current.chains if chain.end_signal_sec is not None)
    return (len(records), current.exchange_id, len(current.chains), ended, len(current.landings))


@dataclass
class SwitchAwareDisplayEMA:
    """EMA 状態を常に表示値へ連続させ、物理イベントのない切替だけ短くブレンドする。"""

    alpha: float = EMA_ALPHA
    blend_sec: float = BLEND_SEC
    grace_sec: float = EVENT_GRACE_SEC
    # True: 評価器が値を持たない間 (waiting_confirmed) は旧評価器の生値をそのまま表示し、EMA状態だけその値へ合わせる
    # (事前登録の構成B。False=構成A は旧評価器の値も EMA を通す)。
    raw_waiting: bool = False
    state: Pair | None = None
    shown: Pair | None = None
    last_sec: float | None = None
    previous_source: str | None = None
    previous_game: int | None = None
    previous_stamp: tuple | None = None
    last_event_sec: float = float("-inf")
    blend_start: float | None = None
    anchor: Pair = (0.0, 0.5)
    counters: dict[str, int] = field(default_factory=lambda: dict(
        frames=0, switches=0, blends=0, event_switches=0, snaps=0, held=0))

    def apply(self, tracker: Any, target: Pair | None, t_sec: float) -> Pair | None:
        """1 フレームの表示値 (adv, 確率) を返す。target が None (旧評価器の値も未知) なら更新しない。"""
        if t_sec == self.last_sec and self.shown is not None:
            return self.shown  # 同一フレームの再参照は二重更新しない。
        if target is None:
            self.counters["held"] += 1
            return self.shown
        self.counters["frames"] += 1
        source = getattr(tracker, "source", None)
        game = getattr(tracker, "_game_idx", None)
        stamp = event_stamp(tracker)
        waiting_raw = self.raw_waiting and getattr(tracker, "probability", None) is None
        if self.state is None or game != self.previous_game or source in IMMEDIATE_SOURCES or waiting_raw:
            self._snap(target)
        else:
            self._step(target)
            self._blend(source, stamp, t_sec)
        self.previous_source, self.previous_game = source, game
        self.previous_stamp, self.last_sec = stamp, t_sec
        return self.shown

    def export_state(self) -> tuple:
        """プロセス間・journal再実行で受け渡す全状態 (純粋な値のタプル)。長さで旧3要素形式と区別できる。"""
        return (STATE_MARKER, self.state, self.shown, self.last_sec, self.previous_source, self.previous_game,
                self.previous_stamp, self.last_event_sec, self.blend_start, self.anchor)

    def restore_state(self, values: tuple) -> None:
        """export_state の値を戻す。旧初期値 (adv, 確率, None) は「状態なし」として何もしない。"""
        if not values or values[0] != STATE_MARKER:
            return
        (_, self.state, self.shown, self.last_sec, self.previous_source, self.previous_game,
         self.previous_stamp, self.last_event_sec, self.blend_start, self.anchor) = values

    def _snap(self, target: Pair) -> None:
        """試合境界・確定死亡・初回は、遅延させず値へ合わせる (EMA 状態も同値、ブレンド中断)。"""
        self.counters["snaps"] += 1
        self.state, self.shown, self.blend_start = target, target, None

    def _step(self, target: Pair) -> None:
        """EMA を 1 手進める。評価器をまたいでも状態は常に直前の表示値から連続する。"""
        old = self.state
        self.state = tuple(self.alpha * new + (1 - self.alpha) * prev for new, prev in zip(target, old))

    def _blend(self, source: str | None, stamp: tuple, t_sec: float) -> None:
        """物理イベントのない由来切替でブレンドを開始し、進行中なら表示を内挿する。"""
        if stamp != self.previous_stamp:
            self.last_event_sec = t_sec
            self.blend_start = None  # 物理イベントは正当な変化として平滑しない。
        switched = self.previous_source is not None and source != self.previous_source
        if switched:
            self.counters["switches"] += 1
            if self._is_event_switch(source, t_sec):
                self.counters["event_switches"] += 1
                self.blend_start = None
            else:
                self.counters["blends"] += 1
                self.anchor, self.blend_start = self.shown, t_sec
        self.shown = self._interpolate(t_sec)

    def _is_event_switch(self, source: str | None, t_sec: float) -> bool:
        """死亡を含む切替か、直近に物理イベントの記録がある切替か。"""
        death = source in DEATH_SOURCES or self.previous_source in DEATH_SOURCES
        return death or t_sec - self.last_event_sec <= self.grace_sec

    def _interpolate(self, t_sec: float) -> Pair:
        """ブレンド中は 直前表示→EMA 状態 を時間で線形内挿し、終われば EMA 状態そのもの。"""
        if self.blend_start is None:
            return self.state
        weight = (t_sec - self.blend_start) / self.blend_sec
        if weight >= 1.0:
            self.blend_start = None
            return self.state
        return tuple(a + (s - a) * weight for a, s in zip(self.anchor, self.state))
