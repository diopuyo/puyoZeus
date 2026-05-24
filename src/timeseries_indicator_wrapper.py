"""Phase H2: 時系列展開ラッパー (TimeseriesWrapper).

各 STABLE frame で計算した IndicatorSet を時系列で保持し、
45 指標を 6 軸で展開して 270 features 辞書に変換する。

6 軸:
    1. static    : 現在値
    2. delta     : 前 STABLE 比 (curr - prev) / max(abs(prev), epsilon)
    3. accel     : 加速度 (delta_curr - delta_prev)
    4. hist_max  : 直近 history_sec の最大値
    5. hist_min  : 直近 history_sec の最小値
    6. hist_mean : 直近 history_sec の平均値

履歴不足時 (試合開始直後) は中立値 (delta/accel=0.0、hist_*=static) で穴埋め。

設計思想:
    - stateless 計算で得た IndicatorSet をこの wrapper が時系列に蓄積
    - 学習データ生成時に 1 STABLE frame ごとに update + expand_features
    - 履歴は (timestamp, IndicatorSet) tuple で list 保持、古いものは drop

Phase H4 の CNN auxiliary supervision target にもこの 270 軸を使う想定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from src.indicators import (
    ALL_INDICATOR_NAMES,
    EXTRA_INDICATOR_NAMES,
    IndicatorSet,
)

# ============================
# 定数
# ============================

# 履歴管理の時間窓 (秒)。Phase H2 仕様: 直近 30 秒。
DEFAULT_HISTORY_SEC: float = 30.0

# サンプリング間隔 (秒)。EVAL_INTERVAL_SEC=0.6 と整合。
DEFAULT_SAMPLE_INTERVAL_SEC: float = 0.6

# Δ 計算時のゼロ割回避 epsilon。
# raw 値ではなく 0..1 正規化済値を扱うため 1e-3 で十分小さい。
DELTA_EPSILON: float = 1e-3

# 履歴不足時の delta/accel 中立値。指標は 0..1 なので 0.0 = 変化無しに対応。
NEUTRAL_DELTA: float = 0.0

# 軸名定数 (CSV 列名 suffix にも使用)。
AXIS_STATIC: str = "static"
AXIS_DELTA: str = "delta"
AXIS_ACCEL: str = "accel"
AXIS_HIST_MAX: str = "hist_max"
AXIS_HIST_MIN: str = "hist_min"
AXIS_HIST_MEAN: str = "hist_mean"

ALL_AXES: tuple[str, ...] = (
    AXIS_STATIC,
    AXIS_DELTA,
    AXIS_ACCEL,
    AXIS_HIST_MAX,
    AXIS_HIST_MIN,
    AXIS_HIST_MEAN,
)

# 全 45 指標名 (ALL=8 + EXTRA=37)。FEATURE_NAMES と同順。
TIMESERIES_INDICATOR_NAMES: tuple[str, ...] = (
    tuple(ALL_INDICATOR_NAMES) + tuple(EXTRA_INDICATOR_NAMES)
)


# ============================
# IndicatorSet → スカラー値抽出
# ============================


def indicator_value(indicator_set: IndicatorSet, name: str) -> float:
    """IndicatorSet から指標値 (0..1) を取得.

    `results` 優先、属性 fallback (next_acceptance のみ属性のため)。
    extract_feature_diff の `_indicator_value` と同じ規約。
    """
    if name in indicator_set.results:
        return float(indicator_set.results[name].score)
    # next_acceptance は IndicatorSet 直接属性
    if name == "next_acceptance":
        return float(indicator_set.next_acceptance)
    return 0.0


def extract_static_vector(
    indicator_set: IndicatorSet,
    names: Iterable[str] = TIMESERIES_INDICATOR_NAMES,
) -> dict[str, float]:
    """1 frame 分の 45 指標スカラー値 dict を返す."""
    return {name: indicator_value(indicator_set, name) for name in names}


# ============================
# Entry: (timestamp, value_vector)
# ============================


@dataclass(frozen=True)
class TimeseriesEntry:
    """履歴 1 件: timestamp + 各指標スカラー値辞書.

    IndicatorSet 全体を保持するとメモリと再計算コストが膨らむため、
    抽出済みスカラー値辞書のみを保持する。
    """
    timestamp: float
    values: dict[str, float]


# ============================
# TimeseriesWrapper 本体
# ============================


@dataclass
class TimeseriesWrapper:
    """1 サイド (1P or 2P) の時系列指標履歴 wrapper.

    Attributes:
        history_sec: 直近何秒間の履歴を保持するか。デフォルト 30.0。
        sample_interval_sec: 想定サンプリング間隔 (情報用、現状未使用)。
        history: list[TimeseriesEntry]。timestamp 昇順で蓄積。
    """
    history_sec: float = DEFAULT_HISTORY_SEC
    sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC
    history: list[TimeseriesEntry] = field(default_factory=list)

    def update(self, timestamp: float, indicator_set: IndicatorSet) -> None:
        """新しい IndicatorSet を履歴末尾に追加し、古い entries を drop する."""
        values = extract_static_vector(indicator_set)
        entry = TimeseriesEntry(timestamp=timestamp, values=values)
        self.history.append(entry)
        self._drop_old_entries(timestamp)

    def _drop_old_entries(self, current_t: float) -> None:
        """current_t - history_sec より古い entries を削除する."""
        cutoff = current_t - self.history_sec
        # timestamp 昇順なので先頭から削除して効率化
        while self.history and self.history[0].timestamp < cutoff:
            self.history.pop(0)

    def reset(self) -> None:
        """履歴を空にする (試合切り替え時等に呼ぶ)."""
        self.history.clear()

    # ------------------------------------------------
    # feature 展開
    # ------------------------------------------------

    def expand_features(
        self, indicator_set: IndicatorSet,
    ) -> dict[str, float]:
        """45 indicator × 6 軸 = 270 features を返す.

        Args:
            indicator_set: 「現在の」指標値 (history 末尾の値と一致する想定)。
                update() を呼んでから本関数を呼ぶ運用を想定するが、引数として
                明示的に受け取ることで「expand 専用」のテストや、
                update せずに評価したいケースにも対応する。

        Returns:
            dict[str, float]: key=f"{indicator_name}__{axis}", value=値。
                例: "main_chain_maturity__static", "main_chain_maturity__delta", ...
        """
        out: dict[str, float] = {}
        static_vec = extract_static_vector(indicator_set)
        for name in TIMESERIES_INDICATOR_NAMES:
            curr = static_vec[name]
            out[f"{name}__{AXIS_STATIC}"] = curr
            out[f"{name}__{AXIS_DELTA}"] = self._compute_delta(name, curr)
            out[f"{name}__{AXIS_ACCEL}"] = self._compute_accel(name, curr)
            hist_stats = self._compute_history_stats(name, curr)
            out[f"{name}__{AXIS_HIST_MAX}"] = hist_stats[0]
            out[f"{name}__{AXIS_HIST_MIN}"] = hist_stats[1]
            out[f"{name}__{AXIS_HIST_MEAN}"] = hist_stats[2]
        return out

    def _compute_delta(self, name: str, curr: float) -> float:
        """前 STABLE 比の Δ を返す (履歴 < 2 entries なら NEUTRAL_DELTA)."""
        if len(self.history) < 2:
            return NEUTRAL_DELTA
        prev = self.history[-2].values.get(name, 0.0)
        denom = max(abs(prev), DELTA_EPSILON)
        return (curr - prev) / denom

    def _compute_accel(self, name: str, curr: float) -> float:
        """加速度 (delta_curr - delta_prev) を返す.

        履歴 < 3 entries なら NEUTRAL_DELTA。
        delta_prev は history[-3] と history[-2] から計算。
        delta_curr は history[-2] と curr から計算 (= _compute_delta と同じ)。
        """
        if len(self.history) < 3:
            return NEUTRAL_DELTA
        prev = self.history[-2].values.get(name, 0.0)
        prev2 = self.history[-3].values.get(name, 0.0)
        delta_curr = (curr - prev) / max(abs(prev), DELTA_EPSILON)
        delta_prev = (prev - prev2) / max(abs(prev2), DELTA_EPSILON)
        return delta_curr - delta_prev

    def _compute_history_stats(
        self, name: str, curr: float,
    ) -> tuple[float, float, float]:
        """直近履歴の (max, min, mean) を返す.

        履歴 == 1 entry のみ (= 現在値 push 後の最初の expand) なら
        max=min=mean=curr とし、変化無しを表現する。
        """
        if not self.history:
            return curr, curr, curr
        # 履歴は curr が push 済みの想定だが、引数 curr を信用する
        values = [e.values.get(name, 0.0) for e in self.history]
        if not values:
            return curr, curr, curr
        return max(values), min(values), sum(values) / len(values)


# ============================
# 全 270 列名タプル (CSV ヘッダ生成用)
# ============================


def build_timeseries_feature_names() -> tuple[str, ...]:
    """45 × 6 = 270 列名を順序通りに返す."""
    out: list[str] = []
    for name in TIMESERIES_INDICATOR_NAMES:
        for axis in ALL_AXES:
            out.append(f"{name}__{axis}")
    return tuple(out)


TIMESERIES_FEATURE_NAMES: tuple[str, ...] = build_timeseries_feature_names()
