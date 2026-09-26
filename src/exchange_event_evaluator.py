"""静止・発火・撃ち合い終了を評価する、状態を保持しない評価器。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np

from src.exchange_event_features import (
    Array, D_COLUMNS, G_COLUMNS, S1_COLUMNS, S3_COLUMNS, SIDE_COLUMNS,
    arrival_features, fill_phase, g_features, score_features, validate_elapsed,
)

MODEL_VERSION = "exchange_event_v1"
MODEL_COLUMNS = {"G_fe": G_COLUMNS, "S1": S1_COLUMNS, "S3": S3_COLUMNS}


def _array(value: Array, shape: tuple[int, ...], name: str,
           allow_missing: bool = False) -> Array:
    """配列を複製して凍結し、呼出側による変更の混入を防ぐ。"""
    result = np.array(value, dtype=float, copy=True)
    invalid = np.isinf(result).any() if allow_missing else not np.isfinite(result).all()
    if result.shape != shape or invalid:
        raise ValueError(f"{name}の形状または数値が不正")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class StaticInput:
    """両側STABLE時の入力。Dはsource_side視点、M0は常に1P勝率。

    source_sideは保存行・観測の出所（0=1P、1=2P）。左右交換時はDを
    維持しsource_sideを交換する。相手視点のDへ単純な符号反転は不可。
    elapsed_secの起点は学習と同じ試合内最初の保存時刻。
    """

    d_features: Array
    m0_probability_1p: float
    elapsed_sec: float
    source_side: int = 0

    def __post_init__(self) -> None:
        validate_elapsed(self.elapsed_sec)
        if self.source_side not in (0, 1):
            raise ValueError("source_sideは0または1が必要")
        if not np.isfinite(self.m0_probability_1p) or not 0 <= self.m0_probability_1p <= 1:
            raise ValueError("M0勝率は0〜1が必要")
        values = _array(self.d_features, (len(D_COLUMNS),), "D", allow_missing=True)
        object.__setattr__(self, "d_features", values)


@dataclass(frozen=True)
class FiringInput:
    """発火前に凍結した入力。prefire_sidesとfiringは1P/2P順。"""

    static: StaticInput
    prefire_sides: Array
    firing: tuple[bool, bool]

    def __post_init__(self) -> None:
        if not isinstance(self.static, StaticInput):
            raise ValueError("staticはStaticInputが必要")
        sides = _array(self.prefire_sides, (2, len(SIDE_COLUMNS)), "発火前", True)
        if np.any((sides < 0) | (sides > 1)):
            raise ValueError("発火前の絶対特徴は0〜1が必要")
        if len(self.firing) != 2 or not any(self.firing) or not all(
                isinstance(v, (bool, np.bool_)) for v in self.firing):
            raise ValueError("発火側は少なくとも一方がTrueの2要素が必要")
        object.__setattr__(self, "prefire_sides", sides)
        object.__setattr__(self, "firing", tuple(self.firing))


@dataclass(frozen=True)
class ExchangeEndInput:
    """絶対終了信号後の入力。得点未確定ならS1を返し続ける。

    score_elapsed_secは得点換算に使う発火時点の経過秒。
    欠測確定はNaNまたは負数、未確定はscores_confirmed=Falseで区別する。
    """

    firing: FiringInput
    scores_before: Array
    scores_after: Array
    score_elapsed_sec: float
    scores_confirmed: bool = True
    from_first_move: bool = False

    def __post_init__(self) -> None:
        validate_elapsed(self.score_elapsed_sec)
        if not isinstance(self.firing, FiringInput):
            raise ValueError("firingはFiringInputが必要")
        if not isinstance(self.scores_confirmed, bool) or not isinstance(self.from_first_move, bool):
            raise ValueError("得点確定・起点フラグはboolが必要")
        for name in ("scores_before", "scores_after"):
            object.__setattr__(self, name, _array(getattr(self, name), (2,), name, True))


EventInput: TypeAlias = StaticInput | FiringInput | ExchangeEndInput


class ExchangeModels(Protocol):
    """ファイル形式や分類器から独立したモデル境界。"""

    elapsed_thresholds: tuple[float, float]

    def predict_source_probability(self, model_name: str, features: Array) -> float:
        """観測元の側の勝率を返す。"""
        ...


def build_features(event: EventInput, thresholds: tuple[float, float]) -> tuple[str, Array, int]:
    """状態からモデル名・自側特徴・観測元を得る。検証済み列順を固定する。"""
    if isinstance(event, ExchangeEndInput):
        name, features, side = build_features(event.firing, thresholds)
        if not event.scores_confirmed:
            return name, features, side
        scores = score_features(event.scores_before, event.scores_after,
                                event.score_elapsed_sec, side, event.from_first_move)
        return "S3", np.r_[features, scores], side
    if isinstance(event, FiringInput):
        side = event.static.source_side
        arrival = arrival_features(event.prefire_sides, event.firing, side)
        return "S1", np.r_[np.nan_to_num(event.static.d_features, nan=0.0), arrival], side
    if not isinstance(event, StaticInput):
        raise TypeError("未知の撃ち合い入力型")
    limits = np.asarray(thresholds)
    if limits.shape != (2,) or not np.isfinite(limits).all() or np.any(limits < 0) or limits[0] > limits[1]:
        raise ValueError("経過秒の閾値が不正")
    d = event.d_features
    fill = fill_phase(d[D_COLUMNS.index("board_puyo_total")], d[D_COLUMNS.index("diff_board_puyo_total")])
    elapsed = np.searchsorted(limits, event.elapsed_sec, side="left")
    x = g_features(np.nan_to_num(d, nan=0.0)[None, :], np.array([event.m0_probability_1p]),
                   np.array([1 - 2 * event.source_side]), np.array([[fill, elapsed]]))
    return "G_fe", x[0], event.source_side


def evaluate_exchange_event(event: EventInput, models: ExchangeModels) -> float:
    """入力状態だけから1P勝率を返す。凍結・遷移管理は呼出側が担当する。"""
    name, features, side = build_features(event, models.elapsed_thresholds)
    probability = float(models.predict_source_probability(name, features))
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("モデル勝率が0〜1の範囲外")
    return probability if side == 0 else 1 - probability


def file_sha256(path: Path) -> str:
    """大きいモデル・学習資産も一定メモリでハッシュ化する。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class FileExchangeModels:
    """信頼できるローカル学習成果物を読み込むsklearnアダプタ。"""

    elapsed_thresholds: tuple[float, float]
    estimators: dict

    @classmethod
    def load(cls, directory: str | Path, lightweight: bool = True) -> FileExchangeModels:
        """版・列順・モデルハッシュを照合してから復元する。"""
        import joblib

        root = Path(directory)
        meta = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if meta["version"] != MODEL_VERSION:
            raise ValueError("非対応のモデル版")
        thresholds = tuple(meta["elapsed_thresholds"])
        build_features(StaticInput(np.zeros(len(D_COLUMNS)), .5, 0), thresholds)
        estimators = {}
        for name, columns in MODEL_COLUMNS.items():
            key = name + "_light" if lightweight and name != "G_fe" else name
            entry = meta["models"][key]
            if tuple(entry["columns"]) != columns:
                raise ValueError(f"{key}の特徴列順が不一致")
            path = root / entry["file"]
            if path.parent.resolve() != root.resolve() or file_sha256(path) != entry["sha256"]:
                raise ValueError(f"{key}のモデルファイルが不一致")
            model = joblib.load(path)
            if model.n_features_in_ != len(columns) or list(model.classes_) != [0, 1]:
                raise ValueError(f"{key}のモデル次元またはラベルが不正")
            estimators[name] = model
        return cls(thresholds, estimators)

    def predict_source_probability(self, model_name: str, features: Array) -> float:
        """分類器内部にイベント状態を蓄積せず推論する。"""
        return float(self.estimators[model_name].predict_proba(features[None, :])[0, 1])
