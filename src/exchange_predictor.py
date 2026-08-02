"""#24 打ち合い計測器 RT推論用「案D単体モデル」ロード+推論モジュール (2026-08-02)。

ΔWinProb接続アーキ設計 (案C=仮想盤面2回評価) の Step1。RT (リアルタイム)
モードでは sim_* 3列のMC計算 (平均1秒/件) が予算オーバーのため、
「案D単体 (41特徴量、AUC 0.786/rho 0.694)」をRT層のモデルとして使う
(併用スタッキングは動画モード層)。

scripts/train_exchange_model_d.py の `--save-model` で保存したモデル
バンドル (joblib) を読み込み、単発イベントの特徴量dictから軽量に推論する。

## 設計方針: scripts/ への依存を持たない
本モジュールは src/ 配下 (本番RTパイプラインが参照する側) のため、
scripts/ 配下 (学習・評価用の重い依存を持つ) には依存しない。推論に必要な
メタ情報 (indicator_bases・phase一覧・fire_side一覧) は全て保存済み
バンドルに埋め込まれている (self-contained)。

## 使い方
    from src.exchange_predictor import load_exchange_model, predict_exchange_event

    model = load_exchange_model("data/models/exchange_model_d_2026-08-02.joblib")
    prob_taiou, net_ojama_pred = predict_exchange_event(model, {
        "fire_current_max_chain": 3.0, "opp_current_max_chain": 1.0,
        "diff_current_max_chain": 2.0, ...,  # indicator_bases分のfire_/opp_/diff_
        "phase": "中", "fire_side": "1P",
    })

## 速度要件
50ms未満/件 (RT予算)。1000回実測の中央値で検収する
(tests/test_exchange_predictor.py 参照)。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass(frozen=True)
class ExchangeModelBundle:
    """RT推論用モデルバンドル (joblib での保存/復元単位)。

    Attributes:
        cls_model: taiou_success (対応成功確率) 予測モデル。
        reg_model: net_ojama_after (正味お邪魔予測) 予測モデル。
        indicator_bases: fire_/opp_/diff_ 3つ組の base 名リスト (学習時の列順)。
        feature_names: 特徴量ベクトルの列名リスト (学習時の列順、参照用)。
        phases: phase one-hot の展開順 (序/中/終)。
        fire_sides: fire_side one-hot の展開順 (1P/2P)。
        metadata: 学習時メタ情報 (ラベルCSV名・日時・サンプル数・ハイパラ等)。
    """
    cls_model: Any
    reg_model: Any
    indicator_bases: list[str]
    feature_names: list[str]
    phases: tuple[str, ...]
    fire_sides: tuple[str, ...]
    metadata: dict[str, Any]


def load_exchange_model(path: "str | Path") -> ExchangeModelBundle:
    """joblib 保存済みのモデルバンドルを読み込む。"""
    raw: dict[str, Any] = joblib.load(Path(path))
    return ExchangeModelBundle(
        cls_model=raw["cls_model"],
        reg_model=raw["reg_model"],
        indicator_bases=list(raw["indicator_bases"]),
        feature_names=list(raw["feature_names"]),
        phases=tuple(raw["phases"]),
        fire_sides=tuple(raw["fire_sides"]),
        metadata=dict(raw["metadata"]),
    )


def _build_feature_vector(model: ExchangeModelBundle, features: dict[str, Any]) -> np.ndarray:
    """features dict から学習時と同じ列順の特徴量ベクトル (shape=(1, n_features)) を組む。

    scripts/train_exchange_model_d.build_feature_matrix と同一の列順
    (fire_/opp_/diff_ 3つ組 → phase one-hot → fire_side one-hot) を、
    scripts/ に依存せずバンドルに埋め込まれたメタ情報だけで再現する。
    """
    values: list[float] = []
    for prefix in ("fire_", "opp_", "diff_"):
        for base in model.indicator_bases:
            values.append(float(features[f"{prefix}{base}"]))
    for phase in model.phases:
        values.append(1.0 if features["phase"] == phase else 0.0)
    for side in model.fire_sides:
        values.append(1.0 if features["fire_side"] == side else 0.0)
    return np.asarray(values, dtype=np.float64).reshape(1, -1)


def predict_exchange_event(
    model: ExchangeModelBundle, features: dict[str, Any],
) -> tuple[float, float]:
    """1発火イベント分の特徴量dictから (対応成功確率, net_ojama_after予測) を返す。

    Args:
        model: load_exchange_model() で読み込んだバンドル。
        features: 必須キーは indicator_bases 分の fire_<base>/opp_<base>/
            diff_<base> (数値) + "phase" (序/中/終のいずれか) + "fire_side"
            (1P/2Pのいずれか)。欠けているキーがあれば KeyError を送出する
            (誤って0埋めして誤判定させないため)。

    Returns:
        (prob_taiou_success, net_ojama_after_pred) のタプル。
    """
    x = _build_feature_vector(model, features)
    prob_taiou = float(model.cls_model.predict_proba(x)[0, 1])
    net_ojama_pred = float(model.reg_model.predict(x)[0])
    return prob_taiou, net_ojama_pred
