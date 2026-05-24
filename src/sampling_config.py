"""
動画サンプリング・評価頻度のプロジェクト共通設定。

ルール（プロジェクト合意事項、2026-04-25 確定）:
    1. 盤面情報の保持・追跡: 0.2 秒に 1 フレーム（5 fps 相当）
       連鎖検出、puyo 配置追跡、ネクスト遷移検出など、
       盤面の変化をカバーするのに必要な時間解像度。

    2. 有利不利の評価更新: 0.6 秒に 1 回（≈ 1.67 Hz）
       指標計算 + スコア表示の更新タイミング。
       盤面サンプリング 3 回ごとに 1 回評価する関係。

これより細かいサンプリングは puyo の落下/連鎖アニメをまたぐ可能性があり
無駄な変動が増える。これより粗いと連鎖の途中状態を見逃す可能性。

使い方:
    from src.sampling_config import BOARD_INTERVAL_SEC, EVAL_INTERVAL_SEC

    next_t += BOARD_INTERVAL_SEC          # 盤面サンプリング
    if frame_idx % EVAL_FRAME_RATIO == 0: # 評価タイミング
        recompute_indicators(...)
"""
from __future__ import annotations

# ============================
# サンプリング間隔（秒）
# ============================

# 盤面情報の保持・追跡用サンプリング間隔
BOARD_INTERVAL_SEC: float = 0.2

# 有利不利の評価更新間隔
EVAL_INTERVAL_SEC: float = 0.6

# 評価頻度は盤面サンプリングの 3 倍に 1 回
# (0.6s / 0.2s = 3)
EVAL_FRAME_RATIO: int = 3

# 安定判定用: 連続 2 フレーム（0.2s 間隔）で同一判定なら採用
STABLE_FRAME_COUNT: int = 2
STABLE_FRAME_INTERVAL_SEC: float = 0.2


def board_sample_times(start_sec: float, end_sec: float) -> list[float]:
    """指定区間で盤面サンプリングすべき時刻リストを返す。"""
    times: list[float] = []
    t = start_sec
    while t < end_sec:
        times.append(t)
        t += BOARD_INTERVAL_SEC
    return times


def eval_sample_times(start_sec: float, end_sec: float) -> list[float]:
    """指定区間で有利不利評価を実施すべき時刻リストを返す。"""
    times: list[float] = []
    t = start_sec
    while t < end_sec:
        times.append(t)
        t += EVAL_INTERVAL_SEC
    return times
