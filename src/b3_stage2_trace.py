"""段2の予測・30Hz表示・レビュー変換。死亡証明と確率を分離する。"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from src.b3_eval_adapter import CERTAIN_SOURCES, EvaluatorTrace, resample_hold
from src.b3_train_model import TwoStateModel
from src.exchange_display_smoothing import SwitchAwareDisplayEMA

ADV_SCALE = 100.0
PROB_EPS = 1e-9


def predict_rows(rows: pd.DataFrame, model: TwoStateModel) -> pd.DataFrame:
    """原向きだけを推論し、非採用側は欠測として保存する。"""
    df = rows.loc[rows.orientation == 0].sort_values(
        ["zenchi_set", "zenchi_game", "t_source_sec"]).copy()
    if df.empty or df.row_key.duplicated().any():
        raise ValueError("原向きの行が空またはキーが重複")
    if not df.mode_exchange.isin([0, 1]).all():
        raise ValueError("状態選択が不正")
    p = model.predict(df)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("予測が有限な確率ではない")
    df["p_selected"] = p
    df["p_static"] = np.where(df.mode_exchange == 0, p, np.nan)
    df["p_exchange"] = np.where(df.mode_exchange == 1, p, np.nan)
    df["mode"] = np.where(df.mode_exchange == 1, "exchange", "static")
    return df


def physical_times(events: list[dict]) -> np.ndarray:
    """Rで観測済みの発火・終了・着地の時刻だけを表示切替の合図に使う。"""
    times = []
    for event in events:
        times.extend(event.get(k) for k in ("trigger_sec", "closed_sec"))
        for chain in event.get("chains", []):
            times.extend(chain.get(k) for k in ("observed_sec", "end_signal_sec"))
        for landing in event.get("landings", []):
            times.extend(landing.get(k) for k in ("fall_start_sec", "t_sec"))
    return np.unique([t for t in times if t is not None])


def to_adv(p: float, k: float | None) -> float:
    """本番 _winprob_to_adv と同じ変換。"""
    clipped = np.clip(p, PROB_EPS, 1 - PROB_EPS)
    raw = np.log(clipped / (1 - clipped)) / k if k else (p - .5) * 2 * ADV_SCALE
    return float(np.clip(raw, -ADV_SCALE, ADV_SCALE))


def smooth_display(raw: np.ndarray, sources: np.ndarray, games: np.ndarray,
                   times: np.ndarray, physical: np.ndarray, k: float | None) -> tuple[np.ndarray, np.ndarray]:
    """本番のEMA実装をフレームごとに呼ぶ。nullは0.5へ即時復帰する。"""
    smoother = SwitchAwareDisplayEMA()
    tracker = SimpleNamespace(current=None, records=(), source=None, _game_idx=None)
    adv, shown = np.empty(len(raw)), np.empty(len(raw))
    # event_stampの件数を、因果的に到着した物理合図数で表す。
    counts = np.searchsorted(physical, times, side="right")
    for i, (p, source, game, stamp) in enumerate(zip(raw, sources, games, times)):
        tracker.source, tracker._game_idx = str(source), int(game)
        tracker.records = range(int(counts[i]))
        if source == "null":
            smoother = SwitchAwareDisplayEMA()
            adv[i], shown[i] = 0.0, .5
            continue
        adv[i], shown[i] = smoother.apply(tracker, (to_adv(float(p), k), float(p)), float(stamp))
    return adv, shown


def make_display(df: pd.DataFrame, reference: dict[str, np.ndarray], games: list[dict],
                 events: list[dict], k: float | None) -> dict[str, np.ndarray]:
    """Rの30Hz時刻を保ち、試合をまたいだ保持を禁止して死亡証明だけをそのまま載せる。"""
    t = reference["t_sec"]
    raw = np.full(len(t), .5)
    source = np.full(len(t), "null", dtype="U24")
    game_ids = np.full(len(t), -1, dtype=int)
    for game in games:
        mask = (t >= game["start"]) & (t < game["end"])
        sub = df[df.zenchi_game == game["game"]].sort_values("t_source_sec")
        if sub.empty:
            raise ValueError(f"試合{game['game']}の行がない")
        if sub.t_source_sec.duplicated().any():
            raise ValueError("同一試合内の時刻重複")
        trace = EvaluatorTrace(sub.t_source_sec.to_numpy(float), sub.p_selected.to_numpy(float))
        part = resample_hold(trace, game["start"], game["end"], grid_times=t[mask])
        idx = np.maximum(0, np.searchsorted(trace.t_sec, t[mask], side="right") - 1)
        raw[mask], source[mask] = part["display_p1"], np.where(
            part["source"] == "null", "null", sub["mode"].to_numpy()[idx])
        game_ids[mask] = game["game"]
    # 死亡証明はB3行の鮮度や確率から作らず、Rのフレームと完全同一にする。
    death = np.isin(reference["source"], CERTAIN_SOURCES)
    raw_with_death = np.where(death, reference["display_p1"], raw)
    source[death] = reference["source"][death]
    adv, shown = smooth_display(raw_with_death, source, game_ids, t, physical_times(events), k)
    return dict(t_sec=t, display_p1=raw_with_death, display_adv=adv, source=source,
                p_selected=raw, p_display=shown, game_idx=game_ids)


def review_rows(rows: pd.DataFrame, predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """c68向け公開口。キーで照合し、元動画時刻へ変換して既存変換器へ渡す。"""
    from src.b3_review_input import from_b3_rows_iter
    if rows.row_key.duplicated().any() or predictions.row_key.duplicated().any():
        raise ValueError("レビュー行キーが重複")
    indexed = rows.set_index("row_key", drop=False)
    if set(indexed.index) != set(predictions.row_key):
        raise ValueError("レビュー行と予測のキーが不一致")
    pred = predictions.sort_values("t_source_sec")
    source = indexed.loc[pred.row_key].copy()
    source["t_sec"] = source["t_source_sec"]
    source["game_idx"] = source["zenchi_game"]
    clean = lambda frame: frame.astype(object).where(pd.notna(frame), None).to_dict("records")
    return from_b3_rows_iter(clean(source), clean(pred))
