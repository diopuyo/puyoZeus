"""既存採点の定義を保った段2集計。区間は対応付き試合bootstrap。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.b3_eval_metrics import row_brier, row_log_loss
from src.b3_train_config import BOOTSTRAP_N, BOOTSTRAP_SEED, CI_LOW_PCT, CI_HIGH_PCT

DELAY_PERCENTILE = 95
NEW_DELAY_SEC = 5.0
SAFETY_RECORDS = 18
SHORT_RECORDS = 5
ZENCHI_GAMES = 115


def read_json(path: Path) -> Any:
    """BOM付き原票を許容する。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def paired_interval(candidate: np.ndarray, reference: np.ndarray, ratio: bool = False) -> dict:
    """試合行を対応再抽出。M_agreeだけは既存どおり分子合計/分母合計。"""
    c, r = np.asarray(candidate, float), np.asarray(reference, float)
    if c.shape != r.shape or len(c) == 0 or not np.isfinite(c).all() or not np.isfinite(r).all():
        raise ValueError("対応する有限な試合配列が必要")
    idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(c), (BOOTSTRAP_N, len(c)))
    if ratio:
        aggregate = lambda a: a[..., 0].sum(axis=-1) / a[..., 1].sum(axis=-1)
    else:
        aggregate = lambda a: a.mean(axis=-1)
    delta = aggregate(c[idx]) - aggregate(r[idx])
    return dict(point=float(aggregate(c) - aggregate(r)), lo=float(np.percentile(delta, CI_LOW_PCT)),
                hi=float(np.percentile(delta, CI_HIGH_PCT)), n_games=len(c),
                bootstrap_n=BOOTSTRAP_N, bootstrap_seed=BOOTSTRAP_SEED)


def metric_intervals(candidate: list[dict], reference: list[dict]) -> dict:
    """採点器の試合順の不一致を拒否し、M_time/M_agreeをそのまま比較する。"""
    if [r["game"] for r in candidate] != [r["game"] for r in reference]:
        raise ValueError("採点器の試合順が不一致")
    return {metric: paired_interval(
        np.array([r["m_time"] if metric == "m_time" else [r["late_hits"], r["late_frames"]]
                  for r in candidate]),
        np.array([r["m_time"] if metric == "m_time" else [r["late_hits"], r["late_frames"]]
                  for r in reference]), ratio=metric == "m_agree") for metric in ("m_time", "m_agree")}


def raw_game_scores(display: dict, reference: dict, games: list[dict]) -> list[dict]:
    """全提供時間の30Hz格子で平滑前の確率を採点。nullも0.5で残す。"""
    np.testing.assert_array_equal(display["t_sec"], reference["t_sec"])
    rows = []
    for game in games:
        mask = (display["t_sec"] >= game["start"]) & (display["t_sec"] < game["end"])
        if not mask.any():
            raise ValueError("採点窓にフレームがない")
        y = np.full(mask.sum(), game["winner"] == "1P", dtype=float)
        row = dict(game=game["game"], frames=int(mask.sum()), null_frames=int(
            np.count_nonzero(mask & (display["source"] == "null"))))
        for name, fn in (("ll", row_log_loss), ("brier", row_brier)):
            row[name + "_c"] = float(fn(y, display["p_selected"][mask]).mean())
            row[name + "_r"] = float(fn(y, reference["display_p1"][mask]).mean())
        rows.append(row)
    return rows


def probability_intervals(rows: list[dict]) -> dict:
    """LL/Brierは各試合の時間平均を試合間で等重み集計する。"""
    values = {name: paired_interval(np.array([r[name + "_c"] for r in rows]),
                                   np.array([r[name + "_r"] for r in rows])) for name in ("ll", "brier")}
    return dict(dll=values["ll"], brier_diff=values["brier"]["point"], brier=values["brier"])


def confirm_changes(before: dict, after: dict) -> dict:
    """同じ試合・側の初出を比較。遅れ0の共通事例もp95の母数に含める。"""
    old = {(r["game"], r["side"]): r for r in before["rows"] if not r["false"]}
    new = {(r["game"], r["side"]): r for r in after["rows"] if not r["false"]}
    common = sorted(old.keys() & new.keys())
    delays = [new[key]["first_sec"] - old[key]["first_sec"] for key in common]
    return dict(lost=len(old.keys() - new.keys()), correct_R=len(old), common=len(common),
                delay_p95_diff=float(np.percentile(delays, DELAY_PERCENTILE)) if delays else None,
                new_delay_over_5s=sum(d > NEW_DELAY_SEC for d in delays), delays=delays,
                lost_cases=[old[k] for k in sorted(old.keys() - new.keys())],
                candidate_only=[new[k] for k in sorted(new.keys() - old.keys())])


def safety_receipt(reference_report: dict, scans: list[dict], changes: list[dict]) -> dict:
    """F10の同一証明に限り、zenchi外の既存安全回帰値を原票から継承する。"""
    old = reference_report["existing5"]
    display = reference_report["display"]["a1"]
    frames5 = sum(v["false_frames"] + v["unknown_frames"] for v in display.values())
    all115 = sum(s["false"] + s["false_frames"] for s in scans)
    delays = [v for c in changes for v in c["delays"]]
    return dict(false_confirms=dict(records18=all115 + frames5 + old["false"] + old["unresolved"],
                    short5=frames5 + old["false"] + old["unresolved"],
                    game14=len(reference_report["regression"]["game14_false_times"]), all115=all115),
                first_confirm_sec_3min=reference_report["regression"]["scene"]["legacy_metric_sec"],
                confirms=dict(lost=sum(c["lost"] for c in changes),
                    delay_p95_diff=float(np.percentile(delays, DELAY_PERCENTILE)) if delays else None,
                    new_delay_over_5s=sum(d > NEW_DELAY_SEC for d in delays), common=len(delays)),
                safety_population=dict(records=SAFETY_RECORDS, short_records=SHORT_RECORDS, games=ZENCHI_GAMES,
                    short_cases=old["after"], certain_frames_short=sum(v["certain_frames"] for v in display.values()),
                    overlap="18記録・短5・第14・115は重複集合。件数は独立標本として加算しない"))
