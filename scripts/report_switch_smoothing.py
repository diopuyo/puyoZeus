"""切替平滑・E19 の採点。事前登録 (exev DECISIONS.md 2026-10-01) の門だけで採否を決める。

標準の E36b 採点器 (report_e36b) を出力先だけ差し替えて動かし、そこへ
  - 飛びの件数 (scripts/measure_switch_jumps.py)
  - 表示ベース q / 場面 (display_adv 由来) / 確定死亡の即時性 / 鮮度
  - OFF との全列一致 (平滑前の列が動いていないこと)
を足す。母数 (フレーム数・件数) を必ず併記する。
使い方: python -m scripts.report_switch_smoothing --variant off|a|e19|a_e19 [--baseline off]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import measure_switch_jumps as jumps
from scripts import report_e36, report_e36b
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.run_switch_smoothing import EXEV_ROOT, OUT_ROOT, SOURCES
from src.display_freshness import display_freshness

# --- 事前登録の門 (採点前に固定) ---
Q_BASELINE = 0.5075669993215693          # logs/pending_expiry/e36b_on 実測
Q_TOLERANCE = 5e-7
DISPLAY_Q_TOLERANCE = 0.001              # 表示ベース q (補助門)
ZENCHI_HITS_MIN, ZENCHI_FRAMES = 7671, 8333
SCENE_DEADLINE = 2766.0
BASELINE_NON_EVENT_SWITCH_JUMPS = 34     # 採点前の実測 (OFF 5記録)
JUMP_REDUCTION_MAX_RATIO = 0.5
LEGIT_JUMP_MIN_RATIO = 0.9
FRESH_EQUAL_TOLERANCE = 0.05
UPDATES_MIN_RATIO = 0.95
E19_Q_IMPROVE = 0.002
E19_ZENCHI_IMPROVE = 50
FPS = 30
RAW_COLUMNS = ("display_p1", "source", "adv_raw_last", "t_sec", "game_idx")
SCENE_START, SCENE_END, WIN_THRESHOLD = 2750.0, 2775.0, 0.95
ADV_CLIP = 100.0


def directory(variant: str, source: str) -> Path:
    """再生出力 (run_e36 と同じ構造)。"""
    root = OUT_ROOT / variant / "on"
    return root / source if source in ("review", "zenchi") else root / "renders" / source / "on"


def load(variant: str, source: str) -> dict[str, np.ndarray]:
    """display.npz を独立コピーで読む。"""
    with np.load(directory(variant, source) / "display.npz") as data:
        return {key: data[key].copy() for key in data.files}


def shown_probability(display: dict[str, np.ndarray]) -> np.ndarray:
    """実表示 display_adv を本番と同じ変換で勝率へ戻す。"""
    return jumps.adv_to_prob(display["display_adv"], jumps.winprob_k())


def scene_first_sec(variant: str) -> float | None:
    """review の実表示 (display_adv) が場面内で初めて 2P勝率<=5% になる時刻。"""
    data = load(variant, "review")
    mask = (data["t_sec"] >= SCENE_START) & (data["t_sec"] <= SCENE_END) & (
        shown_probability(data) >= WIN_THRESHOLD)
    return float(data["t_sec"][mask][0]) if mask.any() else None


def display_q(variant: str) -> dict:
    """q 動画の表示ベース log loss (display_p1 を display_adv 由来の勝率へ置換して標準採点器へ通す)。"""
    from scripts import aggregate_e3_exchange_eval_20260926 as e3
    source = SOURCES[0]
    data = load(variant, source)
    windows, _ = e3.outcomes(source)
    replaced = dict(data, display_p1=shown_probability(data))
    return e3.m3_scores(replaced, windows)["groups"]["all"]


def death_immediacy(variant: str) -> dict:
    """確定死亡の行で display_adv が確定値そのもの (遅延0) か。分母=確定死亡行数。"""
    exact = total = 0
    for source in SOURCES:
        data = load(variant, source)
        rows = data["source"] == "confirmed_death"
        p = np.clip(data["display_p1"][rows], 1e-9, 1 - 1e-9)
        k = jumps.winprob_k()
        expected = np.clip(np.log(p / (1 - p)) / k if k else (p - 0.5) * 200.0, -ADV_CLIP, ADV_CLIP)
        total += int(rows.sum())
        exact += int(np.count_nonzero(np.abs(data["display_adv"][rows] - expected) <= 1e-9))
    return dict(frames=total, exact=exact)


def freshness(variant: str, column: str) -> dict:
    """5記録を記録ごとに測って合算 (隣接同値率・最長同値区間・更新回数/分)。"""
    parts = [display_freshness(load(variant, s)[column], FPS, allow_missing=True) for s in SOURCES]
    pairs = sum(p["adjacent_pairs"] for p in parts)
    minutes = sum(p["minutes"] for p in parts)
    return dict(frames=sum(p["frames"] for p in parts), adjacent_pairs=pairs,
                equal_fraction=sum(p["equal_pairs"] for p in parts) / pairs,
                longest_equal_seconds=max(p["longest_equal_seconds"] for p in parts),
                updates_per_minute=sum(p["updates"] for p in parts) / minutes)


def same_array(left: np.ndarray, right: np.ndarray) -> bool:
    """浮動小数は NaN 位置も含めて完全一致、その他は値の完全一致。"""
    return bool(np.array_equal(left, right, equal_nan=left.dtype.kind == "f"))


def raw_columns_identical(variant: str, baseline: str) -> dict:
    """平滑前の列 (確率・由来・保持値・時刻・試合) が OFF と全記録で完全一致か。"""
    result = {}
    for source in SOURCES:
        left, right = load(baseline, source), load(variant, source)
        result[source] = all(same_array(left[c], right[c]) for c in RAW_COLUMNS)
    return result


def jump_summary(variant: str) -> dict:
    """5記録の飛びを同一測定器で数える。"""
    results = {s: jumps.measure(directory(variant, s) / "display.npz", directory(variant, s) / "events.jsonl")
               for s in SOURCES}
    return jumps.summarize(results)


def standard(variant: str) -> dict:
    """E36b 標準採点器 (q・zenchi・誤発火・監査・第14試合)。場面だけ実表示由来へ差し替える。"""
    out = OUT_ROOT / variant
    report_e36b.OUT = out
    report_e36.RECORDS = EXEV_ROOT / "logs/r1b/records"
    legacy_scene = report_e36.first_scene_sec
    report_e36.first_scene_sec = lambda: scene_first_sec(variant)
    try:
        summary = report_e36b.report()
    finally:
        report_e36.first_scene_sec = legacy_scene
    return summary


def evaluate(variant: str, baseline: str) -> dict:
    """全指標を集めて事前登録の門を判定する。"""
    summary = standard(variant)
    metrics = dict(standard=summary, jumps=jump_summary(variant), display_q=display_q(variant),
                   death=death_immediacy(variant), fresh_adv=freshness(variant, "display_adv"),
                   fresh_eval=freshness(variant, "display_p1"),
                   raw_identical=raw_columns_identical(variant, baseline) if variant != baseline else {})
    save_json(OUT_ROOT / variant / "FULL_METRICS.json", metrics)
    return metrics


def gates(variant: str, baseline: str, metrics: dict, base: dict) -> dict:
    """事前登録の門を機械的に判定する (結果を見て調整しない)。"""
    std, b_std = metrics["standard"], base["standard"]
    common = dict(
        q=std["q"]["frames"] == 6526 and std["q"]["log_loss"] <= Q_BASELINE + Q_TOLERANCE,
        zenchi=std["zenchi"]["frames"] == ZENCHI_FRAMES and std["zenchi"]["hits"] >= ZENCHI_HITS_MIN,
        false_fire=std["deaths"]["unlabelled"] == 0 and std["deaths"]["false"] == 0,
        false_certainty=std["audit"]["bound_false"] == 0 and std["audit"]["unresolved"] == 0,
        scene=std["scene_first_sec"] is not None and std["scene_first_sec"] <= SCENE_DEADLINE,
        game14=std["game14_false_times"] == [])
    if "e19" in variant:
        improve = (std["q"]["log_loss"] <= Q_BASELINE - E19_Q_IMPROVE
                   or std["zenchi"]["hits"] >= ZENCHI_HITS_MIN + E19_ZENCHI_IMPROVE)
        fresh = (metrics["fresh_eval"]["equal_fraction"] <= base["fresh_eval"]["equal_fraction"] + FRESH_EQUAL_TOLERANCE
                 and metrics["fresh_eval"]["longest_equal_seconds"] <= base["fresh_eval"]["longest_equal_seconds"])
        common.update(improvement=improve, freshness_eval=fresh)
    if variant[0] in "ab":
        common.update(switch_gates(metrics, base))
    return dict(common, passed=all(common.values()))


def switch_gates(metrics: dict, base: dict) -> dict:
    """切替平滑の必須門 (飛び削減・正当変化の保存・鮮度・表示q・平滑前の列の不変)。"""
    j, bj = metrics["jumps"], base["jumps"]
    legit = lambda s: s["by_kind"].get("fire", 0) + s["by_kind"].get("death", 0)  # noqa: E731
    fa, ba = metrics["fresh_adv"], base["fresh_adv"]
    return dict(
        display_q=metrics["display_q"]["log_loss"] <= base["display_q"]["log_loss"] + DISPLAY_Q_TOLERANCE,
        jump_reduction=j["non_event_switch_jumps"] <= BASELINE_NON_EVENT_SWITCH_JUMPS * JUMP_REDUCTION_MAX_RATIO,
        baseline_jump_count=bj["non_event_switch_jumps"] == BASELINE_NON_EVENT_SWITCH_JUMPS,
        death_immediate=metrics["death"]["exact"] == metrics["death"]["frames"] > 0,
        legit_jumps_kept=legit(j) >= LEGIT_JUMP_MIN_RATIO * legit(bj),
        fresh_equal=fa["equal_fraction"] <= ba["equal_fraction"] + FRESH_EQUAL_TOLERANCE,
        fresh_longest=fa["longest_equal_seconds"] <= ba["longest_equal_seconds"],
        fresh_updates=fa["updates_per_minute"] >= UPDATES_MIN_RATIO * ba["updates_per_minute"],
        raw_columns=all(metrics["raw_identical"].values()) and len(metrics["raw_identical"]) == len(SOURCES))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--baseline", default="off")
    args = parser.parse_args()
    metrics = evaluate(args.variant, args.baseline)
    if args.variant == args.baseline:
        print(json.dumps(dict(standard={k: v for k, v in metrics["standard"].items() if k != "risky"},
                              jumps=metrics["jumps"]["non_event_switch_jumps"]), ensure_ascii=False))
        return
    base = json.loads((OUT_ROOT / args.baseline / "FULL_METRICS.json").read_text(encoding="utf-8"))
    verdict = gates(args.variant, args.baseline, metrics, base)
    save_json(OUT_ROOT / args.variant / "VERDICT.json", verdict)
    print(json.dumps(verdict, ensure_ascii=False))


if __name__ == "__main__":
    main()
