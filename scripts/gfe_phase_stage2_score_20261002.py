"""案Cの採点関数をそのまま読み、事前登録条件1〜4・6だけを判定する。"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.gfe_phase_stage2_replay_20261002 import OUT, PREFIRE, EXEV, EVALSET, SET1, SET2, EXISTING, SOURCES
from scripts.gfe_phase_stage1_20261002 import save_json
from src.exchange_event_evaluator import file_sha256
import numpy as np

LIMIT_SET_LL, LIMIT_AGREE, FLIP_RATIO = .002, -.01, 1.2
SCENE_DEADLINE, LIMIT_LATENCY = 2766., 20.
MODES = ("baseline", "V3", "V1b")


def extract(path: Path, names: tuple[str, ...], namespace: dict) -> SimpleNamespace:
    """参照実装の純粋関数をASTで無改変読込みし、起動・外部書込みを避ける。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def metric_functions() -> SimpleNamespace:
    """案Cが使う試合採点・同seed対bootstrapのコードを共有する。"""
    from scripts.report_e7_exchange_20260927 import m4
    legacy = SimpleNamespace(flips=lambda display: int(m4(display)["flips"]))
    namespace = dict(np=np, json=json, Path=Path, BOOTSTRAPS=10000, SEED=0,
        EPSILON=1e-7, LATE_FRACTION=2/3, CI_PERCENTILES=(2.5, 97.5), legacy=legacy,
        JOIN_COLUMNS=("t_sec", "display_p1", "display_adv", "game_idx", "prefire_mask", "prediction_delta"))
    return extract(PREFIRE / "scripts/prefire_v6_score.py",
                   ("load", "events", "paired", "game_metrics", "join_displays", "false_deaths"), namespace)


def root(mode: str, source: str) -> Path:
    """全比較をこのworktreeに保存した同構成再生へ揃える。"""
    return OUT / mode / "replay" / source


def score_set(path: Path, sources: tuple[str, ...], metric: SimpleNamespace) -> dict:
    """同一表示行・固定公式試合窓で母数を変えずに採点する。"""
    games = json.loads(path.read_text())["games"]
    frames, deaths, display_frames = {}, {}, {}
    for mode in MODES:
        parts, cases = [], set()
        for source in sources:
            directory = root(mode, source)
            part, old = metric.load(directory), metric.load(root("baseline", source))
            np.testing.assert_array_equal(part["t_sec"], old["t_sec"])
            parts.append(part)
            cases.update(metric.false_deaths(metric.events(directory), games))
        display = metric.join_displays(parts)
        frames[mode] = [metric.game_metrics(display, game, np.zeros(len(display["t_sec"]), dtype=bool))
                        for game in games]
        for row, game in zip(frames[mode], games):
            row["frames"] = int(((display["t_sec"] >= game["start"]) & (display["t_sec"] < game["end"])).sum())
        deaths[mode], display_frames[mode] = len(cases), len(display["t_sec"])
    comparisons = {}
    for mode in MODES[1:]:
        comparisons[mode] = {}
        for name in ("m_time", "agree"):
            old, new = [np.asarray([row[name] for row in frames[m]]) for m in ("baseline", mode)]
            value = metric.paired(new, old, name == "agree")
            value["candidate"] = value.pop("phase6")
            comparisons[mode][name] = value
    return dict(games=len(games), display_frames=display_frames,
                evaluated_frames={m: sum(r["frames"] for r in rows) for m, rows in frames.items()},
                late_frames={m: sum(r["agree"][1] for r in rows) for m, rows in frames.items()},
                rows=frames, false_deaths=deaths,
                flips={m: sum(r["flips"] for r in rows) for m, rows in frames.items()},
                comparisons=comparisons)


def regression(mode: str, metric: SimpleNamespace) -> dict:
    """案Cと同じWIN・パネル補完・観測死亡で既存5記録を採点する。"""
    from scripts import report_e35 as auditor, report_e36b
    from scripts import report_e10b_exchange_20260927 as labels
    from scripts import aggregate_e3_exchange_eval_20260926 as metrics
    from src.exchange_event_record import read_records
    fixed = extract(PREFIRE / "scripts/prefire_v6_regression.py", ("fixed_labels",),
        dict(EXEV=EXEV, BASELINE_DIRS=EXISTING, OUT=OUT/mode, labels=labels, metrics=metrics))
    existing = fixed.fixed_labels()
    old_events, old_read = auditor.events, auditor.read_records
    auditor.events = lambda source: metric.events(root(mode, source))
    auditor.read_records = lambda path: read_records(EXEV / "logs/pending_expiry/full/records" / path.name)
    try:
        rows = [row for source in EXISTING for row in auditor.deaths(source, existing)]
    finally:
        auditor.events, auditor.read_records = old_events, old_read
    assert not any(row["unresolved"] for row in rows)
    legacy = extract(PREFIRE / "scripts/prefire_gate_report_20260930.py", ("scene_first",),
        dict(Path=Path, SCENE_START=2750., SCENE_END=2775., WIN_THRESHOLD=.95))
    return dict(records=len(EXISTING), cases=len(rows), false=sum(bool(r["false"]) for r in rows),
        rows=rows, scene_first_sec=legacy.scene_first(root(mode, "review") / "display.npz"),
        game14=report_e36b.game14_false_rows(metric.events(root(mode, "q_7gc4TgFig"))))


def latency(mode: str) -> dict:
    """全静止評価通知のP95を取り、HistGBC推論だけの時間も併記する。"""
    arrays = {"total_ms": [], "inference_ms": []}
    sources, calls, inference_calls = {}, [], []
    for source in SOURCES:
        with np.load(root(mode, source) / "latency.npz") as data:
            for name in arrays:
                arrays[name].extend(data[name])
            sources[source] = len(data["total_ms"])
            calls.extend(data["call_ms"])
            inference_calls.extend(data["call_inference_ms"])
            assert int(data["calls_per_notification"].sum()) == len(data["call_ms"])
    assert len(arrays["total_ms"]) == len(arrays["inference_ms"]) > 0
    return dict(notifications=len(arrays["total_ms"]), sources=sources,
        p95_ms=float(np.percentile(arrays["total_ms"], 95)),
        inference_p95_ms=float(np.percentile(arrays["inference_ms"], 95)),
        gfe_calls=len(calls), call_p95_ms=float(np.percentile(calls, 95)),
        inference_call_p95_ms=float(np.percentile(inference_calls, 95)),
        maximum_ms=float(np.max(arrays["total_ms"])))


def judge(sets: list[dict], regressions: dict, speeds: dict, metric: SimpleNamespace) -> dict:
    """条件の不等号と母数115試合を事前登録のまま適用する。"""
    combined = {m: np.array([r["m_time"] for s in sets for r in s["rows"][m]]) for m in MODES}
    assert len(combined["baseline"]) == 115
    result = {}
    for mode in MODES[1:]:
        overall = metric.paired(combined[mode], combined["baseline"])
        overall["candidate"] = overall.pop("phase6")
        old = regressions[mode]
        gates = dict(condition1=overall["ci95"][1] < 0,
            condition2=all(s["comparisons"][mode]["m_time"]["ci95"][1] < LIMIT_SET_LL for s in sets),
            condition3=all(s["comparisons"][mode]["agree"]["ci95"][0] > LIMIT_AGREE and
                s["flips"][mode] <= FLIP_RATIO*s["flips"]["baseline"] for s in sets),
            condition4=all(s["false_deaths"][mode] <= s["false_deaths"]["baseline"] for s in sets)
                and old["false"] <= regressions["baseline"]["false"]
                and old["scene_first_sec"] is not None and old["scene_first_sec"] <= SCENE_DEADLINE
                and not old["game14"], condition6=speeds[mode]["p95_ms"] <= LIMIT_LATENCY)
        result[mode] = dict(overall=overall, gates=gates, passed=all(gates.values()))
    return result


def main() -> None:
    """全成果物が揃ってから機械判定を保存する。採否は統括が決める。"""
    metric = metric_functions()
    sets = [score_set(EVALSET/"labels.json", SET1, metric),
            score_set(EVALSET/"set2/labels.json", SET2, metric)]
    assert [s["games"] for s in sets] == [57, 58]
    regressions = {m: regression(m, metric) for m in MODES}
    speeds = {m: latency(m) for m in MODES}
    result = dict(status="COMPLETE", set1=sets[0], set2=sets[1], regression=regressions,
        latency=speeds, decisions=judge(sets, regressions, speeds, metric), adopted=False,
        scorer_sha256={name: file_sha256(PREFIRE/"scripts"/name) for name in
            ("plan_c_t_score.py", "prefire_v6_score.py", "prefire_v6_regression.py", "prefire_gate_report_20260930.py")})
    save_json(OUT/"METRICS.json", result)
    from scripts.gfe_phase_stage2_report_20261002 import write_summary
    write_summary(result)
    print(json.dumps(result["decisions"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
