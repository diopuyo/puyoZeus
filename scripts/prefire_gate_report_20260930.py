"""発火前予測の事前登録門 (exev DECISIONS.md 2026-09-30) で ON 再生を採点する (Phase 2)。

- OFF = 現本番記録 `exev/logs/pending_expiry/e36b_on/on` (同じ5記録・同じ行)。ON = logs/prefire_prediction/replay/on
- 発火前区間・撃ち合いの目標値は OFF の表示とイベントで固定する (ON の結果で母数を変えない)
使い方: PYTHONPATH=. python -m scripts.prefire_gate_report_20260930
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts import aggregate_e3_exchange_eval_20260926 as e3
from scripts import prefire_oracle_ceiling_20260930 as oracle
from scripts.run_prefire_replay_20260930 import BASELINE, BASELINE_DIRS, OUT as REPLAY

ON = REPLAY / 'on'
REPORT = Path('logs/prefire_prediction/GATE.json')
# 構成は --variant で選ぶ (on=構成A、on_mc=構成B)。出力は GATE_<variant>.json
Q_SOURCE, ZENCHI, REVIEW = 'q_7gc4TgFig', 'zenchi', 'review'
Q_MAX, Q_IMPROVED = .507567, .507567 - .002
PREFIRE_LL_MAX = .7308 - .035
ANTICIPATION_MIN = 23 + 10
ANTICIPATION_LEAD_SEC = 3.0
ZENCHI_HITS_MIN, DEATHS_TOTAL = 7671, 37
SCENE_START, SCENE_END, SCENE_DEADLINE, WIN_THRESHOLD = 2750., 2775., 2766., .95
FLIP_RATIO_MAX = 1.2
ORACLE_Q_LL = .4769


def load(root: Path, source: str) -> dict:
    """display.npz を読む。"""
    with np.load(root / 'display.npz') as data:
        return {k: data[k].copy() for k in data.files}


def dirs(source: str) -> tuple[Path, Path]:
    """(OFF, ON) の出力先。"""
    return BASELINE / BASELINE_DIRS[source], ON / source


def fixed_rows(source: str) -> tuple[dict, list[dict]]:
    """OFF の表示・イベントから撃ち合い行 (発火・目標値) を作る。"""
    off_display, events = oracle.load(source)
    return off_display, oracle.exchange_rows(off_display, events)


def anticipation(display_on: dict, rows: list[dict]) -> tuple[int, int]:
    """(半分以上を先取りした件数, 分母)。目標値は OFF で固定し、区間の値は ON の表示を使う。"""
    items = [oracle.window_freshness(display_on, r, ANTICIPATION_LEAD_SEC) for r in rows]
    ratios = [i['anticipation'] for i in items if i is not None and i['anticipation'] is not None]
    return sum(r >= .5 for r in ratios), len(ratios)


def events_identical(source: str) -> bool:
    """ON のイベント原票が OFF とバイト一致するか (予測層は表示だけを変える契約)。"""
    off, on = dirs(source)
    return (off / 'events.jsonl').read_bytes() == (on / 'events.jsonl').read_bytes()


def scene_first(path: Path) -> float | None:
    """3:00 場面で 2P≤5% (表示 EMA 後の 1P≥.95) の初時刻 (report_e36.first_scene_sec と同定義)。"""
    from scripts.measure_e23_display_20260928 import displayed
    times, smooth = displayed(path)
    hit = times[(times >= SCENE_START) & (times <= SCENE_END) & (smooth >= WIN_THRESHOLD)]
    return float(hit[0]) if len(hit) else None


def placebo_p1(display_on: dict, trace_path: Path) -> np.ndarray:
    """対照: 予測の増分 (p_shown − p_current) を trace 長の半分だけ循環ずらしして現在値へ足す。"""
    with np.load(trace_path) as data:
        table = dict(zip([str(c) for c in data['columns']], data['values'].T))
    p1 = display_on['display_p1'].copy()
    if not len(table['t_sec']):
        return p1
    delta = np.roll(table['p_shown'] - table['p_current'], len(table['t_sec']) // 2)
    index = np.searchsorted(display_on['t_sec'], table['t_sec'])
    ok = (index < len(p1)) & np.isclose(display_on['t_sec'][np.minimum(index, len(p1) - 1)], table['t_sec'])
    p1[index[ok]] = np.clip(table['p_current'][ok] + delta[ok], 0.0, 1.0)
    return p1


def flips(display: dict) -> int:
    """±3帯の反転回数 (report_e7.m4 と同定義)。"""
    from scripts.report_e7_exchange_20260927 import m4
    return int(m4(display)['flips'])


def q_block(display_off: dict, display_on: dict, rows: list[dict], p1_on: np.ndarray) -> dict:
    """q の全体と発火前3秒 (OFF 行で固定) の LL/AUC。"""
    windows = e3.outcomes(Q_SOURCE)[0]
    subset = oracle.inject(display_off, rows, ANTICIPATION_LEAD_SEC, 0.0)[1]
    return dict(all=oracle.q_scores(display_on, p1_on, windows, None),
                prefire3s=oracle.q_scores(display_on, p1_on, windows, subset))


def variant_scores(displays: dict, rows: dict, p1: dict) -> dict:
    """条件1つ (ON または対照) の門2・3・4 の値。"""
    q = q_block(displays['off'][Q_SOURCE], displays['on'][Q_SOURCE], rows[Q_SOURCE], p1[Q_SOURCE])
    counts = [anticipation(dict(displays['on'][s], display_p1=p1[s]), rows[s]) for s in BASELINE_DIRS]
    hits, total = sum(c[0] for c in counts), sum(c[1] for c in counts)
    return dict(q=q, anticipation=dict(hits=hits, denominator=total),
                gate2=q['all']['log_loss'] <= Q_IMPROVED, gate3=q['prefire3s']['log_loss'] <= PREFIRE_LL_MAX,
                gate4=hits >= ANTICIPATION_MIN)


def non_worse(displays: dict) -> dict:
    """門1 (悪化なし) の各項目を母数つきで返す。"""
    q = variant_scores_q_all(displays)
    zenchi = oracle.zenchi_hits(displays['on'][ZENCHI], displays['on'][ZENCHI]['display_p1'],
                                np.zeros(len(displays['on'][ZENCHI]['t_sec']), dtype=bool), False)
    identical = {s: events_identical(s) for s in BASELINE_DIRS}
    scene = scene_first(ON / REVIEW / 'display.npz')
    flip_off = sum(flips(displays['off'][s]) for s in BASELINE_DIRS)
    flip_on = sum(flips(displays['on'][s]) for s in BASELINE_DIRS)
    return dict(q=q, zenchi=zenchi, events_identical=identical, scene_first_sec=scene,
                flips=dict(off=flip_off, on=flip_on),
                passed=bool(q['frames'] == 6526 and q['log_loss'] <= Q_MAX and zenchi['frames'] == 8333
                            and zenchi['hits'] >= ZENCHI_HITS_MIN and all(identical.values())
                            and scene is not None and scene <= SCENE_DEADLINE
                            and flip_on <= FLIP_RATIO_MAX * flip_off))


def variant_scores_q_all(displays: dict) -> dict:
    """ON の q 全体 LL (e3.m3_scores と同定義)。"""
    return e3.m3_scores(displays['on'][Q_SOURCE], e3.outcomes(Q_SOURCE)[0])['groups']['all']


def timing() -> dict:
    """予測層の1通知あたり所要 (ms)。"""
    values = np.concatenate([np.load(ON / s / 'prefire_trace.npz')['values'][:, -1] for s in BASELINE_DIRS])
    pct = lambda q: float(np.percentile(values, q)) if len(values) else None
    return dict(n=int(len(values)), p50=pct(50), p95=pct(95), p99=pct(99), max=pct(100),
                mean=float(values.mean()) if len(values) else None)


def main() -> None:
    import argparse
    global ON, REPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='on', choices=('on', 'on_mc'))
    variant = parser.parse_args().variant
    ON, REPORT = REPLAY / variant, REPORT.with_name(f'GATE_{variant}.json')
    displays = {'off': {s: load(dirs(s)[0], s) for s in BASELINE_DIRS},
                'on': {s: load(dirs(s)[1], s) for s in BASELINE_DIRS}}
    rows = {s: fixed_rows(s)[1] for s in BASELINE_DIRS}
    on = variant_scores(displays, rows, {s: displays['on'][s]['display_p1'] for s in BASELINE_DIRS})
    off = variant_scores(dict(displays, on=displays['off']), rows,
                         {s: displays['off'][s]['display_p1'] for s in BASELINE_DIRS})
    placebo = variant_scores(displays, rows, {s: placebo_p1(displays['on'][s], ON / s / 'prefire_trace.npz')
                                              for s in BASELINE_DIRS})
    gate1 = non_worse(displays)
    placebo_failed = not (placebo['gate2'] and placebo['gate3'] and placebo['gate4'])
    result = dict(off=off, on=on, placebo=placebo, gate1=gate1, gate5_placebo_failed=placebo_failed,
                  leak_suspect=on['q']['all']['log_loss'] < ORACLE_Q_LL, timing_ms=timing(),
                  passed=bool(gate1['passed'] and on['gate2'] and on['gate3'] and on['gate4'] and placebo_failed))
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=1, default=float), flush=True)


if __name__ == '__main__':
    main()
