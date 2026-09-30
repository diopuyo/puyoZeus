"""D5固定検収とR1先行確認を別母数のまま集計する。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts import report_e35 as e35
from scripts import run_e17_ablation_20260928 as prior
from scripts.diagnose_d5 import directory, read_events, LONG_LEAD_SEC
from scripts.measure_e23_display_20260928 import displayed
from scripts.run_d5 import OUT, SOURCES
from scripts.run_e3_exchange_eval_20260926 import save_json

SCENE_START, SCENE_END, DEADLINE = 2750., 2775., 2766.
SCENE_STAMPS = (2750.85, 2751.3833333333, 2753.3166666667, 2755.45, 2758.95,
                2759.05, 2760., 2762., 2764., 2766., 2768., 2770.)


def read(path: Path) -> Any:
    """UTF-8原票を読む。"""
    return json.loads(path.read_text())


def all_single_cases() -> dict:
    """初回経路が別だった試合も含め、全単発確定を試合・側で集計する。"""
    baseline = read(OUT/'BASELINE_PATH_AUDIT.json')
    rows = []
    for source in SOURCES:
        cases = {}
        events = read_events(source)
        for event in events:
            for value in event['values']:
                if value['source'] != 'unavoidable_death':
                    continue
                for side in value['dead_sides']:
                    idx = int(side == '2P')
                    if value.get('multi_landing', [{}, {}])[idx].get('reason') != 'already_dead':
                        continue
                    if any(v.get('dead') and v['side'] == side for v in value.get('post_counter_bound', [])):
                        continue
                    key = (event['game_idx'], side)
                    cases.setdefault(key, value)
        for (game, side), value in cases.items():
            reference = next(r for r in baseline if (r['source'], r['game'], r['side']) == (source, game, side))
            own = reference['observed_death_sec']
            lead = None if own is None else own-value['t_sec']
            chains = [dict(trigger=c['trigger_sec'], score=c['score_delta'], formula=c['formula_total'])
                for e in events if e['game_idx'] == game for c in e['chains']
                if c['side'] == side and c['trigger_sec'] > value['t_sec']]
            rows.append(dict(source=source, game=game, side=side, first=value['t_sec'], lead=lead,
                long_lead=lead is not None and lead > LONG_LEAD_SEC, false=reference['false'],
                later_chains=chains, verified_counter=any((c['score'] or 0) >= 70 for c in chains)))
    result = dict(records=len(SOURCES), cases=len(rows), rows=rows,
        other_risky=sum(not r['false'] and (r['long_lead'] or r['verified_counter']) for r in rows),
        later_notification=sum(bool(r['later_chains']) for r in rows),
        long_lead=sum(r['long_lead'] for r in rows))
    save_json(OUT/'SINGLE_PATH_AUDIT.json', result)
    return result


def timeline() -> dict:
    """採用確率と実表示、予測得点・残量・証明理由を同じ時刻軸へ載せる。"""
    root = OUT/'r1_e35/review'
    data = np.load(root/'display.npz')
    times, smooth = displayed(root/'display.npz')
    traces = read(root/'scene_timeline.json')
    events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    values = [v for e in events for v in e['values'] if SCENE_START <= v['t_sec'] <= SCENE_END]
    mask = (times >= SCENE_START) & (times <= SCENE_END)
    selected = times[mask & (data['display_p1'] >= .95)]
    shown = times[mask & (smooth >= .95)]
    dead = [v['t_sec'] for v in values if v['source'] == 'unavoidable_death' and '2P' in v['dead_sides']]
    samples = sorted(set((*SCENE_STAMPS, *(dead[:1]), *(selected[:1]), *(shown[:1]))))
    rows = []
    for stamp in samples:
        index = int(np.argmin(abs(times-stamp)))
        trace = next((v for v in reversed(traces) if v['t_sec'] <= times[index]), None)
        rows.append(dict(t_sec=float(times[index]), selected_p2=1-float(data['display_p1'][index]),
                         displayed_p2=1-float(smooth[index]), trace=trace))
    result = dict(first_selected=None if not len(selected) else float(selected[0]),
        first_display=None if not len(shown) else float(shown[0]), first_death=min(dead) if dead else None,
        deadline=DEADLINE, rows=rows)
    result['decomposition'] = scene_decomposition(traces, result)
    save_json(OUT/'R1_E35_TIMELINE.json', result)
    return result


def scene_decomposition(traces: list[dict], scene: dict) -> dict:
    """予測の公開、証明入力成立、断定と平滑化の待ち時間を区別する。"""
    predicted, final_score, proofs = [], [], []
    for row in traces:
        for chain in row['chains']:
            if chain['side'] != '2P' or chain['trigger'] < SCENE_START:
                continue
            if chain['score']:
                predicted.append(row['t_sec'])
            if chain['score'] is not None and abs(chain['score']-75440.) < .01:
                final_score.append(row['t_sec'])
        last = row['last'] or {}
        for proof in last.get('post_counter_bound', []):
            if proof['side'] == '2P' and 'proofs' in proof:
                proofs.append(row['t_sec'])
    selected, shown = scene['first_selected'], scene['first_display']
    return dict(prediction_first=min(predicted) if predicted else None,
        prediction_75440_first=min(final_score) if final_score else None,
        proof_inputs_first=min(proofs) if proofs else None,
        confirmation_first=scene['first_death'],
        ema_sec=None if shown is None or selected is None else shown-selected,
        deadline_delay=None if shown is None else max(0., shown-DEADLINE))


def report() -> dict:
    """入力・採点行を固定し、母数と未照合数を含めて合否を保存する。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, .509977, .9167
    result = prior.report('on')
    assert result['q']['frames'] == 6526 and result['zenchi']['frames'] == 8333
    e35.OUT = OUT
    audit = e35.audit_all(prior.report_base.death_metrics.deaths())
    result['gates']['false_certainty'] = audit['false'] == 0 and audit['unresolved'] == 0
    result.update(death_audit=audit, path_audit=all_single_cases(), r1_scene=timeline())
    result['passed'] = all(result['gates'].values())
    checks = {}
    for source in SOURCES:
        before = np.load(directory(Path('logs/e35/on'), source)/'display.npz')
        after = np.load(directory(OUT/'on', source)/'display.npz')
        for key in ('t_sec', 'game_idx', 'state1', 'state2', 'score1', 'score2'):
            np.testing.assert_array_equal(before[key], after[key])
        checks[source] = len(after['t_sec'])
    save_json(OUT/'FIXED_ROWS.json', checks)
    save_json(OUT/'SUMMARY.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('death_audit', 'path_audit', 'r1_scene')},
                     ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    report()
