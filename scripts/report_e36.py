"""E36の事前登録(logs/e36/PREREGISTRATION.md)だけで採否を決める。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts import report_e35 as e35
from scripts import run_e17_ablation_20260928 as prior
from scripts.measure_e23_display_20260928 import displayed
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.run_e36 import OUT, RECORDS, SOURCES
from src.exchange_event_record import read_records

Q_MAX = .508040 + 5e-7      # R1b値の6桁丸め許容
Q_FRAMES, ZENCHI_FRAMES, ZENCHI_HITS_MIN = 6526, 8333, 7671
FALSE_MAX = 1/35
SCENE_START, SCENE_END, DEADLINE = 2750., 2775., 2766.
WIN_THRESHOLD = .95         # 2P勝率<=5% == P1表示>=.95
LONG_LEAD_SEC = 10.
COUNTER_SCORE = 70


def classify_route(first: dict, side: str) -> str:
    """初回確定が単発着弾(already_dead)でE35証明を伴わないかを返す。"""
    idx = int(side == '2P')
    single = first.get('multi_landing', [{}, {}])[idx].get('reason') == 'already_dead'
    bound = any(p.get('dead') and p['side'] == side for p in first.get('post_counter_bound', []))
    return 'single' if single and not bound else 'other'


def risky_row(case: dict, first: dict, chains: list[dict]) -> dict:
    """1件の確定を危険度(長い先行・後続の実反撃)で分類する。"""
    lead = case['lead_sec']
    return dict(source=case['source'], game=case['game'], side=case['side'], first=case['first_sec'],
        lead=lead, false=case['false'], route=classify_route(first, case['side']),
        long_lead=lead is not None and lead > LONG_LEAD_SEC, later_chains=chains,
        verified_counter=any((c['score'] or 0) >= COUNTER_SCORE for c in chains))


def summarize_risky(rows: list[dict]) -> dict:
    """誤りでないが危険な例を数えるだけ(合否に含めない)。"""
    risky = [r for r in rows if not r['false'] and (r['long_lead'] or r['verified_counter'])]
    return dict(cases=len(rows), single=sum(r['route'] == 'single' for r in rows),
        other_risky=len(risky), long_lead=sum(r['long_lead'] for r in rows),
        risky_rows=[dict(r, later_chains=len(r['later_chains'])) for r in risky])


def evaluate_gates(q: dict, zenchi: dict, deaths: dict, audit: dict, scene_sec: float | None) -> dict:
    """事前登録5条件を母数一致つきで判定する。"""
    return dict(
        q=q['frames'] == Q_FRAMES and q['log_loss'] <= Q_MAX,
        zenchi=zenchi['frames'] == ZENCHI_FRAMES and zenchi['hits'] >= ZENCHI_HITS_MIN,
        false_fire=deaths['unlabelled'] == 0 and deaths['false']/max(1, deaths['total']) <= FALSE_MAX,
        false_certainty=audit['bound_false'] == 0 and audit['unresolved'] == 0,
        scene=scene_sec is not None and scene_sec <= DEADLINE)


def first_scene_sec() -> float | None:
    """review表示列で2P勝率<=5%になる場面内の初時刻(D5のfirst_displayと同定義)。"""
    times, smooth = displayed(e35.directory('review')/'display.npz')
    hit = times[(times >= SCENE_START) & (times <= SCENE_END) & (smooth >= WIN_THRESHOLD)]
    return float(hit[0]) if len(hit) else None


def risky_cases(audit: dict) -> dict:
    """監査の全確定から、単発経路の危険例を列挙する。"""
    rows = []
    for source in SOURCES:
        events = e35.events(source)
        for case in (c for c in audit['rows'] if c['source'] == source):
            values = [v for e in events if e['game_idx'] == case['game'] for v in e['values']]
            first = next(v for v in values if v['source'] == 'unavoidable_death'
                         and abs(v['t_sec']-case['first_sec']) < 1e-6)
            chains = [dict(trigger=c['trigger_sec'], score=c['score_delta']) for e in events
                if e['game_idx'] == case['game'] for c in e['chains']
                if c['side'] == case['side'] and c['trigger_sec'] > case['first_sec']]
            rows.append(risky_row(case, first, chains))
    result = summarize_risky(rows)
    save_json(OUT/'RISKY_CASES.json', result)
    return result


def report() -> dict:
    """R1b記録を入力にした採点。E35監査の記録読込先だけR1bへ差し替える。"""
    prior.OUT = OUT
    result = prior.report('on')
    e35.OUT = OUT
    e35.read_records = lambda path: read_records(RECORDS/Path(path).name)
    audit = e35.audit_all(prior.report_base.death_metrics.deaths())
    scene = first_scene_sec()
    gates = evaluate_gates(result['q'], result['zenchi'], result['deaths'], audit, scene)
    summary = dict(q=result['q'], zenchi=result['zenchi'], deaths=result['deaths'],
        audit={k: v for k, v in audit.items() if k != 'rows'}, scene_first_sec=scene,
        gates=gates, passed=all(gates.values()), risky=risky_cases(audit))
    save_json(OUT/'SUMMARY.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'risky'}, ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    report()
