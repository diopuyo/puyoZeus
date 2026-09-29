"""旧E32の固定母数と事前登録閾値でE34cを判定する。"""
from __future__ import annotations

from scripts import e34b_residuals as residual
from scripts import report_e34b as prior
from scripts.collect_e34c import OUT
from scripts.run_e17_ablation_20260928 import ALL_SOURCES
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.replay_exchange_event_20260926 import compare
from pathlib import Path

Q_MAX, ZENCHI_MIN, SCORE_MAX = .509977, .9167, 836.


def old_equivalence() -> dict:
    """ON開始前に旧E32とOFFの全出力を厳密照合する。"""
    prior.OUT = OUT
    equivalence = {}
    for source in ALL_SOURCES:
        suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        equivalence[source] = compare(Path('logs/e32/on')/suffix, prior.directory('off', source))
    return equivalence


def report() -> dict:
    """OFFの旧出力再現と同じ採点行を確認し、基準値の再定義を禁止する。"""
    assert prior.read(OUT/'INPUT_CHECK.json')['passed']
    prior.OUT = residual.OUT = OUT
    equivalence = old_equivalence()
    values = {v: prior.metrics(v) for v in ('off', 'on')}
    scores, b = prior.score_metrics(), values['on']
    old_metrics = prior.read(Path('logs/e32/on/METRICS.json'))
    old_scores = prior.read(Path('logs/e33b/SCORES.json'))['off']
    for key in ('q', 'zenchi', 'deaths'):
        assert values['off'][key] == old_metrics[key], key
    for key in ('n', 'mean', 'p50', 'p95'):
        assert scores['off'][key] == old_scores[key], key
    assert values['off']['q']['frames'] == b['q']['frames'] == 6526
    assert values['off']['zenchi']['frames'] == b['zenchi']['frames'] == 8333
    gates = dict(q=b['q']['log_loss'] <= Q_MAX, zenchi=b['zenchi']['agreement'] >= ZENCHI_MIN,
        deaths=b['deaths']['unlabelled'] == 0 and b['deaths']['total'] > 0 and
               b['deaths']['false']/b['deaths']['total'] <= 1/28,
        score_mean=scores['on']['mean'] is not None and scores['on']['mean'] <= SCORE_MAX)
    for variant, value in values.items():
        value.pop('gates', None)
        value.pop('candidate', None)
        save_json(OUT/variant/'METRICS.json', value)
    result = dict(metrics=values, scores=scores, gates=gates, passed=all(gates.values()),
        limits=dict(q=Q_MAX, zenchi=ZENCHI_MIN, score_mean=SCORE_MAX, deaths=1/28),
        old_e32=old_metrics, old_e32_score=old_scores,
        comparisons=prior.compare_rows(), old_equivalence=equivalence,
        inputs=prior.read(OUT/'INPUT_CHECK.json'), cohort=prior.read(OUT/'COHORT.json'),
        adoptions={v: prior.adoptions(v) for v in ('off', 'on')}, residuals=residual.residuals())
    save_json(OUT/'SUMMARY.json', result)
    return result


if __name__ == '__main__':
    print(report())
