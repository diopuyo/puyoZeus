"""推定ファイルを凍結した後に採点する。神託盤はこの入口以外から読まない。"""
from __future__ import annotations

from collections import Counter
import json
import numpy as np
from scripts.audit_exchange_counts_v33 import OUT
from scripts.exchange_review_v32_data import BASE, VIDEOS
from scripts.exchange_review_io import save, sha256
from scripts.exchange_review_v3_fit import write_csv
from src.exchange_count_prediction import simulate, histogram, interval, LOW_SPEC_RATIO
from src.exchange_count_ledger import gross, STEP_SEC, FORMULA_SCORE_DELAY_SEC
from src.b3_search import points_to_ojama


def target(truth: dict, ocr: dict, ledger: dict) -> dict:
    steps = truth['steps']
    times = [next(r['t'] for r in ocr['records'] if r['side'] == truth['side'] and
                  r['t'] > s['t_sec'] and r['score'] is not None) for s in steps]
    scores = [s['left']*s['right'] for s in steps]
    sent = gross(scores, times, ledger['carry'], ledger['first_move'])[0]
    return dict(erased=[s['left']//10 for s in steps], bonuses=[s['right'] for s in steps],
                remaining=histogram(truth['end_hsv']), sent=sent, score=sum(scores), score_times=times,
                carry_before=ledger['carry'], first_move=ledger['first_move'], all_clear_bonus=0,
                rate_origin_sensitivity=[gross(scores, times, ledger['carry'], ledger['first_move']+d)[0]
                                         for d in (-.5, .5)],
                rate_origin_frame_sensitivity=[gross(scores, times, ledger['carry'], ledger['first_move']+d)[0]
                                               for d in (-1/30, 1/30)])


def score(values: list[dict], actual: dict) -> dict:
    best = max(values, key=lambda v: v['probability']) if values else {}
    predicted = best.get('erased', [])
    stages = [i < len(predicted) and predicted[i] == n for i, n in enumerate(actual['erased'])]
    coverage, widths = [], []
    for i, n in enumerate(actual['erased']):
        bounds = interval(values, 'erased', i)
        coverage.append(bool(bounds and bounds['low'] <= n <= bounds['high']))
        widths.append(bounds['high']-bounds['low'] if bounds else None)
    send = interval(values, 'sent')
    remain = [interval(values, 'remaining', i) for i in range(len(actual['remaining']))]
    return dict(stage_matches=sum(stages), stage_n=len(stages),
                unseen_stage_matches=sum(stages[1:]), unseen_stage_n=max(0, len(stages)-1),
                event_stage_match=predicted == actual['erased'], sent_match=best.get('sent') == actual['sent'],
                remaining_match=best.get('remaining') == actual['remaining'],
                stage_coverage=sum(coverage), stage_widths=widths,
                sent_covered=bool(send and send['low'] <= actual['sent'] <= send['high']),
                remaining_covered=all(b and b['low'] <= n <= b['high'] for b, n in zip(remain, actual['remaining'])),
                sent_interval=send, remaining_intervals=remain,
                predicted_erased=predicted, predicted_remaining=best.get('remaining'), predicted_sent=best.get('sent'))


def event_scores(truth: dict, prediction: dict, item: dict, ocr: dict, native: object) -> dict:
    actual = target(truth, ocr, prediction['ledger'])
    oracle = simulate(truth['oracle_origin'], native)
    old_board = item['origin'].get('origin')
    old = simulate(old_board, native) if old_board else None
    if old:
        old.update(sent=points_to_ojama(old['score'], item['fire']['exchange_input']['elapsed_sec']), probability=1.)
    variants = {'v32': [old] if old else [], **prediction['variants']}
    results = {name: score(values, actual) for name, values in variants.items()}
    updates = []
    for update in prediction.get('updates', []):
        if update['observed_steps'] > len(actual['erased']):
            continue
        measured = score(update['values'], actual)
        k = update['observed_steps']
        best = measured['predicted_erased']
        measured['future_n'] = len(actual['erased'][k:])
        measured['future_matches'] = sum(i < len(best) and best[i] == n for i, n in enumerate(actual['erased']) if i >= k)
        updates.append(dict(update_time=update['input_sec'], observed=k, metrics=measured))
    return dict(event_id=truth['event_id'], actual=actual, variants=results, updates=updates,
                oracle_erased=oracle['erased'] == actual['erased'],
                oracle_stage_matches=sum(a == b for a, b in zip(oracle['erased'], actual['erased'])),
                oracle_bonus_matches=sum(a == b for a, b in zip(oracle['bonuses'], actual['bonuses'])),
                oracle_sent=gross(oracle['scores'],
                    [prediction['input_sec']+FORMULA_SCORE_DELAY_SEC+STEP_SEC*i for i in range(len(oracle['scores']))],
                    prediction['ledger']['carry'], prediction['ledger']['first_move'])[0] == actual['sent'],
                oracle_remaining=oracle['remaining'] == actual['remaining'])


def sequential_summary(rows: list[dict]) -> dict:
    matches, coverage, n = 0, 0, 0
    final_sent, final_remaining = 0, 0
    for row in rows:
        snapshots = [(1, row['variants']['board_candidates'])]+[(u['observed'], u['metrics']) for u in row['updates']]
        for k, metrics in snapshots:
            if k >= len(row['actual']['erased']):
                continue
            n += 1
            prediction = metrics['predicted_erased']
            matches += k < len(prediction) and prediction[k] == row['actual']['erased'][k]
        last = snapshots[-1][1]
        final_sent += last['sent_match']
        final_remaining += last['remaining_match']
    return dict(next_unseen_matches=matches, next_unseen_n=n,
                final_observation_sent=final_sent, final_observation_remaining=final_remaining,
                events=len(rows), note='最終観測後の送り量・残存数を初回予測の成績に混ぜない')


def aggregate(rows: list[dict], name: str) -> dict:
    values = [r['variants'][name] for r in rows]
    keys = ('stage_matches', 'stage_n', 'unseen_stage_matches', 'unseen_stage_n', 'event_stage_match',
            'sent_match', 'remaining_match', 'stage_coverage', 'sent_covered', 'remaining_covered')
    result = {k: sum(v[k] for v in values) for k in keys}
    widths = [w for v in values for w in v['stage_widths'] if w is not None]
    result.update(events=len(values), mean_stage_interval_width=float(np.mean(widths)) if widths else None)
    return result


def causal_audit(predictions: dict) -> dict:
    policy = json.loads((OUT/'policy.json').read_text())
    assert policy['training_videos'] == list(VIDEOS) and policy['excluded_video'] == 80
    assert predictions['policy_sha256'] == sha256(OUT/'policy.json')
    comparisons, times = 0, []
    legacy = json.loads((BASE/'video_80_g7_v3_2/events.json').read_text())['origins']
    for item in legacy:
        origin, stamp = item['origin'], item['fire']['t_sec']
        evidence = [origin.get('before_sec'), origin.get('input_sec'),
                    origin.get('image_evidence', {}).get('latest_sec'),
                    origin.get('pair_metadata', {}).get('observed_sec')]
        for time in evidence:
            if time is not None:
                assert time <= stamp
                comparisons += 1
    for row in predictions['rows']:
        for stamp in row.get('evidence_times', []):
            assert stamp <= row['input_sec']
            comparisons += 1
        if 'ledger' in row:
            assert row['ledger']['observed_sec'] <= row['input_sec']
            assert row['first']['t'] <= row['input_sec']
            times.append(row['low_spec_ms'])
            comparisons += 2
        for update in row.get('updates', []):
            assert update['input_sec'] >= row['input_sec']
            for formula in row['formulas'][:update['observed_steps']]:
                assert formula['t'] <= update['input_sec']
                comparisons += 1
    return dict(comparisons=comparisons, future_references=0, predictions=len(times),
                low_spec_ms=dict(median=float(np.median(times)), p95=float(np.percentile(times, 95)), maximum=max(times)),
                over_500ms=sum(t > 500 for t in times), ratio=LOW_SPEC_RATIO,
                note='低スペック機の実測ではなく既存の実測比による換算。デコード・配信キューは含まない。')


def main() -> None:
    import puyo_core
    predictions = json.loads((OUT/'predictions.json').read_text())
    by_id = {r['event_id']: r for r in predictions['rows']}
    truth = json.loads((OUT/'truth_probe.json').read_text())
    ocr = json.loads((OUT/'ocr.json').read_text())
    items = {r['event_id']: r for r in json.loads((BASE/'video_80_g7_v3_2/events.json').read_text())['origins']}
    rows = [event_scores(t, by_id[t['event_id']], items[t['event_id']], ocr, puyo_core)
            for t in truth if 'duplicate_of' not in t]
    save(OUT/'scored_events.json', rows)
    result = dict(notification_n=len(truth), event_n=len(rows), stage_n=sum(len(r['actual']['erased']) for r in rows),
                  variants={name: aggregate(rows, name) for name in rows[0]['variants']},
                  oracle=dict(events=len(rows), erased=sum(r['oracle_erased'] for r in rows),
                              stage_matches=sum(r['oracle_stage_matches'] for r in rows),
                              bonus_matches=sum(r['oracle_bonus_matches'] for r in rows),
                              sent=sum(r['oracle_sent'] for r in rows),
                              remaining=sum(r['oracle_remaining'] for r in rows)),
                  sequential=sequential_summary(rows),
                  causal=causal_audit(predictions))
    save(OUT/'verification.json', result)
    write_csv(OUT/'truth.csv', [dict(notification_n=len(truth), physical_event_n=len(rows),
        total_stage_n=result['stage_n'], event_id=r['event_id'], **r['actual']) for r in rows])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
