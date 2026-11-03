"""動画80への固定設定適用。正解表・神託盤・終了盤は読み込まない。"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter
import json
import numpy as np
from scripts.audit_exchange_counts_v33 import OUT
from scripts.exchange_review_v32_data import BASE, VIDEOS, source, video_path
from scripts.exchange_review_io import save, sha256
from scripts.replay_exchange_projection_shadow import updates
from scripts.exchange_counts_v33_collect import ImageSources, extract_sources
from src.exchange_count_prediction import candidates, simulate, condition, distribution, LOW_SPEC_RATIO
from src.exchange_count_ledger import ScorePrefix, gross, STEP_SEC, FORMULA_SCORE_DELAY_SEC

GAME, MS = 7, 1000


def with_send(values: list[dict], stamp: float, ledger: dict) -> list[dict]:
    for value in values:
        times = [stamp+FORMULA_SCORE_DELAY_SEC+STEP_SEC*i for i in range(len(value['scores']))]
        value['sent'] = gross(value['scores'], times, ledger['carry'], ledger['first_move'])[0]
    return values


def predict(item: dict, raw: object, sources: list[dict], first: dict | None,
            ledger: dict, policy: dict, native: object) -> dict:
    stamp, began = item['fire']['t_sec'], perf_counter()
    if raw.chain_event.mechanism != 'formula_read' or first is None:
        return dict(event_id=item['event_id'], input_sec=stamp, status='新しい初段式未確認', variants={})
    if ledger['carry'] is None or ledger['first_move'] is None:
        raise ValueError('送り量の予測には試合開始からの得点履歴が必要です')
    full = candidates(sources, item['candidates'], item['fire']['exchange_input']['colors'], policy, stamp, native)
    accepted, full_conflict = condition(full['candidates'], [first], stamp)
    main_values = distribution(with_send(deepcopy(accepted or full['candidates']), stamp, ledger), policy['temperature'])
    elapsed = (perf_counter()-began)*MS+sum(s.get('image_compute_ms', 0.) for s in sources)
    no_tape = candidates(sources, [], item['fire']['exchange_input']['colors'], policy, stamp, native)
    legacy = [dict(simulate(c['origin'], native), log_weight=c['rank_score']) for c in item['candidates']]
    variants = dict(board_candidates=full['candidates'], without_tape=no_tape['candidates'], legacy_formula=legacy)
    variants['no_formula'] = full['candidates']
    result, conflicts = {}, {}
    for key, values in variants.items():
        filtered, conflict = condition(values, [] if key == 'no_formula' else [first], stamp)
        # 矛盾時は未支持と明示した元分布を残す。正解を見て穴埋めしない。
        selected = filtered if filtered else values
        result[key] = distribution(with_send(deepcopy(selected), stamp, ledger), policy['temperature'])
        conflicts[key] = conflict
    result['board_candidates'] = main_values
    conflicts['board_candidates'] = full_conflict
    return dict(event_id=item['event_id'], input_sec=stamp, status='分布', variants=result,
                candidates=full['candidates'], conflicts=conflicts, first=first, ledger=ledger,
                elapsed_ms=elapsed, low_spec_ms=elapsed*LOW_SPEC_RATIO,
                benchmark_total_ms=(perf_counter()-began)*MS, evidence_times=[s['t'] for s in sources],
                exhausted=full['exhausted'])


def formula_step(row: dict, step: int) -> dict:
    return dict(t=row['t_sec'], step=step, erased=row['left']//10, bonus=row['right'])


def sequential(active: dict, step: dict, policy: dict) -> None:
    side = step['side']
    if side not in active:
        return
    record = active[side]
    if step['t_sec'] <= record['input_sec']:
        return
    formulas = record.setdefault('formulas', [record['first']])
    if step['t_sec']-formulas[-1]['t'] > 2.:  # 既存式セッションの切断時間。
        return
    formulas.append(formula_step(step, len(formulas)+1))
    began = perf_counter()
    values, conflict = condition(record['candidates'], formulas, step['t_sec'])
    values = distribution(with_send(deepcopy(values), record['input_sec'], record['ledger']), policy['temperature'])
    record.setdefault('updates', []).append(dict(input_sec=step['t_sec'], observed_steps=len(formulas),
        values=values, conflict=conflict, elapsed_ms=(perf_counter()-began)*MS))


def run(policy: dict, native: object) -> list[dict]:
    manifest = json.loads((BASE/'video_80_g7_v3_2/events.json').read_text())
    if video_path(80) is None:
        raise FileNotFoundError('検証動画80の画像が必須です。共有先を同じWSLセッションで接続してください')
    targets = {r['fire']['t_sec']: r for r in manifest['origins']}
    ocr = json.loads((OUT/'ocr.json').read_text())
    score_stream, step_stream = iter(ocr['records']), iter(ocr['steps'])
    score, step = next(score_stream, None), next(step_stream, None)
    ledger = ScorePrefix(policy['first_move_quiet_sec'])
    images, recent, active, rows = ImageSources(video_path(80)), {}, {}, []
    for args in updates(source(80)/'inputs.jsonl.gz', {GAME}):
        stamp = args[3]
        while score and score['t'] <= stamp:
            ledger.observe(score['side'], score['t'], score['score'])
            score = next(score_stream, None)
        while step and step['t_sec'] <= stamp:
            recent[step['side']] = step
            sequential(active, step, policy)
            step = next(step_stream, None)
        if stamp not in targets:
            continue
        item = targets[stamp]
        side, trigger = item['side'], item['fire']['trigger_sec']
        raw = (args[0].p1, args[0].p2)[side]
        sources = extract_sources(raw, stamp, trigger, side, images)
        first = recent.get(side)
        first = formula_step(first, 1) if first and trigger-1/60 <= first['t_sec'] <= stamp else None
        row = predict(item, raw, sources, first, ledger.snapshot(side, stamp), policy, native)
        rows.append(row)
        if row['variants']:
            active[side] = row
        print('予測', item['event_id'], row['status'], row.get('low_spec_ms'), flush=True)
    images.cap.release()
    return rows


def main() -> None:
    import puyo_core
    policy = json.loads((OUT/'policy.json').read_text())
    assert policy['locked'] and policy['training_videos'] == list(VIDEOS) and policy['excluded_video'] == 80
    rows = run(policy, puyo_core)
    save(OUT/'predictions.json', dict(rows=rows, policy_sha256=sha256(OUT/'policy.json')))


if __name__ == '__main__':
    main()
