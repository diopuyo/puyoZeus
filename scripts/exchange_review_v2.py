"""動画80・第7試合だけの勝率加算試作。既存配信処理には接続しない。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import csv
import json
from pathlib import Path
from scripts.exchange_review_io import jsonl, save
from scripts.exchange_review_timeline import frame_rows
from scripts.exchange_review_v2_fit import fit, value
from scripts.exchange_review_v2_budget import compute, BUDGET_MS, BACK_RATIO, CLOCK_SCALE
from scripts.exchange_review_engine import leaf_value

BASE = Path(_public_dev_text('<work>/exch_review_render'))
SOURCE, OUTPUT = BASE / 'video_80_g7', BASE / 'video_80_g7_v2'
# 追補4(c)の通常/低スペックP50・P95比の大きい方を全呼出へ適用。
FAST_RATIO = max(22.8 / 14.0, 418.7 / 260.2)
POLICIES = {'main': '本線', 'counter': '対応', 'quick_counter': '対応',
            'provocation': '催促'}
POLICY_ORDER = {'main': 0, 'counter': 1, 'quick_counter': 2}


def recommendation(event: dict, recursive: bool) -> dict:
    fg = event['foreground']
    if recursive:
        candidates = [(n, leaf_value(n)) for n in event['background']['tree']['own']]
        candidates = [(n, v) for n, v in candidates if v is not None]
        if candidates:
            node, balance = min(candidates, key=lambda p: (-p[1], POLICY_ORDER[p[0]['policy']],
                                 p[0]['deadline_hands'], p[0]['target_score']))
            amount = node['outcome']['cancelled'] + node['outcome']['excess']
            k = node['available_hands']
            return dict(balance=balance, amount=amount, policy=POLICIES[node['policy']], route='再帰', k=k,
                        truncated=bool(event['background']['tree'].get('budget_exhausted') or node['truncated']))
    quick = fg['quick']
    options = quick.get('assumptions', [quick])
    # 不明時は保守的な仮定。quickの本線判断と実際に選んだ経路の量を保持する。
    q = min(options, key=lambda r: r['cancelled'] + r['excess'] - r['fallen'] - r['pending'])
    chosen = q if q.get('coverage') else fg['f']
    if not chosen.get('coverage'):
        return {}
    amount = chosen['cancelled'] + chosen['excess']
    k = fg['k']
    return dict(balance=amount - fg['incoming'], amount=amount,
        policy=POLICIES.get(q.get('selected_policy'), '対応'), route='毎コマ経路', k=k,
        truncated=recursive)


def prepare() -> tuple[dict, dict, list[dict], list[dict]]:
    import puyo_core
    model_path = OUTPUT / 'model.json'
    model = json.loads(model_path.read_text()) if model_path.exists() else fit(OUTPUT)
    model = {k: v for k, v in model.items() if k not in
             ('discount', 'n_discount', 'k_offset', 'n_k', 'correction_basis')}
    save(model_path, model)
    assert 80 not in model['videos'] and model['excluded_video'] == 80
    manifest = json.loads((SOURCE / 'events.json').read_text())
    run = Path(manifest['match']['run'])
    shadow = [r for r in jsonl(run / 'out/spool/exchange_projection_shadow.jsonl')
              if r.get('game_idx') == 7 or r.get('kind') == 'outcome']
    ids = {e['event_id'] for e in manifest['events']}
    ends = {r['event_id']: r['t_sec'] for r in shadow
            if r.get('kind') == 'outcome' and r['event_id'] in ids}
    for event in manifest['events']:
        assert event['seed'].get('input_sec') is None or event['seed']['input_sec'] <= event['input_sec']
        if 'foreground' not in event:
            continue
        event['background'] = compute(event, puyo_core)
        event['fast_ready_sec'] = event['input_sec'] + event['foreground']['compute_ms'] * FAST_RATIO / 1000
        event['background_ready_sec'] = max(event['fast_ready_sec'] + 1 / manifest['fps'],
            event['input_sec'] + event['background']['compute_ms'] * BACK_RATIO / 1000)
        assert event['background_ready_sec'] - event['input_sec'] <= BUDGET_MS / 1000
        event['close_sec'] = ends.get(event['event_id'], manifest['match']['end_sec'])
        event['recommendations'] = {route: recommendation(event, route == 'back') for route in ('fast', 'back')}
        print(f"再帰 {event['event_id']}: {event['background']['low_spec_ms']:.2f}ms / "
              f"打切り={event['recommendations']['back']['truncated']}", flush=True)
    manifest['v2_model'] = model
    manifest['budget_ms'] = BUDGET_MS
    manifest['latency'] = dict(fast_ratio=FAST_RATIO, background_ratio=BACK_RATIO,
        budget_clock_scale=CLOCK_SCALE, method='追補4(c)の実測倍率を探索時計と公開遅延に適用、返却余裕50ms')
    save(OUTPUT / 'events.json', manifest)
    return model, manifest, shadow, manifest['events']


def make_rows(model: dict, manifest: dict, shadow: list[dict], events: list[dict]) -> list[dict]:
    pending = sorted([r for r in shadow if 'pending_p1' in r], key=lambda r: r['t_sec'])
    cursor, now, pending_stamp = 0, 0., None
    result = []
    for row in frame_rows(events, manifest['display'], manifest['match'], manifest['fps']):
        t = row['t_sec']
        while cursor < len(pending) and pending[cursor]['t_sec'] <= t:
            p = pending[cursor]
            now, pending_stamp = p['pending_p2'] - p['pending_p1'], p['t_sec']
            cursor += 1
        seen = [e for e in events if e['input_sec'] <= t]
        event = max(seen, key=lambda e: e['input_sec']) if seen else None
        usable = bool(event and 'foreground' in event and event['fast_ready_sec'] <= t < event['close_sec'])
        route = 'back' if usable and event['background_ready_sec'] <= t and event['background'].get('tree') else 'fast'
        row.update(now_balance=now, pending_input_sec=pending_stamp, selected_input_sec=None,
                   selected_ready_sec=None, selected_event_id=None, route='待機', receiver='—')
        choice = event['recommendations'][route] if usable else {}
        sign = 1 if usable and event['receiver'] == '1P' else -1
        post = sign * choice['balance'] if choice else now
        probability = row['b3_probability']
        delta = model['c'] * (value(post, model) - value(now, model)) if choice else 0.
        row.update(exchange_probability=None if probability is None else min(1., max(0., probability + delta)),
            post_balance=post, predicted_return=choice.get('amount'), policy=choice.get('policy', '未計算'),
            exchange_delta=delta, k=choice.get('k'), truncated=choice.get('truncated', False))
        if usable:
            row.update(selected_input_sec=event['input_sec'], selected_ready_sec=event[
                'background_ready_sec' if route == 'back' else 'fast_ready_sec'],
                selected_event_id=event['event_id'], route='再帰' if route == 'back' else '毎コマ経路',
                receiver=event['receiver'])
        result.append(row)
    return result


def write_rows(rows: list[dict]) -> None:
    with (OUTPUT / 'frames.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if v is None else json.dumps(v, ensure_ascii=False) for k, v in row.items()})


def main() -> None:
    model, manifest, shadow, events = prepare()
    rows = make_rows(model, manifest, shadow, events)
    write_rows(rows)
    from scripts.exchange_review_v2_render import render
    render(OUTPUT, manifest, rows, shadow)


if __name__ == '__main__':
    main()
