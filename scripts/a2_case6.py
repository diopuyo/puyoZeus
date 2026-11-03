"""#6の連鎖ID・段照合を既存診断と候補原票から抽出する。"""
from __future__ import annotations

import json
from pathlib import Path
from scripts.a2_replay import OUT
from scripts._diag_set2_falsedeath_audit import events, read


def first_proof(values: list[dict]) -> dict | None:
    """末尾の鮮度切れだけでなく、最初に通った実際の死亡証明も残す。"""
    hit = next((v for v in values if v['source'] == 'unavoidable_death'), None)
    if hit is None:
        return None
    return {k: hit.get(k) for k in ('t_sec', 'p1', 'dead_sides', 'held_from_sec', 'post_counter_bound')}


def main() -> None:
    """予測の一致を探さず、原票にあるIDと直近段の比較をそのまま保存する。"""
    base = OUT/'replay/a2/s1'
    chains, values = [], []
    for event in events(base/'events.jsonl'):
        for chain in event['chains']:
            if 4258 <= chain['trigger_sec'] <= 4275:
                chains.append(chain)
        values += [v for v in event['values'] if 4271 <= v['t_sec'] <= 4273
                   and v['source'] in ('unavoidable_death', 'S3_landing')]
    audit = read(base/'post_counter_bound_audit.json')['rows']
    checks = [r for r in audit if 4271 <= r['t_sec'] <= 4273]
    prefire = [r for r in read(base/'prefire_audit.json')['rows'] if 4258 <= r['trigger_sec'] <= 4275]
    result = dict(chains=chains, values=values, checks=checks, prefire=prefire)
    (OUT/'CASE6.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    summary = dict(first_computed_death=first_proof(values),
        chains=[{k: r.get(k) for k in ('chain_id', 'side', 'trigger_sec',
        'predicted_chain_count', 'predicted_final_score')} for r in chains],
        prefire=[{k: r.get(k) for k in ('chain_id', 'trigger_sec', 'reason', 'accepted', 'withdrawn')}
                 for r in prefire],
        checks=[dict(t=r['t_sec'], reason=r.get('reason'), check=r.get('completion_prefix_check'))
                for r in checks[-2:]])
    (OUT/'CASE6_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
