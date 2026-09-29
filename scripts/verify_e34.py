"""E34のOFF互換とON検収に必要な入力の不足を保存する。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.replay_exchange_event_20260926 import compare
from scripts.run_e34 import OUT, e32
from scripts.run_e3_exchange_eval_20260926 import save_json, SOURCES


def read(path: Path) -> Any:
    """保存済みの原票をUTF-8で読む。"""
    return json.loads(path.read_text(encoding='utf-8'))


def verify_off() -> dict:
    """全5入力の列・型・バイト・イベントをE32と厳密照合する。"""
    result = {}
    for source in e32.e31.prior.ALL_SOURCES:
        suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        baseline, actual = Path('logs/e32/on')/suffix, OUT/'off'/suffix
        value = compare(baseline, actual)
        with np.load(actual/'display.npz') as data:
            result[source] = dict(rows=len(data['t_sec']), columns=len(data.files), **value)
    return result


def main() -> None:
    """未計測のON値を基準値で埋めず、実測済みOFFと不足の証拠を分ける。"""
    checks = verify_off()
    preflight = read(OUT/'PREFLIGHT.json')
    failures = {p.stem: read(p) for p in (OUT/'records').glob('*.json')}
    baseline = read(Path('logs/e32/on/METRICS.json'))
    rows = []
    for source in SOURCES:
        rows.extend(read(Path('logs/e32/on/renders')/source/'on/snapshot_final_audit.json')['rows'])
    result = dict(state='input_blocked', passed=None, off=checks,
        baseline={k: baseline[k] for k in ('q', 'zenchi', 'deaths')},
        baseline_score=read(Path('logs/e33b/SCORES.json'))['off'],
        baseline_prefire=dict(fires=len(rows), accepted=sum(r['accepted'] for r in rows),
            origin_difference=sum(r['reason'] == 'origin_difference' for r in rows)),
        preflight=preflight, enrichment=failures, on=None)
    save_json(OUT/'VERIFICATION.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
