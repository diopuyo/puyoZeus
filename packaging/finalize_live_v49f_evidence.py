"""原票の保管と、レビュー用の小さな検収票を作る。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('D:/puyo_analyzer/packaging/build_20261019a_release')
VERIFY = Path(_public_dev_text('<work>/build_20261019a_stage/verification_r2'))
RECEIPTS = ROOT/'docs/verification/live_v49f_obs_20261019'


def environment_summary() -> dict:
    guard = json.loads((VERIFY/'benchmark_guard.jsonl').read_text().splitlines()[0])
    smoke = json.loads((VERIFY/'smoke_result.json').read_text())
    records = [json.loads(s) for s in (OUT/'environment.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    selected = [r for r in records if guard['time'] <= datetime.fromisoformat(r['time']).timestamp()
                <= guard['time']+smoke['elapsed_sec']+5]
    cpu = [p['CurrentClockSpeed'] for r in selected for p in r['cpu']]
    obs = [p['PercentProcessorTime'] for r in selected for p in r['obs']]
    return dict(samples=len(selected), first=selected[0]['time'], last=selected[-1]['time'],
        free_gib_min=min(r['free_gib'] for r in selected), clock_mhz_min=min(cpu), clock_mhz_max=max(cpu),
        cpu=selected[0]['cpu'][0]['Name'], obs_cpu_one_core_percent_max=max(obs),
        obs_cpu_one_core_percent_mean=sum(obs)/len(obs), ama_seen=any(r['ama'] for r in selected),
        benchmark_guard=guard, heavy_samples=selected)


def main() -> None:
    RECEIPTS.mkdir(exist_ok=True)
    environment = environment_summary()
    (OUT/'environment_summary.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    archive = OUT/'PuyoLive_20261019a.zip'
    with archive.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = json.loads((OUT/'archive_receipt.json').read_text())
    assert sha == receipt['sha256']
    receipt['unchanged_after_smoke'] = True
    changes = json.loads((OUT/'diff_20261017b.json').read_text())
    receipt['change_categories'] = {kind: dict(pyc=sum(n.endswith('.pyc') for n in names),
        other=sum(not n.endswith('.pyc') for n in names)) for kind, names in changes.items()}
    (OUT/'final_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    for name in ('speed_summary.json', 'publication_parity.json', 'final_receipt.json', 'diff_20261017b.json'):
        shutil.copy2(OUT/name, RECEIPTS/name)
    brief = {k: v for k, v in environment.items() if k != 'heavy_samples'}
    (RECEIPTS/'environment_summary.json').write_text(json.dumps(brief, indent=2), encoding='utf-8')
    for pair in ('80_7', '51_4'):
        shutil.copy2(OUT/f'packaged_parity_{pair}/receipt.json', RECEIPTS/f'parity_{pair}.json')
    for name in ('smoke_result.json', 'dom_smoke.json'):
        shutil.copy2(VERIFY/name, RECEIPTS/name)
    shutil.copytree(VERIFY, OUT/'verified_smoke_r2')
    print(json.dumps(dict(environment=brief, archive=receipt)), flush=True)


if __name__ == '__main__':
    main()
