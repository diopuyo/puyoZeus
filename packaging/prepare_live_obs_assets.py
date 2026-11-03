"""既存成果を上書きせず、本人用試験の固定資産をSHA確認して配置する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path(_public_dev_text('<work>/build_20261020a_stage'))


def copy_checked(source: Path, target: Path, expected: str | None = None) -> dict:
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None:
        assert digest == expected, (source, digest)
    if target.exists():
        assert target.read_bytes() == data, target
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    return dict(source=str(source), target=str(target), sha256=digest, bytes=len(data))


def main() -> None:
    shared = Path('D:/puyo_analyzer/share/tasks/done/dj4f_b3_finalize_only/M3/final_large')
    jobs = [(shared/'large_seed1003.joblib', ROOT/'models/b3_m3/model.joblib',
             '11f7b32a26a9b70dbfda6544d64bc83695a6218531deaf25204d518704d26788'),
            (shared/'b3_ledger_experiment.json', ROOT/'models/b3_m3/b3_ledger_experiment.json', None),
            (Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive/app/models/b3/model.joblib')), ROOT/'models/v49f/b3_v6.joblib',
             '6113e42b8d3aadad6251f0749ad09281d7c63a65fc09a4cc659e68694ee2903c'),
            (STAGE/'ama_mingw_r3/ama_reference.exe', ROOT/'models/ama_plain/ama_reference.exe', None),
            (Path(_public_dev_text('<work>/wt_ama_plain/docs/verification/ama_plain/model.json')), ROOT/'models/ama_plain/model.json',
             '39abc1654dc417dba6d90fa30816d26656d8a5835cf9f48deed6d14fc14f9860'),
            (Path(_public_dev_text('<work>/wt_ama_eval/logs/ama_eval_v5/seed_table_cache.npz')), ROOT/'models/ama_plain/seed_table_cache.npz', None)]
    old = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive/app/models/v49f'))
    jobs.extend((old/name, ROOT/'models/v49f'/name, None) for name in ('policy.json','pre_model.json','model.json'))
    receipts = [copy_checked(*job) for job in jobs]
    output = ROOT/'docs/verification/live_obs_20261020/assets.json'
    output.write_text(json.dumps(receipts, indent=2), encoding='utf-8')
    record = Path('D:/puyo_analyzer/b3_eval_records/attempts/b3_zenchi_s1/1791168806059117750/record')
    records = [copy_checked(record/name, STAGE/'m3_input'/name) for name in ('observations.jsonl.gz', 'events.jsonl', 'lean.npz')]
    (STAGE/'m3_input/receipt.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
