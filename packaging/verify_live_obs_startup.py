"""三系統とv4.9f用v6のprocess所有・終了を検査する。映像smokeの前段。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main() -> None:
    os.environ.update(PUYO_B3_M3='1',PUYO_LIVE_AMA_PLAIN='1',PUYO_LIVE_V49F='1',
                      PUYO_B3_TIME_GATE='1',PUYO_B3_LEDGER_MODE='1',PUYO_B3_LEDGER_LANDING='1')
    from src.phase_j.live_b3 import LiveB3
    from src.phase_j.live_b3_time_gate import ACTIVE
    out = Path(sys.argv[1])
    out.mkdir(exist_ok=False)
    table = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive/app/models/b3/timing_table.json'))
    owner = LiveB3(ROOT/'models/b3_m3/model.joblib',table,out,async_enabled=True)
    workers = [owner.ama.worker.process,owner.v49f.worker.process,owner.async_engine.process,
        owner.async_engine.compute,owner.v49f_baseline.async_engine.process,owner.v49f_baseline.async_engine.compute]
    assert all(p.is_alive() for p in workers)
    assert owner.time_gate is not None and owner.v49f_baseline.time_gate is None
    owner.close()
    assert not any(p.is_alive() for p in workers)
    value = dict(passed=True,owned_workers=len(workers),pids=[p.pid for p in workers],alive_after_close=[])
    (out/'receipt.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(value,flush=True)


if __name__ == '__main__':
    main()
