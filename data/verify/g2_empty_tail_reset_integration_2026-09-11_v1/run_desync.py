"""観測信号→原資格→原reset→新25更新を、原76と同じcollectorへ接続する。"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
import run_qualification as Q

R = Q.module('_desync_continuation_entry', Q.ROOT/'run_continuation.py')

TRIGGER = Path(__file__).resolve().parent.parent/'g2_empty_tail_desync_trigger_2026-09-11_v1'
sys.path.insert(0, str(TRIGGER))
import extension as X


def main() -> int:
    files = sorted(TRIGGER.glob('*.py'))
    pins = lambda: {str(p): sha256(p.read_bytes()).hexdigest() for p in files}
    before = pins()
    # R.inputsのclosureの期待全Jへ観測4件も含める。
    R.I = N(POST=X.FRAMES+X.POST)
    R.C = N(extend=X.extend)
    R.Q.wrap_loader, R.Q.checked_loader = R.wrap_loader, R.checked_loader
    code = FunctionType(R.Q.F.main.__code__, dict(vars(R.Q.F), ROOT=R.Q.ROOT, execute=R.Q.execute))()
    result = dict(source_before=before, source_unchanged=pins()==before,
        actual_exit_code=code, quality_gate_clear=False)
    with (R.Q.ROOT/sys.argv[1]/'DESYNC_INPUTS.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return code or int(not result['source_unchanged'])


if __name__ == '__main__':
    raise SystemExit(main())
