"""原full collectの開始/遅延対照を保存し、源と復元を検収する。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import test_anchor as T

ROOT = Path(__file__).resolve().parent


def sources() -> dict[str, str]:
    paths = [ROOT / n for n in ('anchor.py', 'test_anchor.py', 'save_proof.py')]
    paths += [T.A.BASE, Path(T.T.U.__file__), Path(T.T.U.U.__file__),
              T.T.U.U.SNAPSHOT / 'scripts/collect_boards_lean.py',
              T.T.U.U.SNAPSHOT / 'src/board.py', T.T.L.SOURCE]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main() -> None:
    start, before = time.monotonic(), sources()
    real = T.real.__wrapped__()
    results = []
    for mode in ('good', 'delayed'):
        output = ROOT / (mode + '_v1')
        saved = T.run(real, output, mode)
        anchor = saved['start_anchor']
        expected = T.QUALIFIED if mode == 'good' else 94
        assert anchor['qualification_frame'] == expected
        assert all(not d['eligible'] for d in saved['start_decisions'] if d['frame'] < expected)
        results.append(dict(mode=mode, path=str(output / 'START_CAPTURE.json'),
                            boundary_frame=anchor['candidate']['boundary_available_frame'],
                            qualified_frame=anchor['qualification_frame'],
                            observed_count=saved['observed_count'], stable_rows=len(saved['stable_snapshots'])))
    assert before == sources()
    value = dict(results=results, source_unchanged=True, sources=before, seconds=time.monotonic() - start,
                 actual_video=False, quality_gate_clear=False, saved_and_restored_verified=True)
    with (ROOT / 'SAVED_START_PROOF_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    print('原開始90/遅延94・遡及なし・同frame2側保存・源/復元: PASS限定')


if __name__ == '__main__':
    main()
