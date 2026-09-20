"""原side入口から基準cascade一回まで延長する。原更新/原J/全archive検査を維持。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
RETIRE = ROOT.parent / 'g2_side_occurrence_retirement_2026-09-11_v1'
HINIT = ROOT.parent / 'g2_hidden_basis_initialization_2026-09-11_v1'
ARCH = ROOT.parent / 'g2_archive_close_boundary_2026-09-11_v1'


def configure() -> Any:
    sys.path[:0] = [str(ARCH), str(HINIT), str(RETIRE)]
    import proof_cpu_finalized_v2 as F
    import retirement as R
    import side_actual_connection as S
    import cascade_inputs as I
    import cascade_loader as L
    import cascade_verify as V
    import cascade_finish as FIN
    import cascade_constructor as CTOR
    import baseline_retirement as D
    hidden, target = F.OLD.BASE.H, F.OLD.BASE.H.OLD
    original_modules = hidden.H.modules
    hidden.H.modules = L.wrap(original_modules, hidden.H.OLD.L.load)
    hidden.Context = D.derived(R.derived(S.derived(hidden.Context)))
    hidden.verify = V.verify
    F.OLD.BASE.install = I.install(F.OLD.BASE.install)
    extension = target.DRIVER.P.R.R.X
    extension.POST = tuple(range(extension.RESET, extension.RESET + I.POST_COUNT * I.STRIDE, I.STRIDE))
    original_derived = target.TARGET.derived
    target.TARGET.derived = lambda: I.extend(original_derived(), extension.RESET)
    continuation = target.DRIVER.P.R.R.R
    original_checked = continuation.OLD_CHECKED
    continuation.OLD_CHECKED = lambda selected, base: CTOR.wrap(original_checked(selected, base), FIN.EVIDENCE)
    F.F = FIN
    return F


def main() -> int:
    import predecessor_go as G
    approval = G.verify()
    started = time.monotonic()
    cascade = ROOT.parent / 'g2_basis_cascade_candidate_2026-09-11_v1'
    sources = [p for directory in (ROOT, RETIRE, cascade) for p in directory.glob('*.py')]
    sources.append(ROOT.parent / 'g2_chain_firing_continuous_cpu_2026-09-10_v1/constructor_input.py')
    sources.append(ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v36/PROBABILISTIC_BASIS.jsonl')
    def digest() -> Any:
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    before = digest()
    F = configure()
    import cascade_finish as FIN
    if '--preflight' in sys.argv:
        generated = F.OLD.BASE.H.OLD.TARGET.derived()
        assert callable(generated)
        print('generated_cascade_entry_ready; actual_update_not_run')
        return 0
    probe = '--constructor-preflight' in sys.argv
    FIN.EVIDENCE['constructor_only_probe'] = probe
    code = F.main()
    evidence = FIN.EVIDENCE
    if code == 0:
        assert evidence['constructor_restored'] and evidence['tracker_constructor_restored']
        assert len(evidence['calls']) == evidence['tracker_constructor_calls'] == 1
    output = F.OLD.BASE.H.OLD.DRIVER.P.R.Q.ROOT / sys.argv[1]
    with (output / 'CASCADE_SOURCE.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(original_exit=code, unchanged=before == digest(), source=before,
            constructor=evidence, predecessor=approval, seconds=time.monotonic()-started,
            quality_gate_clear=False), stream, indent=2)
    if probe:
        assert code == 1 and evidence['actual_constructor_seen']['tracker_class_verified']
        assert evidence['constructor_restored'] and evidence['tracker_constructor_restored']
        print('actual_constructor_preflight_only; requested_stop; G2_OPEN')
        return int(before != digest())
    return code or int(before != digest())


if __name__ == '__main__':
    raise SystemExit(main())
