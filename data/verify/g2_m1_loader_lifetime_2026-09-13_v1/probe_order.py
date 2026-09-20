"""実creatorが保持するloadと後発修復loadの寿命差をCPUで確認する。"""
from __future__ import annotations
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'g2_m1_ram_runtime_2026-09-13_v15'
sys.path.insert(0, str(RUNTIME))
import owned_adapter as R


def main() -> None:
    """実createの保持参照を使う。Session生成/画像処理はまだ行わない。"""
    with ExitStack() as stack:
        R.configured(stack)
        import probe_native_merge
        owner = sys.modules[R.A.A.A.V4.OWNED_ALIAS]
        bootstrap = owner.bootstrap()
        early = bootstrap.load
        creator = R.A.A.A.V4.A.D.session_creator(early, (35370, 35410))
        captured = creator.__globals__['sys'].modules['inflight_loader'].load
        assert captured is early
        parts = owner.dependencies().modules()
        current = bootstrap.load
        assert current is not early
        names = ('journal_context', 'journal_witness', 'second_observation', 'second_basis',
                 'second_physical', 'second_tracking', 'live_binding_v3', 'trained_sidecar', 'serialization')
        path = ROOT.parent / 'g2_belief_live_publication_2026-09-11_v1/live_session.py'
        value = captured('_g2_pub_runtime_live_session', path, {name: N() for name in names})
        before = value.Session
        assert Path(before.completed.__code__.co_filename).resolve() == path
        selected = current('_g2_pub_runtime_live_session', path, {name: N() for name in names})
        after = selected.Session
        assert Path(after.completed.__code__.co_filename).name == 'scheduled_session.py'
        assert parts.mode.Q.verify.__code__.co_filename.endswith('stable_capture_v2.py')
        result = dict(actual_creator_retains_early_load=True, later_load_replaced=True,
            captured_load_session=str(Path(before.completed.__code__.co_filename)),
            current_load_session=str(Path(after.completed.__code__.co_filename)),
            session_dependencies_stubbed=True, actual_session_created=False, actual_video=False,
            quality_gate_clear=False)
    assert value.Session is before
    result['class_restored'] = True
    with (ROOT / 'ORDER_REPRO_v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
