"""実入口用に必要な型/Session生成関数だけ読む。CPU run/main/connectは輸入しない。"""
from __future__ import annotations
from collections import ChainMap
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
PUB = VERIFY / 'g2_belief_live_publication_2026-09-11_v1'
RUNTIME = VERIFY / 'g2_belief_publication_runtime_2026-09-11_v1'
JOINT = VERIFY / 'g2_joint_collector_runtime_2026-09-12_v1'
LEDGER = VERIFY / 'g2_joint_ledger_engine_2026-09-12_v1'
ALIASES = ('_whole_accounting_upgrade', '_whole_start_anchor', '_whole_start_anchor_v2',
           '_g2_repaired_accounting_observer_v11', '_whole_parent_transport', '_whole_parent_validation',
           '_whole_parent_publication', '_whole_parent_client', '_whole_session_create_v6',
           '_whole_unused_cpu_boundary', '_whole_runtime_session', '_joint_parent_protocol_v2',
           '_joint_parent_transport_v2', '_joint_parent_event_reader_v2', '_whole_start_trace',
           '_whole_start_qualification', '_whole_start_capture', '_whole_parent_client_base')


def source_paths() -> tuple[Path, ...]:
    """追加経路の遅延importも実prepareの既存SHA票へ含める。"""
    roots = (PUB, JOINT, LEDGER, VERIFY / 'g2_model_process_bridge_2026-09-11_v1',
             VERIFY / 'g2_journal_writer_witness_repair_2026-09-11_v1')
    explicit = (RUNTIME / 'run_v6.py', RUNTIME / 'collector_session.py',
        VERIFY / 'g2_frozen_accounting_upgrade_2026-09-12_v1/upgrade.py',
        VERIFY / 'g2_observed_start_anchor_2026-09-12_v1/anchor.py',
        VERIFY / 'g2_producer_time_capture_2026-09-12_v1/capture.py',
        VERIFY / 'g2_persistent_ledger_process_2026-09-12_v1/protocol.py',
        VERIFY / 'g2_persistent_ledger_process_2026-09-12_v1/transport.py',
        VERIFY / 'g2_probabilistic_scope_candidate_2026-09-11_v1/joint_evaluation.py',
        ROOT.parents[2] / 'src/event_accounting_observer_v1.py',
        ROOT.parents[2] / 'src/event_source_v1.py')
    start = tuple(ROOT / name for name in ('start_trace.py', 'start_qualification.py', 'start_capture.py',
                                         'start_join.py', 'start_client.py', 'start_worker.py'))
    return tuple(sorted(set(explicit + start) | {path for root in roots for path in root.glob('*.py')}))


class Scope:
    """新依存だけを所有する。既存bootstrap/共有src moduleは所有しない。"""
    def __init__(self, stack: Any) -> None:
        if any(name in sys.modules for name in ALIASES):
            raise ValueError('whole_dependency_foreign_alias')
        self.owned: dict[str, Any] = {}
        self.closed = False
        self.started = False
        self.error: str | None = None
        stack.push(self.close)

    def prepare(self, load: Any, frames: tuple[int, ...]) -> tuple[Any, Any, Any]:
        if self.started or self.closed:
            raise ValueError('whole_dependency_reentry')
        self.started = True
        paths = list(sys.path)
        try:
            recorder, anchor = observer_and_anchor(load)
            return recorder, anchor, session_creator(load, frames)
        finally:
            self.owned = {name: sys.modules[name] for name in ALIASES if name in sys.modules}
            sys.path[:] = paths

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        try:
            changed = [name for name, value in self.owned.items() if sys.modules.get(name) is not value]
            for name, value in self.owned.items():
                if sys.modules.get(name) is value:
                    sys.modules.pop(name)
            if changed:
                raise ValueError('whole_dependency_alias_changed:' + ','.join(changed))
        except BaseException as error:
            self.error = repr(error)
            if body is None:
                raise
        finally:
            self.closed = True
        return False


def observer_and_anchor(load: Any) -> tuple[Any, Any]:
    guard = sys.modules['continuous_guard']  # 実live tail_authが既に使う同一guard。
    upgrade = load('_whole_accounting_upgrade', VERIFY / 'g2_frozen_accounting_upgrade_2026-09-12_v1/upgrade.py',
                   {'continuous_guard': guard})
    anchor = load('_whole_start_anchor', VERIFY / 'g2_observed_start_anchor_2026-09-12_v1/anchor.py')
    corrected = load('_whole_start_anchor_v2', JOINT / 'anchor_v2.py', {'anchor': anchor})
    trace = load('_whole_start_trace', ROOT / 'start_trace.py')
    qualification = load('_whole_start_qualification', ROOT / 'start_qualification.py')
    captured = load('_whole_start_capture', ROOT / 'start_capture.py',
                    {'anchor_v2': corrected, 'start_trace': trace, 'start_qualification': qualification})
    return upgrade.modern(), captured


def session_creator(load: Any, frames: tuple[int, ...]) -> Any:
    """CPU入口の起動副作用を避け、原createへ同じ依存pathと明示評価frameを与える。"""
    lazy: dict[str, Any] = {'parent_validation': None}
    transport = load('_whole_parent_transport', LEDGER / 'parent_transport.py', lazy)
    validation = load('_whole_parent_validation', LEDGER / 'parent_validation.py', {'parent_transport': transport})
    lazy['parent_validation'] = validation
    publication = load('_whole_parent_publication', LEDGER / 'parent_publication.py',
                       {'parent_transport': transport, 'parent_validation': validation})
    base_client = load('_whole_parent_client_base', JOINT / 'parent_client_v2.py', {'parent_transport': transport})
    client = load('_whole_parent_client', ROOT / 'start_client.py', {'parent_client_v2': base_client})
    prior = N(OLD=N(PUB=PUB, FRAMES=frames),
              V4=N(BRIDGE=VERIFY / 'g2_model_process_bridge_2026-09-11_v1'),
              WRITER=VERIFY / 'g2_journal_writer_witness_repair_2026-09-11_v1')
    sixth = load('_whole_session_create_v6', RUNTIME / 'run_v6.py', {'run_v5': prior})
    # この境界型は原runtime_sessionのimportだけを満たす。startは使わない。
    boundary = load('_whole_unused_cpu_boundary', RUNTIME / 'collector_session.py')
    runtime = load('_whole_runtime_session', JOINT / 'runtime_session.py',
                   {'parent_client_v2': client, 'parent_publication': publication,
                    'run_v6': sixth, 'collector_session': boundary})
    proxy = N(modules=ChainMap({'inflight_loader': N(load=load)}, sys.modules))
    create = FunctionType(runtime.create.__code__, dict(vars(runtime), sys=proxy))
    # parent resultのactual_video=Falseは未認定モデル候補。実動画GOの根拠にはしない。
    return create
