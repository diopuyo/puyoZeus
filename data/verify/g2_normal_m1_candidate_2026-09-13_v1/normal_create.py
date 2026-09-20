"""原親transport/実モデルclientを再用し、正常専用Sessionへ明示配線する。"""
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import original_run_v6 as ORIGINAL
import normal_dependencies as D
import normal_settings as K

ROOT = Path(__file__).resolve().parent
V4, OLD, WRITER = ORIGINAL.V4, ORIGINAL.OLD, ORIGINAL.WRITER


def inner_mode() -> type:
    base = D.mode.BASE
    D.binding.B.require(base is sys.modules['_g2_real_basis_cascade_mode_v2']
        and base.Mode.__mro__[1] is base.V1.Mode and base.B is D.binding.B, 'normal_inner_mode_type')
    return base.Mode


def dependencies_ready() -> None:
    native = sys.modules.get('_g2_tracking_native')
    policy = sys.modules.get('_g2_hidden_initialization_gate')
    D.binding.B.require(native is D.mode.CORE.V1.N and native.B is D.binding.B, 'normal_native_ready')
    source = ROOT.parent / 'g2_hidden_basis_initialization_2026-09-11_v1/hidden_basis_gate.py'
    D.binding.B.require(policy is not None and Path(policy.__file__).resolve() == source
                        and policy.initial_hidden.__globals__ is vars(policy), 'normal_policy_ready')
    inner_mode()


def saved_verifier(load: Any) -> Any:
    paths = list(sys.path)
    try:
        original = load('_normal_original_saved', ROOT.parent /
            'g2_joint_collector_runtime_2026-09-12_v1/verify_joint_saved.py',
            dict(parent_transport=sys.modules['_whole_parent_transport'],
                 parent_validation=sys.modules['_whole_parent_validation']))
        schedule = load('_normal_original_saved_schedule', ROOT.parent /
                        'g2_m1_completion_candidate_2026-09-12_v1/schedule_saved.py')
        selected = load('_normal_saved_verifier', ROOT / 'normal_saved.py',
                        dict(original_saved=original, original_schedule_saved=schedule, normal_settings=K))
        return selected.verify
    finally:
        sys.path[:] = paths


def create(stack: Any, context: Any) -> Any:
    dependencies_ready()
    # 親runtime_sessionがこの関数のsysへ輸送した実loadを参照する。
    load = sys.modules['inflight_loader'].load
    D.binding.B.require(callable(load), 'normal_loader_ready')
    ORIGINAL.transport(context)
    state = context['state']
    client = load('_g2_model_process_client', V4.BRIDGE / 'client.py')
    contract = load('_g2_model_process_contract', V4.BRIDGE / 'pure_contract.py')
    loader = load('_normal_publication_loader', ROOT / 'normal_loader.py',
                  dict(normal_dependencies=D, normal_settings=K))
    def supplied(alias: str, path: Any, injection: Any = None) -> Any:
        if Path(path).resolve() == OLD.PUB / 'trained_sidecar.py':
            return client
        if Path(path).resolve() == OLD.PUB / 'journal_witness.py':
            original = load('_g2_J_writer_contract_original', WRITER / 'writer_contract.py', injection)
            proof = load('_g2_J_writer_contract_v2', WRITER / 'writer_contract_v2.py',
                         injection | dict(writer_contract=original))
            return load('_g2_J_stream_witness', WRITER / 'stream_witness.py',
                        injection | dict(writer_contract=proof))
        return load(alias, path, injection)
    modules = loader.modules(state, supplied)
    D.binding.B.require(modules['trained_sidecar'] is client
                        and modules['journal_witness'] is sys.modules['_g2_J_stream_witness'], 'normal_actual_transport')
    config = client.Configuration(D.binding.S, D.binding.B.grid, contract.RowInputs)
    verify = saved_verifier(load)
    session = modules['normal_session'].Session(stack, context, sys.modules['_g2_hidden_initialization_gate'],
        N(Mode=inner_mode()), contract, config, K.EARLIEST)
    session.saved_verifier = verify
    return session
