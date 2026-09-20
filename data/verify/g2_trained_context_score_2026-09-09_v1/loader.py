"""固定video38の外側fold6だけを復元する私有CPU検収用loader。"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
VERIFY = PROJECT / 'data/verify'
PREFIX = 'advantage_m1_zero_counterfactual_46v_clean_2026-09-06_v1'
SOURCE_ID = 'sha256:b3728078cc2e8282065e5dd78ca1c13e6a443ffdf1dc4333e3da06f0852757d3'
SEEDS = (20260904, 20260905, 20260906)
FOLD = 6
OWN = ('loader.py', 'test_loader.py', 'run_loader_cpu.py')
SPECS = {
    20260904: ('folds5_6_seed20260904', 'allfolds_seed20260904_merged',
        'bf0c77da92ed12c5019a0a24105a82496bb7ff1bbebc81ec282fc0028f53ab4f',
        'f44ea22232bc77d693f121f0a0b8f88a5a3feab34ef96d1fd0123e3151feb061',
        'ad95252c55ca8e105d9761f5cc135ddc511a7ce9a6331a9cdaf2dd02988b8c6a', 0.4085864275522194),
    20260905: ('folds5_6_seeds20260905_20260906', 'allfolds_seeds20260905_20260906_merged',
        '3c5d762c486bf3d43fb921c1d5cc0d02a0e3810f8449cef04973c896d177dc1d',
        '49e1f1035494174f3d7e97b54fbf1e2b940f75e9a7d435e70312c99f02dadc8b',
        '1bbf79c482e0d3c99f853beb74e3a82afc20e37986d22d5dc513a66ec482629c', 0.3250513247405564),
    20260906: ('folds5_6_seeds20260905_20260906', 'allfolds_seeds20260905_20260906_merged',
        '1b57c064d9b721645b2840b3bb7a5cba6ad898a0b9a605e8abf5162de2fa4d0e',
        'f40b59d4efe09b9854bd774fff49f7e2966e23070df689245a14481064101465',
        'c4737e7cde52f28e4563b25d0ac96cee406f8691dba88b5fed8fa05de1c6836a', 0.34384472111301934),
}
ROOT_HASHES = {
    'allfolds_seed20260904_merged': ('2928fe7bc6da8e1461f63339a2af7aef5543cb7282dfce517696631bea2a7ada',
        '12eb31fc56f1e3b8ada5575b47cf48c37765355c7a41b4a42acdddc1b7f12200',
        'ee095a8bf713b4fe1917fce715e25e01dcac4890657726596be2fdfd2183245a'),
    'allfolds_seeds20260905_20260906_merged': ('f54db1d9311cd4dbaa7286dc7337ad3f4480cf676917e7a4f490e2b5b488eacb',
        '4634d78d228c16d5ca7f550b426210f8e986441ae2b8548a015fb472f310c04b',
        '6adc8dc67c43f41637db878d8f3e2da5dde526d3bf0639901f47f2d53a58cb6f'),
}
CODE = {
    'scripts/train_advantage_m1_fixed_m0_anchor_v1.py': '855b15f902632259c9fd923827bcc00283679db38e57ba438e7435fde5b52100',
    'scripts/advantage_m1_outer_fold_baseline_contract_v1.py': 'bb7e170d1421fa5fa13a75e7ebab1082852231776ccc3fb9cdbf568520b6279f',
    'src/advantage_m0_current_cnn_v1.py': '84af708bfc348570b651834aa2e3f51f96e83551ccd11285c9f3b530347c1554',
    'src/advantage_m1_zero_counterfactual_v3.py': 'fae55575e58796d99b35c519445855bf6839d0c21a6c707e2c5bd039deaf0728',
}


@dataclass(frozen=True, slots=True)
class TrainedMember:
    """公開属性のSHAは自己申告として再検証する。model自体はmutable。"""
    seed: int
    fold: int
    slope: float
    model: Any
    checkpoint_sha256: str
    m0_state_sha256: str
    model_state_sha256: str


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def guards() -> dict[str, str]:
    """親担当sourceをown guardへ混ぜない。"""
    return {str(ROOT / name): sha(ROOT / name) for name in OWN}


def backend() -> Any:
    """runtime factoryへ設置する部品ではないが、importも必要時に限定する。"""
    for name, expected in CODE.items():
        require(sha(PROJECT / name) == expected, 'loader_source_changed:' + name)
    from scripts import train_advantage_m1_fixed_m0_anchor_v1 as fixed
    from scripts import advantage_m1_outer_fold_baseline_contract_v1 as outer
    require(Path(fixed.__file__).resolve() == PROJECT / next(iter(CODE)), 'loader_module_path')
    return SimpleNamespace(fixed=fixed, outer=outer, torch=fixed.torch)


def input_guards() -> dict[str, str]:
    paths = {PROJECT / name for name in CODE}
    for spec in SPECS.values():
        paths.update(VERIFY / (PREFIX + '_' + spec[1]) / n for n in ('COMPLETE', 'PLAN.json', 'results.json'))
    for seed in SEEDS:
        paths.update(_paths(seed))
    return {str(path): sha(path) for path in sorted(paths)}


def _paths(seed: int) -> tuple[Path, Path]:
    component = VERIFY / (PREFIX + '_' + SPECS[seed][0])
    return (component / f'm0__seed_{seed}__fold_{FOLD}.pt',
            component / f'm1_zero_values_and_masks__seed_{seed}__fold_{FOLD}.pt')


def _root(api: Any, seed: int) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    suffix = SPECS[seed][1]
    root = VERIFY / (PREFIX + '_' + suffix)
    for name, digest in zip(('COMPLETE', 'PLAN.json', 'results.json'), ROOT_HASHES[suffix], strict=True):
        require(sha(root / name) == digest, 'training_root_changed:' + name)
    complete, plan, report = [api.fixed._read_json(root / n) for n in ('COMPLETE', 'PLAN.json', 'results.json')]
    api.fixed._validate_anchor_plan(plan, report)
    require(complete['format_version'] == api.fixed.merger.MERGE_COMPLETE_VERSION, 'complete_format')
    require(api.fixed._anchor_dataset_receipt(plan) == api.fixed.EXPECTED_ANCHOR_DATASET_RECEIPT, 'dataset_identity')
    return root, complete, report['results']


def _record(api: Any, results: dict[str, Any], seed: int, kind: str) -> dict[str, Any]:
    records = results[f'{kind}__seed_{seed}']['records']
    matches = [row for row in records if type(row.get('eval_fold')) is int and row['eval_fold'] == FOLD]
    require(len(matches) == 1, 'fold_record_count')
    row = matches[0]
    api.outer.validate_outer_fold_record(row, FOLD)
    index = 0 if kind == 'm0' else 1
    path, expected = _paths(seed)[index], SPECS[seed][2 + index]
    ref = row['checkpoint_reference']
    require(Path(ref['path']).resolve() == path and Path(ref['component_root']).resolve() == path.parent,
            'checkpoint_path')
    require(row['model'] == path.name and row['model_sha256'] == ref['sha256'] == sha(path) == expected,
            'checkpoint_sha')
    return row


def _records(api: Any, seed: int) -> tuple[Any, dict[str, Any]]:
    root, complete, results = _root(api, seed)
    base, record = [_record(api, results, seed, name) for name in ('m0', 'm1_zero_values_and_masks')]
    reference = api.fixed._reference_from_record(base, seed, root, complete)
    spec = SPECS[seed]
    require(reference.state_sha256 == spec[4] and reference.slope == spec[5], 'm0_reference')
    require(record['m0_model_sha256'] == spec[2] and record['m0_model'] == reference.path.name,
            'm0_link')
    require(record['m0_checkpoint_reference'] == base['checkpoint_reference'], 'm0_reference_link')
    require(record['m0_state_sha256_before'] == record['m0_state_sha256_after'] == spec[4], 'm0_state_link')
    require(type(record['fixed_m0_symmetric_platt_slope']) is float
            and record['fixed_m0_symmetric_platt_slope'] == spec[5]
            and record['fallback_to_m0'] is False, 'record_policy')
    return reference, record


def _payload(api: Any, seed: int) -> dict[str, Any]:
    path = _paths(seed)[1]
    require(sha(path) == SPECS[seed][3], 'checkpoint_changed')
    payload = api.torch.load(path, map_location='cpu', weights_only=True)
    _validate_payload(api, payload, seed)
    return payload


def _validate_payload(api: Any, payload: Any, seed: int) -> None:
    require(type(payload) is dict, 'checkpoint_mapping')
    require(type(payload.get('seed')) is int and payload['seed'] == seed
            and type(payload.get('fold')) is int and payload['fold'] == FOLD, 'checkpoint_seed_fold')
    require(payload.get('variant') == 'values_and_masks' and payload.get('not_production') is True
            and payload.get('fallback_to_m0') is False, 'checkpoint_policy')
    require(payload.get('model_version') == api.fixed.v3.ZERO_COUNTERFACTUAL_MODEL_VERSION
            and payload.get('input_schema_version') == api.fixed.v3.M1_INPUT_SCHEMA_VERSION, 'checkpoint_schema')
    require(type(payload.get('fixed_m0_symmetric_platt_slope')) is float
            and payload['fixed_m0_symmetric_platt_slope'] == SPECS[seed][5]
            and payload.get('m0_state_sha256') == SPECS[seed][4], 'checkpoint_m0_link')
    require(isinstance(payload.get('state_dict'), dict), 'checkpoint_state')


def _state_digest(api: Any, state: dict[str, Any]) -> str:
    """既存のtensor名/型/shape/値hashをpayloadへも同じまま使う。"""
    return api.fixed.frozen.model_state_sha256(SimpleNamespace(state_dict=lambda: state))


def _load_one(api: Any, seed: int, device: str) -> TrainedMember:
    reference, _ = _records(api, seed)
    anchor = api.fixed.load_m0_anchor(reference, api.torch.device(device))
    payload = _payload(api, seed)
    model = api.fixed.v3.AdvantageM1ZeroCounterfactualV3(anchor.model, 'values_and_masks').to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    expected = _state_digest(api, payload['state_dict'])
    require(api.fixed.frozen.model_state_sha256(model) == expected, 'loaded_state')
    require(api.fixed.frozen.model_state_sha256(model.m0) == reference.state_sha256, 'loaded_m0_link')
    return TrainedMember(seed, FOLD, reference.slope, model, SPECS[seed][3], reference.state_sha256, expected)


def _runtime_model(api: Any, member: TrainedMember, device: str) -> None:
    model = member.model
    require(type(model) is api.fixed.v3.AdvantageM1ZeroCounterfactualV3
            and type(model.m0) is api.fixed.AdvantageM0CurrentCNNV2, 'model_type')
    require(model.variant == 'values_and_masks', 'model_variant')
    require(all(not sub.training for sub in model.modules()), 'model_eval')
    require(all(not parameter.requires_grad for parameter in model.parameters()), 'model_grad')
    require(all(value.device == api.torch.device(device) for value in
                (*model.parameters(), *model.buffers())), 'model_device')
    for sub in model.modules():
        require(not any(name in vars(sub) for name in ('forward', 'infer', 'state_dict')), 'model_method_override')
        require(not sub._forward_hooks and not sub._forward_pre_hooks, 'model_hooks')
    # 無重みactivation差替えも拒否し、検証用構築によるCPU乱数の進行は戻す。
    with api.torch.random.fork_rng(devices=[]):
        reference = api.fixed.v3.AdvantageM1ZeroCounterfactualV3(
            api.fixed.AdvantageM0CurrentCNNV2(), 'values_and_masks')
    require(_layout(api, model) == _layout(api, reference), 'model_structure')


def _layout(api: Any, model: Any) -> tuple[Any, ...]:
    """固定nn.Moduleの型/非tensor設定を検査し、未登録callableも拒否する。"""
    internal = set(vars(api.torch.nn.Module()))
    rows = []
    for name, sub in model.named_modules():
        attrs = {key: value for key, value in vars(sub).items() if key not in internal}
        require(all(type(value) in (str, int, float, bool, type(None), list, tuple)
                    for value in attrs.values()), 'model_instance_attribute')
        require(all(not value for key, value in vars(sub).items() if key.endswith('_hooks')), 'model_hooks')
        rows.append((name, type(sub), repr(sorted(attrs.items()))))
    return tuple(rows)


def verify_members(source_id: str, members: tuple[TrainedMember, ...], device: str = 'cpu') -> None:
    """callerのbool/SHA自己申告を信じず、固定原本と現stateを再照合する。"""
    require(type(source_id) is str and source_id == SOURCE_ID, 'source_id')
    require(type(device) is str and device == 'cpu', 'cpu_only')
    require(type(members) is tuple and len(members) == len(SEEDS), 'members_count')
    api = backend()
    for seed, member in zip(SEEDS, members, strict=True):
        require(type(member) is TrainedMember and type(member.seed) is int and member.seed == seed,
                'member_seed_type')
        require(type(member.fold) is int and member.fold == FOLD and type(member.slope) is float
                and member.slope == SPECS[seed][5], 'member_fold_slope')
        _records(api, seed)
        expected = _state_digest(api, _payload(api, seed)['state_dict'])
        require(member.checkpoint_sha256 == SPECS[seed][3] and member.m0_state_sha256 == SPECS[seed][4]
                and member.model_state_sha256 == expected, 'member_receipt')
        _runtime_model(api, member, device)
        require(api.fixed.frozen.model_state_sha256(member.model) == expected, 'model_state_changed')
        require(api.fixed.frozen.model_state_sha256(member.model.m0) == SPECS[seed][4], 'model_m0_changed')


def load_members(source_id: str, device: str = 'cpu') -> tuple[TrainedMember, ...]:
    """現単位はCPUのみ。学習/forward/inferを呼ばず、全stateを復元する。"""
    require(type(source_id) is str and source_id == SOURCE_ID, 'source_id')
    require(type(device) is str and device == 'cpu', 'cpu_only')
    api = backend()
    members = tuple(_load_one(api, seed, device) for seed in SEEDS)
    verify_members(source_id, members, device)
    return members
