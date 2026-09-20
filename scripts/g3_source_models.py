"""G3の6動画を既存heldoutモデルへ結ぶ。学習・推論・本番登録はしない。"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterator
import uuid

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / 'data/verify'
LOADER = VERIFY / 'g2_trained_context_score_2026-09-09_v1/loader.py'
LOADER_SHA = '2fd2f8cefc9d0bd513480e884b05164e232b52862b7552a9b22a52dfd8670d0f'
PREREG = ROOT / 'docs/agent_coordination/G3_VIDEO_VALIDATION_PREREG_2026-09-07.json'
PREREG_SHA = 'ba08076933601a21ad4457ccb4f73375b9f454ef12bc96851aa2f566f17aba0a'
DATASET = VERIFY / 'advantage_m1_canonical_dataset_46v_2026-09-06_v4_quarantine_clean'
MANIFEST_SHA = '75e9e726e01b77ea932eecbfff8154536aa5f3b53cd42402e9b8448579f7405d'
COMPLETE_SHA = '68775da288277b543d44c968f0bb641cd2de8edf645232f7b3d79aeb618e9826'
SOURCE_FOLDS = dict(video_38=6, video_39=1, video_c74=3,
                    video_c50=3, video_c80=4, video_c138=1)


def require(ok: bool, reason: str) -> None:
    """不明なsource/来歴を既定foldへ落とさない。"""
    if not ok:
        raise ValueError('g3_source_models:' + reason)


def digest(path: Path) -> str:
    """大きいcheckpointも一括メモリ展開せず照合する。"""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def document(path: Path, expected: str) -> dict[str, Any]:
    """読んだbyte列を認証してから解釈する。"""
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'document_changed:' + str(path))
    return json.loads(raw)


def one(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any]:
    """重複や未登録を拒否する。"""
    selected = [row for row in rows if row.get(key) == value]
    require(len(selected) == 1, 'unique_record:' + key)
    return selected[0]


def owned_path(value: str) -> Path:
    """原票のWSL絶対pathを同じworkspaceへ限定する。"""
    path = Path(value).resolve()
    require(path.is_relative_to(VERIFY.resolve()), 'foreign_artifact_path')
    return path


def source_contract(source: str, source_sha256: str) -> dict[str, Any]:
    """preregと元dataset/tableの所属を結合し、caller指定foldを使わない。"""
    require(type(source) is str and source in SOURCE_FOLDS, 'unknown_source')
    item = one(document(PREREG, PREREG_SHA)['sources'], 'source_video_id', source)
    require(type(source_sha256) is str
            and item['source_video_sha_expected_from_manifest'] == source_sha256, 'source_sha')
    manifest = document(DATASET / 'manifest.json', MANIFEST_SHA)
    document(DATASET / 'COMPLETE', COMPLETE_SHA)
    record = one(manifest['sources'], 'target_id', source.removeprefix('video_'))
    table = document(owned_path(record['table_root']) / 'manifest.json', record['table_manifest_sha256'])
    require(type(item['fold']) is int and item['fold'] == SOURCE_FOLDS[source]
            == record['partition_fold'] == table['partition_fold'], 'source_fold')
    require(table['source_video_id'] == source and table['source_group_id'] == item['source_group'],
            'source_group')
    return dict(source=source, source_id='sha256:' + source_sha256, fold=item['fold'],
                group=item['source_group'], prereg_sha256=PREREG_SHA)


def route_spec(module: ModuleType, api: Any, seed: int, fold: int) -> tuple[Any, ...]:
    """固定allfolds rootの原recordから元loader形式を構築する。"""
    suffix = module.SPECS[seed][1]
    root = VERIFY / (module.PREFIX + '_' + suffix)
    docs = [document(root / name, sha) for name, sha in zip(
        ('COMPLETE', 'PLAN.json', 'results.json'), module.ROOT_HASHES[suffix], strict=True)]
    api.fixed._validate_anchor_plan(docs[1], docs[2])
    require(api.fixed._anchor_dataset_receipt(docs[1])
            == api.fixed.EXPECTED_ANCHOR_DATASET_RECEIPT, 'training_dataset')
    rows = [one(docs[2]['results'][f'{kind}__seed_{seed}']['records'], 'eval_fold', fold)
            for kind in ('m0', 'm1_zero_values_and_masks')]
    for row in rows:
        api.outer.validate_outer_fold_record(row, fold)
    base, record = rows
    paths = [owned_path(row['checkpoint_reference']['path']) for row in rows]
    require(paths[0].parent == paths[1].parent, 'component_mismatch')
    require(record['m0_checkpoint_reference'] == base['checkpoint_reference'], 'anchor_reference')
    require(record['m0_state_sha256_before'] == record['m0_state_sha256_after'], 'anchor_changed')
    for row, path in zip(rows, paths, strict=True):
        ref = row['checkpoint_reference']
        require(owned_path(ref['component_root']) == path.parent
                and digest(path) == row['model_sha256'] == ref['sha256'], 'checkpoint_sha')
    prefix = module.PREFIX + '_'
    require(paths[0].parent.name.startswith(prefix), 'component_prefix')
    return (paths[0].parent.name[len(prefix):], suffix, base['model_sha256'],
            record['model_sha256'], record['m0_state_sha256_before'],
            record['fixed_m0_symmetric_platt_slope'])


@contextmanager
def source_loader(source: str, source_sha256: str) -> Iterator[ModuleType]:
    """元loaderの専用namespaceだけを所有。共有backendは削除しない。"""
    contract = source_contract(source, source_sha256)
    require(digest(LOADER) == LOADER_SHA, 'loader_changed')
    name = '_g3_source_loader_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, LOADER)
    require(spec is not None and spec.loader is not None, 'loader_spec')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        api = module.backend()
        specs = {seed: route_spec(module, api, seed, contract['fold']) for seed in module.SEEDS}
        if source == 'video_38':
            require(specs == module.SPECS, 'video38_original_specs')
        module.SOURCE_ID, module.FOLD, module.SPECS = contract['source_id'], contract['fold'], specs
        module.G3_CONTRACT = contract
        module.G3_GUARDS = {str(Path(__file__).resolve()): digest(Path(__file__)), str(LOADER): LOADER_SHA}
        yield module
    finally:
        # 他所有への差替えを削除しない。memberの寿命は呼出元が所有する。
        if sys.modules.get(name) is module:
            del sys.modules[name]
