"""段3合格票と内容SHAを照合してから比較監督器を起動する唯一の入口。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

LAUNCH_TOKEN = 'verified-full-comparison'
MODEL_NAMES = ('ui.joblib', 'digit_student.joblib')
INPUT_KEYS = ('mask_model', 'b3_model', 'b3_table')
MIN_PROOF_BOUNDARIES = 3


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_segmentation(config: dict, gate: dict) -> None:
    """同等性の実測票と所有区間全体を固定し、未検証の分割を拒否する。"""
    from scripts.image_free_segment_merge import validate_segments
    root = Path(config['output'])
    proof_path = root / 'split_proof.json'
    plan_path = root / 'segment_plan.json'
    for name, path in (('split_proof_sha256', proof_path), ('segment_plan_sha256', plan_path)):
        if sha(path) != gate.get(name):
            raise ValueError(f'区間分割の証拠または計画SHAが不一致: {name}')
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    cases = proof.get('cases', [])
    if config.get('comparison_mode') == 'paired_cold':
        verify_paired(proof, gate)
    else:
        verify_continuous(proof, cases)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('comparison_mode') != config.get('comparison_mode'):
        raise ValueError('比較方式が計画と一致しません')
    validate_segments(plan['segments'], plan['expected_frames'])
    assigned = [s for s in plan['segments'] if s['host'] == config['host']]
    keys = ('id', 'video_id', 'start_frame', 'owned_start', 'end_frame', 'fps')
    signature = lambda values: [{k: v[k] for k in keys} for v in values]
    if signature(assigned) != signature(config['videos']):
        raise ValueError('実行区間が割当済み計画と一致しません')


def verify_continuous(proof: dict, cases: list[dict]) -> None:
    boundaries = {(c['video'], c['start']) for c in cases}
    if (proof.get('passed') is not True or len(boundaries) < MIN_PROOF_BOUNDARIES
            or any(c.get('passed') is not True or not c.get('complete_reference')
                   or c.get('mismatches') or c.get('feature_rows', 0) <= 0 for c in cases)):
        raise ValueError('3境界の全項目同等性が未合格です')


def verify_paired(proof: dict, gate: dict) -> None:
    from scripts.image_free_paired import COMPARISON_MODE, DISCLAIMER, KNOWN_WINDOWS
    if (proof.get('comparison_mode') != COMPARISON_MODE or proof.get('disclaimer') != DISCLAIMER
            or proof.get('continuous_equivalence') is not False or proof.get('passed') is not True):
        raise ValueError('対の比較の段3が未合格です')
    cases = {c['video']: c for c in proof.get('cases', [])}
    for video, (first, last) in KNOWN_WINDOWS.items():
        case = cases.get(video, {})
        if (case.get('passed') is not True or case.get('start') != first or case.get('end') != last+1
                or case.get('frames') != last-first+1 or case.get('board_mismatches') != []):
            raise ValueError('既知区間の盤面差0を再現していません')
    if gate.get('fix2_r2_exit') != 0 or not gate.get('fix2_r2_commit'):
        raise ValueError('fix2後半検証の完了確認が必要です')
    if proof.get('model_sha256') != gate.get('model_sha256'):
        raise ValueError('対の比較の段3モデルが実行モデルと違います')
    if proof.get('source_manifest') != gate.get('source_manifest'):
        raise ValueError('対の比較の段3ソースが実行ソースと違います')


def verify(config: dict, source: Path, gate: dict) -> None:
    if gate.get('gate_passed') is not True:
        raise ValueError('段3が未合格のため全区間比較は起動しません')
    actual = {str(p.relative_to(source)): sha(p) for base in ('src', 'scripts')
              for p in (source / base).rglob('*.py')}
    if actual != gate.get('source_manifest'):
        raise ValueError('段3合格ソースと実行ソースが一致しません')
    for name in MODEL_NAMES:
        if sha(Path(config['output']) / name) != gate['model_sha256'][name]:
            raise ValueError(f'段3合格モデル不一致: {name}')
    for key in INPUT_KEYS:
        if sha(Path(config[key])) != gate['input_sha256'][key]:
            raise ValueError(f'段3合格入力不一致: {key}')
    if config.get('split_streams'):
        verify_routes(config, gate)
    elif config.get('segmented'):
        verify_segmentation(config, gate)
    elif any(v.get('start_frame', 0) or v.get('end_frame') for v in config['videos']):
        raise ValueError('分割同等性未合格のため途中開始・終了は禁止')


def verify_routes(config: dict, gate: dict) -> None:
    from scripts.image_free_stream_merge import validate
    root = Path(config['output'])
    for name in ('route_proof', 'route_plan'):
        if sha(root/(name+'.json')) != gate.get(name+'_sha256'):
            raise ValueError('単独通しの証拠・計画SHAが違います')
    proof = json.loads((root/'route_proof.json').read_text())
    cases = proof.get('cases', [])
    if (proof.get('passed') is not True or len(cases) < 4
            or {c['route'] for c in cases} != {'old', 'new'}
            or any(c.get('passed') is not True or c.get('mismatches')
                   or c.get('feature_rows', 0) <= 0 for c in cases)):
        raise ValueError('別プロセス・CP再開の全項目一致が未合格です')
    if proof.get('source_manifest') != gate['source_manifest'] or proof.get('model_sha256') != gate['model_sha256']:
        raise ValueError('単独通しの段3ソース・モデルが違います')
    if gate.get('fix2_r2_exit') != 0 or not gate.get('fix2_r2_commit'):
        raise ValueError('fix2後半検証が未完了です')
    plan = json.loads((root/'route_plan.json').read_text())
    validate(plan)
    expected = [j for j in plan['jobs'] if j['host'] == config['host']]
    keys = ('id', 'video_id', 'route', 'start_frame', 'end_frame', 'owned_start', 'owned_end', 'seed_sha256')
    if [{k: j.get(k) for k in keys} for j in expected] != [{k: j.get(k) for k in keys} for j in config['videos']]:
        raise ValueError('通し経路の担当が計画と違います')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--execute', required=True, choices=[LAUNCH_TOKEN])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    gate = json.loads(args.gate.read_text(encoding='utf-8-sig'))
    verify(config, args.source, gate)
    command = [sys.executable, '-m', 'scripts.image_free_parallel',
               '--config', str(args.config.resolve()), '--source', str(args.source.resolve()),
               '--workers', str(args.workers), '--auto-workers']
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
