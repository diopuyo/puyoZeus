"""事前登録のメタデータだけを固定する。学習・採点・モデル復元はしない。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[1])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import types
import unicodedata

import numpy as np

ROOT = Path('/mnt/d/puyo_analyzer/wt_b3')
OUT = ROOT / 'docs/b3_prereg_fill'
EXEV = Path(_public_dev_text('<repo>'))
ASSETS = Path(_public_dev_text('<repo>'))
B2 = Path('/mnt/d/puyo_analyzer/wt_b2/data/verify/recollect_2026-10-03')
REV = 'ccc9623'
sys.path.insert(0, str(ROOT))
from src.b3_train_folds import (video_to_group, outer_assignment, write_assignment_tsv,
                                fold_summary, normalize_player)


def sha(path: Path) -> str:
    """内容を解釈せずバイト列だけをハッシュする。"""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     sort_keys=True) + '\n', encoding='utf-8')


def blob(name: str) -> bytes:
    return subprocess.check_output(['git', '--git-dir', str(ASSETS / '.git'), 'show', f'{REV}:{name}'])


def hashes(name: str, paths: list[Path]) -> None:
    rows = ['path\tbytes\tsha256']
    for path in sorted(set(paths)):
        rows.append(f'{path}\t{path.stat().st_size}\t{sha(path)}')
    (OUT / name).write_text('\n'.join(rows) + '\n', encoding='utf-8')


def reference() -> None:
    """固定版の設定だけを実行し、モデルはバイト列として扱う。"""
    source = blob('src/production_config.py')
    module = types.ModuleType('prereg_reference_config')
    sys.modules[module.__name__] = module
    exec(compile(source, 'ccc9623:src/production_config.py', 'exec'), module.__dict__)
    values = {}
    for name, fn in vars(module).items():
        if inspect.isfunction(fn) and fn.__module__ == module.__name__ and not inspect.signature(fn).parameters:
            value = fn()
            values[name] = sorted(value) if isinstance(value, (set, frozenset)) else value
    save('r_flags.json', values)
    save('r_config.json', {'commit': subprocess.check_output(['git', '--git-dir', str(ASSETS / '.git'),
         'rev-parse', REV]).decode().strip(), 'production_config_sha256': hashlib.sha256(source).hexdigest(),
         'flags_sha256': sha(OUT / 'r_flags.json')})
    paths = [EXEV / 'models/cnn_phase_b_large_v2.pt', EXEV / 'models/cnn_global_best.pt']
    for directory, keys in [('exchange_event_v1', ['G_fe']),
                            ('exchange_event_v3', ['S1_prime_light', 'S3_prime_light']),
                            ('exchange_event_v1_r1_hgb', ['G_fe'])]:
        root = EXEV / 'models' / directory
        meta = json.loads((root / 'manifest.json').read_text())
        paths.append(root / 'manifest.json')
        paths.extend(root / meta['models'][key]['file'] for key in keys)
    paths.extend([EXEV / 'models/exchange_event_v1/M0/model.pt',
                  EXEV / 'models/exchange_event_v1/M0/manifest.json'])
    paths.extend(EXEV / 'data/verify/retrain148_2026-08-14' / n for n in
                 ['model_full148_full_features.joblib', 'feature_cols_full.json'])
    hashes('r_models.sha256.tsv', paths)
    names = ['src/production_config.py', 'src/exchange_display_smoothing.py',
             'scripts/visualize_advantage_overlay.py', 'src/recognition_pipeline.py',
             'src/exchange_event_evaluator.py', 'src/exchange_event_m0.py',
             'scripts/eval_set_score_20261001.py', 'scripts/eval_set2_score_20261001.py',
             'scripts/eval_set_noise_20261001.py', 'scripts/a2_score.py',
             'scripts/a1b_score.py', 'scripts/a1_completion_score.py',
             'scripts/falsedeath_score_20261001.py', 'scripts/falsedeath_v2_score_20261002.py',
             'scripts/falsedeath_v2d_display.py', 'scripts/_diag_set2_falsedeath_audit.py',
             'scripts/eval_set2_regression_20261001.py', 'scripts/_diag_set2_falsedeath_batch.py']
    rows = ['commit\tpath\tsha256']
    rows.extend(f'{REV}\t{n}\t{hashlib.sha256(blob(n)).hexdigest()}' for n in names)
    (OUT / 'r_source.sha256.tsv').write_text('\n'.join(rows) + '\n', encoding='utf-8')


def training() -> list[dict]:
    targets = json.loads((B2 / 'targets.json').read_text())
    rows = []
    for target in sorted(targets, key=lambda t: t['video_id']):
        path = B2 / 'receipts' / (target['video_id'] + '.json')
        receipt = json.loads(path.read_text())
        assert receipt['status'] == 'complete'
        attempt = Path(receipt['attempt'])
        if not attempt.is_absolute():
            attempt = B2.parents[2] / attempt
        lean = attempt / 'record/lean.npz'
        count = receipt.get('match_coverage', {}).get('denominator_real_matches')
        method = 'receipt.match_coverage.denominator_real_matches'
        if count is None:
            with np.load(lean, allow_pickle=False) as data:
                count = len(np.unique(data['game_idx']))
            method = 'lean.game_idx の一意数'
        rows.append(dict(video_id=target['video_id'], youtube_id=target['youtube_id'],
                         title=target['title'], tier=target['tier'], n_games=int(count),
                         count_source=method, receipt=str(path), receipt_sha256=sha(path),
                         attempt=str(attempt), lean_sha256=sha(lean),
                         observations_sha256=receipt['observations_sha256']))
    assert len(rows) == 60
    save('training_manifest.json', rows)
    hashes('receipts.sha256.tsv', [Path(r['receipt']) for r in rows])
    groups = video_to_group({r['video_id']: r['youtube_id'] for r in rows},
                            [('zB8d1FuCStU', 'iZZyNGtJ3G8')])
    counts, videos = {}, {}
    for row in rows:
        key = groups[row['video_id']]
        counts[key] = counts.get(key, 0) + row['n_games']
        videos.setdefault(key, []).append(row['video_id'])
    assignments = outer_assignment(counts)
    digest = write_assignment_tsv(OUT / 'outer_folds.tsv', counts, assignments, videos)
    save('training_summary.json', dict(videos=len(rows), groups=len(counts),
         games=sum(counts.values()), folds=fold_summary(counts, assignments),
         folds_sha256=digest, manifest_sha256=sha(OUT / 'training_manifest.json'),
         fold_source_sha256=sha(ROOT / 'src/b3_train_folds.py'),
         config_sha256=sha(ROOT / 'src/b3_train_config.py'), salt='B3-outer-v1|'))
    return rows


def winners(rows: list[dict]) -> None:
    """指定された学習側2動画だけの既存勝者列を照合する。"""
    result = {}
    for row in rows:
        if row['video_id'] not in ('video_c110', 'video_c112'):
            continue
        games = []
        with np.load(Path(row['attempt']) / 'record/lean.npz', allow_pickle=False) as data:
            for game in np.unique(data['game_idx']):
                sides = {}
                for side in np.unique(data['side']):
                    mask = (data['game_idx'] == game) & (data['side'] == side)
                    sides[str(side)] = np.unique(data['won'][mask]).tolist()
                games.append(dict(game=int(game), won_by_side=sides))
        sequence = []
        for game in games:
            values = game['won_by_side']
            valid = values in ({'1P': [1.0], '2P': [0.0]}, {'1P': [0.0], '2P': [1.0]})
            sequence.append(('1P' if values['1P'] == [1.0] else '2P') if valid else 'unknown')
        result[row['video_id']] = dict(receipt_games=row['n_games'], games=games, winner_sequence=sequence)
    result['exact_sequence_equal'] = (result['video_c110']['winner_sequence'] ==
                                      result['video_c112']['winner_sequence'])
    result['decision'] = '試合数・勝者列は完全一致しない。同一内容の再編集か否かは未確定。保守的な同一groupを維持。'
    save('c110_c112_comparison.json', result)


def players(rows: list[dict]) -> None:
    lines = ['video_id\tplayer_raw\tnormalized']
    names = {}
    for row in rows:
        title = unicodedata.normalize('NFKC', row['title'])
        pairs = re.findall(r'「([^「」]+?)\s+vs\s+([^「」]+?)」', title, re.I)
        if not pairs:
            title = re.sub(r'^(?:【[^】]*】)*', '', title)
            match = re.search(r'^\s*(.+?)\s+vs\s+(.+?)\s*\d+先', title, re.I)
            assert match, row['video_id']
            pairs = [match.groups()]
        for player in sorted(set(p.strip() for pair in pairs for p in pair)):
            key = normalize_player(player)
            lines.append(f"{row['video_id']}\t{player}\t{key}")
            item = names.setdefault(key, {'raw': set(), 'videos': set()})
            item['raw'].add(player)
            item['videos'].add(row['video_id'])
    (OUT / 'players.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save('players_normalized.json', {k: {n: sorted(v) for n, v in d.items()} for k, d in names.items()})


def sealed_hashes() -> None:
    """封印領域では列挙・stat・SHA256だけ。JSON/NPZ/ラベルは解釈しない。"""
    root = Path('/mnt/d/puyo_analyzer/wt_sealed/data/verify/sealed_eval_2026-10-03')
    paths = [p for p in root.rglob('*') if p.is_file()]
    evaluation = Path('/mnt/d/puyo_analyzer/b3_eval_records')
    paths.append(evaluation / 'b3_targets.json')
    for pattern in ('b3_sealed*', 'b3_v43*'):
        paths.extend(evaluation.glob(f'receipts/{pattern}.json'))
        for directory in (evaluation / 'attempts').glob(pattern):
            paths.extend(p for p in directory.rglob('*') if p.is_file())
    hashes('sealed.sha256.tsv', paths)


def regression(rows: list[dict]) -> None:
    root = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/falsedeath_fix/collect/records')
    ids = {r['youtube_id'] for r in rows}
    result = []
    for path in sorted(root.glob('*.jsonl.json')):
        data = json.loads(path.read_text())
        command = data.get('command', [])
        video = command[command.index('--video') + 1] if '--video' in command else ''
        candidates = re.findall(r'^([A-Za-z0-9_-]{11})_first_', Path(video).name)
        if not candidates:
            candidates = re.findall(r'^video_zenchi_([A-Za-z0-9_-]{11})\.mp4$', Path(video).name)
        youtube = candidates[0] if candidates else None
        result.append(dict(record=path.name.removesuffix('.jsonl.json'), source_video=video,
                           youtube_id=youtube, overlap=youtube in ids if youtube else None,
                           receipt=str(path), receipt_sha256=sha(path)))
    save('regression18.json', result)


def environment() -> None:
    from threadpoolctl import threadpool_info
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode()
    (OUT / 'pip_freeze_pc1.txt').write_text(freeze, encoding='utf-8')
    save('environment_pc1.json', dict(python=sys.version, executable=sys.executable,
         platform=platform.platform(), os_release=Path('/etc/os-release').read_text(),
         thread_env={k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}, threadpools=threadpool_info(),
         cpuinfo=subprocess.check_output(['lscpu']).decode()))
    pc2 = Path('/mnt/d/puyo_analyzer/share/results/pc2/setup/pip_freeze_pc2.txt')
    (OUT / 'pip_freeze_pc2.txt').write_bytes(pc2.read_bytes())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    reference()
    rows = training()
    winners(rows)
    players(rows)
    sealed_hashes()
    regression(rows)
    environment()
    print((OUT / 'training_summary.json').read_text())


if __name__ == '__main__':
    main()
