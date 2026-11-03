"""v4.8入力の読取専用検収。画像ZIPを含む全SHAを照合する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


from collections import Counter
from pathlib import Path
import json
import subprocess
import sys
from typing import Iterator
from scripts.ed_save_prediction_inputs import read
from scripts.exchange_review_io import sha256, jsonl, fires, save
from scripts.prefire_v44_data import source, VIDEOS

ROOT = Path('/mnt/d/puyo_analyzer')
OUT = Path(_public_dev_text('<work>/exch_review_render/v4_8'))
EVIDENCE = Path('docs/verification/prefire_v48')
BASE = ROOT/'ed_inputs_rebuild_20261015'
FIXED = (39, 94)


def native_hashes(path: Path) -> dict[int, str]:
    windows = 'D:/'+path.relative_to('/mnt/d').as_posix()
    command = ['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        _public_dev_text('<work>/wt_prefire/scripts/prefire_v48_hash.ps1'), '-SavedPath', windows]
    result = {}
    with subprocess.Popen(command, stdout=subprocess.PIPE, encoding='utf-8-sig') as process:
        for line in process.stdout:
            row = json.loads(line)
            result[row['game']] = row['complete_sha256']
            print('全ファイルSHA', windows, row['game'], flush=True)
        assert process.wait() == 0, 'Windows SHA検査失敗'
    return result


def folder(video: int) -> Path:
    if video in FIXED:
        return ROOT/f'ed_inputs_rebuild_20261015_v{video}fix/saved'
    paths = list(BASE.glob(f'lane_*/video_{video}/saved'))
    assert len(paths) == 1, (video, paths)
    return paths[0]


def receipt_for(video: int, path: Path, complete: dict) -> dict:
    identity = complete['identity']
    manifest = next(r for r in read(Path('docs/tasks/ed_prediction_inputs/manifest.json')) if r['video'] == video)
    assert all(identity[k] == manifest[k] for k in ('record_sha256','shadow_sha256','video_sha256'))
    assert identity['policy_sha256'] == sha256(Path('docs/verification/exch_review_v34/policy.json'))
    assert identity['video'] == video and identity['games'] is None
    assert identity['image_color_mode'] == 'live601to709'
    assert ('score_alignment' in identity) == (video in FIXED)
    if video in FIXED:
        receipt = read(path.parent/'full_summary.json')
        download = read(path.parent/'download_verified.json')
        assert receipt['passed'] and receipt['output_sha_verified']
        assert download['sha256'] == identity['video_sha256']
        assert sha256(path.parent/'download/video.mp4') == identity['video_sha256']
    else:
        receipt = read(path.parent/'receipt.json')
        assert receipt['video'] == video and receipt['games'] == complete['games']
    return receipt


def verify_video(video: int) -> dict:
    path = folder(video)
    complete = read(path/'complete.json')
    receipt_for(video, path, complete)
    identity = complete['identity']
    record = source(video)/'inputs.jsonl.gz'
    shadow = source(video)/'exchange_projection_shadow.jsonl'
    assert sha256(record) == identity['record_sha256']
    assert sha256(shadow) == identity['shadow_sha256']
    targets = [f for f in fires(list(jsonl(shadow))) if f['game_idx'] > 0 and f.get('exchange_input')]
    assert set(complete['games']) == {f['game_idx'] for f in targets}
    counts, reasons, checked = Counter(), Counter(), []
    hashes = native_hashes(path)
    for game in complete['games']:
        local = path/f'game_{game}'
        assert sha256(local/'complete.json') == hashes[game]
        ticket = read(local/'complete.json')
        assert ticket['identity'] == identity
        inputs, rejected = read(local/'prediction_inputs.json.gz'), read(local/'rejected.json')
        ids = [r['input']['event_id'] for r in inputs]+[r['event_id'] for r in rejected]
        expected = {f['event_id'] for f in targets if f['game_idx'] == game}
        assert len(ids) == len(set(ids)) and set(ids) == expected
        assert (len(inputs), len(rejected), len(ids)) == tuple(ticket[k] for k in ('inputs','rejected','targets'))
        counts.update({k:ticket[k] for k in ('inputs','rejected','targets')})
        reasons.update(r['reason'] for r in rejected)
        checked.append(dict(game=game, complete_sha256=sha256(local/'complete.json'), files=ticket['sha256']))
        print('SHA照合', video, game, flush=True)
    return dict(video=video, passed=True, path=str(path), **counts, reasons=dict(reasons),
        alignment=identity.get('score_alignment'), complete_sha256=sha256(path/'complete.json'), games=checked)


def sources(video: int, games: set[int] | None = None) -> Iterator[tuple[dict, dict]]:
    verified = read(EVIDENCE/'inputs_verified.json')
    item = next(r for r in verified if r['video'] == video)
    assert item['passed'], item
    path = folder(video)
    assert sha256(path/'complete.json') == item['complete_sha256']
    for game in item['games']:
        if games is not None and game['game'] not in games:
            continue
        local = path/f'game_{game["game"]}'
        assert sha256(local/'prediction_inputs.json.gz') == game['files']['prediction_inputs.json.gz']
        yield from ((r['input'], r['stream']) for r in read(local/'prediction_inputs.json.gz'))


def main() -> None:
    if '--preflight' in sys.argv:
        for video in VIDEOS:
            path = folder(video)
            receipt_for(video,path,read(path/'complete.json'))
            print('receipt事前照合', video, flush=True)
        return
    results = []
    for video in VIDEOS:
        try:
            results.append(verify_video(video))
        except Exception as error:
            results.append(dict(video=video, passed=False, reason=str(error)))
            print('使用不可', video, repr(error), flush=True)
        save(EVIDENCE/'inputs_verified.json', results)
    assert all(r['passed'] for r in results), '不合格動画をfitへ渡さない'


if __name__ == '__main__':
    main()
