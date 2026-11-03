"""採点前にローカル動画・除外標本・動画単位の分割を固定する。"""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import cv2
from scripts.exchange_review_v32_data import BASE, RUNS, source, fire_rows, video_path
from scripts.audit_exchange_counts_v33 import OUT as V33
from scripts.audit_exchange_cells_v34 import OUT as V34
from scripts.exchange_review_io import save, sha256

OUT = BASE/'v3_5'
EVIDENCE = Path('docs/verification/exch_review_v35')
VIDEOS = (32, 34, 51, 61, 92)
DEVELOPMENT, HOLDOUT = (32, 51, 92), (34, 61)
MAX_GAMES = 10


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding='utf-8'))


def excluded(video: int) -> set[tuple[int, int]]:
    rows = read(V33/f'calibration/input_{video}.json')
    keys = {(r['game'], r['event']) for r in rows}
    # 前版で原画を見た5件は今回の未見集合からも外す。
    keys.update((r['game'], r['event_id']) for r in read(V34/'holdout_scored.json') if r['video'] == video)
    return keys


def inventory() -> list[dict]:
    titles = {int(r['video_idx']): r['title'] for r in csv.DictReader(
        Path('data/phase_e_dl_index.tsv').open(encoding='utf-8-sig'), delimiter='\t')}
    rows = []
    roots = ('tasks/done/ei_ed_parallel_after_bo5/artifacts', 'exchange_runs',
             'tasks/done/ed_exchange_shadow_12videos_recompute/artifacts',
             'tasks/done/eh_ed_resume_from_3rd/artifacts')
    numbers = sorted({int(p.name.removeprefix('ed_video_')) for r in roots
                      for p in (RUNS/r).glob('ed_video_*') if (p/'video.mp4').is_file()})
    for video in numbers:
        path = video_path(video)
        if path is None:
            continue
        cap = cv2.VideoCapture(str(path))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        title = titles.get(video, '')
        tier = any(word in title for word in ('マスター', 'チャレンジャー', 'S級', 'A級'))
        rows.append(dict(video=video, path=str(path), bytes=path.stat().st_size,
                         fps=fps, frames=int(frames), seconds=frames/fps, title=title, tier=tier))
    return rows


def compact_manifest(result: dict) -> dict:
    """コミットには選定キーだけ残し、不要な予測木・共通手時間表を重複させない。"""
    selected = []
    for entry in result['selected']:
        targets = [dict(event_id=f['event_id'], game=f['game_idx'], t=f['t_sec'],
                        trigger=f['trigger_sec'], side=f['exchange_input']['attacker']) for f in entry['targets']]
        selected.append(dict(entry, targets=targets))
    return dict(result, selected=selected, full_manifest_sha256=sha256(OUT/'manifest.json'))


def main() -> None:
    if (OUT/'manifest.json').exists():
        raise FileExistsError('採点前固定のmanifestは上書きしません')
    local, selected = inventory(), []
    for video in VIDEOS:
        assert next(r['tier'] for r in local if r['video'] == video)
        fires, omitted = fire_rows(source(video)), excluded(video)
        games = sorted({f['game_idx'] for f in fires})[:MAX_GAMES]
        targets = [f for f in fires if f['game_idx'] in games
                   and (f['game_idx'], f['event_id']) not in omitted
                   and len(f['exchange_input']['colors']) == 4]
        selected.append(dict(video=video, games=games, targets=targets,
            all_notifications=len(fires), target_notifications=len(targets),
            by_game=dict(Counter(f['game_idx'] for f in targets)),
            input_sha256=sha256(source(video)/'inputs.jsonl.gz'),
            fit_sha256=sha256(V33/f'calibration/input_{video}.json')))
    result = dict(version='v3.5', inventory=local, selected=selected,
        development=list(DEVELOPMENT), holdout=list(HOLDOUT),
        rule='動画番号昇順の交互分割。各動画の最初の10完全試合内のfit・既監査外通知を全採取。初段式通知のみ採点。',
        limitation='ローカル元映像の動画自体はfitと共通。標本重複を除外する検証で、完全未知動画の検証ではない。',
        policy_sha256={v: sha256(p/'policy.json') for v, p in [('v33', V33), ('v34', V34)]})
    save(OUT/'manifest.json', result)
    save(EVIDENCE/'manifest.json', compact_manifest(result))
    print(json.dumps(dict(inventory=local, selected=[{k:v for k,v in r.items() if k != 'targets'} for r in selected]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
