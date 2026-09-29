"""固定記録に発火前の画像窓だけを追記する。"""
from __future__ import annotations
import gzip
import json
import argparse
from pathlib import Path
import cv2
from src.exchange_event_record import read_records, encode
from src.prefire_snapshot_reader import PrefireSnapshotReader, vote_window, WINDOW_SEC
from scripts.enrich_e26_midchain import frame_at
from scripts.enrich_e16_records_20260928 import ZENCHI_VIDEO
from scripts.run_e3_exchange_eval_20260926 import SOURCES, VIDEO_ROOT, save_json, digest

OUT = Path('logs/e31/records')
SAMPLE_HZ = 30


def fire_keys(source: str) -> set[tuple]:
    """E27の全発火時刻を列挙し、途中段の通知による重複画像取得を省く。"""
    suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
    records = [json.loads(line) for line in (Path('logs/e27/on')/suffix/'events.jsonl').read_text().splitlines()]
    return {(r['game_idx'], int(c['side'] == '2P'), c['trigger_sec']) for r in records for c in r['chains']}


def capture_window(reader: PrefireSnapshotReader, cap: cv2.VideoCapture,
                   idx: int, stamp: float, fps: float, game_start: float) -> dict:
    """原記録と同じ30Hz時刻を使い、検出時刻以後の画像を参照しない。"""
    reader.filters[idx].reset()
    frames = []
    for offset in range(round(WINDOW_SEC*SAMPLE_HZ), 0, -1):
        when = stamp-offset/SAMPLE_HZ
        if when < game_start:
            continue
        index = round(when*fps)
        frames.append(reader.read_frame(frame_at(cap, index), idx, when))
    return dict(vote_window(frames, stamp), raw_frames=frames)


def pack_manifest(path: Path) -> None:
    """セル別観測を可逆圧縮し、軽い出所情報から参照できるようにする。"""
    value = json.loads(path.read_text())
    if 'windows' not in value:
        return
    windows = value.pop('windows')
    archive = path.with_suffix('.windows.json.gz')
    with gzip.open(archive,'wt',encoding='utf-8') as stream:
        json.dump(windows,stream,ensure_ascii=False,separators=(',',':'))
    value.update(windows_path=str(archive), windows_sha256=digest(archive), window_count=len(windows))
    save_json(path,value)


def enrich(source: str) -> None:
    """入力原票を保持し、異なる発火ごとに一度だけ原映像を読む。"""
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT/f'{source}.jsonl.gz'
    if dest.exists() and dest.with_suffix('.json').exists():
        return
    record = Path('logs/e26/records')/dest.name
    video = VIDEO_ROOT/f'{source}_first_0_900_20260925_v1.mp4' if source in SOURCES else ZENCHI_VIDEO
    reader, cap = PrefireSnapshotReader(), cv2.VideoCapture(str(video))
    fps, cache, game, game_start = cap.get(cv2.CAP_PROP_FPS), {}, None, 0.
    required = fire_keys(source)
    assert fps > 0
    with gzip.open(dest.with_suffix('.tmp'), 'wt', encoding='utf-8') as stream:
        for row in read_records(record):
            if row['kind'] == 'update':
                result, stamp, current_game = row['args'][0], row['args'][3], row['args'][4]
                if current_game != game:
                    game, game_start = current_game, stamp
                for idx, side in enumerate((result.p1, result.p2)):
                    event = side.chain_event
                    if event is not None:
                        key = (game, idx, event.trigger_sec)
                        if key not in required:
                            continue
                        if key not in cache:
                            cache[key] = capture_window(reader, cap, idx, event.trigger_sec, fps, game_start)
                            print(source, key, cache[key]['reason'], flush=True)
                        side.prefire_snapshot = {k:v for k,v in cache[key].items() if k != 'raw_frames'}
            stream.write(json.dumps(encode(row), ensure_ascii=False, separators=(',', ':'))+'\n')
    cap.release()
    dest.with_suffix('.tmp').replace(dest)
    save_json(dest.with_suffix('.json'), dict(source_sha256=digest(record), video=str(video), fps=fps,
        windows=[dict(game=k[0], side=k[1], trigger_sec=k[2], **v) for k,v in cache.items()]))
    pack_manifest(dest.with_suffix('.json'))


def main() -> None:
    """GPU読取を一プロセスに制限する。"""
    cv2.setNumThreads(1)
    for source in ('review', *SOURCES, 'zenchi'):
        enrich(source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-only', action='store_true')
    if parser.parse_args().pack_only:
        for path in OUT.glob('*.jsonl.json'):
            pack_manifest(path)
    else:
        main()
