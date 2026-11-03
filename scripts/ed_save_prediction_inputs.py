"""ed完了後・動画削除前の任意追加保存。既定OFF、既存成果は読取専用。"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
from itertools import groupby
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from scripts.collect_exchange_history_v34 import observe, HISTORY_SIZE
from scripts.collect_exchange_holdout_v34 import observation
from scripts.ed_prediction_capture import CropArchive, SavedImages, capture_ocr
from scripts.exchange_counts_v33_collect import extract_sources
from scripts.exchange_review_io import fires, jsonl, sha256
from scripts.exchange_stream_v310 import capture
from scripts.predict_exchange_v35 import make_input
from scripts.replay_exchange_projection_shadow import updates

SCHEMA = 1
POLICY = Path('docs/verification/exch_review_v34/policy.json')
CAPTURE_CODE = ('scripts/ed_save_prediction_inputs.py', 'scripts/ed_prediction_capture.py',
    'scripts/exchange_counts_v33_collect.py', 'scripts/exchange_stream_v310.py',
    'scripts/predict_exchange_v35.py', 'src/image_reader.py', 'src/score_ocr.py',
    'src/ui_mask.py', 'src/phase_j/dshow_color.py')
TEMPLATE_DIGITS = 10
IMAGE_FREE_FLAGS = ('PUYO_IMAGE_FREE_ALL', 'PUYO_IMAGE_FREE_UI_MASK', 'PUYO_IMAGE_FREE_SCORE_OCR')


def write(path: Path, value: Any) -> None:
    """中断ファイルを完了原票と取り違えないよう置換する。"""
    temporary = path.with_name(path.name+'.tmp')
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    temporary.write_bytes(gzip.compress(payload, mtime=0) if path.suffix == '.gz' else payload)
    temporary.replace(path)


def read(path: Path) -> Any:
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == '.gz' else payload)


def validate_record(path: Path) -> None:
    last = None
    for row in jsonl(path):
        last = row
    if last is None or last.get('kind') != 'complete':
        raise ValueError('ed認識入力の完了マーカーがありません')


def capture_identity(templates: Path) -> dict:
    return dict(code={name: sha256(Path(name)) for name in CAPTURE_CODE},
        templates={p.relative_to(templates.parent).as_posix(): sha256(p)
                   for p in sorted(templates.parent.rglob('*.png'))})


def validate_templates(templates: Path) -> None:
    required = [templates/f'digit_{i}.png' for i in range(TEMPLATE_DIGITS)]
    required.append(templates/'formula_mult.png')
    if not all(p.is_file() for p in required) or not list(templates.parent.glob('x_mark*.png')):
        raise FileNotFoundError('数字・式・×印のテンプレートが不足しています')
    if any(os.environ.get(flag, '0') != '0' for flag in IMAGE_FREE_FLAGS):
        raise ValueError('この採取はv3.4/v3.10と同じテンプレート方式に固定します')


def snapshot(fire: dict, raw: Any, histories: list, images: SavedImages) -> dict:
    side = fire['exchange_input']['attacker']
    history = [h for h in histories[side] if h['t'] <= fire['trigger_sec']]
    return dict(fire=fire, sources=extract_sources(raw, fire['t_sec'], fire['trigger_sec'], side, images),
        history=dict(history=deepcopy(history[-HISTORY_SIZE:]),
            history_latest=max((h['t'] for h in history), default=None),
            hidden_history=[dict(Counter(h['board'][c] for h in history)) for c in range(6)]))


def collect_game(rows: list[tuple], targets: list[dict], images: SavedImages) -> tuple:
    histories, observations, items, rejected = [[], []], [[], []], [], []
    wanted = {(f['t_sec'], f['exchange_input']['attacker']): f for f in targets}
    seen = set()
    for args in rows:
        stamp = args[3]
        for side, raw in enumerate((args[0].p1, args[0].p2)):
            observe(histories, raw, side, stamp)
            observations[side].append(observation(raw, stamp))
            fire = wanted.get((stamp, side))
            if fire is None:
                continue
            seen.add(fire['event_id'])
            event = raw.chain_event
            if event is None or event.mechanism != 'formula_read' or event.chain_count != 1:
                rejected.append(dict(event_id=fire['event_id'], reason='独立した初段式通知でない'))
            else:
                items.append(snapshot(fire, raw, histories, images))
    rejected.extend(dict(event_id=f['event_id'], reason='対応する認識通知がない')
                    for f in targets if f['event_id'] not in seen)
    return items, observations, rejected


def finish_game(video: int, folder: Path, items: list, observations: list,
                ocr: dict, images: SavedImages, rejected: list) -> dict:
    policy, inputs = read(POLICY), []
    for item in items:
        try:
            inp = make_input(video, item, ocr, policy)
        except ValueError as error:
            rejected.append(dict(event_id=item['fire']['event_id'], reason=str(error)))
            continue
        stream = capture(inp, ocr, observations[inp['side']], images)
        inputs.append(dict(input=inp, stream=stream))
    write(folder/'raw_inputs.json.gz', items)
    write(folder/'observations.json.gz', observations)
    write(folder/'ocr.json.gz', ocr)
    write(folder/'prediction_inputs.json.gz', inputs)
    write(folder/'rejected.json', rejected)
    return dict(inputs=len(inputs), rejected=len(rejected), targets=len(items)+sum(
        r['reason'] in ('独立した初段式通知でない', '対応する認識通知がない') for r in rejected))


def game_done(folder: Path, identity: dict) -> bool:
    receipt = folder/'complete.json'
    if not receipt.exists():
        return False
    saved = read(receipt)
    if saved['identity'] != identity:
        raise ValueError(f'再開入力が前回と異なります: {folder}')
    for name, digest in saved['sha256'].items():
        if sha256(folder/name) != digest:
            raise ValueError(f'完了済み原票が破損: {folder/name}')
    return True


def save_game(args: argparse.Namespace, game: int, rows: list, targets: list,
              identity: dict) -> None:
    folder = args.output/f'game_{game}'
    folder.mkdir(parents=True, exist_ok=True)
    if game_done(folder, identity):
        print('再開時スキップ', game, flush=True)
        return
    began = perf_counter()
    archive = CropArchive(folder/'crops.zip')
    live_color = args.image_color_mode == 'live601to709'
    images = SavedImages(args.video, args.templates, archive, live_color)
    try:
        items, observations, rejected = collect_game(rows, targets, images)
        ocr = capture_ocr(args.video, rows[0][3], rows[-1][3], args.templates, archive, live_color,
                          getattr(args, 'score_region_offsets', None))
        summary = finish_game(args.video_id, folder, items, observations, ocr, images, rejected)
    finally:
        images.cap.release()
        archive.archive.close()
    files = [p for p in folder.iterdir() if p.is_file() and p.name != 'complete.json']
    summary.update(identity=identity, game=game, start=rows[0][3], end=rows[-1][3],
        wall_sec=perf_counter()-began, bytes=sum(p.stat().st_size for p in files),
        sha256={p.name: sha256(p) for p in files})
    write(folder/'complete.json', summary)
    print('試合保存完了', game, summary['inputs'], summary['rejected'], summary['wall_sec'], flush=True)


def run(args: argparse.Namespace) -> None:
    if not args.save_prediction_inputs:
        return  # OFF時は存在検査・mkdir・既存ファイル書込みも一切しない。
    validate_templates(args.templates)
    if args.output.resolve().is_relative_to(args.ed_run.resolve()):
        raise ValueError('既存ed成果とは別の出力先を指定してください')
    args.output.mkdir(parents=True, exist_ok=True)
    record = args.ed_run/'out/inputs.jsonl.gz'
    shadow = args.ed_run/'out/spool/exchange_projection_shadow.jsonl'
    identity = dict(schema=SCHEMA, video=args.video_id, video_sha256=sha256(args.video),
        record_sha256=sha256(record), shadow_sha256=sha256(shadow), policy_sha256=sha256(POLICY),
        games=args.games, image_color_mode=args.image_color_mode, **capture_identity(args.templates))
    # gzip/JSONの途中終了を成功扱いする既存readerの前に完全性を検査する。
    validate_record(record)
    if getattr(args, 'align_score_roi', False):
        from scripts.ed_score_alignment import calibrate, offsets
        first_game, first_rows = next((g, list(r)) for g, r in groupby(
            updates(record), key=lambda a: a[4]) if g > 0)
        alignment = calibrate(args.video, args.templates, first_rows[0][3], first_rows[-1][3],
                              args.image_color_mode == 'live601to709')
        alignment['game'] = first_game
        args.score_region_offsets = offsets(alignment)
        identity['score_alignment'] = alignment
        identity['code']['scripts/ed_score_alignment.py'] = sha256(Path('scripts/ed_score_alignment.py'))
        write(args.output/'score_alignment.json', alignment)
    targets = [f for f in fires(list(jsonl(shadow))) if f['game_idx'] > 0 and f.get('exchange_input')]
    selected = set(args.games) if args.games else {f['game_idx'] for f in targets}
    completed = []
    for game, group in groupby(updates(record), key=lambda a: a[4]):
        if game not in selected:
            continue
        rows = list(group)
        save_game(args, game, rows, [f for f in targets if f['game_idx'] == game], identity)
        completed.append(game)
    if set(completed) != selected:
        raise ValueError(f'試合欠測: {selected-set(completed)}')
    write(args.output/'complete.json', dict(identity=identity, games=completed))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--save-prediction-inputs', action='store_true')
    result.add_argument('--align-score-roi', action='store_true',
                        help='最初の試合序盤の式で得点枠を動画ごとに固定較正（既定OFF）')
    result.add_argument('--ed-run', type=Path, required=True)
    result.add_argument('--video', type=Path, required=True)
    result.add_argument('--video-id', type=int, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--templates', type=Path, default=Path('models/ui_templates/score_digits'))
    result.add_argument('--games', type=int, nargs='+')
    result.add_argument('--image-color-mode', choices=('live601to709', 'review_raw'),
                        default='live601to709', help='liveと同じ補正。未補正は既存レビューとの比較専用')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
