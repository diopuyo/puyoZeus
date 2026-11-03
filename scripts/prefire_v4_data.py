"""v4の保存通知アダプター。実測ラベルは計算とは別の段で読む。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Iterator
import argparse
import json
import numpy as np
from scripts.exchange_review_io import grid, jsonl, pickles, save, sha256
from scripts.exchange_review_seed import updates
from scripts.exchange_review_v32_data import BASE, source, VIDEOS
from src.b3_timing import TimingTable
from src.exchange_tsumo_tape import TsumoTape
from src.prefire_worker import PrefireInput, calculate, ROLLOUTS

OUT = BASE/'v4'
EVIDENCE = Path('docs/verification/prefire_v4')
ROOTS = (Path('/mnt/d/puyo_analyzer/share/tasks/done/ei_ed_parallel_after_bo5/artifacts'),
         Path('/mnt/d/puyo_analyzer/share/exchange_runs'))
RANDOM_SEED = 20261015
SAMPLE_SEC = .5


def publication_fps(path: Path) -> float:
    """公開frameは入力動画のフレーム番号。認識通知30Hzとは別の時計。"""
    receipt = json.loads((path/'video_receipt.json').read_text(encoding='utf-8'))
    fps = float(receipt['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('入力動画fpsが不正です')
    return fps


def run_path(video: int) -> Path:
    for root in ROOTS:
        path = root/f'ed_video_{video}'
        if (path/'death.jsonl').exists():
            return path
    raise FileNotFoundError(f'動画{video}の勝敗原票なし')


def ledger(path: Path) -> list[dict]:
    import csv
    with (path/'review_data.csv').open(encoding='utf-8-sig') as stream:
        values = [dict(t_sec=float(r['t_sec']), game=int(r['game_idx']),
            pending=[int(float(r[f'{s}_incoming'] or 0)) for s in ('1P', '2P')]) for r in csv.DictReader(stream)]
    values += [dict(t_sec=r['t_sec'], game=r['game_idx'], pending=[r['pending_p1'], r['pending_p2']])
        for r in jsonl(path/'exchange_projection_shadow.jsonl') if 'pending_p1' in r and 'game_idx' in r]
    return sorted(values, key=lambda r: r['t_sec'])


def requests(path: Path, video: int, game_filter: int | None = None) -> Iterator[tuple]:
    tape, game, generation, previous = TsumoTape(), None, 0, None
    pending_rows, cursor, pending = ledger(path), 0, None
    colors: set[int] = set()
    for row in updates(path/'inputs.jsonl.gz'):
        stamp = row['t_sec']
        if row['game_idx'] != game:
            tape.reset()
            game, previous, start, colors = row['game_idx'], None, stamp, set()
        for side, raw in enumerate(row['sides']):
            tape.observe(side, stamp, raw['next'], raw['dnext'], raw['slide'])
            colors.update(c for p in tape.known_future(side) for c in p)
        while cursor < len(pending_rows) and pending_rows[cursor]['t_sec'] <= stamp:
            pending = pending_rows[cursor]
            cursor += 1
        if game <= 0 or game_filter is not None and game != game_filter:
            continue
        stable = tuple(s['state'] == 'STABLE' and s['board'] is not None for s in row['sides'])
        if not all(stable):
            continue
        boards = tuple(tuple(grid(s['board'])) for s in row['sides'])
        if boards == previous:
            continue
        previous = boards
        generation += 1
        # STABLE確定盤は直前の操作組を含む。NEXT以降を次の設置列とする。
        known = tuple(tape.known_future(s)[:3] for s in (0, 1))
        if pending is None or pending['game'] != game:
            yield None, dict(game=game, input_sec=stamp, reason='台帳未到達')
            continue
        req = PrefireInput(game, generation, stamp, (stamp, pending['t_sec']), boards,
            known, tuple(sorted(colors)), tuple(pending['pending']), stamp-start, stable,
            RANDOM_SEED+video*1000+game)
        yield req, None


def compute(video: int, game: int | None = None) -> None:
    import puyo_core
    output = OUT/f'compute_{video}.jsonl'
    if output.exists():
        raise FileExistsError(output)
    OUT.mkdir(parents=True, exist_ok=True)
    path, counts = source(video), dict(computed=0, missing=0, over_budget=0)
    with output.open('w', encoding='utf-8') as stream, ThreadPoolExecutor(ROLLOUTS) as pool:
        for request, missing in requests(path, video, game):
            try:
                value = calculate(request, puyo_core, TimingTable(), pool, enabled=True) if request else missing
            except ValueError as error:
                value = dict(game=request.game, input_sec=request.input_sec, reason=str(error))
            value['video'] = video
            if request:
                value['request'] = asdict(request)
            counts['computed' if 'sides' in value else 'missing'] += 1
            counts['over_budget'] += int(value.get('over_budget', False))
            stream.write(json.dumps(value, ensure_ascii=False)+'\n')
            if sum(counts.values()) % 500 == 0:
                stream.flush()
                print(video, counts, flush=True)
    save(OUT/f'compute_{video}_receipt.json', dict(counts=counts,
        input_sha256=sha256(path/'inputs.jsonl.gz'), output_sha256=sha256(output),
        timing=TimingTable().fit_scope, native_sha256=sha256(Path(puyo_core.__file__))))
    print(video, counts, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=int, required=True)
    parser.add_argument('--game', type=int)
    args = parser.parse_args()
    compute(args.video, args.game)


if __name__ == '__main__':
    main()
