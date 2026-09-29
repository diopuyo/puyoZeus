"""E23保存原票だけから、全評価時点・着弾枝・原映像フレームを取り出す。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.run_e3_exchange_eval_20260926 import VIDEO_ROOT, save_json
from src.exchange_event_record import read_records

OUT = Path('logs/e24')
WINDOWS = dict(review=(2755., 2771.), q_7gc4TgFig=(825., 854.), fcXG83vInDY=(483., 540.))
SIDES = dict(review=1, q_7gc4TgFig=1, fcXG83vInDY=0)
TIMES = dict(q_7gc4TgFig=(828., 831., 832.733333, 833.4, 835., 838., 842., 848., 852.6),
             fcXG83vInDY=(486., 489., 490.866667, 492., 494., 497., 502., 515., 537.866667))
TILE_SIZE = (640, 360)
CONTACT_COLUMNS = 3
CAPTION_POSITION = (8, 22)
CAPTION_SCALE = .7
CAPTION_COLOR = (255, 255, 255)
CAPTION_THICKNESS = 2
MSEC_PER_SEC = 1000


def write_csv(path: Path, rows: list[dict]) -> None:
    """nullを空欄で保持し、未実行を応手0と混同しない。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_rows(path: Path, rows: list[dict]) -> None:
    """盤面の各セルを改行せず、評価1件を1行にして原票の差分量を抑える。"""
    text = '[\n' + ',\n'.join(json.dumps(row, ensure_ascii=False,
        separators=(',', ':')) for row in rows) + '\n]\n'
    assert json.loads(text) == rows
    path.write_text(text, encoding='utf-8')


def event_path(source: str) -> Path:
    """既存E23のON出力を読み、再評価による結果変更を避ける。"""
    root = Path('logs/e23/on')
    return (root/source if source == 'review' else root/'renders'/source/'on')/'events.jsonl'


def evaluations(source: str) -> None:
    """各評価の最終判定と、各着弾の全枝の応手量をCSV化する。"""
    begin, end = WINDOWS[source]
    idx, rows, branches, events = SIDES[source], [], [], []
    for line in event_path(source).read_text().splitlines():
        event = json.loads(line)
        selected = [v for v in event['values'] if begin <= v['t_sec'] <= end and 'multi_landing' in v]
        if not selected:
            continue
        events.append({k: v for k, v in event.items() if k != 'values'})
        for value in selected:
            proof = value['multi_landing'][idx]
            rounds = proof.get('rounds', [])
            row = dict(t_sec=value['t_sec'], exchange_id=event['exchange_id'], side=f'{idx+1}P',
                source=value['source'], p1=value['p1'], incoming=value['incoming'][idx],
                hands=value['hands'][idx], certain=value['completion_certain'][idx],
                proof_dead=proof['dead'], reason=proof['reason'], nodes=proof.get('nodes'),
                rounds=len(rounds), death_round=len(rounds) if proof['dead'] else None,
                held_from_sec=value.get('held_from_sec'), declared_dead=f'{idx+1}P' in value['dead_sides'])
            for turn, states in enumerate(rounds, 1):
                sends = [s['maximum_send'] for s in states]
                row[f'round{turn}_send_min'], row[f'round{turn}_send_max'] = min(sends), max(sends)
                for branch, state in enumerate(states):
                    branches.append(dict(t_sec=value['t_sec'], round=turn, branch=branch, **state))
            rows.append(row)
    write_csv(OUT/f'{source}_evaluations.csv', rows)
    write_csv(OUT/f'{source}_landing_branches.csv', branches)
    save_json(OUT/f'{source}_events.json', events)
    print(source, len(rows), dict((r, sum(v['reason'] == r for v in rows)) for r in {v['reason'] for v in rows}), flush=True)


def raw_inputs(source: str) -> None:
    """物理入力の時系列を保存し、投下累計・通知・得点を独立に追跡する。"""
    begin, end = WINDOWS[source]
    rows, changes, previous = [], [], None
    for item in read_records(Path('logs/e16/records')/f'{source}.jsonl.gz'):
        if item['kind'] != 'update':
            continue
        result, snapshot, finalized, stamp, game, formulas, displayed, visible = item['args']
        if not begin <= stamp <= end:
            continue
        row = dict(t_sec=stamp, game=game, **vars(snapshot), **vars(finalized))
        boards = {}
        for i, side in enumerate((result.p1, result.p2)):
            label, chain = f'{i+1}P', side.chain_event
            row.update({label+'_state': side.state.name, label+'_score': side.score,
                label+'_next_slide_motion': side.next_slide_motion,
                label+'_displayed': displayed[i], label+'_formula': formulas[i], label+'_visible': visible[i],
                label+'_trigger': None if chain is None else chain.trigger_sec,
                label+'_count': None if chain is None else chain.chain_count,
                label+'_mechanism': None if chain is None else chain.mechanism})
            grid = getattr(side.confirmed_board, '_grid', side.confirmed_board)
            boards[label] = dict(board=None if grid is None else np.asarray(grid).tolist(),
                next=None if side.next_pair is None else list(side.next_pair),
                dnext=None if side.dnext_pair is None else list(side.dnext_pair))
        rows.append(row)
        key = json.dumps(boards)
        if key != previous:
            changes.append(dict(t_sec=stamp, sides=boards))
            previous = key
    write_csv(OUT/f'{source}_inputs.csv', rows)
    save_rows(OUT/f'{source}_boards.json', changes)


def frames(source: str, suffix: str = '') -> None:
    """原映像の無加工フレームと、一覧点検用の縮小画像を保存する。"""
    path = VIDEO_ROOT/f'{source}_first_0_900_20260925_v1.mp4'
    capture, tiles, manifest = cv2.VideoCapture(str(path)), [], []
    for stamp in TIMES[source]:
        capture.set(cv2.CAP_PROP_POS_MSEC, stamp*MSEC_PER_SEC)
        ok, frame = capture.read()
        assert ok
        dest = OUT/f'{source}_{stamp:.3f}.jpg'
        assert cv2.imwrite(str(dest), frame)
        tile = cv2.resize(frame, TILE_SIZE)
        cv2.putText(tile, f'{stamp:.3f}s', CAPTION_POSITION, cv2.FONT_HERSHEY_SIMPLEX,
                    CAPTION_SCALE, CAPTION_COLOR, CAPTION_THICKNESS)
        tiles.append(tile)
        manifest.append(dict(requested_sec=stamp, frame_index=int(capture.get(cv2.CAP_PROP_POS_FRAMES))-1,
            actual_sec=capture.get(cv2.CAP_PROP_POS_MSEC)/MSEC_PER_SEC, path=str(dest)))
    capture.release()
    assert cv2.imwrite(str(OUT/f'{source}_contact{suffix}.jpg'), np.vstack([
        np.hstack(tiles[start:start+CONTACT_COLUMNS])
        for start in range(0, len(tiles), CONTACT_COLUMNS)]))
    save_json(OUT/f'{source}_frames{suffix}.json', dict(video=str(path), frames=manifest))


def main() -> None:
    """評価器・フラグ・本番設定を変更せず、固定原票の診断だけを行う。"""
    OUT.mkdir(parents=True, exist_ok=True)
    for source in WINDOWS:
        evaluations(source)
        raw_inputs(source)
        if source in TIMES:
            frames(source)
        runtime = OUT/f'{source}_runtime.json'
        if runtime.exists():
            save_rows(runtime, json.loads(runtime.read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()
