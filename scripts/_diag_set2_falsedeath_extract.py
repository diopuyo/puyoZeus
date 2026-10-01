"""誤確定6件の原票・入力・前後画像を変更なしで抽出する。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import cv2

ROOT = Path('/mnt/d/puyo_analyzer/wt_evalset')
OUT = ROOT / 'logs/eval_set/set2_falsedeath'
SET = ROOT / 'logs/eval_set/set2'
VIDEO = Path('/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/frames/video_zenchi_c0BQoMJwwQU.mp4')
OFFSETS = (-1.0, -0.2, 0.0, 0.2, 1.0)
WINDOW = 2.0


def save(path: Path, value: object) -> None:
    """UTF-8で原票を保存する。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')


def extract_cases() -> list[dict]:
    """既存採点とイベントの最初の誤確定を結び付ける。"""
    result = json.loads((SET / 'RESULT.json').read_text())
    games = json.loads((SET / 'labels.json').read_text())['games']
    cases = []
    for row in result['deaths']['prod']['rows']:
        if not row['false']:
            continue
        game = next(g for g in games if g['game'] == row['game'])
        case = dict(row, label=game)
        for line in (SET / 'replay/prod' / game['part'] / 'events.jsonl').read_text().splitlines():
            event = json.loads(line)
            values = [v for v in event['values'] if v['t_sec'] == row['first_sec']
                      and row['side'] in v.get('dead_sides', [])]
            if values:
                case.update(first_value=values[0], event=event)
                break
        cases.append(case)
    return cases


def extract_inputs(cases: list[dict]) -> None:
    """同じ記録を一度だけ走査し、前後の評価入力を保存する。"""
    for part in sorted({c['label']['part'] for c in cases}):
        selected = [c for c in cases if c['label']['part'] == part]
        buffers = {c['game']: [] for c in selected}
        with gzip.open(SET / 'collect/records' / f'{part}.jsonl.gz', 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['kind'] != 'update':
                    continue
                stamp = row['args']['tuple'][3]
                for case in selected:
                    if abs(stamp - case['first_sec']) <= WINDOW:
                        buffers[case['game']].append(row)
        for game, rows in buffers.items():
            save(OUT / f'game{game:02d}_inputs.json', rows)


def extract_frames(cases: list[dict]) -> None:
    """元映像のフレームを保存し、実際に取得したフレーム番号も残す。"""
    capture = cv2.VideoCapture(str(VIDEO))
    assert capture.isOpened(), VIDEO
    fps = capture.get(cv2.CAP_PROP_FPS)
    for case in cases:
        frames = []
        for offset in OFFSETS:
            frame_no = round((case['first_sec'] + offset) * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = capture.read()
            assert ok, frame_no
            name = f"game{case['game']:02d}_{offset:+.1f}.jpg"
            assert cv2.imwrite(str(OUT / name), frame)
            frames.append(dict(file=name, frame=frame_no, time=frame_no / fps))
        case['frames'] = frames
    capture.release()


def main() -> None:
    """原票と画像を指定ディレクトリだけに出力する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    cases = extract_cases()
    extract_inputs(cases)
    extract_frames(cases)
    save(OUT / 'cases.json', cases)
    print(json.dumps([dict(game=c['game'], source=c['first_value']) for c in cases]))


if __name__ == '__main__':
    main()
