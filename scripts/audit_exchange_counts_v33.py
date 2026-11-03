"""個数監査の原票採取。未来の式と盤面は採点ファイルにだけ保存する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


from dataclasses import asdict
from pathlib import Path
import json
import cv2
from scripts.exchange_review_v32_data import BASE, source, video_path
from scripts.exchange_review_io import save
from scripts.replay_exchange_projection_shadow import updates
from src.score_ocr import ScoreOcr, FormulaStepAccumulator

OUT = BASE/'video_80_g7_v3_3'
GAME, FPS, STRIDE = 7, 60, 2
TEMPLATES = Path(_public_dev_text('<repo>/models/ui_templates/score_digits'))


def inspect() -> None:
    """保存通知の利用可能な観測列を確認する。"""
    data = json.loads((BASE/'video_80_g7_v3_2/events.json').read_text())
    print('video', video_path(80))
    fires = {o['fire']['t_sec']: o for o in data['origins']}
    for args in updates(source(80)/'inputs.jsonl.gz', {GAME}):
        if args[3] in fires:
            item = fires[args[3]]
            raw = (args[0].p1, args[0].p2)[item['side']]
            print(item['event_id'], args[3], item['fire']['trigger_sec'], raw.state,
                  vars(raw.chain_event) if raw.chain_event else None)
            if item is data['origins'][0]:
                print('keys', vars(raw).keys())
                print('snapshot', raw.prefire_snapshot)


def initial() -> None:
    previous = None
    for args in updates(source(80)/'inputs.jsonl.gz', {GAME}):
        if args[3] > 565.5:
            break
        values = [(r.score, r.next_pair, r.dnext_pair, r.state.name,
                   r.confirmed_board._grid.sum() if r.confirmed_board else None) for r in (args[0].p1, args[0].p2)]
        if values != previous:
            print(args[3], values)
        previous = values


def extract() -> None:
    """既存OCRで試合全体を走査し、式の両辺と通常得点を保存する。"""
    data = json.loads((BASE/'video_80_g7_v3_2/events.json').read_text())
    # 描画区間より前の初手・得点0も読む。クリップ先頭の得点を0点扱いしない。
    start = next(updates(source(80)/'inputs.jsonl.gz', {GAME}))[3]
    end = data['match']['end_sec']
    cap = cv2.VideoCapture(str(video_path(80)))
    if not (TEMPLATES/'formula_mult.png').exists():
        raise FileNotFoundError('既存の得点テンプレートが必要です')
    ocr = ScoreOcr.load_default(TEMPLATES, enable_matmul_ncc=True)
    if not cap.isOpened():
        raise FileNotFoundError('元動画が必要です')
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(start*FPS))
    accumulators = [FormulaStepAccumulator(), FormulaStepAccumulator()]
    records, steps = [], []
    for index in range(round((end-start)*FPS)):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('動画が途中で終了')
        if index % STRIDE:
            continue
        stamp = (round(start*FPS)+index)/FPS
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        scores = ocr.read(frame)
        for side, name in enumerate(('1P', '2P')):
            formula = ocr.read_formula_side(frame, name)
            score = (scores.score_1p, scores.score_2p)[side]
            step = accumulators[side].update(stamp, formula, score_displayed=score is not None)
            records.append(dict(t=stamp, side=side, score=score, **asdict(formula)))
            if step:
                steps.append(dict(side=side, **asdict(step)))
        if index % (FPS*10) == 0:
            print('OCR', round(stamp, 2), len(steps), flush=True)
    cap.release()
    save(OUT/'ocr.json', dict(records=records, steps=steps, start=start, end=end))


if __name__ == '__main__':
    import sys
    if '--initial' in sys.argv:
        initial()
    else:
        extract() if '--extract' in sys.argv else inspect()
