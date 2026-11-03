"""画像あり5動画の設定外発火を固定規則で採取。式・終了盤は採点側へ隔離する。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
import cv2
from scripts.audit_exchange_counts_v33 import TEMPLATES
from scripts.audit_exchange_cells_v34 import OUT
from scripts.exchange_review_v32_data import source, fire_rows, video_path
from scripts.exchange_counts_v33_collect import ImageSources, extract_sources
from scripts.collect_exchange_history_v34 import observe, HISTORY_SIZE
from scripts.exchange_review_io import save
from scripts.replay_exchange_projection_shadow import updates
from src.score_ocr import ScoreOcr, FormulaStepAccumulator

VIDEOS, INDEX, STRIDE, FPS = (32, 34, 51, 61, 92), 2, 2, 60


def selected(video: int) -> tuple[dict, float]:
    groups: dict = defaultdict(list)
    fires = fire_rows(source(video))
    for fire in fires:
        if len(fire['exchange_input']['colors']) == 4:
            groups[fire['game_idx']].append(fire)
    group = next(g for _, g in sorted(groups.items()) if len(g) > INDEX)
    targets = {f['t_sec']: f for f in group[INDEX:]}
    fire, end = None, None
    for args in updates(source(video)/'inputs.jsonl.gz', {group[0]['game_idx']}):
        end = args[3]
        item = targets.get(args[3])
        if item is None:
            continue
        raw = (args[0].p1, args[0].p2)[item['exchange_input']['attacker']]
        event = raw.chain_event
        if event is None or event.mechanism != 'formula_read' or event.chain_count != 1:
            continue
        if fire is None:
            fire = item
        elif item['attacker'] == fire['attacker']:
            return fire, item['trigger_sec']
    if fire is None:
        raise ValueError('設定外の独立初段通知がありません')
    return fire, end


def observation(raw: object, stamp: float) -> dict:
    event = raw.chain_event
    return dict(t=stamp, state=raw.state.name, score=raw.score,
        next=list(raw.next_pair) if raw.next_pair else None,
        board=raw.confirmed_board._grid.ravel().tolist() if raw.confirmed_board is not None else None,
        event={k: getattr(event, k) for k in ('chain_count', 'total_score', 'trigger_sec', 'mechanism')} if event else None)


def collect_input(video: int, fire: dict, end: float) -> dict:
    images, histories, observations = ImageSources(video_path(video)), [[], []], []
    result, start = {}, None
    side = fire['exchange_input']['attacker']
    for args in updates(source(video)/'inputs.jsonl.gz', {fire['game_idx']}):
        stamp, raw = args[3], (args[0].p1, args[0].p2)[side]
        start = stamp if start is None else start
        if stamp > end:
            break
        for s, player in enumerate((args[0].p1, args[0].p2)):
            observe(histories, player, s, stamp)
        observations.append(observation(raw, stamp))
        if stamp == fire['t_sec']:
            hs = [h for h in histories[side] if h['t'] <= fire['trigger_sec']]
            from collections import Counter
            result = dict(video=video, game=fire['game_idx'], fire=fire, next=observations[-1]['next'],
                sources=extract_sources(raw, stamp, fire['trigger_sec'], side, images),
                history=dict(history=hs[-HISTORY_SIZE:], history_latest=max((h['t'] for h in hs), default=None),
                    hidden_history=[dict(Counter(h['board'][c] for h in hs)) for c in range(6)]))
    images.cap.release()
    save(OUT/f'holdout_{video}_input.json', result)
    save(OUT/f'holdout_{video}_observations.json', observations)
    return dict(start=start, end=end)


def extract_ocr(video: int, start: float, end: float) -> None:
    cap = cv2.VideoCapture(str(video_path(video)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ocr = ScoreOcr.load_default(TEMPLATES, enable_matmul_ncc=True)
    accumulators = [FormulaStepAccumulator(), FormulaStepAccumulator()]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start*fps))
    records, steps = [], []
    for frame_index in range(int(start*fps), int(end*fps)+1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('検証動画のフレーム欠測')
        if frame_index % max(1, round(fps/(FPS/STRIDE))):
            continue
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        scores = ocr.read(frame)
        stamp = frame_index/fps
        for side, name in enumerate(('1P', '2P')):
            formula = ocr.read_formula_side(frame, name)
            score = (scores.score_1p, scores.score_2p)[side]
            step = accumulators[side].update(stamp, formula, score_displayed=score is not None)
            records.append(dict(t=stamp, side=side, score=score, **asdict(formula)))
            if step:
                steps.append(dict(side=side, **asdict(step)))
    cap.release()
    save(OUT/f'holdout_{video}_ocr.json', dict(records=records, steps=steps, start=start, end=end,
                                          fps=fps, selection_version=2))


def main() -> None:
    for video in VIDEOS:
        path = OUT/f'holdout_{video}_ocr.json'
        if path.exists() and json.loads(path.read_text()).get('selection_version') == 2:
            continue
        fire, end = selected(video)
        window = collect_input(video, fire, end)
        print('検証採取', video, fire['event_id'], window, flush=True)
        extract_ocr(video, **window)
        print('検証OCR完了', video, flush=True)


if __name__ == '__main__':
    main()
