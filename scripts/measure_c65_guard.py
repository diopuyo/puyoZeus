"""cycle65 対整合ガードの影子測定 (認識は本番構成のまま=フラグOFFで再生し、各呼出で ON の判断も併記)。

使い方:
  run     SOURCE   認識を再生し cycle65 の各呼出を logs/c65_guard/calls_SOURCE.jsonl へ記録
  window  START END on|off  q の窓を実フラグ ON/OFF で再生し 1P 確認盤面の指定セルを記録 (端から端の確認)
  analyze          calls_*.jsonl と logs/r1b/followup の生CNN/生HSV観測から OFF/ON の正誤を集計
真値 = 書込 0.1〜0.5 秒後の「生CNN==生HSV かつ有色」の最頻値 (診断 c65_truth と同一基準)。
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

OUT = Path('logs/c65_guard')
FOLLOWUP = Path('logs/r1b/followup')
TRUTH_FRAME_OFFSETS = range(3, 16)  # 30fps 換算で 0.1〜0.5 秒後
FPS_NORM = 30
ZENCHI_START_SEC = 2580.566
RECOGNITION_ARGS = ['--per-side-settled', '--sample-interval', '0', '--normalize-fps-30',
                    '--production-recognition', '--resize-1080p', '--no-render',
                    '--placement-signal-reconcile', '--early-fire-reaction',
                    '--no-score-lead-bias', '--no-pressure', '--counter-reach']
D = '/mnt/d/puyo_analyzer/videos/source/'
SOURCES = {
    'q_7gc4TgFig': dict(video=D + 'q_7gc4TgFig_first_0_900_20260925_v1.mp4', span=['--max-sec', '900']),
    'fcXG83vInDY': dict(video=D + 'fcXG83vInDY_first_0_900_20260925_v1.mp4', span=['--max-sec', '900']),
    'mia8KCjr52g': dict(video=D + 'mia8KCjr52g_first_0_900_20260925_v1.mp4', span=['--max-sec', '900']),
    'zenchi': dict(
        video='/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/frames/video_zenchi_c0BQoMJwwQU.mp4',
        span=['--start-sec', str(ZENCHI_START_SEC), '--end-sec', '3427.166', '--warmup-sec', '30.0']),
}


def install(log_path: Path, t_min: float) -> None:
    """infer_placement を差し替え、cycle65 呼出ごとに OFF の結果と ON の判断を記録する。"""
    from src import next_recolor_pair_guard as guard
    from src import recognition_pipeline as rp
    original = rp.infer_placement
    stream = log_path.open('w')

    def cells_of(board, cells):  # noqa: ANN001
        return None if board is None else [int(board._grid[r, c]) for r, c in cells]

    def wrapper(prev, cur, pair, *args, **kwargs):  # noqa: ANN001
        frame = sys._getframe(1)
        result = original(prev, cur, pair, *args, **kwargs)
        if 'falling_pair_b' not in frame.f_locals or prev is None or cur is None:
            return result
        t = frame.f_locals['ctx'].time_sec
        cells = [(int(r), int(c)) for r, c in zip(*((prev._grid == 0) & (cur._grid != 0)).nonzero())]
        if t < t_min:
            return result
        observed = cells_of(cur, cells)
        queue = [tuple(p) for p in frame.f_locals['prev_next_queue']]
        chosen, outcome = guard.select_recolor_pair(observed, queue)
        if outcome == guard.OUTCOME_KEPT_OBSERVED:
            on_cells = observed
        elif outcome == guard.OUTCOME_ACCEPTED:
            on_cells = cells_of(result, cells)
        else:
            on_cells = cells_of(original(prev, cur, chosen, *args, **kwargs), cells)
        stream.write(json.dumps(dict(
            t=round(t, 3), side=frame.f_locals['side'], cells=cells, observed=observed,
            pair=list(pair), queue=queue, outcome=outcome, chosen=chosen,
            off=cells_of(result, cells), on=on_cells)) + '\n')
        stream.flush()
        return result

    rp.infer_placement = wrapper


def run(source: str) -> None:
    """指定動画の認識を再生 (--no-render)。"""
    spec = SOURCES[source]
    OUT.mkdir(parents=True, exist_ok=True)
    t_min = ZENCHI_START_SEC if source == 'zenchi' else 0.0
    sys.argv = ['visualize_advantage_overlay', '--video', spec['video'],
                '--out', str(OUT / f'{source}_scan.mp4')] + spec['span'] + RECOGNITION_ARGS
    import random
    import numpy as np
    import torch
    random.seed(20260926)
    np.random.seed(20260926)
    torch.manual_seed(20260926)
    install(OUT / f'calls_{source}.jsonl', t_min)
    from scripts.visualize_advantage_overlay import main
    main()


WINDOW_CELLS = [(5, 2), (6, 2), (1, 5), (2, 5), (3, 5)]  # 診断書の q第14試合 5 セル
WINDOW_REPORT = (883.0, 884.5)


def window(start: float, end: float, mode: str) -> None:
    """q の窓を実フラグで再生し、1P 確認盤面の 5 セルを時刻順に記録する。"""
    from src import recognition_pipeline as rp
    OUT.mkdir(parents=True, exist_ok=True)
    stream = (OUT / f'window_{mode}.jsonl').open('w')
    update = rp.RecognitionPipeline.update

    def wrapped(self, frame_idx, time_sec, frame, *args, **kwargs):  # noqa: ANN001
        result = update(self, frame_idx, time_sec, frame, *args, **kwargs)
        board = result.p1.confirmed_board
        if WINDOW_REPORT[0] <= time_sec <= WINDOW_REPORT[1] and board is not None:
            stream.write(json.dumps(dict(
                t=round(time_sec, 3), state=result.p1.state.name,
                cells=[int(board._grid[r, c]) for r, c in WINDOW_CELLS])) + '\n')
            stream.flush()
        return result

    rp.RecognitionPipeline.update = wrapped
    spec = SOURCES['q_7gc4TgFig']
    extra = ['--next-recolor-pair-guard'] if mode == 'on' else []
    sys.argv = ['visualize_advantage_overlay', '--video', spec['video'],
                '--out', str(OUT / f'window_{mode}.mp4'),
                '--start-sec', str(start), '--end-sec', str(end)] + RECOGNITION_ARGS + extra
    from scripts.visualize_advantage_overlay import main
    main()


def load_obs(source: str, keys: set[int]) -> dict[int, list]:
    """必要フレームだけ観測を読む (キー = round(t*30))。"""
    found: dict[int, list] = {}
    with gzip.open(FOLLOWUP / f'{source}.observations.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = round(row['t'] * FPS_NORM)
            if key in keys:
                found[key] = row['sides']
    return found


def truth_of(observations: dict, key: int, side: str, r: int, c: int) -> tuple[int | None, int]:
    """書込セルの真値 (最頻) と得票数。"""
    votes: collections.Counter = collections.Counter()
    for offset in TRUTH_FRAME_OFFSETS:
        sides = observations.get(key + offset)
        if not sides:
            continue
        cell = sides[0 if side == '1P' else 1]
        cnn, hsv = cell['cnn'][r][c], cell['hsv'][r][c]
        if cnn == hsv and cnn not in (0, 10):
            votes[cnn] += 1
    if not votes:
        return None, 0
    return votes.most_common(1)[0][0], sum(votes.values())


def analyze_source(source: str) -> dict:
    """1 動画の集計。"""
    path = OUT / f'calls_{source}.jsonl'
    calls = [json.loads(line) for line in path.open()]
    keys = {round(call['t'] * FPS_NORM) + d for call in calls for d in TRUTH_FRAME_OFFSETS}
    obs = load_obs(source, keys)
    tot: collections.Counter = collections.Counter()
    worse: list = []
    for call in calls:
        tot['calls'] += 1
        tot['outcome_' + call['outcome']] += 1
        key = round(call['t'] * FPS_NORM)
        off_cells = call['observed'] if call['off'] is None else call['off']  # 推論なし=観測色のまま
        on_cells = call['observed'] if call['on'] is None else call['on']
        tot['infer_returned_none'] += call['off'] is None
        for (r, c), observed, off, on in zip(call['cells'], call['observed'], off_cells, on_cells):
            truth, _ = truth_of(obs, key, call['side'], r, c)
            tot['cells'] += 1
            tot['recolored_by_off'] += off != observed
            if truth is None:
                tot['no_truth'] += 1
                continue
            tot['truth_cells'] += 1
            tot['observed_right'] += observed == truth
            tot['off_right'] += off == truth
            tot['on_right'] += on == truth
            if on != truth and off == truth:
                worse.append(dict(t=call['t'], side=call['side'], cell=[r, c], observed=observed,
                                  off=off, on=on, truth=truth, outcome=call['outcome']))
    tot['on_worse_cells'] = len(worse)
    return dict(source=source, totals=dict(tot), on_worse=worse)


def analyze() -> None:
    """全動画を集計して JSON へ。"""
    finished = [s for s in SOURCES if '[done]' in (OUT / f'run_{s}.log').read_text(errors='ignore')]
    results = [analyze_source(s) for s in finished]
    (OUT / 'measure.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))
    for result in results:
        print(result['source'], result['totals'])


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(sys.argv[2])
    elif sys.argv[1] == 'window':
        window(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
    else:
        analyze()
