"""式・得点と独立したSTABLE終了盤で採点。初回推定器へ戻さない。"""
from __future__ import annotations

import argparse
from collections import Counter
from scripts.exchange_v35_manifest import OUT, read, excluded
from scripts.exchange_review_io import save
from scripts.verify_exchange_counts_v33 import score, aggregate
from scripts.correct_exchange_counts_v34 import correct
from scripts.exchange_counts_v33_collect import ImageSources
from scripts.exchange_v35_capture import image_sources
from scripts.exchange_review_v32_data import video_path
from src.exchange_count_prediction import histogram, simulate
from src.exchange_count_ledger import gross

GAP_SEC, TOLERANCE, CONFIRM = 2., .1, 2
ALL_CLEAR_SCORE = 2100


def next_fire(inp: dict, obs: list[dict], end: float,
              step_times: list[float] | None = None) -> float:
    for r in obs:
        event = r['event']
        if (r['t'] > inp['input_sec'] and event and event['mechanism'] == 'formula_read'
                and event['chain_count'] == 1 and event['trigger_sec'] > inp['trigger_sec']+TOLERANCE):
            if step_times is None or any(abs(t-event['trigger_sec']) <= TOLERANCE for t in step_times):
                return event['trigger_sec']
    return end


def confirmed_score(records: list[dict], start: float, end: float) -> dict | None:
    pending, count = None, 0
    for row in records:
        if not start < row['t'] < end:
            continue
        value = row['score']
        count = count+1 if value is not None and value == pending else int(value is not None)
        pending = value
        if count >= CONFIRM:
            return row
    return None


def prior_all_clear(records: list[dict], steps: list[dict], first_time: float) -> bool:
    """直前の式の得点差で全消しの持越しを判定。今回獲得分と分離する。"""
    prior = [s for s in steps if s['t_sec'] < first_time-TOLERANCE]
    if not prior:
        return False
    step = prior[-1]
    before = confirmed_score(list(reversed(records)), float('-inf'), step['t_sec'])
    after = confirmed_score(records, step['t_sec'], first_time)
    return bool(before and after and after['score']-before['score']-step['left']*step['right'] == ALL_CLEAR_SCORE)


def truth(inp: dict, ocr: dict, obs: list[dict], native: object) -> dict:
    limit = next_fire(inp, obs, ocr['end'], [s['t_sec'] for s in ocr['steps'] if s['side'] == inp['side']])
    stages = [s for s in ocr['steps'] if s['side'] == inp['side']
              and inp['trigger_sec']-TOLERANCE <= s['t_sec'] < limit-TOLERANCE]
    steps = []
    for step in stages:
        if steps and step['t_sec']-steps[-1]['t_sec'] > GAP_SEC:
            break
        steps.append(step)
    if not steps:
        raise ValueError('初段式欠測')
    records = [r for r in ocr['records'] if r['side'] == inp['side']]
    reflected = [confirmed_score(records, s['t_sec'],
                 steps[i+1]['t_sec'] if i+1 < len(steps) else limit) for i, s in enumerate(steps)]
    if any(r is None for r in reflected):
        raise ValueError('段の得点反映未確認')
    scores, times = [s['left']*s['right'] for s in steps], [r['t'] for r in reflected]
    deltas = [reflected[i]['score']-(reflected[i-1]['score'] if i else inp['ledger']['score'])-s
              for i, s in enumerate(scores)]
    # 通常点と式が矛盾するラベルを黙って正解にしない。2100点は今回の全消し表示。
    score_valid = all(d in (0, 2100) for d in deltas)
    end_board = automatic_end(obs, times[-1], limit, inp['colors'], native)
    all_clear = prior_all_clear(records, [s for s in ocr['steps'] if s['side'] == inp['side']], steps[0]['t_sec'])
    sent = gross(scores, times, inp['ledger']['carry'], inp['ledger']['first_move'], all_clear=all_clear)[0]
    return dict(erased=[s['left']//10 for s in steps], scores=scores, sent=sent,
        remaining=histogram(end_board['board']) if end_board else None, end=end_board,
        steps=steps, score_times=times, score_deltas=deltas, score_valid=score_valid,
        limit=limit, all_clear_achieved=ALL_CLEAR_SCORE in deltas, all_clear_carried=all_clear,
        hidden_uncertain=bool(end_board and any(end_board['board'][6:12])))


def automatic_end(obs: list[dict], after: float, limit: float,
                  colors: list[int], native: object) -> dict | None:
    """保存認識の最初のSTABLE確定盤をRustで検査。画像補正器とは独立。"""
    for row in obs:
        board = row['board']
        if not after <= row['t'] < limit or row['state'] != 'STABLE' or board is None:
            continue
        if any(c not in (0, 9, *colors) for c in board) or simulate(board, native)['erased']:
            continue
        return dict(t=row['t'], board=board)
    return None


def classify(values: list[dict], actual: dict, measured: dict) -> dict:
    """盤面の神託がない全件では、出力候補の支持と順位を別々に記録する。"""
    result = {}
    for key, match in (('erased', 'event_stage_match'), ('sent', 'sent_match'), ('remaining', 'remaining_match')):
        if actual[key] is None:
            result[key] = '正解欠測'
        elif measured[match]:
            result[key] = '一致'
        else:
            rank = next((i+1 for i, v in enumerate(sorted(values, key=lambda x: -x['probability']))
                         if v[key] == actual[key]), None)
            result[key] = dict(category='順位' if rank else '候補にない', candidate_rank=rank,
                               hidden_unresolved=actual['hidden_uncertain'] if key == 'remaining' else False)
    return result


def score_event(row: dict, actual: dict, correction: dict) -> dict:
    variants, causes = {}, {}
    proxy = dict(actual, remaining=actual['remaining'] or [-1]*6)
    for name in ('v33', 'v34'):
        values = row[name]['variants']['board_candidates']
        variants[name] = score(values, proxy)
        causes[name] = classify(values, actual, variants[name])
    inp = row['inputs']
    return dict(video=inp['video'], game=inp['game'], event_id=inp['event_id'],
        input_sec=inp['input_sec'], side=inp['side'], actual=actual, variants=variants, causes=causes,
        correction=correction, correction_match=actual['remaining'] is not None and
        correction['remaining'] == actual['remaining'])


def run(video: int) -> None:
    import puyo_core
    entry = next(r for r in read(OUT/'manifest.json')['selected'] if r['video'] == video)
    images = image_sources(video_path(video))
    for game in entry['games']:
        folder = OUT/f'video_{video}/game_{game}'
        cached = {}
        if (folder/'scored.json').exists():
            previous = read(folder/'scored.json')
            if previous.get('boundary_version') == 3:
                continue
            old = previous.get('boundary_version', 1)
            if old >= 2:
                cached = {r['event_id']: r['correction'] for r in previous['rows']}
            (folder/'scored.json').rename(folder/f'scored_boundary_v{old}.json')
        ocr, obs = read(folder/f'holdout_{video}_ocr.json'), read(folder/'observations.json')
        rows, skipped = [], []
        for row in read(folder/'predictions.json')['rows']:
            inp = row['inputs']
            assert (game, inp['event_id']) not in excluded(video) and video != 80
            try:
                actual = truth(inp, ocr, obs[inp['side']], puyo_core)
            except ValueError as error:
                skipped.append(dict(event_id=inp['event_id'], reason=str(error)))
                continue
            correction = cached.get(inp['event_id'])
            if correction is None:
                correction = correct(inp, [r for r in obs[inp['side']] if r['t'] < actual['limit']],
                    [r for r in ocr['records'] if r['side'] == inp['side']], images, puyo_core)
            rows.append(score_event(row, actual, correction))
        save(folder/'scored.json', dict(rows=rows, skipped=skipped, boundary_version=3))
        print('採点完了', video, game, len(rows), flush=True)
    images.cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=int, required=True)
    run(parser.parse_args().video)
