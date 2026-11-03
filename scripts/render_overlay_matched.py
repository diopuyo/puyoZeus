"""c68の同時刻見本を生成し、認識イベント基準の連鎖表示差を保存する。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from functools import lru_cache

VIDEO = 'b3_c68_DBfHe3dA244'
SOURCE_SHA = '67149c6b9052cfffc1a5be4a17dcd30efde176bb7b56fe2a327ebf230846d194'
SIDES = ('p1', 'p2')
FPS = 60
MIN_GROUP = 4
MAX_COLORS = 4
MAX_CHAIN = 19
SCORE_FACTOR = 10
BOARD_CELLS = 78
FORMULA_CONFIRM_GAP = .05
SCREEN_CHAIN_GAP = 2.5
DIAGNOSTIC_FRAMES = (98600, 110994, 148146)
LABELS = {'three': '3連鎖の場面', 'five': '5連鎖以上の場面', 'exchange': '撃ち合い'}
ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def verify_source(source: Path) -> None:
    """再収集記録と同一の動画だけを許可する。"""
    with source.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != SOURCE_SHA:
        raise ValueError(f'元動画SHA不一致: {actual}')


@lru_cache(maxsize=None)
def group_bonuses(erased: int) -> frozenset[tuple[int, int]]:
    """消去数の全分割から（連結ボーナス合計、群数）を列挙する。"""
    from src.scoring import connection_bonus
    if erased == 0:
        return frozenset({(0, 0)})
    return frozenset((bonus+connection_bonus(size), groups+1)
        for size in range(MIN_GROUP, erased+1)
        for bonus, groups in group_bonuses(erased-size))


@lru_cache(maxsize=None)
def formula_number(left: int, right: int) -> int | None:
    """式の左右から段数を逆算。一意にならない式を推測で埋めない。"""
    from src.scoring import chain_power, color_bonus
    if left % SCORE_FACTOR or not MIN_GROUP*SCORE_FACTOR <= left <= BOARD_CELLS*SCORE_FACTOR:
        return None
    bonuses = {bonus+color_bonus(colors) for bonus, groups in group_bonuses(left//SCORE_FACTOR)
               for colors in range(1, min(groups, MAX_COLORS)+1)}
    candidates = [n for n in range(1, MAX_CHAIN+1)
                  if any(max(1, chain_power(n)+bonus) == right for bonus in bonuses)]
    return candidates[0] if len(candidates) == 1 else None


def read_observations(record: Path) -> Any:
    """画面認識の連鎖数を取り出す。式の観測順序を正解にはしない。"""
    import pandas as pd
    from src.b3_live_display import B3Display
    display = B3Display()
    values = []
    with gzip.open(record/'observations.jsonl.gz', 'rt', encoding='utf-8') as stream:
        header = json.loads(next(stream))
        assert header['video_sha256'] == SOURCE_SHA
        for line in stream:
            frame = json.loads(line)
            if frame.get('kind') != 'frame':
                continue
            display.boundary(frame['game_idx'])
            display.chain.observe(frame)
            for side, raw, tracker in zip(SIDES, frame['sides'], display.chain.sides):
                chain = raw.get('chain') or {}
                formula = raw.get('formula') or {}
                left, right = formula.get('left'), formula.get('right')
                number = formula_number(left, right) if formula.get('valid') and left and right else None
                values.append(dict(frame=frame['frame'], t_sec=frame['t_sec'],
                    game=frame['game_idx'], side=side, state=raw['state'],
                    match_active=frame['match_active'], match_end_locked=frame['match_end_locked'],
                    actual=chain.get('chain_count'), mechanism=chain.get('mechanism'),
                    trigger=chain.get('trigger_sec'), screen_step=number,
                    display_step=tracker.value, display_observed_at=tracker.observed_at,
                    formula_left=left, formula_right=right))
    return pd.DataFrame(values)


def quantiles(values: Any) -> dict:
    import numpy as np
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    return dict(n=len(data), **{key: float(value) for key, value in
        zip(('min', 'p50', 'p90', 'p95', 'p99', 'max'),
            np.quantile(data, [0, .5, .9, .95, .99, 1]))}) if len(data) else dict(n=0)


def compare_timeline(rows: Any, obs: Any, out: Path) -> tuple[Any, dict]:
    """各認識時刻に、それ以前の最新公開行を保持して比較する。"""
    import pandas as pd
    from src.b3_eval_adapter import MAX_HOLD_SEC
    parts = []
    for side in SIDES:
        cols = ['frame_idx', 'game_idx', 't_sec', 'chain_active_'+side,
                'chain_n_steps_'+side, 'chain_pred_l_'+side]
        right = rows[cols].rename(columns=dict(zip(cols,
            ['source_frame', 'game', 'source_t', 'active', 'displayed', 'predicted'])))
        right['game'] = right.game.astype('int64')
        left = obs[obs.side.eq(side)].sort_values('t_sec')
        joined = pd.merge_asof(left, right.sort_values('source_t'),
            left_on='t_sec', right_on='source_t', by='game', direction='backward')
        joined['age_sec'] = joined.t_sec-joined.source_t
        joined['b3_displayed'] = joined.displayed
        joined['displayed'] = joined.display_step
        expired = joined.age_sec.gt(MAX_HOLD_SEC)
        joined.loc[expired, ['active', 'displayed', 'predicted']] = float('nan')
        joined['difference'] = joined.displayed-joined.actual
        parts.append(joined)
    timeline = pd.concat(parts, ignore_index=True).sort_values(['frame', 'side'])
    timeline.to_csv(out/'timeline.csv', index=False)
    chain = timeline[timeline.state.eq('CHAIN') & timeline.actual.gt(0)]
    valid = chain[chain.active.eq(1) & chain.displayed.notna()]
    delta = valid.difference.value_counts().sort_index()
    summary = dict(recognition_chain_samples=len(chain), comparable=len(valid),
        chain_state_without_count=int((timeline.state.eq('CHAIN') & timeline.actual.isna()).sum()),
        no_display=len(chain)-len(valid), equal=int(valid.difference.eq(0).sum()),
        mismatch=int(valid.difference.ne(0).sum()),
        difference_histogram={str(int(k)): int(v) for k, v in delta.items()},
        source_age_sec=quantiles(valid.age_sec),
        reference='診断専用: chain_countは全体予測・累積持越しを含むため現在段の正解ではない。主集計はscreen')
    return timeline, summary


def episode_audit(timeline: Any, record: Path, out: Path) -> dict:
    """全発火区間を検査。完了・打切りを分離し、初回と最終予想を別集計する。"""
    import pandas as pd
    events = [json.loads(s) for s in (record/'events.jsonl').read_text().splitlines()]
    complete = {e['fire_id']: e for e in events if e['kind'] == 'fire_complete'}
    records, delays = [], []
    for start in (e for e in events if e['kind'] == 'fire_start'):
        end = complete.get(start['fire_id'])
        if end is None:
            records.append(dict(fire_id=start['fire_id'], status='censored'))
            continue
        side = 'p1' if start['side'] == '1P' else 'p2'
        window = timeline[timeline.side.eq(side) & timeline.game.eq(start['game_idx']) &
            timeline.t_sec.between(start['trigger_sec'], end['observed_at_sec'])]
        actual = window.loc[window.actual.gt(0), 'actual']
        predictions = window.loc[window.active.eq(1), 'predicted']
        predicted = predictions.dropna()
        final = int(actual.max()) if len(actual) else None
        item = dict(fire_id=start['fire_id'], side=side, start=start['trigger_sec'],
            end=end['observed_at_sec'], status='complete', actual=final,
            formula_steps=len(end['steps']), first=float(predicted.iloc[0]) if len(predicted) else None,
            last=float(predicted.iloc[-1]) if len(predicted) else None,
            terminal=float(predictions.iloc[-1]) if len(predictions) and pd.notna(predictions.iloc[-1]) else None)
        records.append(item)
        delays.extend(step_delays(window, start['fire_id']))
    pd.DataFrame(records).to_csv(out/'episodes.csv', index=False)
    pd.DataFrame(delays).to_csv(out/'step_delays.csv', index=False)
    done = [r for r in records if r['status'] == 'complete' and r['actual'] is not None]
    result = dict(starts=len(records), complete=len(done), completed_events=len(complete),
        censored=sum(r['status'] == 'censored' for r in records),
        unmeasurable_complete=sum(r['status'] == 'complete' and r['actual'] is None for r in records))
    for key in ('first', 'last', 'terminal'):
        have = [r for r in done if r[key] is not None]
        result[key] = dict(predicted=len(have), hit=sum(r[key] == r['actual'] for r in have),
            missing=len(done)-len(have), error_histogram=pd.Series(
                [int(r[key]-r['actual']) for r in have]).value_counts().sort_index().to_dict())
    result['step_delay_sec'] = quantiles([d['delay'] for d in delays if d['delay'] is not None])
    result['steps_never_reached'] = sum(d['delay'] is None for d in delays)
    result['prediction_policy'] = 'first/last=初回/最終の有限予想、terminal=終了直前の予想（撤回後は欠測）'
    return result


def screen_audit(timeline: Any, record: Path, out: Path) -> dict:
    """独立に逆算した画面の段数だけで一致率を出し、内部イベント差と分離する。"""
    import pandas as pd
    accepted = []
    for _, group in timeline.groupby(['game', 'side']):
        same = group.screen_step.eq(group.screen_step.shift()) & group.screen_step.notna()
        same &= group.formula_left.eq(group.formula_left.shift()) & group.formula_right.eq(group.formula_right.shift())
        same &= group.t_sec.diff().le(FORMULA_CONFIRM_GAP)
        # 2観測以上連続した式だけ採用し、最初の観測も残す。
        accepted.append(group[same | same.shift(-1, fill_value=False)])
    visible = pd.concat(accepted).sort_values(['frame', 'side']).reset_index(drop=True)
    visible['stage_age_sec'] = 0.0
    for _, group in visible.groupby(['game', 'side']):
        reset = group.screen_step.ne(group.screen_step.shift()) | group.t_sec.diff().gt(SCREEN_CHAIN_GAP)
        onset = group.t_sec.where(reset).ffill()
        visible.loc[group.index, 'stage_age_sec'] = group.t_sec-onset
    visible['difference'] = visible.displayed-visible.screen_step
    visible.to_csv(out/'screen_timeline.csv', index=False)
    valid = visible[visible.active.eq(1) & visible.displayed.notna()]
    hist = valid.difference.value_counts().sort_index()
    result = dict(visible_samples=len(visible), comparable=len(valid),
        no_display=len(visible)-len(valid), mismatch=int(valid.difference.ne(0).sum()),
        mismatch_after_half_sec=int((valid.stage_age_sec.ge(.5) & valid.difference.ne(0)).sum()),
        difference_histogram={str(int(k)): int(v) for k, v in hist.items()},
        source_age_sec=quantiles(valid.age_sec),
        reference='同一動画の掛け算式の左右を公式得点表で逆算。一意かつ同一式2観測以上のみ。非表示・曖昧は未判定')
    result['episodes'] = screen_episodes(visible, timeline, record, out)
    result['display_coverage'] = display_coverage(visible, out)
    result['publisher_model'] = publisher_model(visible, timeline, out)
    return result


def display_coverage(visible: Any, out: Path) -> dict:
    """旧B3活動フラグに依存しない表示欠測と残差分類も隠さず保存する。"""
    known = visible[visible.displayed.notna()]
    wrong = known[known.difference.ne(0)].copy()
    wrong['cause'] = '式の連続確認待ち・欠読保持'
    wrong.loc[wrong.stage_age_sec.ge(.5), 'cause'] = '段開始0.5秒後の残差（要確認）'
    wrong.to_csv(out/'display_mismatches.csv', index=False)
    missing = visible[visible.displayed.isna()].copy()
    missing['cause'] = '初回式の連続確認待ち（同じ段番号の別連鎖を含む）'
    terminal = missing.match_end_locked | ~missing.match_active
    missing.loc[terminal, 'cause'] = '試合終了ロック・試合外による非表示'
    missing.to_csv(out/'display_missing.csv', index=False)
    late = visible[visible.stage_age_sec.ge(.5)]
    return dict(comparable=len(known), no_display=len(visible)-len(known),
        mismatch=len(wrong), after_half_sec=len(late),
        missing_after_half_sec=int(late.displayed.isna().sum()),
        mismatch_after_half_sec=int((late.displayed.notna() & late.difference.ne(0)).sum()),
        causes=wrong.cause.value_counts().to_dict(),
        missing_causes=missing.cause.value_counts().to_dict(),
        missing_after_half_sec_causes=missing[missing.stage_age_sec.ge(.5)].cause.value_counts().to_dict())


def publisher_model(visible: Any, timeline: Any, out: Path) -> dict:
    """0秒位相・計算待ちゼロの0.5秒周期モデル。実機測定とは区別する。"""
    import numpy as np
    import pandas as pd
    from src.phase_j.live_publish import PUBLISH_PERIOD_SEC
    period_frames = round(PUBLISH_PERIOD_SEC*FPS)
    parts = []
    for side in SIDES:
        left = visible[visible.side.eq(side)].copy()
        left['publish_frame'] = left.frame//period_frames*period_frames
        right = timeline[timeline.side.eq(side)][['frame', 'game', 'displayed', 'active']]
        right = right.rename(columns=dict(frame='publish_frame', displayed='published_step', active='published_active'))
        joined = pd.merge_asof(left.sort_values('publish_frame'), right.sort_values('publish_frame'),
            on='publish_frame', by='game', direction='backward')
        parts.append(joined)
    data = pd.concat(parts)
    valid = data[data.published_active.eq(1) & data.published_step.notna()]
    data.to_csv(out/'publisher_model.csv', index=False)
    onsets = visible.sort_values(['side', 'frame'])
    onsets = onsets[onsets.screen_step.ne(onsets.groupby('side').screen_step.shift())]
    wait = (np.ceil(onsets.frame/period_frames)*period_frames-onsets.frame)/FPS
    return dict(period_sec=PUBLISH_PERIOD_SEC, comparable=len(valid),
        mismatch=int(valid.published_step.ne(valid.screen_step).sum()),
        mismatch_after_half_sec=int((valid.stage_age_sec.ge(.5) &
                                    valid.published_step.ne(valid.screen_step)).sum()),
        mismatch_after_confirmation_and_period=int((valid.stage_age_sec.ge(PUBLISH_PERIOD_SEC+FORMULA_CONFIRM_GAP) &
                                                   valid.published_step.ne(valid.screen_step)).sum()),
        additional_wait_sec=quantiles(wait),
        note='0秒位相・処理待ちゼロの周期モデル。PID2088実測ではない。非同期処理待ち・SSE通信は未計測')


def screen_chains(visible: Any, record: Path) -> list[dict]:
    """式の段数の増加を接続し、fire_completeの途中分割を連鎖終了と誤認しない。"""
    from bisect import bisect_right
    events = [json.loads(s) for s in (record/'events.jsonl').read_text().splitlines()]
    chains = []
    for (game, side), group in visible.groupby(['game', 'side']):
        name = '1P' if side == 'p1' else '2P'
        placements = sorted(e['observed_at_sec'] for e in events if
            e['kind'] == 'placement_signal' and e['game_idx'] == game and e['side'] == name)
        current = None
        for row in group.sort_values('t_sec').itertuples():
            number = int(row.screen_step)
            previous = current['samples'][-1] if current else None
            placed = previous and bisect_right(placements, row.t_sec) > bisect_right(placements, previous[0])
            split = previous and (number < previous[1] or row.t_sec-previous[0] > SCREEN_CHAIN_GAP
                                 or (number == previous[1] == 1 and placed))
            if current is None or split:
                current = dict(game=int(game), side=side, samples=[])
                chains.append(current)
            current['samples'].append((row.t_sec, number))
    return chains


def screen_episodes(visible: Any, timeline: Any, record: Path, out: Path) -> dict:
    """画面式の連続1～Nと終了後の静穏観測を満たす連鎖だけ最終段数を採点する。"""
    import pandas as pd
    records, delays = [], []
    for index, chain in enumerate(screen_chains(visible, record)):
        samples = chain['samples']
        start, end = samples[0][0], samples[-1][0]
        sequence = [v for i, (_, v) in enumerate(samples) if i == 0 or v != samples[i-1][1]]
        side = timeline[timeline.side.eq(chain['side']) & timeline.game.eq(chain['game'])]
        quiet = side[side.t_sec.between(end, end+SCREEN_CHAIN_GAP) &
                     side.state.isin(['STABLE', 'TSUMO_FALL', 'OJAMA_FALL'])]
        complete = sequence == list(range(1, max(sequence)+1)) and len(quiet) >= 2
        actual = max(sequence) if complete else None
        window = side[side.t_sec.between(start-FORMULA_CONFIRM_GAP, end)]
        preds = window.loc[window.active.eq(1), 'predicted']
        finite = preds.dropna()
        records.append(dict(chain_id=index, side=chain['side'], start=start, end=end,
            actual=actual, sequence=str(sequence), complete=complete,
            first=float(finite.iloc[0]) if len(finite) else None,
            terminal=float(preds.iloc[-1]) if len(preds) and pd.notna(preds.iloc[-1]) else None))
        if complete:
            for number in sequence:
                stamp = next(t for t, n in samples if n == number)
                reached = window[window.active.eq(1) & window.displayed.eq(number)]
                delay = float(reached.t_sec.min()-stamp) if len(reached) else None
                delays.append(dict(chain_id=index, step=number, actual_t=stamp, delay=delay))
    pd.DataFrame(records).to_csv(out/'screen_episodes.csv', index=False)
    pd.DataFrame(delays).to_csv(out/'screen_step_delays.csv', index=False)
    return screen_scores(records, delays)


def screen_scores(records: list[dict], delays: list[dict]) -> dict:
    import pandas as pd
    known = [r for r in records if r['actual'] is not None]
    measured = [r['delay'] for r in delays if r['delay'] is not None]
    summary = dict(screen_intervals=len(records), evaluable=len(known), unknown=len(records)-len(known),
        delay_sec=quantiles(measured), nonnegative_delay_sec=quantiles([v for v in measured if v >= -1e-6]),
        ahead=sum(v < -1e-6 for v in measured), never_reached=sum(r['delay'] is None for r in delays))
    for key in ('first', 'terminal'):
        have = [r for r in known if r[key] is not None]
        summary[key] = dict(predicted=len(have), hit=sum(r[key] == r['actual'] for r in have),
            missing=len(known)-len(have), error_histogram=pd.Series(
                [int(r[key]-r['actual']) for r in have]).value_counts().sort_index().to_dict())
    return summary


def step_delays(window: Any, fire_id: str) -> list[dict]:
    """認識が各段に到達した時刻と、公開段数が到達した時刻の符号付き差。"""
    values = []
    observed = window[window.state.eq('CHAIN') & window.actual.gt(0)]
    for number, group in observed.groupby('actual'):
        reached = window[window.active.eq(1) & window.displayed.ge(number)]
        first = float(group.t_sec.min())
        delay = float(reached.t_sec.min()-first) if len(reached) else None
        values.append(dict(fire_id=fire_id, step=int(number), actual_t=first, delay=delay))
    return values


def replay_public(rows: Any, args: argparse.Namespace) -> None:
    """v7評価行を製品の推論→平滑化→公開DTO経路へ時刻順に再生する。"""
    from scripts.check_overlay_b3_contents import ASSETS, live_wrapper, publish
    from src.b3_realtime import RealtimeRows
    from src.b3_timing import TimingTable
    from src.b3_train_model import load_model
    from src.phase_j.live_publish import initial_snapshot
    from src.b3_source import iter_frames, resolve_attempt
    table = TimingTable.from_json(args.table.read_text())
    live = live_wrapper(RealtimeRows(table, VIDEO, measure_timing=False), load_model(args.model))
    initial = initial_snapshot(ASSETS)
    frames = iter(iter_frames(resolve_attempt(VIDEO, args.records).record))
    observation = next(frames, None)
    with (args.output/'snapshots.jsonl').open('w', encoding='utf-8') as output:
        for index, row in enumerate(rows.to_dict('records')):
            while observation is not None and observation['frame'] <= row['frame_idx']:
                live.display.boundary(observation['game_idx'])
                live.display.chain.observe(observation)
                observation = next(frames, None)
            notice = SimpleNamespace(frame=int(row['frame_idx']), t_sec=row['t_sec'], captured_at=0.)
            live._predict(row, notice)
            payload = publish(live, row, initial)
            verify_display_payload(payload, live.display.chain)
            output.write(json.dumps(payload, ensure_ascii=False, allow_nan=False)+'\n')
            if index % 2000 == 0:
                print(f'公開経路照合 {index}/{len(rows)}', flush=True)


def verify_display_payload(payload: dict, chain: Any) -> None:
    """最新観測の表示段と公開DTOを照合し、B3基準時刻との分離も確認する。"""
    contents = payload['evaluations']['b3_contents']
    assert contents['chain_display_frame'] == chain.frame
    assert contents['chain_display_t_sec'] == chain.t_sec
    for side, tracker in zip(SIDES, chain.sides):
        assert contents[side]['chain_display_step'] == tracker.value
        assert contents[side]['chain_display_observed_at_sec'] == tracker.observed_at


def prepare(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.b3_source import resolve_attempt
    verify_source(args.source)
    rows = pd.read_parquet(args.rows)
    rows = rows[rows.orientation.eq(0)].sort_values('frame_idx')
    record = resolve_attempt(VIDEO, args.records).record
    timeline, result = compare_timeline(rows, read_observations(record), args.output)
    result['episodes'] = episode_audit(timeline, record, args.output)
    result['screen'] = screen_audit(timeline, record, args.output)
    result.update(rows=len(rows), source_sha256=SOURCE_SHA,
        primary_metric='screen（画面の式から一意に逆算できた観測のみ）',
        replay='v7保存評価行→LiveB3._predict→B3Display→decorate→result_snapshot。非同期実時間遅延は対象外')
    write_json(args.output/'audit.json', result)
    if args.mode == 'prepare':
        replay_public(rows, args)
        result['public_differences'] = 0
        write_json(args.output/'public_parity.json', dict(rows=len(rows), differences=0,
            schema_valid=True, path=result['replay'], model=str(args.model)))
    elif (args.output/'public_parity.json').exists():
        parity = json.loads((args.output/'public_parity.json').read_text(encoding='utf-8'))
        result['public_differences'] = parity.get('differences', parity.get('public_differences'))
    write_json(args.output/'audit.json', result)


def choose_scenes(rows: list[dict], timeline: Any) -> dict[str, dict]:
    """真の時刻対応を満たす認識フレームと、同フレームの公開行を選ぶ。"""
    picked = {}
    observations = {(int(r.frame), r.side): r for r in timeline.itertuples()}
    for row in rows:
        c = row['evaluations']['b3_contents']
        raw = [observations[(c['frame'], side)] for side in SIDES]
        active = [r.state == 'CHAIN' and r.screen_step > 0 for r in raw]
        if any(a and r.screen_step == 3 for a, r in zip(active, raw)):
            picked.setdefault('three', row)
        if any(a and r.screen_step >= 5 for a, r in zip(active, raw)):
            picked.setdefault('five', row)
        if all(active) and all(c[s]['chain_active'] for s in SIDES):
            picked.setdefault('exchange', row)
    assert set(picked) == set(LABELS), picked.keys()
    return {key: picked[key] for key in LABELS}


def extract_frame(capture: Any, row: dict, target: Path) -> None:
    import cv2
    c = row['evaluations']['b3_contents']
    assert abs(c['frame']/FPS-c['t_sec']) < 1e-6
    capture.set(cv2.CAP_PROP_POS_FRAMES, c['frame'])
    ok, frame = capture.read()
    assert ok and round(capture.get(cv2.CAP_PROP_POS_FRAMES))-1 == c['frame']
    assert cv2.imwrite(str(target), frame)


def compose(page: Any, frame: Path, target: Path, stamp: float) -> Any:
    import io
    from PIL import Image, ImageDraw
    from scripts.render_overlay_design import SIZE, JPEG_QUALITY, font
    game = page.evaluate('OverlayOptions.geometry(overlayOptions)')
    base = Image.new('RGBA', SIZE, '#0b1423')
    source = Image.open(frame).convert('RGBA').resize((game['width'], game['height']))
    base.paste(source, (game['x'], game['y']))
    overlay = Image.open(io.BytesIO(page.screenshot(omit_background=True))).convert('RGBA')
    result = Image.alpha_composite(base, overlay).convert('RGB')
    draw = ImageDraw.Draw(result)
    draw.rectangle((8, 8, 1230, 49), fill='#0b1423')
    draw.text((20, 12), f'画面と数値は同じ瞬間 (c68 {stamp:.3f}秒)', font=font(23), fill='#ffda9a')
    result.save(target, quality=JPEG_QUALITY, subsampling=0)
    return result


def mobile(key: str, pictures: list, details: list, target: Path) -> None:
    from PIL import Image, ImageDraw
    from scripts.render_overlay_design import font, JPEG_QUALITY
    width, gap, header, label = 1080, 32, 190, 70
    thumbs = []
    for caption, image in pictures:
        copy = image.copy()
        copy.thumbnail((width, width*9//16), Image.Resampling.LANCZOS)
        thumbs.append((caption, copy))
    height = header+sum(im.height+label for _, im in thumbs)+label+sum(im.height+gap for im in details)
    sheet = Image.new('RGB', (width, height), '#0b1423')
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 20), f'{key} ｜ 同じ瞬間の見本', font=font(37), fill='white')
    draw.text((gap, 80), '画面と数値は同じ瞬間（c68・各画像に秒数）', font=font(27), fill='#c4d4e8')
    draw.text((gap, 126), '連鎖の進みは画面の式を読取。予想連鎖は従来どおり。', font=font(25), fill='#ffda9a')
    y = header
    for caption, image in thumbs:
        draw.text((gap, y), caption, font=font(28), fill='white')
        sheet.paste(image, (0, y+45))
        y += image.height+label
    draw.text((gap, y), '表示の拡大（撃ち合い）', font=font(30), fill='#c4d4e8')
    y += label
    for image in details:
        sheet.paste(image, (gap, y))
        y += image.height+gap
    sheet.save(target, quality=JPEG_QUALITY, subsampling=0)


def render_layout(browser: Any, key: str, query: str, rows: list, scenes: dict, out: Path) -> None:
    import io
    from PIL import Image
    from scripts.render_overlay_design import new_page
    from scripts.render_overlay_b3_contents import render_scene, detail_images
    pictures, details = [], []
    for scene, selected in scenes.items():
        page = new_page(browser, query)
        render_scene(page, rows, selected)
        stamp = selected['evaluations']['b3_contents']['t_sec']
        picture = compose(page, out/'frames'/f'{scene}.png', out/f'{key}_{scene}.jpg', stamp)
        pictures.append((f'{LABELS[scene]} · c68 {stamp:.3f}秒', picture))
        if scene == 'exchange':
            picture.save(out/f'{key}.jpg', quality=90)
            if query.startswith('card'):
                image = Image.open(io.BytesIO(page.locator('#panel').screenshot())).convert('RGB')
                details = [image.resize((1016, round(image.height*1016/image.width)))]
            else:
                details = detail_images(page)
        page.close()
    mobile(key, pictures, details, out/'mobile'/f'{key}.jpg')


def render_diagnostics(browser: Any, rows: list[dict], source: Path, out: Path) -> None:
    """実画像でも確認した大きな不一致を、公開値を補正せず残す。"""
    import cv2
    from scripts.render_overlay_design import new_page
    from scripts.render_overlay_b3_contents import render_scene
    indexed = {r['evaluations']['b3_contents']['frame']: r for r in rows}
    capture = cv2.VideoCapture(str(source))
    for frame in DIAGNOSTIC_FRAMES:
        row = indexed[frame]
        target = out/'frames'/f'diagnostic_{frame}.png'
        extract_frame(capture, row, target)
        page = new_page(browser, 'l-right')
        render_scene(page, rows, row)
        compose(page, target, out/f'diagnostic_{frame}.jpg', frame/FPS)
        page.close()
    capture.release()


def render(args: argparse.Namespace) -> None:
    import cv2
    import pandas as pd
    from playwright.sync_api import sync_playwright
    from scripts.render_overlay_trend import THEMES
    verify_source(args.source)
    rows = [json.loads(s) for s in (args.output/'snapshots.jsonl').read_text(encoding='utf-8').splitlines()]
    scenes = choose_scenes(rows, pd.read_csv(args.output/'timeline.csv'))
    for folder in ('frames', 'mobile', 'trend/mobile', 'trend/frames'):
        (args.output/folder).mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.source))
    assert capture.get(cv2.CAP_PROP_FPS) == FPS
    for scene, row in scenes.items():
        extract_frame(capture, row, args.output/'frames'/f'{scene}.png')
        extract_frame(capture, row, args.output/'trend/frames'/f'{scene}.png')
    capture.release()
    with sync_playwright() as runner:
        browser = runner.chromium.launch(channel='msedge', headless=True)
        for key, query in (('l_right', 'l-right'), ('data', 'data')):
            render_layout(browser, key, query, rows, scenes, args.output)
        render_layout(browser, 'l_right', 'l-right&theme=neon', rows, scenes, args.output/'trend')
        for theme in THEMES:
            render_layout(browser, theme, 'card&theme='+theme, rows, scenes, args.output/'trend')
        render_diagnostics(browser, rows, args.source, args.output)
        browser.close()
    write_json(args.output/'manifest.json', dict(video=VIDEO, source_sha256=SOURCE_SHA,
        timing='frame_idx / 60; 公開行と元フレームを同フレームで対応', scenes=scenes,
        diagnostic_frames=DIAGNOSTIC_FRAMES))
    if args.samples is not None:
        copy_samples(args.output, args.samples)


def copy_samples(out: Path, target: Path) -> None:
    """既存のローカル見本置場には画像と出典だけを更新する。"""
    import shutil
    if target.resolve().is_relative_to(ROOT):
        raise ValueError('見本はリポジトリ外に保存してください')
    for folder in ('', 'mobile', 'trend', 'trend/mobile'):
        (target/folder).mkdir(parents=True, exist_ok=True)
        for path in (out/folder).glob('*.jpg'):
            if path.name.startswith(('inspect', 'screen_mismatches')):
                continue
            shutil.copy2(path, target/folder/path.name)
    shutil.copy2(out/'manifest.json', target/'manifest_matched.json')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'audit', 'render'))
    parser.add_argument('--samples', type=Path, help='既存ローカル見本を更新する場合の保存先')
    for key in ('source', 'output', 'rows', 'records', 'model', 'table'):
        parser.add_argument('--'+key, type=Path, required=key in ('source', 'output'))
    args = parser.parse_args()
    if args.mode != 'render' and any(getattr(args, key) is None for key in ('rows', 'records', 'model', 'table')):
        parser.error('prepare/auditには --rows --records --model --table が必要です')
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError('画像・検査結果はリポジトリ外に保存してください')
    args.output.mkdir(parents=True, exist_ok=True)
    (render if args.mode == 'render' else prepare)(args)


if __name__ == '__main__':
    main()
