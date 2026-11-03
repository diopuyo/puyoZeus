"""撃った連鎖の火力 X が配信へ出る早さと正しさを、固定72場面で採点する測定台 (段0)。

物差しは D:/puyo_analyzer/obs_vs_video_20261022/score_saved.py と同じ数え方:
- 得点一致は実 X 得点の ±10%。公開の窓は「保存発火時刻〜攻め連鎖終了+2秒」。
- 当該 X は、公開票の v4.9f 完了IDで結合した結果の発火 trigger が、場面の発火通知に
  「同じ側で 1.5 秒以内に最も近い発火」(summarize_replay.align と同じ nearest_trigger) として
  対応付いたもの、またはその発火から攻め連鎖終了までの後続断片。
  (2026-10-23 是正) 旧規則は「trigger ≥ 通知時刻」のみで、流し直し・新しい走行で自分の発火が
  通知より少し早く出ると、それを「別の発火の保持値」として捨て、後半の断片を採点していた。
  21c 原票では全場面で発火 = 通知時刻なので、旧規則と新規則は同じ値になる (既知値の再現で確認)。
M1 初回公開の得点一致 / M2 最初の当該 X の得点一致 (分母はともに実 X 比較可能63) /
M3 発火通知→当該 X 公開の P50/P95 / M4 誤発火 / M5 二走行の副作用 (別関数)。
読むのは保存原票だけ。録画の流し直しは scripts/run_live_x_replay.py (合言葉の門つき)。
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
from bisect import bisect_left
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle
from typing import Any, Iterable, Iterator

ACTUAL = Path(_public_dev_text('<work>/prefire_v412_20261022/actual/exchanges.json'))
SAVED_21C = Path('D:/PuyoLive_OBS_20261021c_USER_ONLY/output/20261021T205226_258349')
SYNC_ROOT = Path('D:/puyo_analyzer/obs_vs_video_20261022')
TOLERANCE = .10            # score_saved.py:14 と同じ ±10%
TRIGGER_EPS = 1e-5         # 発火時刻の一致許容 (score_saved.py:50-52)
TRIGGER_RADIUS_SEC = 1.5   # 場面の発火通知と発火を結ぶ近傍 (summarize_replay.TRIGGER_RADIUS と同じ)
SIDES = (0, 1)
NO_TRIGGER = float('-inf')  # 近傍に発火が無い場面の印 (どの trigger とも一致しない)
TAIL_SEC = 2.              # 公開窓の後端 = 攻め連鎖終了+2秒 (score_saved.py:74)
ORIGIN_RANGE_LIMIT_SEC = 1e-3   # captured_at−t_sec の幅。実時間入力なら一定
PERCENT_POINTS = (50, 95, 100)
METRIC_KEYS = ('chain', 'score', 'gross')
# 21c の既知値 (OBS_VS_VIDEO_20261022.md §1・§3、first_fresh_delay.json)。物差しの健全性の門。
KNOWN_21C = dict(first_score=4, fresh_score=30, first_present=14, fresh_n=51,
                 fresh_p50=.7039574000518769, fresh_p95=.9593627999420278)
KNOWN_FULL = dict(first_score=50, fresh_score=52)
KNOWN_FLOAT_EPS = 1e-9


def pickle_stream(path: Path) -> Iterator[Any]:
    """spool の追記 pickle を順に読む。途中切れの末尾は読まない。"""
    with path.open('rb') as stream:
        while True:
            try:
                yield pickle.load(stream)
            except EOFError:
                return


def jsonl_rows(path: Path) -> Iterator[dict]:
    """jsonl / jsonl.gz を読む。未クローズ gzip や途中行で止まる (原票は直さない)。"""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        try:
            for line in stream:
                if not line.endswith('\n'):
                    return
                yield json.loads(line)
        except EOFError:
            return


def run_origin(evaluations: Iterable[dict]) -> float:
    """壁時計→記録秒の対応。実時間入力では captured_at−t_sec が全行で一定。"""
    values = [row['captured_at']-row['t_sec'] for row in evaluations]
    if not values or max(values)-min(values) > ORIGIN_RANGE_LIMIT_SEC:
        raise ValueError('captured_at−t_sec が一定でない (実時間入力でない) ため壁時計で採点できない')
    return min(values)


def load_publications(output: Path, placement: str = 'wall') -> list[dict]:
    """公開票を評価行 (frame, evaluated_at) で結合し、v4.9f 欄と公開時刻を復元する。

    placement='wall' は公開の壁時計を記録秒へ写す (OBS と同じ)。'input' は入力時刻へ置く (F 条件)。
    """
    evaluations = {(r['frame'], r['evaluated_at']): r
                   for r in pickle_stream(output/'spool/evaluations.pickle')}
    origin = run_origin(evaluations.values()) if placement == 'wall' else None
    rows = []
    for publication in pickle_stream(output/'spool/publications.pickle'):
        row = evaluations.get((publication['frame'], publication['evaluated_at']))
        if placement == 'input' and row is None:
            continue
        stamp = publication['published_at']-origin if origin is not None else row['t_sec']
        rows.append(dict(t=stamp, frame=publication['frame'],
                         input_t=None if row is None else row['t_sec'],
                         vf=(row or {}).get('v49f') or {}))
    return rows


def index_results(updates: Iterable[dict]) -> dict:
    """score_saved.index_results と同じ。最新の X 付き base を値付き結果へ添える。"""
    base, result = None, {}
    for row in updates:
        if row.get('base') and 'predicted_scores' in row['base']:
            base = row['base']
        if row.get('value'):
            result[row['completed_at']] = (row, base)
    return result


def fire_ids(updates: Iterable[dict]) -> list[set[float]]:
    """v4.9f 更新票の base (X つき) が持つ発火時刻 chain_ids を、側ごとに集める。"""
    ids: list[set[float]] = [set() for _ in SIDES]
    for row in updates:
        base = row.get('base')
        if base and 'predicted_scores' in base:
            for side, trigger in enumerate(base['chain_ids']):
                if trigger is not None:
                    ids[side].add(trigger)
    return ids


def align_triggers(scenes: list[dict], ids: list[set[float]], radius: float = TRIGGER_RADIUS_SEC) -> list[dict]:
    """場面の発火通知を、同じ側で radius 秒以内に最も近い未使用の発火へ対応付ける。

    summarize_replay.align (nearest_trigger) と同じ規則。得点の正否は選択に使わない。
    当該 X = 対応した発火と、その発火から攻め連鎖終了までの後続断片。入力の scenes は変えない。
    """
    used: set[tuple[int, float]] = set()
    result = []
    for actual in scenes:
        item = dict(actual, scene=dict(actual['scene']))
        scene = item['scene']
        side = scene['attacker']
        candidates = sorted((abs(t-scene['start']), t) for t in ids[side]
                            if abs(t-scene['start']) <= radius and (side, t) not in used)
        trigger = candidates[0][1] if candidates else NO_TRIGGER
        fragments = sorted(t for t in ids[side] if trigger <= t < scene['attack']['end']) if candidates else []
        if candidates:
            used.add((side, trigger))
        scene['_replay_trigger'], scene['_replay_triggers'] = trigger, fragments
        item['_alignment'] = dict(found=bool(candidates), candidates=len(candidates),
                                  offset=trigger-scene['start'] if candidates else None)
        result.append(item)
    return result


def alignment_summary(aligned: list[dict]) -> dict:
    """対応付けの母数 (場面数・見つかった数) と、発火 − 通知時刻のずれ。見つからない場面は M3 から外れる。"""
    offsets = [a['_alignment']['offset'] for a in aligned if a['_alignment']['found']]
    return dict(method='nearest_trigger', radius_sec=TRIGGER_RADIUS_SEC, scenes=len(aligned),
                found=len(offsets), offset_sec=percentiles(offsets))


def within(predicted: float | None, actual: float | None) -> bool | None:
    if actual is None or predicted is None:
        return None
    return abs(predicted-actual) <= TOLERANCE*abs(actual)


def valid_trigger(trigger: float | None, scene: dict) -> bool:
    """再生の対応付け (_replay_trigger(s)) があればそれを、無ければ攻め連鎖の区間を使う。"""
    fragments, aligned = scene.get('_replay_triggers'), scene.get('_replay_trigger')
    if trigger is None:
        return False
    if fragments is not None:
        return any(abs(trigger-t) < TRIGGER_EPS for t in fragments)
    if aligned is not None:
        return abs(trigger-aligned) < TRIGGER_EPS
    return scene['start']-TRIGGER_EPS <= trigger < scene['attack']['end']


def prediction(publication: dict, indexed: dict, scene: dict) -> dict:
    """公開票1件の当該 X。理由の文言も score_saved.prediction と同じにする。"""
    pair = indexed.get(publication['vf'].get('completed_at'))
    result = dict(public_sec=publication['t'], input_sec=None, chain=None, score=None,
                  gross=None, trigger=None, reason='初回公開に予測Xなし')
    if pair is None or pair[1] is None:
        return result
    row, base = pair
    side = scene['attacker']
    trigger = base['chain_ids'][side]
    result.update(input_sec=row['t'], trigger=trigger)
    if not valid_trigger(trigger, scene):
        result['reason'] = '別の発火の保持値'
        return result
    value = row['value']
    scores, account = value['predicted_scores'][side], value['reconciliation'][side]
    if not scores:
        return result
    result.update(chain=len(scores), score=sum(scores), reason=None,
                  gross=account['total'] if account else None,
                  prediction_active=value['board_prediction_active'][side])
    return result


def truth_values(actual: dict) -> dict:
    truth = actual['X']
    valid = bool(truth and truth['comparable'])
    account = (truth.get('accounting') or {}) if truth else {}
    gross = (account.get('reconstructed', {}).get('generated_total')
             if account.get('actual_sent') is not None else None)
    return dict(actual_chain=truth['chain'] if valid else None,
                actual_score=truth['score'] if valid else None, actual_gross=gross)


def score_scene(actual: dict, display: list[dict], times: list[float], indexed: dict) -> dict:
    """1場面を採点する (score_saved.score_scene と同じ窓・同じ判定)。"""
    scene = actual['scene']
    start, end = scene['start'], scene['attack']['end']
    publications = display[bisect_left(times, start):bisect_left(times, end+TAIL_SEC)]
    predictions = [prediction(p, indexed, scene) for p in publications]
    first = predictions[0] if predictions else dict(reason='公開票なし')
    fresh = next((p for p in predictions if p.get('score') is not None), {})
    result = dict(id=scene['id'], start=start, side=scene['attacker'], first=first,
                  first_fresh=fresh, **truth_values(actual))
    for key in METRIC_KEYS:
        target = result['actual_'+key]
        result['first_'+key+'_within10'] = within(first.get(key), target)
        result['fresh_'+key+'_within10'] = within(fresh.get(key), target)
        matched = [p for p in predictions if p.get('prediction_active') and within(p.get(key), target)]
        result[key+'_seconds_to_within10'] = matched[0]['public_sec']-start if matched else None
    return result


def percentiles(values: list[float]) -> dict | None:
    import numpy as np
    if not values:
        return None
    return dict(zip(('p50', 'p95', 'max'), map(float, np.percentile(values, PERCENT_POINTS))))


def summary(rows: list[dict]) -> dict:
    """M1・M2 と初回公開の X 有無 (score_saved.summary と同じ母数)。"""
    result = dict(scenes=len(rows),
                  first_prediction_present=sum(r['first'].get('score') is not None for r in rows),
                  first_missing_reasons=dict(Counter(str(r['first'].get('reason')) for r in rows)))
    for key in METRIC_KEYS:
        eligible = [r for r in rows if r['actual_'+key] is not None]
        result[key] = dict(actual_comparable=len(eligible),
            first_within10=sum(r['first_'+key+'_within10'] is True for r in eligible),
            first_fresh_within10=sum(r['fresh_'+key+'_within10'] is True for r in eligible),
            first_fresh_present=sum(r['first_fresh'].get(key) is not None for r in eligible),
            arrival_seconds=percentiles([r[key+'_seconds_to_within10'] for r in eligible
                                         if r[key+'_seconds_to_within10'] is not None]))
    return result


def fresh_delays(rows: list[dict]) -> dict:
    """M3: 発火通知 (保存 CHAIN 通知) → 最初の当該 X の公開。母数は得点比較可能かつ当該 X あり。"""
    items = []
    for row in rows:
        fresh = row['first_fresh']
        if row['actual_score'] is None or fresh.get('score') is None:
            continue
        delay = fresh['public_sec']-row['start']
        waiting = max(0., fresh['input_sec']-row['start'])
        items.append(dict(id=row['id'], first_fresh_delay=delay, input_delay=waiting,
                          publication_delay=delay-waiting, correct=row['fresh_score_within10']))
    return dict(n=len(items), rows=items,
                first_fresh_delay=percentiles([i['first_fresh_delay'] for i in items]),
                input_delay=percentiles([i['input_delay'] for i in items]),
                publication_delay=percentiles([i['publication_delay'] for i in items]))


def score_run(display: list[dict], indexed: dict, scenes: list[dict]) -> dict:
    """公開票列と v4.9f 結果から M1〜M3 をまとめる。"""
    times = [r['t'] for r in display]
    if times != sorted(times):
        raise ValueError('公開票の時刻が単調でない')
    rows = [score_scene(a, display, times, indexed) for a in scenes]
    result = summary(rows)
    return dict(M1_first_score=result['score']['first_within10'],
                M2_fresh_score=result['score']['first_fresh_within10'],
                comparable=result['score']['actual_comparable'],
                first_present=result['first_prediction_present'],
                M3=fresh_delays(rows), summary=result, rows=rows)


def load_scenes(path: Path = ACTUAL) -> list[dict]:
    return json.loads(path.read_text(encoding='utf-8'))


def score_saved_output(output: Path, scenes: list[dict], placement: str = 'wall', align: bool = True) -> dict:
    """配信の保存原票 (spool と v49f_updates.jsonl) を採点する。

    align=True (既定) は発火通知を最も近い発火へ対応付ける (align_triggers)。False は旧規則
    (trigger が [通知時刻, 攻め連鎖終了) に入るもの) で、新旧の比較・過去値の再現用。
    """
    display = load_publications(output, placement)
    updates = output/'v49f_updates.jsonl'
    indexed = index_results(jsonl_rows(updates))
    summary_of_alignment = None
    if align:
        scenes = align_triggers(scenes, fire_ids(jsonl_rows(updates)))
        summary_of_alignment = alignment_summary(scenes)
    return dict(score_run(display, indexed, scenes), placement=placement, output=str(output),
                alignment=summary_of_alignment)


def load_sync_replay(root: Path, prefix: str = 'full') -> tuple[list[dict], dict, list[dict]]:
    """F 条件 (全コマ同期再生) の保存成果物を、summarize_replay と同じ規則で読む。"""
    summary_path = root/f'{prefix}_comparison/summary.json'
    shift = json.loads(summary_path.read_text(encoding='utf-8'))['shift']
    with gzip.open(root/f'{prefix}_comparison/display.json.gz', 'rt', encoding='utf-8') as stream:
        display = json.load(stream)
    indexed, base = {}, None
    for row in jsonl_rows(root/f'{prefix}_vf_exact/v49f.jsonl.gz'):
        if row['base'] and 'predicted_scores' in row['base']:
            base = dict(row['base'], chain_ids=[t+shift if t is not None else None
                                                for t in row['base']['chain_ids']])
        if row['value'] is not None:
            indexed[row['frame']] = (dict(row, t=row['t']+shift), base)
    alignment = {a['id']: a for a in json.loads(
        (root/f'{prefix}_comparison/trigger_alignment.json').read_text(encoding='utf-8'))}
    return display, indexed, alignment


def aligned_scenes(scenes: list[dict], alignment: dict) -> list[dict]:
    """summarize_replay.align の保存結果を場面へ戻す (見つからない発火は -inf)。"""
    result = []
    for actual in scenes:
        item = dict(actual, scene=dict(actual['scene']))
        found = alignment[actual['scene']['id']]
        trigger = found['trigger']
        item['scene']['_replay_trigger'] = NO_TRIGGER if trigger is None else trigger
        item['scene']['_replay_triggers'] = found['fragments']
        result.append(item)
    return result


def check_known(values: dict, known: dict) -> dict:
    """既知値との一致を項目ごとに残す。1件でも違えば物差しは使わない。"""
    checks = {key: dict(expected=expected, observed=values.get(key),
                        passed=values.get(key) is not None and abs(values[key]-expected) <= KNOWN_FLOAT_EPS)
              for key, expected in known.items()}
    return dict(passed=all(c['passed'] for c in checks.values()), checks=checks)


def reproduce_known(scenes: list[dict], saved: Path = SAVED_21C, sync_root: Path = SYNC_ROOT) -> dict:
    """段0 の最初の仕事: 21c 保存原票と全コマ同期再生の既知値を再現する。"""
    obs = score_saved_output(saved, scenes)
    delays = obs['M3']['first_fresh_delay'] or {}
    obs_values = dict(first_score=obs['M1_first_score'], fresh_score=obs['M2_fresh_score'],
                      first_present=obs['first_present'], fresh_n=obs['M3']['n'],
                      fresh_p50=delays.get('p50'), fresh_p95=delays.get('p95'))
    display, indexed, alignment = load_sync_replay(sync_root)
    full = score_run(display, indexed, aligned_scenes(scenes, alignment))
    full_values = dict(first_score=full['M1_first_score'], fresh_score=full['M2_fresh_score'])
    return dict(obs=check_known(obs_values, KNOWN_21C), full=check_known(full_values, KNOWN_FULL),
                scenes=len(scenes), comparable=obs['comparable'], alignment=obs['alignment'])


def compact(report: dict) -> dict:
    """出力用に場面別の明細を落とす (明細は --rows で別保存)。"""
    return {k: (dict(v, rows=None) if k == 'M3' else v) for k, v in report.items() if k != 'rows'}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    known = sub.add_parser('known', help='21c と全コマ同期再生の既知値を再現する')
    known.add_argument('--actual', type=Path, default=ACTUAL)
    score = sub.add_parser('score', help='配信の保存原票 1 走行を採点する')
    score.add_argument('--run', type=Path, required=True)
    score.add_argument('--actual', type=Path, default=ACTUAL)
    score.add_argument('--placement', choices=('wall', 'input'), default='wall')
    score.add_argument('--rows', type=Path)
    score.add_argument('--legacy-window', action='store_true',
                       help='旧規則 (trigger が通知時刻以上かつ攻め連鎖終了前) で採点する。新旧比較用')
    args = parser.parse_args(argv)
    scenes = load_scenes(args.actual)
    if args.command == 'known':
        result = reproduce_known(scenes)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not (result['obs']['passed'] and result['full']['passed']):
            raise SystemExit('既知値を再現できない。物差しを使わずに原因を調べる')
        return
    report = score_saved_output(args.run, scenes, args.placement, align=not args.legacy_window)
    if args.rows:
        args.rows.write_text(json.dumps(report['rows'], ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(compact(report), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
