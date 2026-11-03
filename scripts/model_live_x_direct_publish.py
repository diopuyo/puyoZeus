"""段1 直接公開の効果見込み: 21c 保存原票で v4.9f の待ち行列と公開の重複排除を模して再生する。

実時間の流し直しではない (それは scripts/run_live_x_replay.py の L 条件)。保存された時刻だけで
OFF を再生して実公開を再現できることを先に確かめ、同じ再生器で ON の規則に替えて M1〜M3 を出す。

模すもの (保存原票の単調時計そのまま):
- 作業側: FIFO・1件ずつ。ON は送信が前の完了を待たない → 開始 = max(届いた時刻, 前の完了)。
  届いた時刻は min(評価準備 evaluation_ready, OFF の送信時刻)。計算時間 compute_ms は同じとする。
- 公開: 評価行 (offer) ごとに判定ID が変われば公開 (live_publish.py:322-328)。ON は判定IDに
  X 初出の完了時刻を足し、評価所有者が空いていれば受信時点で起こす (忙しければ処理中の通知の終わり)。
- 公開の遅れ: 実際に公開された行は実測 (published_at−evaluated_at)、新たな行はその中央値。
模さないもの: 公開増による負荷、terminal 保持の受理判定、入力世代の切替での X 追跡の初期化 (試合単位で代用)。
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from pathlib import Path
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.measure_live_x_timing import (ACTUAL, KNOWN_21C, SAVED_21C, check_known, index_results,
                                           jsonl_rows, load_scenes, pickle_stream, run_origin, score_run)
from src.phase_j.live_x_direct import XFirstTracker

MS = 1000.
WAKE_OFFER_SEC = .002      # 起床 → 判定行の作成 (decorate) までの見込み。感度は --wake-offer-ms で見る
TIME_GATE = 'b3_time_gate.jsonl'


def load_inputs(saved: Path) -> dict:
    """保存原票から、再生に要る列だけを読む。"""
    results = [dict(id=r['id'], game=r['game'], t=r['t'], completed_at=r['completed_at'], compute_ms=r['compute_ms'],
                    wait_ms=r['wait_ms'], base=r['base'], value=r['value'], origins=r['origins'])
               for r in jsonl_rows(saved/'v49f_updates.jsonl')]
    ready: dict[int, float] = {}
    for row in jsonl_rows(saved/TIME_GATE):
        if row.get('kind') == 'evaluation_ready':
            ready.setdefault(row['frame'], row['monotonic'])
    evaluations = list(pickle_stream(saved/'spool/evaluations.pickle'))
    busy = sorted((r['started_at'], r['finished_at']) for r in pickle_stream(saved/'spool/profile.pickle'))
    return dict(results=results, ready=ready, evaluations=evaluations, busy=busy,
                publications=list(pickle_stream(saved/'spool/publications.pickle')),
                origin=run_origin(evaluations))


def worker_timeline(results: list[dict], ready: dict, direct: bool) -> list[float]:
    """各結果を評価所有者側で受け取れる時刻。OFF は実測の完了、ON は送受信分離後の完了。"""
    ipc = statistics.median(r['wait_ms'] for r in results)/MS
    if not direct:
        return [r['completed_at']+ipc for r in results]
    received, previous = [], float('-inf')
    for r in results:
        began = r['completed_at']-r['compute_ms']/MS
        sent = began-r['wait_ms']/MS
        arrived = min(ready.get(r['id'], sent), sent)+ipc
        start = min(began, max(arrived, previous))
        previous = start+r['compute_ms']/MS
        received.append(previous+ipc)
    return received


X_RULES = ('trigger', 'origins', 'both')   # trigger が本番 (live_x_direct.py)。他は比較用


def x_first(results: list[dict], generations: dict, rule: str = 'trigger') -> list[bool]:
    """本番と同じ XFirstTracker (新しい発火の X の初出)。origins は設計 §2 の字義、both は和集合。"""
    tracker = XFirstTracker()
    flags = []
    for r in results:
        flag = tracker.observe(r, (generations.get(r['id']), r['game']))
        origin = bool(r['origins']) and r['value'] is not None
        flags.append(origin if rule == 'origins' else flag or origin if rule == 'both' else flag)
    return flags


def generation_of_frames(evaluations: list[dict]) -> dict:
    """結果 id (=通知 frame) → その通知の B3 世代 (判定IDの先頭、LiveV49F.update と同じ値)。"""
    generations: dict = {}
    for row in evaluations:
        if row.get('judgment_id') is not None:
            generations.setdefault(row['frame'], row['judgment_id'][0])
    return generations


def wake_time(busy: list[tuple], starts: list[float], stamp: float, offer_sec: float) -> float:
    """受信時に評価所有者が通知を処理中なら、その通知の終わりに起きる。"""
    index = bisect_right(starts, stamp)-1
    if index >= 0 and busy[index][0] <= stamp < busy[index][1]:
        return busy[index][1]+offer_sec
    return stamp+offer_sec


def offer_events(data: dict, received: list[float], fresh: list[bool], direct: bool,
                 offer_sec: float) -> list[tuple]:
    """(時刻, 種類, 評価行) を時刻順に。ON は X 初出の起床を加える。"""
    events = [(row['evaluated_at'], 0, row) for row in data['evaluations']]
    if direct:
        starts = [b[0] for b in data['busy']]
        events += [(wake_time(data['busy'], starts, stamp, offer_sec), 1, None)
                   for stamp, flag in zip(received, fresh) if flag]
    return sorted(events, key=lambda e: (e[0], e[1]))


class ResultCursor:
    """受信時刻順に結果を進め、LiveV49F と同じく (世代, 試合) が違う結果は捨て、変われば忘れる。"""

    def __init__(self, results: list[dict], received: list[float], fresh: list[bool],
                 generations: dict) -> None:
        self.order = sorted(range(len(results)), key=lambda i: received[i])
        self.results, self.received, self.fresh = results, received, fresh
        self.generations = generations
        self.position, self.latest, self.x_done, self.identity = 0, None, None, None

    def advance(self, stamp: float, identity: tuple) -> None:
        if identity != self.identity:
            self.identity, self.latest, self.x_done = identity, None, None
        while self.position < len(self.order) and self.received[self.order[self.position]] <= stamp:
            index = self.order[self.position]
            result = self.results[index]
            self.position += 1
            if result['value'] is None or (self.generations.get(result['id']), result['game']) != identity:
                continue
            self.latest = result['completed_at']
            if self.fresh[index]:
                self.x_done = result['completed_at']


def fixed_rows(data: dict) -> list[dict]:
    """評価行から作られない公開 (入力確認の HOLD、鮮度切れ・決着保持の再公開) は実測のまま置く。"""
    evaluations = {(r['frame'], r['evaluated_at']): r for r in data['evaluations']}
    seen, rows = set(), []
    for p in data['publications']:
        key = (p['frame'], p['evaluated_at'])
        if key not in evaluations or key in seen:
            vf = (evaluations.get(key) or {}).get('v49f') or {}
            rows.append(dict(t=p['published_at']-data['origin'], vf=dict(completed_at=vf.get('completed_at')),
                             frame=p['frame']))
        seen.add(key)
    return rows


def simulate(data: dict, direct: bool, offer_sec: float = WAKE_OFFER_SEC, rule: str = 'trigger') -> dict:
    """判定IDの重複排除を時刻順に再生し、公開票列 (記録秒, v4.9f 完了ID) を作る。"""
    results = data['results']
    generations = generation_of_frames(data['evaluations'])
    received = worker_timeline(results, data['ready'], direct)
    fresh = x_first(results, generations, rule) if direct else [False]*len(results)
    cursor = ResultCursor(results, received, fresh, generations)
    actual = {(p['frame'], p['evaluated_at']): p['published_at']-p['evaluated_at'] for p in data['publications']}
    latency = statistics.median(actual.values())
    display, last_key, last_row, agree = [], None, None, [0, 0]
    for stamp, kind, row in offer_events(data, received, fresh, direct, offer_sec):
        row = row if row is not None else last_row
        if row is None or row.get('judgment_id') is None:
            continue
        cursor.advance(stamp, (row['judgment_id'][0], row['judgment_id'][5]))  # (B3 世代, B3 試合)
        if kind == 0:
            agree[0] += cursor.latest == (row.get('v49f') or {}).get('completed_at')
            agree[1] += 1
        last_row = row
        key = (row['game'], *row['judgment_id'], *((cursor.x_done,) if direct else ()))
        if key == last_key:
            continue
        last_key = key
        delay = actual.get((row['frame'], row['evaluated_at']), latency) if kind == 0 else latency
        display.append(dict(t=stamp+delay-data['origin'], vf=dict(completed_at=cursor.latest),
                            frame=row['frame'], offered=(row['frame'], row['evaluated_at'], kind)))
    display += fixed_rows(data)
    return dict(display=sorted(display, key=lambda r: r['t']), x_first=sum(fresh), latency=latency,
                vf_agreement=agree[0]/agree[1] if agree[1] else None)


def compare_actual(simulated: dict, data: dict) -> dict:
    """OFF 再生が実公開を再現するか: 評価行由来の公開の集合と件数。"""
    actual = {(p['frame'], p['evaluated_at']) for p in data['publications']}
    made = {r['offered'][:2] for r in simulated['display'] if 'offered' in r}
    return dict(simulated=len(made), actual=len(actual), both=len(made & actual),
                only_simulated=len(made-actual), only_actual=len(actual-made))


def scored(simulated: dict, data: dict, scenes: list[dict]) -> dict:
    report = score_run(simulated['display'], index_results(data['results']), scenes)
    delays = report['M3']['first_fresh_delay'] or {}
    return dict(M1_first_score=report['M1_first_score'], M2_fresh_score=report['M2_fresh_score'],
                first_present=report['first_present'], fresh_n=report['M3']['n'],
                fresh_p50=delays.get('p50'), fresh_p95=delays.get('p95'),
                publication_delay=report['M3']['publication_delay'], input_delay=report['M3']['input_delay'],
                publications=len(simulated['display']), x_first=simulated['x_first'],
                vf_agreement=simulated['vf_agreement'])


def estimate(saved: Path, scenes: list[dict], offer_ms: tuple[float, ...]) -> dict:
    """OFF 再生で既知値の再現を示し、同じ再生器の ON 規則で見込みを出す。"""
    data = load_inputs(saved)
    off = simulate(data, False)
    result = dict(off=scored(off, data, scenes), off_vs_actual=compare_actual(off, data))
    off_values = result['off']
    known = dict(first_score=off_values['M1_first_score'], fresh_score=off_values['M2_fresh_score'],
                 first_present=off_values['first_present'], fresh_n=off_values['fresh_n'],
                 fresh_p50=off_values['fresh_p50'], fresh_p95=off_values['fresh_p95'])
    result['off_known'] = check_known(known, KNOWN_21C)
    result['on'] = {f'trigger_{ms:g}ms': scored(simulate(data, True, ms/MS), data, scenes) for ms in offer_ms}
    result['on'].update({f'{rule}_{offer_ms[0]:g}ms': scored(simulate(data, True, offer_ms[0]/MS, rule), data, scenes)
                         for rule in X_RULES[1:]})
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--saved', type=Path, default=SAVED_21C)
    parser.add_argument('--actual', type=Path, default=ACTUAL)
    parser.add_argument('--wake-offer-ms', type=float, nargs='+', default=[WAKE_OFFER_SEC*MS, 0., 20.])
    args = parser.parse_args(argv)
    print(json.dumps(estimate(args.saved, load_scenes(args.actual), tuple(args.wake_offer_ms)),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
