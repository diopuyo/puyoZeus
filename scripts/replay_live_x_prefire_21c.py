"""段3 の保存原票再生 (読み取りのみ)。21c の通知列で着地を再生し、暫定 X の当否を数える。

画像は保存原票に無いため、着地は STABLE 確定盤 (3通知一致) の「追加2個」だけで見る。
専用の画像読取 (live_v49f_prefire の image 経路) の再現率はこの台では測れない (録画の流し直しが必要)。
起動は明示の --out 指定時だけ。PuyoLive・OBS・runner は起動しない。

例 (配布物の python、優先度を下げて):
  python scripts/replay_live_x_prefire_21c.py --inputs <21c>/inputs.jsonl.gz \
      --chains <diag>/chains.json.gz --scenes <prefire_v412>/actual/exchanges.json \
      --policy <app>/models/v49f/policy.json --out <dir>/replay.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.phase_j.live_v49f_prefire import PrefireTracker, SideView  # noqa: E402

GZIP_WBITS = 16+zlib.MAX_WBITS
NEAR_SCORE_ABS, NEAR_SCORE_REL = 40, .10   # v4.12 分解と同じ近似一致 (得点 ±max(40,10%))
MATCH_EPS_SEC = .05
PERCENTILES = (25, 50, 75, 95)


def read_updates(path: Path) -> list[dict]:
    """末尾が欠けた gzip も読める範囲で読む (21c は終了時に gzip の末尾が書かれていない)。"""
    text = zlib.decompressobj(GZIP_WBITS).decompress(path.read_bytes()).decode('utf-8', 'replace')
    rows = []
    for line in text.split('\n'):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get('kind') == 'update':
            rows.append(update_row(item['args']['tuple']))
    return rows


def update_row(args: list) -> dict:
    """ExchangeEventRecorder.update の引数 (result, snapshot, finalization, t, game, 式累計, 表示得点, ...)。"""
    result = args[0]['namespace']
    sides = [side_row(result[key]['namespace']) for key in ('p1', 'p2')]
    displayed = args[6]['tuple'] if isinstance(args[6], dict) else [None, None]
    return dict(t=float(args[3]), game=int(args[4]), sides=sides, displayed=displayed)


def side_row(value: dict) -> dict:
    board = value['confirmed_board']
    grid = None if board is None else tuple(c for row in board['board']['array'] for c in row)
    event = value['chain_event']
    trigger = None if event is None else float(event['namespace']['trigger_sec'])
    pairs = [None if value[k] is None else tuple(value[k]['tuple']) for k in ('next_pair', 'dnext_pair')]
    return dict(state=value['state']['state'], board=grid, trigger=trigger, next=pairs[0], dnext=pairs[1])


def replay(rows: list[dict], native: Any, quiet_sec: float | None) -> tuple[list[dict], list[float]]:
    """保存の通知順・片側順 (1P→2P) で追跡器へ渡す。得点台帳は保存の表示得点で進める。"""
    from src.exchange_count_ledger import ScorePrefix
    tracker, events, costs = PrefireTracker(native), [], []
    ledger, game = ScorePrefix(quiet_sec), None
    for row in rows:
        if row['game'] != game:
            ledger, game = ScorePrefix(quiet_sec), row['game']
        for side, score in enumerate(row['displayed']):
            ledger.observe(side, row['t'], score)
        for side, value in enumerate(row['sides']):
            snapshot = ledger.snapshot(side, row['t']) if quiet_sec is not None else None
            view = SideView(value['state'], value['board'], value['next'], value['trigger'], snapshot,
                            value['dnext'])
            began = perf_counter()
            for event in tracker.observe(row['game'], side, row['t'], view):
                events.append(dict(event, game=row['game']))
            costs.append((perf_counter()-began)*1000.)
    return events, costs


def quantiles(values: list[float]) -> dict:
    values = sorted(values)
    if not values:
        return dict(count=0)
    pick = {f'P{p}': values[min(len(values)-1, int(p/100*len(values)))] for p in PERCENTILES}
    return dict(count=len(values), **pick, max=values[-1])


def near(a: int, b: int) -> bool:
    return abs(a-b) <= max(NEAR_SCORE_ABS, NEAR_SCORE_REL*b)


def resolve(events: list[dict]) -> list[dict]:
    """暫定 X ごとに確定・取り下げを対応づける (同じ側の次の confirm / retract)。"""
    open_: dict[tuple, dict] = {}
    out = []
    for event in events:
        key = (event['game'], event['side'])
        if event['kind'] == 'provisional':
            open_[key] = dict(event, outcome=None)
            out.append(open_[key])
        elif event['kind'] in ('confirm', 'retract') and key in open_:
            item = open_.pop(key)
            item.update(outcome=event['kind'], reason=event.get('reason'), resolved_t=event['t'],
                        delay_sec=event['delay_sec'])
    return out


def chain_for(item: dict, chains: list[dict]) -> dict | None:
    """確定した暫定 X に当たる連鎖 = 同じ試合・側で、着地以後・確定通知以前に始まった最初の連鎖。"""
    found = [c for c in chains if c['game'] == item['game'] and c['side'] == item['side']
             and item['landing_t']-MATCH_EPS_SEC <= c['start'] <= item['resolved_t']+MATCH_EPS_SEC]
    return min(found, key=lambda c: c['start']) if found else None


def score_summary(items: list[dict], chains: list[dict]) -> dict:
    confirmed = [i for i in items if i['outcome'] == 'confirm']
    exact = near_count = unmatched = 0
    for item in confirmed:
        chain = chain_for(item, chains)
        item['actual_score'] = None if chain is None else chain['score']
        if chain is None:
            unmatched += 1
        else:
            exact += item['score'] == chain['score']
            near_count += near(item['score'], chain['score'])
    return dict(confirmed=len(confirmed), score_exact=exact, score_near=near_count, unmatched=unmatched,
                lead_sec=quantiles([i['delay_sec'] for i in confirmed]))


def recall(items: list[dict], targets: list[dict]) -> dict:
    """発火 (連鎖の開始) のうち、確定した暫定 X が付いた件数と得点一致。"""
    hits = exact = 0
    for target in targets:
        mine = [i for i in items if i['outcome'] == 'confirm' and i['game'] == target['game']
                and i['side'] == target['side']
                and i['landing_t']-MATCH_EPS_SEC <= target['start'] <= i['resolved_t']+MATCH_EPS_SEC]
        if mine:
            hits += 1
            exact += mine[0]['score'] == target['score']
    return dict(targets=len(targets), hit=hits, score_exact=exact)


def scene_targets(scenes: list[dict], chains: list[dict]) -> list[dict]:
    """72 場面の攻め X を、連鎖一覧の (試合・側・開始) へ引き当てる。"""
    targets = []
    for scene in scenes:
        x = scene['X']
        if x is None:
            continue
        found = [c for c in chains if c['side'] == x['side'] and abs(c['start']-x['trigger_sec']) <= MATCH_EPS_SEC]
        if found:
            targets.append(dict(found[0], score=x['score']))
    return targets


def landing_fires(rows: list[dict], chains: list[dict]) -> list[dict]:
    """撃った着地: 着地 (TSUMO_FALL→STABLE) の後、次の落下・おじゃま・MENU より先に連鎖の初段が始まった件。

    設計書 §3.4 の「撃った着地 66 件」は定義が原票に残っておらず再現できなかった。ここでは上の定義で数える。
    """
    starts = {(c['game'], c['side'], round(c['start'], 4)): c for c in chains}
    found = []
    for side in (0, 1):
        previous, landing, trigger = None, None, None
        for row in rows:
            value = row['sides'][side]
            fresh = value['trigger'] is not None and value['trigger'] != trigger
            trigger = value['trigger'] if value['trigger'] is not None else trigger
            chain = starts.get((row['game'], side, round(value['trigger'], 4))) if fresh else None
            if landing is not None and chain is not None:
                found.append(dict(chain, landing_t=landing))
                landing = None
            elif landing is not None and value['state'] != previous and value['state'] in ('TSUMO_FALL', 'OJAMA_FALL', 'MENU'):
                landing = None
            if previous == 'TSUMO_FALL' and value['state'] == 'STABLE':
                landing = row['t']
            previous = value['state']
    return found


def summarize(events: list[dict], costs: list[float], chains: list[dict], scenes: list[dict]) -> dict:
    items = resolve(events)
    retracted = [i for i in items if i['outcome'] == 'retract']
    reasons: dict[str, int] = {}
    for item in retracted:
        reasons[item['reason']] = reasons.get(item['reason'], 0)+1
    tables = [e for e in events if e['kind'] == 'table']
    return dict(
        notifications_sides=len(costs), tables=len(tables), provisional=len(items),
        confirmed=score_summary(items, chains),
        false_fire=dict(count=len(retracted), reasons=reasons,
                        retract_delay_sec=quantiles([i['delay_sec'] for i in retracted if i['delay_sec'] is not None])),
        unresolved=sum(i['outcome'] is None for i in items),
        recall_all_chains=recall(items, chains), recall_72_scenes_x=recall(items, scene_targets(scenes, chains)),
        landings_nofire=sum(e['kind'] == 'landing' for e in events),
        table_build_ms=quantiles([e['build_ms'] for e in tables]),
        table_firing=quantiles([float(e['firing']) for e in tables]),
        observe_ms=quantiles(costs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'chains', 'scenes', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--events', type=Path, help='暫定 X の全事象を jsonl で保存')
    options = parser.parse_args()
    import puyo_core
    quiet = json.loads(options.policy.read_text(encoding='utf-8'))['first_move_quiet_sec'] if options.policy else None
    rows = read_updates(options.inputs)
    chains = json.loads(gzip.open(options.chains, 'rt', encoding='utf-8').read())
    scenes = json.loads(options.scenes.read_text(encoding='utf-8'))
    events, costs = replay(rows, puyo_core, quiet)
    import hashlib
    summary = dict(inputs_sha256=hashlib.sha256(options.inputs.read_bytes()).hexdigest(),
                   source='confirmed_board_only', updates=len(rows), chains=len(chains), scenes=len(scenes),
                   **summarize(events, costs, chains, scenes))
    fires = landing_fires(rows, chains)
    summary['recall_landing_fires'] = recall(resolve(events), fires)
    options.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if options.events:
        with options.events.open('w', encoding='utf-8') as stream:
            for event in events:
                stream.write(json.dumps(event, ensure_ascii=False)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
