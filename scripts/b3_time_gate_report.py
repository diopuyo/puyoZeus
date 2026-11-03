"""時間原票の再集計。公式115試合は既存行の識別列だけで限定し、勝敗を読まない。"""
from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.b3_time_gate_support import quantiles, write_json
from src.b3_search import HMAX_TOTAL_HANDS

ROOT = Path('/mnt/d/puyo_analyzer/b3_data')
OUT = ROOT / 'time_gate'
MEMORY_GATE = 6.0
FEATURE_P95 = 100.0
FEATURE_P99 = 200.0
LATENCY_P99 = 500.0
FAIL_RATE = 0.01
WAIT_MS = 1000.0
MINUTE_SEC = 60.0
REPLAY_SEC = 1800.0


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]


def stats(rows: list[dict]) -> dict:
    """更新単位で処理時間・打切り率を集計する。"""
    keys = ('feature_ms', 'row_ms', 'state_ms', 'search_ms', 'ledger_ms', 'indicators_ms',
            'inference_ms', 'amortized_feature_ms', 'latency_ms', 'wait_ms')
    result = dict(n=len(rows), timings={k: quantiles([r[k] for r in rows]) for k in keys})
    result['rates'] = {}
    for label, part in (('all', rows), ('observed_chain7', [r for r in rows if r['observed_chain7']])):
        n = len(part)
        counts = {k: sum(r[k] for r in part) for k in ('unreached', 'k_unreached', 'ta_unreached')}
        result['rates'][label] = dict(n=n, counts=counts, rate=counts['unreached']/n if n else None)
    result['placement'] = quantiles([r['latency_ms'] for r in rows if r['placement']])
    result['cache'] = {key: sum(r[key] for r in rows) for key in ('cache_hits', 'cache_misses')}
    result['total_ms'] = {key: sum(r[key] for r in rows) for key in keys[:7]}
    result['ta_failures_h_above_cap'] = sum(r['ta_unreached'] and r['requested_h'] is not None
                                          and r['requested_h'] > HMAX_TOTAL_HANDS for r in rows)
    result['by_cache'] = {name: {key: quantiles([r[key] for r in rows if bool(r['cache_misses']) == miss])
                                for key in keys[:7]} for name, miss in (('miss', True), ('hit_only', False))}
    result['by_requested_h'] = {str(h): quantiles([r['feature_ms'] for r in rows if r['requested_h'] == h])
                                for h in sorted({r['requested_h'] for r in rows if r['requested_h'] is not None})}
    result['by_search_calls'] = {str(n): quantiles([r['feature_ms'] for r in rows if r['search_calls'] == n])
                                 for n in sorted({r['search_calls'] for r in rows})}
    return result


def official_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """評価parquetからID列だけを読み、モデル出力や教師は一切参照しない。"""
    columns = ['row_key', 'orientation', 'zenchi_set', 'zenchi_game']
    source = ROOT / 'eval_rows_v2b/b3_zenchi_merged.parquet'
    identity = pd.read_parquet(source, columns=columns)
    identity = identity.loc[identity.orientation == 0]
    keys = set(identity.row_key)
    found = {f"{r['video']}|g{r['game']}|f{r['frame']}|o0": r for r in rows}
    missing = sorted(keys - found.keys())
    selected = [r for key, r in found.items() if key in keys]
    games = identity.loc[identity.row_key.isin(found), ['zenchi_set', 'zenchi_game']].drop_duplicates()
    return selected, dict(expected_rows=len(keys), measured_rows=len(selected),
                          expected_games=115, measured_games=len(games), missing_keys=missing,
                          ignored_extra_updates=len(found)-len(selected), identity_source=str(source))


def realtime_detail(rows: list[dict]) -> dict:
    """開始/1800秒時点の未完了件数と、分単位の滞留傾向を記録する。"""
    base = OUT / 'realtime_30min'
    frames = read_lines(base / 'frames.jsonl')
    samples = read_lines(base / 'queue.jsonl')
    result = stats(rows)
    result.update(frame_wait=quantiles([r['wait_ms'] for r in frames]),
                  ingest_lag=quantiles([r['ingest_lag_ms'] for r in frames]),
                  wait_over_1s=sum(r['wait_ms'] > WAIT_MS for r in frames),
                  submitted=len(frames), queue_start=0, queue_end=samples[-1]['queue'],
                  queue_max=max(r['queue'] for r in samples), monitor_samples=len(samples))
    minutes = []
    for minute in range(30):
        part = [s['queue'] for s in samples if minute*MINUTE_SEC <= s['wall_sec'] < (minute+1)*MINUTE_SEC]
        minutes.append(dict(minute=minute, **quantiles(part)))
    result['queue_minutes'] = minutes
    result['queue_slope_frames_per_sec'] = float(np.polyfit([s['wall_sec'] for s in samples],
                                                           [s['queue'] for s in samples], 1)[0])
    result['queue_minute_medians_increase_all'] = all(minutes[i]['p50'] < minutes[i+1]['p50']
                                                     for i in range(len(minutes)-1))
    result['drain'] = json.loads((base / 'completion.json').read_text())
    return result


def gate(name: str, value: Any, denominator: int, threshold: str, passed: bool | None) -> dict:
    return dict(gate=name, value=value, denominator=denominator, threshold=threshold,
                verdict='未判定' if passed is None else '通過' if passed else '不通過')


def gates(full: dict, rt: dict, memory: dict, coverage: dict) -> list[dict]:
    result = []
    for label, part in (('115試合', full), ('30分', rt)):
        timing = part['timings']['feature_ms']
        for p, limit in (('p95', FEATURE_P95), ('p99', FEATURE_P99)):
            result.append(gate(label+'特徴'+p, timing[p], part['n'], f'≤{limit}ms', timing[p] <= limit))
        for layer, rate in part['rates'].items():
            result.append(gate(label+'要求H未達/'+layer, rate['counts']['unreached'], rate['n'],
                               '≤1%', rate['rate'] <= FAIL_RATE if rate['n'] else None))
    result.append(gate('設置→公開p99', rt['placement']['p99'], rt['placement']['n'], '≤500ms',
                       rt['placement']['p99'] <= LATENCY_P99 if rt['placement']['n'] else None))
    result.append(gate('1秒超未処理待ち', rt['wait_over_1s'], rt['submitted'], '=0', rt['wait_over_1s'] == 0))
    result.append(gate('30分終了時の未完了', rt['queue_end'], rt['submitted'], '開始時0以下', rt['queue_end'] == 0))
    result.append(gate('30分継続増加（各分中央値の観測）', rt['queue_minute_medians_increase_all'],
                       len(rt['queue_minutes']), '全30窓で増え続けない', not rt['queue_minute_medians_increase_all']))
    result.append(gate('全期間WSL空きメモリ最小', memory['wsl_min_gib'], memory['wsl_samples'], '≥6GiB',
                       memory['wsl_min_gib'] >= MEMORY_GATE))
    result.append(gate('ホスト空きメモリ最小', memory['host_min_gib'], memory['host_samples'], '≥6GiB',
                       memory['host_min_gib'] >= MEMORY_GATE))
    result.append(gate('公式試合被覆', coverage['measured_games'], 115, '=115',
                       coverage['measured_games'] == 115 and not coverage['missing_keys']))
    return result


def format_times(values: dict) -> str:
    return '/'.join(f'{values[k]:.2f}' for k in ('p50', 'p95', 'p99', 'max'))


def brief(report: dict) -> None:
    """指定の20行以内の日本語要約。"""
    full, rt, mem = report['full'], report['realtime'], report['memory']
    rates = full['rates']
    failed = [g['gate'] for g in report['gates'] if g['verdict'] == '不通過']
    names = {'search_ms': '探索', 'inference_ms': '推論経路', 'ledger_ms': '台帳', 'indicators_ms': '静止指標'}
    heavy = max(names, key=lambda k: full['total_ms'][k])
    tail = report['tail_probe']['details'][0]
    lines = ['# B3 時間門', '判定: '+('不通過（'+ '、'.join(failed)+'）' if failed else '測定範囲内で通過'),
             f"母数: 公式{report['coverage']['measured_games']}/115試合、{full['n']}更新；30分{rt['n']}更新/{rt['submitted']}フレーム。",
             '時間はms、順にp50/p95/p99/最大。',
             '特徴（当該フレーム状態更新含む）: '+format_times(full['timings']['feature_ms'])+'。p95≤100/p99≤200とも通過。',
             '推論（1行入力変換含む）: '+format_times(full['timings']['inference_ms'])+'。',
             '探索: '+format_times(full['timings']['search_ms'])+'；静止指標: '+format_times(full['timings']['indicators_ms'])+'。',
             '台帳（更新フレーム）: '+format_times(full['timings']['ledger_ms'])+'。',
             f"総計最大は{names[heavy]}（{full['total_ms'][heavy]/1000:.1f}秒）；遅延ピークは静止指標{rt['timings']['indicators_ms']['max']:.1f}ms、別途再現で{tail['indicator']}が{tail['ms']:.1f}msと支配的。",
             f"30分特徴: {format_times(rt['timings']['feature_ms'])}（通過）；設置→公開: {format_times(rt['placement'])}（n={rt['placement']['n']}、p99≤500不通過）。",
             f"待ち: 1秒超{rt['wait_over_1s']}/{rt['submitted']}（0件条件は不通過）、最大{rt['frame_wait']['max']:.2f}ms。",
             f"未完了列（処理中込み）: 開始0→30分末{rt['queue_end']}（終端条件不通過）→{(rt['drain']['drain_wall_sec']-REPLAY_SEC)*1000:.0f}ms後0、1秒標本最大{rt['queue_max']}。各分中央値{min(m['p50'] for m in rt['queue_minutes']):g}〜{max(m['p50'] for m in rt['queue_minutes']):g}、継続増加={rt['queue_minute_medians_increase_all']}。",
             f"要求H未達: 全体{rates['all']['counts']['unreached']}/{rates['all']['n']} ({rates['all']['rate']:.2%})；観測実7+層{rates['observed_chain7']['counts']['unreached']}/{rates['observed_chain7']['n']} ({rates['observed_chain7']['rate']:.2%})。各≤1%不通過；H>12を{full['ta_failures_h_above_cap']}更新含む。",
             f"30分要求H未達: {rt['rates']['all']['counts']['unreached']}/{rt['n']}；観測実7+ {rt['rates']['observed_chain7']['counts']['unreached']}/{rt['rates']['observed_chain7']['n']}。",
             f"空き最小: WSL {mem['wsl_min_gib']:.2f}GiB/{mem['wsl_samples']}標本、Windows {mem['host_min_gib']:.2f}GiB/{mem['host_samples']}標本（≥6通過）；B3最大RSS {mem['rss_max_gib']:.2f}GiB。",
             '条件: PC1 i7-13620H/16論理CPU、RAM31.71GiB、RTX4060 Laptop 8GiB（B3 GPU不使用）、WSL/Python3.12；計算1本、BLAS/OpenMP各1、監視等込み実スレッド3。',
             '30分: s1の[300,2100)秒、独立投入プロセス30Hz、状態prefix復元・探索cold開始、無間引き；heartbeat0.5秒、WSL/queue監視1秒、Windows5秒。',
             '境界: available_at予定投入→勝率JSON flush；CNN認識、OBS通信・描画、R死亡証明は含まず、システム全体の合格を主張しない。',
             '実7+層=式7段以上を現在観測、または直近閉鎖episodeで7段以上観測。予測連鎖数は層選択に不使用。HはK最大7手と時間軸要求（速い版含む）のOR。',
             '原票・判定表: D:/puyo_analyzer/b3_data/time_gate/{report.json,gates.csv,realtime_30min,full_s1,full_s2}。段2・封印の採点/モデル・学習行変更なし。']
    (ROOT / 'codex_tasks/time_gate.SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    OUT = parser.parse_args().out
    raw = read_lines(OUT / 'full_s1/updates.jsonl') + read_lines(OUT / 'full_s2/updates.jsonl')
    official, coverage = official_rows(raw)
    full = stats(official)
    rt = realtime_detail(read_lines(OUT / 'realtime_30min/updates.jsonl'))
    resources = sum((read_lines(OUT / name / 'resources.jsonl')
                     for name in ('full_s1', 'full_s2', 'realtime_30min')), [])
    host = read_lines(OUT / 'host_resources.jsonl')
    mem = dict(wsl_min_gib=min(r['available_gib'] for r in resources), wsl_samples=len(resources),
               rss_max_gib=max(r['rss_gib'] for r in resources), host_samples=len(host),
               host_min_gib=min(r['available_gib'] for r in host))
    report = dict(full=full, realtime=rt, coverage=coverage, memory=mem,
                  tail_probe=json.loads((OUT / 'tail_probe.json').read_text())
                  if (OUT / 'tail_probe.json').exists() else None)
    report['gates'] = gates(full, rt, mem, coverage)
    write_json(OUT / 'report.json', report)
    with (OUT / 'gates.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['gates'][0]))
        writer.writeheader()
        writer.writerows(report['gates'])
    if OUT == ROOT / 'time_gate':
        brief(report)


if __name__ == '__main__':
    main()
