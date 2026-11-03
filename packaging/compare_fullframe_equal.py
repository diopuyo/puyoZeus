"""F 条件 (全コマ・時間制約なし) の流し直し 2 本の出力を比べる (読み取りのみ)。

確定記録 (計時以外の全列で一致するはずのもの) と、非同期の揺れ (OFF 同士でも揺れるもの) を分けて数える。
  確定記録: review_data.csv・認識行 (spool/recognition_rows.pickle)・B3 更新 (b3_updates.jsonl)・v4.9f 更新 (v49f_updates.jsonl)。
           件数が両側で同じで、同じ鍵の行が全列一致なら identical。
  非同期の揺れ: 評価行 (evaluations.pickle)・状態更新・ama 更新・公開行数・表示確率の差 (frame ごとの最後の行)。
           OFF 同士の組 (baseline) の揺れの幅と比べ、各組が幅の内に収まるかを判定する。幅そのものも結果へ残す。
元は <work>/live_fast_20261023/tools/compare_equal.py・prob_diff.py (live-fast 担当の台) で、判定を足した。
使い方: python -B compare_fullframe_equal.py --worktree W --out OUT.json --baseline NAME --pair NAME 左 右 [--pair ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

SAMPLES = 10
WIDTH_FACTOR = 1.5             # 組の揺れが baseline の揺れのこの倍まで (+ 下限) なら幅の内
COUNT_FLOOR = 2                # 件数の比較の下限 (baseline が 0 でも 2 件までは幅の内)
PROB_FLOOR_PT = .5             # 確率差 (pt) の比較の下限
# 幅の判定に使う項目。評価行・状態更新の「違う件数」は、ON の腕が X の欄 (暫定 X など) を足すため OFF との組では構造の違いで
# ほぼ全行が違う (OFF 同士でも状態更新は全行の計時以外の列が揺れる)。判定には使わず、参考として残す。
CRITERIA = ('ama_different', 'probability_nonzero', 'probability_p95_pt', 'probability_max_pt')
DETERMINISTIC_UPDATES = ('b3', 'v49f')
PICKLES = ('recognition_rows', 'evaluations', 'state_updates')


def csv_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return {row['frame_index']: row for row in csv.DictReader(stream)}


def compare_csv(left: Path, right: Path) -> dict:
    a, b = csv_rows(left), csv_rows(right)
    common = a.keys() & b.keys()
    rows, columns = [], {}
    for key in sorted(common, key=int):
        changed = [c for c in a[key] if a[key].get(c) != b[key].get(c)]
        if changed:
            rows.append(key)
            for column in changed:
                columns[column] = columns.get(column, 0)+1
    return dict(left=len(a), right=len(b), common=len(common), different=len(rows), samples=rows[:SAMPLES],
                columns=dict(sorted(columns.items(), key=lambda item: -item[1])[:20]))


def keyed(rows: Any, semantic: Any) -> dict:
    """frame を鍵に、同じ frame の行は出た順の列にまとめる (frame が無い行は通し番号)。"""
    out: dict = {}
    for i, row in enumerate(rows):
        key = row.get('frame', f'#{i}') if isinstance(row, dict) else f'#{i}'
        out.setdefault(key, []).append(semantic(row))
    return out


def compare_pickle(left: Path, right: Path, semantic: Any, pickle_stream: Any) -> dict:
    if not (left.exists() and right.exists()):
        return dict(missing=[str(p) for p in (left, right) if not p.exists()])
    a, b = keyed(pickle_stream(left), semantic), keyed(pickle_stream(right), semantic)
    common = a.keys() & b.keys()
    different = sorted((k for k in common if a[k] != b[k]), key=str)
    return dict(left_keys=len(a), right_keys=len(b), left_rows=sum(map(len, a.values())),
                right_rows=sum(map(len, b.values())), common=len(common), different=len(different),
                samples=[str(k) for k in different[:SAMPLES]])


def last_by_frame(path: Path, pickle_stream: Any, field: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in pickle_stream(path/'spool/evaluations.pickle'):
        if isinstance(row, dict) and row.get(field) is not None:
            out[row['frame']] = float(row[field])
    return out


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, max(0, math.ceil(q/100*len(ordered))-1))] if ordered else 0.


def prob_diff(left: Path, right: Path, pickle_stream: Any) -> dict:
    """評価行の表示確率 (pt) と優劣スコアの差 (frame ごとの最後の行)。"""
    result = {}
    for field, scale in (('probability', 100.), ('advantage', 1.)):
        a, b = last_by_frame(left, pickle_stream, field), last_by_frame(right, pickle_stream, field)
        diffs = [abs(a[k]-b[k])*scale for k in sorted(a.keys() & b.keys())]
        result[field] = dict(n=len(diffs), nonzero=sum(d > 0 for d in diffs), p50=round(percentile(diffs, 50), 4),
                             p95=round(percentile(diffs, 95), 4), p99=round(percentile(diffs, 99), 4),
                             max=round(max(diffs, default=0.), 4))
    return result


def compare_pair(worktree: Path, left: Path, right: Path) -> dict:
    sys.path.insert(0, str(worktree))
    from scripts.measure_live_x_effects import compare_runs, semantic
    from scripts.measure_live_x_timing import pickle_stream
    result: dict[str, Any] = dict(left=str(left), right=str(right))
    result['review_data_csv'] = compare_csv(left/'review_data.csv', right/'review_data.csv')
    for name in PICKLES:
        result[name] = compare_pickle(left/f'spool/{name}.pickle', right/f'spool/{name}.pickle', semantic, pickle_stream)
    result['updates'] = compare_runs(left, right)
    result['prob_diff'] = prob_diff(left, right, pickle_stream)
    result['deterministic'] = deterministic(result)
    return result


def same_counts(entry: dict, *keys: str) -> bool:
    values = [entry.get(k) for k in keys]
    return None not in values and len(set(values)) == 1


def deterministic(pair: dict) -> dict:
    """確定記録が全件一致か。記録が無い・件数が違う・1 件でも違えば False (0 件を一致と読まない)。"""
    csv_ = pair['review_data_csv']
    rows = pair['recognition_rows']
    checks = {'review_data.csv': csv_['different'] == 0 and csv_['common'] > 0 and same_counts(csv_, 'left', 'right', 'common'),
              'recognition_rows': rows.get('different') == 0 and rows.get('common', 0) > 0
              and same_counts(rows, 'left_rows', 'right_rows') and same_counts(rows, 'left_keys', 'right_keys', 'common')}
    for kind in DETERMINISTIC_UPDATES:
        update = pair['updates'].get(kind, {})
        checks[kind+'_updates'] = update.get('different') == 0 and update.get('common', 0) > 0 \
            and same_counts(update, 'left', 'right', 'common')
    populations = dict(review_data=csv_['common'], recognition_rows=rows.get('common'),
                       b3=pair['updates'].get('b3', {}).get('common'), v49f=pair['updates'].get('v49f', {}).get('common'))
    return dict(identical=all(checks.values()), checks=checks, populations=populations)


def async_width(pair: dict) -> dict[str, float]:
    """非同期の揺れの大きさ (件数と確率差 pt)。"""
    updates, prob = pair['updates'], pair['prob_diff']['probability']
    return dict(evaluations_different=pair['evaluations'].get('different', -1),
                state_updates_different=pair['state_updates'].get('different', -1),
                ama_different=updates.get('ama', {}).get('different', -1),
                probability_nonzero=prob['nonzero'], probability_p95_pt=prob['p95'], probability_max_pt=prob['max'])


def within(pair_width: dict[str, float], base_width: dict[str, float]) -> dict[str, bool]:
    """各項目が baseline (OFF 同士) の幅の WIDTH_FACTOR 倍 (+ 下限) 以内か。"""
    out = {}
    for key, value in pair_width.items():
        floor = PROB_FLOOR_PT if key.endswith('_pt') else COUNT_FLOOR
        out[key] = value >= 0 and value <= WIDTH_FACTOR*max(base_width[key], 0)+floor
    return out


def baseline_width(pairs: dict[str, dict], baselines: list[str]) -> dict[str, float]:
    """OFF 同士の組すべての揺れの、項目ごとの最大。1 組だけだと偶然の小ささに引きずられるため、使える組を全部使う。"""
    widths = [async_width(pairs[name]) for name in baselines]
    return {key: max(width[key] for width in widths) for key in widths[0]}


def verdict(pairs: dict[str, dict], baselines: list[str]) -> dict:
    """baselines (OFF 同士の組) の幅に対し、他の組を、確定記録の一致と非同期の揺れの幅 (CRITERIA) で判定する。"""
    base = baseline_width(pairs, baselines)
    result: dict[str, Any] = dict(baselines=list(baselines), baseline_width=base, criteria=list(CRITERIA), rule=dict(
        width_factor=WIDTH_FACTOR, count_floor=COUNT_FLOOR, prob_floor_pt=PROB_FLOOR_PT))
    for name, pair in pairs.items():
        width = async_width(pair)
        inside = {k: v for k, v in within(width, base).items() if k in CRITERIA}
        if name in baselines:
            inside = {k: True for k in inside}
        result[name] = dict(deterministic_identical=pair['deterministic']['identical'], width=width, within_width=inside,
                            all_within_width=all(inside.values()), is_baseline=name in baselines,
                            passed=pair['deterministic']['identical'] and all(inside.values()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', nargs='+', required=True, help='OFF 同士の組の名前 (幅は項目ごとの最大)')
    parser.add_argument('--pair', nargs=3, action='append', metavar=('NAME', 'LEFT', 'RIGHT'), required=True)
    args = parser.parse_args()
    pairs = {name: compare_pair(args.worktree, Path(left), Path(right)) for name, left, right in args.pair}
    result = dict(pairs=pairs, verdict=verdict(pairs, args.baseline))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps(result['verdict'], ensure_ascii=False, default=str)[:4000], flush=True)


if __name__ == '__main__':
    main()
