"""B15の5分推移、RSS回帰、境界前後のRSSと認識品質を集計する。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from scripts.analyze_live_b9 import lines, windows, WINDOW_SEC
from scripts.analyze_live_b12 import report as contamination
from scripts.measure_live_b6 import save

MB, HOUR, BOUNDARY_TOLERANCE = 1_000_000, 3600.0, 10.0
BOUNDARY_RADIUS, BOUNDARY_GUARD = 15.0, 2.0


def regression(rows: list[dict]) -> dict:
    if len(rows) < 2:
        return dict(samples=len(rows), slope_mb_per_hour=None, r_squared=None)
    x = np.array([r['at'] for r in rows], dtype=float)
    y = np.array([r['rss_bytes']/MB for r in rows])
    x = (x-x[0])/HOUR
    slope, intercept = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
    residual, total = float(np.sum((y-(slope*x+intercept))**2)), float(np.sum((y-y.mean())**2))
    return dict(samples=len(rows), slope_mb_per_hour=float(slope), intercept_mb=float(intercept),
                r_squared=1-residual/total if total else None, elapsed_hours=float(x[-1]),
                first_mb=float(y[0]), last_mb=float(y[-1]), delta_mb=float(y[-1]-y[0]))


def rss_boundary(rows: list[dict], at: float) -> dict:
    before = [r['rss_bytes']/MB for r in rows if at-BOUNDARY_RADIUS <= r['at'] < at-BOUNDARY_GUARD]
    after = [r['rss_bytes']/MB for r in rows if at+BOUNDARY_GUARD < r['at'] <= at+BOUNDARY_RADIUS]
    delta = float(np.median(after)-np.median(before)) if before and after else None
    return dict(at=at, before_samples=len(before), after_samples=len(after), delta_mb=delta,
                decreased=None if delta is None else delta < 0)


def match_boundaries(expected: list[dict], actual: list[dict]) -> dict:
    """順序を保ち、対応数最大→時刻差最小で境界を突き合わせる。"""
    from scripts.analyze_live_b8 import match_times, MAX_SHIFT_SEC
    # 元の2秒許容を固定比で拡大する。番号の付け直しでは一致扱いにしない。
    scale = BOUNDARY_TOLERANCE/MAX_SHIFT_SEC
    def normalized(rows: list[dict]) -> list[dict]:
        return [dict(side=0, game=0, t_sec=r['t_sec']/scale) for r in rows]
    pairs = match_times(normalized(expected), normalized(actual), 't_sec')
    paired_left, paired_right = {a for a, _ in pairs}, {b for _, b in pairs}
    return dict(expected=len(expected), actual=len(actual), matched=len(pairs), tolerance_sec=BOUNDARY_TOLERANCE,
        missing=[r for i, r in enumerate(expected) if i not in paired_left],
        extra=[r for i, r in enumerate(actual) if i not in paired_right],
        pairs=[dict(expected=expected[a], actual=actual[b], delta_sec=actual[b]['t_sec']-expected[a]['t_sec'])
               for a, b in pairs])


def runtime_report(root: Path) -> dict:
    plan = json.loads((root/'plan.json').read_text())
    path, start, end = root/'realtime', plan['start'], plan['end']
    runtime, resources = lines(path/'runtime.jsonl'), lines(path/'resources.jsonl')
    with np.load(path/'recognition.npz') as data:
        origin = float(np.median(data['captured_at']-data['t_sec']))
        last_frame = float(data['t_sec'][-1])
    measured = [r for r in resources if origin+start <= r['at'] < origin+end]
    games = [r for r in runtime[-1]['game_starts'] if start <= r['t_sec'] < end]
    changes = []
    for game in games[1:]:
        tick = next((r for r in runtime if any(g['game'] == game['game'] for g in r['game_starts'])), None)
        changes.append(dict(game=game, **rss_boundary(measured, tick['at'])) if tick else dict(game=game))
    timeline = windows(path, start, end)
    for row in timeline:
        row['rss_p50_mb'] = row['rss_p50']/MB if row['rss_p50'] is not None else None
    result = dict(windows=timeline, rss_regression=regression(measured),
        rss_after_first_5min=regression([r for r in measured if r['at'] >= origin+start+WINDOW_SEC]),
        rss_boundaries=changes, boundaries=match_boundaries(plan['games'], games),
        expected_games=plan['expected_games'], expected_internal_boundaries=plan['expected_internal_boundaries'],
        detected_games=len(games), detected_internal_boundaries=max(0, len(games)-1),
        last_recognized_sec=last_frame, reached_end=last_frame >= end-1,
        note='MB=10^6 bytes。RSS低下は物理常駐量の観測で、非低下だけでは破棄失敗と断定できない。')
    save(root/'report.json', result)
    contamination(path, start, end)
    return result


def quality_report(root: Path) -> dict:
    from scripts.analyze_live_b8 import load_audit, board_comparison
    plan = json.loads((root/'plan.json').read_text())
    output = []
    for start, end in plan['comparison_windows']:
        # 履歴は全60分を再生して保持。集計だけを事前固定した20分へ絞る。
        reference = load_audit(root/'reference/recognition.npz', start, end)
        current = load_audit(root/'realtime/recognition.npz', start, end)
        result = board_comparison(reference, current)
        output.append(dict(start=start, end=end, **{k: result[k] for k in
            ('reference_notifications', 'candidate_notifications', 'placements',
             'matched_board_events', 'visible_cells', 'visible_differences')}))
    result = dict(windows=output, sampled_seconds=sum(r['end']-r['start'] for r in output),
                  note='30Hz正規化後の間引きなし参照。可視セル差は対応した確定盤面の母数。')
    save(root/'quality.json', result)
    return result
