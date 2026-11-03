"""採点専用勝敗。WIN原票を時刻で対応し、原票のない動画だけ死亡で補完する。"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import os
import pandas as pd
import pyarrow.parquet as parquet
from scripts.exchange_review_io import jsonl

ROWS_REL = 'puyo_analyzer/share/results/pc2/fast_pipeline/runs/v6/rows'
B3_ROWS = Path('D:/'+ROWS_REL) if os.name == 'nt' else Path('/mnt/d/'+ROWS_REL)
MIN_OVERLAP = .5
TIME_EPS = 1e-9
SIDES = frozenset(('1P', '2P'))


def death_games(rows: list[dict]) -> dict[int, dict]:
    """activeを無視し、S3以外の最初の死亡を採用。両側死亡は全試合を棄却。"""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if int(row['game']) > 0:
            grouped[int(row['game'])].append(row)
    result = {}
    for game, part in grouped.items():
        part.sort(key=lambda r: r['t_sec'])
        deaths = [r for r in part if r.get('dead') and not str(r.get('source', '')).startswith('S3')]
        sides = set().union(*(set(r['dead']) for r in deaths))
        reason = '両側dead' if len(sides) > 1 else '死亡なし/S3のみ'
        valid = len(sides) == 1 and sides <= SIDES
        winner = ('2P' if deaths[0]['dead'][0] == '1P' else '1P') if valid else None
        result[game] = dict(start=float(part[0]['t_sec']), end=float(part[-1]['t_sec']),
            death_winner=winner, death_reason='最初の非S3死亡' if valid else reason,
            first_dead_t=float(deaths[0]['t_sec']) if deaths else None)
    return result


def panel_spans(rows: pd.DataFrame) -> list[dict]:
    """対称化前のorientation=0だけを読み、WINの欠落・矛盾を保存する。"""
    rows = rows[rows['orientation'] == 0]
    result = []
    for game, part in rows.groupby('game_idx'):
        votes = set(part['won_p1'].dropna().tolist())
        valid = len(votes) == 1 and votes <= {0., 1.}
        winner = ('1P' if next(iter(votes)) else '2P') if valid else None
        result.append(dict(game=int(game), start=float(part['t_sec'].min()),
            end=float(part['t_sec'].max()), winner=winner, has_win=bool(votes),
            reason='WIN' if valid else 'WIN欠落/矛盾'))
    return result


def match_panel(span: dict, panels: list[dict]) -> dict:
    """b3_source同様、重なり最大・短い区間の50%以上。番号は比較しない。"""
    ranked = sorted([(max(0., min(span['end'], p['end'])-max(span['start'], p['start'])), i)
                     for i, p in enumerate(panels)], reverse=True)
    if not ranked or ranked[0][0] <= 0:
        return dict(panel_winner=None, panel_reason='時間重なりなし')
    overlap, index = ranked[0]
    panel = panels[index]
    frac = overlap/max(min(span['end']-span['start'], panel['end']-panel['start']), TIME_EPS)
    tied = len(ranked) > 1 and abs(overlap-ranked[1][0]) <= TIME_EPS
    valid = frac >= MIN_OVERLAP and not tied
    reason = '重なり同率' if tied else '重なり不足' if not valid else panel['reason']
    return dict(panel_game=panel['game'], overlap_fraction=frac,
        panel_winner=panel['winner'] if valid else None, panel_reason=reason)


@lru_cache(maxsize=None)
def label_audit(run: Path, b3_root: Path = B3_ROWS) -> dict[int, dict]:
    """動画にWIN原票があれば未対応試合を死亡で埋めず、欠落理由を残す。"""
    games = death_games(list(jsonl(run/'death.jsonl')))
    video = int(run.name.split('_')[-1])
    path = b3_root/f'video_{video}.parquet'
    columns = ['game_idx', 'orientation', 't_sec', 'won_p1']
    available = path.exists() and 'won_p1' in parquet.read_schema(path).names
    panels = panel_spans(pd.read_parquet(path, columns=columns)) if available else []
    has_win = any(p['has_win'] for p in panels)
    for game, span in games.items():
        match = match_panel(span, panels) if has_win else dict(panel_winner=None, panel_reason='動画WINなし')
        winner = match['panel_winner'] if has_win else span['death_winner']
        reason = match['panel_reason'] if has_win else span['death_reason']
        span.update(match, game=game, video=video, winner=winner, reason=reason,
            label_source='B3 rows WIN' if has_win else '非S3最初dead')
        span['agreement'] = (span['death_winner'] == match['panel_winner']
                             if span['death_winner'] and match['panel_winner'] else None)
    matches: dict[int, list[dict]] = defaultdict(list)
    for span in games.values():
        if has_win and span['winner']:
            matches[span['panel_game']].append(span)
    for duplicates in matches.values():
        if len(duplicates) > 1:
            for span in duplicates:
                span.update(winner=None, reason='同じWIN試合へ複数対応')
    return games


def corrected_winners(run: Path, b3_root: Path = B3_ROWS) -> dict[int, str]:
    """旧exchange_review_v2_fit.winnersの置換先。推論入力には渡さない。"""
    return {g: r['winner'] for g, r in label_audit(run, b3_root).items() if r['winner']}
