"""H1の履歴欠測時間・実際の証明抑止・表示時間を原票から集計する。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.falsedeath_replay_20261001 import OUT, records

DEST = OUT/'v2d'


def display_blocked(directory: Path, rows: list[dict], game: int) -> dict:
    """直近の抑止評価が表示のS3_landingに使われたフレームを数える。"""
    with np.load(directory/'display.npz') as data:
        time, games, source = data['t_sec'], data['game_idx'], data['source']
        eligible = np.flatnonzero(games == game)
        if not rows or not len(eligible):
            return dict(blocked_display_frames=0, blocked_display_seconds=0.)
        stamps = np.asarray([r['t_sec'] for r in rows])
        indices = np.searchsorted(stamps, time[eligible], side='right')-1
        valid = indices >= 0
        flags = np.array([r['blocked'] for r in rows], dtype=bool)
        selected = eligible[valid & flags[np.maximum(indices, 0)] & (source[eligible] == 'S3_landing')]
        # 最後の行を先へ外挿せず、同一試合の次の表示行までだけ積算する。
        following = np.minimum(selected+1, len(time)-1)
        duration = np.where(games[following] == game, time[following]-time[selected], 0.)
        return dict(blocked_display_frames=len(selected), blocked_display_seconds=float(duration.sum()))


def summarize(directory: Path) -> dict:
    """履歴不完全である時間と、証明が成立して実際に止めた時間を混同しない。"""
    data = json.loads((directory/'history_integrity_audit.json').read_text())
    games = []
    for game in data['games']:
        audits = [r for r in data['rows'] if r['game'] == game['game']]
        blocked = [r for r in audits if r['blocked']]
        complete_at = game['match_start_observed_sec']
        end = complete_at if complete_at is not None else game['last_observation_sec']
        games.append(dict(**game, incomplete_seconds=end-game['first_observation_sec'],
            blocked_evaluations=len(blocked), first_blocked_sec=blocked[0]['t_sec'] if blocked else None,
            last_blocked_sec=blocked[-1]['t_sec'] if blocked else None,
            **display_blocked(directory, audits, game['game'])))
    return dict(games=games, observed_games=len(games),
        incomplete_games=sum(not g['history_complete'] for g in games),
        blocked_games=sum(g['blocked_evaluations'] > 0 for g in games),
        blocked_evaluations=sum(g['blocked_evaluations'] for g in games),
        incomplete_seconds=sum(g['incomplete_seconds'] for g in games),
        blocked_display_frames=sum(g['blocked_display_frames'] for g in games),
        blocked_display_seconds=sum(g['blocked_display_seconds'] for g in games))


def main() -> None:
    """短区間は常に、全体は梯子通過時だけ別母数で保存する。"""
    ladder = json.loads((DEST/'LADDER.json').read_text())
    groups = dict(short={s:summarize(DEST/'short'/s) for s in ('c5','mia8KCjr52g','review')})
    if ladder['passed']:
        groups['full'] = {s:summarize(OUT/'replay/v2d'/s) for s in records()}
    with (DEST/'HISTORY.json').open('x', encoding='utf-8') as stream:
        json.dump(groups, stream, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
