"""評価セット (zenchi 30先セット1 全57試合) の試合範囲と勝者ラベルを作る (2026-10-01)。

ラベル源: 8/31 の set1 公式試合割当 (official_game_assignment、WIN パネル数値の増分由来)。
現行 zenchi 一致率の official_games.json (第41〜57試合) と同じ源・同じ抽出規則。
照合: (1) 既存 official_games.json と第41〜57試合で完全一致 (2) WIN パネル目視読取の勝者列
(PANEL_WINNERS、各試合開始+5秒のパネル数値の増分。第57試合は 3411秒の「やった!」と 28-29 表示)。
出力: logs/eval_set/labels.json (試合・勝者・公式範囲・パート)。
"""
from __future__ import annotations

import json
from pathlib import Path

SET1_RUN = Path('/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/verify/'
                'zenchi_two_sets_review_source_2026-08-31/runs/schema=v1/video=video_zenchi_c0BQoMJwwQU/'
                'build=build-a2d3d70a5d2ebfb316f4eb993fab9a8ddcf079045917330aea65abfc59feaa24/'
                'attempt=zenchi-review-set1-20260831')
EXISTING = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/review_zenchi_part3/official_games.json')
OUT = Path('logs/eval_set/labels.json')
GAMES = 57
MS_PER_SEC = 1000
# 20試合ずつのレビュー動画パート (videos/review/zenchi_two_30first_redesign48_graph_v2_2026-08-31 の manifest)
PARTS = ((1, 1, 20, 0.0, 1357.633), (2, 21, 40, 1357.633, 2580.566), (3, 41, 57, 2580.566, 3427.166))
# WIN パネル目視 (logs/eval_set/panel_montage.png)。1文字 = 1試合の勝者 (A=1P, B=2P)。
PANEL_WINNERS = ('AAAAAABBBBBBABAABABBBABBBBBBBABAAABAABABAAABABBAABABABABA')


def official_games() -> list[dict]:
    """公式試合割当から57試合の勝者と範囲を読む。"""
    rows = []
    for file in sorted((SET1_RUN/'events').glob('*.jsonl')):
        for line in file.read_text().splitlines():
            event = json.loads(line)
            if event.get('event_type') != 'official_game_assignment':
                continue
            timing = event['timing']
            rows.append(dict(game=event['payload']['official_game_number'],
                             winner=event['payload']['winner_side'],
                             start=timing['occurred_earliest_ms']/MS_PER_SEC,
                             end=timing['occurred_latest_ms']/MS_PER_SEC))
    rows.sort(key=lambda row: row['game'])
    assert [row['game'] for row in rows] == list(range(1, GAMES+1))
    return rows


def part_of(game: int) -> int:
    """試合番号から所属パートを返す。"""
    return next(part for part, first, last, _, _ in PARTS if first <= game <= last)


def checks(rows: list[dict]) -> dict:
    """既存ラベル・パネル目視との一致を母数つきで返す。"""
    existing = {row['game']: row for row in json.loads(EXISTING.read_text())}
    same_existing = sum({k: rows[g-1][k] for k in existing[g]} == existing[g] for g in existing)
    panel = ['1P' if c == 'A' else '2P' for c in PANEL_WINNERS]
    assert len(panel) == GAMES
    mismatch = [row['game'] for row in rows if row['winner'] != panel[row['game']-1]]
    return dict(existing_match=same_existing, existing_total=len(existing),
                panel_match=GAMES-len(mismatch), panel_total=GAMES, panel_mismatch_games=mismatch)


def main() -> None:
    """ラベルを書き出し、照合結果を表示する。"""
    rows = official_games()
    for row in rows:
        row['part'] = part_of(row['game'])
    result = dict(games=rows, parts=[dict(part=p, first=f, last=l, start_sec=s, end_sec=e)
                                     for p, f, l, s, e in PARTS],
                  winners={side: sum(r['winner'] == side for r in rows) for side in ('1P', '2P')},
                  checks=checks(rows), source=str(SET1_RUN))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(dict(winners=result['winners'], checks=result['checks']), ensure_ascii=False))


if __name__ == '__main__':
    main()
