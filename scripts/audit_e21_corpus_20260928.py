"""148原票を全件走査し、死亡候補・生還・通知欠測を分けて監査する。"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts import train_exchange_event_models_v2_20260927 as data
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.board import COLOR_EMPTY, COLOR_UNKNOWN, DEATH_COL, DEATH_ROW

OUT = Path('logs/e21/corpus')


def episodes(raw: dict, ids: np.ndarray, video: str, game: int, side: str) -> list[dict]:
    """STABLE原票の占有区間を、クリア・勝敗・欠測に分離する。"""
    rows, active = [], None
    labels = set(float(v) for v in raw['won'][ids] if v in (0., 1.))
    outcome = next(iter(labels)) if len(labels) == 1 else None
    for idx in ids:
        cell = int(raw['grids'][idx, DEATH_ROW, DEATH_COL])
        occupied = cell not in (COLOR_EMPTY, COLOR_UNKNOWN)
        stamp = float(raw['t_sec'][idx])
        if occupied and active is None:
            active = dict(video=video, game=game, side=side, start_sec=stamp,
                          end_sec=None, outcome='unresolved', won=outcome, notifications=[])
            rows.append(active)
        if active is not None:
            trigger = float(raw['chain_trigger_sec'][idx])
            mechanism = str(raw['chain_mechanism'][idx])
            if mechanism and trigger >= active['start_sec']:
                notice = dict(trigger_sec=trigger, mechanism=mechanism)
                if notice not in active['notifications']:
                    active['notifications'].append(notice)
            if cell == COLOR_EMPTY:
                active.update(end_sec=stamp, outcome='cleared_before_end')
                active = None
    if active is not None:
        active.update(end_sec=float(raw['t_sec'][ids[-1]]),
            outcome='survived_winner' if outcome == 1 else 'terminal_loser' if outcome == 0 else 'unlabelled')
    return rows


def video_job(path: Path) -> dict:
    """全試合・両側を走査し、候補のない敗北側も分母へ残す。"""
    raw = data.read_npz(path)
    rows, counts = [], Counter()
    for game in np.unique(raw['game_idx']):
        for side in data.SIDES:
            ids = np.flatnonzero((raw['game_idx'] == game) & (raw['side'] == side))
            ids = ids[np.argsort(raw['t_sec'][ids], kind='stable')]
            if not len(ids):
                continue
            labels = set(float(v) for v in raw['won'][ids] if v in (0., 1.))
            counts['game_sides'] += 1
            counts['labelled_loser_sides'] += labels == {0.}
            counts['labelled_winner_sides'] += labels == {1.}
            counts['unknown_sides'] += len(labels) != 1
            found = episodes(raw, ids, path.stem, int(game), side)
            rows.extend(found)
            counts['loser_sides_with_candidate'] += labels == {0.} and bool(found)
    counts.update(r['outcome'] for r in rows)
    counts['candidate_episodes'] = len(rows)
    counts['known_survivor_candidates'] = sum(r['won'] == 1 for r in rows)
    counts['candidate_outcome_not_proven'] = sum(r['won'] != 1 for r in rows)
    counts['terminal_loser_with_recorded_notice'] = sum(r['outcome'] == 'terminal_loser' and bool(r['notifications']) for r in rows)
    value = dict(video=path.stem, rows=rows, counts=dict(counts), fields=list(raw))
    save_json(OUT/'videos'/f'{path.stem}.json', value)
    return value


def main() -> None:
    """ティア確認済み148件を除外なしで監査し、不足列を明示する。"""
    paths = sorted(data.RAW.glob('*.npz'))
    assert len(paths) == 148
    tiers = pd.read_csv(data.SHARED/'data/video_tier_index_2026-08-07.tsv', sep='\t').set_index('video_name')
    counts, unavailable = Counter(), []
    for path in paths:
        assert any(t in str(tiers.loc['video_'+path.stem, 'tier']) for t in data.v1.ALLOWED_TIERS)
        value = video_job(path)
        counts.update(value['counts'])
        if not (data.SHARED/'data/frames'/f'video_{path.stem}.mp4').exists():
            unavailable.append(path.stem)
        print(path.stem, value['counts']['candidate_episodes'], flush=True)
    save_json(OUT/'SUMMARY.json', dict(videos=len(paths), counts=dict(counts),
        exact_runtime_hold_correct=None, source_video_unavailable=unavailable,
        limitation='原票はSTABLE抽出。非STABLEの発火通知・event.before_board・ばたんきゅー確定時刻なし。実行時の正しい保留件数は同定不能。terminal_loserは勝敗ラベルによる代理であり死亡画像確認ではない。',
        source=str(data.RAW)))


if __name__ == '__main__':
    main()
