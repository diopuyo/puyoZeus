"""評価セット (zenchi セット1 57試合) の採点器 (2026-10-01、定義は logs/eval_set/METRIC.md)。

入力: 3パートの display.npz (時刻で連結) と labels.json。出力: 試合別の行と集計。
使い方: python -m scripts.eval_set_score_20261001 --name NAME --display P1.npz P2.npz P3.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABELS = Path('logs/eval_set/labels.json')
SCORES = Path('logs/eval_set/scores')
PROBABILITY_EPSILON = 1e-7          # q (aggregate_e3) と同じ
CHECKPOINTS = (.25, .50, .75)
LATE_FRACTION = 2/3                 # 現行 zenchi 一致率の終盤1/3の開始位置
COLUMNS = ('t_sec', 'display_p1', 'display_adv')


def load_display(paths: list[Path]) -> dict[str, np.ndarray]:
    """パートの表示列を時刻順に連結する。時刻の重複は区間の重なりとして拒否する。"""
    parts = []
    for path in paths:
        with np.load(path) as data:
            parts.append({name: np.asarray(data[name], dtype=float) for name in COLUMNS})
    joined = {name: np.concatenate([part[name] for part in parts]) for name in COLUMNS}
    order = np.argsort(joined['t_sec'], kind='stable')
    joined = {name: values[order] for name, values in joined.items()}
    if np.any(np.diff(joined['t_sec']) <= 0):
        raise ValueError('パート間で表示時刻が重複または逆行している')
    return joined


def frame_loss(p1: np.ndarray, winner: str) -> np.ndarray:
    """1フレームごとの log loss (y=勝者が1P)。"""
    if not np.isfinite(p1).all():
        raise ValueError('表示勝率に非有限値がある')
    p = np.clip(p1, PROBABILITY_EPSILON, 1-PROBABILITY_EPSILON)
    return -np.log(p) if winner == '1P' else -np.log(1-p)


def checkpoint_loss(t: np.ndarray, loss: np.ndarray, start: float, end: float, q: float) -> float:
    """開始 + q·試合長 以前で最後の表示フレームの loss。"""
    index = np.searchsorted(t, start + q*(end-start), side='right') - 1
    if index < 0:
        raise ValueError('時点以前に表示フレームがない')
    return float(loss[index])


def game_row(display: dict[str, np.ndarray], game: dict) -> dict:
    """1試合の全指標。"""
    t = display['t_sec']
    mask = (t >= game['start']) & (t < game['end'])
    if not mask.any():
        raise ValueError(f"試合{game['game']}の窓に表示フレームがない")
    loss = frame_loss(display['display_p1'][mask], game['winner'])
    row = dict(game=game['game'], winner=game['winner'], part=game.get('part'),
               frames=int(mask.sum()), m_time=float(loss.mean()), loss_sum=float(loss.sum()))
    for q in CHECKPOINTS:
        row[f'm_cp{round(q*100)}'] = checkpoint_loss(t[mask], loss, game['start'], game['end'], q)
    begin = game['start'] + (game['end']-game['start'])*LATE_FRACTION
    late = (t >= begin) & (t < game['end'])
    sign = 1 if game['winner'] == '1P' else -1
    row['late_frames'] = int(late.sum())
    row['late_hits'] = int(np.count_nonzero(display['display_adv'][late]*sign > 0))
    return row


def summarize(rows: list[dict]) -> dict:
    """57試合を等重みで平均し、フレーム加重の参考値と母数を添える。"""
    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))
    cp = [f'm_cp{round(q*100)}' for q in CHECKPOINTS]
    late_frames = sum(r['late_frames'] for r in rows)
    return dict(games=len(rows), frames=sum(r['frames'] for r in rows), m_time=mean('m_time'),
                **{k: mean(k) for k in cp}, m_cp=float(np.mean([mean(k) for k in cp])),
                m_agree=sum(r['late_hits'] for r in rows)/late_frames,
                agree_hits=sum(r['late_hits'] for r in rows), agree_frames=late_frames,
                m_frame=sum(r['loss_sum'] for r in rows)/sum(r['frames'] for r in rows))


def score(paths: list[Path], labels: Path = LABELS) -> dict:
    """表示列を採点して試合別の行と集計を返す。"""
    games = json.loads(labels.read_text(encoding='utf-8'))['games']
    display = load_display(paths)
    rows = [game_row(display, game) for game in games]
    return dict(summary=summarize(rows), rows=rows, inputs=[str(p) for p in paths])


def main() -> None:
    """採点して logs/eval_set/scores/<name>.json に保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--display', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    result = score(args.display)
    SCORES.mkdir(parents=True, exist_ok=True)
    (SCORES/f'{args.name}.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(dict(name=args.name, **result['summary']), ensure_ascii=False))


if __name__ == '__main__':
    main()
