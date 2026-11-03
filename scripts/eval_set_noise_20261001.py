"""評価セットの物差しのゆらぎ (再学習・試合標本) と最小検出差を測る (2026-10-01、定義は logs/eval_set/METRIC.md)。

入力: 再生済みの display.npz (logs/eval_set/replay_rejudge, replay_cli、第3パートの一部は wt_nextfix の既存再生)。
出力: logs/eval_set/NOISE.json。参考の対差 (補正−旧、E19−本番) は報告のみで採否に使わない。
使い方: python -m scripts.eval_set_noise_20261001
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import eval_set_score_20261001 as scorer

ROOT = Path('logs/eval_set')
NEXTFIX = Path('/mnt/d/puyo_analyzer/wt_nextfix')
GATE = NEXTFIX/'logs/next_shift_rejudge/GATE.json'
SEEDS = range(5)
KINDS = ('orig', 'T')
METRICS = ('m_time', 'm_cp', 'm_cp25', 'm_cp50', 'm_cp75', 'm_frame', 'm_agree')
BOOT_N, BOOT_SEED = 10_000, 20261001
Z_SUM = 1.959964 + 0.841621      # 両側 α=.05 + 検出力 .8
CI = (2.5, 97.5)
CHUNKS = ('c1', 'c2', 'c3', 'c4', 'c5', 'c6')   # 第1〜40試合の収集区間 (METRIC.md)


def rejudge_paths(variant: str) -> list[Path]:
    """再判定構成の7区間。第3パートは自前の再生が無ければ wt_nextfix の既存再生を使う。"""
    base = ROOT/'replay_rejudge'/variant/'on'
    own = base/'zenchi/display.npz'
    part3 = own if own.exists() else NEXTFIX/'logs/next_shift_train/replay'/variant/'on/zenchi/display.npz'
    return [base/'renders'/chunk/'on/display.npz' for chunk in CHUNKS] + [part3]


def cli_paths(variant: str) -> list[Path]:
    """再生CLI (本番指定) の7区間。"""
    return [ROOT/'replay_cli'/variant/part/'display.npz' for part in (*CHUNKS, 'zenchi')]


def configurations() -> dict[str, list[Path]]:
    """採点する全構成と入力。"""
    result = {f'J_{k}_rs{s}': rejudge_paths(f'J_{k}_rs{s}') for k in KINDS for s in SEEDS}
    result.update(prod=cli_paths('prod'), e19=cli_paths('e19'))
    return result


def per_game(rows: list[dict], metric: str) -> np.ndarray:
    """試合別の値 (一致率は試合別の hits と frames の2列)。"""
    if metric == 'm_agree':
        return np.array([[r['late_hits'], r['late_frames']] for r in rows], dtype=float)
    if metric == 'm_cp':
        return np.array([np.mean([r['m_cp25'], r['m_cp50'], r['m_cp75']]) for r in rows])
    if metric == 'm_frame':
        return np.array([[r['loss_sum'], r['frames']] for r in rows], dtype=float)
    return np.array([r[metric] for r in rows])


def aggregate(values: np.ndarray) -> np.ndarray | float:
    """試合別の値を集計値へ (比の指標は和の比、他は平均)。最後の軸の前で集計。"""
    if values.ndim >= 2 and values.shape[-1] == 2:
        return values[..., 0].sum(axis=-1) / values[..., 1].sum(axis=-1)
    return values.mean(axis=-1)


def bootstrap_index(games: int) -> np.ndarray:
    """試合の復元抽出の添字 (全構成で共通の乱数列)。"""
    return np.random.default_rng(BOOT_SEED).integers(0, games, size=(BOOT_N, games))


def boot_ci(values: np.ndarray, index: np.ndarray) -> list[float]:
    """1構成の集計値の95%区間。"""
    samples = aggregate(values[index])
    return [float(x) for x in np.percentile(samples, CI)]


def paired(a: np.ndarray, b: np.ndarray, index: np.ndarray) -> dict:
    """同じ試合で比べた差 (a−b) の点推定・95%区間・標準誤差・最小検出差。"""
    diff = aggregate(a) - aggregate(b)
    samples = aggregate(a[index]) - aggregate(b[index])
    se = float(np.std(samples, ddof=1))
    low, high = np.percentile(samples, CI)
    return dict(diff=float(diff), ci95=[float(low), float(high)], se=se, mdd=Z_SUM*se)


def seed_spread(values: list[float]) -> dict:
    """同じ特徴の random_state 5通りの散らばりと、再学習だけによる最小検出差。"""
    arr = np.array(values)
    sd = float(np.std(arr, ddof=1))
    return dict(values=[float(v) for v in arr], mean=float(arr.mean()), sd=sd,
                range=float(arr.max()-arr.min()), mdd_single=Z_SUM*np.sqrt(2)*sd,
                mdd_mean5=Z_SUM*np.sqrt(2/len(arr))*sd)


def q_spread() -> dict:
    """q (4試合) の同じ10構成の散らばり (GATE.json)。"""
    gate = json.loads(GATE.read_text())
    return {kind: seed_spread(gate['q'][kind]) for kind in KINDS}


def score_all() -> dict[str, dict]:
    """全構成を採点して scores/<name>.json に保存する。"""
    results = {}
    for name, paths in configurations().items():
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            print(f'skip {name}: 未再生 {missing}', flush=True)
            continue
        results[name] = scorer.score(paths)
        (ROOT/'scores').mkdir(parents=True, exist_ok=True)
        (ROOT/'scores'/f'{name}.json').write_text(json.dumps(results[name], ensure_ascii=False, indent=1))
    return results


def spreads(results: dict[str, dict]) -> dict:
    """指標ごとの再学習ゆらぎ (旧・補正それぞれ5 seed)。"""
    out = {}
    for metric in METRICS:
        out[metric] = {kind: seed_spread([results[f'J_{kind}_rs{s}']['summary'][metric] for s in SEEDS])
                       for kind in KINDS if all(f'J_{kind}_rs{s}' in results for s in SEEDS)}
    return out


def comparisons(results: dict[str, dict], index: np.ndarray) -> dict:
    """試合標本ゆらぎ: 本番の区間、seed 違いの対 (純粋な再学習差)、参考の対差。"""
    pairs = {f'orig_rs{s}-orig_rs0': (f'J_orig_rs{s}', 'J_orig_rs0') for s in range(1, 5)}
    pairs.update({f'T_rs{s}-orig_rs{s}': (f'J_T_rs{s}', f'J_orig_rs{s}') for s in SEEDS})
    pairs['e19-prod'] = ('e19', 'prod')
    out = {}
    for metric in METRICS:
        values = {n: per_game(r['rows'], metric) for n, r in results.items()}
        row = dict(prod_ci95=boot_ci(values['prod'], index) if 'prod' in values else None)
        row['pairs'] = {k: paired(values[a], values[b], index) for k, (a, b) in pairs.items()
                        if a in values and b in values}
        out[metric] = row
    return out


def main() -> None:
    """採点・ゆらぎ・最小検出差を NOISE.json にまとめる。"""
    results = score_all()
    games = len(next(iter(results.values()))['rows'])
    summary = dict(games=games, configurations={n: r['summary'] for n, r in results.items()},
                   seed_spread=spreads(results), q_seed_spread=q_spread(),
                   bootstrap=dict(n=BOOT_N, seed=BOOT_SEED, unit='match'),
                   match_sampling=comparisons(results, bootstrap_index(games)))
    (ROOT/'NOISE.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(dict(seed_spread={m: {k: (round(v['sd'], 6), round(v['range'], 6)) for k, v in s.items()}
                                       for m, s in summary['seed_spread'].items()},
                          q={k: (round(v['sd'], 6), round(v['range'], 6)) for k, v in summary['q_seed_spread'].items()}),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
