"""E23: E22固定入力で複数着弾の採否と新規死亡判定を再生する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from scripts import run_e17_ablation_20260928 as prior
from scripts.replay_exchange_event_20260926 import compare
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from src.exchange_event_multilanding import MAX_SEARCH_NODES, MAX_LANDINGS

OUT = Path('logs/e23')
OPTIONS = dict(death_guard=True, confirmed_death_hold=True, death_formula_guard=True,
               multi_landing_death=True)
LOSS_MAX, AGREEMENT_MIN = .513079, .8262
SCENE = (2755., 2771.)
P2_MAX = .05
PROGRESS_FRAMES = 3000


class ProgressTrace(prior.AuditTrace):
    """再生フレーム数と入力時刻だけを進捗として公開する。"""
    def __call__(self, overlay: Any, inputs: tuple) -> None:
        super().__call__(overlay, inputs)
        if self.frames % PROGRESS_FRAMES == 0:
            print(dict(source=self.source, frames=self.frames, t_sec=inputs[3]), flush=True)


def worker(source: str, off: bool) -> None:
    """新フラグ以外はE22と同一のモデル・入力・採用条件を使う。"""
    prior.OUT = OUT
    prior.AuditTrace = ProgressTrace
    options = dict(OPTIONS, multi_landing_death=not off)
    variant = 'off_final' if off else 'on'
    prior.locked_worker(variant, source, options)
    if off:
        save_json(OUT/'OFF_EQUIVALENCE.json', compare(
            Path('logs/e22/on/review'), prior.directory(variant, source)))


def launch(source: str) -> None:
    """固定した5記録（3動画・zenchi・review）を最大2プロセスで再生する。"""
    dest = prior.directory('on', source)
    if (dest/'DONE.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'runner.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'scripts.run_e23_multilanding_20260928',
                        '--source', source], stdout=log, stderr=subprocess.STDOUT, check=True)


def first_low(path: Path, end: float | None = None) -> float | None:
    """採用確率が2P 5%以下へ達した初時刻を、指定窓の開始から求める。"""
    data = np.load(path)
    mask = (data['t_sec'] >= SCENE[0]) & (1-data['display_p1'] <= P2_MAX)
    if end is not None:
        mask &= data['t_sec'] <= end
    times = data['t_sec'][mask]
    return float(times[0]) if len(times) else None


def death_audit() -> dict:
    """元の発火母数を保存し、新規と前倒しを同じイベント・側で対応させる。"""
    metrics = prior.report_base.death_metrics
    metrics.OUT = Path('logs/e22/on')
    baseline = metrics.deaths()
    metrics.OUT = OUT/'on'
    rows = metrics.deaths()
    key = lambda r: (r['source'], r['game_idx'], r['exchange_id'], r['side'])
    old = {key(r): r for r in baseline}
    changed = []
    for row in rows:
        if row['source'] not in SOURCES:
            continue
        before = old.get(key(row))
        if before is None or row['first_sec'] < before['first_sec']:
            changed.append(dict(row, change='new' if before is None else 'earlier',
                baseline_first_sec=None if before is None else before['first_sec']))
    return dict(all_firings=rows, baseline_firings=baseline, three_videos_changed=changed,
                new=sum(r['change'] == 'new' for r in changed),
                earlier=sum(r['change'] == 'earlier' for r in changed),
                survived=sum(r['false_positive'] is True for r in changed),
                unlabelled=sum(r['winner'] is None for r in changed))


def report() -> dict:
    """事前登録した閾値を丸め前の数値で判定する。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, LOSS_MAX, AGREEMENT_MIN
    result = prior.report('on')
    baseline = first_low(Path('logs/e22/on/review/display.npz'))
    first = first_low(prior.directory('on', 'review')/'display.npz', SCENE[1])
    from scripts.measure_e23_display_20260928 import main as measure_display
    measure_display()
    shown = json.loads((OUT/'SCENE_DISPLAY.json').read_text())
    result['scene'] = dict(shown, selected_first_sec=first, baseline_selected_first_sec=baseline,
        advance_sec=None if shown['first_sec'] is None else shown['baseline_first_sec']-shown['first_sec'])
    result['gates']['scene'] = shown['scene_gate']
    audit = death_audit()
    result['three_videos'] = {k: v for k, v in audit.items() if k in ('new', 'earlier', 'survived', 'unlabelled')}
    result['candidate'] = all(result['gates'].values())
    save_json(OUT/'DEATH_AUDIT.json', audit)
    save_json(OUT/'on/METRICS.json', result)
    return result


def main() -> None:
    """評価前にプロトコルを登録し、後から基準を変更しない。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=prior.ALL_SOURCES)
    parser.add_argument('--off', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    prior.OUT = OUT
    if args.source:
        worker(args.source, args.off)
        return
    if not args.report_only:
        save_json(OUT/'PROTOCOL.json', dict(baseline='E22 / 541b44a', options=OPTIONS,
            q_max=LOSS_MAX, zenchi_min=AGREEMENT_MIN, max_false=1, reference_firings=28,
            scene=SCENE, p2_max=P2_MAX, criterion='初到達時刻がE22より早い',
            max_search_nodes=MAX_SEARCH_NODES, max_landings=MAX_LANDINGS,
            uncertainty='UNKNOWN・探索打切り・生存応手ありは新たな死亡にしない'))
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(launch, prior.ALL_SOURCES))
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
