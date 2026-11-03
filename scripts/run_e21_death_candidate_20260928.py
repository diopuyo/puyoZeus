"""E21: E19候補へ死亡候補ゲートだけを追加し、固定入力で採否を決める。"""
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
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.replay_exchange_event_20260926 import compare

OUT = Path('logs/e21')
OPTIONS = dict(death_guard=True, confirmed_death_hold=True, death_candidate_guard=True)
LOSS_MAX, AGREEMENT_MIN, SCENE_MIN = .511079+.002, .8235, .85
SCENE = (2633.8, 2634.75)


class Trace(prior.AuditTrace):
    """候補の真偽・保留通知を表示指標と別に監査する。"""
    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.gate: Any = None
        self.scene: list[dict] = []

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        super().__call__(overlay, inputs)
        self.gate = overlay._candidate_gate
        if self.source == 'review' and SCENE[0] <= inputs[3] <= SCENE[1]:
            chain = overlay.tracker.latest_chain('2P')
            self.scene.append(dict(t_sec=inputs[3], p1=overlay.tracker.probability,
                source=overlay.tracker.source, chain=None if chain is None else dict(
                    trigger_sec=chain.trigger_sec, predicted_chain_count=chain.predicted_chain_count,
                    predicted_final_score=chain.predicted_final_score)))


def worker(source: str, off: bool) -> None:
    """実行条件ごとに独立し、OFFと既存E19の一致も保存する。"""
    prior.OUT = OUT
    trace = Trace(source)
    prior.AuditTrace = lambda _: trace
    variant = 'off' if off else 'on'
    options = dict(death_guard=True, confirmed_death_hold=True) if off else OPTIONS
    prior.locked_worker(variant, source, options)
    dest = prior.directory(variant, source)
    if trace.gate is not None:
        save_json(dest/'death_candidates.json', trace.gate.audit)
    if trace.scene:
        save_json(dest/'scene.json', trace.scene)
    if off:
        save_json(OUT/'OFF_EQUIVALENCE.json', compare(Path('logs/e19/hold/review'), dest))


def launch(source: str) -> None:
    """再生は2並列に制限し、148動画監査の枠を一つ残す。"""
    dest = prior.directory('on', source)
    if (dest/'DONE.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'runner.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'scripts.run_e21_death_candidate_20260928',
                        '--source', source], stdout=log, stderr=subprocess.STDOUT, check=True)


def report() -> dict:
    """未丸めの固定基準で判定し、母数を省略しない。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, LOSS_MAX, AGREEMENT_MIN
    value = prior.report('on')
    data = np.load(prior.directory('on', 'review')/'display.npz')
    mask = (data['t_sec'] >= SCENE[0]) & (data['t_sec'] <= SCENE[1])
    value['scene'] = dict(frames=int(mask.sum()), minimum=float(data['display_p1'][mask].min()))
    value['gates']['scene'] = value['scene']['minimum'] >= SCENE_MIN
    value['candidate'] = all(value['gates'].values())
    save_json(OUT/'on/METRICS.json', value)
    return value


def main() -> None:
    """評価前に閾値・候補定義・除外なしの監査対象を登録する。"""
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
        save_json(OUT/'PROTOCOL.json', dict(options=OPTIONS, baseline='E19', q_max=LOSS_MAX,
            zenchi_min=AGREEMENT_MIN, false_max=1, reference_firings=28, scene=SCENE,
            scene_min=SCENE_MIN, erasure='発火後のSTABLE確定盤面で同色4個以上減少',
            candidate='STABLE確定盤面[DEATH_ROW=1,DEATH_COL=2]が既知ぷよで占有',
            scope='固定記録3動画+zenchi+reviewと全148原票。未観測は欠測と記録'))
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(launch, prior.ALL_SOURCES))
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
