"""E25の固定条件再生・既定OFF等価性・事前登録基準を検証する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from scripts import run_e17_ablation_20260928 as prior
from scripts import run_e23_multilanding_20260928 as e23
from scripts import audit_e23_deaths_20260928 as audit
from scripts import measure_e23_display_20260928 as display
from scripts.inspect_e24_saved_20260928 import save_rows
from scripts.replay_exchange_event_20260926 import compare
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e25')
OPTIONS = dict(e23.OPTIONS, landing_state_safety=True)
WORKERS = 2
SCENE_DEADLINE = 2766.


class SafetyTrace(e23.ProgressTrace):
    """復元候補の棄却理由と指定場面の各評価を、再生を変えずに保存する。"""
    instances: list = []

    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.completion_rows, self.scene_rows = [], []
        self.recovery_id, self.cursor = None, 0
        self.instances.append(self)

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        super().__call__(overlay, inputs)
        projection, stamp = overlay._landing_projection, inputs[3]
        if projection.safety is None:
            return
        recovery = projection.safety.recovery
        if self.recovery_id is not recovery:
            self.recovery_id, self.cursor = recovery, 0
        self.completion_rows.extend(dict(game=overlay._game, **r) for r in recovery.audit[self.cursor:])
        self.cursor = len(recovery.audit)
        value = projection.last
        if self.source == 'review' and 2755. <= stamp <= 2771. and value and value['t_sec'] == stamp:
            self.scene_rows.append(dict(game=overlay._game, **value))


def worker(source: str, variant: str) -> None:
    """明示ONと二種類のOFF経路を、同じ記録・モデルで再生する。"""
    prior.OUT, prior.AuditTrace = OUT, SafetyTrace
    options = dict(OPTIONS, landing_state_safety=variant == 'on')
    if variant == 'off_e22':
        options['multi_landing_death'] = False
    prior.locked_worker(variant, source, options)
    if SafetyTrace.instances:
        trace = SafetyTrace.instances[-1]
        save_rows(prior.directory(variant, source)/'completion_audit.json', trace.completion_rows)
        save_rows(prior.directory(variant, source)/'scene_evaluations.json',
                  json.loads(json.dumps(trace.scene_rows)))
    scene_path = prior.directory(variant, source)/'scene_evaluations.json'
    if source == 'review' and not scene_path.exists():
        events = [json.loads(s) for s in scene_path.with_name('events.jsonl').read_text().splitlines()]
        save_rows(scene_path, [dict(game=e['game_idx'], **v) for e in events for v in e['values']
            if 2755. <= v['t_sec'] <= 2771. and 'multi_landing' in v])
    if variant != 'on':
        baseline = 'e22' if variant == 'off_e22' else 'e23'
        save_json(OUT/f'{variant}_EQUIVALENCE.json', compare(
            Path(f'logs/{baseline}/on/review'), prior.directory(variant, source)))


def launch(task: tuple[str, str]) -> None:
    """最大2プロセスで固定入力を再生し、標準出力を個別に保持する。"""
    source, variant = task
    dest = prior.directory(variant, source)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'runner.log').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'scripts.run_e25_state_safety_20260928',
            '--source', source, '--variant', variant], stdout=stream, stderr=subprocess.STDOUT, check=True)


def report() -> dict:
    """基準は事前登録のまま、実発火母数と新規誤り件数も必須条件へ含める。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, e23.LOSS_MAX, e23.AGREEMENT_MIN
    result = prior.report('on')
    e23.OUT, display.OUT, audit.OUT = OUT, OUT, OUT
    display.main()
    shown = json.loads((OUT/'SCENE_DISPLAY.json').read_text())
    shown['deadline_sec'] = SCENE_DEADLINE
    shown['scene_gate'] = shown['first_sec'] is not None and shown['first_sec'] <= SCENE_DEADLINE
    save_json(OUT/'SCENE_DISPLAY.json', shown)
    table = OUT/'SCENE_DISPLAY.csv'
    table.write_text(table.read_text().replace('E23_p1_display', 'E25_p1_display'))
    result['scene'] = shown
    result['gates']['scene'] = shown['scene_gate']
    save_json(OUT/'DEATH_AUDIT.json', e23.death_audit())
    audit.main()
    cases = json.loads((OUT/'NEW_DEATH_CASES.json').read_text())
    for change in ('new', 'earlier'):
        rows = [r for r in cases['cases'] if r['change'] == change]
        cases['summary'][change+'_errors'] = sum(r['actually_died'] is False for r in rows)
        cases['summary'][change+'_unresolved'] = sum(r['actually_died'] is None for r in rows)
    save_json(OUT/'NEW_DEATH_CASES.json', cases)
    result['three_videos'] = cases['summary']
    result['gates']['new_errors'] = cases['summary']['new_errors'] <= 1 and cases['summary']['new_unresolved'] == 0
    result['candidate'] = all(result['gates'].values())
    save_json(OUT/'on/METRICS.json', result)
    return result


def main() -> None:
    """事前登録済みの条件だけを走らせ、採否を機械的に保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=prior.ALL_SOURCES)
    parser.add_argument('--variant', choices=('on', 'off_e22', 'off_e23'), default='on')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    prior.OUT = OUT
    if args.source:
        worker(args.source, args.variant)
        return
    if not args.report_only:
        tasks = [('review', 'on'), ('review', 'off_e23'), ('review', 'off_e22')]
        tasks += [(s, 'on') for s in prior.ALL_SOURCES if s != 'review']
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(launch, tasks))
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
