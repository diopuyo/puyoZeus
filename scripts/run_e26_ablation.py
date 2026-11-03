"""E26: E22固定入力で各要素を独立比較し、悪化しない要素だけを選ぶ。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from scripts import run_e17_ablation_20260928 as prior
from scripts.run_e22_death_formula_20260928 import OPTIONS as BASE
from scripts.run_e23_multilanding_20260928 import ProgressTrace
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e26')
VARIANTS = dict(ledger=dict(pending_ledger=True), color=dict(color_score_safety=True),
    multi=dict(multi_landing_death=True), safety=dict(landing_state_safety=True),
    recovery=dict(completion_recovery=True))
WORKERS = 2


def launch(task: tuple[str, str]) -> None:
    """固定入力を二並列までで再生し、完了済み単位は再利用する。"""
    variant, source = task
    dest = prior.directory(variant, source)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'runner.log').open('a') as stream:
        subprocess.run([sys.executable, '-m', 'scripts.run_e26_ablation',
            '--variant', variant, '--source', source], stdout=stream,
            stderr=subprocess.STDOUT, check=True)


def report() -> dict:
    """未丸めE22実測との比較で、単独のq悪化を除外する。"""
    prior.LOSS_MAX, prior.AGREEMENT_MIN = .513079, .8262
    baseline = json.loads(Path('logs/e22/on/METRICS.json').read_text())
    results = {'E22': baseline}
    for variant in VARIANTS:
        results[variant] = prior.report(variant)
    selected = [v for v in ('ledger', 'color', 'multi')
                if results[v]['q']['log_loss'] <= baseline['q']['log_loss']]
    options = dict(BASE, **{k: value for v in selected for k, value in VARIANTS[v].items()})
    result = dict(results=results, selected=selected, options=options)
    save_json(OUT/'ABLATION.json', result)
    return result


def main() -> None:
    """単独条件の測定を途中予測の実装・選定より先に完了する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--source', choices=prior.ALL_SOURCES)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    prior.OUT, prior.AuditTrace = OUT, ProgressTrace
    if args.source:
        prior.locked_worker(args.variant, args.source, dict(BASE, **VARIANTS[args.variant]))
        return
    if args.report_only and args.variant:
        prior.LOSS_MAX, prior.AGREEMENT_MIN = .513079, .8262
        print(json.dumps(prior.report(args.variant), ensure_ascii=False), flush=True)
        return
    if not args.report_only:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(launch, [(v, s) for v in VARIANTS for s in prior.ALL_SOURCES]))
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
