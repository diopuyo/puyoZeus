"""E27構成に一観測候補化だけを追加し、固定閾値で検収する。"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from scripts import run_e17_ablation_20260928 as prior
from scripts.run_e27 import OPTIONS as E27_OPTIONS
from scripts.audit_e29_predictions import PredictionTrace, counts
from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.measure_e23_display_20260928 import displayed, first
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

OUT = Path('logs/e29')
OPTIONS = dict(E27_OPTIONS, midchain_single_observation=True)
WORKERS = 2
LOSS_MAX, AGREEMENT_MIN, DEADLINE = .512778, .8468, 2766.


def worker(source: str, control: bool = False) -> None:
    """同じ記録とモデルを使い、OFFはE27の全出力と照合する。"""
    prior.OUT, prior.AuditTrace = OUT, PredictionTrace
    variant, options = ('off', E27_OPTIONS) if control else ('on', OPTIONS)
    def enriched(record: Path, dest: Path, *args: object, **kwargs: object) -> dict:
        """隠し段上限にも最終得点監査を追加する。"""
        result = replay(Path('logs/e26/records')/record.name, dest, *args, **kwargs)
        save_json(dest/'hidden_final_audit.json', args[2].hidden_summary())
        return result
    prior.replay = enriched
    prior.locked_worker(variant, source, options)
    if control:
        suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        save_json(OUT/f'OFF_{source}.json', compare(Path('logs/e27/on')/suffix, prior.directory(variant, source)))


def launch(task: tuple[str, bool]) -> None:
    """最大二再生で実行し、完了済みは同じ条件でのみ再利用する。"""
    source, control = task
    dest = prior.directory('off' if control else 'on', source)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'runner.log').open('a') as stream:
        command = [sys.executable, '-m', 'scripts.run_e29', '--source', source]
        subprocess.run(command+(['--control'] if control else []),
            stdout=stream, stderr=subprocess.STDOUT, check=True)


def prediction_counts() -> dict:
    """指定3動画の件数と、別枠のzenchi・レビューを分けて保存する。"""
    totals, sources = dict(midchain=Counter(), hidden=Counter()), {}
    for source in prior.ALL_SOURCES:
        dest = prior.directory('on', source)
        sources[source] = {name: counts(json.loads((dest/file).read_text()))
            for name, file in (('midchain', 'midchain_audit.json'), ('hidden', 'hidden_final_audit.json'))}
        if source in SOURCES:
            for name in totals:
                totals[name].update(sources[source][name])
    value = dict(three_videos={k: dict(v) for k, v in totals.items()}, sources=sources)
    save_json(OUT/'PREDICTION_COUNTS.json', value)
    return value


def report() -> dict:
    """事前登録の条件を全て満たした場合だけ動画生成可能とする。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, LOSS_MAX, AGREEMENT_MIN
    value = prior.report('on')
    stamp = first(*displayed(prior.directory('on', 'review')/'display.npz'))
    value['scene'] = dict(first_sec=stamp, deadline_sec=DEADLINE, scenes=1, baseline_first_sec=2771.25)
    value['gates']['scene'] = stamp is not None and stamp <= DEADLINE
    prediction = prediction_counts()['three_videos']
    value['prediction_counts'] = prediction
    value['gates']['final_scores'] = all(v['final_mismatch'] == 0 and v['final_unresolved'] == 0
        for v in prediction.values())
    value['candidate'] = all(value['gates'].values())
    save_json(OUT/'on/METRICS.json', value)
    return value


def main() -> None:
    """閾値固定後にOFF再現とON検収を行い、集計を保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=prior.ALL_SOURCES)
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    prior.OUT = OUT
    assert (OUT/'PREREGISTRATION.md').exists()
    if args.source:
        worker(args.source, args.control)
        return
    if not args.report_only:
        tasks = [(s, off) for off in (True, False) for s in ('review', *SOURCES, 'zenchi')]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(launch, tasks))
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
