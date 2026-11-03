"""本番描画CLIで構築し、封印済み二記録を各系列と突き合わせる。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


from collections import Counter
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any
from unittest.mock import patch

import numpy as np

from src import production_config as config
from scripts.verify_static_gfe_cli_20261002 import render_overlay, memory_available

OUT = Path('logs/integration_r1s_dv2d_20261002')
PHASE = Path('/mnt/d/puyo_analyzer/wt_phase/logs/gfe_phase_stage2b_20261002/R1s/replay')
EVAL = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/falsedeath_fix')
EXEV = Path(_public_dev_text('<repo>'))
DURATIONS = {'review': 220., 'q_7gc4TgFig': 120.}
MODES = ('r1s_old', 'r1s_d4', 'v2d', 'integrated')


def audit_saved_result() -> None:
    """保存された最終結果の一致・実モデル・死亡証拠・資産SHAを再検算する。"""
    report = json.loads((OUT/'native_m0/RESULT.json').read_text())
    evidence = dict(sources={}, nice=report['nice'], available_bytes=report['available_bytes'])
    assert report['nice'] == 10 and report['available_bytes'] >= 6*1024**3
    for source, result in report['sources'].items():
        for key in ('r1s_saved_identity', 'v2d_saved_identity'):
            assert all(c['changed'] == 0 for c in result['comparisons'][key]['columns'].values())
        calls = result['runs']['integrated']['calls']
        assert any(c['model'] == 'G_fe' and c['estimator'] == 'HistGradientBoostingClassifier' for c in calls)
        if source == 'review':
            assert any(c['purpose'] == 'landing' and c['estimator'] == 'LogisticRegression' for c in calls)
        directory = OUT/'native_m0'/source/'integrated'
        blocked = 0
        for line in (directory/'events.jsonl').read_text().splitlines():
            for value in json.loads(line)['values']:
                if not value.get('history_integrity', {}).get('blocked'):
                    continue
                blocked += 1
                assert not value['dead_sides']
                for key in ('single_death_proof', 'post_counter_bound'):
                    assert all(not row['dead'] for row in value.get(key, []) if row is not None)
        with np.load(directory/'display.npz') as display:
            stamps = display['t_sec'][(display['source'] == 'unavoidable_death') & (display['t_sec'] >= 2700.)]
            evidence['sources'][source] = dict(blocked_values=blocked,
                first_computed_death_after_2700=float(stamps[0]) if len(stamps) else None)
    model = Path('models/exchange_event_v1_r1_hgb')
    expected = json.loads((model/'manifest.json').read_text())['models']['G_fe']['sha256']
    actual = hashlib.sha256((model/'G_fe.joblib').read_bytes()).hexdigest()
    assert expected == actual
    evidence['static_gfe_sha256'] = actual
    (OUT/'VERIFICATION.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence), flush=True)


def shorten(source: str, original: bool) -> tuple[Path, dict]:
    """入力は読み取り専用。先頭履歴を維持し、短縮記録だけ作業先へ保存する。"""
    from src.exchange_event_record import read_records, encode
    root = EXEV/'logs/pending_expiry/full' if original else EVAL/'collect'
    path = root/'records'/f'{source}.jsonl.gz'
    stream, count, start = read_records(path), 0, None
    header = next(stream)
    target = OUT/f'{source}_{"old" if original else "d4"}.jsonl.gz'
    with gzip.open(target, 'wt', encoding='utf-8') as output:
        output.write(json.dumps(encode(header))+'\n')
        for row in stream:
            if row['kind'] == 'complete':
                break
            if row['kind'] == 'update':
                stamp = row['args'][3]
                start = stamp if start is None else start
                if stamp > start + DURATIONS[source]:
                    break
                count += 1
            output.write(json.dumps(encode(row))+'\n')
        output.write(json.dumps(dict(kind='complete', frames=count))+'\n')
    return target, header


def flags_for(mode: str) -> str:
    """新登録だけを外した対照を作り、既存本番は同一に保つ。"""
    entries = config.EXCHANGE_EVENT_ADOPTED
    if mode.startswith('r1s'):
        entries = tuple(e for e in entries if e.reason != config.DV2D_ADOPTION_REASON)
    if mode == 'v2d':
        entries = tuple(e for e in entries if not e.flag.startswith('--exchange-event-static-gfe-dir'))
    return ' '.join(e.flag for e in entries)


def run(source: str, mode: str) -> dict:
    """main→argparse→generate→overlayを通し、推論呼出しと全表示を保存する。"""
    from scripts import replay_exchange_event_20260926 as replay
    from src.exchange_event_evaluator import FileExchangeModels
    record, header = shorten(source, mode == 'r1s_old')
    flags, counts = flags_for(mode), Counter()
    with patch.object(config, 'exchange_event_flags', lambda: flags):
        overlay = render_overlay(['--production-exchange-event'], record, header)
    original = FileExchangeModels.predict_source_probability

    def predict(models: Any, name: str, values: np.ndarray) -> float:
        purpose = 'landing' if models is overlay.tracker.models.landing_models else 'static_or_s3'
        counts[(purpose, name, type(models.estimators[name]).__name__)] += 1
        return original(models, name, values)

    destination = OUT/source/mode
    destination.mkdir(parents=True, exist_ok=True)
    with patch.object(config, 'exchange_event_flags', lambda: flags), \
            patch.object(sys, 'argv', ['replay', str(record), '--out', str(destination),
                                      '--production-exchange-event']), \
            patch.object(replay, 'ExchangeEventOverlay', lambda *a, **kw: overlay), \
            patch.object(FileExchangeModels, 'predict_source_probability', predict):
        replay.main()
    return dict(flags=shlex.split(flags), calls=[dict(purpose=p, model=n, estimator=k, count=v)
        for (p, n, k), v in counts.items()], available_bytes=memory_available())


def compare(left: Path, right: Path) -> dict:
    """同じ時刻の表示列だけを比較し、値差と由来差を分ける。"""
    with np.load(left/'display.npz') as a, np.load(right/'display.npz') as b:
        times, ai, bi = np.intersect1d(a['t_sec'], b['t_sec'], return_indices=True)
        result = dict(rows=len(times), start=float(times[0]), end=float(times[-1]), columns={})
        for key in a.files:
            if key == 't_sec' or key not in b or a[key].ndim != 1 or len(a[key]) != len(a['t_sec']):
                continue
            x, y = a[key][ai], b[key][bi]
            if x.dtype.kind in 'fiu' and y.dtype.kind in 'fiu':
                same = (x == y) | (np.isnan(x) & np.isnan(y))
                finite = np.isfinite(x) & np.isfinite(y)
                delta = np.abs(x[finite]-y[finite])
                common = np.result_type(x.dtype, y.dtype)
                stored = np.dtype('float32') if 'float32' in (str(x.dtype), str(y.dtype)) else common
                rounded = (x.astype(stored) == y.astype(stored)) | (np.isnan(x) & np.isnan(y))
                result['columns'][key] = dict(changed=int((~same).sum()),
                    left_dtype=str(x.dtype), right_dtype=str(y.dtype),
                    changed_at_storage_precision=int((~rounded).sum()),
                    max_abs=float(delta.max()) if len(delta) else None)
            else:
                result['columns'][key] = dict(changed=int((x != y).sum()))
        return result


def main() -> None:
    """証拠を上書きせず、各対照の保存表示への一致も明記する。"""
    global OUT
    from src import exchange_event_m0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--native-m0', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    OUT = args.output
    if args.native_m0:
        OUT = OUT/'native_m0'
    else:
        exchange_event_m0.CPU_THREADS = 1
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'RESULT.json').exists() and not args.report_only:
        raise FileExistsError('完成済み結果を保持するため、--outputで新しい出力先を指定する')
    report = dict(nice=os.nice(0), available_bytes=memory_available(),
                  m0_cpu_threads=exchange_event_m0.CPU_THREADS, sources={})
    previous = json.loads((OUT/'RESULT.json').read_text()) if args.report_only else None
    for source in DURATIONS:
        runs = {}
        for mode in MODES:
            print(f'{source} {mode}', flush=True)
            runs[mode] = previous['sources'][source]['runs'][mode] if previous else run(source, mode)
        directory = OUT/source
        report['sources'][source] = dict(runs=runs, comparisons={
            'r1s_saved_identity': compare(directory/'r1s_old', PHASE/source),
            'v2d_saved_identity': compare(directory/'v2d', EVAL/'replay/v2d'/source),
            'd4_effect': compare(directory/'r1s_old', directory/'r1s_d4'),
            'dv2d_effect': compare(directory/'r1s_d4', directory/'integrated'),
            'static_gfe_effect': compare(directory/'v2d', directory/'integrated'),
            'combined_vs_r1s': compare(PHASE/source, directory/'integrated')})
        (OUT/'RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('二記録の比較完了', flush=True)


if __name__ == '__main__':
    if '--audit-only' in sys.argv:
        audit_saved_result()
    else:
        main()
