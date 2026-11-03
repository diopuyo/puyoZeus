"""保存本番表示と静止入力を対応付け、監査専用領域へ抽出する。"""
from __future__ import annotations
import gzip
import json
import os
from pathlib import Path
from typing import Any

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import joblib
import numpy as np
import pandas as pd
from src.exchange_event_evaluator import StaticInput, build_features
from src.exchange_event_features import D_COLUMNS, G_COLUMNS

OUT = Path('/mnt/d/puyo_analyzer/review_2026-10-03/static_audit')
EVAL = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set')
BASE = EVAL / 'falsedeath_fix'
SOURCES = [*(f'c{i}' for i in range(1, 7)), 'zenchi', *(f's{i}' for i in range(7))]
RESERVE = 6 * 1024 ** 3
FPS = 30


def guard() -> None:
    """常に低優先度で6GiBの余裕を要求する。"""
    os.nice(max(0, 10 - os.nice(0)))
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(values['MemAvailable'].split()[0]) * 1024 < RESERVE:
        raise RuntimeError('利用可能メモリ6GiB未満')


def save_json(name: str, value: Any) -> None:
    """許可された管理外領域だけに保存する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def parse(source: str) -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
    """直前の静止呼出と各表示行を対応させ、盤面標本は毎秒保存する。"""
    cache, calls, states, samples = {}, [], [], []
    current, last_sample, call_id = None, -np.inf, -1
    with gzip.open(BASE / f'collect/records/{source}.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            if r['kind'] == 'update':
                current = r['args']['tuple']
            elif r['kind'] == 'static':
                cache[r['key']] = r['input']['d_features']['array']
            elif r['kind'] == 'static_call':
                calls.append((cache[r['key']], r['m0_probability_1p'], r['elapsed_sec'], r['source_side'], current[3]))
                call_id = len(calls) - 1
            elif r['kind'] == 'display':
                sides = [current[0]['namespace'][s]['namespace'] for s in ('p1', 'p2')]
                snap = current[1]['namespace']
                elapsed = calls[-1][2] + r['t_sec'] - calls[-1][4] if calls else np.nan
                states.append((r['t_sec'], call_id, snap['forecast_p1'], snap['net_balance_capped'], elapsed))
                stable = all(s['state']['state'] == 'STABLE' and s['confirmed_board'] for s in sides)
                if stable and r['t_sec'] - last_sample >= 1 - 1e-8:
                    samples.append(dict(t=r['t_sec'], sides=sides, source=source))
                    last_sample = r['t_sec']
    thresholds = json.loads(Path('models/exchange_event_v1/manifest.json').read_text())['elapsed_thresholds']
    x = np.array([build_features(StaticInput(np.array(d), p, e, s), tuple(thresholds))[1]
                  for d, p, e, s, t in calls])
    return pd.DataFrame(states, columns=['t', 'call', 'forecast', 'net', 'model_elapsed']), x, samples


def extract(source: str, lr: Any, hgb: Any) -> None:
    """R1sを直接再計算し、保存本番確率との同一性を数値確認する。"""
    guard()
    states, x, samples = parse(source)
    with np.load(BASE / f'replay/v2h2b/{source}/display.npz') as z:
        frame = pd.DataFrame({k: z[k] for k in z.files if z[k].ndim == 1})
    np.testing.assert_allclose(frame.t_sec, states.t, atol=1e-8, rtol=0)
    frame = pd.concat([frame.rename(columns={'t_sec': 't'}), states.drop(columns='t')], axis=1)
    pr, pl = hgb.predict_proba(x)[:, 1], lr.predict_proba(x)[:, 1]
    ids = frame['call'].to_numpy()
    frame['p_lr'], frame['p_r1_calc'] = pl[ids], pr[ids]
    frame['source_record'] = source
    valid = (ids >= 0) & frame.source.eq('G_fe').to_numpy()
    delta = np.abs(frame.display_p1.to_numpy() - frame.p_r1_calc.to_numpy())
    report = dict(source=source, frames=len(frame), gfe=int(valid.sum()),
                  exact=int((delta[valid] < 1e-10).sum()), max_error=float(delta[valid].max()),
                  old_lr_exact=int((abs(frame.display_p1.to_numpy()[valid] - pl[ids[valid]]) < 1e-10).sum()))
    frame['input_verified'] = valid & (delta < 1e-10)
    frame.to_pickle(OUT / f'{source}_frames.pkl')
    np.savez_compressed(OUT / f'{source}_inputs.npz', x=x)
    with gzip.open(OUT / f'{source}_samples.json.gz', 'wt') as stream:
        json.dump(samples, stream)
    save_json(f'{source}_identity.json', report)
    print(report, flush=True)


def main() -> None:
    """14分割を逐次処理し、共有作業への資源競合を抑える。"""
    guard()
    OUT.mkdir(parents=True, exist_ok=True)
    lr = joblib.load('models/exchange_event_v1/G_fe.joblib')
    hgb = joblib.load('models/exchange_event_v1_r1_hgb/G_fe.joblib')
    for source in SOURCES:
        extract(source, lr, hgb)


if __name__ == '__main__':
    main()
