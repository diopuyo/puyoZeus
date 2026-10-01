"""発火前予測 (Phase 2) の記録再生器。E36b 本番構成 (run_e36b.options) で5記録を再生する。

- 入力: exev の本番構成記録 `logs/pending_expiry/full/records` (e36b_on の元記録、読むだけ)
- 構成: run_e36b.options() に `prefire_exchange_prediction` を足すだけ。E35 証明器は d5_runtime で 0f28a04 版を固定
- 出力: logs/prefire_prediction/replay/<variant>/<source>/ (display.npz / events.jsonl / status.json / prefire_trace.npz)
使い方: PYTHONPATH=. python -m scripts.run_prefire_replay_20260930 --variant off|on --source S [--compare]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXEV = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer')
RECORDS = EXEV / 'logs/pending_expiry/full/records'
BASELINE = EXEV / 'logs/pending_expiry/e36b_on/on'
BASELINE_DIRS = {'q_7gc4TgFig': 'renders/q_7gc4TgFig/on', 'fcXG83vInDY': 'renders/fcXG83vInDY/on',
                 'mia8KCjr52g': 'renders/mia8KCjr52g/on', 'review': 'review', 'zenchi': 'zenchi'}
OUT = Path('logs/prefire_prediction/replay')
VARIANTS = ('off', 'on', 'on_mc', 'bestplay', 'stable')   # on=構成A、on_mc=構成B、bestplay=Phase 3 最善手 (hazard 不使用)


def options(variant: str, latency: float = 0.0, evaluator: str = 'full') -> dict:
    """本番 E36b の再生オプション。on だけ発火前予測を足す (off は引数自体を渡さない = 既存呼出と同一)。"""
    from scripts import run_e36b
    base = run_e36b.options()
    if variant == 'stable':
        return dict(base, prefire_best_play_stable=True, prefire_best_play_latency=latency)
    if variant == 'bestplay':
        return dict(base, prefire_best_play=True, prefire_best_play_latency=latency,
                    prefire_best_play_evaluator=evaluator)
    if variant == 'on':
        return dict(base, prefire_exchange_prediction=True)
    if variant == 'on_mc':
        return dict(base, prefire_exchange_prediction=True, prefire_counter_mc=True)
    return base


def run(variant: str, source: str, compare_baseline: bool, latency: float = 0.0, tag: str = '',
        evaluator: str = 'full') -> dict:
    """1記録を再生する。off は本番記録とバイト一致を照合できる (原則3: 台がまず既知の値を再現する)。"""
    from scripts import d5_runtime
    name = variant + tag
    d5_runtime.OUT = OUT / f'runtime_{name}_{source}'   # 並列起動で manifest の一時ファイルが衝突しないよう分ける
    d5_runtime.install()
    from scripts.replay_exchange_event_20260926 import compare, replay
    opts = options(variant, latency, evaluator)
    model = 'v4' if opts.get('count_sync') or opts.get('e16') else 'v3'
    dest = OUT / name / source
    dest.mkdir(parents=True, exist_ok=True)
    result = replay(RECORDS / f'{source}.jsonl.gz', dest, Path('models/exchange_event_' + model), True,
                    None, **opts)
    result['options'] = opts
    if compare_baseline:
        result['equivalence'] = compare(BASELINE / BASELINE_DIRS[source], dest)
    (dest / 'DONE.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--source', choices=tuple(BASELINE_DIRS), required=True)
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--latency', type=float, default=0.0, help='bestplay の結果が届くまでの遅れ秒')
    parser.add_argument('--tag', default='', help='出力先の名前に足す接尾辞 (例 _l0)')
    parser.add_argument('--evaluator', default='full', choices=('full', 'fast', 's3', 'gfe'))
    args = parser.parse_args()
    print(json.dumps(run(args.variant, args.source, args.compare, args.latency, args.tag, args.evaluator),
                     ensure_ascii=False, default=str), flush=True)


if __name__ == '__main__':
    main()
