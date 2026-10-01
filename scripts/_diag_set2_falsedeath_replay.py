"""本番再生に診断観測だけを挿入する。srcへの書込みは行わない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path('/mnt/d/puyo_analyzer/wt_evalset')
OUT = ROOT / 'logs/eval_set/set2_falsedeath'
TARGETS = (4417.4, 4598.033333333334, 4620.7, 6059.283333333334, 6172.6,
           6309.933333333333, 6379.2, 7002.85)
TOLERANCE = 0.02
PROGRESS_INTERVAL = 30.0


def install_progress() -> None:
    """結果に触れず、長い再生の到達時刻だけを記録する。"""
    from src.exchange_event_overlay import ExchangeEventOverlay
    original = ExchangeEventOverlay.update
    previous, started = float('-inf'), time.monotonic()

    def update(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal previous
        result = original(self, *args, **kwargs)
        stamp = args[3]
        if stamp - previous >= PROGRESS_INTERVAL:
            print(f'再生 {stamp:.3f}s / 実行 {time.monotonic()-started:.1f}s', flush=True)
            previous = stamp
        return result

    ExchangeEventOverlay.update = update


def plain(value: Any) -> Any:
    """診断用オブジェクトをJSONに変換する。"""
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, '_grid'):
        return plain(value._grid)
    if hasattr(value, '__dict__'):
        return plain(vars(value))
    return value


def install_trace(directory: Path) -> None:
    """死亡入力と履歴を対象時刻だけ採取する。"""
    from src import exchange_death_inputs as inputs
    original = inputs.death_inputs

    def traced(projection: Any, overlay: Any, latest: tuple, incoming: list) -> dict:
        context = original(projection, overlay, latest, incoming)
        stamp = overlay.tracker._score_elapsed + overlay._start
        # 評価時計は呼出元のフレーム時刻を使う。
        import inspect
        stamp = inspect.currentframe().f_back.f_locals['t_sec']
        if any(abs(stamp - t) < TOLERANCE for t in TARGETS):
            data = dict(t_sec=stamp, context=context, latest=latest,
                        chains=projection.safety.chains, elapsed=projection.safety.elapsed,
                        ledger=projection.safety.ledger, guards=projection.safety.guards,
                        history=[list(h)[-3:] for h in overlay._history])
            (directory / f'trace_{stamp:.6f}.json').write_text(
                json.dumps(plain(data), ensure_ascii=False, indent=1), encoding='utf-8')
        return context

    inputs.death_inputs = traced


def main() -> None:
    """共通CLIから本番フラグを適用して再生する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('record', type=Path)
    parser.add_argument('name')
    parser.add_argument('--mode', default='baseline', choices=('baseline', 'guard', 'score', 'combined', 'attributed', 'corroborated'))
    args = parser.parse_args()
    directory = OUT / args.mode / args.name
    directory.mkdir(parents=True, exist_ok=True)
    install_progress()
    if args.mode == 'baseline':
        install_trace(directory)
    else:
        from scripts._diag_set2_falsedeath_patch import install
        install(args.mode)
        install_trace(directory)
    sys.argv = ['replay', str(args.record), '--out', str(directory), '--production-exchange-event']
    from scripts.replay_exchange_event_20260926 import main as replay
    replay()


if __name__ == '__main__':
    main()
