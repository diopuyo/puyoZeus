"""ライブ評価 (通知単位の表示EMA) の切替平滑を、保存済みの評価入力で ON/OFF 比較する。

保存記録 (R1b互換の評価入力) を B18b と同じ手順で NotificationExchangeOverlay へ流し、
通知ごとの表示 (latest_display) から、試合境界を含む飛びの件数を ON/OFF で数える。
認識・評価値は変えない (表示のみ)。出力は logs/live_switch_smoothing/<on|off>/<記録>/。
使い方: python -m scripts.verify_live_switch_smoothing --enabled|--disabled --source S [--source S ...]
        python -m scripts.verify_live_switch_smoothing --summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from scripts import measure_switch_jumps as jumps
from src.phase_j.live_notification_eval import NotificationExchangeOverlay, latest_display

OUT = Path('logs/live_switch_smoothing')
RECORDS = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/pending_expiry/full/records')
SOURCES = ('q_7gc4TgFig', 'review', 'fcXG83vInDY', 'mia8KCjr52g', 'zenchi')
FALLBACK_ADV, FALLBACK_PROBABILITY = 0., .5   # 評価値なし時の呼出元フォールバック (B18b と同じ)


def replay(record: Path, dest: Path, enabled: bool) -> list[dict]:
    """保存記録をライブの評価器で再生し、通知ごとの表示を集める。"""
    from scripts import replay_exchange_event_20260926 as cli
    rows: list[dict] = []

    class Measured(NotificationExchangeOverlay):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, switch_smoothing=enabled, **kwargs)

        def update(self, *args: Any, **kwargs: Any) -> None:
            super().update(*args, **kwargs)
            adv, probability = latest_display(self, FALLBACK_ADV, FALLBACK_PROBABILITY)
            rows.append(dict(t_sec=args[3], game=args[4], source=self.tracker.source, adv=adv,
                             p1=self.tracker.probability))
    with patch.object(cli, 'ExchangeEventOverlay', Measured), \
            patch('scripts.visualize_advantage_overlay._exchange_display', latest_display), \
            patch.object(sys, 'argv', ['replay', str(record), '--out', str(dest), '--production-exchange-event']):
        cli.main()
    return rows


def write_display(dest: Path, rows: list[dict]) -> Path:
    """測定器 (measure_switch_jumps) が読む列だけの display.npz を作る。"""
    path = dest / 'live_display.npz'
    np.savez_compressed(path, t_sec=np.array([r['t_sec'] for r in rows]),
                        game_idx=np.array([r['game'] for r in rows]),
                        source=np.array([r['source'] for r in rows]),
                        display_adv=np.array([r['adv'] for r in rows]))
    return path


def run(enabled: bool, sources: list[str]) -> None:
    """指定記録を再生して保存する (1プロセス1記録ずつ順次)。"""
    for source in sources:
        dest = OUT / ('on' if enabled else 'off') / source
        dest.mkdir(parents=True, exist_ok=True)
        rows = replay(RECORDS / f'{source}.jsonl.gz', dest, enabled)
        write_display(dest, rows)
        print(json.dumps(dict(source=source, enabled=enabled, updates=len(rows))), flush=True)


def measure_one(variant: str, source: str) -> dict:
    """1 記録の飛びを測る。events.jsonl は replay の出力 (評価は ON/OFF 同一)。"""
    root = OUT / variant / source
    return jumps.measure(root / 'live_display.npz', root / 'events.jsonl')


def summary() -> dict:
    """ON/OFF の飛びを合算する。試合境界近傍と、切替に帰属する非イベントの飛びを併記する。"""
    result = {}
    for variant in ('off', 'on'):
        runs = {s: measure_one(variant, s) for s in SOURCES}
        result[variant] = jumps.summarize(runs)
    (OUT / 'SUMMARY.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enabled', action='store_true')
    parser.add_argument('--disabled', action='store_true')
    parser.add_argument('--source', action='append', choices=SOURCES)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.summary:
        data = summary()
        print(json.dumps({v: {k: d[k] for k in ('frames', 'switches', 'jumps', 'by_kind', 'non_event_switch_jumps')}
                          for v, d in data.items()}, ensure_ascii=False))
        return
    if args.enabled == args.disabled:
        parser.error('--enabled か --disabled のどちらか一方を指定')
    run(args.enabled, args.source or list(SOURCES))


if __name__ == '__main__':
    main()
