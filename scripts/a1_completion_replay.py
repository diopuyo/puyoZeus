"""A1の18記録を同じ再収集入力・現行本番設定で再生する。"""
from __future__ import annotations

import argparse
import fcntl
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'logs/a1_completion'
INPUT = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/falsedeath_fix/collect/records')


def install_capture(folder: Path) -> Any:
    """監査と同じ更新直後に予測を記録し、初回値は集計時に固定する。"""
    from src.exchange_event_overlay import ExchangeEventOverlay
    stream = gzip.open(folder/'observations.jsonl.gz', 'wt', encoding='utf-8')
    original = ExchangeEventOverlay.update

    def update(self: Any, *args: Any, **kwargs: Any) -> Any:
        result, _, _, stamp, game = args[:5]
        answer = original(self, *args, **kwargs)
        chains = self.tracker.current.chains if self.tracker.current else []
        row = dict(t=stamp, game=game, states=[result.p1.state.name, result.p2.state.name],
            chains=[{k: v for k, v in asdict(c).items() if k != 'predicted_final_board'} for c in chains],
            score_elapsed=self.tracker._score_elapsed,
            observed=[vars(s.chain_event) if s.chain_event else None for s in (result.p1, result.p2)])
        stream.write(json.dumps(row, default=str)+'\n')
        return answer

    ExchangeEventOverlay.update = update
    return stream


def main() -> None:
    """旧コードは保存したHEADから読み、全出力をこのworktree内へ限定する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('original', 'prod', 'a1'))
    parser.add_argument('name')
    args = parser.parse_args()
    folder = OUT/'replay'/args.mode/args.name
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (folder/'status.json').exists():
            run(args.mode, args.name, folder)


def run(mode: str, name: str, folder: Path) -> None:
    """起動元のパッケージと旧コードのパッケージを明示的に分ける。"""
    if mode == 'original':
        sys.path.insert(0, str(OUT/'baseline'))
        import scripts
        scripts.__path__ = [str(OUT/'baseline/scripts')]
    from scripts.falsedeath_replay_20261001 import save_v2e_resources
    from scripts._diag_set2_falsedeath_replay import install_progress
    from scripts import replay_exchange_event_20260926 as replay
    assert not (folder/'status.json').exists()
    save_v2e_resources(folder)
    install_progress()
    stream = install_capture(folder) if mode != 'original' else None
    sys.argv = ['replay', str(INPUT/f'{name}.jsonl.gz'), '--out', str(folder),
                '--production-exchange-event']
    if mode == 'a1':
        sys.argv.append('--completion-score-coherent')
    (folder/'COMMAND.json').write_text(json.dumps(sys.argv), encoding='utf-8')
    try:
        replay.main()
    finally:
        if stream is not None:
            stream.close()


if __name__ == '__main__':
    main()
