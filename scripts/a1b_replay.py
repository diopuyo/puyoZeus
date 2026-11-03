"""A1と同じ記録・捕捉器でA1b単独とA2併用を再生する。"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import sys
from scripts.a1_completion_replay import ROOT, INPUT, install_capture
from typing import Any


def install_republish_capture() -> list[dict]:
    """更新後の出力では失われる、送り量換算直前の再公開分岐を計装する。"""
    from src.exchange_prefire_candidates import PrefireCandidates
    original = PrefireCandidates.sends
    rows: list[dict] = []

    def sends(self: Any, chains: list, elapsed: float) -> Any:
        pending = [dict(side=c.side, chain_id=c.chain_id, elapsed=elapsed,
            game=entry['audit'].get('game'), trigger_sec=getattr(c, 'trigger_sec', None),
            same_record=entry['chain'] is c, selected_score=entry['stats'].get('selected_score'))
            for c in chains if self.max_chain and (entry := self.active(c)) is not None
            and c.predicted_final_score is None]
        rows.extend(pending)
        return original(self, chains, elapsed)

    PrefireCandidates.sends = sends
    return rows


def main() -> None:
    """本番再生は再利用し、新しい構成の出力だけを生成する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('single', 'combo'))
    parser.add_argument('name')
    args = parser.parse_args()
    folder = ROOT/'logs/a1b'/args.mode/'replay/a1'/args.name
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (folder/'status.json').exists():
            run(args.mode, args.name, folder)


def run(mode: str, name: str, folder: Path) -> None:
    """採点器の互換キーa1は保持し、実際のフラグを原票へ保存する。"""
    from scripts.falsedeath_replay_20261001 import save_v2e_resources
    from scripts._diag_set2_falsedeath_replay import install_progress
    from scripts import replay_exchange_event_20260926 as replay
    save_v2e_resources(folder)
    install_progress()
    sys.argv = ['replay', str(INPUT/f'{name}.jsonl.gz'), '--out', str(folder),
                '--production-exchange-event', '--completion-score-max-chain']
    if mode == 'combo':
        sys.argv.append('--a2-safety-review')
    (folder/'COMMAND.json').write_text(json.dumps(sys.argv), encoding='utf-8')
    stream = install_capture(folder)
    republished = install_republish_capture()
    try:
        replay.main()
    finally:
        stream.close()
        (folder/'REPUBLISH_BRANCH.json').write_text(json.dumps(dict(count=len(republished), rows=republished)), encoding='utf-8')


if __name__ == '__main__':
    main()
