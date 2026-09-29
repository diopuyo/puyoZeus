"""D5b: D5と同じ固定E31 5記録で、単発死亡証明の取消を生存枝のみに限定して再生する。"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any

from scripts import run_d5
from scripts import run_e32 as baseline
from scripts import replay_exchange_event_20260926 as replay_module
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.exchange_event_record import encode

OUT = Path('logs/d5b')
RECORDS = Path('logs/e31/records')


def options() -> dict:
    """run_d5のONと同じ構成 + D5b。"""
    return dict(baseline.OPTIONS, post_counter_death_bound=True, single_death_proof_guard=True,
                single_death_proof_negative_only=True)


def worker(source: str) -> None:
    """0f28a04版証明器を固定して読み、E31記録を再生する。"""
    from scripts import d5_runtime
    d5_runtime.OUT = OUT/'runtime'
    d5_runtime.install()
    prior = baseline.e31.prior
    prior.OUT, prior.AuditTrace = OUT, run_d5.Trace

    def enriched(record: Path, dest: Path, *args: Any, **kwargs: Any) -> dict:
        result = replay_module.replay(RECORDS/record.name, dest, *args, **kwargs)
        save_json(dest/'snapshot_final_audit.json', args[2].summary())
        save_json(dest/'scene_timeline.json', encode(args[2].timeline))
        return result
    prior.replay = enriched
    prior.locked_worker('on', source, options())


def main() -> None:
    """記録ごとに1プロセスを順次起動する (並列1)。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=run_d5.SOURCES)
    args = parser.parse_args()
    assert (OUT/'PREREGISTRATION.md').exists()
    if args.source:
        worker(args.source)
        return
    order = ('q_7gc4TgFig', *[s for s in run_d5.SOURCES if s != 'q_7gc4TgFig'])
    for source in order:
        with (OUT/f'{source}.log').open('a') as stream:
            subprocess.run([sys.executable, '-B', '-m', 'scripts.run_d5b', '--source', source],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
