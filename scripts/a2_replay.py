"""A2の現行本番と候補を、同じ再収集入力から再生する。"""
from pathlib import Path
import argparse
import sys
from scripts.falsedeath_collect_20261001 import OUT as INPUT
from scripts.falsedeath_replay_20261001 import save_v2e_resources

OUT = Path(__file__).resolve().parents[1] / 'logs/a2_safety'


def main() -> None:
    """明示出力先以外に書かず、既存のトレースと再生器を利用する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prod', 'a2', 'off'))
    parser.add_argument('source')
    args = parser.parse_args()
    dest = OUT / 'replay' / args.mode / args.source
    assert not dest.exists(), dest
    dest.mkdir(parents=True)
    save_v2e_resources(dest)
    from scripts._diag_set2_falsedeath_replay import install_progress, install_trace
    install_progress()
    install_trace(dest)
    from scripts import replay_exchange_event_20260926 as replay
    flags = ['--a2-safety-review'] if args.mode == 'a2' else []
    sys.argv = ['replay', str(INPUT/'collect/records'/f'{args.source}.jsonl.gz'),
                '--out', str(dest), '--production-exchange-event', *flags]
    replay.main()


if __name__ == '__main__':
    main()
