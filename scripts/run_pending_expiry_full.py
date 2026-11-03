"""本番構成 (+対整合ガード) に pending連鎖失効フラグを足した全長ゲート実走 (run_c65_guard_full の薄い派生)。

段: collect (認識を再収集→記録) → replay (E36b 構成で再生) → report (E36b 門で採点)。
出力は logs/pending_expiry/full と logs/pending_expiry/e36b_on に隔離し、既存 logs は不変。
使い方: python -m scripts.run_pending_expiry_full --step collect|replay|report --source S
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts import run_c65_guard_full as base

EXPIRY_FLAG = '--verification-pending-chain-expiry'
COLLECT_OUT = Path('logs/pending_expiry/full')
REPLAY_OUT = Path('logs/pending_expiry/e36b_on')


def configure() -> None:
    """基底モジュールの出力先  を差し替える。"""
    base.COLLECT_OUT, base.REPLAY_OUT = COLLECT_OUT, REPLAY_OUT


def collect_with_both(source: str) -> None:
    """collect は 1 個のフラグ文字列を足す実装のため、2 個へ分けて足す。"""
    from scripts import collect_e34c as checked
    from scripts import collect_r1 as capture
    import json
    assert json.loads(base.STAGE2_SUMMARY.read_text())['passed']
    original = checked.command
    checked.command = lambda name: original(name) + [
        '--next-recolor-pair-guard', EXPIRY_FLAG]
    capture.OUT = COLLECT_OUT
    capture.collect(source, 'on')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', choices=('collect', 'replay', 'report'), required=True)
    parser.add_argument('--source', choices=base.SOURCES)
    args = parser.parse_args()
    configure()
    if args.step == 'collect':
        collect_with_both(args.source)
    elif args.step == 'replay':
        base.replay(args.source)
    else:
        base.report()


if __name__ == '__main__':
    main()
